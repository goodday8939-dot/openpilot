"""Fixed-shape NV12 to letterboxed RGB and YOLO on the internal GPU."""
from __future__ import annotations

from functools import partial
import hashlib
import os
from pathlib import Path

import numpy as np

BACKEND = 'QCOM:IR3'
MESA_SHA256 = '9436d1bd3da1c4394523a3debef4df9451b967c264f5d48bf9aeee03430c1364'


def configure_environment(directory: Path):
  """Select the pinned YOLO compiler before importing tinygrad in this process."""
  library = directory / 'lib' / 'libtinymesa.so'
  if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != MESA_SHA256:
    raise ValueError('internal YOLO requires the verified experiment libtinymesa.so')
  # tinygrad's DLL loader accepts an absolute MESA_PATH. No system installation
  # or changes to the driving process's library search path are required.
  os.environ.update(DEV=BACKEND, WARP_DEV='QCOM', MESA_PATH=str(library), IMAGE='0', FLOAT16='1',
                    NOLOCALS='1', JIT_BATCH_SIZE='0', OPENPILOT_HACKS='1', QCOM_PRIORITY='15')


def local_size_for(global_size):
  """A legal, bounded workgroup for kernels without shared local memory."""
  remaining = 64
  result = []
  for size in global_size:
    local = next(value for value in (64, 32, 16, 8, 4, 2, 1) if value <= remaining and size % value == 0)
    result.append(local)
    remaining //= local
  return tuple(result)


def _bounded_local_size(call, prg):
  from tinygrad.engine import realize
  if (str(prg.src[1].arg).startswith('QCOM') and prg.arg.local_size is None and prg.arg.global_size is not None
      and all(isinstance(x, int) for x in prg.arg.global_size)):
    realize.local_size_cache.setdefault(prg.key, local_size_for(prg.arg.global_size))
  return realize.optimize_local_size(call, prg)


def configure_compiler():
  """Only call in the dedicated compiler process, never in driving modeld."""
  from tinygrad.engine import realize
  from tinygrad.uop.ops import Ops, PatternMatcher, UPat

  realize.pm_optimize_local_size = PatternMatcher([
    (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name='prg'),), name='call', allow_any_len=True), _bounded_local_size),
  ])


def letterbox_geometry(camera_size: tuple[int, int], model_size: tuple[int, int]):
  cw, ch = camera_size
  mw, mh = model_size
  if min(cw, ch, mw, mh) <= 0:
    raise ValueError("positive image dimensions required")
  scale = min(mw / cw, mh / ch)
  width, height = round(cw * scale), round(ch * scale)
  left, top = (mw - width) // 2, (mh - height) // 2
  return width, height, left, top


def camera_transform(camera_size: tuple[int, int], model_size: tuple[int, int]):
  cw, ch = camera_size
  width, height, left, top = letterbox_geometry(camera_size, model_size)
  return np.array([[cw / width, 0, -left * cw / width],
                   [0, ch / height, -top * ch / height], [0, 0, 1]], dtype=np.float32)


def _nv12_to_rgb_kernel(dest, frame, *, layout, model_size):
  from tinygrad import UOp, dtypes
  from tinygrad.uop.ops import KernelInfo
  cw, ch, stride, y_height, _uv_height, _size = layout
  mw, mh = model_size
  width, height, left, top = letterbox_geometry((cw, ch), model_size)
  iy, ix = UOp.range(mh, 0), UOp.range(mw, 1)
  x = ((ix.cast(dtypes.float32) - left + .5) * (cw / width) - .5).clip(0, cw - 1)
  y = ((iy.cast(dtypes.float32) - top + .5) * (ch / height) - .5).clip(0, ch - 1)

  def sample(xp, yp, offset, step, w, h):
    xp, yp = xp.clip(0, w - 1), yp.clip(0, h - 1)
    # Coordinates are nonnegative, so integer truncation equals floor.
    x0, y0 = xp.cast(dtypes.int32).cast(dtypes.weakint), yp.cast(dtypes.int32).cast(dtypes.weakint)
    x1, y1 = (x0 + 1).minimum(w - 1), (y0 + 1).minimum(h - 1)
    wx, wy = xp - x0.cast(dtypes.float32), yp - y0.cast(dtypes.float32)
    a = frame[offset + y0 * stride + x0 * step].cast(dtypes.float32)
    b = frame[offset + y0 * stride + x1 * step].cast(dtypes.float32)
    c = frame[offset + y1 * stride + x0 * step].cast(dtypes.float32)
    d = frame[offset + y1 * stride + x1 * step].cast(dtypes.float32)
    return ((a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy) / 255

  luma = sample(x, y, 0, 1, cw, ch)
  u = sample(x / 2, y / 2, stride * y_height, 2, cw // 2, ch // 2) - .5
  v = sample(x / 2, y / 2, stride * y_height + 1, 2, cw // 2, ch // 2) - .5
  channels = [luma + 1.402 * v, luma - .344 * u - .714 * v, luma + 1.772 * u]
  inside = (ix >= left) & (ix < left + width) & (iy >= top) & (iy < top + height)
  stores = [dest[0, c, iy, ix].store(inside.where(value.clip(0, 1), 114 / 255)) for c, value in enumerate(channels)]
  return UOp.group(*stores).end(iy, ix).sink(arg=KernelInfo(name='nv12_letterbox_rgb'))


def nv12_to_rgb(frame, layout, model_size):
  from tinygrad import Tensor
  if tuple(frame.shape) != (layout[-1],):
    raise ValueError('unexpected NV12 buffer size')
  mw, mh = model_size
  # Direct indexed loads avoid the Tensor gather/stack reduction graph, which
  # took about 52 ms for preprocessing alone on the internal GPU.
  output = Tensor.empty(1, 3, mh, mw, dtype='float32', device=frame.device)
  return output.custom_kernel(frame, fxn=partial(_nv12_to_rgb_kernel, layout=layout, model_size=model_size))[0]


def make_runner(onnx_runner, layout, model_size):
  from tinygrad import Tensor
  from tinygrad.helpers import Context
  if hasattr(onnx_runner, 'onnx_ops'):
    # QCOM's linker rejects YOLO's fused branch-convolution/Concat kernel.
    # Materialize only this runner's branch joins; leave all driving ops alone.
    def concat(*xs, axis):
      return Tensor.cat(*(x.contiguous().realize() for x in xs), dim=axis).contiguous().realize()
    onnx_runner.onnx_ops = {**onnx_runner.onnx_ops, 'Concat': concat}
    if 'Conv' in onnx_runner.onnx_ops:
      original_conv = onnx_runner.onnx_ops['Conv']

      def conv(x, weight, bias=None, **kwargs):
        # ONNX stores unit strides/dilations as tuples. tinygrad's Winograd
        # dispatch requires scalar 1, otherwise it silently uses plain Conv.
        for key in ('strides', 'dilations'):
          if kwargs.get(key) in ((1, 1), [1, 1]):
            kwargs[key] = 1
        with Context(WINO=1):
          value = original_conv(x, weight, bias, **kwargs)
        # Preserve shared branch results instead of recomputing earlier Conv
        # chains when each Concat input is materialized.
        return value.contiguous().realize()

      onnx_runner.onnx_ops['Conv'] = conv
  name, spec = next(iter(onnx_runner.graph_inputs.items()))
  mw, mh = model_size
  if len(onnx_runner.graph_inputs) != 1 or tuple(spec.shape) != (1, 3, mh, mw):
    raise ValueError("expected fixed NCHW YOLO model")

  def run(frame):
    # Materialize the direct NV12 kernel before the convolution graph. The
    # Qualcomm compiler has much tighter kernel limits than the USB AMD GPU.
    rgb = nv12_to_rgb(frame, layout, model_size).contiguous().realize()
    raw, = onnx_runner({name: rgb.cast(spec.dtype)}).values()
    scores = raw[:, 4:]
    return raw[:, :4].cat(scores.max(axis=1, keepdim=True),
                         scores.argmax(axis=1).unsqueeze(1).cast(raw.dtype), dim=1).cast('float32').realize()
  return run
