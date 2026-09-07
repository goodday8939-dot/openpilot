"""Fixed-shape NV12 to letterboxed RGB and YOLO on the internal GPU."""
from __future__ import annotations

import numpy as np


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


def nv12_to_rgb(frame, layout, model_size):
  from tinygrad import Tensor
  cw, ch, stride, y_height, _uv_height, size = layout
  mw, mh = model_size
  if tuple(frame.shape) != (size,):
    raise ValueError("unexpected NV12 buffer size")
  width, height, left, top = letterbox_geometry((cw, ch), model_size)
  # Bilinear resize in RGB with chroma sampled at its native half resolution.
  x = ((Tensor.arange(mw).to(frame.device).float() - left + .5) * cw / width - .5).clip(0, cw - 1)
  y = ((Tensor.arange(mh).to(frame.device).float() - top + .5) * ch / height - .5).clip(0, ch - 1)

  def sample(xp, yp, offset, step, w, h):
    xp, yp = xp.clip(0, w - 1), yp.clip(0, h - 1)
    x0, y0 = xp.floor().cast('int'), yp.floor().cast('int')
    x1, y1 = (x0 + 1).clip(0, w - 1), (y0 + 1).clip(0, h - 1)
    wx, wy = xp - x0, (yp - y0).reshape(mh, 1)
    a = frame[offset + y0.reshape(mh, 1) * stride + x0 * step].float()
    b = frame[offset + y0.reshape(mh, 1) * stride + x1 * step].float()
    c = frame[offset + y1.reshape(mh, 1) * stride + x0 * step].float()
    d = frame[offset + y1.reshape(mh, 1) * stride + x1 * step].float()
    return ((a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy) / 255

  luma = sample(x, y, 0, 1, cw, ch)
  u = sample(x / 2, y / 2, stride * y_height, 2, cw // 2, ch // 2) - .5
  v = sample(x / 2, y / 2, stride * y_height + 1, 2, cw // 2, ch // 2) - .5
  rgb = Tensor.stack(luma + 1.402 * v, luma - .344 * u - .714 * v, luma + 1.772 * u).clip(0, 1)
  xx, yy = Tensor.arange(mw).to(frame.device), Tensor.arange(mh).to(frame.device).reshape(mh, 1)
  inside = (xx >= left) & (xx < left + width) & (yy >= top) & (yy < top + height)
  return inside.where(rgb, 114 / 255).unsqueeze(0)


def make_runner(onnx_runner, layout, model_size):
  from tinygrad import Tensor
  if hasattr(onnx_runner, 'onnx_ops'):
    # QCOM's linker rejects YOLO's fused branch-convolution/Concat kernel.
    # Materialize only this runner's branch joins; leave all driving ops alone.
    def concat(*xs, axis):
      return Tensor.cat(*(x.contiguous().realize() for x in xs), dim=axis).contiguous().realize()
    onnx_runner.onnx_ops = {**onnx_runner.onnx_ops, 'Concat': concat}
  name, spec = next(iter(onnx_runner.graph_inputs.items()))
  mw, mh = model_size
  if len(onnx_runner.graph_inputs) != 1 or tuple(spec.shape) != (1, 3, mh, mw):
    raise ValueError("expected fixed NCHW YOLO model")

  def run(frame):
    # Keep the gather-heavy NV12 conversion out of convolution fusion. The
    # Qualcomm compiler has much tighter kernel limits than the USB AMD GPU.
    rgb = nv12_to_rgb(frame, layout, model_size).contiguous().realize()
    raw, = onnx_runner({name: rgb.cast(spec.dtype)}).values()
    scores = raw[:, 4:]
    return raw[:, :4].cat(scores.max(axis=1, keepdim=True),
                         scores.argmax(axis=1).unsqueeze(1).cast(raw.dtype), dim=1).cast('float32').realize()
  return run
