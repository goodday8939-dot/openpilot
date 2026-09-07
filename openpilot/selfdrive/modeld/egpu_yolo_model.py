"""YUV420 (four luma planes + U + V) adapter; all image work stays on GPU."""
from tinygrad import Tensor


def _native_rgb_kernel(dest, frame):
  from tinygrad import UOp, dtypes
  from tinygrad.uop.ops import KernelInfo
  height, width = frame.shape[-2:]
  y, x = UOp.range(height * 2, 0), UOp.range(width * 2, 1)
  channel = (x % 2) * 2 + y % 2

  def plane(c):
    return frame[frame.shape[0]-1, c, y // 2, x // 2] if len(frame.shape) == 4 else frame[c, y // 2, x // 2]

  luma = plane(channel).cast(dtypes.float32) / 255
  u = plane(4).cast(dtypes.float32) / 255 - .5
  v = plane(5).cast(dtypes.float32) / 255 - .5
  rgb = (luma + 1.402 * v, luma - .344 * u - .714 * v, luma + 1.772 * u)
  return UOp.group(*(dest[0, c, y, x].store(value.clip(0, 1)) for c, value in enumerate(rgb))) \
    .end(y, x).sink(arg=KernelInfo(name='native_packed_yuv_rgb'))


def native_packed_yuv_to_rgb(frame: Tensor) -> Tensor:
  if len(frame.shape) not in (3, 4) or frame.shape[-3] != 6:
    raise ValueError('expected packed (6, H/2, W/2) frame or (history, 6, H/2, W/2) queue')
  height, width = frame.shape[-2:]
  output = Tensor.empty(1, 3, height * 2, width * 2, dtype='float32', device=frame.device)
  return output.custom_kernel(frame, fxn=_native_rgb_kernel)[0].realize()


def packed_yuv_to_rgb(frame: Tensor, size: tuple[int, int]) -> Tensor:
  if len(frame.shape) != 3 or frame.shape[0] != 6:
    raise ValueError("expected packed (6, H/2, W/2) YUV")
  _, h, w = frame.shape
  # frames_to_tensor packs channels as x_parity * 2 + y_parity.
  y = frame[:4].reshape(2, 2, h, w).permute(2, 1, 3, 0).reshape(h * 2, w * 2).float() / 255
  uv = frame[4:6].float().interpolate((h * 2, w * 2), mode="nearest") / 255 - 0.5
  # Match the camera renderer's full-range YUV matrix, including UV centering.
  rgb = Tensor.stack(y + 1.402 * uv[1], y - 0.344 * uv[0] - 0.714 * uv[1], y + 1.772 * uv[0])
  return rgb.clip(0, 1).interpolate(size, mode="linear").unsqueeze(0)


def make_yolo_runner(onnx_runner, width: int, height: int, *, native: bool = False, output_dtype: str = 'float32'):
  name, spec = next(iter(onnx_runner.graph_inputs.items()))
  if len(onnx_runner.graph_inputs) != 1 or tuple(spec.shape) != (1, 3, height, width):
    raise ValueError("YOLO must have one fixed NCHW image input")

  def run(queue):
    if native and tuple(queue.shape[-2:]) != (height // 2, width // 2):
      raise ValueError('native reuse cannot resize the driving image')
    # Pass the whole input allocation to the custom kernel. A realized slice
    # can keep the capture-time address in an HCQ graph after serialization.
    # Selecting the latest frame inside the kernel keeps this a direct JIT PARAM.
    rgb = native_packed_yuv_to_rgb(queue) if native else packed_yuv_to_rgb(queue[-1], (height, width))
    raw, = onnx_runner({name: rgb.cast(spec.dtype)}).values()
    scores = raw[:, 4:, :]
    # Return only each anchor's winning class, score and box (6 rather than 84
    # channels), reducing USB readback while keeping class-aware NMS on CPU.
    return raw[:, :4, :].cat(scores.max(axis=1, keepdim=True),
                            scores.argmax(axis=1).unsqueeze(1).cast(raw.dtype), dim=1).cast(output_dtype).realize()
  return run
