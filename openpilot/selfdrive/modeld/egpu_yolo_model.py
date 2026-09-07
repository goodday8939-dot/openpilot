"""YUV420 (four luma planes + U + V) adapter; all image work stays on GPU."""
from tinygrad import Tensor


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


def make_yolo_runner(onnx_runner, width: int, height: int):
  name, spec = next(iter(onnx_runner.graph_inputs.items()))
  if len(onnx_runner.graph_inputs) != 1 or tuple(spec.shape) != (1, 3, height, width):
    raise ValueError("YOLO must have one fixed NCHW image input")

  def run(queue):
    rgb = packed_yuv_to_rgb(queue[-1], (height, width))
    raw, = onnx_runner({name: rgb.cast(spec.dtype)}).values()
    scores = raw[:, 4:, :]
    # Return only each anchor's winning class, score and box (6 rather than 84
    # channels), reducing USB readback while keeping class-aware NMS on CPU.
    return raw[:, :4, :].cat(scores.max(axis=1, keepdim=True),
                            scores.argmax(axis=1).unsqueeze(1).cast(raw.dtype), dim=1).cast("float32").realize()
  return run
