"""YOLO-only USB submission helpers, used by the single modeld GPU owner."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def serial_graph_type():
  from tinygrad.runtime.graph.hcq import HCQGraph

  class SerialYoloGraph(HCQGraph):
    single_queue = True

  return SerialYoloGraph


def custom_usb(device) -> bool:
  return callable(getattr(device, "is_usb", None)) and device.is_usb() and device.iface.pci_dev.usb.usb.is_custom


def replay_yolo(run, queue):
  """Reuse validated JIT arguments only while the exact input UOp is unchanged."""
  from tinygrad.engine.jit import TinyJit, _prepare_jit_inputs, capturing
  from tinygrad.helpers import JIT
  if type(run) is not TinyJit or not JIT or capturing:
    return run(queue=queue)
  cached = getattr(run, '_yolo_validated_input', None)
  if run.cnt >= 2 and cached is not None and cached[0] is run.captured and cached[1] is queue.uop:
    result = run.captured(cached[2], cached[3])
    run.cnt += 1
    return result
  # Normal TinyJit validates the input representation, dtype, device and names.
  # A changed allocation or view always returns through that validation.
  result = run(queue=queue)
  if run.cnt >= 2:
    inputs, values, _, _ = _prepare_jit_inputs((), {'queue': queue})
    run._yolo_validated_input = (run.captured, queue.uop, inputs, values)
  return result


def run_yolo(run, queue):
  from tinygrad.device import Device
  device = Device[queue.device]
  if not custom_usb(device):
    return replay_yolo(run, queue)
  # Graph construction happens on the modeld owner thread. Always restore the
  # normal factory before the next driving call, including initialization errors.
  previous = device.graph
  device.graph = serial_graph_type()
  try:
    return replay_yolo(run, queue)
  finally:
    device.graph = previous


def read_yolo(raw):
  from tinygrad.device import Device
  from tinygrad import dtypes
  device = Device[raw.device]
  nbytes = raw.numel() * raw.dtype.itemsize
  if not custom_usb(device) or raw.dtype not in (dtypes.float16, dtypes.float32) or nbytes > device.allocator.b[0].size:
    return raw.numpy()
  dtype = np.float16 if raw.dtype == dtypes.float16 else np.float32
  return read_usb_output(device, raw.uop.buffer._buf, nbytes, raw.shape, dtype=dtype)


def read_usb_output(device, source, nbytes: int, shape, *, dtype=np.float32):
  if device.error_state is not None:
    raise device.error_state
  if device.timeline_value > 1 << 31:
    device.synchronize()  # preserve HCQ's rare timeline rollover handling
  staging = device.allocator.b[0]
  device.iface.pci_dev.usb.scsi_read_arm(nbytes)
  # Submit the copy while compute is in flight. The GPU timeline dependency,
  # rather than an extra CPU wait, orders it after the complete YOLO graph.
  device.hw_copy_queue_t().wait(device.timeline_signal, device.timeline_value - 1) \
    .copy(staging, source, nbytes).write(device.iface.cq_buf.offset(12), 0) \
    .signal(device.timeline_signal, device.next_timeline()).submit(device)
  data = staging.cpu_view().view(size=nbytes, fmt="B")[:]
  return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
