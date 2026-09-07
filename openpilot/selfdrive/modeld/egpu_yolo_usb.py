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


def run_yolo(run, queue):
  from tinygrad.device import Device
  device = Device[queue.device]
  if not custom_usb(device):
    return run(queue=queue)
  # Graph construction happens on the modeld owner thread. Always restore the
  # normal factory before the next driving call, including initialization errors.
  previous = device.graph
  device.graph = serial_graph_type()
  try:
    return run(queue=queue)
  finally:
    device.graph = previous


def read_yolo(raw):
  from tinygrad.device import Device
  from tinygrad import dtypes
  device = Device[raw.device]
  nbytes = raw.numel() * 4
  if not custom_usb(device) or raw.dtype != dtypes.float32 or nbytes > device.allocator.b[0].size:
    return raw.numpy()
  return read_usb_output(device, raw.uop.buffer._buf, nbytes, raw.shape)


def read_usb_output(device, source, nbytes: int, shape):
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
  return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
