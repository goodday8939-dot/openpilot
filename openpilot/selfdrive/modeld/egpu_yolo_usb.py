"""YOLO-only USB submission helpers, used by the single modeld GPU owner."""
from functools import cache, lru_cache
import array

import numpy as np


@lru_cache(maxsize=1)
def serial_graph_type():
  from tinygrad.runtime.graph.hcq import HCQGraph

  class SerialYoloGraph(HCQGraph):
    single_queue = True

  return SerialYoloGraph


def custom_usb(device) -> bool:
  return callable(getattr(device, "is_usb", None)) and device.is_usb() and device.iface.pci_dev.usb.usb.is_custom


def submit_bound_compute(queue, device):
  """Write a bound graph's indirect packet in contiguous USB transfers."""
  commands = queue.indirect_cmd
  ring = device.compute_queue.ring
  position = device.compute_queue.put_value
  offset = position % len(ring)
  first = min(len(commands), len(ring)-offset)
  ring[offset:offset+first] = array.array('I', commands[:first])
  if first < len(commands):
    ring[:len(commands)-first] = array.array('I', commands[first:])
  device.compute_queue.put_value = position+len(commands)
  device.compute_queue.signal_doorbell(device)


@cache
def batched_compute_type(base):
  class YoloComputeQueue(base):
    def _submit(self, device):
      if self.binded_device == device and device.xccs == 1 and custom_usb(device):
        submit_bound_compute(self, device)
      else:
        super()._submit(device)
  return YoloComputeQueue


def run_yolo(run, queue):
  from tinygrad.device import Device
  device = Device[queue.device]
  if not custom_usb(device):
    return run(queue=queue)
  # Graph construction happens on the modeld owner thread. Always restore the
  # normal factory before the next driving call, including initialization errors.
  previous = device.graph
  previous_compute = device.hw_compute_queue_t
  device.graph = serial_graph_type()
  device.hw_compute_queue_t = batched_compute_type(previous_compute)
  try:
    return run(queue=queue)
  finally:
    device.graph = previous
    device.hw_compute_queue_t = previous_compute


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
