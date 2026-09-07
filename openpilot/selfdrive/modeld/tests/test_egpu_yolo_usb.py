from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld.egpu_yolo_usb import read_usb_output


def device_for_readback():
  events = []
  data = bytearray(np.arange(6, dtype=np.float32).tobytes())

  class Queue:
    def wait(self, signal, value):
      events.append(("wait", signal, value))
      return self

    def copy(self, staging, source, nbytes):
      events.append(("copy", source, nbytes))
      return self

    def write(self, pointer, value):
      events.append(("trigger", pointer, value))
      return self

    def signal(self, signal, value):
      events.append(("signal", signal, value))
      return self

    def submit(self, device):
      events.append(("submit",))
      return self

  class View:
    def cpu_view(self):
      return self

    def view(self, **kwargs):
      events.append(("read", kwargs["size"]))
      return memoryview(data)

  device = SimpleNamespace(error_state=None, timeline_value=12, timeline_signal="timeline", synchronize=lambda: events.append(("sync",)),
                           allocator=SimpleNamespace(b=[View()]), hw_copy_queue_t=Queue,
                           iface=SimpleNamespace(pci_dev=SimpleNamespace(usb=SimpleNamespace(scsi_read_arm=lambda n: events.append(("arm", n)))),
                                                 cq_buf=SimpleNamespace(offset=lambda n: n)))

  def next_timeline():
    value = device.timeline_value
    device.timeline_value += 1
    return value

  device.next_timeline = next_timeline
  return device, events, data


def test_copy_is_gpu_ordered_after_inference_and_readback_has_owned_storage():
  device, events, data = device_for_readback()
  result = read_usb_output(device, "output", 24, (1, 6))
  assert events == [("arm", 24), ("wait", "timeline", 11), ("copy", "output", 24),
                    ("trigger", 12, 0), ("signal", "timeline", 12), ("submit",), ("read", 24)]
  assert device.timeline_value == 13
  data[:] = bytes(24)
  np.testing.assert_array_equal(result, np.arange(6).reshape(1, 6))


def test_device_fault_does_not_submit_optional_work():
  device, events, _ = device_for_readback()
  device.error_state = RuntimeError("device failed")
  with pytest.raises(RuntimeError, match="device failed"):
    read_usb_output(device, "output", 24, (1, 6))
  assert events == []


def test_timeline_rollover_is_handled_before_submitting_copy():
  device, events, _ = device_for_readback()
  device.timeline_value = (1 << 31) + 1

  def rollover():
    events.append(("sync",))
    device.timeline_value = 2
    device.timeline_signal = "new timeline"

  device.synchronize = rollover
  read_usb_output(device, "output", 24, (1, 6))
  assert events[0] == ("sync",)
  assert events[2] == ("wait", "new timeline", 1)
  assert events[5] == ("signal", "new timeline", 2)


@pytest.mark.parametrize("fail", [False, True])
def test_driving_graph_factory_restored_after_optional_run(monkeypatch, fail):
  from tinygrad import device as device_module
  from openpilot.selfdrive.modeld import egpu_yolo_usb
  device = SimpleNamespace(graph="driving")
  monkeypatch.setattr(device_module, "Device", {"AMD": device})
  monkeypatch.setattr(egpu_yolo_usb, "custom_usb", lambda d: True)
  monkeypatch.setattr(egpu_yolo_usb, "serial_graph_type", lambda: "yolo")
  queue = SimpleNamespace(device="AMD")

  def run(**kwargs):
    assert kwargs["queue"] is queue and device.graph == "yolo"
    if fail:
      raise RuntimeError("optional initialization failed")
    return "output"

  if fail:
    with pytest.raises(RuntimeError, match="optional initialization failed"):
      egpu_yolo_usb.run_yolo(run, queue)
  else:
    assert egpu_yolo_usb.run_yolo(run, queue) == "output"
  assert device.graph == "driving"
