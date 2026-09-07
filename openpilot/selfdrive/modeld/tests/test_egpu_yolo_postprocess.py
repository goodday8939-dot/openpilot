from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld.egpu_yolo_postprocess import OutputWorker, decode_packet


def test_full_cpu_output_socket_skips_without_waiting():
  worker = OutputWorker.__new__(OutputWorker)
  worker.process = SimpleNamespace(poll=lambda: None)

  def full(data):
    raise BlockingIOError

  worker.socket = SimpleNamespace(send=full)
  assert not worker.send({'frame': 1})
  worker.process = SimpleNamespace(poll=lambda: 1)
  with pytest.raises(RuntimeError, match='exited'):
    worker.send({'frame': 2})


def test_cpu_decoder_keeps_originating_frame_and_maps_only_its_transform(monkeypatch):
  from openpilot.selfdrive.modeld import egpu_yolo
  timestamps = iter([100., 100.001, 100.002])
  monkeypatch.setattr(egpu_yolo, 'camera_time', lambda: next(timestamps))
  metadata = {'frameId': 321, 'state': 'run', 'detections': []}
  values = np.zeros((1, 6, 2688), dtype=np.float32)
  values[0, :, 0] = [256, 128, 100, 60, .8, 1]
  result = decode_packet((metadata, values, np.eye(3)*2, (1024, 512), ['person', 'bicycle'], 0.))
  assert result['frameId'] == 321
  assert result['detections'][0]['label'] == 'bicycle'
  assert result['detections'][0]['cameraPoints'][0] == pytest.approx((256-50)/1024)
  assert result['postprocessTime'] == pytest.approx(.001)


def test_cpu_decoder_rejects_wrong_compact_shape_and_preserves_paused_status():
  with pytest.raises(ValueError, match='compact'):
    decode_packet(({}, np.zeros((1, 84, 2688)), np.eye(3), (1344, 760), [], 0.))
  metadata = {'frameId': 456, 'state': 'paused', 'detections': []}
  assert decode_packet((metadata, None, None, None, None, None)) == metadata
