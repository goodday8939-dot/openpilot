import importlib.util
import json
from pathlib import Path


def load_feature():
  path = Path(__file__).parents[1] / "features/egpu_yolo.py"
  spec = importlib.util.spec_from_file_location("egpu_yolo_feature_test", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_web_payload_distinguishes_installed_and_compiled(tmp_path):
  feature = load_feature()
  state = {"status": {}, "frame": None, "received": 0}
  assert feature.status_payload(state, tmp_path, 100)["status"]["state"] == "not_prepared"
  (tmp_path / "model.onnx").touch()
  (tmp_path / "manifest.json").write_text("{}")
  assert feature.status_payload(state, tmp_path, 100)["status"]["state"] == "downloaded"
  (tmp_path / "yolo_qcom.pkl").touch()
  assert feature.status_payload(state, tmp_path, 100)["status"]["state"] == "prepared"
  (tmp_path / "qcom_enabled").touch()
  assert feature.status_payload(state, tmp_path, 100)["status"]["state"] == "ready"


def test_web_keeps_detection_timestamp_independent_of_diagnostic_heartbeat(tmp_path):
  feature = load_feature()
  frame = {"frameId": 8, "timestampEof": 100_000_000_000, "detections": []}
  state = {"status": {"state": "no_budget", "frameId": 15}, "frame": frame, "received": 100.5}
  result = feature.status_payload(state, tmp_path, 100.51)
  assert result["stale"] is True
  assert result["frame"]["frameId"] == 8
  assert result["status"]["frameId"] == 15
  assert result["ageSeconds"] > .5


def test_web_exposes_only_fresh_supervisor_recovery_state(tmp_path):
  feature = load_feature()
  state = {'status': {}, 'frame': None, 'received': 0}
  path = tmp_path / 'live_reuse_status.json'
  path.write_text(json.dumps({'updated_camera_time': 100, 'stage': 'cooldown', 'retry_count': 1,
                              'stable_seconds_required': 30, 'recovery_reason': 'modelV2 raw gap 152.090 ms'}))
  assert feature.status_payload(state, tmp_path, 101)['supervisor']['stage'] == 'cooldown'
  assert feature.status_payload(state, tmp_path, 111)['supervisor'] is None
  path.write_text('{')
  assert feature.status_payload(state, tmp_path, 101)['supervisor'] is None
