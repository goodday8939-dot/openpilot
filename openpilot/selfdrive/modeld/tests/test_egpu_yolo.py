import ast
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.modeld.egpu_yolo import CameraPrefetch, IdleBudget, camera_detections, decode_detections, source_fingerprint


def settled_budget(runtime=0.006):
  budget = IdleBudget(runtime)
  for frame in range(25):
    budget.observe(frame, 100 + frame * .05, 100 + frame * .05 + .01)
  return budget


def test_admit_only_when_whole_job_fits_and_rate_limit_does_not_override_deadline():
  budget = settled_budget()
  assert budget.admit(budget.deadline - .03) == "run"
  budget.finish(budget.last_run, budget.last_run + .006)
  assert budget.admit(budget.last_run + .06) == "rate_limit"
  budget.observe(30, 101.5, 101.51)
  assert budget.admit(101.52) == "warming"
  budget = settled_budget(.040)
  assert budget.admit(budget.deadline - .035) == "no_budget"


def test_late_camera_receive_does_not_create_false_free_time():
  budget = settled_budget(.010)
  budget.observe(25, 101.25, 101.297)
  assert budget.deadline == pytest.approx(101.31)
  assert budget.admit(101.300) == "no_budget"


def test_every_frame_admission_removes_only_the_rate_limit():
  budget = settled_budget()
  first = budget.deadline-.02
  assert budget.admit(first, min_interval=0.) == 'run'
  budget.finish(first, first+.006)
  budget.observe(25, 101.25, 101.26)
  assert budget.admit(101.29) == 'rate_limit'
  assert budget.admit(101.29, min_interval=0.) == 'run'
  budget.finish(101.29, 101.296)
  budget.observe(26, 101.3, 101.31)
  assert budget.admit(101.34, camera_pending=True, min_interval=0.) == 'camera_pending'
  assert budget.admit(101.355, min_interval=0.) == 'no_budget'
  assert budget.admit(101.34, min_interval=0.) == 'run'
  budget.finish(101.34, budget.deadline+.002)
  assert budget.admit(102., min_interval=0.) == 'overrun'


def test_isolated_early_camera_does_not_remove_idle_slots_from_later_frames():
  budget = IdleBudget(.006)
  for frame in range(25):
    sof = 100 + frame * .05
    budget.observe(frame, sof, sof + (.04 if frame == 2 else .06))
  assert budget.deadline == pytest.approx(sof + .11)
  assert budget.admit(sof + .095) == "run"
  # A delayed receive still cannot shift the deadline by that delay.
  budget.observe(25, 101.25, 101.34)
  assert budget.deadline == pytest.approx(101.36)


def test_waiting_primary_frame_takes_priority_without_consuming_yolo_rate_slot():
  budget = settled_budget()
  start = budget.deadline - .03
  assert budget.admit(start, camera_pending=True) == "camera_pending"
  assert budget.runs == 0 and budget.skipped == 1
  assert budget.admit(start, camera_pending=False) == "run"


def test_camera_probe_preserves_frame_and_metadata_for_primary_inference():
  class Client:
    frame = 0
    calls = []

    def recv(self, timeout_ms=100):
      self.calls.append(timeout_ms)
      self.frame += 1
      return f"frame{self.frame}"

  client = Client()
  prefetch = CameraPrefetch(client, lambda c: c.frame)
  assert prefetch.ready() and prefetch.ready()
  assert client.calls == [0]
  assert prefetch.recv() == ("frame1", 1)
  assert client.calls == [0]
  assert prefetch.recv() == ("frame2", 2)
  assert client.calls == [0, 100]


def test_empty_camera_probe_does_not_cache_a_missing_frame():
  class Client:
    def recv(self, timeout_ms=100):
      return None if timeout_ms == 0 else "next"

  prefetch = CameraPrefetch(Client(), lambda c: "metadata")
  assert not prefetch.ready()
  assert prefetch.recv() == ("next", "metadata")


def test_budget_uses_complete_camera_pair_readiness():
  budget = IdleBudget(.004)
  for frame in range(25):
    sof = 100 + frame * .05
    budget.observe(frame, sof, sof + .055)
  assert budget.admit(sof + .090) == "run"


def test_overrun_disables_subsequent_gpu_submission():
  budget = settled_budget()
  start = budget.deadline - .025
  assert budget.admit(start) == "run"
  budget.finish(start, budget.deadline + .001)
  assert budget.overruns == 1
  assert budget.admit(budget.deadline + 1) == "overrun"


@pytest.mark.parametrize("runtime", [0, -1, float("nan"), float("inf")])
def test_invalid_profile_cannot_admit(runtime):
  with pytest.raises(ValueError):
    IdleBudget(runtime)


def test_timestamp_discontinuity_and_drops_require_new_stable_cadence():
  budget = settled_budget()
  budget.observe(25, 101.25, 101.26, dropped=True)
  assert budget.admit(101.27) == "warming"
  budget.observe(26, 102.3, 101.31)
  assert budget.deadline == 0
  assert budget.settled == 0


def test_camera_clock_includes_suspended_time(monkeypatch):
  from openpilot.selfdrive.modeld import egpu_yolo
  monkeypatch.setattr(egpu_yolo.time, "CLOCK_BOOTTIME", 7, raising=False)
  monkeypatch.setattr(egpu_yolo.time, "clock_gettime", lambda clock: 105.0 if clock == 7 else 100.0, raising=False)
  monkeypatch.setattr(egpu_yolo.time, "monotonic", lambda: 100.0)
  assert egpu_yolo.camera_time() == 105.0


def test_nms_keeps_different_classes_and_bounds_coordinates():
  raw = np.zeros((1, 6, 4), dtype=np.float32)
  raw[0, :4, :] = np.array([[100, 50, 100, 60], [102, 50, 100, 60], [100, 50, 100, 60], [-1, 5, 10, 10]]).T
  raw[0, 4] = [.95, .9, .01, .8]
  raw[0, 5] = [.01, .01, .85, .01]
  results = decode_detections(raw, 200, 100)
  assert len(results) == 3
  assert [d["classId"] for d in results] == [0, 1, 0]
  assert results[-1]["x1"] == 0
  assert results[0]["x1"] == pytest.approx(.25)


def test_nms_empty_dense_and_nonfinite_outputs():
  assert decode_detections(np.zeros((1, 84, 100)), 320, 160) == []
  raw = np.ones((1, 84, 8400), dtype=np.float32)
  assert len(decode_detections(raw, 320, 160)) == 1
  raw[0, 0, 0] = np.nan
  with pytest.raises(ValueError):
    decode_detections(raw, 320, 160)


def test_resident_yolo_uses_primary_queue_only_after_all_driving_publications():
  assert len(source_fingerprint()) == 64
  source = (Path(__file__).parents[1] / "modeld.py").read_text()
  tree = ast.parse(source)
  calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
  optional = [node.lineno for node in calls if node.func.attr == "after_publish"]
  driving = [node.lineno for node in calls if node.func.attr == "send" and node.args
             and isinstance(node.args[0], ast.Constant) and node.args[0].value in ("modelV2", "drivingModelData", "cameraOdometry")]
  assert len(driving) == 3
  assert len(optional) == 1 and optional[0] > max(driving)
  assert "ReuseRuntime.load(self.input_queues['img_q'])" in source
  assert 'frequency=ModelConstants.MODEL_RUN_FREQ' in source


def test_yuv_adapter_preserves_luma_parity_and_chroma():
  from tinygrad import Tensor
  from openpilot.selfdrive.modeld.egpu_yolo_model import packed_yuv_to_rgb
  y = np.arange(32, dtype=np.uint8).reshape(4, 8) * 7
  u = np.array([[0, 64, 128, 255], [255, 192, 128, 0]], dtype=np.uint8)
  v = u[:, ::-1].copy()
  packed = np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], u, v))
  uf, vf = [(c.repeat(2, 0).repeat(2, 1).astype(np.float32) / 255 - .5) for c in (u, v)]
  yf = y.astype(np.float32) / 255
  expected = np.stack((yf + 1.402 * vf, yf - .344 * uf - .714 * vf, yf + 1.772 * uf)).clip(0, 1)[None]
  actual = packed_yuv_to_rgb(Tensor(packed), (4, 8)).numpy()
  np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_cereal_yolo_roundtrip_preserves_frame_and_detections():
  from openpilot.cereal import log
  msg = log.Event.new_message()
  payload = msg.init("carrotYolo")
  payload.frameId = 123
  payload.cameraWidth, payload.cameraHeight = 1928, 1208
  payload.submitTime, payload.readbackTime, payload.postprocessTime = .0003, .0017, .0004
  payload.detections = [{"classId": 0, "label": "person", "confidence": .9, "x1": .1, "y1": .2, "x2": .3, "y2": .4}]
  with log.Event.from_bytes(msg.to_bytes()) as result:
    assert result.which() == "carrotYolo"
    assert result.carrotYolo.frameId == 123
    assert result.carrotYolo.detections[0].label == "person"
    assert result.carrotYolo.cameraWidth == 1928
    assert result.carrotYolo.readbackTime == pytest.approx(.0017)


def test_camera_coordinates_use_warp_and_original_size():
  detection = {"x1": .1, "y1": .2, "x2": .3, "y2": .4}
  transform = np.array([[2, 0, 10], [0, 3, 20], [0, 0, 1.]])
  result = camera_detections([detection], transform, (100, 50), (200, 150))
  np.testing.assert_allclose(result[0]["cameraPoints"], [.15, 1/3, .35, 1/3, .35, 8/15, .15, 8/15])
  assert camera_detections([detection], np.zeros((3, 3)), (100, 50), (200, 150)) == []


def test_compact_output_matches_raw_class_selection():
  raw = np.zeros((1, 84, 3), dtype=np.float32)
  raw[0, :4] = np.array([[20, 30, 20, 10], [40, 40, 20, 10], [100, 100, 20, 10]]).T
  raw[0, 9, 0], raw[0, 6, 1], raw[0, 83, 2] = .95, .85, .75
  compact = np.concatenate((raw[:, :4], raw[:, 4:].max(1, keepdims=True), raw[:, 4:].argmax(1)[:, None]), axis=1)
  assert decode_detections(compact, 320, 160, compact=True) == decode_detections(raw, 320, 160)


def test_gpu_adapter_returns_compact_results_and_consumes_newest_queue_frame():
  from types import SimpleNamespace
  from tinygrad import Tensor, dtypes
  from openpilot.selfdrive.modeld.egpu_yolo_model import make_yolo_runner

  class Runner:
    graph_inputs = {"images": SimpleNamespace(shape=(1, 3, 4, 8), dtype=dtypes.float32)}

    def __call__(self, inputs):
      mean = inputs["images"].mean().reshape(1, 1, 1)
      return {"output": mean.expand(1, 6, 2) + Tensor([20, 10, 5, 5, .2, .5]).reshape(1, 6, 1)}

  queue = np.full((5, 6, 2, 4), 128, dtype=np.uint8)
  queue[-1, :4] = 0
  run = make_yolo_runner(Runner(), 8, 4)
  result = run(Tensor(queue)).numpy()
  assert result.shape == (1, 6, 2)
  assert (result[0, 5] == 1).all()
  assert result[0, 0, 0] < 20.01


def test_native_color_kernel_preserves_full_luma_parity_and_chroma():
  from tinygrad import Tensor
  from openpilot.selfdrive.modeld.egpu_yolo_model import native_packed_yuv_to_rgb, packed_yuv_to_rgb
  values = np.random.default_rng(42).integers(0, 256, (6, 8, 16), dtype=np.uint8)
  frame = Tensor(values)
  actual = native_packed_yuv_to_rgb(frame).numpy()
  expected = packed_yuv_to_rgb(frame, (16, 32)).numpy()
  np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_native_runner_refuses_to_resize_an_existing_driving_image():
  from types import SimpleNamespace
  from tinygrad import Tensor, dtypes
  from openpilot.selfdrive.modeld.egpu_yolo_model import make_yolo_runner
  runner = SimpleNamespace(graph_inputs={'images': SimpleNamespace(shape=(1, 3, 4, 8), dtype=dtypes.float32)})
  with pytest.raises(ValueError, match='cannot resize'):
    make_yolo_runner(runner, 8, 4, native=True)(Tensor.zeros(2, 6, 4, 8, dtype='uint8'))


def test_native_jit_and_serialized_replay_follow_new_queue_and_changed_pixels():
  import pickle
  from tinygrad import Tensor, TinyJit
  from openpilot.selfdrive.modeld.egpu_yolo_model import native_packed_yuv_to_rgb, packed_yuv_to_rgb
  rng = np.random.default_rng(8)
  frames = [rng.integers(0, 256, (3, 6, 4, 8), dtype=np.uint8) for _ in range(3)]
  queues = [Tensor(frame).realize() for frame in frames]
  run = TinyJit(native_packed_yuv_to_rgb, prune=True)
  for queue in queues+queues:
    expected = packed_yuv_to_rgb(queue[-1], (8, 16)).numpy()
    np.testing.assert_allclose(run(queue).numpy(), expected, atol=1e-6)
  restored = pickle.loads(pickle.dumps(run))
  for queue in queues:
    expected = packed_yuv_to_rgb(queue[-1], (8, 16)).numpy()
    np.testing.assert_allclose(restored(queue).numpy(), expected, atol=1e-6)
  queues[0].assign(queues[1]).realize()
  np.testing.assert_allclose(restored(queues[0]).numpy(), packed_yuv_to_rgb(queues[1][-1], (8, 16)).numpy(), atol=1e-6)
