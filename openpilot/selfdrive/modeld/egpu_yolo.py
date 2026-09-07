"""Best-effort YOLO after the driving publications, on the same USB GPU owner.

The experimental result is observational only. Nothing in controls subscribes
to it. Preparation/compilation is an explicit maintenance operation.
"""
from __future__ import annotations

import math
import hashlib
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

ARTIFACT_VERSION = 1
FRAME_PERIOD = 0.05
MIN_INTERVAL = 0.2
GUARD = 0.005
MAX_DETECTIONS = 40
MAX_CANDIDATES = 200


def camera_time() -> float:
  """Camera/cereal timestamps use CLOCK_BOOTTIME, including suspended time."""
  return time.clock_gettime(time.CLOCK_BOOTTIME) if hasattr(time, "CLOCK_BOOTTIME") else time.monotonic()


def artifact_path() -> Path:
  return Path(os.getenv("EGPU_YOLO_DIR", "/data/egpu_yolo")) / "yolo.pkl"


def source_fingerprint() -> str:
  directory = Path(__file__).parent
  digest = hashlib.sha256()
  for name in ("egpu_yolo.py", "egpu_yolo_model.py", "egpu_yolo_prepare.py"):
    digest.update((directory / name).read_bytes())
  root = directory.parents[2]
  for name in ("engine/jit.py", "runtime/ops_amd.py", "runtime/graph/hcq.py"):
    digest.update((root / "tinygrad_repo/tinygrad" / name).read_bytes())
  return digest.hexdigest()


class IdleBudget:
  """Use the camera's cadence, not a fresh 50 ms allowance after a late recv."""
  def __init__(self, measured_seconds: float):
    if not math.isfinite(measured_seconds) or measured_seconds <= 0:
      raise ValueError("YOLO needs a measured positive runtime")
    self.estimate = measured_seconds * 1.4
    self.offsets: deque[float] = deque(maxlen=120)
    self.last_frame = -1
    self.last_sof = 0.0
    self.last_run = -math.inf
    self.deadline = 0.0
    self.settled = 0
    self.runs = self.skipped = self.overruns = 0
    self.disabled_reason = ""

  def observe(self, frame_id: int, sof: float, received: float, dropped: bool = False):
    discontinuity = self.last_frame >= 0 and (frame_id != self.last_frame + 1 or not 0.025 <= sof - self.last_sof <= 0.075)
    self.last_frame, self.last_sof = frame_id, sof
    if not all(math.isfinite(x) for x in (sof, received)) or sof <= 0 or not 0 <= received - sof < 1:
      self.offsets.clear()
      self.settled = 0
      self.deadline = 0
      return
    if discontinuity or dropped:
      self.settled = 0
    else:
      self.settled += 1
    self.offsets.append(received - sof)
    self.deadline = min(received + FRAME_PERIOD, sof + min(self.offsets) + FRAME_PERIOD)

  def admit(self, now: float) -> str:
    if self.disabled_reason:
      reason = self.disabled_reason
    elif self.settled < 20:
      reason = "warming"
    elif now - self.last_run < MIN_INTERVAL:
      return "rate_limit"
    elif now + self.estimate + GUARD > self.deadline:
      reason = "no_budget"
    else:
      self.last_run = now
      return "run"
    self.skipped += 1
    return reason

  def finish(self, start: float, end: float):
    self.runs += 1
    self.estimate = max(self.estimate, (end - start) * 1.4)
    if end + GUARD > self.deadline:
      self.overruns += 1
      self.disabled_reason = "overrun"


def decode_detections(raw: np.ndarray, width: int, height: int, confidence: float = 0.35, *, compact: bool = False) -> list[dict]:
  """YOLOv8 raw [1, 4 + classes, anchors], bounded class-aware NMS.

  Top candidates are selected before the quadratic step. Coordinates refer to
  the warped image actually inferred, never to an unrelated live camera frame.
  """
  raw = np.asarray(raw)
  if raw.ndim != 3 or raw.shape[0] != 1 or not 5 <= raw.shape[1] <= 1004 or width <= 0 or height <= 0:
    raise ValueError(f"unsupported YOLO output {raw.shape}")
  if not np.isfinite(raw).all():
    raise ValueError("nonfinite YOLO output")
  predictions = raw[0].T
  if compact:
    if raw.shape[1] != 6 or (predictions[:, 5] < 0).any() or not np.equal(predictions[:, 5], np.floor(predictions[:, 5])).all():
      raise ValueError("invalid compact YOLO classes")
    classes, scores = predictions[:, 5].astype(np.int32), predictions[:, 4]
  else:
    classes = predictions[:, 4:].argmax(axis=1)
    scores = predictions[np.arange(len(predictions)), classes + 4]
  ids = np.flatnonzero((scores >= confidence) & (scores <= 1) & (predictions[:, 2] > 0) & (predictions[:, 3] > 0))
  ids = ids[np.argsort(-scores[ids], kind="stable")[:MAX_CANDIDATES]]
  xywh = predictions[ids, :4].astype(np.float32)
  boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
  boxes /= np.array([width, height, width, height], dtype=np.float32)
  boxes = np.clip(boxes, 0, 1)
  pending = list(range(len(ids)))
  result = []
  while pending and len(result) < MAX_DETECTIONS:
    i = pending.pop(0)
    box = boxes[i]
    if np.any(box[2:] <= box[:2]):
      continue
    result.append({"classId": int(classes[ids[i]]), "confidence": float(scores[ids[i]]),
                   "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])})
    if pending:
      other = boxes[pending]
      inter = np.maximum(0, np.minimum(box[2:], other[:, 2:]) - np.maximum(box[:2], other[:, :2])).prod(axis=1)
      union = (box[2:] - box[:2]).prod() + np.maximum(0, other[:, 2:] - other[:, :2]).prod(axis=1) - inter
      suppress = (inter / np.maximum(union, 1e-9) > 0.45) & (classes[ids[pending]] == classes[ids[i]])
      pending = [j for j, remove in zip(pending, suppress, strict=True) if not remove]
  return result


def camera_detections(detections: list[dict], transform: np.ndarray, model_size: tuple[int, int],
                      camera_size: tuple[int, int]) -> list[dict]:
  """Map all four warped-box corners back to the original camera coordinates."""
  if np.shape(transform) != (3, 3) or not np.isfinite(transform).all() or min(*model_size, *camera_size) <= 0:
    return []
  result = []
  mw, mh = model_size
  cw, ch = camera_size
  for detection in detections:
    x1, y1, x2, y2 = (detection[k] for k in ("x1", "y1", "x2", "y2"))
    corners = np.array([[x1 * mw, y1 * mh, 1], [x2 * mw, y1 * mh, 1],
                        [x2 * mw, y2 * mh, 1], [x1 * mw, y2 * mh, 1]]) @ transform.T
    if not np.isfinite(corners).all() or (corners[:, 2] <= 1e-6).any():
      continue
    normalized = corners[:, :2] / corners[:, 2:] / np.array([cw, ch])
    if np.max(np.abs(normalized)) > 10:
      continue
    result.append({**detection, "cameraPoints": normalized.flatten().tolist()})
  return result


class YoloRuntime:
  def __init__(self, input_queue, bundle):
    if (bundle["version"] != ARTIFACT_VERSION or bundle.get("source_fingerprint") != source_fingerprint()
        or tuple(bundle["queue_shape"]) != tuple(input_queue.shape)):
      raise ValueError("YOLO artifact does not match driving input queue")
    self.queue = input_queue
    self.run = bundle["run"]
    self.width, self.height = bundle["width"], bundle["height"]
    self.names = bundle["names"]
    self.model_id = bundle["model_id"]
    self.budget = IdleBudget(bundle["measured_seconds"])
    self.last_publish = 0.0
    self.last_execution = 0.0
    # JIT replay/queue initialization must finish before entering the main loop.
    for i in range(3):
      start = time.monotonic()
      self.infer()
      if i > 0:
        self.budget.estimate = max(self.budget.estimate, (time.monotonic() - start) * 1.4)

  @classmethod
  def load(cls, input_queue):
    if os.getenv("EGPU_YOLO", "1") == "0" or not artifact_path().is_file():
      return None
    from openpilot.selfdrive.modeld.helpers import load_oob
    with artifact_path().open("rb") as f:
      bundle = load_oob(f)
    return cls(input_queue, bundle)

  def infer(self):
    raw = self.run(queue=self.queue)
    detections = decode_detections(raw.numpy(), self.width, self.height, compact=True)
    for detection in detections:
      detection["label"] = self.names[detection["classId"]]
    return detections

  def after_publish(self, pm, frame_id: int, sof_ns: int, eof_ns: int, received: float,
                    driving_published: float, dropped: bool, camera: str, transform: np.ndarray, camera_size: tuple[int, int]):
    from openpilot.cereal import messaging
    self.budget.observe(frame_id, sof_ns / 1e9, received, dropped)
    start = camera_time()
    reason = self.budget.admit(start)
    detections = []
    if reason == "run":
      try:
        detections = camera_detections(self.infer(), transform, (self.queue.shape[3] * 2, self.queue.shape[2] * 2), camera_size)
      except Exception:
        from openpilot.common.swaglog import cloudlog
        cloudlog.exception("optional YOLO failed; disabling it for this modeld session")
        self.budget.disabled_reason = "error"
        reason = "error"
      self.last_execution = camera_time() - start
    elif start - self.last_publish < 1.0:
      return
    msg = messaging.new_message("carrotYolo")
    msg.valid = reason == "run"
    msg.carrotYolo = {
      "frameId": frame_id, "timestampSof": sof_ns, "timestampEof": eof_ns,
      "modelId": self.model_id, "camera": camera, "state": reason,
      "executionTime": self.last_execution, "budgetTime": max(0, self.budget.deadline - start),
      "drivingPublishTime": int(driving_published * 1e9), "drivingLatency": max(0, driving_published - eof_ns / 1e9),
      "runs": self.budget.runs + int(reason == "run"), "skipped": self.budget.skipped, "overruns": self.budget.overruns,
      "cameraWidth": camera_size[0], "cameraHeight": camera_size[1], "detections": detections,
    }
    pm.send("carrotYolo", msg)
    self.last_publish = camera_time()
    if reason == "run":
      self.budget.finish(start, self.last_publish)
