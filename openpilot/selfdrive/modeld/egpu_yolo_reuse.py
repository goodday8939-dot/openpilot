"""Manually leased, stationary reuse of the driving owner's existing eGPU queue."""
import json
import math

from openpilot.selfdrive.modeld.egpu_yolo import YoloRuntime, artifact_path, camera_detections, camera_time, decode_detections


def session_path():
  return artifact_path().parent / 'reuse_session.json'


def leased(value, now, *, enabled=True):
  expiry = value.get('expires', 0)
  return (isinstance(expiry, (float, int)) and math.isfinite(expiry) and 0 < expiry-now <= 5
          and value.get('prepared') is True and (not enabled or value.get('enabled') is True))


def read_session(*, enabled=True):
  try:
    value = json.loads(session_path().read_text())
    return isinstance(value, dict) and leased(value, camera_time(), enabled=enabled)
  except (OSError, ValueError, TypeError):
    return False


def stationary_permitted(*, fresh, started, parked, standstill, speed, enabled, lat_active, long_active,
                         primary_seconds, dropped):
  return (fresh and started and parked and standstill and math.isfinite(speed) and abs(speed) < .01
          and not enabled and not lat_active and not long_active
          and math.isfinite(primary_seconds) and 0 < primary_seconds < .06 and not dropped)


class ReuseRuntime(YoloRuntime):
  @classmethod
  def load(cls, input_queue):
    directory = artifact_path().parent
    target = directory / 'egpu2-reuse/yolo_reuse.pkl'
    if not read_session(enabled=False) or not target.is_file() or (directory/'qcom_enabled').exists():
      return None
    from openpilot.selfdrive.modeld.helpers import load_oob
    with target.open('rb') as stream:
      bundle = load_oob(stream)
    if not bundle.get('native') or (bundle['width'], bundle['height']) != (input_queue.shape[-1]*2, input_queue.shape[-2]*2):
      raise ValueError('reuse artifact must retain native driving input resolution')
    # This is a compiled artifact replay on the driving owner's startup thread.
    # No ONNX runner or compiler is invoked on live camera frames.
    runtime = cls(input_queue, bundle)
    from openpilot.common.swaglog import cloudlog
    cloudlog.info('resident YOLO prepared: %s, required %.3f ms', bundle['variant'], runtime.budget.estimate*1000+.001*1000)
    return runtime

  def infer(self):
    from openpilot.selfdrive.modeld.egpu_yolo_usb import run_yolo, read_yolo
    start = camera_time()
    raw = run_yolo(self.run, self.queue)
    submitted = camera_time()
    values = read_yolo(raw)
    copied = camera_time()
    detections = decode_detections(values, self.width, self.height, compact=True)
    for detection in detections:
      detection['label'] = self.names[detection['classId']]
    self.phases = (submitted-start, copied-submitted, camera_time()-copied)
    return detections

  def after_publish(self, pm, frame_id, sof_ns, eof_ns, received, driving_published, dropped, camera,
                    transform, camera_size, next_frame_ready, inference_started, inference_ended, *, permitted):
    from openpilot.cereal import messaging
    self.budget.observe(frame_id, sof_ns/1e9, received, dropped)
    if not permitted or not read_session(enabled=False):
      return
    enabled = read_session()
    pending = next_frame_ready() if enabled else False
    start = camera_time()
    reason = self.budget.admit(start, pending) if enabled else 'paused'
    detections = []
    if reason == 'run':
      try:
        detections = camera_detections(self.infer(), transform, (self.width, self.height), camera_size)
      except Exception:
        from openpilot.common.swaglog import cloudlog
        cloudlog.exception('optional resident YOLO failed; disabling this session')
        self.budget.disabled_reason = reason = 'error'
      self.last_execution = camera_time()-start
    elif start-self.last_publish < 1:
      return
    msg = messaging.new_message('carrotYolo')
    msg.valid = reason == 'run'
    msg.carrotYolo = {
      'frameId': frame_id, 'timestampSof': sof_ns, 'timestampEof': eof_ns,
      'modelId': self.model_id, 'camera': camera, 'state': reason,
      'executionTime': self.last_execution, 'budgetTime': max(0, self.budget.deadline-start),
      'drivingPublishTime': int(driving_published*1e9), 'drivingLatency': max(0, driving_published-eof_ns/1e9),
      'inputReadyTime': int(received*1e9), 'inferenceStartTime': int(inference_started*1e9),
      'inferenceEndTime': int(inference_ended*1e9), 'deadlineTime': int(max(0, self.budget.deadline)*1e9),
      'requiredTime': self.budget.estimate+.001, 'runs': self.budget.runs+int(reason == 'run'),
      'skipped': self.budget.skipped, 'overruns': self.budget.overruns,
      'cameraWidth': camera_size[0], 'cameraHeight': camera_size[1], 'detections': detections,
      'submitTime': self.phases[0], 'readbackTime': self.phases[1], 'postprocessTime': self.phases[2],
    }
    pm.send('carrotYolo', msg)
    self.last_publish = camera_time()
    if reason == 'run':
      self.budget.finish(start, self.last_publish)
