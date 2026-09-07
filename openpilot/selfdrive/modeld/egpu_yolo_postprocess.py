"""Bounded CPU-only delivery for the driving owner's resident YOLO output."""
import os
import pickle
import socket
import subprocess
import sys


class OutputWorker:
  def __init__(self):
    self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    self.socket.setblocking(False)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
    try:
      self.process = subprocess.Popen([sys.executable, '-m', __name__, str(child.fileno())], pass_fds=(child.fileno(),),
                                      stdin=subprocess.DEVNULL)
    finally:
      child.close()

  def send(self, packet):
    if self.process.poll() is not None:
      raise RuntimeError('CPU-only YOLO output worker exited')
    data = pickle.dumps(packet, protocol=5)
    if len(data) > 131072:
      raise ValueError('YOLO output packet exceeds bound')
    try:
      return self.socket.send(data) == len(data)
    except BlockingIOError:
      return False


def project_detections(detections, transform, model_size, camera_size):
  """Batch corner projection for crowded frames; retain the small-frame path."""
  import numpy as np
  from openpilot.selfdrive.modeld.egpu_yolo import camera_detections
  if len(detections) < 2:
    return camera_detections(detections, transform, model_size, camera_size)
  if np.shape(transform) != (3, 3) or not np.isfinite(transform).all() or min(*model_size, *camera_size) <= 0:
    return []
  mw, mh = model_size
  boxes = np.array([[d[k] for k in ('x1', 'y1', 'x2', 'y2')] for d in detections])
  corners = np.ones((len(detections), 4, 3))
  corners[:, :, 0] = boxes[:, [0, 2, 2, 0]]*mw
  corners[:, :, 1] = boxes[:, [1, 1, 3, 3]]*mh
  corners = corners @ transform.T
  valid = np.isfinite(corners).all(axis=(1, 2)) & (corners[:, :, 2] > 1e-6).all(axis=1)
  indices = np.flatnonzero(valid)
  normalized = corners[indices, :, :2]/corners[indices, :, 2:]/np.asarray(camera_size)
  return [{**detections[index], 'cameraPoints': points.flatten().tolist()} for index, points in zip(indices, normalized, strict=True)
          if np.max(np.abs(points)) <= 10]


def decode_packet(packet):
  import numpy as np
  from openpilot.selfdrive.modeld.egpu_yolo import decode_detections, camera_time
  metadata, values, transform, size, names, started = packet
  if values is None:
    return metadata
  if values.shape != (1, 6, 2688) or values.dtype not in (np.float32, np.float16):
    raise ValueError('native compact FP32 or FP16 output required')
  cpu_start = camera_time()
  # Keep transport compact, but run filtering/NMS in FP32 on this CPU worker.
  detections = decode_detections(values.astype(np.float32, copy=False), 512, 256, compact=True)
  for detection in detections:
    detection['label'] = names[detection['classId']]
  metadata['detections'] = project_detections(detections, np.asarray(transform), (512, 256), size)
  metadata['postprocessTime'] = camera_time()-cpu_start
  metadata['executionTime'] = camera_time()-started
  return metadata


def main():
  # Reset inherited FIFO/CPU-7 policy before importing NumPy or cereal. This
  # process never imports tinygrad, opens an image buffer or owns a GPU device.
  os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
  os.sched_setaffinity(0, {4})
  from openpilot.cereal import messaging
  from openpilot.selfdrive.modeld.egpu_yolo import camera_time
  from openpilot.selfdrive.modeld.egpu_yolo_reuse import read_session
  stream = socket.socket(fileno=int(sys.argv[1]))
  pm = messaging.PubMaster(['carrotYolo'])
  while True:
    data = stream.recv(131073)
    if not data:
      return
    # Keep only the newest complete packet when CPU processing falls behind.
    while True:
      try:
        newer = stream.recv(131073, socket.MSG_DONTWAIT)
      except BlockingIOError:
        break
      if not newer:
        return
      data = newer
    if len(data) > 131072 or not read_session(enabled=False):
      continue
    # Only this owner's inherited private socket can supply pickle data.
    packet = pickle.loads(data)
    if camera_time()-packet[-1] > .25:
      continue
    try:
      metadata = decode_packet(packet)
    except Exception:
      from openpilot.common.swaglog import cloudlog
      cloudlog.exception('CPU YOLO postprocessing failed')
      return
    if camera_time()-packet[-1] > .25 or not read_session(enabled=False):
      continue
    msg = messaging.new_message('carrotYolo')
    msg.valid = metadata['state'] == 'run'
    msg.carrotYolo = metadata
    pm.send('carrotYolo', msg)


if __name__ == '__main__':
  main()
