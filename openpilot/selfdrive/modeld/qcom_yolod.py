#!/usr/bin/env python3
"""Optional internal-GPU detections, independent of the driving inference loop."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time

import numpy as np

from openpilot.selfdrive.modeld.egpu_yolo import camera_detections, camera_time, decode_detections
from openpilot.selfdrive.modeld.qcom_yolo_model import BACKEND, camera_transform, configure_environment

DIRECTORY = Path('/data/egpu_yolo')
INTERVAL = .25
MAX_FRAME_AGE = .2
# 40 ms is the optimization target. Cooperative batches yield to driving/UI;
# this total frame limit also gets checked before each new batch submission.
MAX_RUNTIME = .25


def permit_reason(*, onroad, egpu_active, loading, dm_disabled, dm_running, model_alive, manager_alive):
  if not onroad:
    return 'offroad'
  if not egpu_active or loading:
    return 'egpu_wait'
  if not dm_disabled or dm_running:
    return 'dm_active'
  if not model_alive or not manager_alive:
    return 'warming'
  return 'run'


def configured(params, directory=DIRECTORY):
  return (params.get_bool('UsbGpuActive') and not params.get_bool('UsbGpuLoading')
          and params.get_int('DisableDM') in (1, 2)
          and (directory / 'qcom_enabled').is_file() and (directory / 'yolo_qcom.pkl').is_file()
          and not (directory / 'qcom_fault').exists())


def main(prepared_bundle=None, commissioning=False):
  try:
    configure_environment(DIRECTORY)
  except (OSError, ValueError) as exc:
    (DIRECTORY / 'qcom_fault').write_text(str(exc))
    return
  from openpilot.cereal import messaging
  from openpilot.common.params import Params
  from openpilot.common.swaglog import cloudlog
  from openpilot.selfdrive.modeld.qcom_yolo_runtime import YoloInterrupted, configure_runtime, is_prepared
  from msgq.visionipc import VisionIpcClient, VisionStreamType
  from tinygrad import Tensor

  # Ordinary scheduling on a non-driving CPU core. GPU execution still shares
  # memory and the QCOM queue with the driving image warp and UI.
  os.sched_setaffinity(0, {4})
  os.nice(10)
  params = Params()
  sm = messaging.SubMaster(['modelV2', 'managerState', 'deviceState'] + (['carState'] if commissioning else []),
                          ignore_avg_freq=['carState'] if commissioning else [])
  pm = messaging.PubMaster(['carrotYolo'])
  client = VisionIpcClient('camerad', VisionStreamType.VISION_STREAM_ROAD, True)
  bundle = prepared_bundle
  runs = skipped = 0
  last_run = -float('inf')
  last_status = -float('inf')
  failed = False
  frame_start = None

  def reason():
    sm.update(0)
    if commissioning and (not sm.all_checks(['carState']) or not sm['carState'].standstill
                          or abs(sm['carState'].vEgo) >= .01 or str(sm['carState'].gearShifter) != 'park'):
      return 'stopped'
    dm_running = any(p.running and p.name in ('dmonitoringmodeld', 'dmonitoringd') for p in sm['managerState'].processes)
    return permit_reason(onroad=params.get_bool('IsOnroad') and sm['deviceState'].started,
                         egpu_active=params.get_bool('UsbGpuActive'), loading=params.get_bool('UsbGpuLoading'),
                         dm_disabled=params.get_int('DisableDM') in (1, 2), dm_running=dm_running,
                         model_alive=sm.all_checks(['modelV2', 'deviceState']), manager_alive=sm.all_checks(['managerState']))

  def publish(state, **values):
    nonlocal last_status
    msg = messaging.new_message('carrotYolo')
    msg.valid = state == 'run'
    msg.carrotYolo = {'modelId': bundle['model_id'] if bundle else '', 'camera': 'road',
                     'state': state, 'runs': runs, 'skipped': skipped, **values}
    pm.send('carrotYolo', msg)
    last_status = time.monotonic()

  def requested():
    return (commissioning and not (DIRECTORY / 'qcom_fault').exists()) or configured(params)

  def admission():
    if not requested() or reason() != 'run':
      return False
    if sm['modelV2'].modelExecutionTime > .06 or sm['modelV2'].frameDropPerc > 1:
      raise RuntimeError('driving latency/drop guard exceeded during optional YOLO')
    if frame_start is not None and camera_time() - frame_start > MAX_RUNTIME:
      raise RuntimeError('cooperative YOLO frame exceeded runtime budget')
    return True

  try:
    expected = hashlib.sha256(Path(__file__).with_name('qcom_yolo_model.py').read_bytes()).hexdigest()
    runtime_hash = hashlib.sha256(Path(__file__).with_name('qcom_yolo_runtime.py').read_bytes()).hexdigest()
    if (bundle is None or bundle['version'] != 2 or bundle['device'] != BACKEND or bundle['adapter_sha256'] != expected
        or bundle.get('runtime_sha256') != runtime_hash or not bundle.get('cooperative') or not is_prepared(bundle['run'])):
      raise ValueError('internal YOLO requires verified maintenance prewarm in this process')
    configure_runtime(admission)
    while requested():
      state = reason()
      now = time.monotonic()
      if state != 'run':
        if now - last_status > 1:
          publish(state)
        time.sleep(.02)
        continue
      if now - last_run < INTERVAL:
        time.sleep(.01)
        continue
      if not client.is_connected() and not client.connect(False):
        time.sleep(.1)
        continue
      buf = client.recv(timeout_ms=100)
      if buf is None:
        continue
      frame_id, sof, eof = client.frame_id, client.timestamp_sof, client.timestamp_eof
      start = camera_time()
      cw, ch, stride, y_height, _uv_height, size = bundle['layout']
      if (client.width, client.height, buf.stride, buf.uv_offset) != (cw, ch, stride, stride * y_height):
        raise ValueError('camera layout changed; recompile internal YOLO')
      if not 0 <= start - eof / 1e9 <= MAX_FRAME_AGE:
        skipped += 1
        continue
      # Own the snapshot: a slow optional inference must not read a recycled
      # VisionIPC ring buffer. No GPU data comes back from the eGPU.
      pixels = np.frombuffer(buf.data, dtype=np.uint8).copy()
      if pixels.size != size:
        raise ValueError('unexpected camera buffer length')
      if not requested() or reason() != 'run':
        continue
      last_run = time.monotonic()
      frame_start = start
      try:
        tensor = Tensor(pixels, device='QCOM').realize()
        raw = bundle['run'](tensor).numpy()
      except YoloInterrupted:
        skipped += 1
        continue
      finally:
        frame_start = None
      detections = decode_detections(raw, bundle['width'], bundle['height'], compact=True)
      for detection in detections:
        detection['label'] = bundle['names'][detection['classId']]
      detections = camera_detections(detections, camera_transform((cw, ch), (bundle['width'], bundle['height'])),
                                     (bundle['width'], bundle['height']), (cw, ch))
      elapsed = camera_time() - start
      runs += 1
      if elapsed > MAX_RUNTIME:
        raise RuntimeError(f'internal YOLO exceeded {MAX_RUNTIME}s: {elapsed:.4f}s')
      # A mode transition revokes new work and suppresses obsolete results.
      # Already submitted GPU work is not claimed to be preemptible.
      if not requested() or reason() != 'run' or camera_time() - eof / 1e9 > .35:
        continue
      publish('run', frameId=frame_id, timestampSof=sof, timestampEof=eof, executionTime=elapsed,
              cameraWidth=cw, cameraHeight=ch, detections=detections)
  except Exception as exc:
    failed = True
    (DIRECTORY / 'qcom_fault').write_text(str(exc))
    cloudlog.exception('optional internal YOLO stopped')
    publish('error')
  finally:
    if not failed:
      publish('stopped')


if __name__ == '__main__':
  main()
