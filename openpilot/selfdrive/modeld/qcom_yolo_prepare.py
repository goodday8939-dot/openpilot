#!/usr/bin/env python3
"""Compile internal YOLO during maintenance, using a saved NV12 camera frame."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def check_maintenance_state(processes, *, onroad, driver_preview, stationary_maintenance=False, parked=False):
  if onroad and not (stationary_maintenance and parked):
    raise RuntimeError('offroad required for maintenance compilation')
  stopped = {'camerad', 'modeld', 'dmonitoringmodeld', 'dmonitoringd', 'qcom_yolod'}
  if stationary_maintenance:
    if not parked:
      raise RuntimeError('fresh stationary park state required for maintenance')
    stopped.update(('controlsd', 'selfdrived', 'joystickd', 'maneuversd', 'lateral_maneuversd'))
  if any(p.running and p.name in stopped for p in processes):
    raise RuntimeError('stop camera/driving/DM/YOLO processes before compiling')
  if driver_preview:
    raise RuntimeError('close driver camera preview before compiling')


def compile_model(directory, input_nv12, camera_size, samples=60, stationary_maintenance=False):
  from openpilot.selfdrive.modeld.qcom_yolo_model import BACKEND, configure_environment
  configure_environment(directory)
  from openpilot.cereal import messaging
  from openpilot.common.params import Params
  from openpilot.selfdrive.modeld.egpu_yolo import decode_detections
  from openpilot.selfdrive.modeld.qcom_yolo_model import configure_compiler, make_runner
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
  import openpilot.selfdrive.modeld.compile_modeld  # noqa: F401 -- serialization support
  from tinygrad import Tensor, TinyJit
  from tinygrad.nn.onnx import OnnxRunner
  configure_compiler()

  sm = messaging.SubMaster(['managerState', 'deviceState'] + (['carState'] if stationary_maintenance else []))
  deadline = time.monotonic() + 5
  while time.monotonic() < deadline:
    sm.update(100)
    if sm.all_checks():
      break
  if not sm.all_checks():
    raise RuntimeError('fresh manager state required for maintenance compilation')
  params = Params()
  parked = (stationary_maintenance and sm['carState'].standstill and abs(sm['carState'].vEgo) < .01
            and str(sm['carState'].gearShifter) == 'park')
  check_maintenance_state(sm['managerState'].processes, onroad=params.get_bool('IsOnroad') or sm['deviceState'].started,
                          driver_preview=params.get_bool('IsDriverViewEnabled'),
                          stationary_maintenance=stationary_maintenance, parked=parked)
  manifest = json.loads((directory / 'manifest.json').read_text())
  source = directory / 'model.onnx'
  if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['sha256']:
    raise ValueError('ONNX checksum mismatch')
  model_size = manifest['width'], manifest['height']
  if model_size != (512, 256):
    raise ValueError('expected 512x256 YOLO model')
  layout = (*camera_size, *get_nv12_info(*camera_size))
  frame = np.load(input_nv12, allow_pickle=False)
  if frame.dtype != np.uint8 or frame.shape != (layout[-1],):
    raise ValueError('saved NV12 frame does not match camera layout')
  run = TinyJit(make_runner(OnnxRunner(source), layout, model_size), prune=True)
  tensor = Tensor(frame, device='QCOM').realize()
  for _ in range(3):
    raw = run(tensor).numpy()
  # Offroad power management can leave only CPUs 0-3 online. Preserve that
  # affinity instead of failing after compilation by requesting offline CPU 4.
  if 4 in os.sched_getaffinity(0):
    os.sched_setaffinity(0, {4})
  os.nice(10)
  elapsed = []
  for _ in range(samples):
    start = time.monotonic()
    raw = run(tensor).numpy()
    decode_detections(raw, *model_size, compact=True)
    elapsed.append(time.monotonic() - start)
    time.sleep(.05)
  bundle = {'version': 1, 'run': run, 'layout': layout, 'model_id': manifest['model_id'],
            'width': model_size[0], 'height': model_size[1], 'names': manifest['names'],
            'onnx_sha256': manifest['sha256'], 'device': BACKEND,
            'adapter_sha256': hashlib.sha256(Path(__file__).with_name('qcom_yolo_model.py').read_bytes()).hexdigest()}
  target = directory / 'yolo_qcom.pkl'
  temporary = target.with_suffix('.pkl.tmp')
  with temporary.open('wb') as f:
    dump_oob(bundle, f)
  with temporary.open('rb') as f:
    restored = load_oob(f)
  np.testing.assert_allclose(restored['run'](tensor).numpy(), raw, rtol=1e-3, atol=1e-3)
  temporary.replace(target)
  report = {'model_id': manifest['model_id'], 'layout': layout, 'elapsed_seconds': elapsed,
            'percentiles_ms': (np.percentile(elapsed, [50, 95, 99, 100]) * 1000).tolist(), 'serialization_passed': True,
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'adapter_sha256': bundle['adapter_sha256'],
            'onnx_sha256': bundle['onnx_sha256']}
  np.savez_compressed(directory / 'qcom_compile_validation.npz', raw=raw)
  (directory / 'qcom_compile_report.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, default=Path('/data/egpu_yolo'))
  parser.add_argument('--input-nv12', type=Path, required=True)
  parser.add_argument('--camera-width', type=int, required=True)
  parser.add_argument('--camera-height', type=int, required=True)
  parser.add_argument('--stationary-maintenance', action='store_true',
                      help='allow ignition ON only with fresh park state and stopped camera/inference/control processes')
  args = parser.parse_args()
  compile_model(args.directory, args.input_nv12, (args.camera_width, args.camera_height),
                stationary_maintenance=args.stationary_maintenance)


if __name__ == '__main__':
  main()
