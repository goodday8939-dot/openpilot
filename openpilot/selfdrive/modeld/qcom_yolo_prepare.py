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


def compile_model(directory, input_nv12, camera_size, samples=60):
  os.environ.update(DEV='QCOM', WARP_DEV='QCOM', IMAGE='0', FLOAT16='1', NOLOCALS='1',
                    JIT_BATCH_SIZE='0', OPENPILOT_HACKS='1')
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

  sm = messaging.SubMaster(['managerState'])
  deadline = time.monotonic() + 5
  while time.monotonic() < deadline:
    sm.update(100)
    if sm.all_checks():
      break
  if not sm.all_checks():
    raise RuntimeError('fresh manager state required for maintenance compilation')
  if any(p.running and p.name in ('modeld', 'dmonitoringmodeld', 'qcom_yolod') for p in sm['managerState'].processes):
    raise RuntimeError('stop driving/DM/YOLO processes before compiling')
  if Params().get_bool('IsDriverViewEnabled'):
    raise RuntimeError('close driver camera preview before compiling')
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
            'onnx_sha256': manifest['sha256'], 'device': 'QCOM',
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
            'percentiles_ms': (np.percentile(elapsed, [50, 95, 99, 100]) * 1000).tolist(), 'serialization_passed': True}
  (directory / 'qcom_compile_report.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, default=Path('/data/egpu_yolo'))
  parser.add_argument('--input-nv12', type=Path, required=True)
  parser.add_argument('--camera-width', type=int, required=True)
  parser.add_argument('--camera-height', type=int, required=True)
  args = parser.parse_args()
  compile_model(args.directory, args.input_nv12, (args.camera_width, args.camera_height))


if __name__ == '__main__':
  main()
