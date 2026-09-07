"""Convert a verified QCOM YOLO artifact to measured, cooperative GPU batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np


def prepare(directory, input_nv12, stationary_maintenance=False):
  from openpilot.selfdrive.modeld.qcom_yolo_model import BACKEND, configure_environment
  configure_environment(directory)
  if 4 in os.sched_getaffinity(0):
    os.sched_setaffinity(0, {4})
  os.nice(10)
  from openpilot.cereal import messaging
  from openpilot.common.params import Params
  from openpilot.selfdrive.modeld.qcom_yolo_prepare import check_maintenance_state
  sm = messaging.SubMaster(['managerState', 'deviceState'] + (['carState'] if stationary_maintenance else []))
  deadline = time.monotonic()+10
  while time.monotonic() < deadline:
    sm.update(100)
    if sm.all_checks():
      break
  if not sm.all_checks():
    raise RuntimeError('fresh maintenance state required')
  params = Params()
  parked = (stationary_maintenance and sm['carState'].standstill and abs(sm['carState'].vEgo) < .01
            and str(sm['carState'].gearShifter) == 'park')
  check_maintenance_state(sm['managerState'].processes, onroad=params.get_bool('IsOnroad') or sm['deviceState'].started,
                          driver_preview=params.get_bool('IsDriverViewEnabled'),
                          stationary_maintenance=stationary_maintenance, parked=parked)

  import openpilot.selfdrive.modeld.compile_modeld  # noqa: F401 -- serialization support
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
  from openpilot.selfdrive.modeld.egpu_yolo import decode_detections
  from openpilot.selfdrive.modeld.qcom_yolo_runtime import configure_runtime, repartition
  from tinygrad import Tensor
  from tinygrad.device import Compiled, Device, ProfileGraphEvent
  from tinygrad.engine.realize import graph_cache
  from tinygrad.helpers import Context

  target = directory/'yolo_qcom.pkl'
  with target.open('rb') as f:
    bundle = load_oob(f)
  adapter = Path(__file__).with_name('qcom_yolo_model.py')
  if (bundle['device'] != BACKEND or bundle['version'] not in (1, 2)
      or bundle['adapter_sha256'] != hashlib.sha256(adapter.read_bytes()).hexdigest()
      or bundle['onnx_sha256'] != hashlib.sha256((directory/'model.onnx').read_bytes()).hexdigest()):
    raise ValueError('verified current QCOM artifact required')
  pixels = np.load(input_nv12, allow_pickle=False)
  if pixels.dtype != np.uint8 or pixels.shape != (bundle['layout'][-1],):
    raise ValueError('saved NV12 layout mismatch')
  tensor = Tensor(pixels, device='QCOM').realize()
  for _ in range(3):
    reference = bundle['run'](tensor).numpy()
  graph_cache.clear()
  Compiled.profile_events.clear()
  with Context(PROFILE=1):
    for _ in range(3):
      bundle['run'](tensor).numpy()
    Device['QCOM'].synchronize()
    graphs = [event for event in Compiled.profile_events if isinstance(event, ProfileGraphEvent)]
    if not graphs:
      raise RuntimeError('GPU kernel profile missing')
    # The source artifact is the original single graph, not a prior partition.
    if bundle['version'] != 1:
      raise ValueError('use the original unsliced artifact when changing batch policy')
    event = graphs[-1]
    profile = [{'name': str(entry.name), 'ms': float(event.sigs[entry.en_id]-event.sigs[entry.st_id])/1000}
               for entry in event.ents]
    graph_cache.clear()
  run = repartition(bundle['run'], profile)
  configure_runtime()
  for _ in range(3):
    np.testing.assert_array_equal(run(tensor).numpy(), reference)
  elapsed = []
  for _ in range(60):
    start = time.monotonic()
    raw = run(tensor).numpy()
    decode_detections(raw, bundle['width'], bundle['height'], compact=True)
    elapsed.append((time.monotonic()-start)*1000)
    time.sleep(.05)
  bundle.update(version=2, run=run, cooperative=True, batch_budget_ms=8.,
                runtime_sha256=hashlib.sha256(Path(__file__).with_name('qcom_yolo_runtime.py').read_bytes()).hexdigest())
  temporary = target.with_suffix('.pkl.tmp')
  with temporary.open('wb') as f:
    dump_oob(bundle, f)
  with temporary.open('rb') as f:
    restored = load_oob(f)
  np.testing.assert_array_equal(restored['run'](tensor).numpy(), reference)
  backup = directory/'yolo_qcom_unsliced.pkl'
  if not backup.exists():
    shutil.copy2(target, backup)
  temporary.replace(target)
  np.savez_compressed(directory/'qcom_cooperative_validation.npz', raw=raw)
  report = {'percentiles_ms': np.percentile(elapsed, [50, 95, 99, 100]).tolist(), 'times_ms': elapsed,
            'batches': len(run.captured.linear.src), 'kernel_profile': profile, 'bit_exact': True,
            'runtime_sha256': bundle['runtime_sha256'], 'adapter_sha256': bundle['adapter_sha256']}
  (directory/'qcom_cooperative_report.json').write_text(json.dumps(report, indent=2))
  print(json.dumps({key: value for key, value in report.items() if key not in ('times_ms', 'kernel_profile')}), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, default=Path('/data/egpu_yolo'))
  parser.add_argument('--input-nv12', type=Path, required=True)
  parser.add_argument('--stationary-maintenance', action='store_true')
  args = parser.parse_args()
  prepare(args.directory, args.input_nv12, args.stationary_maintenance)
