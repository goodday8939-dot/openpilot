#!/usr/bin/env python3
"""Compare resident driving-queue adapters during exclusive parked maintenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from openpilot.tools.egpu_yolo.benchmark_saved import check_state, stats


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, required=True)
  parser.add_argument('--input-packed', type=Path, required=True)
  parser.add_argument('--samples', type=int, default=3000)
  parser.add_argument('--profile-replays', type=int, default=0,
                      help='profile additional replays after timing; never include them in latency samples')
  parser.add_argument('--variants', nargs='+', choices=('original_fp32', 'native_fp32', 'native_fp16'),
                      default=['original_fp32', 'native_fp32', 'native_fp16'])
  parser.add_argument('--stationary-maintenance', action='store_true')
  args = parser.parse_args()
  if args.samples < 1000:
    parser.error('at least 1000 samples are required')
  check_state(args.stationary_maintenance)
  out = args.directory
  manifest = json.loads((out/'manifest.json').read_text())
  source = out/'model.onnx'
  if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['sha256']:
    raise ValueError('model checksum mismatch')
  if (manifest['width'], manifest['height']) != (512, 256):
    raise ValueError('native 512x256 model required')
  pixels = np.load(args.input_packed, allow_pickle=False)
  if pixels.shape != (6, 128, 256) or pixels.dtype != np.uint8:
    raise ValueError('native packed driving frame required')
  os.environ.update(DEV='USB+AMD:LLVM', GMMU='0', FLOAT16='1', JIT_BATCH_SIZE='0', TC_OPT='2')
  import openpilot.selfdrive.modeld.compile_modeld  # noqa: F401
  from openpilot.common.file_chunker import open_file_chunked
  from openpilot.common.realtime import config_realtime_process
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.selfdrive.modeld.egpu_yolo import ARTIFACT_VERSION, source_fingerprint, decode_detections
  from openpilot.selfdrive.modeld.egpu_yolo_model import make_yolo_runner
  from openpilot.selfdrive.modeld.egpu_yolo_usb import run_yolo, read_yolo
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob, usbgpu_compiled_path
  from tinygrad import Tensor, TinyJit
  from tinygrad.device import Device
  from tinygrad.nn.onnx import OnnxRunner

  with open_file_chunked(usbgpu_compiled_path()) as f:
    driving = load_oob(f)
  img = driving['metadata']['input_shapes']['img']
  skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  shape = (skip*(img[1]//6-1)+1, 6, img[2], img[3])
  host_queue = np.broadcast_to(pixels, shape).copy()
  queue = Tensor(host_queue, device=Device.DEFAULT).realize()
  variants = {}
  for name, native, dtype in [('original_fp32', False, 'float32'), ('native_fp32', True, 'float32'), ('native_fp16', True, 'float16')]:
    if name not in args.variants:
      continue
    print('compiling', name, flush=True)
    run = TinyJit(make_yolo_runner(OnnxRunner(source), 512, 256, native=native, output_dtype=dtype), prune=True)
    cold = []
    for _ in range(4):
      start = time.perf_counter()
      raw = read_yolo(run_yolo(run, queue))
      cold.append((time.perf_counter()-start)*1000)
    variants[name] = {'run': run, 'native': native, 'output_dtype': dtype, 'cold_ms': cold,
                      'samples': [], 'raw': raw.copy(), 'output_bytes': raw.nbytes}
    print('prepared', name, cold, flush=True)
  reference = next(iter(variants.values()))['raw']
  for value in variants.values():
    relevant = np.maximum(reference[:, 4], value['raw'][:, 4]) >= .35
    np.testing.assert_allclose(value['raw'][:, :4], reference[:, :4], atol=.5, rtol=.001)
    np.testing.assert_allclose(value['raw'][:, 4], reference[:, 4], atol=.001, rtol=0)
    np.testing.assert_array_equal(value['raw'][:, 5][relevant], reference[:, 5][relevant])
  # A same-buffer/same-pixels serialization check cannot detect a graph that
  # retained the compiler's image pointer. Check new allocations AND mutation
  # of the original allocation against an uncaptured evaluation.
  changed = host_queue.copy()
  changed[:, :4] = 255-changed[:, :4]
  other_queue = Tensor(changed, device=Device.DEFAULT).realize()
  for value in variants.values():
    expected = read_yolo(run_yolo(value['run'].fxn, other_queue))
    assert not np.allclose(expected, value['raw'], rtol=1e-3, atol=1e-3), 'challenge image did not change the result'
    np.testing.assert_allclose(read_yolo(run_yolo(value['run'], other_queue)), expected, rtol=1e-3, atol=1e-3)
    queue.assign(other_queue).realize()
    np.testing.assert_allclose(read_yolo(run_yolo(value['run'], queue)), expected, rtol=1e-3, atol=1e-3)
    queue.assign(Tensor(host_queue, device=Device.DEFAULT)).realize()
    np.testing.assert_allclose(read_yolo(run_yolo(value['run'], queue)), value['raw'], rtol=1e-3, atol=1e-3)
    value['challenge_raw'] = expected
    value['input_rebinding_passed'] = True
  print('new allocation and in-place input changes passed', flush=True)
  config_realtime_process(7, 54)
  for index in range(args.samples):
    for value in variants.values():
      start = time.perf_counter()
      raw_tensor = run_yolo(value['run'], queue)
      submitted = time.perf_counter()
      raw = read_yolo(raw_tensor)
      read = time.perf_counter()
      decode_detections(raw.astype(np.float32, copy=False), 512, 256, compact=True)
      end = time.perf_counter()
      value['samples'].append([(submitted-start)*1000, (read-submitted)*1000, (end-read)*1000, (end-start)*1000])
    if (index+1) % 500 == 0:
      print('round', index+1, {name: stats([r[-1] for r in v['samples']]) for name, v in variants.items()}, flush=True)
  report = {'input_kind': 'resident native driving queue', 'queue_shape': shape,
            'input_sha256': hashlib.sha256(args.input_packed.read_bytes()).hexdigest(), 'onnx_sha256': manifest['sha256'],
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'device': Device.DEFAULT,
            'driving_weights_resident': True, 'input_upload_in_timing': False, 'variants': {}}
  for name, value in variants.items():
    report['variants'][name] = {k: v for k, v in value.items() if k not in ('run', 'raw', 'challenge_raw')}
    report['variants'][name]['phases_ms'] = {phase: stats([r[i] for r in value['samples']]) for i, phase in
                                            enumerate(('submit', 'readback_wait', 'nms', 'total'))}
    np.savez_compressed(out/f'{name}_validation.npz', raw=value['raw'])
    if args.profile_replays:
      import cProfile
      import pstats
      profiler = cProfile.Profile()
      profiler.enable()
      for _ in range(args.profile_replays):
        read_yolo(run_yolo(value['run'], queue))
      profiler.disable()
      entries = pstats.Stats(profiler).stats
      report['variants'][name]['replay_profile'] = {
        'replays': args.profile_replays,
        'functions': [{'file': Path(key[0]).name, 'line': key[1], 'function': key[2],
                       'calls': data[1], 'self_ms': data[2]*1000, 'cumulative_ms': data[3]*1000}
                      for key, data in sorted(entries.items(), key=lambda item: item[1][3], reverse=True)[:40]],
        'input_copied_by_jit': queue.uop.base in value['run'].captured._written_uops,
        'captured_calls': len(value['run'].captured.linear.src),
      }
  # Admission reserves the measured worst cost; p99 alone can choose a variant
  # whose isolated stall makes it ineligible for the driving frame's idle slot.
  selected = min(variants, key=lambda name: (report['variants'][name]['phases_ms']['total']['max'],
                                           report['variants'][name]['phases_ms']['total']['p99']))
  report['selection_metric'] = 'maximum total, then p99 total'
  value = variants[selected]
  report['selected'] = selected
  bundle = {'version': ARTIFACT_VERSION, 'run': value['run'], 'queue_shape': shape, 'width': 512, 'height': 256,
            'names': manifest['names'], 'model_id': manifest['model_id'], 'onnx_sha256': manifest['sha256'],
            'device': Device.DEFAULT, 'source_fingerprint': source_fingerprint(),
            'measured_seconds': max(r[-1] for r in value['samples'])/1000,
            'native': value['native'], 'output_dtype': value['output_dtype'], 'variant': selected,
            'input_rebinding_passed': value['input_rebinding_passed']}
  target = out/'yolo_reuse.pkl'
  temporary = target.with_suffix('.pkl.tmp')
  with temporary.open('wb') as f:
    dump_oob(bundle, f)
  with temporary.open('rb') as f:
    restored = load_oob(f)
  np.testing.assert_array_equal(read_yolo(run_yolo(restored['run'], queue)), value['raw'])
  np.testing.assert_allclose(read_yolo(run_yolo(restored['run'], other_queue)), value['challenge_raw'], rtol=1e-3, atol=1e-3)
  report['serialized_rebinding_passed'] = True
  report['serialization_passed'] = True
  report['device_error'] = repr(Device[Device.DEFAULT].error_state) if Device[Device.DEFAULT].error_state is not None else None
  if report['device_error'] is not None:
    raise RuntimeError(report['device_error'])
  temporary.replace(target)
  (out/'reuse_report.json').write_text(json.dumps(report, indent=2)+'\n')
  print('selected', selected, report['variants'][selected]['phases_ms'], flush=True)


if __name__ == '__main__':
  main()
