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
  config_realtime_process(7, 54)
  for index in range(args.samples):
    for value in variants.values():
      start = time.perf_counter()
      raw_tensor = run_yolo(value['run'], queue)
      submitted = time.perf_counter()
      raw = read_yolo(raw_tensor)
      read = time.perf_counter()
      decode_detections(raw, 512, 256, compact=True)
      end = time.perf_counter()
      value['samples'].append([(submitted-start)*1000, (read-submitted)*1000, (end-read)*1000, (end-start)*1000])
    if (index+1) % 500 == 0:
      print('round', index+1, {name: stats([r[-1] for r in v['samples']]) for name, v in variants.items()}, flush=True)
  report = {'input_kind': 'resident native driving queue', 'queue_shape': shape,
            'input_sha256': hashlib.sha256(args.input_packed.read_bytes()).hexdigest(), 'onnx_sha256': manifest['sha256'],
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'device': Device.DEFAULT,
            'driving_weights_resident': True, 'input_upload_in_timing': False, 'variants': {}}
  for name, value in variants.items():
    report['variants'][name] = {k: v for k, v in value.items() if k not in ('run', 'raw')}
    report['variants'][name]['phases_ms'] = {phase: stats([r[i] for r in value['samples']]) for i, phase in
                                            enumerate(('submit', 'readback_wait', 'nms', 'total'))}
    np.savez_compressed(out/f'{name}_validation.npz', raw=value['raw'])
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
            'native': value['native'], 'output_dtype': value['output_dtype'], 'variant': selected}
  target = out/'yolo_reuse.pkl'
  with target.open('wb') as f:
    dump_oob(bundle, f)
  with target.open('rb') as f:
    restored = load_oob(f)
  np.testing.assert_array_equal(read_yolo(run_yolo(restored['run'], queue)), value['raw'])
  report['serialization_passed'] = True
  report['device_error'] = repr(Device[Device.DEFAULT].error_state) if Device[Device.DEFAULT].error_state is not None else None
  (out/'reuse_report.json').write_text(json.dumps(report, indent=2)+'\n')
  print('selected', selected, report['variants'][selected]['phases_ms'], flush=True)


if __name__ == '__main__':
  main()
