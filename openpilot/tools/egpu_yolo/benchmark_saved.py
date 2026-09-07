#!/usr/bin/env python3
"""Saved full-camera NV12 benchmark with exclusive eGPU access; never publishes YOLO."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def stats(values):
  return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist(), strict=True))


def check_state(stationary_maintenance):
  from openpilot.cereal import messaging
  from openpilot.common.params import Params
  from openpilot.selfdrive.modeld.qcom_yolo_prepare import check_maintenance_state

  services = ['managerState', 'deviceState'] + (['carState'] if stationary_maintenance else [])
  sm = messaging.SubMaster(services)
  deadline = time.monotonic() + 5
  while time.monotonic() < deadline:
    sm.update(100)
    if sm.all_checks():
      break
  if not sm.all_checks():
    raise RuntimeError('fresh maintenance state required before opening eGPU')
  params = Params()
  parked = (stationary_maintenance and sm['carState'].standstill and abs(sm['carState'].vEgo) < .01
            and str(sm['carState'].gearShifter) == 'park')
  check_maintenance_state(sm['managerState'].processes,
                          onroad=params.get_bool('IsOnroad') or sm['deviceState'].started,
                          driver_preview=params.get_bool('IsDriverViewEnabled'),
                          stationary_maintenance=stationary_maintenance, parked=parked)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, required=True)
  parser.add_argument('--input-nv12', type=Path, required=True)
  parser.add_argument('--camera-width', type=int, default=1344)
  parser.add_argument('--camera-height', type=int, default=760)
  parser.add_argument('--samples', type=int, default=5000)
  parser.add_argument('--paced-samples', type=int, default=300)
  parser.add_argument('--interval', type=float, default=.2)
  parser.add_argument('--stationary-maintenance', action='store_true')
  args = parser.parse_args()
  if args.samples < 1 or args.paced_samples < 1 or args.interval < .05:
    parser.error('positive sample counts and an interval of at least 50 ms required')
  check_state(args.stationary_maintenance)
  out = args.directory
  manifest = json.loads((out / 'manifest.json').read_text())
  source = out / 'model.onnx'
  if source.stat().st_size != manifest['size'] or hashlib.sha256(source.read_bytes()).hexdigest() != manifest['sha256']:
    raise ValueError('ONNX size/checksum mismatch')
  width, height = manifest['width'], manifest['height']
  if (width, height) not in ((512, 256), (640, 384)):
    raise ValueError('benchmark supports only the 512x256 baseline and 640x384 candidate')
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  layout = (args.camera_width, args.camera_height, *get_nv12_info(args.camera_width, args.camera_height))
  frame = np.load(args.input_nv12, allow_pickle=False)
  if frame.dtype != np.uint8 or frame.shape != (layout[-1],):
    raise ValueError('NV12 shape/dtype mismatch')

  os.environ.update(DEV='USB+AMD:LLVM', GMMU='0', FLOAT16='1', JIT_BATCH_SIZE='0', TC_OPT='2')
  import openpilot.selfdrive.modeld.compile_modeld  # noqa: F401 -- device firmware and serialization
  from openpilot.common.file_chunker import open_file_chunked
  from openpilot.common.realtime import config_realtime_process
  from openpilot.selfdrive.modeld.egpu_yolo import decode_detections
  from openpilot.selfdrive.modeld.egpu_yolo_usb import run_yolo, read_yolo, custom_usb
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob, usbgpu_compiled_path
  from openpilot.selfdrive.modeld.qcom_yolo_model import nv12_to_rgb, letterbox_geometry
  from tinygrad import Tensor, TinyJit
  from tinygrad.device import Device, Compiled, ProfileGraphEvent
  from tinygrad.engine.realize import graph_cache
  from tinygrad.helpers import Context
  from tinygrad.nn.onnx import OnnxRunner

  driving_path = usbgpu_compiled_path()
  if driving_path is None:
    raise RuntimeError('compiled driving model required for resident-weight benchmark')
  print('loading_driving_weights', flush=True)
  with open_file_chunked(driving_path) as f:
    driving = load_oob(f)
  device = Device[Device.DEFAULT]
  if not custom_usb(device):
    raise RuntimeError('custom USB AMD device required')
  network = OnnxRunner(source)
  name, spec = next(iter(network.graph_inputs.items()))
  if len(network.graph_inputs) != 1 or tuple(spec.shape) != (1, 3, height, width):
    raise ValueError('fixed NCHW model shape mismatch')

  def infer(queue):
    # Reuse only the camera conversion, without QCOM compiler/Conv overrides.
    rgb = nv12_to_rgb(queue, layout, (width, height)).contiguous().realize()
    raw, = network({name: rgb.cast(spec.dtype)}).values()
    scores = raw[:, 4:]
    return raw[:, :4].cat(scores.max(axis=1, keepdim=True),
                         scores.argmax(axis=1).unsqueeze(1).cast(raw.dtype), dim=1).cast('float32').realize()

  queue = Tensor(frame, device=Device.DEFAULT).realize()
  run = TinyJit(infer, prune=True)
  cold = []
  for index in range(4):
    start = time.perf_counter()
    raw = read_yolo(run_yolo(run, queue))
    cold.append((time.perf_counter() - start) * 1000)
    print('warmup', index, cold[-1], flush=True)
  config_realtime_process(7, 54)
  report = {'model_id': manifest['model_id'], 'onnx_sha256': manifest['sha256'], 'layout': layout,
            'input_sha256': hashlib.sha256(args.input_nv12.read_bytes()).hexdigest(),
            'letterbox': letterbox_geometry(layout[:2], (width, height)),
            'driving_path': str(driving_path), 'driving_weights_resident': bool(driving),
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'scheduler': os.sched_getscheduler(0),
            'device': Device.DEFAULT, 'cold_ms': cold, 'output_bytes': raw.nbytes,
            'input_bytes': frame.nbytes, 'started_unix': time.time(),  # noqa: TID251 -- artifact wall time; durations use perf_counter
            'scope': 'saved full-camera input; no live camera acquisition or publication; driving weights resident but idle'}

  def save(stage):
    report['stage'] = stage
    report['updated_unix'] = time.time()  # noqa: TID251 -- artifact wall time; durations use perf_counter
    temporary = out / 'benchmark_report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(out / 'benchmark_report.json')
    print(stage, flush=True)

  resident = []
  for index in range(args.samples):
    start = time.perf_counter()
    raw = read_yolo(run_yolo(run, queue))
    detections = decode_detections(raw, width, height, compact=True)
    resident.append((time.perf_counter() - start) * 1000)
    if (index + 1) % 500 == 0:
      print('resident', index + 1, stats(resident), flush=True)
  report['resident_ms'] = stats(resident)
  report['resident_samples_ms'] = resident
  report['detections'] = detections
  save('resident_complete')

  split = []
  for _ in range(120):
    start = time.perf_counter()
    output = run_yolo(run, queue)
    submitted = time.perf_counter()
    device.synchronize()
    complete = time.perf_counter()
    raw = read_yolo(output)
    copied = time.perf_counter()
    decode_detections(raw, width, height, compact=True)
    end = time.perf_counter()
    split.append([(submitted-start)*1000, (complete-submitted)*1000, (copied-complete)*1000, (end-copied)*1000])
  report['serialized_phases_ms'] = {name: stats([row[i] for row in split]) for i, name in
                                   enumerate(('host_submit', 'gpu_wait', 'usb_readback', 'nms'))}
  report['serialized_samples_ms'] = split
  save('split_complete')

  paced = []
  source_view = memoryview(frame).cast('B')
  for index in range(args.paced_samples):
    start = time.perf_counter()
    queue.uop.buffer.copyin(source_view)
    uploaded = time.perf_counter()
    raw = read_yolo(run_yolo(run, queue))
    read = time.perf_counter()
    decode_detections(raw, width, height, compact=True)
    end = time.perf_counter()
    paced.append([(uploaded-start)*1000, (read-uploaded)*1000, (end-read)*1000, (end-start)*1000])
    if (index + 1) % 100 == 0:
      print('paced', index + 1, stats([row[-1] for row in paced]), flush=True)
    time.sleep(max(0, args.interval - (time.perf_counter() - start)))
  report['paced_interval_seconds'] = args.interval
  report['paced_ms'] = {name: stats([row[i] for row in paced]) for i, name in
                        enumerate(('input_upload', 'inference_readback', 'nms', 'total'))}
  report['paced_samples_ms'] = paced
  save('paced_complete')

  rgb = nv12_to_rgb(queue, layout, (width, height)).numpy()
  np.savez_compressed(out / 'benchmark_validation.npz', rgb=rgb, raw=raw)
  # Separate artifact name cannot activate either legacy eGPU or QCOM workers.
  target = out / 'benchmark.pkl'
  with target.open('wb') as f:
    dump_oob({'run': run, 'layout': layout, 'model_id': manifest['model_id'], 'onnx_sha256': manifest['sha256']}, f)
  with target.open('rb') as f:
    restored = load_oob(f)
  np.testing.assert_allclose(read_yolo(run_yolo(restored['run'], queue)), raw, rtol=1e-3, atol=1e-3)
  report['serialization_passed'] = True
  save('serialization_complete')

  # Profiling uses a rebuilt graph after the uninstrumented distributions.
  device.synchronize()
  graph_cache.clear()
  Compiled.profile_events.clear()
  with Context(PROFILE=1):
    for _ in range(6):
      read_yolo(run_yolo(run, queue))
    device.synchronize()
    events = [event for event in Compiled.profile_events if isinstance(event, ProfileGraphEvent)]
    if events:
      event = events[-1]
      kernels = [{'name': str(entry.name), 'device': entry.device,
                  'us': float(event.sigs[entry.en_id] - event.sigs[entry.st_id])} for entry in event.ents]
      report['gpu_profile'] = {'kernels': kernels, 'kernel_sum_ms': sum(k['us'] for k in kernels)/1000,
                               'span_ms': float(max(event.sigs[e.en_id] for e in event.ents) -
                                                min(event.sigs[e.st_id] for e in event.ents))/1000,
                               'profile_graph_count': len(events)}
    graph_cache.clear()
  report['device_error'] = repr(device.error_state) if device.error_state is not None else None
  save('complete')
  print(json.dumps({k: v for k, v in report.items() if not k.endswith('samples_ms') and k != 'gpu_profile'}), flush=True)


if __name__ == '__main__':
  main()
