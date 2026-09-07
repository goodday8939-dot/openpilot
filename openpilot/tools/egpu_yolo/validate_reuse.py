#!/usr/bin/env python3
"""Compare resident-queue benchmark output with independent NumPy/ONNX Runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.egpu_yolo import decode_detections


def reference_rgb(packed):
  if packed.shape != (6, 128, 256) or packed.dtype != np.uint8:
    raise ValueError('native packed 512x256 uint8 input required')
  planes = packed.astype(np.float32)/255
  y = np.empty((256, 512), dtype=np.float32)
  y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2] = planes[:4]
  u, v = [plane.repeat(2, 0).repeat(2, 1)-.5 for plane in planes[4:]]
  return np.stack((y+1.402*v, y-.344*u-.714*v, y+1.772*u)).clip(0, 1)[None]


def main():
  import onnxruntime as ort
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, required=True)
  parser.add_argument('--input-packed', type=Path, required=True)
  parser.add_argument('--model', type=Path, required=True)
  args = parser.parse_args()
  report = json.loads((args.directory/'reuse_report.json').read_text())
  for path, key in [(args.input_packed, 'input_sha256'), (args.model, 'onnx_sha256')]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
      raise ValueError(f'{key} mismatch')
  rgb = reference_rgb(np.load(args.input_packed, allow_pickle=False))
  session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
  reference, = session.run(None, {session.get_inputs()[0].name: rgb})
  scores, classes = reference[:, 4:].max(1), reference[:, 4:].argmax(1)
  result = {'scope': 'saved packed queue input; no live GPU-buffer read', 'variants': {},
            'reference_bicycle_score': float(reference[:, 5].max())}
  for name, variant in report['variants'].items():
    with np.load(args.directory/f'{name}_validation.npz', allow_pickle=False) as values:
      raw = values['raw'].astype(np.float32)
    relevant = np.maximum(raw[:, 4], scores) >= .35
    half = variant['output_dtype'] == 'float16'
    np.testing.assert_allclose(raw[:, :4], reference[:, :4], atol=.5 if half else .02, rtol=.001 if half else 0)
    np.testing.assert_allclose(raw[:, 4], scores, atol=.001 if half else .0001, rtol=0)
    np.testing.assert_array_equal(raw[:, 5][relevant], classes[relevant])
    result['variants'][name] = {
      'box_max_abs_pixels': float(np.abs(raw[:, :4]-reference[:, :4]).max()),
      'score_max_abs': float(np.abs(raw[:, 4]-scores).max()), 'relevant_anchors': int(relevant.sum()),
      'detections': decode_detections(raw, 512, 256, compact=True),
    }
  (args.directory/'numerical_validation.json').write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
