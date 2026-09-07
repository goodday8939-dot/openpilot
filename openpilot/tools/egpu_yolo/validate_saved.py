#!/usr/bin/env python3
"""Independent workstation NumPy/ONNX Runtime check of saved eGPU benchmark output."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def reference_rgb(frame, layout, model_size):
  cw, ch, stride, y_height, _uv_height, size = layout
  width, height = model_size
  if frame.dtype != np.uint8 or frame.shape != (size,):
    raise ValueError('saved NV12 shape/dtype mismatch')
  scale = min(width / cw, height / ch)
  content_width, content_height = round(cw * scale), round(ch * scale)
  left, top = (width - content_width) // 2, (height - content_height) // 2
  x = ((np.arange(width, dtype=np.float32) - left + .5) * cw / content_width - .5).clip(0, cw - 1)
  y = ((np.arange(height, dtype=np.float32) - top + .5) * ch / content_height - .5).clip(0, ch - 1)

  def sample(x, y, offset, step, w, h):
    x, y = x.clip(0, w - 1), y.clip(0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = (x0 + 1).clip(0, w - 1), (y0 + 1).clip(0, h - 1)
    wx, wy = x - x0, (y - y0)[:, None]
    a = frame[offset + y0[:, None] * stride + x0 * step].astype(np.float32)
    b = frame[offset + y0[:, None] * stride + x1 * step].astype(np.float32)
    c = frame[offset + y1[:, None] * stride + x0 * step].astype(np.float32)
    d = frame[offset + y1[:, None] * stride + x1 * step].astype(np.float32)
    return ((a * (1-wx) + b * wx) * (1-wy) + (c * (1-wx) + d * wx) * wy) / 255

  luma = sample(x, y, 0, 1, cw, ch)
  u = sample(x/2, y/2, stride*y_height, 2, cw//2, ch//2) - .5
  v = sample(x/2, y/2, stride*y_height+1, 2, cw//2, ch//2) - .5
  rgb = np.stack([luma + 1.402*v, luma - .344*u - .714*v, luma + 1.772*u]).clip(0, 1)
  inside = ((np.arange(width) >= left) & (np.arange(width) < left + content_width))[None, :] & \
           ((np.arange(height) >= top) & (np.arange(height) < top + content_height))[:, None]
  return np.where(inside[None], rgb, 114/255)[None].astype(np.float32)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, required=True)
  parser.add_argument('--model', type=Path, required=True)
  parser.add_argument('--input-nv12', type=Path, required=True)
  args = parser.parse_args()
  report = json.loads((args.directory / 'benchmark_report.json').read_text())
  if hashlib.sha256(args.model.read_bytes()).hexdigest() != report['onnx_sha256']:
    raise ValueError('reference model checksum mismatch')
  if hashlib.sha256(args.input_nv12.read_bytes()).hexdigest() != report['input_sha256']:
    raise ValueError('reference input checksum mismatch')
  session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
  _, _, height, width = session.get_inputs()[0].shape
  rgb = reference_rgb(np.load(args.input_nv12, allow_pickle=False), report['layout'], (width, height))
  expected, = session.run(None, {session.get_inputs()[0].name: rgb})
  scores = expected[:, 4:]
  compact = np.concatenate([expected[:, :4], scores.max(1, keepdims=True), scores.argmax(1)[:, None]], axis=1)
  with np.load(args.directory / 'benchmark_validation.npz', allow_pickle=False) as values:
    actual_rgb, actual = values['rgb'], values['raw']
  relevant = np.maximum(compact[:, 4], actual[:, 4]) >= .35
  result = {'rgb_max_abs_error': float(np.max(np.abs(actual_rgb-rgb))),
            'box_max_abs_error_pixels': float(np.max(np.abs(actual[:, :4]-compact[:, :4]))),
            'score_max_abs_error': float(np.max(np.abs(actual[:, 4]-compact[:, 4]))),
            'relevant_anchors': int(relevant.sum()),
            'relevant_box_max_abs_error_pixels': float(np.max(np.abs(actual[:, :4]-compact[:, :4])[0, :, relevant[0]]))
              if relevant.any() else None,
            'relevant_classes_match': bool(np.array_equal(actual[:, 5][relevant], compact[:, 5][relevant])),
            'reference_bicycle_max_score': float(scores[:, 1].max()),
            'actual_winning_bicycle_max_score': float(np.max(actual[:, 4][actual[:, 5] == 1], initial=0)),
            'threshold': .35, 'scope': 'one saved frame; numerical validation, not detection accuracy evaluation'}
  # Retain failures as evidence before raising. Thresholds are fixed before comparison.
  (args.directory / 'numerical_validation.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2), flush=True)
  np.testing.assert_allclose(actual_rgb, rgb, atol=3e-5, rtol=1e-5)
  np.testing.assert_allclose(actual[:, :4], compact[:, :4], atol=.5, rtol=.005)
  np.testing.assert_allclose(actual[:, 4], compact[:, 4], atol=.01, rtol=0)
  np.testing.assert_array_equal(actual[:, 5][relevant], compact[:, 5][relevant])
  if relevant.any():
    np.testing.assert_allclose(actual[0, :4, relevant[0]], compact[0, :4, relevant[0]], atol=1, rtol=0)
  result['passed'] = True
  (args.directory / 'numerical_validation.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
  main()
