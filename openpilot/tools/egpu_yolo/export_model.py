#!/usr/bin/env python3
"""Export the experimental YOLOv8n model on a workstation, never in modeld.

Reproduction environment: torch 2.5.1 CPU, ultralytics 8.3.40, onnx 1.17.0,
onnxruntime 1.20.1. Output big_driving_supercombo.onnx and manifest.json belong
in the separate YOLO directory on the NAS (the filename is an endpoint constraint).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--weights", type=Path, required=True, help="official yolov8n.pt")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  width, height = 320, 160
  model = YOLO(args.weights)
  exported = Path(model.export(format="onnx", imgsz=(height, width), batch=1, dynamic=False, simplify=False,
                              opset=13, half=False, nms=False, device="cpu"))
  onnx.checker.check_model(str(exported))
  session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
  errors = []
  for seed in range(3):
    rgb = np.random.default_rng(seed).random((1, 3, height, width), dtype=np.float32)
    with torch.no_grad():
      expected = model.model.eval()(torch.from_numpy(rgb))[0].numpy()
    actual, = session.run(None, {session.get_inputs()[0].name: rgb})
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    errors.append(float(np.max(np.abs(actual - expected))))
  args.output.mkdir(parents=True, exist_ok=True)
  destination = args.output / "big_driving_supercombo.onnx"
  shutil.copyfile(exported, destination)
  manifest = {"model_id": "yolov8n-coco-320x160-v1", "filename": destination.name, "size": destination.stat().st_size,
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "width": width, "height": height,
              "url": "https://upload.shind0.synology.me/models/carrot-egpu-yolo/" + destination.name,
              "names": [model.names[i] for i in range(len(model.names))],
              "source_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "source_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
              "torch_onnx_max_abs_errors": errors}
  (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps({k: v for k, v in manifest.items() if k != "names"}, indent=2))


if __name__ == "__main__":
  main()
