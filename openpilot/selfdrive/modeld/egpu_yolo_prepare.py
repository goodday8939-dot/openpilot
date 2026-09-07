#!/usr/bin/env python3
"""Explicit offline preparation: download without GPU; compile with exclusive GPU access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

from openpilot.selfdrive.modeld.egpu_yolo import ARTIFACT_VERSION, GUARD, RUNTIME_MARGIN, artifact_path, decode_detections, source_fingerprint

MANIFEST_URL = "https://upload.shind0.synology.me/models/carrot-egpu-yolo-512x256/manifest.json"
REMOTE_FILENAME = "big_driving_supercombo.onnx"  # NAS model endpoint's supported filename; this directory contains only YOLO.


def download(directory: Path):
  with urlopen(MANIFEST_URL, timeout=30) as response:
    manifest = json.loads(response.read(64 * 1024))
  expected = manifest["sha256"]
  if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
    raise ValueError("invalid SHA256")
  directory.mkdir(parents=True, exist_ok=True)
  target = directory / "model.onnx"
  if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as out:
      tmp = Path(out.name)
      try:
        with urlopen(MANIFEST_URL.rsplit('/', 1)[0] + "/" + REMOTE_FILENAME, timeout=60) as response:
          while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > manifest["size"]:
              raise ValueError("YOLO download exceeds manifest size")
            digest.update(chunk)
            out.write(chunk)
        if size != manifest["size"] or digest.hexdigest() != expected:
          raise ValueError("YOLO download checksum mismatch")
      except BaseException:
        out.close()
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(target)
  (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"Verified {target}; eGPU has not been opened", flush=True)


def compile_model(directory: Path):
  # This command must not run beside modeld: the USB device is exclusive.
  os.environ.setdefault("DEV", "USB+AMD:LLVM")
  os.environ.setdefault("GMMU", "0")
  os.environ.setdefault("FLOAT16", "1")
  os.environ.setdefault("JIT_BATCH_SIZE", "0")
  os.environ.setdefault("TC_OPT", "2")
  import openpilot.selfdrive.modeld.compile_modeld  # noqa: F401 -- firmware and serialization patches
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob, usbgpu_compiled_path
  from openpilot.selfdrive.modeld.egpu_yolo_model import make_yolo_runner
  from openpilot.selfdrive.modeld.egpu_yolo_usb import run_yolo, read_yolo
  from tinygrad import Tensor, TinyJit
  from tinygrad.device import Device
  from tinygrad.nn.onnx import OnnxRunner
  manifest = json.loads((directory / "manifest.json").read_text())
  source = directory / "model.onnx"
  if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
    raise ValueError("ONNX checksum mismatch")
  driving_path = usbgpu_compiled_path()
  if driving_path is None:
    raise RuntimeError("compile the Cinque Terre driving model first")
  # Read the driving metadata without opening another GPU or running it.
  # load_oob also restores driving buffers: this is intentional, to benchmark
  # YOLO with the actual driving weights resident, and verify memory headroom.
  from openpilot.common.file_chunker import open_file_chunked
  with open_file_chunked(driving_path) as f:
    driving = load_oob(f)
  img = driving["metadata"]["input_shapes"]["img"]
  from openpilot.selfdrive.modeld.constants import ModelConstants
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  queue_shape = (frame_skip * (img[1] // 6 - 1) + 1, 6, img[2], img[3])
  width, height = manifest["width"], manifest["height"]
  if width != 2 * height or width * height > 512 * 256 or width % 32 or height % 32:
    raise ValueError("expected fixed 2:1 image, multiples of 32, at most 512x256")
  runner = OnnxRunner(source)
  run = TinyJit(make_yolo_runner(runner, width, height), prune=True)
  queue = Tensor.full(queue_shape, 128, dtype="uint8").contiguous().realize()
  for _ in range(3):
    raw = run_yolo(run, queue)
    read_yolo(raw)
  # Match modeld's CPU affinity and scheduling for the synchronous GPU
  # submission/readback benchmark; compile itself can use ordinary scheduling.
  from openpilot.common.realtime import config_realtime_process
  config_realtime_process(7, 54)
  for _ in range(5):
    decode_detections(read_yolo(run_yolo(run, queue)), width, height, compact=True)
  elapsed = []
  for _ in range(30):
    start = time.monotonic()
    raw = run_yolo(run, queue)
    decode_detections(read_yolo(raw), width, height, compact=True)
    elapsed.append(time.monotonic() - start)
  bundle = {"version": ARTIFACT_VERSION, "run": run, "queue_shape": queue_shape,
            "width": width, "height": height, "names": manifest["names"], "model_id": manifest["model_id"],
            "measured_seconds": max(elapsed), "onnx_sha256": manifest["sha256"], "device": Device.DEFAULT,
            "source_fingerprint": source_fingerprint()}
  target = directory / "yolo.pkl"
  temporary = target.with_suffix(".pkl.tmp")
  with temporary.open("wb") as f:
    dump_oob(bundle, f)
  with temporary.open("rb") as f:
    restored = load_oob(f)
  raw2 = run_yolo(restored["run"], queue)
  import numpy as np
  np.testing.assert_allclose(raw2.numpy(), raw.numpy(), rtol=1e-3, atol=1e-3)
  temporary.replace(target)
  report = {"model_id": manifest["model_id"], "measured_seconds": elapsed, "queue_shape": queue_shape,
            "admission_seconds": max(elapsed) * RUNTIME_MARGIN + GUARD, "compiled_monotonic": time.monotonic()}
  (directory / "compile_report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=("download", "compile"))
  parser.add_argument("--directory", type=Path, default=artifact_path().parent)
  args = parser.parse_args()
  (download if args.action == "download" else compile_model)(args.directory)


if __name__ == "__main__":
  main()
