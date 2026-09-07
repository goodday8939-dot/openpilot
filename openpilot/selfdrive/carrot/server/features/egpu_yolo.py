"""Read-only experimental YOLO view. IPC has a single event-loop owner."""
from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from aiohttp import web

from openpilot.selfdrive.modeld.egpu_yolo import artifact_path

STATE = web.AppKey("egpu_yolo_state", dict)


async def collector(app: web.Application):
  from openpilot.cereal import messaging
  sock = messaging.sub_sock("carrotYolo", conflate=True)
  state = app[STATE]
  try:
    while True:
      event = messaging.recv_one_or_none(sock)
      if event is not None:
        data = event.carrotYolo.to_dict()
        state["status"] = {k: v for k, v in data.items() if k != "detections"}
        state["received"] = time.monotonic()
        if event.valid:
          state["frame"] = data
      await asyncio.sleep(0.05)
  finally:
    del sock


async def lifecycle(app: web.Application):
  app[STATE] = {"status": {}, "frame": None, "received": 0.0}
  task = asyncio.create_task(collector(app))
  yield
  task.cancel()
  with contextlib.suppress(asyncio.CancelledError):
    await task


def status_payload(state: dict, directory: Path, now: float) -> dict:
  frame = state["frame"]
  age = max(0, now - frame["timestampEof"] / 1e9) if frame else None
  live = now - state["received"] < 2
  prepared = (directory / "yolo.pkl").is_file()
  downloaded = (directory / "model.onnx").is_file() and (directory / "manifest.json").is_file()
  status = state["status"] if live else {"state": "ready" if prepared else "downloaded" if downloaded else "not_prepared"}
  return {"ok": True, "status": status, "frame": frame, "ageSeconds": age, "stale": age is None or age > .35,
          "compiled": prepared, "downloaded": downloaded}


async def api_status(request: web.Request):
  return web.json_response(status_payload(request.app[STATE], artifact_path().parent, time.monotonic()),
                           headers={"Cache-Control": "no-store"})


def register(app: web.Application):
  app.cleanup_ctx.append(lifecycle)
  app.router.add_get("/api/egpu/yolo", api_status)
