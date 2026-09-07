"""Read-only experimental YOLO view. IPC has a single event-loop owner."""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from aiohttp import web

from openpilot.selfdrive.modeld.egpu_yolo import artifact_path, camera_time
from openpilot.selfdrive.carrot.server.services.egpu_yolo_tracking import RadarAssociator, VisualTracker

STATE = web.AppKey("egpu_yolo_state", dict)


async def collector(app: web.Application):
  from openpilot.cereal import car, messaging
  from openpilot.common.params import Params
  sock = messaging.sub_sock("carrotYolo", conflate=True)
  services = ('liveTracks', 'liveCalibration', 'roadCameraState', 'deviceState')
  context = messaging.SubMaster(list(services))
  tracker, radar = VisualTracker(), RadarAssociator()
  next_params_read = 0.
  state = app[STATE]
  try:
    while True:
      now = camera_time()
      if now >= next_params_read:
        next_params_read = now+5.
        raw = Params().get('CarParams')
        if raw:
          with car.CarParams.from_bytes(raw) as cp:
            radar.radar_delay = cp.radarDelay
      context.update(0)
      for service in services:
        if context.updated[service]:
          radar.ingest(service, context.logMonoTime[service]/1e9, context.valid[service], context[service].to_dict())
      event = messaging.recv_one_or_none(sock)
      if event is not None:
        data = event.carrotYolo.to_dict()
        state["status"] = {k: v for k, v in data.items() if k != "detections"}
        state["received"] = camera_time()
        if event.valid:
          state["frame"] = radar.associate(tracker.update(data))
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
  prepared = (directory / "yolo_qcom.pkl").is_file()
  downloaded = (directory / "model.onnx").is_file() and (directory / "manifest.json").is_file()
  status = state["status"] if live else {"state": "ready" if prepared else "downloaded" if downloaded else "not_prepared"}
  if not live and (directory / 'qcom_fault').exists():
    status = {"state": "error"}
  elif not live and prepared and not (directory / 'qcom_enabled').exists():
    status = {"state": "prepared"}
  supervisor = None
  try:
    report = json.loads((directory / 'live_reuse_status.json').read_text())
    if 0 <= now-report.get('updated_camera_time', 0) < 10:
      supervisor = {key: report.get(key) for key in ('stage', 'reason', 'recovery_reason', 'retry_count', 'stable_seconds_required')}
  except (OSError, ValueError, TypeError, AttributeError):
    pass
  return {"ok": True, "status": status, "frame": frame, "ageSeconds": age, "stale": age is None or age > .35,
          "compiled": prepared, "downloaded": downloaded, "supervisor": supervisor}


async def api_status(request: web.Request):
  return web.json_response(status_payload(request.app[STATE], artifact_path().parent, camera_time()),
                           headers={"Cache-Control": "no-store"})


def register(app: web.Application):
  app.cleanup_ctx.append(lifecycle)
  app.router.add_get("/api/egpu/yolo", api_status)
