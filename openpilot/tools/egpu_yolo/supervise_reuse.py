"""Explicit stationary A/B/display session with bounded timing-fault recovery.

Uses an already verified compiled artifact and the device-local frozen normal
restart helper. Never creates another GPU owner. --attach-model-pid avoids a
restart only when that exact owner has already loaded the reuse artifact.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import numpy as np
from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.egpu_yolo import camera_time
from openpilot.tools.egpu_yolo.recovery import RecoveryPolicy
from openpilot.selfdrive.modeld.egpu_yolo_reuse import observation_state_permitted


def wall_time():
  # Wall time is only for correlating saved evidence; every deadline is boot/monotonic time.
  return time.time()  # noqa: TID251


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stationary-maintenance', action='store_true')
parser.add_argument('--dongle-id', required=True)
parser.add_argument('--attach-model-pid', type=int)
parser.add_argument('--max-session-seconds', type=int, default=3600)
parser.add_argument('--road-observation', action='store_true', help='allow motion after stationary preparation; no control output')
args = parser.parse_args()
assert args.stationary_maintenance and 240 <= args.max_session_seconds <= (14400 if args.road_observation else 3600)
assert not args.road_observation or args.attach_model_pid is None, 'road mode requires preparation of a new owner while parked'
mode = 'road_observation' if args.road_observation else 'stationary'
session_deadline = camera_time()+args.max_session_seconds
recovery_policy = RecoveryPolicy()

out, root = Path('/data/egpu_yolo'), Path('/data/openpilot')
assert Path('/data/params/d/DongleId').read_text().strip() == args.dongle_id
assert subprocess.check_output(['git', '-C', str(root), 'branch', '--show-current'], text=True).strip() == 'carrot-egpu-yolo2'
lock = (out/'offroad.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (out/'qcom_enabled').exists()
lease_file = out/'reuse_session.json'
lease_file.unlink(missing_ok=True)
params = Params()
required = ['carState', 'selfdriveState', 'carControl', 'deviceState', 'managerState', 'modelV2',
            'roadCameraState', 'wideRoadCameraState', 'driverCameraState']
sm = messaging.SubMaster(required+['onroadEvents', 'carrotYolo'])
streams = ['modelV2', 'roadCameraState', 'wideRoadCameraState', 'driverCameraState', 'carrotYolo']
sockets = {s: messaging.sub_sock(s, conflate=False) for s in streams}
report = {'stage': 'preflight', 'phases': {}, 'coordinator_pid': os.getpid(), 'updated_unix': wall_time(), 'mode': mode}
restarting = False
restart = None
base_env = None
last_lease = 0.
model_pid = None


def save():
  report['updated_unix'] = wall_time()
  report['updated_camera_time'] = camera_time()
  temp = out/'live_reuse_status.tmp'
  temp.write_text(json.dumps(report, indent=2))
  temp.replace(out/'live_reuse_status.json')


def parked():
  cs = sm['carState']
  return (sm.all_checks(['carState', 'deviceState']) and sm['deviceState'].started
          and cs.standstill and abs(cs.vEgo) < .01 and str(cs.gearShifter) == 'park')


def permitted(*, maintenance=False):
  events = {str(e.name) for e in sm['onroadEvents']}
  if args.road_observation and not maintenance:
    state_allowed = observation_state_permitted(fresh=sm.all_checks(required), started=sm['deviceState'].started,
                                               gear=str(sm['carState'].gearShifter), speed=sm['carState'].vEgo)
  else:
    state_allowed = (parked() and not sm['selfdriveState'].enabled
                     and not sm['carControl'].latActive and not sm['carControl'].longActive)
  return (state_allowed and sm.all_checks(required)
          and params.get_bool('UsbGpuActive') and not params.get_bool('UsbGpuLoading')
          and not (out/'qcom_enabled').exists()
          and not events.intersection({'cameraMalfunction', 'cameraFrameRate', 'processNotRunning'})
          and not any(p.running and p.name in ('qcom_yolod', 'dmonitoringmodeld', 'dmonitoringd')
                      for p in sm['managerState'].processes))


def lease(enabled):
  global last_lease
  now = camera_time()
  if now-last_lease < .25:
    return
  temp = lease_file.with_suffix('.tmp')
  temp.write_text(json.dumps({'expires': now+3, 'prepared': True, 'enabled': enabled, 'pid': os.getpid(), 'mode': mode}))
  temp.replace(lease_file)
  last_lease = now


def alive(pid):
  try:
    return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] != 'Z'
  except FileNotFoundError:
    return False


def stop_manager(pid):
  assert 'manager.py' in Path(f'/proc/{pid}/cmdline').read_text()
  os.kill(pid, signal.SIGTERM)
  deadline = time.monotonic()+45
  while alive(pid) and time.monotonic() < deadline:
    time.sleep(.2)
  if alive(pid):
    raise RuntimeError('manager cleanup incomplete; refusing duplicate manager')


def managers():
  result = []
  for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
      continue
    try:
      args = (proc/'cmdline').read_text().split('\0')
      if any(a == 'manager.py' or a.endswith('/manager.py') for a in args) and alive(int(proc.name)):
        result.append(int(proc.name))
    except (OSError, ProcessLookupError):
      continue
  return result


def restore():
  global restart
  with (out/'live_reuse_restart.log').open('a') as log:
    restart = subprocess.Popen(['/bin/bash', str(out/'restart_egpu2_recovery.sh')], cwd=root, env=base_env,
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def stat(values):
  if not values:
    return {}
  return dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(values, [50, 95, 99, 100])), strict=True))


def temperature():
  state = sm['deviceState'].to_dict()
  return {k: state.get(k) for k in ('thermalStatus', 'cpuTempC', 'gpuTempC', 'memoryTempC', 'ambientTempC')}


def phase(name, seconds, enabled):
  report['stage'] = name
  started = camera_time()
  rows = {s: [] for s in streams}
  previous = {}
  last_save = 0.
  last_result = started
  path = out/f'live_reuse_{name}_samples.json'
  summary = report['phases'][name] = {'started_unix': wall_time(), 'temperature_start': temperature()}
  for sock in sockets.values():
    while messaging.recv_one_or_none(sock) is not None:
      pass
  try:
    while seconds is None or camera_time()-started < seconds:
      sm.update(50)
      now = camera_time()
      if now >= session_deadline:
        raise RuntimeError('authorized session time expired')
      if not permitted():
        guard = 'observation/eGPU/camera' if args.road_observation else 'Park/disabled/eGPU/camera'
        raise RuntimeError(f'{name}: fresh {guard} guard changed')
      current_pid = next((p.pid for p in sm['managerState'].processes if p.name == 'modeld' and p.running), None)
      if current_pid != model_pid:
        raise RuntimeError('driving model process changed')
      lease(enabled)
      for service, sock in sockets.items():
        while (event := messaging.recv_one_or_none(sock)) is not None:
          if event.logMonoTime < started*1e9:
            continue
          value = getattr(event, service)
          if service == 'carrotYolo':
            result = value.to_dict()
            rows[service].append({'published_ns': event.logMonoTime, **result})
            report['latest'] = result
            if result['state'] == 'run':
              last_result = now
            if result['state'] in ('error', 'overrun') or result['overruns']:
              raise RuntimeError(f'YOLO {result["state"]}: overruns={result["overruns"]}')
            continue
          stamp = event.logMonoTime if service == 'modelV2' else value.timestampSof
          row = {'published_ns': event.logMonoTime, 'stamp_ns': stamp, 'frame': value.frameId, 'valid': event.valid}
          if service == 'modelV2':
            row.update(execution_ms=value.modelExecutionTime*1000, drop_percent=value.frameDropPerc,
                       camera_to_publish_ms=(event.logMonoTime-value.timestampEof)/1e6)
          rows[service].append(row)
          if not event.valid:
            raise RuntimeError(f'{service} invalid')
          if service in previous and stamp-previous[service] > 120_000_000:
            raise RuntimeError(f'{service} raw gap {(stamp-previous[service])/1e6:.3f} ms')
          previous[service] = stamp
          if service == 'modelV2' and (value.modelExecutionTime > .06 or value.frameDropPerc > 1):
            raise RuntimeError('driving execution/drop guard exceeded')
      if enabled and now-last_result > 10:
        raise RuntimeError('no admitted YOLO result for 10 seconds')
      if now-last_save > 5:
        summary.update(elapsed_seconds=now-started, counts={s: len(r) for s, r in rows.items()})
        save()
        last_save = now
      # Keep the indefinite display's in-memory evidence bounded. Completed
      # 30/180/30-second trial samples remain complete and separate.
      if seconds is None and now-started > 60:
        break
  finally:
    lease_file.unlink(missing_ok=True)
    primary = rows['modelV2']
    results = [r for r in rows['carrotYolo'] if r['state'] == 'run']
    summary.update(elapsed_seconds=camera_time()-started, counts={s: len(r) for s, r in rows.items()},
                   temperature_end=temperature(), model_ms=stat([r['execution_ms'] for r in primary]),
                   camera_to_publish_ms=stat([r['camera_to_publish_ms'] for r in primary]),
                   max_drop_percent=max((r['drop_percent'] for r in primary), default=None),
                   streams={s: {'gap_ms': stat([(b['stamp_ns']-a['stamp_ns'])/1e6 for a, b in zip(rows[s], rows[s][1:], strict=False)]),
                                'frame_delta_max': max((b['frame']-a['frame'] for a, b in zip(rows[s], rows[s][1:], strict=False)), default=None)}
                            for s in streams if s != 'carrotYolo'},
                   yolo={key: stat([r[key]*1000 for r in results]) for key in
                         ('executionTime', 'submitTime', 'readbackTime', 'postprocessTime', 'drivingLatency')},
                   yolo_publication_age_ms=stat([(r['published_ns']-r['timestampEof'])/1e6 for r in results]),
                   results=len(results))
    path.write_text(json.dumps(rows))
    save()


def recovery_event(reason, stable_seconds, phase_name):
  history = report.setdefault('recovery_history', [])
  count = report.get('recoveries_total', 0)
  slot = count % 3
  report['recoveries_total'] = count+1
  source = out/f'live_reuse_{phase_name}_samples.json'
  evidence = out/f'live_reuse_fault_{slot}_samples.json'
  if source.exists():
    shutil.copyfile(source, evidence)
  event = {'unix': wall_time(), 'reason': reason, 'cause': 'timing_anomaly_cause_unresolved',
           'stable_seconds_required': stable_seconds, 'phase': phase_name,
           'evidence': str(evidence), 'summary': report['phases'].get(phase_name, {})}
  history.append(event)
  del history[:-12]
  report.update(stage='cooldown', recovery_reason=reason, retry_count=len(recovery_policy.attempts), stable_seconds_required=stable_seconds)
  save()
  print(json.dumps({'recovery': {k: v for k, v in event.items() if k != 'summary'}}), flush=True)


def guarded_phase(name, seconds, enabled):
  while True:
    try:
      phase(name, seconds, enabled)
      return
    except RuntimeError as exc:
      reason, failed_phase = str(exc), name
    # The phase's finally block has already removed the enabled lease. No
    # manager restart, compiler, retry inference or GPU work occurs in cooldown.
    while True:
      stable_seconds = recovery_policy.reserve(reason, camera_time())
      if stable_seconds is None:
        raise RuntimeError(f'recovery refused: {reason}')
      recovery_event(reason, stable_seconds, failed_phase)
      try:
        phase('cooldown', stable_seconds, False)
        break
      except RuntimeError as exc:
        reason, failed_phase = str(exc), 'cooldown'
    report['last_resume_unix'] = wall_time()
    save()


def interrupted(signum, frame):
  raise RuntimeError(f'interrupted {signum}')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
  deadline = time.monotonic()+20
  while time.monotonic() < deadline:
    sm.update(100)
    if permitted(maintenance=True):
      break
  assert permitted(maintenance=True), 'fresh Park/disabled/eGPU state required for preparation'
  original = {p.name: p.pid for p in sm['managerState'].processes if p.running}
  old_model = original['modeld']
  if args.attach_model_pid is not None:
    assert old_model == args.attach_model_pid, 'prepared owner PID changed'
  else:
    manager_pid = int(next(line for line in Path(f'/proc/{old_model}/status').read_text().splitlines()
                           if line.startswith('PPid:')).split()[1])
    base_env = {**os.environ, **dict(item.split('=', 1) for item in Path(f'/proc/{manager_pid}/environ').read_text().split('\0') if '=' in item)}
    report.update(stage='restarting_to_load_compiled_reuse', old_manager_pid=manager_pid)
    save()
    restarting = True
    stop_manager(manager_pid)
    deadline = time.monotonic()+15
    while managers() and time.monotonic() < deadline:
      time.sleep(.2)
    assert not managers(), 'old manager wrapper remains'
    assert not alive(old_model), 'old USB GPU owner remains'
    restore()
    # The old process epoch includes the deliberate manager-stop gap. Do not
    # let its frequency history suppress preparation of the new owner.
    sm = messaging.SubMaster(required+['onroadEvents', 'carrotYolo'])
  deadline = time.monotonic()+180
  report['stage'] = 'waiting_prepared_owner'
  save()
  last_preparation_report = 0.
  while time.monotonic() < deadline:
    sm.update(100)
    if parked():
      # Replay initialization only. Execution remains disabled until every
      # primary/control stream is fresh and the owner reports its paused state.
      lease(False)
    else:
      lease_file.unlink(missing_ok=True)
    if restart is not None and restart.poll() is not None and restart.returncode:
      raise RuntimeError('normal restart launcher failed')
    if camera_time()-last_preparation_report >= 5:
      report['preparation'] = {'parked': parked(), 'alive': sm.alive.copy(), 'valid': sm.valid.copy(),
                               'freq_ok': sm.freq_ok.copy(), 'seen': sm.seen.copy(),
                               'usb_active': params.get_bool('UsbGpuActive'), 'yolo_state': str(sm['carrotYolo'].state)}
      save()
      last_preparation_report = camera_time()
    if permitted(maintenance=True) and sm.updated['carrotYolo'] and sm['carrotYolo'].state == 'paused':
      break
  assert permitted(maintenance=True) and sm['carrotYolo'].state == 'paused', 'prepared driving owner not observed'
  restarting = False
  model_pid = next(p.pid for p in sm['managerState'].processes if p.name == 'modeld' and p.running)
  report.update(model_pid=model_pid, head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
  guarded_phase('baseline', 30, False)
  guarded_phase('enabled', 180, True)
  guarded_phase('recovery', 30, False)
  report['trial_passed'] = True
  save()
  while True:
    guarded_phase('display', None, True)
except BaseException as exc:
  report.update(stage='stopped', reason=str(exc), stopped_unix=wall_time())
  print(json.dumps({'stage': report['stage'], 'reason': report['reason']}), flush=True)
finally:
  lease_file.unlink(missing_ok=True)
  if restarting and base_env is not None and not managers() and (restart is None or restart.poll() is not None):
    restore()
  save()
