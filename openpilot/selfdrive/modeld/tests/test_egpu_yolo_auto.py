import json
import os
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.modeld.egpu_yolo_auto import AutomaticRecovery, configured


def ready(policy, now, **kwargs):
  return policy.update(now, **{'owner': 123, 'started': True, 'healthy': True, **kwargs})


def test_each_ignition_starts_without_manual_session_or_park():
  p = AutomaticRecovery()
  assert not ready(p, 0, owner=None, started=False)
  assert not ready(p, 1)
  assert ready(p, 3)
  assert not ready(p, 4, owner=None, started=False)
  assert not ready(p, 10, owner=456)
  assert ready(p, 12, owner=456)
  assert ready(p, 20000, owner=456)  # no four-hour session limit


def test_transient_health_failure_recovers_after_continuous_healthy_data():
  p = AutomaticRecovery()
  ready(p, 0)
  assert ready(p, 2)
  assert not ready(p, 3, healthy=False, reason='camera gap')
  assert not ready(p, 4)
  assert not ready(p, 5, healthy=False, reason='stale device state')
  assert not ready(p, 6)
  assert not ready(p, 7.99)
  assert ready(p, 8)
  assert p.generation == 2


def test_repeat_faults_back_off_but_do_not_permanently_stop_supervision():
  p = AutomaticRecovery()
  ready(p, 0)
  assert ready(p, 2)
  now = 3
  for expected in [2, 5, 10, 30, 30, 30]:
    assert not ready(p, now, healthy=False, reason='temporary data gap')
    assert p.delay == expected
    assert not ready(p, now+1)
    assert ready(p, now+1+expected)
    now += expected+2
  assert ready(p, now+300)
  assert p.failures == 0


def test_gpu_execution_failure_waits_for_normal_new_owner():
  p = AutomaticRecovery()
  ready(p, 0)
  assert ready(p, 2)
  assert not ready(p, 3, fatal='GPU execution failed')
  assert not ready(p, 10000)
  assert not ready(p, 10001, owner=456)
  assert ready(p, 10003, owner=456)


@pytest.fixture
def auto_files(tmp_path, monkeypatch):
  monkeypatch.setenv('EGPU_YOLO_DIR', str(tmp_path))
  (tmp_path/'egpu2-reuse').mkdir()
  (tmp_path/'egpu2-reuse/yolo_reuse.pkl').write_bytes(b'prepared')
  (tmp_path/'auto_enabled.json').write_text('{"version":1,"enabled":true}')
  return tmp_path


def test_persistent_opt_in_requires_prepared_artifact_and_no_competing_owner(auto_files):
  assert configured()
  (auto_files/'qcom_enabled').touch()
  assert not configured()
  (auto_files/'qcom_enabled').unlink()
  (auto_files/'egpu2-reuse/yolo_reuse.pkl').unlink()
  assert not configured()


@pytest.mark.parametrize('value', ['null', '[]', '{', '{"enabled":true}', '{"version":1,"enabled":false}'])
def test_invalid_configuration_never_opts_in(auto_files, value):
  (auto_files/'auto_enabled.json').write_text(value)
  assert not configured()


def test_previous_boot_lease_is_rejected_even_when_expiry_coincidentally_matches(auto_files, monkeypatch):
  from openpilot.selfdrive.modeld import egpu_yolo_auto, egpu_yolo_reuse as reuse
  monkeypatch.setattr(egpu_yolo_auto, 'boot_id', lambda: 'new-boot')
  monkeypatch.setattr(reuse, 'camera_time', lambda: 100)
  session = {'automatic': True, 'boot_id': 'old-boot', 'expires': 101, 'prepared': True, 'enabled': True,
             'mode': 'road_observation', 'owner_pid': os.getpid()}
  path = auto_files/'reuse_session.json'
  path.write_text(json.dumps(session))
  assert reuse.read_session_details() is None
  session['boot_id'] = 'new-boot'
  path.write_text(json.dumps(session))
  assert reuse.read_session_details() is not None


def test_boot_prepares_resident_runtime_before_supervisor_has_a_lease(auto_files, monkeypatch):
  import sys
  from openpilot.selfdrive.modeld import egpu_yolo_reuse as reuse, egpu_yolo_postprocess
  bundle = {'input_rebinding_passed': True, 'output_dtype': 'float16', 'native': True, 'width': 512, 'height': 256, 'variant': 'native_fp16'}
  monkeypatch.setitem(sys.modules, 'openpilot.selfdrive.modeld.helpers', SimpleNamespace(load_oob=lambda stream: bundle))
  monkeypatch.setattr(egpu_yolo_postprocess, 'OutputWorker', lambda **kw: 'cpu-output')
  monkeypatch.setitem(sys.modules, 'openpilot.common.swaglog', SimpleNamespace(cloudlog=SimpleNamespace(info=lambda *a: None)))

  class Runtime(reuse.ReuseRuntime):
    def __init__(self, queue, bundle):
      self.budget = SimpleNamespace(estimate=.006)

  runtime = Runtime.load(SimpleNamespace(shape=(1, 2, 128, 256)))
  assert runtime.automatic and runtime.mode == 'road_observation'
  assert not reuse.read_session()  # preparation is not permission to infer


def test_output_recovery_never_blocks_primary_on_spawn_or_exited_worker():
  import threading
  from openpilot.selfdrive.modeld.egpu_yolo_postprocess import OutputWorker
  worker = OutputWorker.__new__(OutputWorker)
  worker.lock = threading.Lock()
  worker.recover = True
  worker.process = SimpleNamespace(poll=lambda: 1)
  assert not worker.ready()
  assert not worker.send('unused')
  worker.process = SimpleNamespace(poll=lambda: None)
  worker.lock.acquire()
  try:
    assert not worker.ready()
    assert not worker.send('unused')
  finally:
    worker.lock.release()


@pytest.mark.parametrize('owner,lease_enabled,latch,expected_calls', [
  (-1, True, 'overrun', 0), (os.getpid(), False, 'overrun', 0),
  (os.getpid(), True, 'error', 0), (os.getpid(), True, 'overrun', 1),
])
def test_resumption_is_owner_bound_and_preserves_gpu_error_and_worst_budget(monkeypatch, owner, lease_enabled, latch, expected_calls):
  import numpy as np
  from openpilot.selfdrive.modeld import egpu_yolo_reuse as reuse
  from openpilot.selfdrive.modeld.egpu_yolo import IdleBudget
  runtime = reuse.ReuseRuntime.__new__(reuse.ReuseRuntime)
  runtime.mode, runtime.automatic, runtime.generation = 'road_observation', True, 1
  runtime.budget = IdleBudget(.004)
  for frame in range(25):
    runtime.budget.observe(frame, 100+frame*.05, 100+frame*.05+.01)
  runtime.budget.disabled_reason, runtime.budget.overruns = latch, 2
  estimate = runtime.budget.estimate
  runtime.model_id, runtime.names, runtime.phases = 'test', [], (0., 0., 0.)
  runtime.last_publish = runtime.last_execution = 0.
  packets, calls = [], []
  runtime.output = SimpleNamespace(ready=lambda: True, send=lambda packet: packets.append(packet))
  runtime.infer = lambda: calls.append(True)
  monkeypatch.setattr(reuse, 'camera_time', lambda: 101.27)
  monkeypatch.setattr(reuse, 'read_session', lambda **kw: True)
  monkeypatch.setattr(reuse, 'read_session_details', lambda **kw: {'automatic': True, 'owner_pid': owner, 'enabled': lease_enabled, 'generation': 2})
  runtime.after_publish(None, 25, 101250000000, 101255000000, 101.26, 101.27, False,
                        'road', np.eye(3), (1344, 760), lambda: False, 101.26, 101.27, permitted=True)
  assert len(calls) == expected_calls
  assert runtime.budget.estimate >= estimate
  assert runtime.budget.overruns == 2
  if latch == 'error':
    assert runtime.budget.disabled_reason == 'error'
