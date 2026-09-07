import ast
from pathlib import Path

import pytest

from openpilot.tools.egpu_yolo.recovery import RecoveryPolicy


@pytest.mark.parametrize('reason', ['modelV2 raw gap 152.090 ms', 'roadCameraState raw gap 125.003 ms',
                                   'driving execution/drop guard exceeded'])
def test_transient_timing_recovery_is_bounded_and_backs_off(reason):
  policy = RecoveryPolicy()
  assert [policy.reserve(reason, t) for t in (0, 31, 92, 213)] == [30, 60, 120, None]
  assert policy.reserve(reason, 992) == 30


@pytest.mark.parametrize('reason', ['YOLO error: overruns=0', 'YOLO overrun: overruns=1',
                                   'driving model process changed', 'interrupted 15',
                                   'display: fresh Park/disabled/eGPU/camera guard changed',
                                   'no admitted YOLO result for 10 seconds', 'modelV2 invalid'])
def test_faults_operator_stop_and_changed_conditions_never_auto_resume(reason):
  policy = RecoveryPolicy()
  assert policy.reserve(reason, 100) is None
  assert not policy.attempts


def supervised_loop(phase, events):
  # Load the loop without importing its hardware/lease-owning command entrypoint.
  path = Path(__file__).parents[3] / 'tools/egpu_yolo/supervise_reuse.py'
  tree = ast.parse(path.read_text())
  function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'guarded_phase')
  context = {'phase': phase, 'recovery_policy': RecoveryPolicy(), 'camera_time': lambda: 100,
             'recovery_event': lambda *args: events.append(args), 'report': {}, 'wall_time': lambda: 1000,
             'save': lambda: None}
  exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), context)
  return context['guarded_phase']


def test_supervisor_observes_disabled_cooldown_before_retrying_same_phase():
  calls, events = [], []

  def phase(name, seconds, enabled):
    calls.append((name, seconds, enabled))
    if len(calls) == 1:
      raise RuntimeError('modelV2 raw gap 152.090 ms')

  supervised_loop(phase, events)('display', None, True)
  assert calls == [('display', None, True), ('cooldown', 30, False), ('display', None, True)]
  assert events[0][1:] == (30, 'display')


def test_repeated_cooldown_failures_exhaust_budget_without_running_yolo():
  calls, events = [], []

  def phase(name, seconds, enabled):
    calls.append((name, seconds, enabled))
    raise RuntimeError('roadCameraState raw gap 130.000 ms')

  with pytest.raises(RuntimeError, match='recovery refused'):
    supervised_loop(phase, events)('display', None, True)
  assert calls == [('display', None, True), ('cooldown', 30, False), ('cooldown', 60, False), ('cooldown', 120, False)]


def test_changed_control_guard_during_cooldown_ends_recovery():
  calls, events = [], []

  def phase(name, seconds, enabled):
    calls.append((name, seconds, enabled))
    raise RuntimeError('modelV2 raw gap 152.090 ms' if len(calls) == 1 else 'cooldown: fresh Park/disabled/eGPU/camera guard changed')

  with pytest.raises(RuntimeError, match='recovery refused'):
    supervised_loop(phase, events)('display', None, True)
  assert len(calls) == 2 and calls[-1][2] is False
