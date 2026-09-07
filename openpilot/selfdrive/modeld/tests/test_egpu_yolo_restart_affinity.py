import pytest

from openpilot.tools.egpu_yolo import restart_affinity as module


@pytest.fixture
def launcher(monkeypatch):
  state = {0: set(range(8)), 123: set(range(8))}
  calls = []
  monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: '123\n')
  monkeypatch.setattr(module.Path, 'read_text', lambda self: 'tmux: server\n')
  monkeypatch.setattr(module.os, 'SCHED_OTHER', 0, raising=False)
  monkeypatch.setattr(module.os, 'sched_getscheduler', lambda pid: 0, raising=False)
  monkeypatch.setattr(module.os, 'sched_getaffinity', lambda pid: state[pid], raising=False)
  monkeypatch.setattr(module.os, 'sched_setaffinity', lambda pid, cores: calls.append((pid, cores)), raising=False)
  return state, calls


def test_existing_server_and_new_server_launcher_are_both_constrained(launcher):
  _, calls = launcher
  result = module.prepare_restart_affinity()
  assert calls == [(0, {0, 1, 2, 3}), (123, {0, 1, 2, 3})]
  assert result['before'] == {0: list(range(8)), 123: list(range(8))}


@pytest.mark.parametrize('failure', ['identity', 'affinity', 'realtime'])
def test_refuses_unexpected_launcher_before_any_mutation(launcher, monkeypatch, failure):
  state, calls = launcher
  if failure == 'identity':
    monkeypatch.setattr(module.Path, 'read_text', lambda self: 'modeld\n')
  elif failure == 'affinity':
    state[123] = {7}
  else:
    monkeypatch.setattr(module.os, 'sched_getscheduler', lambda pid: 1 if pid == 123 else 0)
  with pytest.raises(RuntimeError):
    module.prepare_restart_affinity()
  assert calls == []
