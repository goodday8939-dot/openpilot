from types import SimpleNamespace

import pytest

from openpilot.selfdrive.modeld import egpu_yolo_supervisor as supervisor


@pytest.fixture
def frequency_tracker():
  return pytest.importorskip('openpilot.cereal.messaging').FrequencyTracker


def test_fast_vehicle_messages_are_sampled_at_the_declared_rate(monkeypatch, frequency_tracker):
  now = [100.]
  monkeypatch.setattr(supervisor.time, 'monotonic', lambda: now[0])
  monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
  tracker = frequency_tracker(service_freq=100, update_freq=20, is_poll=False)
  calls = []

  def update(timeout):
    assert timeout == 0
    calls.append(now[0])
    tracker.record_recv_time(now[0])

  next_poll = 0.
  for _ in range(100):
    next_poll = supervisor.poll_inputs(SimpleNamespace(update=update), next_poll)
    now[0] += .001  # immediately-ready high-rate input cannot wake us early
  assert tracker.valid
  assert calls[-1]-calls[0] == pytest.approx(4.95)
  assert min(b-a for a, b in zip(calls, calls[1:], strict=False)) == pytest.approx(.05)


def test_slow_iteration_does_not_trigger_a_catchup_burst(monkeypatch):
  now = [100.]
  monkeypatch.setattr(supervisor.time, 'monotonic', lambda: now[0])
  monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
  sm = SimpleNamespace(update=lambda timeout: None)
  next_poll = supervisor.poll_inputs(sm, 0.)
  now[0] += .24
  next_poll = supervisor.poll_inputs(sm, next_poll)
  assert next_poll == pytest.approx(100.29)
  supervisor.poll_inputs(sm, next_poll)
  assert now[0] == pytest.approx(100.29)


def test_real_frequency_drop_is_still_rejected(monkeypatch, frequency_tracker):
  now = [100.]
  monkeypatch.setattr(supervisor.time, 'monotonic', lambda: now[0])
  monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
  tracker = frequency_tracker(service_freq=100, update_freq=20, is_poll=False)
  count = [0]

  def update(timeout):
    count[0] += 1
    if count[0] % 4 == 0:  # 5 Hz arrivals, rather than healthy 100 Hz input
      tracker.record_recv_time(now[0])

  next_poll = 0.
  for _ in range(100):
    next_poll = supervisor.poll_inputs(SimpleNamespace(update=update), next_poll)
  assert not tracker.valid
