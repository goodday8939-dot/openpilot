import pytest

from openpilot.selfdrive.modeld.egpu_yolo_reuse import leased, read_session, stationary_permitted


@pytest.mark.parametrize('expires', [0, 100, 99, 106, float('inf'), float('nan'), '103', None])
def test_expired_unbounded_or_invalid_lease_never_enables(expires):
  assert not leased({'expires': expires, 'prepared': True, 'enabled': True}, 100)


def test_prepared_session_can_load_without_enabling_execution():
  session = {'expires': 103, 'prepared': True, 'enabled': False}
  assert leased(session, 100, enabled=False)
  assert not leased(session, 100)
  session['enabled'] = True
  assert leased(session, 100)
  session['prepared'] = False
  assert not leased(session, 100, enabled=False)


@pytest.mark.parametrize('content', ['{', 'null', '[]', '{}', '{"expires": "bad"}'])
def test_incomplete_or_corrupt_session_fails_closed(tmp_path, monkeypatch, content):
  from openpilot.selfdrive.modeld import egpu_yolo_reuse
  target = tmp_path/'session.json'
  monkeypatch.setattr(egpu_yolo_reuse, 'session_path', lambda: target)
  assert not read_session()
  target.write_text(content)
  assert not read_session()


@pytest.mark.parametrize('override', [
  {}, {'fresh': False}, {'started': False}, {'parked': False}, {'standstill': False},
  {'speed': .01}, {'speed': -.01}, {'speed': float('nan')}, {'enabled': True},
  {'lat_active': True}, {'long_active': True}, {'primary_seconds': .06},
  {'primary_seconds': float('nan')}, {'primary_seconds': 0}, {'dropped': True},
])
def test_only_fresh_disabled_parked_frame_with_timely_primary_can_run(override):
  state = {'fresh': True, 'started': True, 'parked': True, 'standstill': True, 'speed': 0., 'enabled': False,
           'lat_active': False, 'long_active': False, 'primary_seconds': .035, 'dropped': False}
  assert stationary_permitted(**{**state, **override}) is (not override)
