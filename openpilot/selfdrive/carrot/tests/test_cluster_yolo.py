from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
from cluster_yolo import build_yolo_display, yolo_text


def display(**kwargs):
  message = {'state': 'run', 'runs': 42, 'executionTime': .007,
             'detections': [{'label': 'car', 'confidence': .8}, {'label': 'car', 'confidence': .6}]}
  return build_yolo_display(**{'message': message, 'enabled': True, 'valid': True,
                               'transport_age': .03, 'image_age': .1, **kwargs})


def test_current_detections_group_counts_and_best_confidence():
  result = display()
  assert result.objects == (('car', 2, 80),)
  assert yolo_text(result, 'ko') == ('YOLO 실행 7.0ms · #42', '자동차 2 (80%)')
  assert yolo_text(result, 'en') == ('YOLO RUN 7.0ms · #42', 'car 2 (80%)')


@pytest.mark.parametrize('state', ['paused', 'error', 'overrun'])
def test_invalid_status_messages_clear_old_objects_but_preserve_state(state):
  assert display(message={'state': state, 'runs': 43}, valid=False).state == state
  assert display(message={'state': state, 'runs': 43}, valid=False).objects == ()


@pytest.mark.parametrize('args', [{'image_age': .36}, {'transport_age': 2.01}, {'image_age': float('nan')}, {'image_age': -.1}])
def test_stale_or_bad_image_time_never_displays_previous_detections(args):
  result = display(**args)
  assert result.state == 'stale' and not result.objects
  assert result.execution_ms is None


def test_absent_disabled_invalid_and_empty_are_distinct():
  assert display(message=None, enabled=False) is None
  assert display(message=None).state == 'waiting'
  assert display(enabled=False).state == 'off'
  assert display(valid=False).state == 'invalid'
  assert yolo_text(replace(display(), objects=()), 'en')[1] == 'No detections'


def test_live_source_uses_publication_age_and_clears_paused_content():
  from cluster_live import OpenpilotLiveSource
  source = object.__new__(OpenpilotLiveSource)
  source._egpu_active = True
  source.sm = SimpleNamespace(seen={'carrotYolo': True}, recv_time={'carrotYolo': 100.},
                              logMonoTime={'carrotYolo': 900_000_000_000}, valid={'carrotYolo': True})
  source._service_data = lambda service: SimpleNamespace(state='run', runs=12, executionTime=.006,
      timestampEof=899_900_000_000, detections=[SimpleNamespace(label='person', confidence=.9)])
  assert source._yolo_display(100.1).objects == (('person', 1, 90),)
  assert source._yolo_display(100.3).state == 'stale'


def test_all_hud_layouts_receive_yolo_summary(monkeypatch):
  import cluster_renderer
  from cluster_live import standby_state
  renderer = object.__new__(cluster_renderer.ClusterUiRenderer)
  calls = []
  renderer._draw_text = lambda text, *args, **kwargs: calls.append(text)
  renderer._ellipsize_text = lambda text, *args: text
  monkeypatch.setattr(cluster_renderer.rl, 'draw_rectangle_rounded', lambda *args: None)
  monkeypatch.setattr(cluster_renderer.rl, 'draw_rectangle_rounded_lines_ex', lambda *args: None)
  for mode in [0, 1, 2, cluster_renderer.CLUSTER_SCREEN_MODE_FULLSCREEN_3D]:
    renderer._draw_yolo_status(replace(standby_state(), yolo=display()), mode)
  assert calls.count('YOLO 실행 7.0ms · #42') == 4
