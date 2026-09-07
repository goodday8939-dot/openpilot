from copy import deepcopy

import pytest

from openpilot.selfdrive.carrot.server.services.egpu_yolo_tracking import RadarAssociator, VisualTracker, mutual_matches


def detection(x=.5, y=.6, cls=1):
  return {'classId': cls, 'label': 'bicycle', 'confidence': .8,
          'cameraPoints': [x-.05, y-.1, x+.05, y-.1, x+.05, y, x-.05, y]}


def frame(t=10., detections=None):
  return {'timestampEof': int(t*1e9), 'frameId': int(t*20), 'camera': 'road', 'cameraWidth': 1344,
          'cameraHeight': 760, 'modelId': 'test', 'detections': [detection()] if detections is None else detections}


def test_visual_identity_survives_short_miss_without_inventing_detection():
  tracker = VisualTracker()
  first = tracker.update(frame())
  assert tracker.update(frame(10.1, []))['detections'] == []
  recovered = tracker.update(frame(10.2, [detection(.51)]))
  assert recovered['detections'][0]['trackId'] == first['detections'][0]['trackId']
  assert recovered['detections'][0]['trackAgeSeconds'] == pytest.approx(.2)
  expired = tracker.update(frame(10.7))
  assert expired['detections'][0]['trackId'] != first['detections'][0]['trackId']


def test_duplicate_frame_does_not_advance_tracker_and_restart_does_not_reuse_id():
  tracker = VisualTracker()
  first = tracker.update(frame())
  assert tracker.update(frame()) is first
  assert tracker.update(frame(9.))['detections'][0]['trackId'] != first['detections'][0]['trackId']
  altered = frame(9.1)
  altered['cameraWidth'] = 1928
  assert tracker.update(altered)['detections'][0]['trackId'] == 3


def test_class_change_and_ambiguous_overlap_do_not_claim_same_identity():
  tracker = VisualTracker()
  first = tracker.update(frame())['detections'][0]['trackId']
  changed = tracker.update(frame(10.1, [detection(cls=2)]))
  assert changed['detections'][0]['trackId'] != first
  tracker = VisualTracker()
  previous = tracker.update(frame(10., [detection(.49), detection(.51)]))
  disputed = tracker.update(frame(10.1))
  assert disputed['detections'][0]['trackId'] not in [d['trackId'] for d in previous['detections']]
  assert tracker.update(frame(10.2))['detections'][0]['trackId'] == disputed['detections'][0]['trackId']
  assert mutual_matches({(0, 0): .9, (1, 0): .85}) == {}


def test_visual_state_is_bounded_and_malformed_box_is_ignored():
  tracker = VisualTracker()
  tracker.update(frame(detections=[detection(i/100) for i in range(100)]))
  assert len(tracker.tracks) <= 40
  bad = detection()
  bad['cameraPoints'][0] = float('nan')
  assert tracker.update(frame(10.1, [bad]))['detections'] == []


def radar_point(**kwargs):
  return {'trackId': 12, 'radarSource': 'frontRadar', 'dRel': 20., 'yRel': 0., 'vRel': -2., 'yvRel': 0., 'measured': True, **kwargs}


def associator(point=None, delay=0.):
  radar = RadarAssociator(delay)
  radar.ingest('liveTracks', 10., True, {'points': [point or radar_point()], 'errors': {}})
  radar.ingest('liveCalibration', 9.9, True, {'calStatus': 'calibrated', 'rpyCalib': [0., 0., 0.], 'height': [1.3]})
  radar.ingest('roadCameraState', 9.95, True, {'sensor': 'os04c10'})
  radar.ingest('deviceState', 9., True, {'deviceType': 'mici'})
  return radar


def projected_frame(t=10., y_rel=0., distance=20.):
  return frame(t, [detection(.5-1141.5*y_rel/(distance+1.52)/1344, .5+1141.5*1.3/(distance+1.52)/760)])


def test_front_geometry_left_sign_and_unvalidated_candidate():
  radar = associator(radar_point(yRel=1.))
  original = projected_frame(y_rel=1.)
  before = deepcopy(original)
  result = radar.associate(original)
  candidate = result['detections'][0]['radar']
  assert candidate['trackId'] == 12 and candidate['validated'] is False
  assert candidate['cameraPoint'][0] < .5
  assert candidate['dRel'] == 20. and candidate['vRel'] == -2.
  assert original == before


@pytest.mark.parametrize('source', ['corner235', 'corner', 'unknown', ''])
def test_corner_or_unknown_source_never_supplies_a_visual_distance(source):
  result = associator(radar_point(radarSource=source)).associate(projected_frame())
  assert result['radarStatus']['state'] == 'no_front_points'
  assert result['detections'][0]['radar'] is None


def test_timestamp_alignment_compensates_declared_delay_and_expires():
  radar = associator(delay=.1)
  result = radar.associate(projected_frame(9.96))
  assert result['radarStatus']['ageSeconds'] == pytest.approx(.06)
  assert result['detections'][0]['radar'] is not None
  assert radar.associate(projected_frame(10.2))['radarStatus']['state'] == 'stale'
  assert RadarAssociator().associate(frame())['radarStatus']['state'] == 'timing_unavailable'


def test_invalid_errors_missing_calibration_and_unmeasured_points_do_not_match():
  radar = associator()
  radar.ingest('liveTracks', 10.01, True, {'points': [radar_point()], 'errors': {'canError': True}})
  assert radar.associate(projected_frame(10.01))['radarStatus']['state'] == 'invalid'
  radar = associator()
  radar.ingest('liveCalibration', 9.99, False, {})
  assert radar.associate(projected_frame())['radarStatus']['state'] == 'calibration_unavailable'
  assert associator(radar_point(measured=False)).associate(frame())['detections'][0]['radar'] is None
  mismatched = projected_frame()
  mismatched['cameraWidth'] = 1928
  assert associator().associate(mismatched)['radarStatus']['state'] == 'calibration_unavailable'


def test_ambiguous_pair_and_duplicate_radar_id_remain_unmatched():
  radar = associator()
  visual = projected_frame()
  visual['detections'] *= 2
  assert all(d['radar'] is None for d in radar.associate(visual)['detections'])
  radar.ingest('liveTracks', 10.01, True, {'points': [radar_point(), radar_point()], 'errors': {}})
  assert radar.associate(projected_frame(10.01))['detections'][0]['radar'] is None
