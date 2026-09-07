"""Bounded, display-only visual IDs and provisional raw-front-radar association.

No output from this module is published to vehicle control. A candidate is a
geometric hypothesis, not a validated object identity or a vision depth estimate.
"""
from collections import Counter, deque
import math
import uuid


def camera_box(detection):
  points = detection.get('cameraPoints', [])
  if len(points) != 8 or not all(math.isfinite(x) for x in points):
    return None
  box = (min(points[::2]), min(points[1::2]), max(points[::2]), max(points[1::2]))
  return box if box[2] > box[0] and box[3] > box[1] else None


def iou(a, b):
  intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
  return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def mutual_matches(scores, margin=.1):
  """Require a unique best choice in both directions; never force a pairing."""
  rows, columns = {}, {}
  for (row, column), score in scores.items():
    rows.setdefault(row, []).append((score, column))
    columns.setdefault(column, []).append((score, row))

  def best(groups):
    result = {}
    for key, values in groups.items():
      values.sort(reverse=True)
      if len(values) == 1 or values[0][0]-values[1][0] > margin:
        result[key] = values[0][1]
    return result

  rb, cb = best(rows), best(columns)
  return {r: c for r, c in rb.items() if cb.get(c) == r}


class VisualTracker:
  def __init__(self):
    self.session = uuid.uuid4().hex[:12]
    self.next_id = 1
    self.tracks = {}
    self.last_frame = None
    self.geometry = None

  def update(self, frame):
    timestamp = frame['timestampEof']/1e9
    geometry = tuple(frame.get(k) for k in ('camera', 'cameraWidth', 'cameraHeight', 'modelId'))
    if self.last_frame and geometry == self.geometry and timestamp == self.last_frame['timestampEof']/1e9:
      return self.last_frame
    if geometry != self.geometry or (self.last_frame and timestamp < self.last_frame['timestampEof']/1e9):
      self.tracks.clear()
    self.geometry = geometry
    self.tracks = {key: t for key, t in self.tracks.items() if 0 <= timestamp-t['time'] <= .4}
    observations = [(d, camera_box(d)) for d in frame.get('detections', [])[:40]]
    observations = [(d, box) for d, box in observations if box is not None]
    scores = {}
    for index, (detection, box) in enumerate(observations):
      for key, track in self.tracks.items():
        overlap = iou(box, track['box'])
        if detection['classId'] == track['classId'] and overlap >= .2:
          scores[index, key] = overlap
    matches = mutual_matches(scores)
    # Retire disputed IDs instead of letting stale duplicates steal later objects.
    disputed = {key for _, key in scores} - set(matches.values())
    for key in disputed:
      del self.tracks[key]
    detections = []
    for index, (detection, box) in enumerate(observations):
      key = matches.get(index)
      if key is None:
        key, self.next_id = self.next_id, self.next_id+1
      previous = self.tracks.get(key)
      first = previous['first'] if previous else timestamp
      self.tracks[key] = {'box': box, 'classId': detection['classId'], 'time': timestamp, 'first': first}
      detections.append({**detection, 'trackId': key, 'trackAgeSeconds': timestamp-first, 'radar': None})
    # Keep at most 40 observations/missed tracks, newest observations first.
    self.tracks = dict(sorted(self.tracks.items(), key=lambda item: item[1]['time'], reverse=True)[:40])
    self.last_frame = {**frame, 'detections': detections, 'trackingSession': self.session,
                       'tracking': {'active': len(self.tracks), 'created': self.next_id-1}}
    return self.last_frame


class RadarAssociator:
  def __init__(self, radar_delay=None):
    self.radar_delay = radar_delay
    self.history = {name: deque(maxlen=48) for name in ('liveTracks', 'liveCalibration', 'roadCameraState', 'deviceState')}

  def ingest(self, service, timestamp, valid, data):
    history = self.history[service]
    if history and timestamp < history[-1][0]:
      history.clear()
    history.append((timestamp, valid, data))

  def context(self, service, timestamp, max_age):
    # Never use a calibration from after the image it is applied to.
    sample = next((item for item in reversed(self.history[service]) if 0 <= timestamp-item[0] <= max_age), None)
    return sample[2] if sample and sample[1] else None

  def projection(self, frame, timestamp):
    calibration = self.context('liveCalibration', timestamp, 2.)
    camera = self.context('roadCameraState', timestamp, 1.)
    device = self.context('deviceState', timestamp, 3.)
    if not calibration or not camera or not device or calibration.get('calStatus') != 'calibrated':
      return None
    rpy, heights = calibration.get('rpyCalib', []), calibration.get('height', [])
    if len(rpy) != 3 or not heights or not all(math.isfinite(x) for x in [*rpy, heights[0]]) or not .5 < heights[0] < 3.:
      return None
    from openpilot.common.transformations.camera import DEVICE_CAMERAS, get_view_frame_from_calib_frame
    config = DEVICE_CAMERAS.get((device.get('deviceType'), camera.get('sensor')))
    if config is None or config.fcam.size != (frame.get('cameraWidth'), frame.get('cameraHeight')) or frame.get('camera') != 'road':
      return None
    return config.fcam.intrinsics @ get_view_frame_from_calib_frame(*rpy, heights[0])

  def associate(self, frame):
    timestamp = frame['timestampEof']/1e9
    status = {'state': 'unavailable', 'sources': {}, 'frontPoints': 0, 'candidates': 0,
              'validated': False, 'ageSeconds': None}
    # Clear old candidates, including when this frame is reprocessed by a caller.
    result = {**frame, 'detections': [{**d, 'radar': None} for d in frame['detections']], 'radarStatus': status}
    if self.radar_delay is None or not math.isfinite(self.radar_delay) or not 0 <= self.radar_delay <= .5:
      status['state'] = 'timing_unavailable'
      return result
    history = self.history['liveTracks']
    if not history:
      return result
    # Find the closest measurement, compensating for the car's declared delay.
    t, valid, radar = min(history, key=lambda item: abs(timestamp-(item[0]-self.radar_delay)))
    age = timestamp-(t-self.radar_delay)
    status['ageSeconds'] = age
    if abs(age) > .12:
      status['state'] = 'stale'
      return result
    errors = radar.get('errors', {})
    if not valid or (any(errors.values()) if isinstance(errors, dict) else bool(errors)):
      status['state'] = 'invalid'
      return result
    points = radar.get('points', [])[:256]
    status['sources'] = dict(Counter(p.get('radarSource') or 'unknown' for p in points))
    front = [p for p in points if p.get('radarSource') == 'frontRadar' and p.get('measured')
             and all(math.isfinite(p.get(k, math.nan)) for k in ('dRel', 'yRel', 'vRel')) and p['dRel'] > 0]
    status['frontPoints'] = len(front)
    if not front:
      status['state'] = 'no_front_points'
      return result
    projection = self.projection(frame, timestamp)
    if projection is None:
      status['state'] = 'calibration_unavailable'
      return result
    status['state'] = 'candidate_only'
    projected = []
    ids = Counter(p.get('trackId') for p in front)
    for point in front:
      if ids[point.get('trackId')] != 1:
        continue
      lateral_velocity = point.get('yvRel', 0.)
      if not math.isfinite(lateral_velocity):
        continue
      # Same provisional longitudinal offset as existing radar fusion. Radar y
      # is left-positive; calibrated device y is right-positive. Ground footpoint.
      distance = point['dRel']+point['vRel']*age
      if distance <= 0:
        continue
      pixel = projection @ [distance+1.52, -(point['yRel']+lateral_velocity*age), 0., 1.]
      if pixel[2] > 0:
        projected.append((point, pixel[0]/pixel[2]/frame['cameraWidth'], pixel[1]/pixel[2]/frame['cameraHeight']))
    scores = {}
    for index, detection in enumerate(result['detections']):
      box = camera_box(detection)
      if box is None or detection['classId'] not in (0, 1, 2, 3, 5, 7):
        continue
      for radar_index, (_, x, y) in enumerate(projected):
        dx = abs(x-(box[0]+box[2])/2)/max((box[2]-box[0])*.6, .012)
        dy = abs(y-box[3])/max((box[3]-box[1])*.3, .025)
        if dx <= 1 and dy <= 1:
          scores[index, radar_index] = 1-(dx+dy)/2
    for index, radar_index in mutual_matches(scores, .2).items():
      point, x, y = projected[radar_index]
      result['detections'][index]['radar'] = {
        'state': 'candidate', 'source': 'frontRadar', 'trackId': point['trackId'],
        'dRel': point['dRel'], 'vRel': point['vRel'], 'ageSeconds': age,
        'geometryScore': scores[index, radar_index], 'cameraPoint': [float(x), float(y)], 'validated': False,
      }
      status['candidates'] += 1
    return result
