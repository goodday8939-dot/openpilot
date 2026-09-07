"""Replay timestamped read-only service captures through the Web tracker.

Usage: python -m openpilot.tools.egpu_yolo.replay_tracking capture.json
Captures contain metadata and streams[name] entries with mono_ns/valid/data.
Reports ID churn; this is not ground-truth identity or radar accuracy scoring.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from openpilot.selfdrive.carrot.server.services.egpu_yolo_tracking import RadarAssociator, VisualTracker


def replay(capture):
  tracker = VisualTracker()
  radar = RadarAssociator(capture.get('metadata', {}).get('radarDelay'))
  events = sorted((event['mono_ns'], service, event) for service, events in capture['streams'].items()
                  if service == 'carrotYolo' or service in radar.history for event in events)
  ids, states, timings, matches, frames = Counter(), Counter(), [], 0, 0
  for timestamp, service, event in events:
    if service == 'carrotYolo':
      if not event['valid']:
        continue
      started = time.perf_counter()
      output = radar.associate(tracker.update(event['data']))
      timings.append((time.perf_counter()-started)*1000)
      frames += 1
      states[output['radarStatus']['state']] += 1
      for detection in output['detections']:
        ids[f"{detection['label']}#{detection['trackId']}"] += 1
        matches += int(detection['radar'] is not None)
    else:
      radar.ingest(service, timestamp/1e9, event['valid'], event['data'])
  timings.sort()
  return {'frames': frames, 'observations_by_id': dict(ids), 'radar_states': dict(states), 'candidate_count': matches,
          'processing_ms': {name: timings[min(len(timings)-1, int(len(timings)*fraction))] if timings else None
                            for name, fraction in [('p50', .5), ('p99', .99), ('max', 1.)]}}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('capture', type=Path)
  args = parser.parse_args()
  print(json.dumps(replay(json.loads(args.capture.read_text())), indent=2))


if __name__ == '__main__':
  main()
