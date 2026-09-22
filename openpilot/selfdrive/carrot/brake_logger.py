#!/usr/bin/env python3
"""
브레이크 개입 관찰 로거 (기존 제어 로직에는 관여하지 않음, 기록만 함)
- 사용자가 직접 브레이크를 밟는 순간
- 시스템이 자동으로 강하게 감속하는 순간
위 두 가지를 CSV로 남겨서 나중에 패턴 분석에 쓴다.
"""
import csv
import os
import time
from datetime import datetime

import openpilot.cereal.messaging as messaging

LOG_PATH = "/data/brake_events.csv"
AUTO_DECEL_THRESHOLD = -2.0  # m/s^2, 이보다 강하게 감속하면 "자동 강감속"으로 기록
MIN_LOG_INTERVAL = 1.0  # 같은 이벤트 연속 기록 방지 (초)


def ensure_header():
  if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
        "timestamp", "event_type", "v_ego_kph", "a_ego",
        "lead_dist_m", "lead_rel_speed", "cruise_enabled"
      ])


def main():
  ensure_header()
  sm = messaging.SubMaster(['carState', 'radarState', 'controlsState'])

  last_brake_pressed = False
  last_log_time = {"brake": 0.0, "auto_decel": 0.0}

  while True:
    sm.update(100)
    if not sm.updated['carState']:
      continue

    cs = sm['carState']
    now = time.time()

    lead_dist = -1.0
    lead_rel_speed = 0.0
    if sm.updated['radarState']:
      lead = sm['radarState'].leadOne
      if lead.status:
        lead_dist = lead.dRel
        lead_rel_speed = lead.vRel

    cruise_enabled = bool(cs.cruiseState.enabled)

    # 1) 사용자가 브레이크를 막 밟기 시작한 순간
    if cs.brakePressed and not last_brake_pressed:
      if now - last_log_time["brake"] > MIN_LOG_INTERVAL:
        with open(LOG_PATH, "a", newline="") as f:
          writer = csv.writer(f)
          writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            "user_brake",
            round(cs.vEgo * 3.6, 1),
            round(cs.aEgo, 2),
            round(lead_dist, 1) if lead_dist >= 0 else "",
            round(lead_rel_speed, 2) if lead_dist >= 0 else "",
            cruise_enabled,
          ])
        last_log_time["brake"] = now
    last_brake_pressed = cs.brakePressed

    # 2) 시스템이 자동으로 강하게 감속하는 순간 (크루즈 활성 + 브레이크 미조작)
    if cruise_enabled and not cs.brakePressed and cs.aEgo < AUTO_DECEL_THRESHOLD:
      if now - last_log_time["auto_decel"] > MIN_LOG_INTERVAL:
        with open(LOG_PATH, "a", newline="") as f:
          writer = csv.writer(f)
          writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            "auto_decel",
            round(cs.vEgo * 3.6, 1),
            round(cs.aEgo, 2),
            round(lead_dist, 1) if lead_dist >= 0 else "",
            round(lead_rel_speed, 2) if lead_dist >= 0 else "",
            cruise_enabled,
          ])
        last_log_time["auto_decel"] = now


if __name__ == "__main__":
  main()
