# selfdrive/controls/lib/standstill_turn_intent.py
#
# YongPilot standstill turn intent v2.1
# - 정차 중 깜빡이로 회전 의도 latch
# - 출발 후 모델 경로가 실제 그 방향으로 꺾일 때만 저속에서 제한적으로 강화
# - v2.1: max_kph/timeout 인자 지원, path_x 단조증가 보정, 일반 종료 시 램프 해제
# 좌표계: openpilot 기준 y > 0 = 왼쪽

import numpy as np

STANDSTILL_V = 0.3       # m/s
LATCH_MIN_T = 0.5        # s
STRAIGHT_MIN_T = 1.0     # s

LOOK_NEAR_X = 2.0        # m
LOOK_FAR_X = 8.0         # m
MIN_TURN_DELTA_Y = 0.35  # m

TURN_MAX_KPH = 20.0
TURN_TIMEOUT = 5.0
TURN_DONE_YAW = 1.4      # rad (약 80도)

STRAIGHT_MAX_KPH = 15.0
STRAIGHT_TIMEOUT = 1.8

RAMP_UP_T = 0.4
RAMP_DOWN_T = 0.3

MAX_TURN_ADD_M = 0.25

IDLE, ARMED, ACTIVE = 0, 1, 2


def _clean_xy(path_x, path_y):
  if path_x is None or path_y is None:
    return None, None
  x = np.asarray(path_x, dtype=float)
  y = np.asarray(path_y, dtype=float)
  if len(x) < 2 or len(x) != len(y):
    return None, None
  if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
    return None, None
  # 저속에서 x가 노이즈로 뒤로 가는 경우 대비 (np.interp는 증가순서 가정)
  x = np.maximum.accumulate(x)
  return x, y


class StandstillTurnIntent:
  def __init__(self):
    self.reset()

  def reset(self):
    self.state = IDLE
    self.dir = 0
    self.hold_t = 0.0
    self.active_t = 0.0
    self.yaw_acc = 0.0
    self.ramp = 0.0
    self.straight = False
    self.exiting = False

  @staticmethod
  def _get_blink(left_blinker, right_blinker):
    if left_blinker and right_blinker:
      return 0, True          # 비상등
    if left_blinker:
      return 1, False
    if right_blinker:
      return -1, False
    return 0, False

  @staticmethod
  def _path_turn_delta(path_x, path_y):
    x, y = _clean_xy(path_x, path_y)
    if x is None:
      return None
    if x[-1] < LOOK_FAR_X:
      return None             # 경로가 너무 짧으면 판단 안 함
    y_near = float(np.interp(LOOK_NEAR_X, x, y))
    y_far = float(np.interp(LOOK_FAR_X, x, y))
    return y_far - y_near

  def update(self, dt, v_ego, left_blinker, right_blinker, steering_pressed,
             yaw_rate, path_x, path_y,
             max_kph=TURN_MAX_KPH, timeout=TURN_TIMEOUT):
    """반환: (dir, ramp, straight)"""
    blink, hazard = self._get_blink(left_blinker, right_blinker)
    standstill = v_ego < STANDSTILL_V

    # 운전자 조향 개입 최우선 (즉시 해제)
    if steering_pressed:
      self.reset()
      return 0, 0.0, False

    # ---------------- 정차 중 latch ----------------
    if standstill:
      self.active_t = 0.0
      self.yaw_acc = 0.0
      self.ramp = 0.0
      self.straight = False
      self.exiting = False

      if hazard:
        self.state = IDLE
        self.dir = 0
        self.hold_t = 0.0
        return 0, 0.0, False

      if blink != self.dir:
        self.dir = blink
        self.hold_t = 0.0
      self.hold_t += dt

      need_t = LATCH_MIN_T if self.dir != 0 else STRAIGHT_MIN_T
      self.state = ARMED if self.hold_t >= need_t else IDLE
      return 0, 0.0, False

    # ---------------- 출발 순간 ----------------
    if self.state == ARMED:
      self.state = ACTIVE
      self.active_t = 0.0
      self.yaw_acc = 0.0
      self.ramp = 0.0
      self.exiting = False
      self.straight = (self.dir == 0)

    if self.state != ACTIVE:
      return 0, 0.0, False

    self.active_t += dt
    self.yaw_acc += yaw_rate * dt

    # 비상등 켜지면 즉시 취소
    if hazard:
      self.reset()
      return 0, 0.0, False

    # ---------------- 직진 출발 모드 ----------------
    if self.dir == 0:
      if blink != 0 or v_ego * 3.6 > STRAIGHT_MAX_KPH or self.active_t > STRAIGHT_TIMEOUT:
        self.reset()
        return 0, 0.0, False
      return 0, 0.0, True

    # ---------------- 회전 모드 종료 (램프로 부드럽게) ----------------
    if (v_ego * 3.6 > max_kph or self.active_t > timeout
        or abs(self.yaw_acc) > TURN_DONE_YAW or blink == -self.dir):
      self.exiting = True

    if self.exiting:
      self.ramp = max(0.0, self.ramp - dt / max(RAMP_DOWN_T, dt))
      if self.ramp <= 0.0:
        self.reset()
        return 0, 0.0, False
      return self.dir, self.ramp, False

    # ---------------- 모델 경로가 실제로 꺾이는지 확인 ----------------
    turn_delta = self._path_turn_delta(path_x, path_y)
    if turn_delta is None:
      target = 0.0
    else:
      target = 1.0 if turn_delta * self.dir > MIN_TURN_DELTA_Y else 0.0

    ramp_time = RAMP_UP_T if target > self.ramp else RAMP_DOWN_T
    step = dt / max(ramp_time, dt)
    if target > self.ramp:
      self.ramp = min(1.0, self.ramp + step)
    elif target < self.ramp:
      self.ramp = max(0.0, self.ramp - step)

    return self.dir, self.ramp, False


def apply_turn_boost(path_xyz, d, ramp, boost_pct, max_add_m=MAX_TURN_ADD_M):
  """2m 지점 대비 회전방향 성분만 제한적으로 증폭"""
  if d == 0 or ramp <= 0.0 or boost_pct <= 0.0 or path_xyz is None or len(path_xyz) < 2:
    return path_xyz

  x, y = _clean_xy(path_xyz[:, 0], path_xyz[:, 1])
  if x is None:
    return path_xyz

  anchor_y = float(np.interp(LOOK_NEAR_X, x, y))
  turn_component = y - anchor_y
  directional = np.where(turn_component * d > 0.0, turn_component, 0.0)

  add = directional * (boost_pct / 100.0) * ramp
  add = np.clip(add, -max_add_m, max_add_m)

  path_xyz[:, 1] = y + add
  return path_xyz
