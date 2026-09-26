import math
import os
import time
import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# from openpilot.common.logger import sLogger
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.standstill_turn_intent import StandstillTurnIntent, apply_turn_boost

TRAJECTORY_SIZE = 33
# positive numbers go right
CAMERA_OFFSET = 0 #0.08
MIN_LANE_DISTANCE = 2.6
MAX_LANE_DISTANCE = 3.7
MAX_LANE_CENTERING_AWAY = 1.85
KEEP_MIN_DISTANCE_FROM_LANE = 1.35
KEEP_MIN_DISTANCE_FROM_EDGELANE = 1.15

def clamp(num, min_value, max_value):
  # weird broken case, do something reasonable
  if min_value > num > max_value:
    return (min_value + max_value) * 0.5
  # ok, basic min/max below
  if num < min_value:
    return min_value
  if num > max_value:
    return max_value
  return num

def sigmoid(x, scale=1, offset=0):
  return (1 / (1 + math.exp(x*scale))) + offset

def lerp(start, end, t):
  t = clamp(t, 0.0, 1.0)
  return (start * (1.0 - t)) + (end * t)

def max_abs(a, b):
  return a if abs(a) > abs(b) else b

class LanePlanner:
  def __init__(self):
    self.ll_t = np.zeros((TRAJECTORY_SIZE,))
    self.ll_x = np.zeros((TRAJECTORY_SIZE,))
    self.lll_y = np.zeros((TRAJECTORY_SIZE,))
    self.rll_y = np.zeros((TRAJECTORY_SIZE,))
    self.le_y = np.zeros((TRAJECTORY_SIZE,))
    self.re_y = np.zeros((TRAJECTORY_SIZE,))
    self.edge_valid = False
    #self.lane_width_estimate = FirstOrderFilter(3.2, 9.95, DT_MDL)
    self.lane_width_estimate = FirstOrderFilter(3.2, 3.0, DT_MDL)
    self.lane_width = 3.2
    self.lane_width_last = self.lane_width
    self.lane_change_multiplier = 1
    #self.lane_width_updated_count = 0

    self.lll_prob = 0.
    self.rll_prob = 0.
    self.d_prob = 0.

    self.lll_std = 0.
    self.rll_std = 0.

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.

    self.debugText = ""
    self.lane_width_left = 0.0
    self.lane_width_right = 0.0
    self.lane_width_left_filtered = FirstOrderFilter(1.0, 1.0, DT_MDL)
    self.lane_width_right_filtered = FirstOrderFilter(1.0, 1.0, DT_MDL)
    self.lane_offset_filtered = FirstOrderFilter(0.0, 2.0, DT_MDL)
    self.avoid_offset_filtered_x = 0.0  # 정차물체 회피 오프셋 (빠르게 진입, 천천히 복귀)
    self._avoid_log_frame = 0  # CARROT_AVOID 로그 스로틀용 프레임 카운터
    self.avoid_left_rear_near_seen = False
    self.avoid_right_rear_near_seen = False
    self.avoid_bsd_hold_time = 0.0

    self.lanefull_mode = False
    self.d_prob_count = 0

    self.params = Params()
    self.turn_intent = StandstillTurnIntent()  # YongPilot 정차 출발 회전 의도
    self._ti_log_frame = 0

  def parse_model(self, md):

    lane_lines = md.laneLines
    edges = md.roadEdges

    if len(lane_lines) >= 4 and len(lane_lines[0].t) == TRAJECTORY_SIZE:
      self.ll_t = (np.array(lane_lines[1].t) + np.array(lane_lines[2].t))/2
      # left and right ll x is the same
      self.ll_x = lane_lines[1].x
      self.lll_y = np.array(lane_lines[1].y)
      self.rll_y = np.array(lane_lines[2].y)
      self.lll_prob = md.laneLineProbs[1]
      self.rll_prob = md.laneLineProbs[2]
      self.lll_std = md.laneLineStds[1]
      self.rll_std = md.laneLineStds[2]

    if len(edges[0].t) == TRAJECTORY_SIZE:
      self.le_y = np.array(edges[0].y) + md.roadEdgeStds[0] * 0.4
      self.re_y = np.array(edges[1].y) - md.roadEdgeStds[1] * 0.4
      self.edge_valid = True
    else:
      self.le_y = self.lll_y
      self.re_y = self.rll_y
      self.edge_valid = False

    desire_state = md.meta.desireState
    if len(desire_state):
      self.l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      self.r_lane_change_prob = desire_state[log.Desire.laneChangeRight]

  def get_d_path(self, CS, v_ego, path_t, path_xyz, curve_speed):
    #if v_ego > 0.1:
    #  self.lane_width_updated_count = max(0, self.lane_width_updated_count - 1)
    # Reduce reliance on lanelines that are too far apart or
    # will be in a few seconds
    l_prob, r_prob = self.lll_prob, self.rll_prob
    width_pts = self.rll_y - self.lll_y
    prob_mods = []
    for t_check in (0.0, 1.5, 3.0):
      width_at_t = np.interp(t_check * (v_ego + 7), self.ll_x, width_pts)
      #prob_mods.append(np.interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
      prob_mods.append(np.interp(width_at_t, [4.5, 6.0], [1.0, 0.0]))
    mod = min(prob_mods)
    l_prob *= mod
    r_prob *= mod

    # Reduce reliance on uncertain lanelines
    l_std_mod = np.interp(self.lll_std, [.15, .3], [1.0, 0.0])
    r_std_mod = np.interp(self.rll_std, [.15, .3], [1.0, 0.0])
    l_prob *= l_std_mod
    r_prob *= r_std_mod

    self.l_prob, self.r_prob = l_prob, r_prob

    # Find current lanewidth
    current_lane_width = abs(self.rll_y[0] - self.lll_y[0])

    max_updated_count = 10.0 * DT_MDL
    both_lane_available = False
    #speed_lane_width = np.interp(v_ego*3.6, [0., 60.], [2.8, 3.5])
    if l_prob > 0.5 and r_prob > 0.5 and self.lane_change_multiplier > 0.5:
      both_lane_available = True
      #self.lane_width_updated_count = max_updated_count
      self.lane_width_estimate.update(current_lane_width)
      self.lane_width_last = self.lane_width_estimate.x
    #elif self.lane_width_updated_count <= 0 and v_ego > 0.1:   # 양쪽차선이 없을때.... 일정시간후(10초)부터 speed차선폭 적용함.
    #  self.lane_width_estimate.update(speed_lane_width)
    else:
      self.lane_width_estimate.update(self.lane_width_last)

    self.lane_width =  self.lane_width_estimate.x
    clipped_lane_width = min(4.0, self.lane_width)
    path_from_left_lane = self.lll_y + clipped_lane_width / 2.0
    path_from_right_lane = self.rll_y - clipped_lane_width / 2.0

    # 가장 차선이 진한쪽으로 골라서..
    self.d_prob = max(l_prob, r_prob) if not both_lane_available else 1.0

    # 좌/우의 차선폭을 필터링.
    if self.lane_width_left > 0:
      self.lane_width_left_filtered.update(self.lane_width_left)
      #self.lane_width_left_filtered.x = self.lane_width_left #바로적용
    if self.lane_width_right > 0:
      self.lane_width_right_filtered.update(self.lane_width_right)
      #self.lane_width_right_filtered.x = self.lane_width_right #바로적용

    self.adjustLaneOffset = float(self.params.get_int("AdjustLaneOffset")) * 0.01
    self.adjustCurveOffset = float(self.params.get_int("AdjustCurveOffset")) * 0.01
    self.adjustCurveMaxBoost = float(self.params.get_int("AdjustCurveMaxBoost")) * 0.01
    self.adjustOffsetLimit = float(self.params.get_int("AdjustOffsetLimitCm")) * 0.01
    ADJUST_OFFSET_LIMIT = self.adjustOffsetLimit #max(self.adjustLaneOffset, self.adjustCurveOffset), default 0.4, now adjustable via AdjustOffsetLimitCm
    offset_curve = 0.0
    ## curve offset
    _t_fade = np.clip((abs(curve_speed) - 30.0) / (200.0 - 30.0), 0.0, 1.0)
    _fade_scale = 1.0 - (_t_fade * _t_fade * (3 - 2 * _t_fade))
    _t_strong = np.clip((abs(curve_speed) - 30.0) / (60.0 - 30.0), 0.0, 1.0)
    _strong_scale = 1.0 + (1.0 - (_t_strong * _t_strong * (3 - 2 * _t_strong))) * self.adjustCurveMaxBoost  # adjustable via AdjustCurveMaxBoost (default 0.7 -> max 1.7x)
    offset_curve = self.adjustCurveOffset * _fade_scale * _strong_scale * np.sign(curve_speed)

    offset_lane = 0.0
    if self.d_prob > 0.3 and self.lane_width > 0:
      if self.lane_width > 3.3: #내 차로 자체가 너무 넓은 경우 - 우측통행이므로 중앙선쪽(왼쪽) 대신 갓길쪽(오른쪽)으로 붙음
        offset_lane = self.adjustLaneOffset
      elif self.lane_width_left_filtered.x > 2.2 and self.lane_width_right_filtered.x > 2.2: #양쪽에 차로가 여유 있는경우
        offset_lane = 0.0
      elif self.lane_width_left_filtered.x < 2.0 and self.lane_width_right_filtered.x < 2.0: #양쪽에 차로가 여유 없는경우
        offset_lane = 0.0
      elif self.lane_width_left_filtered.x > self.lane_width_right_filtered.x:
        offset_lane = np.interp(self.lane_width, [2.5, 2.9], [0.0, self.adjustLaneOffset]) # 차선이 좁으면 안함..
      else:
        offset_lane = np.interp(self.lane_width, [2.5, 2.9], [0.0, -self.adjustLaneOffset]) # 차선이 좁으면 안함..
    elif self.edge_valid:
      ## laneless: 도로 경계(연석 등) 기반 폭으로 갓길쪽 붙임 판단 - 근거리 점 평균 사용
      edge_width = float(np.mean(self.re_y[0:5] - self.le_y[0:5]))
      if 2.0 < edge_width < 8.0 and edge_width > 3.3:
        offset_lane = self.adjustLaneOffset

    #select lane path
    # 차선이 좁아지면, 도로경계쪽에 있는 차선 위주로 따라가도록함.
    if self.lane_width < 2.5:
      if r_prob > 0.5 and self.lane_width_right_filtered.x < self.lane_width_left_filtered.x:
        lane_path_y = path_from_right_lane
      elif l_prob > 0.5 and self.lane_width_left_filtered.x < 2.0:
        lane_path_y = path_from_left_lane
      else:
        lane_path_y = path_from_left_lane if l_prob > 0.5 or l_prob > r_prob else path_from_right_lane
    elif l_prob > 0.7 and r_prob > 0.7:
      lane_path_y = (path_from_left_lane + path_from_right_lane) / 2.
      # lane_width filtering에 의해서, 점점 줄어들때, 중앙선으로 붙어가는 현상이 생김.. 
      #if self.lane_width > 3.2:
      #  lane_path_y = path_from_right_lane
      #else:
      #  lane_path_y = (path_from_left_lane + path_from_right_lane) / 2.
    # 그외 진한차선을 따라가도록함.
    else:
      lane_path_y = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)

    ## 0.5초 앞의 중심을 보도록함. (정차물체 회피 감지용으로 항상 계산)
    lane_path_y_center = np.interp(0.5, path_t, lane_path_y)
    path_xyz_y_center = np.interp(0.5, path_t, path_xyz[:,1])
    diff_center = (lane_path_y_center - path_xyz_y_center) if not self.lanefull_mode else 0.0
    #print("center = {:.2f}={:.2f}-{:.2f}, lanefull={}".format(diff_center, lane_path_y_center, path_xyz_y_center, self.lanefull_mode))
    #diff_center = lane_path_y[5] - path_xyz[:,1][5] if not self.lanefull_mode else 0.0
    if offset_curve * offset_lane < 0:
      offset_total = np.clip(offset_curve + offset_lane, - ADJUST_OFFSET_LIMIT, ADJUST_OFFSET_LIMIT)
    else:
      offset_total = np.clip(max(offset_curve, offset_lane, key=abs), - ADJUST_OFFSET_LIMIT, ADJUST_OFFSET_LIMIT)

    ## self.d_prob = 0 if lane_changing
    self.d_prob *= self.lane_change_multiplier  ## 차선변경중에는 꺼버림.
    if self.lane_change_multiplier < 0.5:
      #self.lane_offset_filtered.x = 0.0
      pass
    else:
      if self.d_prob > 0.3:
        _offset_gate = 1.0
      elif self.edge_valid:
        _offset_gate = 1.0
      else:
        _offset_gate = np.interp(self.d_prob, [0, 0.3], [0, 1])
      self.lane_offset_filtered.update(_offset_gate * offset_total)

    ## YONG_AVOID_V2
    ## 실제 측전방 물체가 확인될 때만 모델 회피량(diff_center)을 추가 증폭한다.
    ##
    ## openpilot vehicle coordinates: +Y = LEFT, -Y = RIGHT
    ## +offset = 오른쪽 물체를 피해 왼쪽으로 회피
    ## -offset = 왼쪽 물체를 피해 오른쪽으로 회피
    ##
    ## 중요:
    ## 피하는 방향에 차량이 있으면 모델 원래 경로 자체를 지우는 것이 아니라
    ## 추가 증폭(avoid_offset)만 막는다.

    avoid_v2_enabled = bool(self.params.get_int("AvoidV2Enabled"))
    avoid_boost = float(self.params.get_int("AvoidOffsetBoostPct")) * 0.01
    avoid_attack_tau = max(0.01, float(self.params.get_int("AvoidAttackTauCs")) * 0.01)
    avoid_release_tau = max(0.01, float(self.params.get_int("AvoidReleaseTauCs")) * 0.01)

    v_kph = v_ego * 3.6

    # 코너레이더 거리
    lf_d = float(getattr(CS, "leftLongDist", 0.0))
    rf_d = float(getattr(CS, "rightLongDist", 0.0))
    lr_d = float(getattr(CS, "leftRearLongDist", 0.0))
    rr_d = float(getattr(CS, "rightRearLongDist", 0.0))

    bsd_l = bool(getattr(CS, "leftBlindspot", False))
    bsd_r = bool(getattr(CS, "rightBlindspot", False))

    # 우선 모델이 요구하는 회피 방향/크기를 계산한다.
    raw_avoid_target = np.clip(
      -diff_center * avoid_boost,
      -ADJUST_OFFSET_LIMIT,
      ADJUST_OFFSET_LIMIT,
    )

    # -----------------------------------------------------
    # 실제 물체 gate
    #
    # +target : 오른쪽 물체 때문에 왼쪽 회피
    # -target : 왼쪽 물체 때문에 오른쪽 회피
    #
    # 현재 Hyundai CarState가 전측방 차단 판단에 사용하는
    # 7m 기준과 동일하게 시작한다.
    # -----------------------------------------------------
    obstacle_gate = 0.0

    if raw_avoid_target > 0.02:
      # 왼쪽으로 피하는 상황 -> 오른쪽 장애물 확인
      obstacle_gate = 1.0 if 0.0 < rf_d < 7.0 else 0.0

    elif raw_avoid_target < -0.02:
      # 오른쪽으로 피하는 상황 -> 왼쪽 장애물 확인
      obstacle_gate = 1.0 if 0.0 < lf_d < 7.0 else 0.0

    # -----------------------------------------------------
    # 피하는 방향(side destination) 차량 확인
    #
    # 왼쪽으로 피할 때는 왼쪽 차량을,
    # 오른쪽으로 피할 때는 오른쪽 차량을 본다.
    #
    # 옆 차량이 있으면 '추가 증폭'만 제거한다.
    # -----------------------------------------------------
    side_blocked = False

    if raw_avoid_target > 0.02:
      # 왼쪽으로 이동하려는 상황 -> 왼쪽 공간 확인
      side_blocked = (
        bsd_l
        or 0.0 < lf_d < 7.0
        or 0.0 < lr_d < 7.0
      )

    elif raw_avoid_target < -0.02:
      # 오른쪽으로 이동하려는 상황 -> 오른쪽 공간 확인
      side_blocked = (
        bsd_r
        or 0.0 < rf_d < 7.0
        or 0.0 < rr_d < 7.0
      )

    side_scale = 0.0 if side_blocked else 1.0

    # -----------------------------------------------------
    # 속도 제한
    #
    # <= 50 km/h : 100 %
    # 50~60 km/h : 선형 감소
    # >= 60 km/h : 추가 증폭 없음
    # -----------------------------------------------------
    speed_scale = float(np.interp(
      v_kph,
      [0.0, 50.0, 60.0],
      [1.0, 1.0, 0.0],
    ))

    avoid_target = (
      raw_avoid_target
      * obstacle_gate
      * side_scale
      * speed_scale
      if avoid_v2_enabled
      else 0.0
    )

    # -----------------------------------------------------
    # 빠른 회피 / 느린 복귀
    # -----------------------------------------------------
    avoid_tau = (
      avoid_attack_tau
      if abs(avoid_target) > abs(self.avoid_offset_filtered_x)
      else avoid_release_tau
    )

    avoid_alpha = DT_MDL / (avoid_tau + DT_MDL)

    # -----------------------------------------------------
    # 복귀할 때 피하는 방향의 후측방 차량 확인
    #
    # +offset = 오른쪽 장애물을 피해 왼쪽에 있음 -> 복귀 전 오른쪽 후측방 확인
    # -offset = 왼쪽 장애물을 피해 오른쪽에 있음 -> 복귀 전 왼쪽 후측방 확인
    # -----------------------------------------------------
    bsd_hold = False

    if self.avoid_offset_filtered_x > 0.05:
      bsd_hold = (
        bsd_r
        or 0.0 < rr_d < 7.0
      )

    elif self.avoid_offset_filtered_x < -0.05:
      bsd_hold = (
        bsd_l
        or 0.0 < lr_d < 7.0
      )

    avoid_releasing = (
      abs(avoid_target) < abs(self.avoid_offset_filtered_x)
      and avoid_target * self.avoid_offset_filtered_x >= 0.0
    )

    # BSD hold 최대 3초.
    #
    # 3초가 지나면 옆 차량이 계속 있어도 추가 회피 오프셋은
    # release_tau 속도로 천천히 제거한다.
    hold_active = False

    if bsd_hold and avoid_releasing:
      if self.avoid_bsd_hold_time < 3.0:
        self.avoid_bsd_hold_time += DT_MDL
        hold_active = True
      else:
        hold_active = False
    else:
      self.avoid_bsd_hold_time = 0.0

    if not hold_active:
      self.avoid_offset_filtered_x = (
        (1.0 - avoid_alpha) * self.avoid_offset_filtered_x
        + avoid_alpha * avoid_target
      )

    # 기존 회피 동작 로그
    self._avoid_log_frame += 1
    if abs(raw_avoid_target) > 0.02 and self._avoid_log_frame % 20 == 0:
      cloudlog.info(
        f"CARROT_AVOID_V2 v={v_kph:.1f}kph "
        f"diff={diff_center:.3f} raw={raw_avoid_target:.3f} "
        f"gate={int(obstacle_gate)} sideBlock={int(side_blocked)} "
        f"speedScale={speed_scale:.2f} "
        f"target={avoid_target:.3f} offset={self.avoid_offset_filtered_x:.3f} "
        f"hold={int(hold_active)} holdTime={self.avoid_bsd_hold_time:.2f} "
        f"LF={lf_d:.2f} RF={rf_d:.2f} LR={lr_d:.2f} RR={rr_d:.2f} "
        f"bsdL={int(bsd_l)} bsdR={int(bsd_r)}"
      )

    # YONG_AVOID_SENSOR_LOG
    # 50km/h 이하에서는 회피가 발생하지 않아도 코너레이더 값을 기록한다.
    if v_ego * 3.6 <= 50.0 and self._avoid_log_frame % 20 == 0:
      lf_d = float(getattr(CS, "leftLongDist", 0.0))
      rf_d = float(getattr(CS, "rightLongDist", 0.0))
      lr_d = float(getattr(CS, "leftRearLongDist", 0.0))
      rr_d = float(getattr(CS, "rightRearLongDist", 0.0))

      lf_y = float(getattr(CS, "leftLatDist", 0.0))
      rf_y = float(getattr(CS, "rightLatDist", 0.0))
      lr_y = float(getattr(CS, "leftRearLatDist", 0.0))
      rr_y = float(getattr(CS, "rightRearLatDist", 0.0))

      if (lf_d > 0.0 or rf_d > 0.0 or lr_d > 0.0 or rr_d > 0.0
          or abs(self.avoid_offset_filtered_x) > 0.05):
        cloudlog.info(
          f"YONG_AVOID v={v_ego*3.6:.1f}kph "
          f"LF={lf_d:.2f}/{lf_y:.2f} RF={rf_d:.2f}/{rf_y:.2f} "
          f"LR={lr_d:.2f}/{lr_y:.2f} RR={rr_d:.2f}/{rr_y:.2f} "
          f"diff={diff_center:.3f} target={avoid_target:.3f} "
          f"offset={self.avoid_offset_filtered_x:.3f} "
          f"hold={int(bsd_hold)} "
          f"bsdL={int(bool(getattr(CS, 'leftBlindspot', False)))} "
          f"bsdR={int(bool(getattr(CS, 'rightBlindspot', False)))}"
        )

    ## laneless at lowspeed
    self.d_prob *= np.interp(v_ego*3.6, [5., 10.], [0.0, 1.0])

    #self.debugText = "OFFSET({:.2f}={:.2f}+{:.2f}+{:.2f}),Vc:{:.2f},dp:{:.1f},lf:{},lrw={:.1f}|{:.1f}|{:.1f}".format(
    #  self.lane_offset_filtered.x,
    #  diff_center, offset_lane, offset_curve,
    #  curve_speed,
    #  self.d_prob, self.lanefull_mode,
    #  self.lane_width_left_filtered.x, self.lane_width, self.lane_width_right_filtered.x)

    adjustLaneTime = self.params.get_float("LatMpcInputOffset") * 0.01 # 0.06 
    laneline_active = False
    self.d_prob_count = self.d_prob_count + 1 if self.d_prob > 0.3 else 0
    if self.lanefull_mode and self.d_prob_count > int(1 / DT_MDL):
      laneline_active = True
      use_dist_mode = False  ## 아무리생각해봐도.. 같은 방법인듯...
      if use_dist_mode:
        lane_path_y_interp = np.interp(path_xyz[:,0] + v_ego * adjustLaneTime, self.ll_x, lane_path_y)
        path_xyz[:,1] = self.d_prob * lane_path_y_interp + (1.0 - self.d_prob) * path_xyz[:,1]
      else:
        safe_idxs = np.isfinite(self.ll_t)
        if safe_idxs[0]:
          lane_path_y_interp = np.interp(path_t * (1.0 + adjustLaneTime), self.ll_t[safe_idxs], lane_path_y[safe_idxs])
          path_xyz[:,1] = self.d_prob * lane_path_y_interp + (1.0 - self.d_prob) * path_xyz[:,1]


    ## YongPilot 정차 출발 회전 의도 (standstill_turn_intent v2.1)
    TURN_INTENT_BOOST_PCT = 25.0
    try:
      _ti_dir, _ti_ramp, _ti_straight = self.turn_intent.update(
        DT_MDL, v_ego,
        bool(getattr(CS, 'leftBlinker', False)),
        bool(getattr(CS, 'rightBlinker', False)),
        bool(getattr(CS, 'steeringPressed', False)),
        float(getattr(CS, 'yawRate', 0.0)),
        path_xyz[:, 0], path_xyz[:, 1])
    except Exception:
      self.turn_intent.reset()
      _ti_dir, _ti_ramp, _ti_straight = 0, 0.0, False

    ## 교차로 등 laneless 저속 급커브에서 회전 부족 보정 - 웹당근에서 조정 가능
    intersection_turn_boost = float(self.params.get_int("IntersectionTurnBoostPct")) * 0.01
    if intersection_turn_boost > 0.0 and abs(curve_speed) > 0.1 and not _ti_straight:
      _laneless_gate = np.clip(1.0 - self.d_prob / 0.3, 0.0, 1.0)
      _sharp_gate = np.clip((40.0 - abs(curve_speed)) / (40.0 - 10.0), 0.0, 1.0)
      _dist_scale = np.clip(path_t / 3.0, 0.0, 1.0)
      turn_boost_offset = intersection_turn_boost * _laneless_gate * _sharp_gate * _dist_scale * np.sign(curve_speed) * 1.0
      path_xyz[:, 1] += turn_boost_offset

    if _ti_dir != 0 and _ti_ramp > 0.0:
      try:
        path_xyz = apply_turn_boost(path_xyz, _ti_dir, _ti_ramp, TURN_INTENT_BOOST_PCT)
      except Exception:
        pass

    _ti_lane_offset = 0.0 if _ti_straight else self.lane_offset_filtered.x
    path_xyz[:, 1] += (CAMERA_OFFSET + _ti_lane_offset + self.avoid_offset_filtered_x)

    # 관찰 전용 기록 (기능 작동 중일 때만, 0.25초 간격)
    if _ti_dir != 0 or _ti_straight:
      self._ti_log_frame += 1
      if self._ti_log_frame % 5 == 1:
        try:
          _new = not os.path.exists('/data/turn_intent_events.csv')
          with open('/data/turn_intent_events.csv', 'a') as _f:
            if _new:
              _f.write('time,kph,dir,ramp,straight,lblink,rblink,yawRate,y8m\n')
            _y8 = float(np.interp(8.0, path_xyz[:, 0], path_xyz[:, 1]))
            _f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{v_ego*3.6:.1f},{_ti_dir},{_ti_ramp:.2f},"
                     f"{int(_ti_straight)},{int(bool(getattr(CS, 'leftBlinker', False)))},"
                     f"{int(bool(getattr(CS, 'rightBlinker', False)))},"
                     f"{float(getattr(CS, 'yawRate', 0.0)):.3f},{_y8:.2f}\n")
        except Exception:
          pass
    else:
      self._ti_log_frame = 0

    self.offset_total = self.lane_offset_filtered.x + self.avoid_offset_filtered_x

    return path_xyz, laneline_active

  def calculate_plan_yaw_and_yaw_rate(self, path_xyz):
    if path_xyz.shape[0] < 3:
        # 너무 짧으면 직진 가정
        N = path_xyz.shape[0]
        return np.zeros(N), np.zeros(N)

    # x, y 추출
    x = path_xyz[:, 0]
    y = path_xyz[:, 1]

    # 모두 동일한 점인지 확인
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return np.zeros(len(x)), np.zeros(len(x))

    # 안전한 diff 계산
    dx = np.diff(x)
    dy = np.diff(y)
    mask = (dx == 0) & (dy == 0)
    dx[mask] = 1e-4
    dy[mask] = 0.0

    yaw = np.arctan2(dy, dx)
    yaw = np.append(yaw, yaw[-1])  # N-1 → N
    yaw = np.unwrap(yaw)

    dx_full = np.clip(np.diff(x), 1e-4, None)
    yaw_rate = np.diff(yaw) / dx_full
    yaw_rate = np.append(yaw_rate, yaw_rate[-1])
    yaw_rate = np.append(yaw_rate, 0.0)

    # NaN/Inf 방어
    if np.any(np.isnan(yaw_rate)) or np.any(np.isinf(yaw_rate)):
        yaw_rate = np.zeros_like(yaw_rate)
    if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
        yaw = np.zeros_like(yaw)

    return yaw, yaw_rate
