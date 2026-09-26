import numpy as np
from openpilot.cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.params import Params

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

HYUNDAI_LONGITUDINAL_KP = 1.0
HYUNDAI_LONGITUDINAL_KI = 0.0
HYUNDAI_LONGITUDINAL_KF = 1.0

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill, a_ego, stopping_accel, radarState):
  stopping_condition = should_stop
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      # YongPilot stop-hold:
      # 저속 정차 직전/직후 shouldStop이 잠깐 흔들려도,
      # 가까운 앞차가 실제로 출발하지 않았다면 브레이크를 풀지 않는다.
      leadOne = radarState.leadOne
      hold_stopped_lead = (
        v_ego < 0.55 and
        leadOne.status and
        0.2 < leadOne.dRel < 4.0 and
        leadOne.vRel <= 0.18
      )

      if not hold_stopped_lead:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        elif starting_condition:
          long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        stopping_accel = stopping_accel if stopping_accel < 0.0 else -0.5
        leadOne = radarState.leadOne
        fcw_stop = leadOne.status and leadOne.dRel < 4.0
        if a_ego > stopping_accel or fcw_stop: # and v_ego < 1.0:
          long_control_state = LongCtrlState.stopping
        if long_control_state == LongCtrlState.starting:
          long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             k_f=CP.longitudinalTuning.kf, rate=1 / DT_CTRL)
    self.last_output_accel = 0.0

    # ========================================================
    # Yong Start / Low-speed Follow v1
    # ========================================================

    # 완전정지 후 앞차 출발 확인 시간
    self.lead_start_confirm_normal_s = 0.10
    self.lead_start_confirm_ready_s = 0.05

    # 옆 차선 차량 움직임 감지 시 READY 유지시간
    self.side_ready_hold_s = 1.00
    self.side_ready_timer_s = 0.0

    self.lead_start_timer_s = 0.0
    self.lead_start_confirm_timer_s = 0.0
    self.lead_start_anchor_drel = None
    self.lead_start_prev_drel = None
    self.lead_start_prev_track_id = -1
    self.lead_start_prev_vrel = 0.0
    self.lead_start_motion_seen = False

    # 감속 중 앞차 재출발
    self.rolling_restart_timer_s = 0.0
    self.rolling_restart_active = False

    # 정차 후 초기 가속
    self.starting_accel_timer_s = 0.0
    self.starting_accel_ramp_s = 0.30
    self.starting_accel_fast_ramp_s = 0.20

    # 저속 안정 추종
    self.follow_cruise_active = False

    self.params = Params()
    self.readParamCount = 0
    self.stopping_accel = self.params.get_float("StoppingAccel") * 0.01
    if CP.brand == "hyundai" and self.stopping_accel == 0.0:
      # Restore the default at startup for Hyundai, Kia, and Genesis instead
      # of retaining the legacy stop target selected by a persisted zero.
      self.params.put_int("StoppingAccel", -50)
      self.stopping_accel = -0.5
    self.j_lead = 0.0

    self.hyundai_fixed_longitudinal_tuning = CP.brand == "hyundai"
    if self.hyundai_fixed_longitudinal_tuning:
      self._apply_hyundai_longitudinal_tuning()

    self.use_accel_pid = False
    if CP.brand == "toyota":
      self.use_accel_pid = True

  def _apply_hyundai_longitudinal_tuning(self):
    # Hyundai, Kia, and Genesis all use the opendbc "hyundai" brand. Keep the
    # complete acceleration/deceleration feedforward path intact instead of
    # allowing a stale or unsafe persistent tuning value to override it.
    self.pid._k_p = ([0.0], [HYUNDAI_LONGITUDINAL_KP])
    self.pid._k_i = ([0.0], [HYUNDAI_LONGITUDINAL_KI])
    self.pid.k_f = HYUNDAI_LONGITUDINAL_KF

  def _refresh_longitudinal_tuning(self):
    if self.hyundai_fixed_longitudinal_tuning:
      self._apply_hyundai_longitudinal_tuning()
    elif len(self.CP.longitudinalTuning.kpBP) == 1 and len(self.CP.longitudinalTuning.kiBP) == 1:
      longitudinalTuningKpV = self.params.get_float("LongTuningKpV") * 0.01
      longitudinalTuningKiV = self.params.get_float("LongTuningKiV") * 0.001
      self.pid._k_p = (self.CP.longitudinalTuning.kpBP, [longitudinalTuningKpV])
      self.pid._k_i = (self.CP.longitudinalTuning.kiBP, [longitudinalTuningKiV])
      self.pid.k_f = self.params.get_float("LongTuningKf") * 0.01

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, long_plan, accel_limits, t_since_plan, radarState):

    soft_hold_active = CS.softHoldActive > 0
    a_target_ff = long_plan.aTarget
    v_target_now = long_plan.vTargetNow
    j_target_now = long_plan.jTargetNow
    should_stop = long_plan.shouldStop

    self.readParamCount += 1
    if self.readParamCount >= 100:
      self.readParamCount = 0
      self.stopping_accel = self.params.get_float("StoppingAccel") * 0.01
    elif self.readParamCount == 10:
      self._refresh_longitudinal_tuning()


    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    # ========================================================
    # Yong Fast Start / SIDE READY / Rolling Restart
    # ========================================================

    lead = radarState.leadOne

    valid_lead = (
      lead.status and
      lead.radar and
      lead.radarTrackId >= 0 and
      lead.dRel > 0.2
    )

    ego_stopped = abs(CS.vEgo) <= 0.10
    low_speed = abs(CS.vEgo) < (20.0 / 3.6)
    rolling_speed = 0.10 < abs(CS.vEgo) < (10.0 / 3.6)

    desired_distance = float(
      getattr(long_plan, "desiredDistance", 0.0)
    )

    # --------------------------------------------------------
    # SIDE READY
    #
    # 옆 차량은 내 차를 직접 출발시키지 않는다.
    # 단지 앞차 출발 감시를 READY 상태로 만든다.
    # --------------------------------------------------------

    side_departure = False

    for side_name in ("leadLeft", "leadRight"):
      side = getattr(radarState, side_name, None)

      if side is None or not getattr(side, "status", False):
        continue

      side_vlead = float(getattr(side, "vLead", 0.0))
      side_vrel = float(getattr(side, "vRel", 0.0))
      side_alead = float(
        getattr(
          side,
          "aLeadK",
          getattr(side, "aLead", 0.0)
        )
      )

      if (
        side_vlead > 0.8 or
        side_vrel > 0.6 or
        side_alead > 0.30
      ):
        side_departure = True
        break

    # YongPilot: corner180 front-side radar READY assist.
    # Adjacent-lane movement only prepares READY state.
    # It never releases should_stop by itself.
    if not side_departure and abs(CS.vEgo) < (3.0 / 3.6):
      for pt in getattr(radarState, "points", []):
        if str(getattr(pt, "radarSource", "")) != "corner180":
          continue
        if not bool(getattr(pt, "measured", False)):
          continue

        side_x = float(getattr(pt, "dRel", 0.0))
        side_y = float(getattr(pt, "yRel", 0.0))
        side_vrel = float(getattr(pt, "vRel", 0.0))
        side_arel = float(getattr(pt, "aRel", 0.0))

        front_side_object = (
          -1.0 < side_x < 15.0 and
          1.2 < abs(side_y) < 5.0
        )

        side_moving = (
          side_vrel > 0.50 or
          side_arel > 0.30
        )

        if front_side_object and side_moving:
          side_departure = True
          break

    if side_departure and abs(CS.vEgo) < (3.0 / 3.6):
      self.side_ready_timer_s = self.side_ready_hold_s
    else:
      self.side_ready_timer_s = max(
        0.0,
        self.side_ready_timer_s - DT_CTRL
      )

    side_ready = self.side_ready_timer_s > 0.0


    # --------------------------------------------------------
    # 앞차 출발 증거
    #
    # 한 센서값만 믿지 않고 여러 움직임을 함께 본다.
    # --------------------------------------------------------

    lead_motion_score = 0

    if valid_lead:

      current_drel = float(lead.dRel)
      current_track_id = int(lead.radarTrackId)

      same_track = (
        self.lead_start_prev_track_id == current_track_id and
        self.lead_start_prev_drel is not None
      )

      drel_delta = (
        current_drel - self.lead_start_prev_drel
        if same_track else 0.0
      )

      lead_vlead = float(lead.vLead)
      lead_vrel = float(lead.vRel)

      # YongPilot vRel spike rejection:
      # A single radar vRel spike must not release stop hold.
      prev_vrel = (
        self.lead_start_prev_vrel
        if same_track else 0.0
      )

      persistent_vrel = (
        lead_vrel > 0.10 and
        prev_vrel > 0.10
      )

      lead_alead = float(
        getattr(
          lead,
          "aLeadK",
          getattr(lead, "aLead", 0.0)
        )
      )

      lead_jlead = float(
        getattr(lead, "jLead", 0.0)
      )

      if lead_vlead > 0.20:
        lead_motion_score += 1

      if lead_vrel > 0.15:
        lead_motion_score += 1

      if drel_delta > 0.03:
        lead_motion_score += 1

      if lead_alead > 0.10:
        lead_motion_score += 1

      if lead_jlead > 0.15:
        lead_motion_score += 1

      self.lead_start_prev_drel = current_drel
      self.lead_start_prev_track_id = current_track_id
      self.lead_start_prev_vrel = lead_vrel

    else:
      self.lead_start_prev_drel = None
      self.lead_start_prev_track_id = -1
      self.lead_start_prev_vrel = 0.0


    # 최소 2개의 증거가 같은 방향일 때 출발 움직임으로 판단
    # Yong Fast Start:
    # 강한 출발은 기존 점수 기반으로 판정.
    # 천천히 출발하는 앞차는 vLead + vRel이 함께 유지되면 조기에 인정.
    # Strong departure: obvious lead movement is accepted immediately.
    strong_departure = (
      valid_lead and
      lead_vrel > 0.40
    )

    # Slow departure:
    # require either one-cycle persistence or actual distance increase.
    # This adds almost no perceptible delay but rejects one-frame vRel spikes.
    slow_departure = (
      valid_lead and
      lead_vlead > 0.12 and
      lead_vrel > 0.10 and
      (
        persistent_vrel or
        drel_delta > 0.02
      )
    )

    # Even a high motion score must have temporal/position confirmation,
    # unless the departure is already strong and obvious.
    scored_departure = (
      valid_lead and
      lead_motion_score >= 3 and
      (
        persistent_vrel or
        drel_delta > 0.02
      )
    )

    lead_motion = (
      strong_departure or
      slow_departure or
      scored_departure
    )


    # --------------------------------------------------------
    # 완전정지 → 출발
    # --------------------------------------------------------

    if not active or CS.brakePressed:

      self.lead_start_timer_s = 0.0
      self.lead_start_confirm_timer_s = 0.0
      self.lead_start_motion_seen = False

      self.rolling_restart_timer_s = 0.0
      self.rolling_restart_active = False

      if not active:
        self.side_ready_timer_s = 0.0

    elif (
      ego_stopped and
      self.long_control_state == LongCtrlState.stopping
    ):

      confirm_time = (
        self.lead_start_confirm_ready_s
        if side_ready
        else self.lead_start_confirm_normal_s
      )

      if lead_motion:
        self.lead_start_confirm_timer_s += DT_CTRL
      else:
        # 완전히 0으로 즉시 지우지 않고 약간의 hysteresis
        self.lead_start_confirm_timer_s = max(
          0.0,
          self.lead_start_confirm_timer_s - DT_CTRL * 2.0
        )

      if self.lead_start_confirm_timer_s >= confirm_time:
        self.lead_start_motion_seen = True

      # ★ 기존 0.50초 추가 강제대기 없음
      if self.lead_start_motion_seen:
        should_stop = False
      else:
        should_stop = True


    # --------------------------------------------------------
    # Rolling Restart disabled
    # 감속 중 lead_motion으로 should_stop을 강제 해제하지 않는다.
    # --------------------------------------------------------

    self.rolling_restart_timer_s = 0.0
    self.rolling_restart_active = False

    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill, CS.aEgo, self.stopping_accel, radarState)
    if active and soft_hold_active:
      self.long_control_state = LongCtrlState.stopping

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel

      # 다음 출발 ramp는 항상 0부터 시작
      self.starting_accel_timer_s = 0.0

      if soft_hold_active:
        output_accel = self.CP.stopAccel

      else:
        stopAccel = (
          self.stopping_accel
          if self.stopping_accel < 0.0
          else self.CP.stopAccel
        )

        if output_accel > stopAccel:
          output_accel = min(output_accel, 0.0)
          output_accel -= self.CP.stoppingDecelRate * DT_CTRL


        # =====================================================
        # Brake Release Taper
        #
        # 강하게 감속한 뒤 마지막 저속구간에서
        # 브레이크를 서서히 풀어 몸이 앞으로 튀는 느낌 감소.
        # =====================================================

        if valid_lead:

          v_kph = abs(CS.vEgo) * 3.6
          lead_vrel = float(lead.vRel)

          # 목표거리가 아주 짧더라도 앞차에 너무 붙은 상태에서는
          # taper를 하지 않는다.
          hard_min_distance = max(
            2.0,
            desired_distance * 0.75
          )

          # YongPilot stop-taper v2: 0.8km/h 아래로 내려가면 taper 재진입 금지
          if v_kph < 0.8:
            self.stop_taper_latched = True
          elif v_kph > 3.0:
            self.stop_taper_latched = False

          safe_to_taper = (
            not getattr(self, 'stop_taper_latched', False) and
            0.8 < v_kph < 8.0 and
            lead_vrel > -0.50 and
            float(lead.dRel) > hard_min_distance
          )

          if safe_to_taper:

            taper_target = float(np.interp(
              v_kph,
              [0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
              [-0.30, -0.32, -0.36, -0.45, -0.55, -0.70]
            ))

            # 한 번에 brake를 풀지 않고 jerk 제한을 둔다.
            brake_release_step = 2.0 * DT_CTRL

            if output_accel < taper_target:
              output_accel = min(
                taper_target,
                output_accel + brake_release_step
              )

      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      # YongPilot v1: 빠르게 출발하되 첫 순간의 jerk만 부드럽게.
      self.starting_accel_timer_s += DT_CTRL

      lead = radarState.leadOne
      valid_start_lead = (
        lead.status and
        lead.radar and
        lead.radarTrackId >= 0 and
        lead.dRel > 0.2
      )

      lead_a = (
        float(getattr(
          lead,
          "aLeadK",
          getattr(lead, "aLead", 0.0)
        ))
        if valid_start_lead else 0.0
      )

      lead_vrel = (
        float(lead.vRel)
        if valid_start_lead else 0.0
      )

      # 앞차가 강하게 빠져나가면 부드러운 구간을 더 짧게.
      strong_lead_launch = (
        lead_a > 1.20 or
        lead_vrel > 1.00
      )

      ramp_time = (
        0.20 if strong_lead_launch
        else 0.30
      )

      ramp = min(
        1.0,
        self.starting_accel_timer_s / max(ramp_time, DT_CTRL)
      )

      # 기존 0.35 시작보다 첫 순간을 약하게.
      # 단, 0.2~0.3초 안에 CP.startAccel까지 바로 회복.
      start_accel_floor = min(
        0.15,
        self.CP.startAccel
      )

      output_accel = (
        start_accel_floor +
        (self.CP.startAccel - start_accel_floor) * ramp
      )

      self.reset()

    else:  # LongCtrlState.pid

      self.starting_accel_timer_s = 0.0

      # =====================================================
      # Low-speed FOLLOW_CRUISE
      #
      # 작은 거리/속도 변화마다 액셀-감속을 반복하지 않는다.
      #
      # 비대칭:
      #   가까워지는 쪽 = 민감
      #   멀어지는 쪽   = 조금 더 여유
      # =====================================================

      follow_cruise = False

      if (
        low_speed and
        valid_lead and
        desired_distance > 0.1
      ):

        gap_margin = (
          float(lead.dRel) -
          desired_distance
        )

        lead_vrel = float(lead.vRel)

        lead_alead = float(
          getattr(
            lead,
            "aLeadK",
            getattr(lead, "aLead", 0.0)
          )
        )

        # 절대거리 보호
        hard_close_distance = max(
          2.0,
          desired_distance * 0.75
        )

        hard_close = (
          float(lead.dRel) < hard_close_distance
        )


        # 이미 FOLLOW_CRUISE 중이면 조금 넓은 범위까지 유지
        if self.follow_cruise_active:

          close_escape = (
            hard_close or
            gap_margin < -0.25 or
            lead_vrel < -0.35 or
            lead_alead < -0.60
          )

          opening_escape = (
            gap_margin > 0.80 or
            lead_vrel > 0.40 or
            lead_alead > 0.50
          )

          follow_cruise = (
            not close_escape and
            not opening_escape
          )

        else:

          # 처음 진입할 때는 더 안정된 상태에서만 들어감
          follow_cruise = (
            not hard_close and
            -0.15 <= gap_margin <= 0.60 and
            -0.25 <= lead_vrel <= 0.30 and
            -0.40 <= lead_alead <= 0.40
          )


      self.follow_cruise_active = follow_cruise


      if self.use_accel_pid:
        error = a_target_ff - CS.aEgo
      else:
        error = v_target_now - CS.vEgo


      raw_output_accel = self.pid.update(
        error,
        speed=CS.vEgo,
        feedforward=a_target_ff
      )


      # =====================================================
      # YONG HIGH-SPEED FOLLOW SMOOTH v1
      # 50km/h 이상에서 작은 거리/상대속도 오차는 허용한다.
      # 45~55km/h에서 서서히 개입해서 경계 꿀렁임을 줄인다.
      # =====================================================
      high_speed_follow_smooth = False
      high_speed_smooth_strength = 0.0

      if (
        valid_lead
        and desired_distance > 0.1
        and CS.vEgo >= (45.0 / 3.6)
      ):
        high_gap_margin = float(lead.dRel) - desired_distance
        high_vrel = float(lead.vRel)
        high_alead = float(
          getattr(
            lead, "aLeadK", getattr(lead, "aLead", 0.0)
          )
        )

        high_close_escape = (
          high_gap_margin < -1.50
          or high_vrel < -0.80
          or high_alead < -1.00
        )

        high_open_escape = (
          high_gap_margin > 2.00
          or high_vrel > 0.80
        )

        high_comfort_zone = (
          -1.00 <= high_gap_margin <= 1.30
          and -0.20 <= high_vrel <= 0.25
          and -0.60 <= high_alead <= 0.60
        )

        high_speed_follow_smooth = (
          high_comfort_zone
          and not high_close_escape
          and not high_open_escape
        )

        if high_speed_follow_smooth:
          high_speed_smooth_strength = float(np.interp(
            CS.vEgo * 3.6,
            [45.0, 55.0],
            [0.0, 1.0]
          ))

      if high_speed_follow_smooth:
        HIGH_SPEED_FOLLOW_TAU = 0.80
        high_alpha = DT_CTRL / (HIGH_SPEED_FOLLOW_TAU + DT_CTRL)

        high_smoothed_accel = (
          self.last_output_accel
          + high_alpha * (raw_output_accel - self.last_output_accel)
        )

        output_accel = (
          raw_output_accel * (1.0 - high_speed_smooth_strength)
          + high_smoothed_accel * high_speed_smooth_strength
        )

      elif follow_cruise:

        # 안정구간에서만 출력 변화 완화.
        # 위험 접근/거리 벌어짐 조건이 발생하면 다음 frame부터
        # FOLLOW_CRUISE를 즉시 빠져나가므로 이 smoothing도 사라진다.
        FOLLOW_CRUISE_TAU = 0.35

        alpha = (
          DT_CTRL /
          (FOLLOW_CRUISE_TAU + DT_CTRL)
        )

        output_accel = (
          self.last_output_accel +
          alpha *
          (raw_output_accel - self.last_output_accel)
        )

      else:
        output_accel = raw_output_accel

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel, a_target_ff, j_target_now
