from openpilot.common.realtime import DT_CTRL
from openpilot.common.params import Params

class CarrotControls:
  def __init__(self, CP):
    self.CP = CP
    self.params = Params()
    self.lat_suspend_active = False
    self.lat_suspend_enter_t = 0.0
    self.lat_suspend_exit_t = 0.0

  def lat_suspend_control(self, CS, latActive):
    suspend_angle = float(self.params.get_int("LatSuspendAngleDeg"))
    resume_angle  = 15
    delay_sec     = 0.5   # lowered from 1.0: angle OR torque only needs to hold this long to yield
    torque_threshold = 1.0   # Nm 등 — CS.steeringTorque 실제 범위 보고 조정 필요
    speed_limit_kph = 40.0   # 이 속도(km/h) 이하에서만 손 조향 양보 기능 작동

    # 1) 진입 조건: 각도 OR 토크(힘) 둘 중 하나만 넘어도 됨 -- 세게 힘을 줬다면
    # 아직 각도가 크게 안 돌아갔어도 바로 양보 시작
    enter_cond = (
      CS.steeringPressed
      and (abs(CS.steeringAngleDeg) > suspend_angle or abs(CS.steeringTorque) > torque_threshold)
      and CS.vEgo < (speed_limit_kph / 3.6)
    )
    if not self.lat_suspend_active:
      if enter_cond:
        self.lat_suspend_enter_t += DT_CTRL
        if self.lat_suspend_enter_t >= delay_sec:
          self.lat_suspend_active = True
      else:
        self.lat_suspend_enter_t = 0.0

    # 2) 손을 떼면(각도/토크 조건 풀리면) 재개 -- 급커브 근처에서 진입/해제가
    # 반복 진동하는 걸 막기 위한 최소 디바운스만 남겨둠(짧게 단축)
    if self.lat_suspend_active:
      exit_cond = (
        (abs(CS.steeringAngleDeg) < resume_angle and not CS.steeringPressed)
        or (CS.vEgo >= (speed_limit_kph / 3.6))
      )
      exit_delay_sec = 0.15   # lowered from 0.3
      if exit_cond:
        self.lat_suspend_exit_t += DT_CTRL
        if self.lat_suspend_exit_t >= exit_delay_sec:
          self.lat_suspend_active = False
          self.lat_suspend_enter_t = 0.0
          self.lat_suspend_exit_t = 0.0
      else:
        self.lat_suspend_exit_t = 0.0

    if self.lat_suspend_active:
      latActive = False
    return latActive
