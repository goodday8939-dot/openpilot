"""Retry policy for explicitly supervised stationary YOLO sessions only."""
from collections import deque
import re


class RecoveryPolicy:
  def __init__(self):
    self.attempts = deque(maxlen=3)

  @staticmethod
  def recoverable(reason):
    # A timing anomaly does not establish its cause. Never retry GPU errors,
    # overruns, changed owners, changed Park/control guards or operator signals.
    return bool(re.fullmatch(r'(modelV2|roadCameraState|wideRoadCameraState|driverCameraState) raw gap [0-9.]+ ms', reason)
                or reason == 'driving execution/drop guard exceeded')

  def reserve(self, reason, now):
    if not self.recoverable(reason):
      return None
    while self.attempts and now-self.attempts[0] >= 900:
      self.attempts.popleft()
    if len(self.attempts) >= 3:
      return None
    stable_seconds = (30, 60, 120)[len(self.attempts)]
    self.attempts.append(now)
    return stable_seconds
