"""Persistent opt-in and recovery policy for observational resident YOLO."""
import json
import os
from functools import cache
from pathlib import Path


def directory():
  return Path(os.getenv('EGPU_YOLO_DIR', '/data/egpu_yolo'))


def configured():
  try:
    value = json.loads((directory()/'auto_enabled.json').read_text())
    return (isinstance(value, dict) and value.get('version') == 1 and value.get('enabled') is True
            and (directory()/'egpu2-reuse/yolo_reuse.pkl').is_file() and not (directory()/'qcom_enabled').exists())
  except (OSError, ValueError):
    return False


@cache
def boot_id():
  return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


class AutomaticRecovery:
  """No gear/standstill requirement, finite sessions, or terminal retry count.

  Transient stream/timing faults need 2/5/10/30 healthy seconds; five minutes of
  successful operation resets backoff. A failed GPU call remains quarantined
  until the normal manager supplies a different model owner.
  """
  def __init__(self):
    self.owner = None
    self.enabled = False
    self.stable_since = None
    self.running_since = None
    self.failures = 0
    self.generation = 0
    self.fatal = ''
    self.reason = 'waiting for driving model'

  @property
  def delay(self):
    return (2., 5., 10., 30.)[min(max(self.failures-1, 0), 3)]

  def update(self, now, *, owner, started, healthy, reason='', fatal=''):
    if owner != self.owner:
      self.__init__()
      self.owner = owner
    if not started or owner is None:
      self.enabled = False
      self.stable_since = self.running_since = None
      self.reason = 'offroad' if not started else 'waiting for driving model'
      return False
    if fatal:
      self.fatal = fatal
    if not healthy or self.fatal:
      if self.enabled:
        self.failures += 1
      self.enabled = False
      self.stable_since = self.running_since = None
      self.reason = self.fatal or reason
      return False
    if self.enabled:
      if now-self.running_since >= 300:
        self.failures = 0
      return True
    if self.stable_since is None:
      self.stable_since = now
    if now-self.stable_since >= self.delay:
      self.enabled = True
      self.running_since = now
      self.generation += 1
      self.reason = ''
    return self.enabled
