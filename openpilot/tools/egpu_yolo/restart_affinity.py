"""Keep a manually supervised restart's launcher off TICI's isolated cores."""
import os
from pathlib import Path
import subprocess


def prepare_restart_affinity():
  # tmux creates the new pane, so constraining only our Popen child is not
  # sufficient when a server already exists. General daemons otherwise inherit
  # the server's isolated CPU and can starve behind modeld/plannerd's FIFO work.
  server = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', 'comma', '#{pid}'], text=True))
  if Path(f'/proc/{server}/comm').read_text().strip() != 'tmux: server':
    raise RuntimeError('restart launcher is not the expected tmux server')
  cores = {0, 1, 2, 3}
  before = {pid: sorted(os.sched_getaffinity(pid)) for pid in (0, server)}
  if any(not cores.issubset(affinity) or os.sched_getscheduler(pid) != os.SCHED_OTHER for pid, affinity in before.items()):
    raise RuntimeError('restart launcher must permit housekeeping cores with ordinary scheduling')
  for pid in before:
    os.sched_setaffinity(pid, cores)
  # No model, camera, planner, controller or existing manager child is moved.
  # Their explicit affinity/priority setup remains in their own startup paths.
  return {'tmux_server_pid': server, 'before': before, 'after': sorted(cores)}
