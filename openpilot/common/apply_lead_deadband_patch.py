#!/usr/bin/env python3
"""Lowers LEAD_ACCEL_DEADBAND so the launch-response logic reacts to a lead
car that has only just started creeping forward (small positive lead accel),
instead of waiting until the lead's acceleration exceeds 0.075 m/s^2.

Run once on the comma device:
  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 /data/openpilot/apply_lead_deadband_patch.py

Safe to run more than once -- it checks whether the patch is already applied
and does nothing if so.
"""
import sys

TARGET = "/data/openpilot/openpilot/selfdrive/controls/lib/longitudinal_preview.py"

OLD_LINE = "LEAD_ACCEL_DEADBAND = 0.075\n"
NEW_LINE = "LEAD_ACCEL_DEADBAND = 0.05  # lowered from 0.075: react sooner to a lead that just started creeping\n"

MARKER = "lowered from 0.075"


def main():
    with open(TARGET, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER in content:
        print("Already patched, nothing to do.")
        return 0

    if OLD_LINE not in content:
        print("ERROR: expected line not found, aborting without changes.")
        return 1

    content = content.replace(OLD_LINE, NEW_LINE, 1)

    backup_path = TARGET + ".bak_before_lead_deadband"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(open(TARGET, "r", encoding="utf-8").read())

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(content)

    print("Patched successfully.")
    print("Backup saved at:", backup_path)
    print("Restart openpilot (reboot the device) for the change to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
