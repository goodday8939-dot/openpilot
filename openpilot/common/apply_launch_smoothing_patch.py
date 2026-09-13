#!/usr/bin/env python3
"""Patches long_mpc.py so the lead-accel response cost factors ramp in and
out smoothly instead of snapping instantly, fixing the launch
crawl -> hard surge -> hard release pattern.

Run once on the comma device:
  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 /data/openpilot/apply_launch_smoothing_patch.py

Safe to run more than once -- it checks whether the patch is already applied
and does nothing if so.
"""
import sys

TARGET = "/data/openpilot/openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"

MARKER = "RESPONSE_FACTOR_RAMP_TIME_S"

EDITS = [
    (
        "A_CHANGE_COST_STARTING = 10. #30.\n",
        "A_CHANGE_COST_STARTING = 10. #30.\n"
        "\n"
        "# Lead-accel response used to switch cost factors instantly (1.0 <-> tuning\n"
        "# value) the frame it engaged or disengaged. That produced a crawl -> hard\n"
        "# surge -> hard release pattern on launch: normal high-cost driving until the\n"
        "# response condition flipped on, then an abrupt full-strength low-cost boost,\n"
        "# then an abrupt full-strength cutoff once the gap was caught up. Ramping the\n"
        "# applied factor toward its target over this many seconds keeps the ego\n"
        "# accelerating in step with the departing lead instead of snapping between\n"
        "# the two extremes.\n"
        "RESPONSE_FACTOR_RAMP_TIME_S = 1.2\n",
    ),
    (
        "    self.a_change_cost = A_CHANGE_COST\n"
        "    self.jerk_cost_factor = 1.0\n"
        "    self.lead_accel_response_active = False\n"
        "    self.lead_accel_response_level = 0\n"
        "\n"
        "    self.reset()\n",
        "    self.a_change_cost = A_CHANGE_COST\n"
        "    self.jerk_cost_factor = 1.0\n"
        "    self.lead_accel_response_active = False\n"
        "    self.lead_accel_response_level = 0\n"
        "    # Smoothed (rate-limited) versions of the response cost factors, applied\n"
        "    # to set_weights instead of the raw on/off request. 1.0 = no boost.\n"
        "    self._response_a_change_factor = 1.0\n"
        "    self._response_jerk_factor = 1.0\n"
        "\n"
        "    self.reset()\n",
    ),
    (
        "    self.lead_accel_response_active = False\n"
        "    self.lead_accel_response_level = 0\n"
        "    # timers\n",
        "    self.lead_accel_response_active = False\n"
        "    self.lead_accel_response_level = 0\n"
        "    self._response_a_change_factor = 1.0\n"
        "    self._response_jerk_factor = 1.0\n"
        "    # timers\n",
    ),
    (
        "    self.lead_accel_response_active = response_request.active\n"
        "    self.lead_accel_response_level = response_request.level if response_request.active else 0\n"
        "    self.set_weights(\n"
        "      prev_accel_constraint,\n"
        "      personality=personality,\n"
        "      jerk_factor=jerk_factor,\n"
        "      a_change_cost_starting=a_change_cost_starting,\n"
        "      a_change_cost_factor=response_request.a_change_cost_factor,\n"
        "      jerk_cost_factor=response_request.jerk_cost_factor,\n"
        "    )\n",
        "    self.lead_accel_response_active = response_request.active\n"
        "    self.lead_accel_response_level = response_request.level if response_request.active else 0\n"
        "\n"
        "    # Rate-limit the applied cost factors toward the requested target instead\n"
        "    # of snapping to it this frame. Same ramp time engaging and disengaging,\n"
        "    # so catching up to a departing lead is a smooth continuous accel rather\n"
        "    # than a hard surge, and settling back into gap-holding is a smooth\n"
        "    # release rather than a sudden cutoff.\n"
        "    max_step = self.dt / max(RESPONSE_FACTOR_RAMP_TIME_S, 1e-3)\n"
        "    self._response_a_change_factor += float(np.clip(\n"
        "      response_request.a_change_cost_factor - self._response_a_change_factor, -max_step, max_step))\n"
        "    self._response_jerk_factor += float(np.clip(\n"
        "      response_request.jerk_cost_factor - self._response_jerk_factor, -max_step, max_step))\n"
        "\n"
        "    self.set_weights(\n"
        "      prev_accel_constraint,\n"
        "      personality=personality,\n"
        "      jerk_factor=jerk_factor,\n"
        "      a_change_cost_starting=a_change_cost_starting,\n"
        "      a_change_cost_factor=self._response_a_change_factor,\n"
        "      jerk_cost_factor=self._response_jerk_factor,\n"
        "    )\n",
    ),
]


def main():
    with open(TARGET, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER in content:
        print("Already patched, nothing to do.")
        return 0

    for old, new in EDITS:
        if old not in content:
            print("ERROR: expected text not found, aborting without changes:")
            print(repr(old[:120]))
            return 1
        content = content.replace(old, new, 1)

    backup_path = TARGET + ".bak_before_launch_smoothing"
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
