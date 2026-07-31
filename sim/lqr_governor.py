"""Simplified reference governor — direct keyboard-to-target mapping (no ramps)."""

import numpy as np


class LqrReferenceGovernor:
    def __init__(self, dt, control_cfg, args):
        self.dt = dt
        target_cfg = control_cfg.get("lqr_target", {})
        self.initial_x = args.target_x if args.target_x is not None else target_cfg.get("x_m", 0.0)
        self.initial_yaw = args.target_yaw if args.target_yaw is not None else target_cfg.get("yaw_rad", 0.0)
        self.initial_x_dot = args.target_x_dot if args.target_x_dot is not None else target_cfg.get("x_dot_mps", 0.0)
        self.initial_yaw_dot = args.target_yaw_dot if args.target_yaw_dot is not None else target_cfg.get("yaw_dot_radps", 0.0)

        keyboard_cfg = control_cfg.get("keyboard_command", {})
        self.max_speed = abs(args.max_keyboard_speed if args.max_keyboard_speed is not None else keyboard_cfg.get("max_speed_mps", 0.6))
        self.max_yaw_rate = abs(args.max_keyboard_yaw_rate if args.max_keyboard_yaw_rate is not None else keyboard_cfg.get("max_yaw_rate_radps", 1.2))
        self.speed_axis_sign = float(keyboard_cfg.get("speed_axis_sign", 1.0))
        self.yaw_axis_sign = float(keyboard_cfg.get("yaw_axis_sign", 1.0))
        self.instant_yaw_rate_command = bool(keyboard_cfg.get("instant_yaw_rate_command", True))
        self.reset()

    def reset(self):
        self.x = self.initial_x
        self.yaw = self.initial_yaw
        self.x_dot = self.initial_x_dot
        self.yaw_dot = self.initial_yaw_dot

    def update(self, speed_axis, yaw_axis, current_yaw=None, current_x=None):
        speed_axis = float(np.clip(speed_axis * self.speed_axis_sign, -1.0, 1.0))
        yaw_axis = float(np.clip(yaw_axis * self.yaw_axis_sign, -1.0, 1.0))

        prev_x_dot = self.x_dot
        # Instant velocity (no ramps — matched to legwheel)
        self.x_dot = speed_axis * self.max_speed

        if self.instant_yaw_rate_command:
            self.yaw_dot = yaw_axis * self.max_yaw_rate
        else:
            self.yaw_dot = yaw_axis * self.max_yaw_rate

        # Snap position target on throttle release to prevent creep-back
        if abs(self.x_dot) < 1e-6 and abs(prev_x_dot) > 1e-6 and current_x is not None:
            self.x = current_x

        # Always integrate position (matched to legwheel)
        self.x += self.x_dot * self.dt
        # Clamp position target to 0.05 m from actual (matched to legwheel)
        if current_x is not None and abs(self.x_dot) > 1e-6:
            if (self.x - current_x) > 0.05:
                self.x = current_x + 0.05
            if (self.x - current_x) < -0.05:
                self.x = current_x - 0.05

        self.yaw += self.yaw_dot * self.dt
        return self.x, self.yaw, self.x_dot, self.yaw_dot
