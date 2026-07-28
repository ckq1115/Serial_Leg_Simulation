import numpy as np
from utils import approach_value

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
        self.speed_accel = abs(args.keyboard_speed_accel if args.keyboard_speed_accel is not None else keyboard_cfg.get("speed_accel_mps2", 0.8))
        self.speed_release_accel = abs(args.keyboard_speed_release_accel if args.keyboard_speed_release_accel is not None else keyboard_cfg.get("speed_release_accel_mps2", self.speed_accel))
        yaw_rate_accel_arg = getattr(args, "keyboard_yaw_rate_accel", None)
        yaw_rate_release_accel_arg = getattr(args, "keyboard_yaw_rate_release_accel", None)
        self.yaw_rate_accel = abs(yaw_rate_accel_arg if yaw_rate_accel_arg is not None else keyboard_cfg.get("yaw_rate_accel_radps2", 1.5))
        self.yaw_rate_release_accel = abs(yaw_rate_release_accel_arg if yaw_rate_release_accel_arg is not None else keyboard_cfg.get("yaw_rate_release_accel_radps2", self.yaw_rate_accel))
        self.speed_axis_sign = float(keyboard_cfg.get("speed_axis_sign", 1.0))
        self.yaw_axis_sign = float(keyboard_cfg.get("yaw_axis_sign", 1.0))
        self.instant_stop_on_release = bool(keyboard_cfg.get("instant_stop_on_release", True))
        self.instant_yaw_rate_command = bool(keyboard_cfg.get("instant_yaw_rate_command", True))
        self.reset()

    def reset(self):
        self.x = self.initial_x
        self.yaw = self.initial_yaw
        self.x_dot = self.initial_x_dot
        self.yaw_dot = self.initial_yaw_dot
        self.x_ddot = 0.0
        self.yaw_ddot = 0.0
        self.last_speed_axis = 0.0
        self.last_yaw_axis = 0.0

    @staticmethod
    def ramp_rate_limit(current, target, accel, release_accel):
        if current == 0.0 or np.sign(current) == np.sign(target):
            slowing_down = abs(target) < abs(current)
        else:
            slowing_down = True
        return release_accel if slowing_down else accel

    def update(self, speed_axis, yaw_axis, current_yaw=None):
        speed_axis = float(np.clip(speed_axis * self.speed_axis_sign, -1.0, 1.0))
        yaw_axis = float(np.clip(yaw_axis * self.yaw_axis_sign, -1.0, 1.0))
        speed_command = speed_axis * self.max_speed
        yaw_rate_command = yaw_axis * self.max_yaw_rate
        prev_x_dot = self.x_dot
        prev_yaw_dot = self.yaw_dot
        yaw_active = abs(yaw_axis) > 1e-6
        yaw_released = abs(self.last_yaw_axis) > 1e-6 and not yaw_active

        if self.instant_stop_on_release and speed_axis == 0.0:
            self.x_dot = 0.0
        else:
            speed_limit = self.ramp_rate_limit(self.x_dot, speed_command, self.speed_accel, self.speed_release_accel)
            self.x_dot = approach_value(self.x_dot, speed_command, speed_limit * self.dt)

        if self.instant_yaw_rate_command:
            self.yaw_dot = yaw_rate_command
        elif self.instant_stop_on_release and yaw_axis == 0.0:
            self.yaw_dot = 0.0
        else:
            yaw_limit = self.ramp_rate_limit(self.yaw_dot, yaw_rate_command, self.yaw_rate_accel, self.yaw_rate_release_accel)
            self.yaw_dot = approach_value(self.yaw_dot, yaw_rate_command, yaw_limit * self.dt)

        if yaw_released and current_yaw is not None:
            # 松开转向键时，锁定当前朝向而不是继续追踪之前积分出来的偏航目标。
            self.yaw = float(current_yaw)

        self.x_ddot = float(np.clip(
            (self.x_dot - prev_x_dot) / self.dt,
            -self.speed_release_accel,
            self.speed_accel,
        ))
        if self.instant_yaw_rate_command:
            self.yaw_ddot = 0.0
        else:
            self.yaw_ddot = float(np.clip(
                (self.yaw_dot - prev_yaw_dot) / self.dt,
                -self.yaw_rate_release_accel,
                self.yaw_rate_accel,
            ))

        self.x += self.x_dot * self.dt
        self.yaw += self.yaw_dot * self.dt
        self.last_speed_axis = speed_axis
        self.last_yaw_axis = yaw_axis
        return self.x, self.yaw, self.x_dot, self.yaw_dot
