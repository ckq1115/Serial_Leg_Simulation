"""Ground contact detection per 轮腿机器人控制理论手册 §9.

Provides per-wheel contact state using only onboard sensors:
  - Fn from VMC force + IMU + leg kinematics (no extra hardware)
  - liftoff / touchdown with hysteresis + debounce
  - stuck detection  (torque saturation + zero wheel speed)
  - slip detection   (wheel vel vs body vel mismatch)

All thresholds are physics-based — no MuJoCo internals.
"""
import numpy as np


class GroundContactDetector:
    def __init__(self, dt, joint_torque_limit=30.0, wheel_torque_limit=5.0):
        self.dt = dt

        # ── thresholds ────────────────────────────────────────────
        self.liftoff_Fn   = 8.0    # [N]  below this → airborne
        self.touchdown_Fn = 30.0   # [N]  above this → re-contact (hysteresis)
        self.debounce_s   = 0.01   # [s]  consecutive steps required (10 ms)
        self.debounce_n   = max(1, int(self.debounce_s / dt))

        self.stuck_torque_ratio = 0.90  # joint torque > 90% of limit
        self.stuck_wheel_vel    = 0.5   # rad/s, wheel nearly stopped
        self.stuck_duration_s   = 0.15  # sustained before flag
        self.stuck_n            = max(1, int(self.stuck_duration_s / dt))

        self.slip_dv_threshold  = 0.3   # m/s wheel-body velocity mismatch
        self.slip_debounce_s    = 0.02
        self.slip_n             = max(1, int(self.slip_debounce_s / dt))

        self.joint_torque_limit = joint_torque_limit
        self.wheel_torque_limit = wheel_torque_limit

        # ── per-wheel state ───────────────────────────────────────
        self._fn_raw    = [0.0, 0.0]       # latest raw Fn estimate
        self._fn_filt   = [115.0, 115.0]   # low-pass filtered (EMA)
        self._fn_alpha  = 0.3              # EMA coefficient

        self._debounce_counter = [0, 0]    # consecutive steps in current transition
        self._stuck_counter    = [0, 0]
        self._slip_counter     = [0, 0]

        self.contact_state  = ["grounded", "grounded"]  # per wheel
        self.slip_active    = [False, False]
        self.stuck_active   = [False, False]

    # ── public API ─────────────────────────────────────────────────
    @property
    def fn_left(self):
        return self._fn_filt[0]

    @property
    def fn_right(self):
        return self._fn_filt[1]

    @property
    def left_airborne(self):
        return self.contact_state[0] == "airborne"

    @property
    def right_airborne(self):
        return self.contact_state[1] == "airborne"

    @property
    def any_airborne(self):
        return self.left_airborne or self.right_airborne

    def reset(self):
        self._debounce_counter = [0, 0]
        self._stuck_counter    = [0, 0]
        self._slip_counter     = [0, 0]
        self.contact_state     = ["grounded", "grounded"]
        self.slip_active       = [False, False]
        self.stuck_active      = [False, False]
        self._fn_filt          = [115.0, 115.0]

    # ── main update ────────────────────────────────────────────────
    def update(self, fn_l_raw, fn_r_raw,
               left_wheel_vel, right_wheel_vel,
               body_vx, z_acc,
               left_joint_torques, right_joint_torques,
               left_wheel_torque, right_wheel_torque):
        """Advance detector state.  Call once per control step."""
        # ── 1. low-pass filter Fn ──
        for i, raw in enumerate((fn_l_raw, fn_r_raw)):
            self._fn_raw[i] = raw
            self._fn_filt[i] = (self._fn_alpha * raw
                                + (1.0 - self._fn_alpha) * self._fn_filt[i])

        for side in (0, 1):
            fn = self._fn_filt[side]
            wv = (left_wheel_vel if side == 0 else right_wheel_vel)
            jt = (left_joint_torques if side == 0 else right_joint_torques)
            wt = (left_wheel_torque if side == 0 else right_wheel_torque)
            wr = 0.05  # wheel radius

            # ── 2. liftoff / touchdown (manual §9: Fn threshold + hysteresis + debounce) ──
            if self.contact_state[side] == "grounded":
                if fn < self.liftoff_Fn:
                    self._debounce_counter[side] += 1
                    if self._debounce_counter[side] >= self.debounce_n:
                        self.contact_state[side] = "airborne"
                        self._debounce_counter[side] = 0
                else:
                    self._debounce_counter[side] = 0
            else:  # airborne
                if fn > self.touchdown_Fn:
                    self._debounce_counter[side] += 1
                    if self._debounce_counter[side] >= self.debounce_n:
                        self.contact_state[side] = "grounded"
                        self._debounce_counter[side] = 0
                else:
                    self._debounce_counter[side] = 0

            # ── 3. stuck detection ──
            jt_max = max(abs(jt[0]), abs(jt[1]))
            jt_saturated = jt_max > self.stuck_torque_ratio * self.joint_torque_limit
            wheel_stopped = abs(wv) < self.stuck_wheel_vel
            fn_high = fn > 300.0  # abnormally high ground reaction

            if jt_saturated and wheel_stopped and fn_high:
                self._stuck_counter[side] += 1
                if self._stuck_counter[side] >= self.stuck_n:
                    self.stuck_active[side] = True
            else:
                self._stuck_counter[side] = max(0, self._stuck_counter[side] - 1)
                if self._stuck_counter[side] == 0:
                    self.stuck_active[side] = False

            # ── 4. slip detection ──
            wheel_vx = wv * wr
            slip_magnitude = abs(wheel_vx - body_vx)
            if slip_magnitude > self.slip_dv_threshold:
                self._slip_counter[side] += 1
                if self._slip_counter[side] >= self.slip_n:
                    self.slip_active[side] = True
            else:
                self._slip_counter[side] = max(0, self._slip_counter[side] - 1)
                if self._slip_counter[side] == 0:
                    self.slip_active[side] = False

        # ── update Kalman noise based on slip ──
        self.kalman_r_vel = 0.5  # nominal
        self.kalman_q_vel = 0.01
        if self.slip_active[0] or self.slip_active[1]:
            self.kalman_r_vel = 50.0   # trust wheel odometry much less
            self.kalman_q_vel = 0.005  # trust model prediction more

    # ── LQR K-suppression mask ─────────────────────────────────────
    def get_K_suppression_mask(self):
        """Return (zero_wheel_L, zero_wheel_R, zero_leg_L, zero_leg_R).

        Airborne → only suppress wheel torque (leg torque preserved for attitude).
        Stuck    → suppress both wheel and leg torque (prevent overheating).
        """
        zero_wheel_L = (self.contact_state[0] != "grounded") or self.stuck_active[0]
        zero_wheel_R = (self.contact_state[1] != "grounded") or self.stuck_active[1]
        zero_leg_L = self.stuck_active[0]
        zero_leg_R = self.stuck_active[1]
        return zero_wheel_L, zero_wheel_R, zero_leg_L, zero_leg_R
