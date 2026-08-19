"""
2-D Kalman filter for wheel-leg robot odometry.

Fuses IMU body-x acceleration (process input) with wheel-odometry velocity
(measurement). Deployable default: velocity is measured from wheel encoders,
not from ground-truth body velocity; noise is slip-adaptive.

Model (control theory manual 7.3):
  State:       x = [position, velocity]
  Process:     x_k = A x_{k-1} + B·body_acc_x
  Measurement: z = [v_odo]  (velocity-only)

Tuning comes from robot_params.yaml -> control_initial.kalman (q_pos, q_vel,
r_vel). r_vel / q_vel are overridden per step by the ground-contact detector's
slip-adaptive values (kalman_r_vel / kalman_q_vel).
"""

import numpy as np


class KalmanOdometry:
    def __init__(self, wheel_radius, dt, q_pos=0.0001, q_vel=0.01, r_vel=0.5):
        self.wheel_radius = wheel_radius
        self.dt = dt
        dt2 = dt * dt

        # State: [position, velocity]
        self.x = np.array([[0.0], [0.0]])
        self.P = np.eye(2)

        # Process model
        self.A = np.array([[1.0, dt],
                           [0.0, 1.0]])
        self.B = np.array([[dt2 / 2.0],
                           [dt]])

        # Measurement: velocity-only (wheel odometry)
        self.H = np.array([[0.0, 1.0]])  # 1x2

        # Noise parameters (single-sourced from config; overridable per step)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_vel = r_vel

        self._odo_x = 0.0
        self._initialized = False

    @property
    def position(self):
        return float(self.x[0, 0])

    @property
    def velocity(self):
        return float(self.x[1, 0])

    def reset(self):
        self.x = np.array([[0.0], [0.0]])
        self.P = np.eye(2)
        self._odo_x = 0.0
        self._initialized = False

    def update(self, left_wheel_vel, right_wheel_vel, imu_body_accel_x,
               body_vx=None, r_vel=None, q_vel=None):
        """Kalman step.

        Velocity measurement defaults to wheel odometry (deployable). Pass
        body_vx to override with ground-truth body velocity (sim validation
        only). r_vel / q_vel override the noise for slip-adaptive tuning.
        """
        v_odo = 0.5 * self.wheel_radius * (left_wheel_vel + right_wheel_vel)
        self._odo_x += v_odo * self.dt

        z_vx = body_vx if body_vx is not None else v_odo
        r_use = self.r_vel if r_vel is None else r_vel
        q_vel_use = self.q_vel if q_vel is None else q_vel
        q_use = np.diag([self.q_pos, q_vel_use])

        if not self._initialized:
            self.x[0, 0] = 0.0
            self.x[1, 0] = z_vx
            self._initialized = True
            return self.position, self.velocity

        # --- Predict ---
        u = np.array([[imu_body_accel_x]])
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + q_use

        # --- Update ---
        z = np.array([[z_vx]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + np.array([[r_use]])
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.position, self.velocity
