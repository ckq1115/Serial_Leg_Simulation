"""
2-D Kalman filter for wheel-leg robot odometry.

Fuses IMU body-x acceleration (process input) with wheel odometry (measurement).
Tuned so odometry position is strongly trusted, IMU smoothes the velocity.

Model (control theory manual 7.3):
  State:       x = [position, velocity]
  Process:     x_k = A x_{k-1} + B·body_acc_x
  Measurement: z = [odo_x, odo_v]

Key tuning: R_pos is SMALL (trust odometry position). IMU body-x acceleration
provides high-frequency velocity smoothing without overriding the odometry trend.
"""

import numpy as np


class KalmanOdometry:
    def __init__(self, wheel_radius, dt):
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

        # Measurement: z = [odo_x] only (position-only observation)
        # Wheel velocity is too noisy at 500 Hz; position trend is reliable.
        self.H = np.array([[1.0, 0.0]])  # 1x2

        # Process noise: moderate
        self.Q = np.diag([0.0001, 0.01])

        # Measurement noise: trust odometry position
        self.R = np.array([[0.01]])  # 1x1

        self.I2 = np.eye(2)
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
               body_vx=None):
        """Kalman step.

        Uses body_vx (from MuJoCo base_linvel) as velocity measurement.
        Position is integrated from velocity by the Kalman —
        wheel odometry position is NOT used because the initial settling
        transient creates an unrecoverable bias.
        """
        v_odo = 0.5 * self.wheel_radius * (left_wheel_vel + right_wheel_vel)
        self._odo_x += v_odo * self.dt

        if body_vx is not None:
            # Velocity-only measurement (position from integration)
            z = np.array([[body_vx]])
            H_use = np.array([[0.0, 1.0]])  # observe only velocity
            R_use = np.array([[0.01]])
        else:
            # Fallback: position-only from odometry
            z = np.array([[self._odo_x]])
            H_use = self.H
            R_use = self.R

        if not self._initialized:
            self.x[0, 0] = 0.0
            self.x[1, 0] = body_vx if body_vx is not None else v_odo
            self._initialized = True
            return self.position, self.velocity

        # --- Predict ---
        u = np.array([[imu_body_accel_x]])
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

        # --- Update ---
        y = z - H_use @ self.x
        S = H_use @ self.P @ H_use.T + R_use
        K = self.P @ H_use.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I2 - K @ H_use) @ self.P

        return self.position, self.velocity
