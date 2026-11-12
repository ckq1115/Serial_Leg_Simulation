"""2-D Kalman filter for wheel-leg robot odometry (control manual §7.3).

State:       x = [velocity, acceleration]
Process:     x_k = F x_{k-1}          (constant-acceleration model)
Measurement: z = [v_est, a_imu]       (wheel-odom velocity + IMU horizontal accel)

Both states are observed (H = identity). During slip the velocity measurement
(wheel odometry) is de-weighted (r_v up) while the IMU acceleration measurement
stays trusted — this is what keeps the velocity estimate anchored to the real
body motion when the wheels spin.

Tuning from robot_params.yaml -> control_initial.kalman (q_v, q_a, r_v, r_a).
r_v / q_v are overridden per step by the ground-contact detector's
slip-adaptive values.
"""

import numpy as np


class KalmanOdometry:
    def __init__(self, dt, q_v=0.01, q_a=1.0, r_v=0.5, r_a=1.0):
        self.dt = dt

        # State: [velocity, acceleration]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)

        # Constant-acceleration process model
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.H = np.eye(2)  # both velocity and acceleration observed

        self.q_v = q_v
        self.q_a = q_a
        self.r_v = r_v
        self.r_a = r_a

        self._pos = 0.0
        self._initialized = False

    @property
    def position(self):
        return self._pos

    @property
    def velocity(self):
        return float(self.x[0, 0])

    @property
    def acceleration(self):
        return float(self.x[1, 0])

    def reset(self):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self._pos = 0.0
        self._initialized = False

    def update(self, v_est, a_imu, r_v=None, q_v=None):
        """Kalman step.

        v_est : wheel-odometry velocity (wheel + leg-swing corrected) [m/s]
        a_imu : IMU horizontal acceleration [m/s^2]
        r_v / q_v : optional per-step overrides (slip-adaptive).
        """
        q_v = self.q_v if q_v is None else q_v
        r_v = self.r_v if r_v is None else r_v
        Q = np.diag([q_v, self.q_a])
        R = np.diag([r_v, self.r_a])

        if not self._initialized:
            self.x[0, 0] = v_est
            self.x[1, 0] = a_imu
            self._initialized = True
            return self.position, self.velocity, self.acceleration

        # --- Predict ---
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        # --- Update ---
        z = np.array([[v_est], [a_imu]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        self._pos += self.velocity * self.dt
        return self.position, self.velocity, self.acceleration


class ChassisOdometry:
    """与 STM32 一致的三状态里程计，状态为位移、速度和纵向加速度零偏。"""

    def __init__(self, dt, half_track):
        self.dt = float(np.clip(dt, 1.e-4, .05))
        self.half_track = half_track
        dt = self.dt
        self.transition = np.array([[1., dt, -.5*dt**2], [0., 1., -dt], [0., 0., 1.]])
        accel_input = np.array([.5*dt**2, dt, 0.])
        self.process_noise = .32**2*np.outer(accel_input, accel_input)
        self.process_noise[2, 2] = .05**2*dt
        self.reset()

    def reset(self):
        self.state = np.zeros(3)
        self.covariance = np.diag([1., 1., .5])
        self.initialized = False

    @property
    def position(self):
        return float(self.state[0])

    @property
    def velocity(self):
        return float(self.state[1])

    def update(self, wheel_speeds, accel, yaw_rate, contacts):
        yaw_speed = yaw_rate*self.half_track
        observations = np.asarray(wheel_speeds) + np.array([yaw_speed, -yaw_speed])
        usable = observations[np.asarray(contacts, dtype=bool)]
        if not self.initialized:
            if usable.size:
                self.state[1] = usable.mean()
            self.initialized = True
        self.state = self.transition @ self.state + accel*np.array([.5*self.dt**2, self.dt, 0.])
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        np.fill_diagonal(self.covariance, np.maximum(np.diag(self.covariance), 1.e-6))

        # 用预测状态计算本拍打滑权重，再依次用当前状态和协方差修正。
        slip = np.clip((np.abs(usable-self.state[1])-.35)/(.35+1.e-6), 0., 1.)
        for observation, variance in zip(usable, .071**2*(1.+20.*slip)):
            velocity_row = self.covariance[1].copy()
            innovation = velocity_row[1] + variance
            if innovation <= 1.e-6:
                continue
            gain = self.covariance[:, 1]/innovation
            self.state += gain*(observation-self.state[1])
            self.covariance -= np.outer(gain, velocity_row)
            self.covariance = .5*(self.covariance+self.covariance.T)
            np.fill_diagonal(self.covariance, np.maximum(np.diag(self.covariance), 1.e-6))
        return self.position, self.velocity
