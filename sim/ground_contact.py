"""估计轮端法向载荷、离地、卡滞和打滑状态。"""
import numpy as np
from self_righting import rotation


class GroundContactDetector:
    def __init__(self, dt, joint_torque_limit=30.0, wheel_torque_limit=5.0,
                 wheel_radius=0.05, half_track=0.0,
                 dby_threshold=0.5, dvb_threshold=0.3, dvw_threshold=0.02,
                 gravity=9.81, static_load=115.0):
        self.dt = dt
        self.wheel_radius = wheel_radius
        self.half_track = half_track
        self.wheel_base = 2.0 * half_track  # L_wheel, left-right wheel spacing

        self.gravity = gravity
        self.static_load = static_load
        self.liftoff_Fn = .085 * static_load
        self.touchdown_Fn = .25 * static_load
        self.debounce_s   = 0.01   # [s]  consecutive steps required (10 ms)
        self.debounce_n   = max(1, int(self.debounce_s / dt))

        self.stuck_torque_ratio = 0.90
        self.stuck_wheel_vel    = 0.5   # rad/s, wheel nearly stopped
        self.stuck_duration_s   = 0.15  # sustained before flag
        self.stuck_n            = max(1, int(self.stuck_duration_s / dt))

        self.dby_threshold = dby_threshold  # [rad/s] condition A: yaw-rate deviation
        self.dvb_threshold = dvb_threshold  # [m/s]   condition B: wheel-body vel deviation
        self.dvw_threshold = dvw_threshold  # [m/s]   condition C: per-step vel-change deviation
        self.dv_wheel_alpha = 0.5           # mean-filter coefficient for Δv_wheel

        self.joint_torque_limit = joint_torque_limit
        self.wheel_torque_limit = wheel_torque_limit

        self._fn_raw    = [0.0, 0.0]
        self._fn_filt = [static_load, static_load]
        self._fn_alpha = dt/(.008+dt)
        self._previous_link_velocity = [None, None]
        self._link_acceleration = np.zeros((2, 7, 4))
        self._specific_force = gravity
        self._freefall_time = 0.0

        self._debounce_counter = [0, 0]    # consecutive steps in current transition
        self._stuck_counter    = [0, 0]

        self._prev_wheel_vx = [0.0, 0.0]   # previous wheel line speed (m/s)
        self._dv_wheel      = [0.0, 0.0]   # mean-filtered wheel speed change (m/s)
        self._f_dv_wheel    = [0.0, 0.0]   # |v_wheel| recorded at slip onset (m/s)

        self.contact_state  = ["grounded", "grounded"]
        self.slip_active    = [False, False]
        self.spinning       = False                    # condition A (whole body)
        self.stuck_active   = [False, False]

        self.R_turn = 0.0  # turn radius diagnostic (m)

        self.kalman_r_vel = 0.5
        self.kalman_q_vel = 0.01

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

    @property
    def any_slip(self):
        return self.spinning or self.slip_active[0] or self.slip_active[1]

    def reset(self):
        self._debounce_counter = [0, 0]
        self._stuck_counter    = [0, 0]
        self._prev_wheel_vx    = [0.0, 0.0]
        self._dv_wheel         = [0.0, 0.0]
        self._f_dv_wheel       = [0.0, 0.0]
        self.contact_state     = ["grounded", "grounded"]
        self.slip_active       = [False, False]
        self.spinning          = False
        self.stuck_active      = [False, False]
        self._fn_filt = [self.static_load, self.static_load]
        self._previous_link_velocity = [None, None]
        self._link_acceleration.fill(0.0)
        self._specific_force = self.gravity
        self._freefall_time = 0.0
        self.R_turn            = 0.0

    def estimate_normal_forces(self, state, hip_torque, wheel_torque, model, lateral_offsets):
        """用轮腿逆动力学的广义力残差估算轮端法向载荷。"""
        R = rotation(state['quat'])
        specific_force = R @ state['accel']
        gyro = state['gyro']
        forces = []
        for side, name in enumerate(('left', 'right')):
            q = np.array([state[name+'_front_pos'], state[name+'_back_pos']])
            qd = np.array([state[name+'_front_vel'], state[name+'_back_vel']])
            positions, _, jacobian, angular_jacobian = model.leg_kinematics(side, q)
            link_mass = model.mass[1+7*side:8+7*side]
            link_inertia = model.inertia[1+7*side:8+7*side]

            # 先求各连杆相对 IMU 的世界系速度，差分后再加 IMU 比力，重力自然计入。
            position_3d = np.column_stack((positions[:, 0], np.full(7, lateral_offsets[side]), positions[:, 1]))
            world_jacobian = jacobian @ R[:, [0, 2]].T
            relative_velocity = np.cross(gyro, position_3d) @ R.T
            relative_velocity += np.einsum('i,ijk->jk', qd, world_jacobian)
            angular_velocity = gyro[1] - qd @ angular_jacobian
            link_velocity = np.column_stack((relative_velocity, angular_velocity))
            previous = self._previous_link_velocity[side]
            acceleration = self._link_acceleration[side]
            if previous is not None:
                acceleration += self.dt/(.010+self.dt)*((link_velocity-previous)/self.dt-acceleration)
            self._previous_link_velocity[side] = link_velocity

            inertial_force = link_mass[:, None]*(specific_force+acceleration[:, :3])
            required_torque = np.einsum('ijk,jk->i', world_jacobian, inertial_force)
            required_torque -= angular_jacobian @ (link_inertia*acceleration[:, 3])
            # 轮电机的反作用力矩经小腿传到髋，需一并从动力学需求中扣除。
            motor_torque = hip_torque[[side, side+2]] + wheel_torque[side]*angular_jacobian[:, 5]
            wheel_jacobian = jacobian[:, -1, :].T
            horizontal = R[:2, 0]
            forward = horizontal/max(float(np.linalg.norm(horizontal)), .15)
            projected = np.vstack((forward @ R[:2, [0, 2]], R[2, [0, 2]])) @ wheel_jacobian
            force = np.linalg.solve(projected @ projected.T + .02**2*np.eye(2),
                                    projected @ (required_torque-motor_torque))
            forces.append(float(np.clip(force[1], 0., 5.*self.static_load)))
        return forces

    def update(self, fn_l_raw, fn_r_raw,
               left_wheel_vel, right_wheel_vel,
               body_vx, z_acc,
               left_joint_torques, right_joint_torques,
               left_wheel_torque, right_wheel_torque,
               yaw_rate=0.0, accel_x=0.0, specific_force_norm=None, wheel_contacts=None):
        """Advance detector state.  Call once per control step.

        yaw_rate : IMU measured yaw rate ω_imu = Gyro[2] [rad/s]
        accel_x  : IMU forward horizontal acceleration a_x [m/s^2]
        """
        if specific_force_norm is None:
            specific_force_norm = abs(z_acc)
        self._specific_force += self.dt/(.004+self.dt)*(specific_force_norm-self._specific_force)
        if self._specific_force < .20*self.gravity:
            self._freefall_time += self.dt
        else:
            self._freefall_time = 0.0
        freefall = wheel_contacts is None and self._freefall_time >= .006

        # 整机失重时，关节仍可输出很大力矩，但两轮都没有地面承载。
        for i, raw in enumerate((fn_l_raw, fn_r_raw)):
            if freefall:
                self.contact_state[i] = "airborne"
                self._debounce_counter[i] = 0
                raw = 0.0
            self._fn_raw[i] = raw
            self._fn_filt[i] = (self._fn_alpha * raw
                                + (1.0 - self._fn_alpha) * self._fn_filt[i])

        v_wheel = [left_wheel_vel * self.wheel_radius,
                   right_wheel_vel * self.wheel_radius]

        # 轮速差与 IMU 偏航角速度的偏差作为整车诊断量。
        if self.wheel_base > 1e-9:
            omega_est = (v_wheel[1] - v_wheel[0]) / self.wheel_base
        else:
            omega_est = 0.0
        self.spinning = abs(omega_est - yaw_rate) >= self.dby_threshold

        # 打滑判据使用前向加速度积分；转弯半径单独用于诊断。
        a_x = accel_x                      # forward specific force (IMU x = forward)
        dv_acc = abs(a_x) * self.dt        # ∫|a_x|dt ≈ |a_x|·Δt
        denom = 2.0 * (v_wheel[0] - v_wheel[1])
        self.R_turn = (self.wheel_base * (v_wheel[0] + v_wheel[1]) / denom
                       if abs(denom) > 1e-6 else 0.0)

        for side in (0, 1):
            fn = self._fn_filt[side]
            wv = (left_wheel_vel if side == 0 else right_wheel_vel)
            jt = (left_joint_torques if side == 0 else right_joint_torques)
            wt = (left_wheel_torque if side == 0 else right_wheel_torque)

            # 有仿真接触数据时直接使用；否则以法向载荷迟滞和连续采样去抖。
            if wheel_contacts is not None:
                self.contact_state[side] = "grounded" if wheel_contacts[side] else "airborne"
                self._debounce_counter[side] = 0
            elif self.contact_state[side] == "grounded":
                if fn < self.liftoff_Fn:
                    self._debounce_counter[side] += 1
                    if self._debounce_counter[side] >= self.debounce_n:
                        self.contact_state[side] = "airborne"
                        self._debounce_counter[side] = 0
                else:
                    self._debounce_counter[side] = 0
            else:
                if not freefall and self._specific_force > .32*self.gravity and fn > self.touchdown_Fn:
                    self._debounce_counter[side] += 1
                    if self._debounce_counter[side] >= self.debounce_n:
                        self.contact_state[side] = "grounded"
                        self._debounce_counter[side] = 0
                else:
                    self._debounce_counter[side] = 0

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

            # 轮速变化超过车身速度变化，同时轮速与车速失配时判为打滑。
            dv_wheel_raw = v_wheel[side] - self._prev_wheel_vx[side]
            self._dv_wheel[side] = (self.dv_wheel_alpha * dv_wheel_raw
                                    + (1.0 - self.dv_wheel_alpha) * self._dv_wheel[side])
            self._prev_wheel_vx[side] = v_wheel[side]

            cond_c = abs(self._dv_wheel[side]) - dv_acc >= self.dvw_threshold
            # 条件B: wheel-body velocity deviation
            cond_b = abs(v_wheel[side]) - abs(body_vx) >= self.dvb_threshold

            if not self.slip_active[side]:
                if cond_b and cond_c:
                    self.slip_active[side] = True
                    self._f_dv_wheel[side] = abs(v_wheel[side])
            else:
                # hysteresis exit: wheel speed fell back to onset value
                if abs(v_wheel[side]) <= self._f_dv_wheel[side]:
                    self.slip_active[side] = False

        self.kalman_r_vel = 0.5
        self.kalman_q_vel = 0.01
        if self.any_slip:
            self.kalman_r_vel = 50.0   # trust wheel odometry much less
            self.kalman_q_vel = 0.005  # trust model prediction more

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
