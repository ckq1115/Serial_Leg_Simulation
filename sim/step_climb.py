"""Step capture using motor feedback, IMU and chassis geometry."""

import numpy as np

from self_righting import RecoveryGeometry


class StepClimb:
    def __init__(self, cad, radius, mass, gravity, guides, torque_limit, dt, wheel_torque_limit=5.):
        self.geometry = RecoveryGeometry(cad, radius)
        self.radius = radius
        self.weight = mass * gravity
        self.gravity = gravity
        self.guides = guides
        self.torque_limit = torque_limit
        self.plant_drive_limit = min(1.2, .24*wheel_torque_limit)
        self.dt = dt
        self.reset()

    def reset(self):
        self.phase = 'idle'
        self.cooldown = 0.
        self.direction = 0.
        self.approach = self.impact = self.confirm = 0.
        self.support_impulse = 0.
        self.inertial_speed = None
        self.light_support = 0.
        self.unweight_time = 0.
        self.landing_window = 0.
        self.defer_recovery = False
        self.landed = 0.
        self.landing_shift = 0.
        self.unloaded_time = 0.
        self.quick_retract = False
        self.asymmetric_lift = False
        self.wheel_odometry_valid = False
        self.support = 1.
        self.wheel_load = np.full(2, .5)
        self.geometry.encoder_previous = [None, None]
        self.geometry.encoder_continuous = [None, None]

    def update(self, state, hip_torque, body_speed, armed):
        dt = self.dt
        self.defer_recovery = False
        self.cooldown = max(0., self.cooldown-dt)
        if self.phase == 'idle' and (not armed or self.cooldown > 0.):
            self.approach = self.impact = self.confirm = 0.
            self.support_impulse = 0.
            self.inertial_speed = None
            self.light_support = 0.
            self.unweight_time = self.landing_window = 0.
            self.support = 1.
            self.wheel_load[:] = .5
            return None

        R, legs = self.geometry.observe(state)
        accel = R @ state['accel']
        forward = self.geometry.last_forward
        ax = accel[:2] @ forward
        roll, pitch, _ = state['euler']
        positions = np.array([leg['p'] for leg in legs])
        velocity = np.array([leg['v'] for leg in legs])
        gravity_torque = [self.geometry.leg_gravity(i, leg['q'], R) for i, leg in enumerate(legs)]
        # 承载估计需要电机与弹簧共同作用在髋关节上的力矩。
        total_hip_torque = hip_torque + state.get("gas_hip_torque_nm", 0.0)
        wheel_load = []
        for i, leg in enumerate(legs):
            J = leg['J']
            torque = total_hip_torque[[i, i+2]] - gravity_torque[i]
            force = np.linalg.solve(J @ J.T + .02**2*np.eye(2), J @ torque)
            wheel_load.append(np.clip(-force[1]/self.weight, 0., 1.))
        # Bounded, filtered estimates. The support fraction is relative to mg.
        alpha = dt/(.012+dt)
        self.support += alpha*(np.clip(accel[2]/self.gravity, 0., 2.)-self.support)
        self.wheel_load += alpha*(np.asarray(wheel_load)-self.wheel_load)
        wheel_support = float(np.sum(self.wheel_load))
        residual = self.support - wheel_support

        # Guide axis is chassis Y. Project cylinder bounds along world Z.
        guide_bottom = (self.guides[:, :3] @ R[2] -
                        self.guides[:, 3]*np.hypot(R[2,0], R[2,2]) -
                        self.guides[:, 4]*abs(R[2,1]))
        deck = min(guide_bottom)
        if self.phase == 'idle':
            if self.inertial_speed is None:
                self.inertial_speed = body_speed
            self.inertial_speed += dt*ax
            self.inertial_speed += dt/(.20+dt)*(body_speed-self.inertial_speed)
            self.approach = max(0., self.approach-dt)
            if abs(body_speed) > .28 and self.support_impulse < .004:
                self.direction = np.sign(body_speed)
                self.approach = .18

            # 积分得到短窗内的纵向速度损失，削弱单帧冲击和普通制动的影响。
            self.impact = max(0., self.impact*(1.-dt/.075)
                              + dt*np.clip(-self.direction*ax-1.4, -7., 22.))
            clearance = deck - (positions[:, 1]-self.radius)
            eligible = (self.approach > 0. and abs(roll) < .30 and abs(pitch) < .50
                        and max(clearance) > .060 and min(clearance) > .035)
            self.light_support = max(0., self.light_support-dt)
            if eligible and self.support < .55:
                self.light_support = .12
            # 主轮滚过下层边缘时，机身可能先失重，再由导轮接住。
            self.landing_window = max(0., self.landing_window-dt)
            if eligible and self.support < .25:
                if self.unweight_time < .018 <= self.unweight_time+dt:
                    self.landing_window = .15
                self.unweight_time += dt
            else:
                self.unweight_time = 0.
            # 进入失重窗口前已确认长腿间隙，碰撞时轮腿回摆会暂时改变当前间隙。
            self.defer_recovery = (self.landing_window > 0. and self.approach > 0.
                                   and abs(roll) < .22 and abs(pitch) < .32)
            # IMU 估计总竖直外力，减去髋力矩反推的两轮承载。
            # 对剩余支撑力积分，只有车身确实分担载荷才继续。
            transferred = (eligible and self.support > .63 and wheel_support < .81
                           and residual > .25)
            self.support_impulse *= 1.-dt/.055
            # 失重后同时出现向上支撑和沿行进方向的制动冲量，且导轮仍接近主轮下缘。
            edge_landing = (self.defer_recovery and self.support > .85 and self.impact > .13
                            and max(clearance) > -.025 and min(clearance) > -.055)
            # 长腿接近极限时，轮端承载估计可能偏高。用失载后的速度失配或单侧卸载补充判据。
            uncertain_load = (eligible and self.light_support > 0. and self.support > .78
                              and self.impact > .10 and min(clearance) > .09
                              and (abs(body_speed-self.inertial_speed) > .32 or
                                   (min(self.wheel_load) < .20 and residual > .10)))
            if transferred:
                self.support_impulse += dt*min(residual-.18, 1.)
                self.confirm += dt
            elif edge_landing:
                self.support_impulse += dt*.65
                self.confirm += dt
            elif uncertain_load:
                self.support_impulse += dt*.40
                self.confirm += dt
            else:
                self.confirm = max(0., self.confirm-2.*dt)
            hit = self.impact > .075 and self.support_impulse > .0035 and self.confirm >= .008
            supported = self.support_impulse > .010 and self.confirm >= .016
            # 两轮已经拖在车后时，导轮接住后的制动再拖几拍就可能使机身回滑。
            quick_retract = (eligible and self.light_support > 0. and self.support > .85
                             and self.impact > .05 and self.confirm >= .002
                             and max(self.direction*positions[:, 0]) < -.13)
            if not (transferred or edge_landing or uncertain_load) or not (hit or supported or quick_retract):
                return None
            self.phase = 'lift'
            self.elapsed = 0.
            self.lift_x = positions[:, 0].copy()
            self.lift_travel = 0.
            self.quick_retract = quick_retract
            self.asymmetric_lift = uncertain_load and abs(self.lift_x[0]-self.lift_x[1]) > .12

        self.elapsed += dt
        if self.elapsed > (1.20 if self.landing_shift > 0. else .95) or abs(roll) > .65 or abs(pitch) > .85:
            return self.finish()
        if self.phase == 'capture':
            self.capture_time += dt
            self.wheel_odometry_valid = bool(min(self.wheel_load) > .12 and wheel_support > .38
                                             and self.support > .5)
            self.unloaded_time = 0. if self.wheel_odometry_valid else self.unloaded_time + dt
            if (self.capture_time >= .15 and self.wheel_odometry_valid and
                    body_speed*self.direction > .12 and abs(pitch) < .14):
                return self.finish()
            if self.unloaded_time < .016 and self.capture_time < .32:
                return None
            # 碰到台阶棱边后可能重新失去承载。回到落轮控制，并把落点向前移。
            self.phase = 'plant'
            self.landing_shift = .045
            self.landed = 0.
            self.wheel_odometry_valid = False

        # A slightly trailing landing point limits the nose-up impulse when
        # the main wheels first transfer load away from the guide rollers.
        goal_x = (self.landing_shift-.025)*self.direction
        bottom = positions[:,1] - self.radius
        trailing = self.impact > .23 and max(self.direction*self.lift_x) < -.13
        lift_clearance = .008 if trailing else .023
        if self.phase == 'lift' and min(bottom-deck) >= lift_clearance:
            self.phase = 'swing'
        horizontal_error = max(abs(positions[:,0]-goal_x))
        if self.phase == 'swing' and horizontal_error < .12:
            self.phase = 'plant'
        planted = (self.phase == 'plant' and min(self.wheel_load) > .17 and wheel_support > .47
                   and self.support > .65 and residual < .52 and
                   max(abs(velocity[:, 1])) < .45 and
                   max(abs(leg['theta']) for leg in legs) < .43 and abs(pitch) < .30)
        self.landed = self.landed+dt if planted else 0.
        self.wheel_odometry_valid = self.landed >= .012
        # Wait for the existing estimator to incorporate loaded-wheel feedback
        # before using its speed to hand control back to balance.
        if self.landed >= .020 and abs(state['gyro'][1]) < 2.:
            self.phase = 'capture'
            self.capture_time = 0.
            self.unloaded_time = 0.
            return None

        ctrl = np.zeros(6)
        lift_speed = .6*np.clip(body_speed*self.direction, 0., 1.6)*self.direction
        if self.phase == 'lift':
            self.lift_travel = np.clip(self.lift_travel+lift_speed*dt, -.16, .16)
        for i, leg in enumerate(legs):
            x = self.lift_x[i]-self.lift_travel if self.phase == 'lift' else goal_x
            if self.phase == 'lift' and self.asymmetric_lift:
                # 斜着搭上时两侧起点不同，收轮时提前对齐两侧的前后位置。
                x = .30*self.lift_x[i]
            clearance = .053
            if self.phase == 'swing':
                clearance = np.clip(.053 - max(0., .16-horizontal_error)*.65, .008, .053)
            if self.phase == 'plant':
                clearance = -.042
            z = deck + self.radius + clearance
            lift_rate = 3.3 if trailing else 2.8
            lower = [-2.2, -.45]
            upper = [2.2, lift_rate]
            if self.asymmetric_lift:
                lower, upper = [-3.2, -.8], [3.2, 3.3]
            rate = np.clip(18.*(np.array([x, z])-positions[i]), lower, upper)
            if self.phase == 'lift':
                rate[0] -= lift_speed
            vertical_gain = 210. if trailing and self.phase == 'lift' else 140.
            force = np.array([135., vertical_gain])*(rate-velocity[i])
            if self.quick_retract and self.phase in ('lift', 'swing') and pitch*self.direction < -.04:
                force[0] -= 30.*state['gyro'][1]
            ctrl[[i, i+2]] = leg['J'].T @ force + gravity_torque[i]
            if self.phase == 'plant' and self.support > .70 and bottom[i] < deck+.06:
                # 先接住的一侧给小幅牵引，帮助机身越过棱边。
                side = 'left' if i == 0 else 'right'
                speed = self.direction*self.radius*state[side+'_wheel_vel']
                load = np.clip((self.wheel_load[i]-.17)/.14, 0., 1.)
                ctrl[i+4] = self.direction*self.plant_drive_limit*load*np.clip((.8-speed)/.4, 0., 1.)
        ctrl[:4] -= state.get("gas_hip_torque_nm", 0.0)
        ctrl[:4] = np.clip(ctrl[:4], -self.torque_limit, self.torque_limit)
        return ctrl

    def finish(self):
        self.reset()
        self.cooldown = .45
        return None
