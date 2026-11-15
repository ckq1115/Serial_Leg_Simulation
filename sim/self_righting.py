"""Six-bar self-righting using IMU and motor encoders only.

The CAD model and support hull are fixed calibration data. This module has no
MuJoCo import or runtime plant-state/contact/force input. Torques are output-joint Nm.
"""
from pathlib import Path
import json
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def smooth(u):
    u = np.clip(u, 0., 1.)
    return 10*u**3 - 15*u**4 + 6*u**5


class RecoveryGeometry:
    def __init__(self, cad, radius=.06):
        self.cad = cad
        self.radius = radius
        calibration = json.loads(Path(__file__).with_name('recovery_geometry.json').read_text(encoding='utf8'))
        self.support = np.array(calibration['base_support_points_m'])
        self.hips = np.array(calibration['hip_positions_m'])
        self.dy = np.array(calibration['wheel_lateral_offsets_m'])
        self.encoder_previous = [None, None]
        self.encoder_continuous = [None, None]

    def jacobian(self, side, q):
        leg = self.cad.legs[side]
        eps = 1e-6
        return np.column_stack([(leg.wheel_position(*(q + np.eye(2)[i]*eps)) -
                                 leg.wheel_position(*(q - np.eye(2)[i]*eps))) / (2*eps) for i in range(2)])

    def inverse(self, side, point, seed):
        """Cartesian residual has no +/-pi discontinuity; continue the CAD branch."""
        q = np.array(seed, dtype=float).copy()
        leg = self.cad.legs[side]
        for _ in range(12):
            error = np.asarray(point) - leg.wheel_position(*q)
            if np.linalg.norm(error) < 1e-7:
                break
            J = self.jacobian(side, q)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-8*np.eye(2), error)
            q += np.clip(dq, -.15, .15)
        return q

    def leg_gravity(self, side, q, R):
        mass = self.cad.mass[1+7*side:8+7*side]
        gradient = []
        for i in range(2):
            delta = np.eye(2)[i]*1e-5
            a = self.cad.pose(side, *(q+delta))[0]
            b = self.cad.pose(side, *(q-delta))[0]
            gradient.append(9.81*np.sum(mass*((a-b)/(2e-5) @ R[2, [0, 2]])))
        return np.array(gradient)

    def observe(self, sample, dynamics=True):
        R = rotation(sample['quat'])
        forward = R[:2, 0]
        if np.linalg.norm(forward) < .15:
            forward = getattr(self, 'last_forward', np.array([1., 0.]))
        else:
            forward = forward / np.linalg.norm(forward)
        forward = getattr(self, 'fixed_forward', forward)
        self.last_forward = forward.copy()
        ex = np.r_[forward, 0.]
        A = np.array([[ex @ R[:, 0], ex @ R[:, 2]], [R[2, 0], R[2, 2]]])
        rest = -float(np.min(self.support @ R[2])) if dynamics else 0.
        result = []
        for i, name in enumerate(('left', 'right')):
            q = np.array([sample[name+'_front_pos'], sample[name+'_back_pos']])
            if self.encoder_previous[i] is None:
                self.encoder_continuous[i] = q.copy()
            else:
                self.encoder_continuous[i] += wrap(q-self.encoder_previous[i])
            self.encoder_previous[i] = q.copy()
            q = self.encoder_continuous[i].copy()
            qd = np.array([sample[name+'_front_vel'], sample[name+'_back_vel']])
            body_p = self.cad.legs[i].wheel_position(*q)
            offset = np.array([ex @ R[:, 1], R[2, 1]]) * self.dy[i]
            p = A @ body_p + offset
            theta = float(np.arctan2(-p[0], -p[1]))
            if not dynamics:
                result.append(dict(theta=theta))
                continue
            Jb = self.jacobian(i, q)
            local = np.array([body_p[0], self.dy[i], body_p[1]])
            v3 = R @ (np.cross(sample['gyro'], local) + np.array([(Jb@qd)[0], 0., (Jb@qd)[1]]))
            result.append(dict(q=q, qd=qd, p=p, J=A@Jb, Jb=Jb, body_p=body_p,
                               A=A, offset=offset, v=np.array([ex@v3, v3[2]]),
                               theta=theta, L=float(np.linalg.norm(body_p)),
                               rest_h=rest + R[2]@self.hips[i], height=self.radius-p[1]))
        return R, result


class SelfRighting:
    def __init__(self, cad, config, hip_limit=40., wheel_limit=5., radius=.06):
        self.geometry = RecoveryGeometry(cad, radius)
        self.cfg = dict(config)
        self.enabled = bool(config.get('enabled', True))
        self.hip_limit, self.wheel_limit, self.radius = hip_limit, wheel_limit, radius
        self.min_length = float(config.get('min_length_m', .123))
        self.swing_length = float(config.get('swing_length_m', .375))
        self.roll_speed = float(config.get('roll_speed_mps', .60))
        self.swing_speed = float(config.get('swing_speed_radps', 4.4/1.5))
        self.capture_duration = float(config.get('capture_blend_s', .04))
        self.capture_pose_duration = 2.*self.capture_duration
        self.capture_length = float(config.get('capture_length_m', .15))
        self.grow_duration = float(config.get('grow_duration_s', .20))
        self.pose_speed_limit = float(config.get('pose_speed_limit_radps', 8.))
        self.preload = float(config.get('ground_preload_n', 3.))
        self.trace_enabled = False
        self.reset()

    def reset(self):
        self.phase = 'watch'
        self.support_action = None
        self.elapsed = self.total = self.clock = self.quiet = 0.
        self.active = False
        self.routes = ['none', 'none']
        self.leg_done = [False, False]
        self.events = []
        self.blend = 0.
        self.handoff = False
        self.capture_pose_blend = 0.
        self.last_angles = None
        self.angle_continuous = np.zeros(2)
        self.ref_q = [None, None]
        self.block_reason = ''
        self.cooldown = 0.
        self.retries = 0
        self.support_attempted = False
        self.rear_flip_roll = False
        self.rear_flip_forward = None
        self.trace = None
        self.previous_wheel_vz = None
        if hasattr(self.geometry, 'fixed_forward'):
            del self.geometry.fixed_forward
        self.geometry.encoder_previous = [None, None]
        self.geometry.encoder_continuous = [None, None]

    def enter(self, phase, support_action=None):
        self.phase, self.elapsed = phase, 0.
        self.support_action = support_action
        self.best_error = np.inf
        self.no_progress = 0.
        if phase != 'watch':
            self.blend = 0.
        self.events.append(dict(time_s=self.clock, phase=phase,
                                routes=self.routes.copy(), reason=self.block_reason, retries=self.retries,
                                support_action=self.support_action))

    def stalled(self, error, improvement, dt, window=.35):
        if error < self.best_error-improvement:
            self.best_error = error
            self.no_progress = 0.
        else:
            self.no_progress += dt
        if self.trace is not None:
            self.trace['watchdog'] = dict(error=error, best_error=self.best_error,
                                          no_progress_s=self.no_progress, window_s=window)
        return self.no_progress >= window

    def restart(self):
        if hasattr(self.geometry, 'fixed_forward'):
            del self.geometry.fixed_forward
        self.last_angles = None
        self.routes = ['none', 'none']
        self.enter('settle')

    def retry(self, reason, R, legs):
        self.block_reason = reason
        if self.retries < 2:
            self.retries += 1
            if self.phase == 'establish_support' and self.support_action == 'ground' and reason != 'wheel_left_roll_path':
                # Keep each short leg on its current side and replan from its measured position.
                self.begin_roll(legs)
            else:
                self.restart()
        else:
            self.blocked_up = R[2].copy()
            self.blocked_q = np.array([leg['q'] for leg in legs])
            self.enter('blocked')

    def start_balance(self):
        self.handoff = True
        self.block_reason = ''
        self.enter('rise')
        self.capture_pose_blend = 0. if self.capture_duration > 0. else 1.
        if self.capture_duration <= 0.:
            self.blend = 1.

    def reachable_point(self, leg, target):
        point = np.linalg.lstsq(leg['A'], target-leg['offset'], rcond=.05)[0]
        length = np.linalg.norm(point)
        # 用机身平面上的长度限幅，避免姿态投影把目标推到行程外。
        if length > self.swing_length:
            point *= self.swing_length / length
        elif 0. < length < self.min_length:
            point *= self.min_length / length
        return point

    def gravity_torque(self, i, leg, R):
        if 'gravity_tau' not in leg:
            leg['gravity_tau'] = self.geometry.leg_gravity(i, leg['q'], R)
        return leg['gravity_tau']

    def estimate_normal_forces(self, sample, motor_ctrl, dt):
        # Quasi-static virtual work from the issued hip torques, plus wheel inertia.
        # The geometry check keeps a leg held above the chassis support plane from
        # being declared loaded solely because its position controller is pushing.
        R, legs = self.contact_observation
        wheel_vz = np.array([leg['v'][1] for leg in legs])
        rel_accel = np.zeros(2) if self.previous_wheel_vz is None else (wheel_vz-self.previous_wheel_vz)/dt
        self.previous_wheel_vz = wheel_vz
        body_accel = R[2]@sample['accel']
        # 从已发出的电机力矩反推地面载荷时，加回被动弹簧力矩。
        total_hip_torque = motor_ctrl[:4] + sample.get("gas_hip_torque_nm", 0.0)
        normal = []
        for i, leg in enumerate(legs):
            torque = total_hip_torque[[i, i+2]]-self.gravity_torque(i, leg, R)
            J = leg['J']
            force = np.linalg.solve(J@J.T+.015**2*np.eye(2), J@torque)
            inertia = .353*np.clip(body_accel+rel_accel[i], -60., 60.)
            clearance = leg['height']-leg['rest_h']
            confidence = np.clip((clearance+.05)/.03, 0., 1.)
            normal.append(float(np.clip(-force[1]+inertia, 0., 350.)*confidence))
        return normal

    def pose_torque(self, i, leg, target, R, dt, kp=90., kd=3.):
        # First initialize from the encoders. Later inverse solves continue from
        # the previous reference, not from a wrapped or fixed nominal angle.
        seed = leg['q'] if self.ref_q[i] is None else self.ref_q[i]
        body_point = self.reachable_point(leg, target)
        reference = self.geometry.inverse(i, body_point, seed)
        qrate = np.zeros(2) if self.ref_q[i] is None else np.clip((reference-self.ref_q[i])/dt, -self.pose_speed_limit, self.pose_speed_limit)
        self.ref_q[i] = reference
        error = np.clip(reference-leg['q'], -.55, .55)
        if self.trace is not None:
            self.trace['targets'][i] = dict(wheel_xz_m=(leg['A']@body_point+leg['offset']).tolist(),
                                             requested_wheel_xz_m=target.tolist(), hip_q_rad=reference.tolist(),
                                             hip_qdot_radps=qrate.tolist())
        return kp*error + kd*(qrate-leg['qd']) + self.gravity_torque(i, leg, R)

    def prepare_routes(self, legs, up_z):
        self.route_length = max(self.min_length, self.swing_length - .018 * self.retries)
        self.starts = [leg['p'].copy() for leg in legs]
        self.start_angles = self.angle_continuous.copy()
        self.ground_z = [self.radius-leg['rest_h'] for leg in legs]
        self.parking = []
        self.swing_targets = []
        self.swing_duration = []
        for i, leg in enumerate(legs):
            z_min = -np.sqrt(self.route_length**2 - .01**2)
            z = float(np.clip(self.ground_z[i], max(-.30, z_min), -.02))
            x = -np.sqrt(max(.01**2, self.route_length**2-z*z))
            self.parking.append(np.array([x, z]))
            angle = np.arctan2(-x, -z)
            direct = leg['p'][1] < -.035 and abs(leg['theta']) < .85 and leg['L'] < .27
            rear_near_floor = (up_z > .90 and leg['p'][0] < -.10 and leg['p'][1] < -.035
                               and leg['L'] < .35 and abs(leg['height']-leg['rest_h']) < .025)
            if direct or rear_near_floor:
                self.routes[i] = 'ground_short'
                target = self.start_angles[i]
            elif leg['p'][0] < -.10 and leg['p'][1] < .04:
                self.routes[i] = 'rear_lower'
                target = self.start_angles[i] + wrap(angle-leg['theta'])
            else:
                self.routes[i] = 'upper_arc'
                # Decreasing theta crosses the upper half-plane at -pi.
                target = angle
                while target >= self.start_angles[i] - .05:
                    target -= 2*np.pi
            self.swing_targets.append(target)
            self.swing_duration.append(max(.375, abs(target-self.start_angles[i])/self.swing_speed))
        self.grow_times = [0. if abs(leg['L']-self.route_length)<.005 else self.grow_duration for leg in legs]
        # 后下方有直达路径时，伸腿与摆轮共用时间段。上方绕行仍先留出车身间隙。
        self.sweep_delay = self.grow_times.copy()
        for i, route in enumerate(self.routes):
            if route == 'rear_lower':
                self.sweep_delay[i] = 0.
                self.swing_duration[i] = max(self.grow_times[i], self.swing_duration[i])
        self.leg_done = [route == 'ground_short' or
                         (route == 'rear_lower' and np.linalg.norm(legs[i]['p']-self.parking[i])<.025
                          and np.linalg.norm(legs[i]['v'])<.4)
                         for i, route in enumerate(self.routes)]
        self.ref_q = [leg['q'].copy() for leg in legs]
        self.enter('establish_support', 'sweep')

    def begin_roll(self, legs, rear_flip=False):
        self.rear_flip_roll = rear_flip
        self.support_starts = None
        self.starts = [leg['p'].copy() for leg in legs]
        self.roll_targets = []
        for leg in legs:
            vertical = max(.015, leg['rest_h']-self.radius)
            offset = np.sqrt(max(0., self.min_length**2-vertical**2))
            direct = self.routes[len(self.roll_targets)] == 'ground_short' or self.retries > 0
            side = 1. if direct and leg['p'][0] > 0 else -1.
            self.roll_targets.append(side*max(.015, offset))
        self.roll_duration = max(.25, max(abs(self.roll_targets[i]-legs[i]['p'][0]) for i in range(2))/self.roll_speed)
        self.roll_start_velocity = np.zeros(2)
        if rear_flip:
            travel = np.asarray(self.roll_targets) - np.asarray(self.starts)[:, 0]
            velocity = np.array([leg['v'][0] for leg in legs])
            self.roll_start_velocity = np.clip(velocity, -self.roll_speed, self.roll_speed)
            self.roll_start_velocity = np.clip(self.roll_start_velocity,
                                                np.minimum(0., 2.*travel/self.roll_duration),
                                                np.maximum(0., 2.*travel/self.roll_duration))
        self.enter('establish_support', 'ground')

    def can_align_support(self, up_z, legs):
        # A folded leg can sit nearly horizontal on the chassis support.
        # Its wheel need only be below the hip to use the lower approach.
        return up_z > .90 and all(leg['L'] < self.min_length+.042 and leg['p'][1] < -.025 for leg in legs)

    def can_lower_folded_legs(self, up_z, legs):
        if up_z <= .92 or not all(leg['L'] < self.min_length+.070 for leg in legs):
            return False
        kicked_up = any(leg['p'][1] > .022 for leg in legs)
        # A chassis perched on an edge can leave both folded wheels below
        # the hips but still far above the surface. They cannot roll in.
        suspended_fold = (all(leg['rest_h']-leg['height'] > .040 for leg in legs)
                          and any(abs(leg['theta']) > .85 for leg in legs))
        return kicked_up or suspended_fold

    def begin_support_alignment(self, legs, R, tilted=False, folded=False):
        if self.phase != 'establish_support' or self.support_action == 'sweep':
            self.routes = ['ground_short']*2
            self.begin_roll(legs)
        self.rear_flip_roll = False
        self.support_attempted = True
        self.support_tilted = tilted
        self.support_folded = folded
        self.damp_support_pitch = tilted and abs(R[2, 1]) < .22
        self.support_action = 'align'
        self.support_starts = [leg['p'].copy() for leg in legs]
        if folded:
            self.folded_angles = np.array([leg['theta'] for leg in legs])
            self.folded_goals = np.array([
                np.arctan2(-np.clip(leg['p'][0], -.028, .028), .147) for leg in legs
            ])
            self.folded_duration = max(.30, float(np.max(abs(self.folded_goals-self.folded_angles)))/4.)
        self.ref_q = [leg['q'].copy() for leg in legs]
        self.elapsed = 0.

    def begin_rear_flip(self, sample, R):
        # 倒扣时车头轴投影可能反向。用横轴确定翻正后的前向并锁定。
        lateral = R[:2, 1]
        forward = np.array([lateral[1], -lateral[0]]) / np.linalg.norm(lateral)
        if self.rear_flip_forward is not None and forward @ self.rear_flip_forward <= 0.:
            return False
        support = self.geometry.support
        z = support @ R[2]
        x = support @ (R[:2].T @ forward)
        rear_edge = np.min(x[z < min(z)+.015])
        radius = self.swing_length
        support_z = self.radius + min(z) - .045
        rear_goal = -np.sqrt(max(0., radius**2 - support_z**2))
        if rear_goal > rear_edge - .025:
            return False

        if self.rear_flip_forward is None:
            self.rear_flip_forward = forward
        self.geometry.fixed_forward = forward
        R, legs = self.geometry.observe(sample)
        self.contact_observation = R, legs
        self.rear_flip_start = [leg['p'].copy() for leg in legs]
        goal_angle = np.arctan2(-rear_goal, -support_z)
        self.rear_flip_angles = np.array([leg['theta'] for leg in legs])
        self.rear_flip_sweep = wrap(goal_angle-self.rear_flip_angles)
        self.rear_flip_edge = rear_edge
        sweep = max(abs(self.rear_flip_sweep))
        self.rear_flip_place_time = max(.25, sweep / 3.2)
        self.rear_flip_supported = False
        self.rear_flip_x = rear_goal
        self.ref_q = [leg['q'].copy() for leg in legs]
        self.enter('rear_flip')
        return True

    def rear_flip_torque(self, legs, R, pitch_rate, wheel_load, dt, tau):
        up_z = R[2, 2]
        if abs(R[2, 1]) > .50:
            self.rear_flip_forward = None
            self.retry('rear_flip_lateral_slip', R, legs)
            return

        clearance = [leg['height']-leg['rest_h'] for leg in legs]
        supported = [wheel_load is not None and wheel_load[i] > 12. and height > -.006
                     for i, height in enumerate(clearance)]
        rear = all(leg['p'][0] < -.055 and leg['p'][1] < -.035 for leg in legs)
        if self.rear_flip_supported and up_z < .25 and any(leg['p'][0] > -.035 for leg in legs):
            self.retry('rear_flip_wheel_left_rear', R, legs)
            return
        has_leverage = all(leg['p'][0] < min(-.12, self.rear_flip_edge-.012) for leg in legs)
        if (all(supported) and has_leverage) or (up_z > .25 and rear):
            self.rear_flip_supported = True
        ready_to_roll = False
        if wheel_load is not None:
            # 车壳支撑面在场地边缘会有误差，几何距离和承载互相补证。
            loaded = ((min(wheel_load) > 18. and min(clearance) > -.0025)
                      or (min(wheel_load) > 28. and min(clearance) > -.016 and abs(pitch_rate) < .45))
            ready_to_roll = self.rear_flip_supported and rear and up_z > .84 and abs(pitch_rate) < 2.5 and loaded
        if self.trace is not None:
            self.trace['rear_flip_support'] = supported

        if not self.rear_flip_supported:
            fraction = smooth(self.elapsed / self.rear_flip_place_time)
            for i, leg in enumerate(legs):
                start = self.rear_flip_start[i]
                angle = self.rear_flip_angles[i] + fraction*self.rear_flip_sweep[i]
                length = (1.-fraction)*np.linalg.norm(start) + fraction*self.swing_length
                if self.retries:
                    # 同方向重试时先缩短摆动半径，避免长腿再次抵住车壳或地面。
                    length -= .075*4.*fraction*(1.-fraction)
                target = -length * np.array([np.sin(angle), np.cos(angle)])
                tau[[i, i+2]] = self.pose_torque(i, leg, target, R, dt, 70., 3.)
            if self.elapsed > self.rear_flip_place_time + .65:
                self.retry('rear_flip_support_missing', R, legs)
            return

        # 先推地形成翻身力矩，接近正面时降低法向力，同时开始往髋下收轮。
        if up_z > -.60:
            self.rear_flip_x = min(-.09, self.rear_flip_x + 1.05 * dt)
        push = np.clip(110. + 90.*max(0., self.elapsed-self.rear_flip_place_time)
                       - 32.*max(0., pitch_rate-1.3), 10., 195.)
        push = self.preload + (push-self.preload) * np.clip((.86-up_z)/.85, .08, 1.)
        for i, leg in enumerate(legs):
            horizontal = np.clip(440.*(self.rear_flip_x-leg['p'][0])-32.*leg['v'][0], -95., 95.)
            downward = push + 55.*leg['v'][1]
            if not supported[i]:
                downward = max(24., downward)
            if up_z > .84 and abs(pitch_rate) < 3.5:
                downward = max(30., downward)
            force = np.array([horizontal, -np.clip(downward, self.preload, 195.)])
            tau[[i, i+2]] = leg['J'].T @ force + self.gravity_torque(i, leg, R) - .5*leg['qd']
            if self.trace is not None:
                self.trace['targets'][i] = dict(wheel_force_xz_n=force.tolist(), wheel_x_m=self.rear_flip_x)
        if ready_to_roll:
            self.routes = ['rear_ground'] * 2
            self.begin_roll(legs, rear_flip=True)
        elif self.elapsed > self.rear_flip_place_time + .45:
            if self.stalled(1.-up_z, .015, dt, .60) or self.elapsed > 3.2:
                self.retry('rear_flip_no_progress', R, legs)

    def begin_flip_brake(self, legs):
        self.brake_start_length = [leg['L'] for leg in legs]
        self.ref_q = [leg['q'].copy() for leg in legs]
        self.enter('flip_brake')

    def ground_target_x(self, leg, x):
        # At low hip height, x=0 may be inside the minimum-length boundary.
        minimum_x = np.sqrt(max(0., (self.min_length+.002)**2-leg['p'][1]**2))
        return (1. if x >= 0 else -1.)*max(abs(x), minimum_x)

    def ground_torque(self, i, leg, x, vx, R, preload=None):
        # Tangential tracking, gentle normal preload. Leg length is a consequence
        # of a feasible ground path, never an independent high-gain command.
        x = self.ground_target_x(leg, x)
        error = x-leg['p'][0]
        extra = np.sign(error)*700.*max(0., abs(error)-.035)
        Fx = np.clip(600.*error+extra+45.*(vx-leg['v'][0]), -110., 110.)
        if leg['L'] < self.min_length and Fx*leg['p'][0] < 0:
            Fx = np.sign(leg['p'][0])*min(20., 1500.*(self.min_length-leg['L']))
        Fz = -self.preload if preload is None else -preload
        tau = leg['J'].T @ np.array([Fx, Fz]) + self.gravity_torque(i, leg, R)
        if self.trace is not None:
            self.trace['targets'][i] = dict(wheel_x_m=x, wheel_vx_mps=vx,
                                             wheel_force_xz_n=[Fx, Fz])
        return tau

    def update(self, sample, dt, jumping=False, trace_idle=False, wheel_load=None):
        self.clock += dt
        self.cooldown = max(0., self.cooldown-dt)
        self.handoff = False
        if not self.enabled:
            return None
        dynamics = self.active or (self.trace_enabled and trace_idle)
        R, legs = self.geometry.observe(sample, dynamics=dynamics)
        angles = np.array([leg['theta'] for leg in legs])
        if self.last_angles is None:
            self.angle_continuous[:] = angles
        else:
            self.angle_continuous += wrap(angles-self.last_angles)
        self.last_angles = angles
        speed = np.linalg.norm(sample['gyro'])
        up_z = R[2, 2]
        max_angle = max(abs(angles))
        abnormal = up_z < .55 or max_angle > .95 or abs(R[2, 0]) > .70
        self.trace = None
        if self.phase == 'watch':
            self.quiet = self.quiet+dt if abnormal and not jumping and not self.cooldown else 0.
        debounce = .01 if up_z < .3 or max_angle > 1.2 else .03
        trigger = self.phase == 'watch' and self.quiet >= debounce
        if trigger and not dynamics:
            R, legs = self.geometry.observe(sample)
        if self.active or trigger:
            self.contact_observation = R, legs
        if self.trace_enabled and (trace_idle or self.active or trigger):
            self.trace = dict(phase_before=self.phase, elapsed_before_s=self.elapsed, abnormal=abnormal,
                              trigger=trigger, debounce_s=debounce, abnormal_duration_s=self.quiet,
                              jump_protected=jumping, cooldown_s=self.cooldown,
                              up_z=up_z, forward_z=R[2, 0], gyro_norm_radps=speed,
                              targets=[None, None], legs=[
                                  dict(length_m=leg['L'], theta_world_rad=leg['theta'],
                                       theta_continuous_rad=self.angle_continuous[i],
                                       wheel_xz_m=leg['p'].tolist(), wheel_vxz_mps=leg['v'].tolist(),
                                       hip_height_if_body_supported_m=leg['rest_h'],
                                       hip_height_if_wheel_supported_m=leg['height'])
                                  for i, leg in enumerate(legs)])
        if self.phase == 'watch':
            if not trigger:
                return None
            self.active = True
            self.total = 0.
            self.retries = 0
            self.support_attempted = False
            self.previous_wheel_vz = None
            self.block_reason = ''
            self.enter('settle')
        self.elapsed += dt
        self.total += dt
        tau = np.zeros(6)
        lateral_tilt = abs(R[2, 1])
        longitudinal_flip = up_z < -.68 and lateral_tilt < .36
        if self.rear_flip_forward is not None and up_z < .3 and lateral_tilt < .50:
            longitudinal_flip = True
        lost_upright = self.phase in ('establish_support', 'rise') and up_z < .3
        if lost_upright:
            # 直通路径收轮时再次倒扣，交给常规恢复重新选路。
            self.retries = int(self.rear_flip_roll and (self.phase == 'rise' or self.support_action in ('ground', 'align')))
            self.support_attempted = False
            self.block_reason = 'overturned_during_recovery'
            self.restart()
        elif self.phase in ('flip_place', 'flip_push', 'flip_brake') and longitudinal_flip:
            # 侧翻可能在摆腿过程中变成纵向倒扣，立即取消原来的撑翻方向。
            for i, leg in enumerate(legs):
                tau[[i, i+2]] = self.gravity_torque(i, leg, R) - 2.*leg['qd']
            if speed >= 1.2 and lateral_tilt > .32:
                self.restart()
            elif not self.begin_rear_flip(sample, R):
                self.retry('rear_flip_no_rear_leverage', R, legs)
        elif self.phase == 'settle':
            for i, leg in enumerate(legs):
                tau[[i, i+2]] = self.gravity_torque(i, leg, R) - 2.*leg['qd']
            if speed < 1.2:
                if longitudinal_flip:
                    if not self.begin_rear_flip(sample, R):
                        self.retry('rear_flip_no_rear_leverage', R, legs)
                elif up_z < .3:
                    self.flip_sign = (1. if R[2, 0] <= 0 else -1.)*(-1. if self.retries % 2 else 1.)
                    self.flip_limit_reach = abs(R[2, 1]) > .55
                    self.flip_start = [leg['p'].copy() for leg in legs]
                    self.geometry.fixed_forward = self.geometry.last_forward.copy()
                    self.flip_grow_time = 0. if all(abs(leg['L']-self.swing_length)<.005 for leg in legs) else self.grow_duration
                    self.flip_angles = angles.copy()
                    target = np.arctan2(-self.flip_sign*self.swing_length, .01)
                    self.flip_goals = self.flip_angles + wrap(target-self.flip_angles)
                    self.flip_place_duration = max(.525, float(np.max(abs(self.flip_goals-self.flip_angles)))/2.4)
                    self.ref_q = [leg['q'].copy() for leg in legs]
                    self.enter('flip_place')
                elif not self.support_attempted and self.can_lower_folded_legs(up_z, legs):
                    self.begin_support_alignment(legs, R, folded=True)
                elif not self.support_attempted and self.can_align_support(up_z, legs):
                    self.begin_support_alignment(legs, R)
                else:
                    self.prepare_routes(legs, up_z)
            elif self.elapsed > 1.5:
                self.retry('body_not_settled', R, legs)
        elif self.phase == 'rear_flip':
            self.rear_flip_torque(legs, R, sample['gyro'][1], wheel_load, dt, tau)
        elif self.phase in ('flip_place', 'flip_push'):
            # Positive rate means the chassis up axis is rotating toward vertical.
            up_rate = R[2, 0]*sample['gyro'][1]-R[2, 1]*sample['gyro'][0]
            if up_z > .45 and (self.phase == 'flip_place' or speed < 9.):
                self.restart()
            elif up_z > .45 or (self.phase == 'flip_push' and up_z > .12 and up_rate > 10.):
                self.begin_flip_brake(legs)
                for i, leg in enumerate(legs):
                    tau[[i, i+2]] = self.gravity_torque(i, leg, R)-2.*leg['qd']
            else:
                for i, leg in enumerate(legs):
                    # Place a long leg beyond one body edge, then lengthen it
                    # against the ground to rotate the body about its resting edge.
                    if self.phase == 'flip_place':
                        b = smooth(max(0., self.elapsed-self.flip_grow_time)/self.flip_place_duration)
                        theta = self.flip_angles[i]+(self.flip_goals[i]-self.flip_angles[i])*b
                        L = np.linalg.norm(self.flip_start[i]) + (self.swing_length-np.linalg.norm(self.flip_start[i]))*(1. if self.flip_grow_time==0. else smooth(self.elapsed/self.flip_grow_time))
                        target = np.array([-L*np.sin(theta), -L*np.cos(theta)])
                        tau[[i, i+2]] = self.pose_torque(i, leg, target, R, dt, 60., 2.5)
                    else:
                        scale = float(np.clip((15.-up_rate)/6., .25, 1.)) if up_z > -.55 else 1.
                        F = scale*np.array([-self.flip_sign*35., -min(260., 65.+120.*self.elapsed)])
                        tau[[i, i+2]] = leg['J'].T@F + self.gravity_torque(i, leg, R) - .5*leg['qd']
                        if self.trace is not None:
                            self.trace['targets'][i] = dict(wheel_force_xz_n=F.tolist())
                    tau[4+i] = np.clip(-.15*sample[('left','right')[i]+'_wheel_vel'], -1.5, 1.5)
                if self.phase == 'flip_place':
                    goals = [np.array([-self.swing_length*np.sin(angle), -self.swing_length*np.cos(angle)]) for angle in self.flip_goals]
                    if self.flip_limit_reach:
                        goals = [leg['A']@self.reachable_point(leg, goal)+leg['offset'] for leg, goal in zip(legs, goals)]
                    error = max(np.linalg.norm(leg['p']-goal) for leg, goal in zip(legs, goals))
                    if error < .06 and max(np.linalg.norm(leg['v']) for leg in legs) < .5:
                        self.enter('flip_push')
                    elif self.elapsed > self.flip_grow_time+self.flip_place_duration:
                        if self.stalled(error, .006, dt) or self.elapsed > self.flip_grow_time+self.flip_place_duration+1.:
                            self.retry('flip_placement_stalled', R, legs)
                elif self.elapsed > .65 and (self.stalled(1.-up_z, .02, dt, .5) or self.elapsed > 2.5):
                    self.retry('flip_push_stalled', R, legs)
        elif self.phase == 'flip_brake':
            b = smooth(self.elapsed/.13)
            for i, leg in enumerate(legs):
                start = self.brake_start_length[i]
                length = start+b*(min(start, .225)-start)
                point = leg['body_p']*(length/max(leg['L'], .01))
                target = leg['A']@point+leg['offset']
                tau[[i, i+2]] = self.pose_torque(i, leg, target, R, dt, 70., 3.)
                tau[4+i] = np.clip(-.15*sample[('left','right')[i]+'_wheel_vel'], -1.5, 1.5)
            if (up_z > .80 and speed < 2.6) or (up_z > .55 and speed < 1.6):
                self.restart()
            elif self.elapsed > .65:
                if up_z > .45:
                    self.restart()
                else:
                    self.retry('flip_brake_stalled', R, legs)
        elif self.phase == 'establish_support':
            if self.support_action == 'sweep':
                for i, leg in enumerate(legs):
                    if self.leg_done[i]:
                        x = self.starts[i][0] if self.routes[i]=='ground_short' else self.parking[i][0]
                        tau[[i, i+2]] = self.ground_torque(i, leg, x, 0., R)
                        continue
                    # Grow first, then sweep at the common maximum usable radius.
                    grow = 1. if self.grow_times[i]==0. else smooth(self.elapsed/self.grow_times[i])
                    u = smooth((self.elapsed-self.sweep_delay[i])/self.swing_duration[i])
                    L = np.linalg.norm(self.starts[i]) + (self.route_length-np.linalg.norm(self.starts[i]))*grow
                    angle = self.start_angles[i]+(self.swing_targets[i]-self.start_angles[i])*u
                    point = np.array([-L*np.sin(angle), -L*np.cos(angle)])
                    tau[[i, i+2]] = self.pose_torque(i, leg, point, R, dt)
                    angle_error = abs(self.angle_continuous[i]-self.swing_targets[i])
                    settled_foot = np.linalg.norm(leg['v']) < .45
                    parked = angle_error < .08 and np.linalg.norm(leg['p']-self.parking[i]) < .03
                    # A deep rear wheel can roll on a lower step even if the
                    # chassis-based flat-floor estimate puts it above ground.
                    low_enough = leg['p'][1] < -.10 or leg['height']-leg['rest_h'] > -.035
                    can_roll = (u > .75 and angle_error < .24 and leg['p'][0] < -.10
                                and leg['p'][1] < -.035 and low_enough)
                    if settled_foot and (parked or can_roll):
                        self.leg_done[i] = True
                        if not parked:
                            self.parking[i] = leg['p'].copy()
                if all(self.leg_done):
                    self.begin_roll(legs)
                else:
                    pending = [i for i, done in enumerate(self.leg_done) if not done]
                    if self.elapsed > max(self.sweep_delay[i]+self.swing_duration[i] for i in pending):
                        error = max(np.linalg.norm(legs[i]['p']-self.parking[i]) for i in pending)
                        if self.stalled(error, .005, dt) or self.elapsed > max(self.sweep_delay)+max(self.swing_duration)+1.:
                            self.retry('reposition_stalled', R, legs)
            elif self.support_action == 'align':
                duration = self.folded_duration if self.support_folded else (.35 if self.support_tilted else .26)
                b = smooth(self.elapsed/duration)
                roll = sample['euler'][0]
                pitch_rate = np.cos(roll)*sample['gyro'][1]-np.sin(roll)*sample['gyro'][2]
                pitch_sine = -R[2, 0]
                for i, leg in enumerate(legs):
                    start = self.support_starts[i]
                    floor_z = self.radius-leg['rest_h']
                    if self.damp_support_pitch:
                        # Move support toward the lean and brake before passing level.
                        x = np.clip(.26*pitch_sine+.08*pitch_rate, -.17, .17)
                        z = min(-.147, floor_z-.035)
                        kp, kd = 55., 3.5
                    elif self.support_tilted:
                        x = np.clip(.38*pitch_sine, -.17, .17)
                        z = min(-.147, floor_z-.09)
                        kp, kd = 90., 3.
                    else:
                        x, z = np.clip(start[0], -.038, .038), -.147
                        kp, kd = 90., 3.
                    goal = np.array([x, z])
                    target = start+b*(goal-start)
                    if self.support_folded:
                        theta = self.folded_angles[i]+b*(self.folded_goals[i]-self.folded_angles[i])
                        start_length = np.linalg.norm(start)
                        radius = max(self.min_length+.003, start_length+b*(.149-start_length))
                        target = -radius*np.array([np.sin(theta), np.cos(theta)])
                    tau[[i, i+2]] = self.pose_torque(i, leg, target, R, dt, kp, kd)
                support_upright = .92 if self.support_tilted else .90
                support_ready = up_z > support_upright and max_angle < .65 and speed < 1.2
                if self.support_folded:
                    loaded = wheel_load is not None and min(wheel_load) > 30.
                    near_floor = all(leg['height']-leg['rest_h'] > -.012 for leg in legs)
                    support_ready = (up_z > support_upright and max_angle < .42
                                     and ((speed < 1.2 and near_floor) or (speed < 1.8 and loaded)))
                if self.damp_support_pitch:
                    loaded = wheel_load is not None and min(wheel_load) > 30.
                    near_floor = all(leg['height']-leg['rest_h'] > -.008 for leg in legs)
                    support_ready = support_ready and max_angle < .48 and abs(pitch_rate) < .70 and (loaded or near_floor)
                if self.trace is not None:
                    self.trace.update(support_mode=('folded' if self.support_folded else 'tilted' if self.support_tilted else 'compact'),
                                      support_ready=support_ready,
                                      pitch_damping=self.damp_support_pitch, pitch_rate_radps=pitch_rate)
                if support_ready:
                    self.roll_targets = [leg['p'][0] for leg in legs]
                    self.start_balance()
                elif self.elapsed > (duration+.45 if self.support_folded else 1.05 if self.support_tilted else .58):
                    self.retry('support_alignment_stalled', R, legs)
            elif (speed < 1.2 and up_z > .55
                  and any(leg['p'][1] > .022 and abs(leg['theta']) > 1.65
                          and np.linalg.norm(leg['v']) < .25 for leg in legs)):
                if not self.support_attempted and self.can_lower_folded_legs(up_z, legs):
                    self.begin_support_alignment(legs, R, folded=True)
                else:
                    self.retry('wheel_left_roll_path', R, legs)
            else:
                u = np.clip(self.elapsed/self.roll_duration, 0., 1.)
                b = smooth(u)
                bd = (30*u*u-60*u**3+30*u**4)/self.roll_duration if u < 1 else 0.
                for i, leg in enumerate(legs):
                    x = self.starts[i][0]+(self.roll_targets[i]-self.starts[i][0])*b
                    vx = (self.roll_targets[i]-self.starts[i][0])*bd
                    if self.rear_flip_roll:
                        v0 = self.roll_start_velocity[i]
                        x += self.roll_duration*v0*u*(1.-u)**3
                        vx += v0*(1.-u)**2*(1.-4.*u)
                    preload = None
                    if self.rear_flip_roll:
                        # 翻身后的车体仍在摆动，用轮端竖直速度补足压地力。
                        preload = np.clip(42.+85.*leg['v'][1], 20., 85.)
                    tau[[i, i+2]] = self.ground_torque(i, leg, x, vx, R, preload)
                error = max(abs(leg['p'][0]-self.ground_target_x(leg, self.roll_targets[i])) for i, leg in enumerate(legs))
                short_under_body = max(leg['L'] for leg in legs) < self.min_length+.014 and max_angle < .78
                slow_feet = max(np.linalg.norm(leg['v']) for leg in legs) < .30
                if self.trace is not None:
                    self.trace.update(roll_target_x_m=self.roll_targets.copy(), roll_end_error_m=error,
                                      short_under_body=short_under_body, slow_feet=slow_feet,
                                      can_align_support=self.can_align_support(up_z, legs))
                at_end = short_under_body or (error < .025 and max_angle < .9)
                if self.rear_flip_roll:
                    at_end = at_end and wheel_load is not None and min(wheel_load) > 12.
                if up_z > .90 and speed < 1.2 and slow_feet and at_end:
                    self.start_balance()
                elif up_z <= .90 and speed < 1.5 and slow_feet and at_end:
                    self.begin_support_alignment(legs, R, tilted=True)
                elif self.elapsed > self.roll_duration and (self.stalled(error, .005, dt) or self.elapsed > self.roll_duration+1.):
                    # A short leg can be stalled on body support before reaching its rolling goal.
                    if speed < 1. and slow_feet and not self.support_attempted and self.can_lower_folded_legs(up_z, legs):
                        self.begin_support_alignment(legs, R, folded=True)
                    elif speed < 1. and slow_feet and self.can_align_support(up_z, legs):
                        self.begin_support_alignment(legs, R)
                    elif (up_z > .68 and speed < 1.2 and slow_feet and max_angle < 1.8
                          and min(leg['p'][1] for leg in legs) < -.06):
                        self.begin_support_alignment(legs, R, tilted=True)
                    else:
                        self.retry('roll_in_stalled', R, legs)
        elif self.phase == 'rise':
            self.blend = 1. if self.capture_duration <= 0. else float(smooth(self.elapsed/self.capture_duration))
            self.capture_pose_blend = (1. if self.capture_pose_duration <= 0.
                                       else float(smooth(self.elapsed/self.capture_pose_duration)))
            clearances = [leg['height']-leg['rest_h'] for leg in legs]
            clearance = min(clearances)
            # With wheels on different levels, one flat-floor clearance can be
            # near zero even though both wheels are already carrying the body.
            split_support = (wheel_load is not None and min(wheel_load) > 40.
                             and up_z > .98 and max_angle < .28 and speed < .65
                             and clearance > -.015 and max(clearances)-clearance > .045)
            ready = split_support or (up_z > .94 and abs(R[2, 0]) < .14 and max_angle < .40
                                      and speed < 1. and clearance > .012)
            if self.trace is not None:
                self.trace.update(estimated_clearance_m=clearance, clearance_by_wheel_m=clearances,
                                  split_support_ready=split_support, balance_ready=ready)
            if ready and self.capture_pose_blend >= 1.:
                self.active = False
                self.cooldown = .2
                self.quiet = 0.
                if hasattr(self.geometry, 'fixed_forward'):
                    del self.geometry.fixed_forward
                    self.last_angles = None
                self.rear_flip_roll = False
                self.rear_flip_forward = None
                self.enter('watch')
            elif self.elapsed > max(.12, self.capture_pose_duration+.10) and (abnormal or
                    self.stalled(max(0., .012-clearance)+.12*max_angle+.1*(1.-up_z), .003, dt, .65) or self.elapsed > 2.):
                self.retry('balance_capture_stalled', R, legs)
        elif self.phase == 'blocked':
            moved = np.linalg.norm(R[2]-self.blocked_up) > .18 or np.max(abs(wrap(np.array([leg['q'] for leg in legs])-self.blocked_q))) > .18
            if self.trace is not None:
                self.trace.update(moved_since_blocked=moved, support_attempted=self.support_attempted,
                                  can_align_support=self.can_align_support(up_z, legs))
            if (up_z < .3 <= self.blocked_up[2]) or (moved and speed < 1.2 and self.elapsed > .2):
                self.retries = 0
                self.support_attempted = False
                self.block_reason = ''
                self.restart()
            elif (not self.support_attempted and speed < 1.
                  and self.block_reason in ('wheel_left_roll_path', 'roll_in_stalled')
                  and self.can_lower_folded_legs(up_z, legs)):
                self.begin_support_alignment(legs, R, folded=True)
            elif (not self.support_attempted and speed < 1. and self.can_align_support(up_z, legs)
                  and all(np.linalg.norm(leg['v']) < .15 for leg in legs)):
                self.begin_support_alignment(legs, R)
        if self.phase in ('settle', 'blocked'):
            for i, leg in enumerate(legs):
                tau[[i, i+2]] = self.gravity_torque(i, leg, R)-2.*leg['qd']
            tau[4:] = 0.
        tau[:4] -= sample.get("gas_hip_torque_nm", 0.0)
        limit = np.array([self.hip_limit]*4+[self.wheel_limit]*2)
        if self.trace is not None:
            self.trace['recovery_ctrl_before_clip_nm'] = tau.tolist()
        tau = np.clip(tau, -limit, limit)
        self.diagnostics = dict(phase=self.phase, routes=self.routes.copy(), legs=[dict(length=leg['L'], theta=leg['theta'], wheel_x=leg['p'][0], height=leg['height']) for leg in legs], up_z=R[2, 2], reason=self.block_reason)
        return tau
