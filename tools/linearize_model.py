"""
MuJoCo Euler linearization for the Serial five-link wheel-leg robot.

This script linearizes the same model used by the simulator:

  z[k+1] = Ad z[k] + Bd u_lqr[k]

The reduced 10-state matches sim/controller.py:
  [x_odom, yaw, pitch, theta_L, theta_R,
   x_odom_dot, yaw_dot, pitch_dot, theta_L_dot, theta_R_dot]

The reduced 4-input matches the LQR output:
  [tau_L_virtual_leg, tau_R_virtual_leg, T_wheel_L, T_wheel_R]

The saved matrices are discrete-time matrices. recompute_lqr.py detects this
and skips continuous-to-discrete conversion.
"""

import argparse
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.linalg import pinv
from scipy.optimize import brentq, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from leg import FiveLinkLeg
from utils import load_config, quat_to_euler, resolve_project_path, sensor_value


def restore_state(model, data, qpos, qvel, ctrl):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mj.mj_forward(model, data)


class SymmetricFiveLinkPose:
    def __init__(self, model, cfg, wheel_penetration):
        self.model = model
        self.cfg = cfg
        self.leg_geom = cfg["geometry"]["five_link"]
        self.wheel_radius = cfg["geometry"]["wheel"]["radius_m"]
        self.wheel_penetration = wheel_penetration
        self.dt = model.opt.timestep
        self.length_probe = FiveLinkLeg("probe", self.leg_geom, self.dt)
        self.passive_qpos = np.array([8, 9, 10, 11, 15, 16, 17, 18], dtype=int)
        self.site_pairs = [
            ("left_rear_to_bridge_connect", "left_bridge_rear_connect"),
            ("left_rear_to_lower_connect", "left_lower_leg_rear_connect"),
            ("right_rear_to_bridge_connect", "right_bridge_rear_connect"),
            ("right_rear_to_lower_connect", "right_lower_leg_rear_connect"),
        ]

    def length_from_front_angle(self, front_angle):
        length, _ = self.length_probe.forward(front_angle, np.pi - front_angle, 0.0)
        return length

    def front_angle_for_length(self, target_h):
        lo = np.deg2rad(-10.0)
        hi = np.deg2rad(50.0)
        return brentq(lambda a: self.length_from_front_angle(a) - target_h, lo, hi)

    def solve(self, data, target_h, qpos_seed=None):
        front = self.front_angle_for_length(target_h)
        rear = np.pi - front

        qpos = self.model.qpos0.copy() if qpos_seed is None else qpos_seed.copy()
        wheel_z = self.wheel_radius - self.wheel_penetration
        qpos[:3] = [0.0, 0.0, target_h + wheel_z]
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos[7] = front
        qpos[13] = rear
        qpos[14] = front
        qpos[20] = rear
        qpos[12] = 0.0
        qpos[19] = 0.0

        y0 = np.r_[qpos[2], qpos[self.passive_qpos]]
        lower = np.r_[0.04, np.full(self.passive_qpos.size, -np.pi)]
        upper = np.r_[0.60, np.full(self.passive_qpos.size, np.pi)]

        def residual(y):
            q = qpos.copy()
            q[2] = y[0]
            q[self.passive_qpos] = y[1:]
            data.qpos[:] = q
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            mj.mj_forward(self.model, data)

            res = []
            for site_a, site_b in self.site_pairs:
                res.extend(100.0 * (data.site(site_a).xpos - data.site(site_b).xpos))
            res.append(20.0 * (data.body("left_wheel").xpos[2] - wheel_z))
            res.append(20.0 * (data.body("right_wheel").xpos[2] - wheel_z))
            return np.asarray(res)

        sol = least_squares(
            residual,
            y0,
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=500,
        )
        qpos[2] = sol.x[0]
        qpos[self.passive_qpos] = sol.x[1:]

        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mj.mj_forward(self.model, data)
        return qpos.copy(), front, rear, np.linalg.norm(residual(sol.x))


class ReducedStateMapper:
    def __init__(self, model, sensor_names, leg_geom, wheel_radius):
        self.model = model
        self.sensor_names = sensor_names
        self.leg_geom = leg_geom
        self.wheel_radius = wheel_radius
        self.dt = model.opt.timestep

    def legs(self, data):
        s = self.sensor_names
        euler = quat_to_euler(sensor_value(data, s["quat"]))
        left = FiveLinkLeg("left", self.leg_geom, self.dt)
        right = FiveLinkLeg("right", self.leg_geom, self.dt)
        left.forward(sensor_value(data, s["left_front_pos"])[0],
                     sensor_value(data, s["left_back_pos"])[0],
                     euler[1])
        right.forward(sensor_value(data, s["right_front_pos"])[0],
                      sensor_value(data, s["right_back_pos"])[0],
                      euler[1])
        return left, right

    def leg_thetas(self, data):
        left, right = self.legs(data)
        return np.array([left.theta, right.theta])

    def theta_pos_jacobian(self, data, eps):
        qpos0 = data.qpos.copy()
        qvel0 = data.qvel.copy()
        ctrl0 = data.ctrl.copy()
        jac = np.zeros((2, self.model.nv))

        for i in range(self.model.nv):
            dq = np.zeros(self.model.nv)

            qpos_p = qpos0.copy()
            dq[i] = eps
            mj.mj_integratePos(self.model, qpos_p, dq, 1.0)
            restore_state(self.model, data, qpos_p, qvel0, ctrl0)
            theta_p = self.leg_thetas(data)

            qpos_m = qpos0.copy()
            dq[i] = -eps
            mj.mj_integratePos(self.model, qpos_m, dq, 1.0)
            restore_state(self.model, data, qpos_m, qvel0, ctrl0)
            theta_m = self.leg_thetas(data)

            jac[:, i] = (theta_p - theta_m) / (2.0 * eps)

        restore_state(self.model, data, qpos0, qvel0, ctrl0)
        return jac

    def reduced_state(self, data, eps):
        s = self.sensor_names
        euler = quat_to_euler(sensor_value(data, s["quat"]))
        gyro = sensor_value(data, s["gyro"])
        left, right = self.legs(data)
        theta_dot = self.theta_pos_jacobian(data, eps) @ data.qvel

        left_wheel_pos = sensor_value(data, s["left_wheel_pos"])[0]
        right_wheel_pos = sensor_value(data, s["right_wheel_pos"])[0]
        left_wheel_vel = sensor_value(data, s["left_wheel_vel"])[0]
        right_wheel_vel = sensor_value(data, s["right_wheel_vel"])[0]

        x_odom = 0.5 * self.wheel_radius * (left_wheel_pos + right_wheel_pos)
        x_odom_dot = 0.5 * self.wheel_radius * (left_wheel_vel + right_wheel_vel)

        return np.array([
            x_odom,
            euler[2],
            euler[1],
            left.theta,
            right.theta,
            x_odom_dot,
            gyro[2],
            gyro[1],
            theta_dot[0],
            theta_dot[1],
        ])

    def jacobian(self, data, eps):
        qpos0 = data.qpos.copy()
        qvel0 = data.qvel.copy()
        ctrl0 = data.ctrl.copy()
        nx = 2 * self.model.nv + self.model.na
        jac = np.zeros((10, nx))

        for i in range(self.model.nv):
            dq = np.zeros(self.model.nv)

            qpos_p = qpos0.copy()
            dq[i] = eps
            mj.mj_integratePos(self.model, qpos_p, dq, 1.0)
            restore_state(self.model, data, qpos_p, qvel0, ctrl0)
            z_p = self.reduced_state(data, eps)

            qpos_m = qpos0.copy()
            dq[i] = -eps
            mj.mj_integratePos(self.model, qpos_m, dq, 1.0)
            restore_state(self.model, data, qpos_m, qvel0, ctrl0)
            z_m = self.reduced_state(data, eps)

            jac[:, i] = (z_p - z_m) / (2.0 * eps)

        for i in range(self.model.nv):
            qvel_p = qvel0.copy()
            qvel_p[i] += eps
            restore_state(self.model, data, qpos0, qvel_p, ctrl0)
            z_p = self.reduced_state(data, eps)

            qvel_m = qvel0.copy()
            qvel_m[i] -= eps
            restore_state(self.model, data, qpos0, qvel_m, ctrl0)
            z_m = self.reduced_state(data, eps)

            jac[:, self.model.nv + i] = (z_p - z_m) / (2.0 * eps)

        restore_state(self.model, data, qpos0, qvel0, ctrl0)
        return jac


def lqr_input_to_actuator_ctrl(mapper, data, left_force, right_force, u_lqr):
    s = mapper.sensor_names
    left, right = mapper.legs(data)

    left_front, left_back = left.vmc(
        left_force,
        u_lqr[0],
        sensor_value(data, s["left_front_pos"])[0],
        sensor_value(data, s["left_back_pos"])[0],
    )
    right_front, right_back = right.vmc(
        right_force,
        u_lqr[1],
        sensor_value(data, s["right_front_pos"])[0],
        sensor_value(data, s["right_back_pos"])[0],
    )

    return np.array([
        left_front,
        right_front,
        left_back,
        right_back,
        u_lqr[2],
        u_lqr[3],
    ])


def lqr_to_actuator_jacobian_analytic(mapper, data, left_force, right_force):
    """Compute G = ∂u_actuator/∂u_lqr using the analytic VMC derivatives.

    The numerical Jacobian around equilibrium gives same-sign front/rear
    torques from virtual torque, which is wrong — virtual torque should
    produce opposite-signed hip torques (rotation, not translation).
    The analytic derivatives from the VMC formulas capture this correctly.
    """
    s = mapper.sensor_names
    left = FiveLinkLeg("left", mapper.leg_geom, mapper.dt)
    right = FiveLinkLeg("right", mapper.leg_geom, mapper.dt)
    euler = quat_to_euler(sensor_value(data, s["quat"]))
    left.forward(sensor_value(data, s["left_front_pos"])[0],
                 sensor_value(data, s["left_back_pos"])[0], euler[1])
    right.forward(sensor_value(data, s["right_front_pos"])[0],
                  sensor_value(data, s["right_back_pos"])[0], euler[1])

    # VMC analytic derivatives:
    # ∂τ_front/∂t_l = Lf * (-sin(θ-φb)) * sin(φf-θf) / (L * sin(φf-φb))
    # ∂τ_back/∂t_l  = Lb * (-sin(θ-φf)) * sin(θb-φb) / (L * sin(φf-φb))
    d_front_d_torque = np.zeros(2)
    d_back_d_torque = np.zeros(2)

    for idx, (leg, lf, lb) in enumerate([(left, mapper.leg_geom["front_big_m"],
                                           mapper.leg_geom["back_big_m"]),
                                          (right, mapper.leg_geom["front_big_m"],
                                           mapper.leg_geom["back_big_m"])]):
        L = leg.length
        th = leg.theta
        pf = leg.phi_front
        pb = leg.phi_back
        th_f = sensor_value(data, s["left_front_pos" if idx == 0 else "right_front_pos"])[0]
        th_b = sensor_value(data, s["left_back_pos" if idx == 0 else "right_back_pos"])[0]
        denom = np.sin(pf - pb)
        if abs(denom) < 1e-9:
            denom = np.copysign(1e-9, denom if denom != 0 else 1.0)

        d_front_d_torque[idx] = (lf * (-np.sin(th - pb)) * np.sin(pf - th_f)
                                 / (L * denom))
        d_back_d_torque[idx] = (lb * (-np.sin(th - pf)) * np.sin(th_b - pb)
                                / (L * denom))

    G = np.zeros((6, 4))
    G[0, 0] = d_front_d_torque[0]   # left front hip from τL
    G[2, 0] = d_back_d_torque[0]    # left rear hip from τL
    G[1, 1] = d_front_d_torque[1]   # right front hip from τR
    G[3, 1] = d_back_d_torque[1]    # right rear hip from τR
    G[4, 2] = 1.0                    # left wheel from TwL
    G[5, 3] = 1.0                    # right wheel from TwR
    return G


def transition_fd(model, data, eps):
    nx = 2 * model.nv + model.na
    A = np.zeros((nx, nx), dtype=np.float64, order="C")
    B = np.zeros((nx, model.nu), dtype=np.float64, order="C")
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = data.ctrl.copy()
    mj.mjd_transitionFD(model, data, eps, 1, A, B, None, None)
    restore_state(model, data, qpos0, qvel0, ctrl0)
    return A, B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--n-lengths", type=int, default=24)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--support-force", type=float, default=104.0)
    parser.add_argument("--wheel-penetration", type=float, default=0.001)
    parser.add_argument("--output", default="data/sys_matrices_serial.npz")
    args = parser.parse_args()

    cfg = load_config()
    xml_path = resolve_project_path(cfg["mujoco"]["xml_path"])
    model = mj.MjModel.from_xml_path(str(xml_path))
    if model.opt.integrator != mj.mjtIntegrator.mjINT_EULER:
        raise RuntimeError("mjd_transitionFD requires Euler. Set the MJCF integrator to Euler.")

    data = mj.MjData(model)
    mj.mj_forward(model, data)

    mapper = ReducedStateMapper(
        model,
        cfg["mujoco"]["sensors"],
        cfg["geometry"]["five_link"],
        cfg["geometry"]["wheel"]["radius_m"],
    )
    pose_solver = SymmetricFiveLinkPose(model, cfg, args.wheel_penetration)

    h_vec = np.linspace(0.12, 0.35, args.n_lengths)
    A_set = np.zeros((args.n_lengths, 10, 10))
    B_set = np.zeros((args.n_lengths, 10, 4))

    print(f"MuJoCo Euler discrete linearization, eps={args.eps:g}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep:g}")
    print(
        f"Static IK pose, support_force={args.support_force:g} N per leg, "
        f"wheel_penetration={args.wheel_penetration:g} m\n"
    )

    # ── Import controller for settling (uses legwheel K as stabiliser) ──
    from controller import StandController

    class DummyArgs:
        pass

    ca = DummyArgs()
    ca.viewer = False
    ca.control_mode = "lqr"
    ca.duration = 0.0
    ca.target_leg_length = None
    ca.support_force = None
    ca.target_x = 0.0
    ca.target_yaw = 0.0
    ca.target_pitch = 0.0
    ca.target_left_leg_theta = 0.0
    ca.target_right_leg_theta = 0.0
    ca.target_x_dot = 0.0
    ca.target_yaw_dot = 0.0
    ca.lqr_gain_table = None
    ca.max_keyboard_speed = None
    ca.max_keyboard_yaw_rate = None
    ca.keyboard_speed_accel = None
    ca.keyboard_speed_release_accel = None
    ca.keyboard_yaw_rate_accel = None
    ca.keyboard_yaw_rate_release_accel = None
    ca.initial_pitch = 0.0
    ca.initial_x_velocity = 0.0
    ca.push_body = "base"
    ca.push_time = 0.0
    ca.push_duration = 0.0
    ca.push_force_x = 0.0
    ca.push_force_y = 0.0
    ca.push_force_z = 0.0
    ca.push_torque_x = 0.0
    ca.push_torque_y = 0.0
    ca.push_torque_z = 0.0
    ca.joint_torque_limit = 40.0
    ca.wheel_torque_limit = 5.0
    ca.length_position_force_limit = 1500.0
    ca.length_velocity_force_limit = 1000.0
    ca.roll_force_limit = 1000.0
    ca.fall_angle = 99.0
    ca.log_every = 0.0

    controller = StandController(model, data, cfg, ca)
    settle_steps = int(3.0 / model.opt.timestep)  # 3 seconds settling

    qpos_seed = None
    for k in range(args.start_idx, args.n_lengths):
        h = h_vec[k]
        print(f"[{k + 1}/{args.n_lengths}] h={h:.3f}", end=" ", flush=True)

        # Start from IK pose (good initial guess)
        if qpos_seed is None:
            qpos_seed, _, _, _ = pose_solver.solve(data, h)
        else:
            qpos_seed, _, _, _ = pose_solver.solve(data, h, qpos_seed)

        data.qpos[:] = qpos_seed
        data.qvel[:] = 0
        data.ctrl[:] = 0
        mj.mj_forward(model, data)

        # Settle with controller to find true equilibrium
        controller.target_leg_length = h
        controller.reset_internal_state()
        print("settle...", end=" ", flush=True)
        for _ in range(settle_steps):
            controller.step((0.0, 0.0, 0.0, 0.0))

        # Record equilibrium
        qpos0 = data.qpos.copy()
        qvel0 = data.qvel.copy()
        ctrl0 = data.ctrl.copy()
        z0 = mapper.reduced_state(data, args.eps)
        left_force = controller.left_leg.force if hasattr(controller.left_leg, 'force') and controller.left_leg.force != 0 else args.support_force * np.cos(z0[3])
        right_force = controller.right_leg.force if hasattr(controller.right_leg, 'force') and controller.right_leg.force != 0 else args.support_force * np.cos(z0[4])
        print("lin...", end=" ", flush=True)

        # Linearize at equilibrium
        A_full, B_act = transition_fd(model, data, args.eps)
        restore_state(model, data, qpos0, qvel0, ctrl0)
        J = mapper.jacobian(data, args.eps)
        G = lqr_to_actuator_jacobian_analytic(mapper, data, left_force, right_force)

        A_10 = J @ A_full @ pinv(J)
        B_10 = J @ B_act @ G

        A_set[k] = A_10
        B_set[k] = B_10

        eig_open = np.max(np.abs(np.linalg.eigvals(A_10)))
        print(
            f"L={0.5*(z0[3]+z0[4]):.4f} "
            f"theta=({z0[3]:+.4f},{z0[4]:+.4f}) "
            f"G[0,0]={G[0,0]:+.2f} G[2,0]={G[2,0]:+.2f} "
            f"max|eig|={eig_open:.3f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        A_set=A_set,
        B_set=B_set,
        h_vec=h_vec,
        discrete=np.array(True),
        dt=np.array(model.opt.timestep),
        wheel_penetration=np.array(args.wheel_penetration),
        source=np.array("mujoco_mjd_transitionFD_euler"),
    )
    print(f"\nSaved {out_path}")
    print(f"Next: python tools/recompute_lqr.py {out_path}")


if __name__ == "__main__":
    main()

#python tools/linearize_model.py --n-lengths 24