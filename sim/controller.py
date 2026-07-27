import re
import socket
import struct
import numpy as np
import mujoco as mj
from utils import (sensor_value, quat_to_euler, clip_symmetric, 
                   resolve_project_path, normalize_angle)
from change_length_fit import ChangeLengthFit
from leg import FiveLinkLeg
from pid import PID
from kalman import KalmanOdometry
from lqr_governor import LqrReferenceGovernor

# ---------- ContinuousAngle 合并 ----------
class ContinuousAngle:
    def __init__(self):
        self.last_wrapped = None
        self.value = 0.0

    def reset(self, wrapped_angle=0.0):
        self.last_wrapped = normalize_angle(wrapped_angle)
        self.value = self.last_wrapped

    def update(self, wrapped_angle):
        wrapped_angle = normalize_angle(wrapped_angle)
        if self.last_wrapped is None:
            self.reset(wrapped_angle)
            return self.value
        self.value += normalize_angle(wrapped_angle - self.last_wrapped)
        self.last_wrapped = wrapped_angle
        return self.value

class StandController:
    def __init__(self, model, data, cfg, args):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.args = args
        self.dt = model.opt.timestep
        self.push_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.push_body)
        if self.push_body_id < 0:
            raise ValueError(f"Unknown push body: {args.push_body}")

        # 轮子刚体 ID，用于着地检测（跳过不可靠的 Fn 估计）
        self.sensor_names = cfg["mujoco"]["sensors"]
        self.geometry = cfg["geometry"]
        self.gravity = cfg["simulation"]["gravity_mps2"]
        self.total_mass = cfg["mass_properties"]["total_robot_mass_kg"]
        self.support_force = args.support_force
        if self.support_force is None:
            self.support_force = 0.5 * self.total_mass * self.gravity

        self.left_leg = FiveLinkLeg("left", self.geometry["five_link"], self.dt)
        self.right_leg = FiveLinkLeg("right", self.geometry["five_link"], self.dt)
        # World-frame Kalman for odometry logging
        self.odometry = KalmanOdometry(self.geometry["wheel"]["radius_m"], self.dt)
        # Body-frame Kalman for LQR state (sagittal-plane model)
        self.body_odom = KalmanOdometry(self.geometry["wheel"]["radius_m"], self.dt)
        self.yaw_unwrapper = ContinuousAngle()

        self.left_length_pos = PID(kp=2000.0, ki=0.0, kd=10000.0)
        self.right_length_pos = PID(kp=2000.0, ki=0.0, kd=10000.0)
        self.left_length_vel = PID(kp=100.0, ki=0.0, kd=0.0)
        self.right_length_vel = PID(kp=100.0, ki=0.0, kd=0.0)
        self.roll_pid = PID(kp=2000.0, ki=0.0, kd=1000.0)

        lqr_table = args.lqr_gain_table if args.lqr_gain_table is not None else cfg["control_initial"]["lqr_gain_table"]
        lqr_path = resolve_project_path(lqr_table)
        self.lqr_lengths, self.lqr_gains = self.load_lqr_table(lqr_path)

        self.change_length = ChangeLengthFit()

        control_cfg = cfg["control_initial"]
        target_cfg = control_cfg.get("lqr_target", {})
        self.control_mode = args.control_mode
        self.initial_target_x = args.target_x if args.target_x is not None else target_cfg.get("x_m", 0.0)
        self.initial_target_yaw = args.target_yaw if args.target_yaw is not None else target_cfg.get("yaw_rad", 0.0)
        self.target_x = self.initial_target_x
        self.target_yaw = self.initial_target_yaw
        self.target_pitch = args.target_pitch if args.target_pitch is not None else target_cfg.get("pitch_rad", 0.0)
        self.target_left_leg_theta = args.target_left_leg_theta if args.target_left_leg_theta is not None else target_cfg.get("left_leg_theta_rad", 0.0)
        self.target_right_leg_theta = args.target_right_leg_theta if args.target_right_leg_theta is not None else target_cfg.get("right_leg_theta_rad", 0.0)
        self.initial_target_x_dot = args.target_x_dot if args.target_x_dot is not None else target_cfg.get("x_dot_mps", 0.0)
        self.initial_target_yaw_dot = args.target_yaw_dot if args.target_yaw_dot is not None else target_cfg.get("yaw_dot_radps", 0.0)
        self.target_x_dot = self.initial_target_x_dot
        self.target_yaw_dot = self.initial_target_yaw_dot
        self.reference_governor = None
        if self.control_mode == "keyboard":
            self.reference_governor = LqrReferenceGovernor(self.dt, control_cfg, args)

        self.target_leg_length = args.target_leg_length
        self.last_log_time = -1.0
        self.last_sim_time = data.time

        self.refresh_kinematics()
        if self.target_leg_length is None:
            self.target_leg_length = 0.5 * (self.left_leg.length + self.right_leg.length)

        self.U_lqr = np.zeros((4,1))

        # ---------- UDP 发送初始化 ----------
        self.udp_ip = "127.0.0.1"
        self.udp_port = 12345
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.vofa_tail = b'\x00\x00\x80\x7f'

    def load_lqr_table(self, path):
        text = path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"(?ms)^\s*(0\.\d+)\s*\n"
            r"\s*theta_balance\s*=\s*[-+0-9.eE]+\s*\n"
            r"\s*F_convergence:\s*\n"
            r"\s*(\[\[.*?\]\])"
        )
        samples = []
        for match in pattern.finditer(text):
            leg_length = float(match.group(1))
            matrix_text = match.group(2).replace("[", " ").replace("]", " ")
            values = np.fromstring(matrix_text, sep=" ")
            if values.size != 40:
                raise ValueError(f"Invalid LQR block at leg_length={leg_length}: {values.size} values")
            samples.append((leg_length, values.reshape(4, 10)))
        if not samples:
            raise ValueError(f"No LQR gain blocks found in {path}")
        samples.sort(key=lambda item: item[0])
        lengths = np.array([item[0] for item in samples])
        gains = np.stack([item[1] for item in samples], axis=0)
        return lengths, gains

    def interpolate_lqr_gain(self, leg_length):
        leg_length = float(np.clip(leg_length, self.lqr_lengths[0], self.lqr_lengths[-1]))
        right = int(np.searchsorted(self.lqr_lengths, leg_length, side="right"))
        if right <= 1:
            return self.lqr_gains[0].copy()
        if right >= len(self.lqr_lengths):
            return self.lqr_gains[-1].copy()
        left = right - 1
        ratio = (leg_length - self.lqr_lengths[left]) / (self.lqr_lengths[right] - self.lqr_lengths[left])
        return self.lqr_gains[left] * (1.0 - ratio) + self.lqr_gains[right] * ratio

    def reset_internal_state(self):
        self.left_leg.reset_dynamic_state()
        self.right_leg.reset_dynamic_state()
        self.odometry.reset()
        self.body_odom.reset()
        self.left_length_pos.reset()
        self.right_length_pos.reset()
        self.left_length_vel.reset()
        self.right_length_vel.reset()
        self.roll_pid.reset()
        current_state = self.read_state()
        self.yaw_unwrapper.reset(current_state["euler"][2])
        self.last_log_time = -1.0
        self.target_x = self.initial_target_x
        self.target_yaw = self.initial_target_yaw
        self.target_x_dot = self.initial_target_x_dot
        self.target_yaw_dot = self.initial_target_yaw_dot
        if self.reference_governor is not None:
            self.reference_governor.reset()
            self.target_x = self.reference_governor.x
            self.target_yaw = self.reference_governor.yaw
            self.target_x_dot = self.reference_governor.x_dot
            self.target_yaw_dot = self.reference_governor.yaw_dot
        self.refresh_kinematics()
        if self.args.target_leg_length is None:
            self.target_leg_length = 0.5 * (self.left_leg.length + self.right_leg.length)

    def read_state(self):
        s = self.sensor_names
        quat = sensor_value(self.data, s["quat"])
        euler = quat_to_euler(quat)
        return {
            "quat": quat,
            "euler": euler,
            "gyro": sensor_value(self.data, s["gyro"]),
            "accel": sensor_value(self.data, s["accel"]),
            "base_pos": sensor_value(self.data, s["base_pos"]),
            "base_linvel": sensor_value(self.data, s["base_linvel"]),
            "left_front_pos": sensor_value(self.data, s["left_front_pos"])[0],
            "left_front_vel": sensor_value(self.data, s["left_front_vel"])[0],
            "left_back_pos": sensor_value(self.data, s["left_back_pos"])[0],
            "left_back_vel": sensor_value(self.data, s["left_back_vel"])[0],
            "right_front_pos": sensor_value(self.data, s["right_front_pos"])[0],
            "right_front_vel": sensor_value(self.data, s["right_front_vel"])[0],
            "right_back_pos": sensor_value(self.data, s["right_back_pos"])[0],
            "right_back_vel": sensor_value(self.data, s["right_back_vel"])[0],
            "left_wheel_pos": sensor_value(self.data, s["left_wheel_pos"])[0],
            "left_wheel_vel": sensor_value(self.data, s["left_wheel_vel"])[0],
            "right_wheel_pos": sensor_value(self.data, s["right_wheel_pos"])[0],
            "right_wheel_vel": sensor_value(self.data, s["right_wheel_vel"])[0],
        }

    def refresh_kinematics(self):
        state = self.read_state()
        state["yaw_continuous"] = self.yaw_unwrapper.update(state["euler"][2])
        pitch = state["euler"][1]
        self.left_leg.forward(state["left_front_pos"], state["left_back_pos"], pitch)
        self.right_leg.forward(state["right_front_pos"], state["right_back_pos"], pitch)
        return state

    def build_lqr_state(self, state):
        yaw_w = state["euler"][2]
        acc_b = state["accel"]

        # World-frame Kalman (yaw-aware, for logging/UDP)
        world_acc_x = acc_b[0] * np.cos(yaw_w) - acc_b[1] * np.sin(yaw_w)
        self.odometry.update(
            state["left_wheel_vel"], state["right_wheel_vel"],
            world_acc_x,
            body_vx=state["base_linvel"][0])

        # Body-frame Kalman for LQR (sagittal-plane model).
        # Process input: body-x acceleration (IMU, no yaw rotation needed).
        # Measurement: world velocity projected onto body-forward direction.
        body_vx_meas = (state["base_linvel"][0] * np.cos(yaw_w)
                        + state["base_linvel"][1] * np.sin(yaw_w))
        body_x, body_vx = self.body_odom.update(
            state["left_wheel_vel"], state["right_wheel_vel"],
            acc_b[0],
            body_vx=body_vx_meas)

        pitch = state["euler"][1]
        yaw = state["yaw_continuous"]
        gyro = state["gyro"]
        return np.array([
            [body_x],
            [yaw],
            [pitch],
            [self.left_leg.theta],
            [self.right_leg.theta],
            [body_vx],
            [gyro[2]],
            [gyro[1]],
            [self.left_leg.theta_dot.value],
            [self.right_leg.theta_dot.value]
        ], dtype=float)

    def build_lqr_target(self, e2, e3_l, e3_r):
        return np.array([
            [self.target_x],
            [self.target_yaw],
            [self.target_pitch - e2],
            [self.target_left_leg_theta - e3_l],
            [self.target_right_leg_theta - e3_r],
            [self.target_x_dot],
            [self.target_yaw_dot],
            [0.0],
            [0.0],
            [0.0]
        ], dtype=float)

    def compute_foot_forces(self, state):
        z_acc = state["accel"][2]
        m_leg = 0.353
        g = self.gravity

        l = self.left_leg.length
        l_dot = self.left_leg.length_dot.value
        l_ddot = self.left_leg.length_ddot.value
        theta = self.left_leg.theta
        theta_dot = self.left_leg.theta_dot.value
        theta_ddot = self.left_leg.theta_ddot.value
        z_wheel_l = (z_acc - l_ddot * np.cos(theta) + 2*l_dot*theta_dot*np.sin(theta)
                     + l*theta_ddot*np.sin(theta) + l*theta_dot**2*np.cos(theta))
        F_l = self.left_leg.force
        P_l = F_l * np.cos(theta) + self.U_lqr[0,0] * np.sin(theta) / l if l > 1e-6 else 0.0
        Fn_l = P_l + m_leg * (g + z_wheel_l)

        l = self.right_leg.length
        l_dot = self.right_leg.length_dot.value
        l_ddot = self.right_leg.length_ddot.value
        theta = self.right_leg.theta
        theta_dot = self.right_leg.theta_dot.value
        theta_ddot = self.right_leg.theta_ddot.value
        z_wheel_r = (z_acc - l_ddot * np.cos(theta) + 2*l_dot*theta_dot*np.sin(theta)
                     + l*theta_ddot*np.sin(theta) + l*theta_dot**2*np.cos(theta))
        F_r = self.right_leg.force
        P_r = F_r * np.cos(theta) + self.U_lqr[1,0] * np.sin(theta) / l if l > 1e-6 else 0.0
        Fn_r = P_r + m_leg * (g + z_wheel_r)

        return Fn_l, Fn_r

    def step(self, keyboard_axes=None):
        if self.data.time + 0.5 * self.dt < self.last_sim_time:
            self.reset_internal_state()

        state = self.refresh_kinematics()

        if keyboard_axes is None:
            speed_axis, yaw_axis, hight_axis, jump_pressed = (0.0, 0.0, 0.0, False)
        else:
            speed_axis, yaw_axis, hight_axis, jump_pressed = keyboard_axes

        if self.reference_governor is not None:
            self.target_x, self.target_yaw, self.target_x_dot, self.target_yaw_dot = \
                self.reference_governor.update(speed_axis, yaw_axis)

        self.target_leg_length += hight_axis
        self.target_leg_length = np.clip(self.target_leg_length, 0.12, 0.35)

        x_lqr = self.build_lqr_state(state)
        avg_length = 0.5 * (self.left_leg.length + self.right_leg.length)
        K_raw = self.interpolate_lqr_gain(avg_length)

        e1 = self.change_length.get_e1(avg_length)
        e2 = self.change_length.get_e2(avg_length)
        e3_l = self.change_length.get_e3(avg_length)
        e3_r = self.change_length.get_e3(avg_length)

        target = self.build_lqr_target(e2, e3_l, e3_r)

        left_pos_force = self.left_length_pos.update(self.target_leg_length, self.left_leg.length)
        right_pos_force = self.right_length_pos.update(self.target_leg_length, self.right_leg.length)
        left_vel_force = self.left_length_vel.update(0.0, self.left_leg.length_dot.value)
        right_vel_force = self.right_length_vel.update(0.0, self.right_leg.length_dot.value)
        roll_force = self.roll_pid.update(0.0, state["euler"][0])

        left_pos_force = clip_symmetric(left_pos_force, self.args.length_position_force_limit)
        right_pos_force = clip_symmetric(right_pos_force, self.args.length_position_force_limit)
        left_vel_force = clip_symmetric(left_vel_force, self.args.length_velocity_force_limit)
        right_vel_force = clip_symmetric(right_vel_force, self.args.length_velocity_force_limit)
        roll_force = clip_symmetric(roll_force, self.args.roll_force_limit)

        G_m = 104
        F_l = left_vel_force + roll_force + left_pos_force + G_m * np.cos(self.left_leg.theta)
        F_r = right_vel_force - roll_force + right_pos_force + G_m * np.cos(self.right_leg.theta)

        U = -K_raw @ (x_lqr - target)
        self.U_lqr = U

        Fn_l, Fn_r = self.compute_foot_forces(state)

        K_adj = K_raw.copy()
        e3_l_adj = e3_l
        e3_r_adj = e3_r
        if Fn_l < 10.0:
            K_adj[0, :] = 0.0
            K_adj[2, :] = 0.0
            e3_l_adj = -e3_l
        if Fn_r < 10.0:
            K_adj[1, :] = 0.0
            K_adj[3, :] = 0.0
            e3_r_adj = -e3_r

        target_adj = self.build_lqr_target(e2, e3_l_adj, e3_r_adj)
        U = -K_adj @ (x_lqr - target_adj)

        left_front, left_back = self.left_leg.vmc(F_l, U[0,0], state["left_front_pos"], state["left_back_pos"])
        right_front, right_back = self.right_leg.vmc(F_r, U[1,0], state["right_front_pos"], state["right_back_pos"])

        ctrl = np.array([
            left_front,
            right_front,
            left_back,
            right_back,
            U[2,0],
            U[3,0]
        ], dtype=float)
        ctrl[:4] = np.clip(ctrl[:4], -self.args.joint_torque_limit, self.args.joint_torque_limit)
        ctrl[4:] = np.clip(ctrl[4:], -self.args.wheel_torque_limit, self.args.wheel_torque_limit)

        roll, pitch, yaw_wrapped = state["euler"]
        yaw = state["yaw_continuous"]
        fallen = abs(roll) > self.args.fall_angle or abs(pitch) > self.args.fall_angle
        if fallen:
            ctrl[:] = 0.0

        # 外部扰动
        self.data.xfrc_applied[:] = 0.0
        if self.args.push_duration > 0.0:
            push_end = self.args.push_time + self.args.push_duration
            if self.args.push_time <= self.data.time < push_end:
                self.data.xfrc_applied[self.push_body_id] = np.array([
                    self.args.push_force_x,
                    self.args.push_force_y,
                    self.args.push_force_z,
                    self.args.push_torque_x,
                    self.args.push_torque_y,
                    self.args.push_torque_z
                ], dtype=float)

        self.data.ctrl[:] = ctrl
        mj.mj_step(self.model, self.data)
        self.last_sim_time = self.data.time

        # ---------- UDP 发送 ----------
        # 准备发送数据：状态10 + 目标10 + 左腿长 + 右腿长 + Fn_l + Fn_r = 24 个 float
        send_list = [
            x_lqr[0,0], x_lqr[1,0], x_lqr[2,0], x_lqr[3,0], x_lqr[4,0],
            x_lqr[5,0], x_lqr[6,0], x_lqr[7,0], x_lqr[8,0], x_lqr[9,0],
            target[0,0], target[1,0], target[2,0], target[3,0], target[4,0],
            target[5,0], target[6,0], target[7,0], target[8,0], target[9,0],
            self.left_leg.length,
            self.right_leg.length,
            Fn_l,
            Fn_r,
        ]
        # 调试：确保长度 24 且后 4 个非零
        if len(send_list) != 24:
            raise RuntimeError(f"send_list length is {len(send_list)}, expected 24")
        # 可选：打印后 4 个值，确认非零
        # print(f"send_list[-4:] = {send_list[-4:]}")
        binary_data = struct.pack('<' + 'f' * len(send_list), *send_list)
        self.sock.sendto(binary_data + self.vofa_tail, (self.udp_ip, self.udp_port))

        # ---------- 可选日志打印 ----------
        if self.args.log_every > 0.0 and self.data.time - self.last_log_time >= self.args.log_every:
            self.last_log_time = self.data.time
            print(
                f"t={self.data.time:6.3f} "
                f"roll={roll:+.4f} pitch={pitch:+.4f} yaw={yaw:+.4f} "
                f"bx={x_lqr[0,0]:+.3f} bv={x_lqr[5,0]:+.3f} "
                f"w_x={self.odometry.position:+.3f} "
                f"yawdot={x_lqr[6,0]:+.3f} "
                f"target=[{self.target_x:+.2f},{self.target_x_dot:+.2f},{self.target_yaw:+.2f},{self.target_yaw_dot:+.2f}] "
                f"L={self.left_leg.length:.4f}/{self.right_leg.length:.4f} "
                f"Fn_l={Fn_l:.1f} Fn_r={Fn_r:.1f} "
                f"u=[{ctrl[0]:+.2f}, {ctrl[1]:+.2f}, {ctrl[2]:+.2f}, {ctrl[3]:+.2f}, {ctrl[4]:+.2f}, {ctrl[5]:+.2f}]"
            )

        return not fallen