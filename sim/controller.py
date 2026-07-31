"""轮腿机器人站立控制器（手册 §8 控制管线编排器）。

每控制步管线（对应手册 §13 chassis_task 顺序）：
  Stage 0: 感知（§2） — 传感器读取 + 运动学正解
  Stage 1: 估计（§7） — 10 维 LQR 状态向量构建 + Kalman 融合
  Stage 2: 规划（§4-5, §10）— 键盘调速器 + 增益调度 + 跳跃状态机
  Stage 3: 控制（§4, §8） — LQR 反馈 + PID 腿长/Roll + 重力前馈
  Stage 4: 映射（§6, §9） — VMC 力矩映射 + K 矩阵抑制（离地检测）
  Stage 5: 执行          — 力矩限幅 + 外力施加 + mj_step
  Stage 6: 遥测          — UDP 发送 + 控制台日志
"""

import sys
from pathlib import Path
import socket
import struct
import numpy as np
import mujoco as mj
from utils import (sensor_value, quat_to_euler, clip_symmetric,
                   normalize_angle, ContinuousAngle, approach_value)

# Import Chebyshev-based LQR gain module (手册 §5.6, §12)
_proj_root = Path(__file__).resolve().parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))
from data.lqr_gain_fit import get_K, get_e2, get_e3

from kinematics import FiveLinkLeg       # 手册 §2: 五连杆运动学
from pid import PID                       # 手册 §8: PID 控制器
from kalman import KalmanOdometry         # 手册 §7: Kalman 状态估计
from lqr_governor import LqrReferenceGovernor  # 手册 §4-5: 参考调速器
from ground_contact import GroundContactDetector  # 手册 §9: 地面接触检测


class StandController:
    """轮腿站立控制器，编排 6 阶段控制管线。

    Public API:
      __init__(model, data, cfg, args)
      step(keyboard_axes=None) -> bool
    """

    def __init__(self, model, data, cfg, args):
        # ── MuJoCo 句柄 ──
        self.model = model
        self.data = data
        self.cfg = cfg
        self.args = args
        self.dt = model.opt.timestep
        self.push_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.push_body)
        if self.push_body_id < 0:
            raise ValueError(f"Unknown push body: {args.push_body}")

        # ── 配置读取 ──
        self.sensor_names = cfg["mujoco"]["sensors"]
        self.geometry = cfg["geometry"]
        self.gravity = cfg["simulation"]["gravity_mps2"]
        self.total_mass = cfg["mass_properties"]["total_robot_mass_kg"]
        self.support_force = args.support_force
        if self.support_force is None:
            self.support_force = 0.5 * self.total_mass * self.gravity

        # ── 运动学模块（手册 §2）──
        self.left_leg = FiveLinkLeg("left", self.geometry["five_link"], self.dt)
        self.right_leg = FiveLinkLeg("right", self.geometry["five_link"], self.dt)

        # ── 状态估计模块（手册 §7）──
        self.odometry = KalmanOdometry(self.geometry["wheel"]["radius_m"], self.dt)
        self.body_odom = KalmanOdometry(self.geometry["wheel"]["radius_m"], self.dt)
        self.yaw_unwrapper = ContinuousAngle()

        # ── PID 模块（手册 §8.1-8.2）──
        self.left_length_pos = PID(kp=1500.0, ki=0.0, kd=4000.0)
        self.right_length_pos = PID(kp=1500.0, ki=0.0, kd=4000.0)
        self.left_length_vel = PID(kp=100.0, ki=0.0, kd=0.0)
        self.right_length_vel = PID(kp=100.0, ki=0.0, kd=0.0)
        self.roll_pid = PID(kp=2000.0, ki=0.0, kd=0.0)

        # ── 规划模块（手册 §4-5, §10）──
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
        leg_traj_cfg = control_cfg.get("leg_length_trajectory", {})
        five_link_cfg = self.geometry["five_link"]
        self.leg_length_min = float(leg_traj_cfg.get("min_length_m", five_link_cfg["min_length_m"]))
        self.leg_length_max = float(leg_traj_cfg.get("max_length_m", five_link_cfg["max_length_m"]))
        self.leg_length_rate = abs(float(leg_traj_cfg.get("max_rate_mps", 0.20)))
        self.reference_governor = None
        if self.control_mode == "keyboard":
            self.reference_governor = LqrReferenceGovernor(self.dt, control_cfg, args)

        self.target_leg_length = args.target_leg_length

        # ── 地面接触检测（手册 §9）──
        self.gc = GroundContactDetector(self.dt, args.joint_torque_limit,
                                        args.wheel_torque_limit)

        # ── 初始状态 ──
        self.U_lqr = np.zeros((4, 1))
        self.last_log_time = -1.0
        self.last_sim_time = data.time

        self.refresh_kinematics()
        self._reset_leg_length_targets()

        # ── UDP 遥测 ──
        self.udp_ip = "127.0.0.1"
        self.udp_port = 12345
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.vofa_tail = b'\x00\x00\x80\x7f'

    # ══════════════════════════════════════════════════════════════════
    #  状态重置
    # ══════════════════════════════════════════════════════════════════

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
        self._reset_leg_length_targets()

        self.gc.reset()

    def _reset_leg_length_targets(self):
        """Reset the requested leg-length endpoint and its smoothed trajectory."""
        if self.args.target_leg_length is None:
            target_leg_length = self.leg_length_min  # start from squat (match legwheel)
        else:
            target_leg_length = self.args.target_leg_length
        target_leg_length = float(np.clip(target_leg_length,
                                          self.leg_length_min,
                                          self.leg_length_max))
        self.target_leg_length_command = target_leg_length
        self.target_leg_length = target_leg_length

    # ══════════════════════════════════════════════════════════════════
    #  传感器读取 + 运动学（手册 §2）
    # ══════════════════════════════════════════════════════════════════

    def read_state(self):
        """读取所有 MuJoCo 传感器数据。"""
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
        """传感器 → 五连杆 FK → (L, θ) 及其微分。"""
        state = self.read_state()
        state["yaw_continuous"] = self.yaw_unwrapper.update(state["euler"][2])
        pitch = state["euler"][1]
        self.left_leg.forward(state["left_front_pos"], state["left_back_pos"], pitch)
        self.right_leg.forward(state["right_front_pos"], state["right_back_pos"], pitch)
        return state

    # ══════════════════════════════════════════════════════════════════
    #  状态估计（手册 §7）
    # ══════════════════════════════════════════════════════════════════

    def build_lqr_state(self, state):
        """构造 10 维 LQR 状态向量 x ∈ ℝ¹⁰（手册 §3.1）。"""
        yaw_w = state["euler"][2]
        acc_b = state["accel"]
        gyro = state["gyro"]
        yaw = state["yaw_continuous"]

        # World-frame Kalman (for odometry logging)
        world_acc_x = acc_b[0] * np.cos(yaw_w) - acc_b[1] * np.sin(yaw_w)
        self.odometry.update(
            state["left_wheel_vel"], state["right_wheel_vel"],
            world_acc_x,
            body_vx=state["base_linvel"][0])

        # Body-frame Kalman for LQR (sagittal-plane model)
        body_vx_meas = (state["base_linvel"][0] * np.cos(yaw_w)
                        + state["base_linvel"][1] * np.sin(yaw_w))
        body_x, body_vx = self.body_odom.update(
            state["left_wheel_vel"], state["right_wheel_vel"],
            acc_b[0],
            body_vx=body_vx_meas)

        pitch = state["euler"][1]
        return np.array([
            [body_x],                       # x[0]: x
            [yaw],                          # x[1]: ψ
            [pitch],                        # x[2]: θ
            [self.left_leg.theta],          # x[3]: θ_L
            [self.right_leg.theta],         # x[4]: θ_R
            [body_vx],                      # x[5]: ẋ
            [gyro[2]],                      # x[6]: ψ̇
            [gyro[1]],                      # x[7]: θ̇
            [self.left_leg.theta_dot.value],  # x[8]: θ̇_L
            [self.right_leg.theta_dot.value], # x[9]: θ̇_R
        ], dtype=float)

    def build_lqr_target(self, e2, e3_l, e3_r):
        """构造 LQR 目标状态向量（含平衡偏移量，手册 §5.4）。"""
        x_ref = self.target_x
        x_dot_ref = self.target_x_dot
        return np.array([
            [x_ref],
            [self.target_yaw],
            [self.target_pitch - e2],
            [self.target_left_leg_theta - e3_l],
            [self.target_right_leg_theta - e3_r],
            [x_dot_ref],
            [self.target_yaw_dot],
            [0.0],
            [0.0],
            [0.0]
        ], dtype=float)

    # ══════════════════════════════════════════════════════════════════
    #  主控制步（手册 §13 chassis_task 管线编排）
    # ══════════════════════════════════════════════════════════════════

    def step(self, keyboard_axes=None):
        """单控制步：感知 → 估计 → 规划 → 控制 → 映射 → 执行（手册 §13）。"""
        if self.data.time + 0.5 * self.dt < self.last_sim_time:
            self.reset_internal_state()

        # ── Stage 0: 感知（§2）──
        state = self._read_sensors(keyboard_axes)

        # ── Stage 1: 估计（§7）──
        x_lqr = self.build_lqr_state(state)

        # ── Stage 2: 规划（§4-5）──
        self._update_navigation(keyboard_axes, state["yaw_continuous"],
                                current_x=float(x_lqr[0, 0]))
        K_raw = get_K(self.left_leg.length, self.right_leg.length)  # 2D Chebyshev
        e2 = get_e2(self.left_leg.length, self.right_leg.length)
        e3_l, e3_r = get_e3(self.left_leg.length, self.right_leg.length)
        target = self.build_lqr_target(e2, e3_l, e3_r)

        # ── Stage 3: 控制（§4, §8）──
        F_l, F_r = self._compute_forces(x_lqr, target, K_raw, state)

        # ── Stage 4: 映射（§6, §9）──
        ctrl = self._map_to_actuators(F_l, F_r, x_lqr, K_raw,
                                      e2, e3_l, e3_r, state)

        # ── Stage 5: 执行 ──
        self._apply_control(ctrl, state, x_lqr)

        # ── Stage 6: 遥测 ──
        self._send_telemetry(x_lqr, target, state, ctrl)

        return True

    # ══════════════════════════════════════════════════════════════════
    #  Stage 0 私有方法
    # ══════════════════════════════════════════════════════════════════

    def _read_sensors(self, keyboard_axes):
        """传感器读取 + FK。"""
        state = self.refresh_kinematics()

        if keyboard_axes is None:
            speed_axis, yaw_axis, hight_axis = (0.0, 0.0, 0.0)
        else:
            speed_axis, yaw_axis, hight_axis = keyboard_axes

        # Store axes for later pipeline stages
        state["_speed_axis"] = speed_axis
        state["_yaw_axis"] = yaw_axis
        state["_hight_axis"] = hight_axis
        return state

    # ══════════════════════════════════════════════════════════════════
    #  Stage 2 私有方法
    # ══════════════════════════════════════════════════════════════════

    def _update_navigation(self, keyboard_axes, current_yaw=None, current_x=None):
        """键盘->参考调速器 + 腿长伸缩限幅。"""
        if keyboard_axes is None:
            speed_axis, yaw_axis, hight_axis = (0.0, 0.0, 0.0)
        else:
            speed_axis, yaw_axis, hight_axis = keyboard_axes

        if self.reference_governor is not None:
            self.target_x, self.target_yaw, self.target_x_dot, self.target_yaw_dot = \
                self.reference_governor.update(speed_axis, yaw_axis,
                                               current_yaw=current_yaw,
                                               current_x=current_x)

        # Keyboard height input changes the requested endpoint; the actual
        # leg-length setpoint follows it through a slew-limited trajectory.
        self.target_leg_length_command = float(np.clip(
            self.target_leg_length_command + hight_axis,
            self.leg_length_min,
            self.leg_length_max,
        ))
        self.target_leg_length = self.target_leg_length_command

    def _compute_forces(self, x_lqr, target, K_raw, state):
        """PID 力控综合。

        Returns
        -------
        F_l, F_r : float    — 左右腿轴向力
        """
        G_m = 0.5 * self.total_mass * self.gravity  # 重力前馈（半车质量/腿）

        # ── PID 位置环（手册 §8.1）──
        left_pos_force = self.left_length_pos.update(
            self.target_leg_length, self.left_leg.length)
        right_pos_force = self.right_length_pos.update(
            self.target_leg_length, self.right_leg.length)
        left_pos_force = clip_symmetric(left_pos_force, self.args.length_position_force_limit)
        right_pos_force = clip_symmetric(right_pos_force, self.args.length_position_force_limit)

        # ── PID 速度环（阻尼，手册 §8.1）──
        left_vel_force = self.left_length_vel.update(0.0, self.left_leg.length_dot.value)
        right_vel_force = self.right_length_vel.update(0.0, self.right_leg.length_dot.value)
        left_vel_force = clip_symmetric(left_vel_force, self.args.length_velocity_force_limit)
        right_vel_force = clip_symmetric(right_vel_force, self.args.length_velocity_force_limit)

        # ── Roll 稳定（手册 §8.2）──
        roll_force = self.roll_pid.update(0.0, state["euler"][0])
        roll_force = clip_symmetric(roll_force, self.args.roll_force_limit)

        # ── 力控综合（手册 §8.3）──
        # Gravity feedforward: G_m * cos(theta)  -- matched to legwheel LQR training model
        left_cos = float(np.clip(np.cos(self.left_leg.theta), 1e-3, 1.0))
        right_cos = float(np.clip(np.cos(self.right_leg.theta), 1e-3, 1.0))
        F_l = left_vel_force + roll_force + left_pos_force + G_m * left_cos
        F_r = right_vel_force - roll_force + right_pos_force + G_m * right_cos

        return F_l, F_r

    def compute_foot_forces(self, state):
        """地面法向力估计（手册 §9-principle）。"""
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
        P_l = self.left_leg.force * np.cos(theta) + self.U_lqr[0, 0] * np.sin(theta) / l if l > 1e-6 else 0.0
        Fn_l = P_l + m_leg * (g + z_wheel_l)

        l = self.right_leg.length
        l_dot = self.right_leg.length_dot.value
        l_ddot = self.right_leg.length_ddot.value
        theta = self.right_leg.theta
        theta_dot = self.right_leg.theta_dot.value
        theta_ddot = self.right_leg.theta_ddot.value
        z_wheel_r = (z_acc - l_ddot * np.cos(theta) + 2*l_dot*theta_dot*np.sin(theta)
                     + l*theta_ddot*np.sin(theta) + l*theta_dot**2*np.cos(theta))
        P_r = self.right_leg.force * np.cos(theta) + self.U_lqr[1, 0] * np.sin(theta) / l if l > 1e-6 else 0.0
        Fn_r = P_r + m_leg * (g + z_wheel_r)

        return Fn_l, Fn_r

    # ══════════════════════════════════════════════════════════════════
    #  Stage 4 私有方法
    # ══════════════════════════════════════════════════════════════════

    def _map_to_actuators(self, F_l, F_r, x_lqr, K_raw,
                          e2, e3_l, e3_r, state):
        """VMC 力矩映射 + 离地 K 抑制 + 力矩限幅（手册 §6, §9）。

        Returns
        -------
        ctrl : (6,) ndarray  — [LF, RF, LB, RB, WL, WR]
        """
        # ── 地面接触检测 → K 行抑制（手册 §9）──
        Fn_l, Fn_r = self.compute_foot_forces(state)

        K_adj = K_raw.copy()
        e3_l_adj = e3_l
        e3_r_adj = e3_r

        zwL, zwR, zlL, zlR = False, False, False, False  # TODO: re-enable ground contact detection
        # self.gc.get_K_suppression_mask()
        if zwL: K_adj[2, :] = 0.0
        if zwR: K_adj[3, :] = 0.0
        if zlL: K_adj[0, :] = 0.0; e3_l_adj = -e3_l
        if zlR: K_adj[1, :] = 0.0; e3_r_adj = -e3_r

        target_adj = self.build_lqr_target(e2, e3_l_adj, e3_r_adj)
        U_adj = -K_adj @ (x_lqr - target_adj)
        self.U_lqr = U_adj

        # ── VMC 力矩映射（手册 §6）──
        # +12N leg self-weight compensation (matched to legwheel)
        left_front, left_back = self.left_leg.vmc(
            F_l + 12.0, U_adj[0, 0], state["left_front_pos"], state["left_back_pos"])
        right_front, right_back = self.right_leg.vmc(
            F_r + 12.0, U_adj[1, 0], state["right_front_pos"], state["right_back_pos"])

        ctrl = np.array([left_front, right_front,
                         left_back, right_back,
                         U_adj[2, 0], U_adj[3, 0]], dtype=float)
        ctrl[:4] = np.clip(ctrl[:4], -self.args.joint_torque_limit, self.args.joint_torque_limit)
        ctrl[4:] = np.clip(ctrl[4:], -self.args.wheel_torque_limit, self.args.wheel_torque_limit)

        return ctrl

    # ══════════════════════════════════════════════════════════════════
    #  Stage 5 私有方法
    # ══════════════════════════════════════════════════════════════════

    def _apply_control(self, ctrl, state, x_lqr):
        """地面接触检测更新 + 外力施加 + MuJoCo 物理步进。"""
        # Ground contact detector update for next step
        left_jt = np.array([ctrl[0], ctrl[2]])
        right_jt = np.array([ctrl[1], ctrl[3]])
        Fn_l, Fn_r = self.compute_foot_forces(state)
        self.gc.update(Fn_l, Fn_r,
                       state["left_wheel_vel"], state["right_wheel_vel"],
                       x_lqr[5, 0], state["accel"][2],
                       left_jt, right_jt,
                       ctrl[4], ctrl[5])

        # External push force
        self.data.xfrc_applied[:] = 0.0
        if self.args.push_duration > 0.0:
            push_end = self.args.push_time + self.args.push_duration
            if self.args.push_time <= self.data.time < push_end:
                self.data.xfrc_applied[self.push_body_id] = np.array([
                    self.args.push_force_x, self.args.push_force_y, self.args.push_force_z,
                    self.args.push_torque_x, self.args.push_torque_y, self.args.push_torque_z,
                ], dtype=float)

        self.data.ctrl[:] = ctrl
        mj.mj_step(self.model, self.data)
        self.last_sim_time = self.data.time

    # ══════════════════════════════════════════════════════════════════
    #  Stage 6 私有方法
    # ══════════════════════════════════════════════════════════════════

    def _send_telemetry(self, x_lqr, target, state, ctrl):
        """UDP 发送（24 floats） + 控制台定时日志。"""
        roll, pitch, yaw_wrapped = state["euler"]
        yaw = state["yaw_continuous"]
        Fn_l, Fn_r = self.compute_foot_forces(state)

        # UDP packet: 24 floats
        send_list = [
            x_lqr[0, 0], x_lqr[1, 0], x_lqr[2, 0], x_lqr[3, 0], x_lqr[4, 0],
            x_lqr[5, 0], x_lqr[6, 0], x_lqr[7, 0], x_lqr[8, 0], x_lqr[9, 0],
            target[0, 0], target[1, 0], target[2, 0], target[3, 0], target[4, 0],
            target[5, 0], target[6, 0], target[7, 0], target[8, 0], target[9, 0],
            self.left_leg.length, self.right_leg.length,
            Fn_l, Fn_r,
        ]
        if len(send_list) != 24:
            raise RuntimeError(f"send_list length is {len(send_list)}, expected 24")
        binary_data = struct.pack('<' + 'f' * len(send_list), *send_list)
        self.sock.sendto(binary_data + self.vofa_tail, (self.udp_ip, self.udp_port))

        # Console log
        if self.args.log_every > 0.0 and self.data.time - self.last_log_time >= self.args.log_every:
            self.last_log_time = self.data.time
            print(
                f"t={self.data.time:6.3f} "
                f"roll={roll:+.4f} pitch={pitch:+.4f} yaw={yaw:+.4f} "
                f"bx={x_lqr[0, 0]:+.3f} bv={x_lqr[5, 0]:+.3f} "
                f"w_x={self.odometry.position:+.3f} "
                f"yawdot={x_lqr[6, 0]:+.3f} "
                f"target=[{self.target_x:+.2f},{self.target_x_dot:+.2f},{self.target_yaw:+.2f},{self.target_yaw_dot:+.2f}] "
                f"L={self.left_leg.length:.4f}/{self.right_leg.length:.4f} "
                f"Fn_l={Fn_l:.1f} Fn_r={Fn_r:.1f} "
                f"u=[{ctrl[0]:+.2f}, {ctrl[1]:+.2f}, {ctrl[2]:+.2f}, {ctrl[3]:+.2f}, {ctrl[4]:+.2f}, {ctrl[5]:+.2f}]"
            )
