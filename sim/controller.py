"""轮腿控制器，运行平衡、跳跃、翻正和台阶接管逻辑。"""

import sys
from pathlib import Path
import socket
import struct
import numpy as np
import mujoco as mj
from utils import (sensor_value, quat_to_euler, clip_symmetric,
                   normalize_angle, ContinuousAngle, approach_value)

_proj_root = Path(__file__).resolve().parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))
from data.lqr_gain_fit import get_K, get_e3, get_u_eq

from cad_kinematics import CadSixBarLeg as FiveLinkLeg
from pid import PID
from kalman import ChassisOdometry, KalmanOdometry
from lqr_governor import LqrReferenceGovernor
from jump_dynamics import JumpMomentumModel
from self_righting import SelfRighting, rotation
from step_climb import StepClimb
from recovery_log import RecoveryLog
from ground_contact import GroundContactDetector


class StandController:
    """读取传感器、估计状态，并按运动模式分配六个电机的输出。"""

    def __init__(self, model, data, cfg, args):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.args = args
        self.dt = cfg["simulation"]["control_period_s"]
        self.push_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.push_body)
        if self.push_body_id < 0:
            raise ValueError(f"Unknown push body: {args.push_body}")
        self.wheel_contact_geom_ids = (
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "left_wheel_contact"),
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "right_wheel_contact"),
        )
        if any(geom_id < 0 for geom_id in self.wheel_contact_geom_ids):
            raise ValueError("Wheel contact geoms are required for liftoff torque suppression")

        self.sensor_names = cfg["mujoco"]["sensors"]
        self.geometry = cfg["geometry"]
        self.wheel_radius = self.geometry["wheel"]["radius_m"]
        self.half_track = self.geometry["chassis"]["half_track_m"]
        self.gravity = cfg["simulation"]["gravity_mps2"]
        mass_cfg = cfg["mass_properties"]
        self.total_mass = mass_cfg["total_robot_mass_kg"]
        self.body_mass = mass_cfg["body"]["mass_kg"]
        self.side_leg_mass = (
            sum(float(v) for v in mass_cfg["links"].values())
            + float(mass_cfg["wheel"]["mass_kg"])
        )
        self.support_force = args.support_force
        if self.support_force is None:
            self.support_force = (
                0.5 * self.body_mass * self.gravity
                + self.side_leg_mass * self.gravity
            )

        self.left_leg = FiveLinkLeg("left", self.geometry["five_link"], self.dt)
        self.right_leg = FiveLinkLeg("right", self.geometry["five_link"], self.dt)
        self.gas_springs = {}
        for side in ("left", "right"):
            spring = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{side}_gas_rod_joint")
            if spring < 0 or model.jnt_stiffness[spring] == 0.0:
                continue
            # 由模型中的轴向弹簧参数读取 CAD 零位下的推力。
            stiffness = model.jnt_stiffness[spring]
            qpos = model.jnt_qposadr[spring]
            reference_force = stiffness * (model.qpos_spring[qpos] - model.qpos0[qpos])
            self.gas_springs[side] = (stiffness, reference_force)

        self.odometry = KalmanOdometry(self.dt)
        self.body_odom = ChassisOdometry(self.dt, self.half_track)
        self.yaw_unwrapper = ContinuousAngle()

        ll_pid = cfg["control_initial"]["leg_length_pid"]
        self.left_length_pos = PID(kp=ll_pid["position_kp"], ki=ll_pid["position_ki"], kd=ll_pid["position_kd"])
        self.right_length_pos = PID(kp=ll_pid["position_kp"], ki=ll_pid["position_ki"], kd=ll_pid["position_kd"])
        self.left_length_vel = PID(kp=ll_pid["velocity_kp"], ki=ll_pid["velocity_ki"], kd=ll_pid["velocity_kd"])
        self.right_length_vel = PID(kp=ll_pid["velocity_kp"], ki=ll_pid["velocity_ki"], kd=ll_pid["velocity_kd"])
        self.support_height_pos = PID(kp=ll_pid["position_kp"], ki=ll_pid["position_ki"],
                                      kd=ll_pid["position_kd"])
        self.support_height_vel = PID(kp=ll_pid["velocity_kp"], ki=ll_pid["velocity_ki"],
                                      kd=ll_pid["velocity_kd"])

        # Roll 控制：高度调度 PD + 速率阻尼 + 斜坡积分（命题 7.2 / 定理 3.1）
        roll_cfg = cfg["control_initial"]["roll_control"]
        self.roll_inertia = cfg["mass_properties"]["body"]["inertia_kg_m2"][0]  # I_xx
        self.wc_roll = float(roll_cfg["wc_radps"])
        self.zeta_roll = float(roll_cfg["zeta"])
        self.ki_slope = float(roll_cfg["ki_slope"])
        self.roll_integral = 0.0

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
        self.leg_length_default = float(np.clip(
            leg_traj_cfg.get("default_length_m", 0.22),
            self.leg_length_min,
            self.leg_length_max,
        ))
        self.reference_governor = None
        if self.control_mode == "keyboard":
            self.reference_governor = LqrReferenceGovernor(self.dt, control_cfg, args)

        self.target_leg_length = args.target_leg_length

        jump_cfg = control_cfg.get("jump", {})
        self.jump_force = float(jump_cfg.get("force_per_leg_n", 500.0))
        self.jump_max_thrust_time = float(jump_cfg.get("max_thrust_time_s", 0.25))
        self.jump_thrust_end_length = float(jump_cfg.get("thrust_end_length_m", 0.39))
        self.jump_min_tuck_time = float(jump_cfg.get("min_retract_duration_s", 0.14))
        # 收腿目标留出轮胎与车身的碰撞余量。
        self.jump_retract_length = float(jump_cfg.get("retract_length_m", 0.123))
        self.jump_retract_duration = float(jump_cfg.get("retract_duration_s", 0.18))
        self.jump_landing_duration = float(jump_cfg.get("landing_duration_s", 0.35))
        self.jump_momentum_gain = float(jump_cfg.get("takeoff_momentum_gain", 50.0))
        self.jump_length_kp = float(jump_cfg.get("length_kp_npm", 2500.0))
        self.jump_length_kd = float(jump_cfg.get("length_kd_nspm", 80.0))
        self.jump_wheel_speed_limit = float(jump_cfg.get("wheel_speed_limit_radps", 180.0))
        self._jump_momentum = JumpMomentumModel(self.left_leg, self.right_leg)
        self.recovery = SelfRighting(self._jump_momentum, control_cfg.get("self_righting", {}), args.joint_torque_limit, args.wheel_torque_limit, self.wheel_radius)
        guide_geometry = np.array([
            np.r_[model.body_pos[model.geom_bodyid[i]] + model.geom_pos[i], model.geom_size[i, :2]]
            for i in range(model.ngeom)
            if model.geom_contype[i] and model.geom(i).name.startswith('guide_wheel_')
        ])
        hip_torque_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, name + '_hip_torque')
                          for name in ('left_front', 'right_front', 'left_rear', 'right_rear')]
        self.climb_torque_adr = model.sensor_adr[hip_torque_ids] if min(hip_torque_ids) >= 0 else None
        self.step_climb = StepClimb(self._jump_momentum, self.wheel_radius, self.total_mass,
                                  self.gravity, guide_geometry, args.joint_torque_limit, self.dt,
                                  wheel_torque_limit=args.wheel_torque_limit)
        self.previous_motor_ctrl = np.zeros(model.nu)
        self.balance_controller = "lqr"
        self.active_balance_controller = None
        self._mpc = None
        self._balance_switch_offset = np.zeros(model.nu)
        self._balance_switch_time = -np.inf
        self._wheel_contact_ticks = np.zeros(2, dtype=int)
        self._wheel_support_confirmed = np.zeros(2, dtype=bool)
        self._wheel_torque_scale = np.zeros(2)
        self.set_balance_controller(getattr(args, "balance_controller", "lqr"))
        self.recovery_log = RecoveryLog(_proj_root / "output" / "recovery_logs", model, self.wheel_contact_geom_ids[0])
        self.recovery.trace_enabled = self.recovery.enabled
        wheel_dofs = [model.jnt_dofadr[model.joint(side + "_wheel_joint").id]
                      for side in ("left", "right")]
        self._jump_wheel_damping = model.dof_damping[wheel_dofs].copy()
        self._jump_vz = 0.0
        self._jump_height = 0.0
        self._jump_sync_wheels = False
        self._jump_trajectory = None
        self._jump_wheel_reference = None
        self._jump_air_time = self._jump_contact_time = 0.0
        self._jump_ground_time = 0.0
        self._jump_request_until = -np.inf
        self._jump_return_length = self.leg_length_default
        self._jump_returning = False
        self.jump_state = "idle"
        self.jump_start_time = 0.0
        self.jump_retract_start_time = 0.0
        self.jump_airborne_seen = False

        slip_cfg = cfg["control_initial"].get("slip_detection", {})
        self.gc = GroundContactDetector(self.dt, args.joint_torque_limit,
                                        args.wheel_torque_limit,
                                        self.wheel_radius, self.half_track,
                                        dby_threshold=slip_cfg.get("dby_threshold", 0.5),
                                        dvb_threshold=slip_cfg.get("dvb_threshold", 0.3),
                                        dvw_threshold=slip_cfg.get("dvw_threshold", 0.02),
                                        gravity=self.gravity, static_load=.5*self.total_mass*self.gravity)

        self.U_lqr = np.zeros((4, 1))
        self.last_log_time = -1.0
        self.last_sim_time = data.time

        self.refresh_kinematics()
        self._reset_leg_length_targets()

        self.udp_ip = "127.0.0.1"
        self.udp_port = 12345
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.vofa_tail = b'\x00\x00\x80\x7f'


    def set_balance_controller(self, mode):
        """选择平衡控制器，专用动作完成后继续使用所选控制器。"""
        if mode not in ("lqr", "mpc"):
            raise ValueError(f"未知平衡控制器 {mode}")
        if mode == "mpc" and self._mpc is None:
            from balance_mpc import BalanceMPC
            self._mpc = BalanceMPC(_proj_root / "data", self.dt)
        if mode != self.balance_controller:
            self._reset_mpc_plans()
        self.balance_controller = mode

    def _reset_mpc_plans(self):
        if self._mpc is not None:
            self._mpc.reset()

    def reset_internal_state(self):
        self.active_balance_controller = None
        self._balance_switch_time = -np.inf
        self._reset_mpc_plans()
        self._pending_jump = False
        self._wheel_contact_ticks.fill(0)
        self._wheel_support_confirmed.fill(False)
        self._wheel_torque_scale.fill(0.)
        self.recovery_log.reset()
        self.step_climb.reset()
        self.recovery.reset()
        self.previous_motor_ctrl.fill(0.)
        self.left_leg.reset_dynamic_state()
        self.right_leg.reset_dynamic_state()
        self.odometry.reset()
        self.body_odom.reset()
        self.left_length_pos.reset()
        self.right_length_pos.reset()
        self.left_length_vel.reset()
        self.right_length_vel.reset()
        self.support_height_pos.reset()
        self.support_height_vel.reset()
        self.roll_integral = 0.0
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
        self.gc.kalman_r_vel = .5
        self.gc.kalman_q_vel = .01
        self.U_lqr.fill(0.)
        self._last_control_time = None
        self.last_sim_time = self.data.time
        for leg in (self.left_leg,self.right_leg):
            for derivative in (leg.length_ddot,leg.theta_ddot):
                if hasattr(derivative,"prev_prev"):del derivative.prev_prev
        self.jump_state = "idle"
        self.jump_start_time = self.data.time
        self.jump_retract_start_time = self.data.time
        self.jump_airborne_seen = False
        self._jump_vz = 0.0
        self._jump_height = 0.0
        self._jump_sync_wheels = False
        self._jump_trajectory = None
        self._jump_wheel_reference = None
        self._jump_air_time = self._jump_contact_time = 0.0
        self._jump_request_until = -np.inf
        self._jump_return_length = self.target_leg_length_command
        self._jump_returning = False
        self._jump_ground_time = 0.0

    def _reset_leg_length_targets(self):
        """Reset the requested leg-length endpoint and its smoothed trajectory."""
        if self.args.target_leg_length is None:
            target_leg_length = self.leg_length_default
        else:
            target_leg_length = self.args.target_leg_length
        target_leg_length = float(np.clip(target_leg_length,
                                          self.leg_length_min,
                                          self.leg_length_max))
        self.target_leg_length_command = target_leg_length
        self.target_leg_length = target_leg_length


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
        """关节 FK + IMU pitch → 世界竖直参考腿角及其微分。"""
        state = self.read_state()
        state["yaw_continuous"] = self.yaw_unwrapper.update(state["euler"][2])
        pitch = state["euler"][1]
        gas_hip_torque = np.zeros(4)
        for i, (side, leg) in enumerate((("left", self.left_leg), ("right", self.right_leg))):
            front = state[f"{side}_front_pos"]
            back = state[f"{side}_back_pos"]
            leg.forward(front, back, pitch)
            if side in self.gas_springs:
                gas_hip_torque[i], gas_hip_torque[i+2] = leg.gas_spring_torque(
                    front, back, *self.gas_springs[side])
        state["gas_hip_torque_nm"] = gas_hip_torque
        return state


    def build_lqr_state(self, state):
        """按增益表顺序构造十状态；位移与纵向速度均以髋轴中点为参考。"""
        acc_b = state["accel"]
        # 将 IMU 比力投影到水平前向轴，消除竖直重力分量。
        body_to_world = rotation(state["quat"])
        horizontal_forward = body_to_world[:2, 0]
        horizontal_length = np.linalg.norm(horizontal_forward)
        horizontal_acc = 0.0
        if horizontal_length > 1.e-6:
            horizontal_acc = horizontal_forward @ (body_to_world[:2] @ acc_b) / horizontal_length
        state["horizontal_accel"] = horizontal_acc
        gyro = state["gyro"]
        yaw = state["yaw_continuous"]
        wheel_body_vx = self.wheel_forward_velocity(state)
        state["wheel_body_vx"] = wheel_body_vx

        # 原始轮速里程计只用于诊断，打滑和离地会造成偏差。
        self.odometry.update(wheel_body_vx, horizontal_acc)

        # 用闭链运动学将轮速换算成髋轴纵向速度；失去可靠接触时暂停相应轮速观测。
        contacts = state["_wheel_contacts"]
        flight_odom = ((self.step_climb.phase != "idle" and not self.step_climb.wheel_odometry_valid)
                       or (self.recovery.active and self.recovery.phase not in ("rise", "blend"))
                       or self.jump_state in ("retract", "extend")
                       or (self.jump_state == "landing" and self.data.time - self._jump_trajectory[0] < .08))
        if flight_odom:
            contacts = (False, False)
        body_x, body_vx = self.body_odom.update(
            self._wheel_odom_to_body_vx(state), horizontal_acc, gyro[2], contacts)

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

    def wheel_forward_velocity(self, state):
        """Wheel-joint forward velocity in the body sagittal direction."""
        return 0.5 * self.wheel_radius * (
            state["left_wheel_vel"] + state["right_wheel_vel"]
        )

    def _wheel_odom_to_body_vx(self, state):
        """分别求左右轮对应的髋轴纵向速度，单位 m/s。"""
        velocities = []
        for side, leg in (("left", self.left_leg), ("right", self.right_leg)):
            velocities.append(self.wheel_radius*(state[side+"_wheel_vel"]+leg.shank_rate)
                              + leg.length_dot.value*np.sin(leg.theta)
                              + leg.length*np.cos(leg.theta)*leg.theta_dot.value)
        return velocities

    def build_lqr_target(self, e3_l, e3_r):
        """构造水平机身、腿角含平衡偏移的 LQR 目标状态。"""
        x_ref = self.target_x
        x_dot_ref = self.target_x_dot
        return np.array([
            [x_ref],
            [self.target_yaw],
            [self.target_pitch],
            [self.target_left_leg_theta + e3_l],
            [self.target_right_leg_theta + e3_r],
            [x_dot_ref],
            [self.target_yaw_dot],
            [0.0],
            [0.0],
            [0.0]
        ], dtype=float)


    def step(self, keyboard_axes=None, viewer_perturb=None):
        """按控制周期更新控制输出，独立运行时推进 MuJoCo 物理步。"""
        if self.data.time + 0.5 * self.dt < self.last_sim_time:
            self.reset_internal_state()

        # Preserve an edge-triggered jump arriving between 500 Hz control ticks.
        self._pending_jump = getattr(self, "_pending_jump", False) or self._unpack_keyboard_axes(keyboard_axes)[3]
        # Keep the original 500 Hz controller independent of 1 kHz physics.
        previous_control = getattr(self, "_last_control_time", None)
        if previous_control is not None and 0.0 <= self.data.time - previous_control < self.dt - 1e-10:
            if not getattr(self, "native_managed", False):
                mj.mj_step(self.model, self.data)
            return True
        self._last_control_time = self.data.time
        if self._pending_jump:
            keyboard_axes = list(keyboard_axes) if keyboard_axes is not None else [0., 0., 0., False]
            while len(keyboard_axes) < 4: keyboard_axes.append(False)
            keyboard_axes[3] = True
            self._pending_jump = False

        state = self._read_sensors(keyboard_axes)

        previous_climb_phase = self.step_climb.phase
        armed = (not self.recovery.active and self.jump_state == "idle" and
                 not state["_jump_requested"] and self.data.time > self._jump_request_until and
                 self.target_leg_length_command >= self.leg_length_max-.015)
        if self.climb_torque_adr is not None and self.step_climb.guides.size:
            climb_ctrl = self.step_climb.update(state, self.data.sensordata[self.climb_torque_adr],
                                                self.body_odom.velocity, armed)
            if climb_ctrl is not None:
                x_lqr = self.build_lqr_state(state)
                self.U_lqr.fill(0.)
                self.active_balance_controller = None
                self._balance_switch_time = -np.inf
                self._reset_mpc_plans()
                self._apply_control(climb_ctrl, state, x_lqr, viewer_perturb)
                self._send_telemetry(x_lqr, self.build_lqr_target(0., 0.), state, climb_ctrl)
                return True
        climb_capture = self.step_climb.phase == "capture"
        if (climb_capture and previous_climb_phase != "capture") or (
                not climb_capture and previous_climb_phase != "idle"):
            self.target_leg_length_command = self.target_leg_length = float(np.clip(
                (self.left_leg.length+self.right_leg.length)/2, self.leg_length_min, self.leg_length_max))
            self.roll_integral = 0.
            if self.reference_governor is not None:
                gov = self.reference_governor
                gov.x, gov.yaw = self.body_odom.position, state["yaw_continuous"]
                gov.x_dot = gov.yaw_dot = 0.
            if climb_capture:
                self.climb_capture_offset = None
        if previous_climb_phase != "idle":
            keyboard_axes = (*self._unpack_keyboard_axes(keyboard_axes)[:3], False)

        # Recovery receives only sensor samples and the intended jump mode.
        wheel_load = None
        support_on_ground = (self.recovery.phase == 'establish_support'
                             and self.recovery.support_action in ('ground', 'align'))
        if self.recovery.phase == 'rear_flip' or (support_on_ground and self.recovery.rear_flip_roll):
            wheel_load = (self.gc.fn_left, self.gc.fn_right)
        elif support_on_ground or self.recovery.phase == 'rise':
            wheel_load = (self.gc.fn_left if not self.gc.left_airborne else 0.,
                          self.gc.fn_right if not self.gc.right_airborne else 0.)
        recovery_ctrl = None
        if self.step_climb.defer_recovery:
            # 直立且刚失重时，先让台阶判据确认有无导轮支撑。
            self.recovery.quiet = 0.
        elif not climb_capture:
            recovery_ctrl = self.recovery.update(state, self.dt,
                jumping=self.jump_state != "idle" and self.data.time-self.jump_start_time < 2.0,
                trace_idle=self.recovery_log.wants_idle(self.data.time), wheel_load=wheel_load)
        if recovery_ctrl is not None:
            self.jump_state = "idle"
            self._jump_returning = False
            self._jump_request_until = -np.inf
            self._pending_jump = False
            self._jump_trajectory = None
            if self.recovery.handoff:
                self.capture_motor_ctrl = self.previous_motor_ctrl.copy()
                self.capture_start_lengths = np.array([self.left_leg.length, self.right_leg.length])
                self.capture_torque_offset = None
                self.odometry.reset()
                self.body_odom.reset()
                self.gc.kalman_r_vel = .5
                self.gc.kalman_q_vel = .01
                self.roll_integral = 0.
                for pid in (self.left_length_pos,self.right_length_pos,self.left_length_vel,self.right_length_vel):pid.reset()
                if self.reference_governor is not None:
                    self.reference_governor.x = 0.
                    self.reference_governor.yaw = state["yaw_continuous"]
                    self.reference_governor.x_dot = self.reference_governor.yaw_dot = 0.
            self.target_x = self.body_odom.position
            self.target_yaw = state["yaw_continuous"]
            self.target_x_dot = self.target_yaw_dot = 0.
            if not self.recovery.active and self.reference_governor is not None:
                self.reference_governor.x = self.target_x
                self.reference_governor.yaw = self.target_yaw
            self.target_leg_length_command = self.target_leg_length = self.recovery.capture_length
            keyboard_axes = (0.,0.,0.,False)

        x_lqr = self.build_lqr_state(state)

        if climb_capture:
            self.target_x = float(x_lqr[0, 0]) + .05*self.step_climb.direction
            self.target_x_dot = .9*self.step_climb.direction
            self.target_yaw = state["yaw_continuous"]
            self.target_yaw_dot = 0.
        elif recovery_ctrl is None:
            self._update_navigation(keyboard_axes, state["yaw_continuous"],
                                    current_x=float(x_lqr[0, 0]))
        e3_l, e3_r = get_e3(self.left_leg.length, self.right_leg.length)
        jump_target = self.jump_state not in ("idle", "landing")
        target_e3_l = 0.0 if jump_target else e3_l
        target_e3_r = 0.0 if jump_target else e3_r
        if self.jump_state=="landing":
            fraction=float(np.clip((self.data.time-self._jump_trajectory[0])/self.jump_landing_duration,0.,1.))
            target_e3_l*=fraction
            target_e3_r*=fraction
        target = self.build_lqr_target(target_e3_l, target_e3_r)

        # 专用动作接管电机期间跳过平衡力矩计算。
        capturing = recovery_ctrl is not None and self.recovery.phase == 'rise'
        if capturing:
            recovery_ctrl = self.capture_motor_ctrl
        if recovery_ctrl is not None and self.recovery.blend == 0. and not capturing:
            self.U_lqr.fill(0.)
            self.active_balance_controller = None
            self._balance_switch_time = -np.inf
            self._reset_mpc_plans()
            forces = balance_ctrl = None
            ctrl = recovery_ctrl
        else:
            K_raw = get_K(self.left_leg.length, self.right_leg.length)
            u_eq = get_u_eq(self.left_leg.length, self.right_leg.length)
            balance_ctrl, forces = self._balance_command(
                x_lqr, target, K_raw, target_e3_l, target_e3_r, u_eq, state)
            ctrl = balance_ctrl
            if climb_capture:
                # 保留落轮末拍的髋力矩，再逐步撤去切换时的输出差。
                if self.climb_capture_offset is None:
                    self.climb_capture_offset = self.previous_motor_ctrl[:4] - balance_ctrl[:4]
                fraction = max(0., 1.-self.step_climb.capture_time/.045)
                ctrl = balance_ctrl.copy()
                ctrl[:4] = np.clip(ctrl[:4] + fraction*self.climb_capture_offset,
                                  -self.args.joint_torque_limit, self.args.joint_torque_limit)
            if capturing and self.recovery.capture_duration > 0.:
                # Cancel the initial step without delaying changes in feedback.
                if self.capture_torque_offset is None:
                    self.capture_torque_offset = recovery_ctrl-balance_ctrl
                ctrl = balance_ctrl+(1.-self.recovery.blend)*self.capture_torque_offset
                limits = np.array([self.args.joint_torque_limit]*4+[self.args.wheel_torque_limit]*2)
                ctrl = np.clip(ctrl, -limits, limits)
            if (capturing or self.recovery.cooldown > 0.) and self.recovery.capture_duration > 0.:
                delta = self.dt*np.array([1000.]*4+[250.]*2)
                ctrl = self.previous_motor_ctrl + np.clip(ctrl-self.previous_motor_ctrl, -delta, delta)

        if self.recovery.enabled:
            self.recovery_log.record(self, state, x_lqr, target, forces, balance_ctrl, recovery_ctrl, ctrl)

        self._apply_control(ctrl, state, x_lqr, viewer_perturb)

        self._send_telemetry(x_lqr, target, state, ctrl)

        return True


    @staticmethod
    def _unpack_keyboard_axes(keyboard_axes):
        if keyboard_axes is None:
            return 0.0, 0.0, 0.0, False
        speed_axis = keyboard_axes[0] if len(keyboard_axes) > 0 else 0.0
        yaw_axis = keyboard_axes[1] if len(keyboard_axes) > 1 else 0.0
        hight_axis = keyboard_axes[2] if len(keyboard_axes) > 2 else 0.0
        jump_requested = bool(keyboard_axes[3]) if len(keyboard_axes) > 3 else False
        return speed_axis, yaw_axis, hight_axis, jump_requested

    def _read_sensors(self, keyboard_axes):
        """传感器读取 + FK。"""
        state = self.refresh_kinematics()
        self._jump_sensor = state
        state["_wheel_contacts"] = self._wheel_contact_flags()
        for side, contact in enumerate(state["_wheel_contacts"]):
            ticks = self._wheel_contact_ticks[side]
            if contact:
                ticks = max(0, ticks) + 1
                if ticks*self.dt >= .004:
                    self._wheel_support_confirmed[side] = True
            else:
                ticks = min(0, ticks) - 1
                if -ticks*self.dt >= .006:
                    self._wheel_support_confirmed[side] = False
                    self._wheel_torque_scale[side] = 0.
            self._wheel_contact_ticks[side] = ticks
            if contact and self._wheel_support_confirmed[side]:
                self._wheel_torque_scale[side] = min(1., self._wheel_torque_scale[side]+self.dt/.012+1e-12)

        # 里程计仍逐轮使用原始接触，避免空转速度进入融合。
        self.gc.contact_state[:] = ["grounded" if contact else "airborne"
                                    for contact in state["_wheel_contacts"]]

        speed_axis, yaw_axis, hight_axis, jump_requested = self._unpack_keyboard_axes(keyboard_axes)

        state["_speed_axis"] = speed_axis
        state["_yaw_axis"] = yaw_axis
        state["_hight_axis"] = hight_axis
        state["_jump_requested"] = jump_requested
        return state


    def _update_navigation(self, keyboard_axes, current_yaw=None, current_x=None):
        """键盘->参考调速器 + 腿长伸缩限幅。"""
        speed_axis, yaw_axis, hight_axis, jump_requested = self._unpack_keyboard_axes(keyboard_axes)

        if self.reference_governor is not None:
            self.target_x, self.target_yaw, self.target_x_dot, self.target_yaw_dot = \
                self.reference_governor.update(speed_axis, yaw_axis,
                                               current_yaw=current_yaw,
                                               current_x=current_x)

        t = self.data.time
        if jump_requested and self.jump_state == "idle":
            # Buffer a press across a brief rolling contact gap; no pose prerequisite.
            self._jump_request_until = t + 0.15
        jump_pending = t <= self._jump_request_until
        contacts = self._wheel_contact_flags() if self.jump_state != "idle" or jump_pending else (False, False)
        if self.jump_state == "idle" and jump_pending and all(contacts):
            self.jump_state = "thrust"
            self._jump_request_until = -np.inf
            self._jump_return_length = self.target_leg_length_command
            self._jump_returning = False
            self.jump_start_time = t
            self.jump_airborne_seen = False
            self._jump_air_time = self._jump_contact_time = 0.0
            self._jump_ground_time = 0.0
            self._jump_vz = 0.0
            self._jump_height = self.wheel_radius + 0.5 * sum(
                leg.length * np.cos(leg.theta) for leg in (self.left_leg, self.right_leg))
            self._jump_sync_wheels = 0.0
            self._jump_wheel_reference = None
            # 短腿前进起跳时，缩短空中伸腿行程，减少落地前腿的后摆。
            self._jump_low_forward = (self._jump_return_length <= self.jump_retract_length + .03
                                      and self.body_odom.velocity > .25)
        if self.jump_state != "idle":
            sensor = self._jump_sensor
            roll,pitch,_ = sensor["euler"]
            ax,ay,az = sensor["accel"]
            vertical = (-np.sin(pitch)*ax+np.cos(pitch)*np.sin(roll)*ay
                        +np.cos(pitch)*np.cos(roll)*az-self.gravity)
            self._jump_height += self._jump_vz * self.dt + 0.5 * vertical * self.dt**2
            self._jump_vz += vertical*self.dt
            self._jump_air_time=self._jump_air_time+self.dt if not any(contacts) else 0.0
            self._jump_contact_time=self._jump_contact_time+self.dt if any(contacts) else 0.0
            self._jump_ground_time=self._jump_ground_time+self.dt if all(contacts) else 0.0
            if self._jump_air_time>=0.012: self.jump_airborne_seen=True
            if self.jump_state == "thrust" and (
                    (self._jump_air_time>=0.012)
                    or t-self.jump_start_time>=self.jump_max_thrust_time
                    or max(self.left_leg.length,self.right_leg.length)>=self.jump_thrust_end_length):
                self.jump_state="retract"
                self.jump_retract_start_time=t
                tuck_time = float(np.clip(0.75 * max(self._jump_vz, 0.0) / self.gravity,
                                          self.jump_min_tuck_time, self.jump_retract_duration))
                self._jump_trajectory=(t,np.array([self.left_leg.length,self.right_leg.length]),
                                       self.jump_retract_length,tuck_time)
            if self.jump_state == "extend":
                clearance = self._jump_height - self.wheel_radius - 0.5 * sum(
                    leg.length * np.cos(leg.theta) for leg in (self.left_leg, self.right_leg))
                clearance_rate = self._jump_vz - 0.5 * sum(
                    leg.length_dot.value * np.cos(leg.theta)
                    - leg.length * np.sin(leg.theta) * leg.theta_dot.value
                    for leg in (self.left_leg, self.right_leg))
                if self._jump_low_forward and clearance_rate < 0.0:
                    touchdown_time = max(0., clearance - .015) / -clearance_rate
                    fraction = float(np.clip((.10 - touchdown_time) / .08, 0., 1.))
                    self._jump_sync_wheels = max(self._jump_sync_wheels, fraction*fraction*(3.-2.*fraction))
                elif (not self._jump_low_forward and clearance_rate < 0.0
                      and clearance < -0.04 * clearance_rate + 0.015):
                    self._jump_sync_wheels = 1.0
            if self.jump_state in ("retract","extend"):
                if self.jump_airborne_seen and any(contacts) and t-self.jump_retract_start_time>=0.12:
                    self.jump_state="landing"
                    landing_length = self.leg_length_default
                    if self._jump_low_forward:
                        landing_length = min(landing_length, self._jump_return_length + .05)
                    self._jump_trajectory=(t,np.array([self.left_leg.length,self.right_leg.length]),
                                           landing_length,self.jump_landing_duration)
                elif (self.jump_state=="retract" and self._jump_vz < -0.3
                      and t-self.jump_retract_start_time>0.12):
                    self.jump_state="extend"
                    extend_length = self.leg_length_default
                    if self._jump_low_forward:
                        extend_length = min(extend_length, self._jump_return_length + .06)
                    self._jump_trajectory=(t,np.array([self.left_leg.length,self.right_leg.length]),
                                           extend_length,self.jump_retract_duration)
            if (self.jump_state=="landing" and self._jump_ground_time >= 0.06
                and t-self._jump_trajectory[0]>=self.jump_landing_duration):
                self.jump_state="idle"
                self.target_leg_length_command=self.leg_length_default
                if self._jump_low_forward:
                    self.target_leg_length_command = self._jump_trajectory[2]
                self._jump_returning = True
                self.jump_airborne_seen=False
                self._jump_trajectory=None
                self.roll_integral=0.0
            # Airborne wheels control attitude, not longitudinal ground speed.
            # Keep the keyboard/governor velocity command for the next touchdown.
            if self.reference_governor is not None and current_x is not None:
                self.reference_governor.x = current_x
                self.target_x = current_x
        elif self.jump_state == "idle":
            if hight_axis:
                self._jump_returning = False
            if self._jump_returning:
                self.target_leg_length_command = approach_value(
                    self.target_leg_length_command, self._jump_return_length, 0.15 * self.dt)
                self._jump_returning = abs(self.target_leg_length_command - self._jump_return_length) > 1e-6
            self.target_leg_length_command = float(np.clip(
                self.target_leg_length_command+hight_axis,self.leg_length_min,self.leg_length_max))
        self.target_leg_length=(self.jump_retract_length if self.jump_state=="retract"
                               else self.target_leg_length_command)

    def _jump_length_reference(self):
        start,initial,goal,duration=self._jump_trajectory
        u=float(np.clip((self.data.time-start)/duration,0.,1.))
        smooth=10*u**3-15*u**4+6*u**5
        speed=(30*u**2-60*u**3+30*u**4)/duration
        return initial+(goal-initial)*smooth,(goal-initial)*speed

    def _compute_forces(self, x_lqr, target, K_raw, state):
        """返回左右腿请求的轴向力，单位 N。"""
        if self._has_double_support(state):
            return self._height_roll_support(target, state)

        self.support_height_pos.reset()
        self.support_height_vel.reset()
        if self.jump_state != "idle":
            if self.jump_state=="thrust":
                forces=np.full(2,self.jump_force)
            else:
                reference,rate=self._jump_length_reference()
                lengths=np.array([self.left_leg.length,self.right_leg.length])
                speeds=np.array([self.left_leg.length_dot.value,self.right_leg.length_dot.value])
                forces=self.jump_length_kp*(reference-lengths)+self.jump_length_kd*(rate-speeds)
                if self.jump_state=="landing":
                    forces=1200.0*(reference-lengths)+240.0*(rate-speeds)
                    forces+=self.support_force*np.array(self._wheel_contact_flags())
            if self.jump_state in ("thrust","landing"):
                difference=-(80.0*state["euler"][0]+10.0*state["gyro"][0])/(2*self.half_track)
                forces+=np.array([difference,-difference])
            return tuple(forces)

        length_l = length_r = self.target_leg_length
        if self.recovery.phase == 'rise':
            height_blend = self.recovery.capture_pose_blend
            length_l, length_r = self.target_leg_length + (1.-height_blend)*(
                self.capture_start_lengths-self.target_leg_length)
            if self.recovery.trace is not None:
                self.recovery.trace['capture_leg_length_ref_m'] = [length_l, length_r]
        left_pos_force = self.left_length_pos.update(length_l, self.left_leg.length)
        right_pos_force = self.right_length_pos.update(length_r, self.right_leg.length)
        left_pos_force = clip_symmetric(left_pos_force, self.args.length_position_force_limit)
        right_pos_force = clip_symmetric(right_pos_force, self.args.length_position_force_limit)

        left_vel_force = self.left_length_vel.update(0.0, self.left_leg.length_dot.value)
        right_vel_force = self.right_length_vel.update(0.0, self.right_leg.length_dot.value)
        left_vel_force = clip_symmetric(left_vel_force, self.args.length_velocity_force_limit)
        right_vel_force = clip_symmetric(right_vel_force, self.args.length_velocity_force_limit)

        # 用世界腿角的竖直投影和轮半径估计髋部高度。
        h = self.wheel_radius + 0.5 * (
            self.left_leg.length * np.cos(self.left_leg.theta)
            + self.right_leg.length * np.cos(self.right_leg.theta))
        roll_force = self._roll_support_force(state, h, self.args.roll_force_limit)

        # 按世界腿角将竖直支撑力换算成轴向力。
        left_cos = float(max(np.cos(self.left_leg.theta_world), 1e-3))
        right_cos = float(max(np.cos(self.right_leg.theta_world), 1e-3))
        left_support_force = self.support_force / left_cos
        right_support_force = self.support_force / right_cos
        F_l = left_vel_force + roll_force + left_pos_force + left_support_force
        F_r = right_vel_force - roll_force + right_pos_force + right_support_force
        return F_l, F_r

    def _has_double_support(self, state):
        return (self.jump_state == "idle" and self.step_climb.phase == "idle"
                and not self.recovery.active and all(state["_wheel_contacts"])
                and all(self._wheel_support_confirmed))

    def _support_measurement(self, target, state):
        world_up = rotation(state["quat"])[2]
        omega = state["gyro"]
        roll = state["euler"][0]
        pitch_rate = np.cos(roll) * omega[1] - np.sin(roll) * omega[2]
        projections, heights, rates = [], [], []
        for leg in (self.left_leg, self.right_leg):
            sine, cosine = np.sin(leg.theta_body), np.cos(leg.theta_body)
            axis = np.array([sine, 0., cosine])
            tangent = np.array([cosine, 0., -sine])
            axis_rate = np.cross(omega, axis) + tangent * (leg.theta_dot.value - pitch_rate)
            projection = float(world_up @ axis)
            projections.append(projection)
            heights.append(leg.length * projection)
            rates.append(leg.length_dot.value * projection + leg.length * (world_up @ axis_rate))

        # 高度相对于两轮接触面的平均高度。轮半径在参考与实测之差中抵消。
        height = float(np.mean(heights))
        height_ref = self.target_leg_length * float(np.mean(np.cos(target[3:5, 0])))
        height_rate = float(np.mean(rates))
        # 跟踪欧拉横滚角，使用对应的角速度而不是直接使用陀螺仪 x 轴。
        pitch = state["euler"][1]
        roll_rate = omega[0] + np.tan(pitch) * (np.sin(roll) * omega[1] + np.cos(roll) * omega[2])
        return (height, height_rate, height_ref, roll, roll_rate), np.asarray(projections)

    def _height_roll_support(self, target, state):
        """在双轮支撑时，用高度与横滚反馈生成两腿轴向力。"""
        (height, height_rate, height_ref, _, _), projections = self._support_measurement(target, state)
        projections = np.maximum(projections, 1e-3)
        pos_force = clip_symmetric(self.support_height_pos.update(height_ref, height),
                                   self.args.length_position_force_limit)
        vel_force = clip_symmetric(self.support_height_vel.update(0., height_rate),
                                   self.args.length_velocity_force_limit)
        mean_force = max(0., self.support_force + pos_force + vel_force)
        # 左右差动力保持总支撑力，并避免要求任一侧向下拉地。
        roll_force = self._roll_support_force(
            state, self.wheel_radius + height, min(mean_force, self.args.roll_force_limit))
        return ((mean_force + roll_force) / projections[0],
                (mean_force - roll_force) / projections[1])

    def _roll_support_force(self, state, h, force_limit):
        roll = state["euler"][0]
        roll_rate = state["gyro"][0]
        I0 = self.roll_inertia + self.total_mass * h * h               # 平行轴定理
        W = 2.0 * self.half_track
        kp_eff = (self.total_mass * self.gravity * h + I0 * self.wc_roll**2) / W   # 抵消 mgh + 极点配置
        kd_eff = (2.0 * self.zeta_roll * self.wc_roll * I0) / W
        self.roll_integral += roll * self.dt
        roll_force = -kp_eff * roll - kd_eff * roll_rate - self.ki_slope * self.roll_integral
        # 抗积分饱和：输出被限幅则回退本步积分
        if abs(roll_force) >= force_limit:
            self.roll_integral -= roll * self.dt
        return clip_symmetric(roll_force, force_limit)

    def compute_foot_forces(self, state):
        """估计两轮法向载荷，每个控制周期更新一次。"""
        hip_torque = self.previous_motor_ctrl[:4]
        if self.climb_torque_adr is not None:
            hip_torque = self.data.sensordata[self.climb_torque_adr]
        return self.gc.estimate_normal_forces(
            state, hip_torque+state['gas_hip_torque_nm'], self.previous_motor_ctrl[4:6],
            self._jump_momentum, self.step_climb.geometry.dy)

    def _balance_command(self, x_lqr, target, K_raw, e3_l, e3_r, u_eq, state):
        """Select the controller and blend commands at a mode change."""
        double_support = self._has_double_support(state)
        suppression = self.gc.get_K_suppression_mask()
        mode = self.balance_controller
        jumping = (self.jump_state not in ("idle", "landing") or
                   (self.jump_state == "landing" and not any(state["_wheel_contacts"])))
        forces = self._compute_forces(x_lqr, target, K_raw, state)
        if mode == "mpc" and not jumping:
            ctrl, forces = self._mpc_balance_command(x_lqr, target, state, suppression, forces)
        else:
            if self._mpc is not None and self._mpc.plan is not None:
                self._reset_mpc_plans()
            if mode == "mpc":
                ctrl = self._jump_torques(*forces, state)
            else:
                ctrl = self._map_to_actuators(*forces, x_lqr, K_raw, e3_l, e3_r, u_eq, state)

        if double_support:
            if self.active_balance_controller not in (None, mode):
                self._balance_switch_offset = self.previous_motor_ctrl - ctrl
                self._balance_switch_time = self.data.time
            blend = max(0., 1. - (self.data.time - self._balance_switch_time) / .04)
            if blend > 0.:
                lower, upper = self._motor_bounds()
                ctrl = np.clip(ctrl + blend * self._balance_switch_offset, lower, upper)
            ctrl[4:][np.asarray(suppression[:2])] = 0.
            if mode == "mpc":
                ctrl[4:] = np.clip(ctrl[4:], *self._mpc_wheel_bounds)
                ctrl[:4][np.tile(np.asarray(suppression[2:], dtype=bool), 2)] = 0.
        else:
            self._balance_switch_time = -np.inf
        self.active_balance_controller = mode if not jumping else None
        return ctrl, forces

    def _motor_bounds(self):
        limits = np.repeat([self.args.joint_torque_limit, self.args.wheel_torque_limit], [4, 2])
        lower, upper = -limits, limits.copy()
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        lower[limited] = np.maximum(lower[limited], self.model.actuator_ctrlrange[limited, 0])
        upper[limited] = np.minimum(upper[limited], self.model.actuator_ctrlrange[limited, 1])
        return lower, upper

    @staticmethod
    def _torque_interval(gains, offset, lower, upper):
        """Intersect the scalar input ranges allowed by the two hip motors."""
        lo, hi = -np.inf, np.inf
        for gain, bias, minimum, maximum in zip(gains, offset, lower, upper):
            if abs(gain) < 1e-8:
                if not minimum <= bias <= maximum:
                    return np.inf, -np.inf
                continue
            first, second = (minimum - bias) / gain, (maximum - bias) / gain
            lo, hi = max(lo, min(first, second)), min(hi, max(first, second))
        return lo, hi

    def _mpc_balance_command(self, x_lqr, target, state, suppression, forces):
        legs = (self.left_leg, self.right_leg)
        jacobians = [leg.jacobian(state[f"{side}_front_pos"], state[f"{side}_back_pos"])
                     for side, leg in zip(("left", "right"), legs)]
        lower, upper = self._motor_bounds()
        wheel_bounds = np.column_stack((lower[4:], upper[4:])) * self._wheel_torque_scale[:, None]
        contacts = np.asarray(state["_wheel_contacts"], dtype=bool)
        swing_enabled = ~np.asarray(suppression[2:], dtype=bool)
        wheel_bounds[~contacts | np.asarray(suppression[:2], dtype=bool)] = 0.
        spring = state["gas_hip_torque_nm"]
        lengths = np.array([leg.length for leg in legs])
        forces = np.array(forces, dtype=float)
        swing_bounds = []
        for i, jac in enumerate(jacobians):
            if not swing_enabled[i]:
                forces[i] = 0.
                swing_bounds.append((0., 0.))
                continue
            motors = slice(i, 4, 2)
            motor_min, motor_max = lower[motors], upper[motors]
            offset = jac[0] * forces[i] - spring[motors]
            lo, hi = self._torque_interval(jac[1], offset, motor_min, motor_max)
            if lo > hi:
                # Limit the PD force only if no swing torque can make it feasible.
                force_lo, force_hi = self._torque_interval(
                    jac[0], -spring[motors], motor_min, motor_max)
                if force_lo <= force_hi:
                    forces[i] = np.clip(forces[i], force_lo, force_hi)
                    offset = jac[0] * forces[i] - spring[motors]
                    lo, hi = self._torque_interval(jac[1], offset, motor_min, motor_max)
            swing_bounds.append((lo, hi) if lo <= hi else (0., 0.))
        result = self._mpc.solve(
            lengths, x_lqr, target, [*swing_bounds, *wheel_bounds])
        if result is None:
            raise RuntimeError("平衡 MPC 未得到有限的力矩解")
        self.U_lqr = result
        self._mpc_wheel_bounds = wheel_bounds.T
        hip = np.zeros(4)
        for i, (leg, jac) in enumerate(zip(legs, jacobians)):
            leg.force = forces[i]
            leg.front_torque, leg.back_torque = jac.T @ np.array([forces[i], result[i, 0]])
            if swing_enabled[i]:
                motors = slice(i, 4, 2)
                hip[motors] = np.array([leg.front_torque, leg.back_torque]) - spring[motors]
        ctrl = np.r_[np.clip(hip, lower[:4], upper[:4]), result[2:, 0]]
        return ctrl, tuple(forces)


    def _map_to_actuators(self, F_l, F_r, x_lqr, K_raw,
                          e3_l, e3_r, u_eq, state):
        """VMC 力矩映射 + 离地 K 抑制 + 力矩限幅。

        Returns
        -------
        ctrl : (6,) ndarray  — [LF, RF, LB, RB, WL, WR]
        """
        if (self.jump_state not in ("idle", "landing") or
                (self.jump_state == "landing" and not any(self._wheel_contact_flags()))):
            self.active_balance_controller = None
            self._balance_switch_time = -np.inf
            return self._jump_torques(F_l, F_r, state)

        e3_l_adj = e3_l
        e3_r_adj = e3_r

        zwL, zwR, zlL, zlR = self.gc.get_K_suppression_mask()
        normal_balance = (self.jump_state == "idle" and self.step_climb.phase == "idle"
                          and not self.recovery.active)
        if normal_balance:
            zwL = zlL or not self._wheel_support_confirmed[0]
            zwR = zlR or not self._wheel_support_confirmed[1]
        if self.jump_state == "landing":
            contactL, contactR = self._wheel_contact_flags()
            zwL, zwR = not contactL, not contactR
            zlL, zlR = not contactL, not contactR
        if zlL:
            e3_l_adj = -e3_l
        if zlR:
            e3_r_adj = -e3_r
        target_adj = self.build_lqr_target(e3_l_adj, e3_r_adj)

        K_adj = K_raw.copy()
        equilibrium = np.asarray(u_eq, dtype=float).reshape(4, 1).copy()
        suppressed = np.array([zlL, zlR, zwL, zwR])
        K_adj[suppressed, :] = 0.
        equilibrium[suppressed] = 0.
        U_adj = equilibrium - K_adj @ (x_lqr - target_adj)
        self.U_lqr = U_adj

        if self.jump_state == "landing":
            forces=[F_l,F_r]
            poses=[(state['left_front_pos'],state['left_back_pos']),
                   (state['right_front_pos'],state['right_back_pos'])]
            for i,(leg,q) in enumerate(zip([self.left_leg,self.right_leg],poses)):
                J=leg.jacobian(*q);lo,hi=-2000.,2000.
                for hip_index, (length_gain, angle_gain) in enumerate(J.T):
                    if abs(length_gain)>1e-8:
                        torque_offset = angle_gain*U_adj[i,0] - state["gas_hip_torque_nm"][i+2*hip_index]
                        a=(-self.args.joint_torque_limit-torque_offset)/length_gain
                        b=(self.args.joint_torque_limit-torque_offset)/length_gain
                        lo,hi=max(lo,min(a,b)),min(hi,max(a,b))
                forces[i]=np.clip(forces[i],lo,hi) if lo<=hi else 0.
            F_l,F_r=forces

        left_front, left_back = self.left_leg.vmc(
            F_l, U_adj[0, 0], state["left_front_pos"], state["left_back_pos"])
        right_front, right_back = self.right_leg.vmc(
            F_r, U_adj[1, 0], state["right_front_pos"], state["right_back_pos"])

        ctrl = np.array([left_front, right_front,
                         left_back, right_back,
                         U_adj[2, 0], U_adj[3, 0]], dtype=float)
        # VMC 给出所需的总关节力矩，电机只输出扣除弹簧后的部分。
        ctrl[:4] -= state["gas_hip_torque_nm"]
        ctrl[:4] = np.clip(ctrl[:4], -self.args.joint_torque_limit, self.args.joint_torque_limit)
        ctrl[4:] = np.clip(ctrl[4:], -self.args.wheel_torque_limit, self.args.wheel_torque_limit)

        if normal_balance:
            ctrl[4:] *= self._wheel_torque_scale
        if zwL:
            ctrl[4] = 0.
        if zwR:
            ctrl[5] = 0.
        return ctrl

    def _jump_torques(self, F_l, F_r, state):
        legs = [self.left_leg, self.right_leg]
        angles = np.array([leg.theta for leg in legs])
        rates = np.array([leg.theta_dot.value for leg in legs])
        forces = np.array([F_l, F_r])
        phi, omega = state["euler"][1], state["gyro"][1]
        bx, _, bz = self.cfg["mass_properties"]["body"]["com_local_m"]
        rx = -(bx*np.cos(phi)+bz*np.sin(phi))
        rz = -(-bx*np.sin(phi)+bz*np.cos(phi))
        pos = [(state['left_front_pos'],state['left_back_pos']),
               (state['right_front_pos'],state['right_back_pos'])]
        js = [leg.jacobian(*q) for leg,q in zip(legs,pos)]
        for _ in range(2):
            moment = rz*np.sum(forces*np.sin(angles))-rx*np.sum(forces*np.cos(angles))
            hip = .5*(40.0*phi+5.0*omega+moment)
            for i,J in enumerate(js):
                lo,hi=-2000.,2000.
                for hip_index, (length_gain, angle_gain) in enumerate(J.T):
                    if abs(length_gain)>1e-8:
                        torque_offset = angle_gain*hip - state["gas_hip_torque_nm"][i+2*hip_index]
                        a=(-self.args.joint_torque_limit-torque_offset)/length_gain
                        b=(self.args.joint_torque_limit-torque_offset)/length_gain
                        lo,hi=max(lo,min(a,b)),min(hi,max(a,b))
                forces[i]=np.clip(forces[i],lo,hi) if lo<=hi else 0.
        moment = rz*np.sum(forces*np.sin(angles))-rx*np.sum(forces*np.cos(angles))
        hips = np.full(2,.5*(40.0*phi+5.0*omega+moment))
        contacts=self._wheel_contact_flags()
        gains=np.full(2,20.0)
        damping=np.full(2,1.2)
        swing = -gains*angles-damping*rates
        alpha=np.array([1.+leg.length*np.cos(leg.theta)/self.wheel_radius if contacts[i] else 1.
                        for i,leg in enumerate(legs)])
        wheel_drag=self._jump_wheel_damping*np.array([state['left_wheel_vel'],state['right_wheel_vel']])
        drag=np.array([wheel_drag[i] if not contacts[i] else 0. for i in range(2)])
        wheels=np.clip((hips-swing)/alpha+drag,-self.args.wheel_torque_limit,self.args.wheel_torque_limit)
        if not any(contacts):
            # Internal torques conserve whole-robot angular momentum in flight.
            # Hip torque levels the body; wheel momentum allows the world-leg
            # angles to converge to zero at the measured body angular rate.
            H,_,_=self._jump_momentum.evaluate(state)
            desired = dict(state)
            for i,name in enumerate(('left','right')):
                dq=np.array([state[name+'_front_vel'],state[name+'_back_vel']])
                length_rate=js[i][0]@dq
                reference=np.linalg.solve(js[i],np.array([length_rate,-10.0*angles[i]-desired['gyro'][1]]))
                desired[name+'_front_vel'],desired[name+'_back_vel']=reference
                desired[name+'_wheel_vel']=0.0
            expected,_,_=self._jump_momentum.evaluate(desired)
            inertia=self._jump_momentum.inertia[[7,14]]
            reference=float(np.clip((H-expected)/np.sum(inertia),-self.jump_wheel_speed_limit,self.jump_wheel_speed_limit))
            if self._jump_sync_wheels:
                ground_reference = np.array([
                    (self.body_odom.velocity - leg.length_dot.value * np.sin(leg.theta)
                     - leg.length * np.cos(leg.theta) * leg.theta_dot.value) / self.wheel_radius
                    - leg.shank_rate for leg in legs])
                reference += self._jump_sync_wheels * (ground_reference - reference)
            if self._jump_wheel_reference is None:
                acceleration=0.0
            else:
                acceleration=np.clip((reference-self._jump_wheel_reference)/self.dt,-5000.,5000.)
            self._jump_wheel_reference=reference
            measured=np.array([state['left_wheel_vel'],state['right_wheel_vel']])
            wheels=np.clip(wheel_drag+inertia*(acceleration+120.0*(reference-measured)),
                           -self.args.wheel_torque_limit,self.args.wheel_torque_limit)
            differential=-20.0*(angles[0]-angles[1])-1.2*(rates[0]-rates[1])
            hips+=np.array([.5*differential,-.5*differential])
        else:
            self._jump_wheel_reference=None
        if self.jump_state=="thrust" and all(contacts):
            # Reduce launch angular momentum through the ground reaction.
            H,center,wheel_center=self._jump_momentum.evaluate(state)
            height=max(0.10,center[1]-wheel_center[1]+self.wheel_radius)
            lever=wheel_center[0]-center[0]
            # Approximate total normal load from body support and leg mass.
            # Hdot = -height * Fx - lever * Fz, Fx ~= (Tw - damping) / r.
            vertical_force=np.sum(forces*np.cos(angles))*self.total_mass/self.body_mass
            momentum_gain = self.jump_momentum_gain
            if self._jump_low_forward:
                # 给离地后的腿角调整保留轮速余量。
                momentum_gain = min(momentum_gain, 10.0)
            total=momentum_gain*H-self.wheel_radius*lever*vertical_force/height+sum(wheel_drag)
            wheels[:]=np.clip(total/2.,-self.args.wheel_torque_limit,self.args.wheel_torque_limit)
        left=legs[0].vmc(forces[0],hips[0],*pos[0]);right=legs[1].vmc(forces[1],hips[1],*pos[1])
        self.U_lqr=np.array([hips[0],hips[1],wheels[0],wheels[1]]).reshape(4,1)
        ctrl = np.r_[left[0],right[0],left[1],right[1],wheels]
        ctrl[:4] -= state["gas_hip_torque_nm"]
        return ctrl.clip(
            [-self.args.joint_torque_limit]*4+[-self.args.wheel_torque_limit]*2,
            [self.args.joint_torque_limit]*4+[self.args.wheel_torque_limit]*2)

    def _wheel_contact_flags(self):
        """逐轮读取 MuJoCo 环境接触，排除本车内部碰撞。"""
        pairs = self.data.contact.geom
        roots = self.model.body_rootid[self.model.geom_bodyid]
        robot_root = roots[self.wheel_contact_geom_ids[0]]
        contacts = []
        for geom in self.wheel_contact_geom_ids:
            rows, columns = np.nonzero(pairs == geom)
            other = pairs[rows, 1-columns]
            contacts.append(bool(np.any((other >= 0) & (roots[other] != robot_root))))
        return tuple(contacts)


    def _apply_control(self, ctrl, state, x_lqr, viewer_perturb=None):
        """地面接触检测更新 + 外力施加 + MuJoCo 物理步进。"""
        left_jt = np.array([ctrl[0], ctrl[2]])
        right_jt = np.array([ctrl[1], ctrl[3]])
        Fn_l, Fn_r = self.compute_foot_forces(state)
        self.previous_motor_ctrl[:] = ctrl
        self.gc.update(Fn_l, Fn_r,
                       state["left_wheel_vel"], state["right_wheel_vel"],
                       x_lqr[5, 0], state["accel"][2],
                       left_jt, right_jt,
                       ctrl[4], ctrl[5],
                       yaw_rate=x_lqr[6, 0],
                       accel_x=state.get("horizontal_accel", state["accel"][0]),
                       specific_force_norm=np.linalg.norm(state['accel']),
                       wheel_contacts=state["_wheel_contacts"])

        if getattr(self, "native_managed", False):
            self.data.ctrl[:] = ctrl
            self.last_sim_time = self.data.time
            return
        self.data.xfrc_applied[:] = 0.0
        if viewer_perturb is not None:
            mj.mjv_applyPerturbForce(self.model, self.data, viewer_perturb)
        if self.args.push_duration > 0.0:
            push_end = self.args.push_time + self.args.push_duration
            if self.args.push_time <= self.data.time < push_end:
                self.data.xfrc_applied[self.push_body_id] += np.array([
                    self.args.push_force_x, self.args.push_force_y, self.args.push_force_z,
                    self.args.push_torque_x, self.args.push_torque_y, self.args.push_torque_z,
                ], dtype=float)

        self.data.ctrl[:] = ctrl
        mj.mj_step(self.model, self.data)
        self.last_sim_time = self.data.time


    def _send_telemetry(self, x_lqr, target, state, ctrl):
        """UDP 发送（24 floats） + 控制台定时日志。"""
        roll, pitch, yaw_wrapped = state["euler"]
        yaw = state["yaw_continuous"]
        Fn_l, Fn_r = self.gc.fn_left, self.gc.fn_right

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

        if self.args.log_every > 0.0 and self.data.time - self.last_log_time >= self.args.log_every:
            self.last_log_time = self.data.time
            wheel_body_vx = state.get("wheel_body_vx", self.wheel_forward_velocity(state))
            body_vx_est = x_lqr[5, 0]  # Kalman 融合车速（轮速 + IMU）
            slip_vx = wheel_body_vx - body_vx_est
            print(
                f"t={self.data.time:6.3f} "
                f"roll={roll:+.4f} pitch={pitch:+.4f} yaw={yaw:+.4f} "
                f"bx={x_lqr[0, 0]:+.3f} bv={x_lqr[5, 0]:+.3f} "
                f"wv={wheel_body_vx:+.3f} slip={slip_vx:+.3f} "
                f"odo_x={self.odometry.position:+.3f} "
                f"yawdot={x_lqr[6, 0]:+.3f} "
                f"target=[{self.target_x:+.2f},{self.target_x_dot:+.2f},{self.target_yaw:+.2f},{self.target_yaw_dot:+.2f}] "
                f"L={self.left_leg.length:.4f}/{self.right_leg.length:.4f} "
                f"Fn_l={Fn_l:.1f} Fn_r={Fn_r:.1f} "
                f"u=[{ctrl[0]:+.2f}, {ctrl[1]:+.2f}, {ctrl[2]:+.2f}, {ctrl[3]:+.2f}, {ctrl[4]:+.2f}, {ctrl[5]:+.2f}]"
            )
