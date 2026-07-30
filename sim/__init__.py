"""轮腿机器人仿真模块（手册各章节对应）。

在线仿真模块映射：
  kinematics.py     — 手册 §2: 五连杆并联腿运动学 FK + IK + Jacobian
  vmc.py            — 手册 §6: VMC 力矩映射 τ = JᵀF
  pid.py            — 手册 §8: PID 控制器
  kalman.py         — 手册 §7: 2-D Kalman 滤波器
  lqr_governor.py   — 手册 §4-5: 参考调速器 + 增益调度
  ground_contact.py — 手册 §9: 地面接触检测
  controller.py     — 手册 §8, §13: 控制管线编排器
  keyboard_reader.py — 键盘输入读取
  utils.py          — 工具函数 + ContinuousAngle
"""

from .kinematics import FiveLinkLeg, DiscreteDerivative, SecondDerivative
from .vmc import VmcMapper
from .pid import PID
from .kalman import KalmanOdometry
from .lqr_governor import LqrReferenceGovernor
from .ground_contact import GroundContactDetector
from .controller import StandController
from .keyboard_reader import KeyboardHoldReader
from .utils import (
    load_config, resolve_project_path, sensor_value,
    quat_to_euler, normalize_angle, clip_symmetric,
    approach_value, require_actuator_order, ContinuousAngle,
    PROJECT_ROOT,
)
