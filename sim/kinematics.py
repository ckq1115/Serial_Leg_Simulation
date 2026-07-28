"""五连杆并联腿运动学（手册 §2）：正解 FK + 逆解 IK + 雅可比 J。

当前 MuJoCo 模型为五连杆并联腿（通过 equality connect 闭环）。正解用交点法求解
五连杆闭合几何 → 虚拟腿长 L 和摆角 θ。正解唯一确定（atan2 选上半支）。

逆解使用数值优化（fsolve）求解 (front_angle, back_angle) 使 FK 输出匹配目标 (L, θ)。
"""

import numpy as np
from scipy.optimize import fsolve
from vmc import VmcMapper

class DiscreteDerivative:
    def __init__(self, dt):
        self.dt = dt
        self.last = None
        self.value = 0.0
        self.prev_value = 0.0

    def reset(self):
        self.last = None
        self.value = 0.0
        self.prev_value = 0.0

    def update(self, current):
        if self.last is None:
            self.last = current
            self.value = 0.0
            self.prev_value = current
            return self.value
        self.value = (current - self.last) / self.dt
        self.last = current
        return self.value

    def diff_num(self):
        return self.value

class SecondDerivative:
    def __init__(self, dt):
        self.dt = dt
        self.prev = None
        self.value = 0.0

    def reset(self):
        self.prev = None
        self.value = 0.0

    def update(self, current):
        if self.prev is None:
            self.prev = current
            self.value = 0.0
            return self.value
        if hasattr(self, 'prev_prev'):
            self.value = (current - 2*self.prev + self.prev_prev) / (self.dt**2)
        else:
            self.value = 0.0
        self.prev_prev = self.prev
        self.prev = current
        return self.value

class FiveLinkLeg:
    def __init__(self, side, link_lengths, dt):
        self.side = side
        self.link_lengths = link_lengths
        self.length = 0.0
        self.theta = 0.0
        self.phi_front = 0.0
        self.phi_back = 0.0
        self.length_dot = DiscreteDerivative(dt)
        self.theta_dot = DiscreteDerivative(dt)
        self.length_ddot = SecondDerivative(dt)
        self.theta_ddot = SecondDerivative(dt)
        self.front_torque = 0.0
        self.back_torque = 0.0
        self.force = 0.0
        self._vmc = VmcMapper(link_lengths)

    def reset_dynamic_state(self):
        self.length_dot.reset()
        self.theta_dot.reset()
        self.length_ddot.reset()
        self.theta_ddot.reset()
        self.front_torque = 0.0
        self.back_torque = 0.0
        self.force = 0.0

    def forward(self, front_angle, back_angle, body_pitch):
        l_base = self.link_lengths["anchor_base_m"]
        l_front_big = self.link_lengths["front_big_m"]
        l_front_small = self.link_lengths["front_small_m"]
        l_back_big = self.link_lengths["back_big_m"]
        l_back_small = self.link_lengths["back_small_m"]

        x_a = -l_base
        y_a = 0.0
        x_e = l_base
        y_e = 0.0

        x_b = x_a + l_back_big * np.cos(back_angle)
        y_b = y_a + l_back_big * np.sin(back_angle)
        x_d = x_e + l_front_big * np.cos(front_angle)
        y_d = y_e + l_front_big * np.sin(front_angle)

        l_bd = np.hypot(x_b - x_d, y_b - y_d)
        a = 2.0 * l_back_small * (x_d - x_b)
        b = 2.0 * l_back_small * (y_d - y_b)
        c = l_back_small**2 + l_bd**2 - l_front_small**2
        discriminant = max(a * a + b * b - c * c, 0.0)

        self.phi_back = 2.0 * np.arctan2(b + np.sqrt(discriminant), a + c)
        x_c = x_b + l_back_small * np.cos(self.phi_back)
        y_c = y_b + l_back_small * np.sin(self.phi_back)

        self.phi_front = np.arctan2(y_c - y_d, x_c - x_d)
        self.length = np.hypot(x_c, y_c)
        self.theta = -np.arctan2(x_c, y_c) + body_pitch

        self.length_dot.update(self.length)
        self.theta_dot.update(self.theta)
        self.length_ddot.update(self.length)
        self.theta_ddot.update(self.theta)
        return self.length, self.theta

    def vmc(self, leg_force, leg_torque, front_angle, back_angle):
        """VMC 力矩映射（委托给 VmcMapper，手册 §6）。"""
        self.force = leg_force
        self.front_torque, self.back_torque = self._vmc.compute(
            self.length, self.theta, self.phi_front, self.phi_back,
            leg_force, leg_torque, front_angle, back_angle)
        return self.front_torque, self.back_torque

    # ── 逆运动学（手册 §2.4）──────────────────────────────────────────
    def inverse(self, L_des, theta_des, body_pitch=0.0):
        """数值逆解：给定目标 (L, θ_leg)，求解 (front_angle, back_angle)。

        使用 fsolve 在 2-DOF 上寻根，残差为 FK 输出与目标值的偏差。
        以当前关节角为初值，收敛快速（通常 <10 次迭代）。

        Parameters
        ----------
        L_des : float
            目标虚拟腿长 [m]。
        theta_des : float
            目标腿摆角（机体系）[rad]。
        body_pitch : float
            机体俯仰角 [rad]（默认 0）。

        Returns
        -------
        (front_angle, back_angle) : (float, float)
            使 FK 输出匹配目标的关节角度 [rad]。
        """
        # 以当前角度为初值
        fa0 = np.arctan2(
            np.sin(self.phi_front) if hasattr(self, 'phi_front') and self.phi_front != 0 else 0.3,
            np.cos(self.phi_front) if hasattr(self, 'phi_front') else 1.0)
        ba0 = np.pi - 0.5  # 合理初值
        # 更鲁棒的初值：从上次 FK 结果反推
        try:
            _ = self.length
            fa0 = float(np.arctan2(np.sin(self.phi_front), np.cos(self.phi_front))) if abs(self.phi_front) > 0.01 else 0.3
            ba0 = float(np.arctan2(np.sin(self.phi_back), np.cos(self.phi_back))) if abs(self.phi_back) > 0.01 else np.pi - 0.5
        except Exception:
            pass

        def residual(x):
            fa, ba = x[0], x[1]
            L, th = self._forward_raw(fa, ba)
            return [L - L_des, (th - body_pitch) - theta_des]

        sol = fsolve(residual, [fa0, ba0], maxfev=100, xtol=1e-8)
        return float(sol[0]), float(sol[1])

    def _forward_raw(self, front_angle, back_angle):
        """内部 FK（不修改 self 状态，不更新微分器）。"""
        link = self.link_lengths
        l_base = link["anchor_base_m"]
        l_front_big = link["front_big_m"]
        l_front_small = link["front_small_m"]
        l_back_big = link["back_big_m"]
        l_back_small = link["back_small_m"]

        x_a, y_a = -l_base, 0.0
        x_e, y_e = l_base, 0.0

        x_b = x_a + l_back_big * np.cos(back_angle)
        y_b = y_a + l_back_big * np.sin(back_angle)
        x_d = x_e + l_front_big * np.cos(front_angle)
        y_d = y_e + l_front_big * np.sin(front_angle)

        l_bd = np.hypot(x_b - x_d, y_b - y_d)
        a = 2.0 * l_back_small * (x_d - x_b)
        b = 2.0 * l_back_small * (y_d - y_b)
        c_val = l_back_small**2 + l_bd**2 - l_front_small**2
        disc = max(a * a + b * b - c_val * c_val, 0.0)

        phi_back = 2.0 * np.arctan2(b + np.sqrt(disc), a + c_val)
        x_c = x_b + l_back_small * np.cos(phi_back)
        y_c = y_b + l_back_small * np.sin(phi_back)

        L = np.hypot(x_c, y_c)
        theta = -np.arctan2(x_c, y_c)
        return L, theta

    # ── 雅可比（手册 §2.5）────────────────────────────────────────────
    def jacobian(self, front_angle, back_angle):
        """数值雅可比 J = ∂(L, θ_leg)/∂(front, back)（有限差分 ε=1e-6）。

        用于 VMC 解析雅可比的交叉验证，以及逆运动学的梯度辅助。
        """
        eps = 1e-6
        L0, th0 = self._forward_raw(front_angle, back_angle)
        J = np.zeros((2, 2))
        for col, (d_fa, d_ba) in enumerate([(eps, 0.0), (0.0, eps)]):
            L1, th1 = self._forward_raw(front_angle + d_fa, back_angle + d_ba)
            J[0, col] = (L1 - L0) / eps
            J[1, col] = (th1 - th0) / eps
        return J