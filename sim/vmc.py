"""VMC 力矩映射（手册 §6）：任务空间力 → 关节力矩。

基于虚功原理 τ_joint = J^T · F，其中 F = [f_l, t_l]^T 为任务空间虚拟力/力矩，
J = ∂(L, θ_leg)/∂(front_angle, back_angle) 为雅可比矩阵。

雅可比解析式由五连杆几何直接展开。
"""

import numpy as np


class VmcMapper:
    """将任务空间力 (f_l, t_l) 映射为前后髋关节力矩 (τ_front, τ_back)。

    J^T 的解析式仅依赖几何参数（杆长）和当前姿态，不涉及质量/惯量。
    """

    def __init__(self, link_lengths):
        self.l_front_big = link_lengths["front_big_m"]
        self.l_back_big = link_lengths["back_big_m"]

    def compute(self, leg_length, leg_theta, phi_front, phi_back,
                leg_force, leg_torque, front_angle, back_angle):
        """τ_joint = J^T · [leg_force, leg_torque]^T。

        Parameters
        ----------
        leg_length, leg_theta : float
            虚拟腿长 L 和摆角 θ（来自运动学正解）。
        phi_front, phi_back : float
            前后连杆内角（来自运动学正解）。
        leg_force, leg_torque : float
            任务空间轴向力 f_l 和虚拟力矩 t_l。
        front_angle, back_angle : float
            前后髋关节角度（传感器直读）。

        Returns
        -------
        (front_torque, back_torque) : (float, float)
        """
        denom = np.sin(phi_front - phi_back)
        if abs(denom) < 1e-6:
            denom = np.copysign(1e-6, denom if denom != 0.0 else 1.0)

        back_torque = (
            leg_force
            * self.l_back_big
            * np.cos(leg_theta - phi_front)
            * np.sin(back_angle - phi_back)
            / denom
            + leg_torque
            * self.l_back_big
            * -np.sin(leg_theta - phi_front)
            * np.sin(back_angle - phi_back)
            / (leg_length * denom)
        )
        front_torque = (
            leg_force
            * self.l_front_big
            * np.cos(leg_theta - phi_back)
            * np.sin(phi_front - front_angle)
            / denom
            + leg_torque
            * self.l_front_big
            * -np.sin(leg_theta - phi_back)
            * np.sin(phi_front - front_angle)
            / (leg_length * denom)
        )
        return front_torque, back_torque
