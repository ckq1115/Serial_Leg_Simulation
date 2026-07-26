import numpy as np

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
        self.force = leg_force
        l_front_big = self.link_lengths["front_big_m"]
        l_back_big = self.link_lengths["back_big_m"]
        denom = np.sin(self.phi_front - self.phi_back)
        if abs(denom) < 1e-6:
            denom = np.copysign(1e-6, denom if denom != 0.0 else 1.0)

        self.back_torque = (
            leg_force
            * l_back_big
            * np.cos(self.theta - self.phi_front)
            * np.sin(back_angle - self.phi_back)
            / denom
            + leg_torque
            * l_back_big
            * -np.sin(self.theta - self.phi_front)
            * np.sin(back_angle - self.phi_back)
            / (self.length * denom)
        )
        self.front_torque = (
            leg_force
            * l_front_big
            * np.cos(self.theta - self.phi_back)
            * np.sin(self.phi_front - front_angle)
            / denom
            + leg_torque
            * l_front_big
            * -np.sin(self.theta - self.phi_back)
            * np.sin(self.phi_front - front_angle)
            / (self.length * denom)
        )
        return self.front_torque, self.back_torque