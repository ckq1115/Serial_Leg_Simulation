import numpy as np

class WheelOdometry:
    def __init__(self, wheel_radius, dt):
        self.wheel_radius = wheel_radius
        self.dt = dt
        self.x = 0.0
        self.x_dot = 0.0

    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0

    def update(self, left_wheel_vel, right_wheel_vel):
        self.x_dot = 0.5 * self.wheel_radius * (left_wheel_vel + right_wheel_vel)
        self.x += self.x_dot * self.dt
        return self.x, self.x_dot