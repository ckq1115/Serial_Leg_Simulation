class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0.0
        self.last_err = 0.0
        self.integral = 0.0
        self.output = 0.0

    def reset(self):
        self.err = 0.0
        self.last_err = 0.0
        self.integral = 0.0
        self.output = 0.0

    def update(self, target, feedback):
        self.last_err = self.err
        self.err = target - feedback
        self.integral += self.err
        self.output = self.kp * self.err + self.ki * self.integral + self.kd * (self.err - self.last_err)
        return self.output