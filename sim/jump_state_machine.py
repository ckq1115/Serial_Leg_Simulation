"""跳跃控制状态机（手册 §10）——基于力控的脉冲驱动跳跃。

手册定义的六阶段模型（§10.1）：
  READY → CROUCH → LAUNCH → TUCK → LANDING → RECOVER

当前实现为简化三阶段版本：
  ready → extending（合并 CROUCH+LAUNCH）→ landing → ready

空中 LQR 降维（§10.3）和完整六阶段迁移留待后续迭代。
"""

import numpy as np


class JumpStateMachine:
    """基于力控的脉冲驱动跳跃状态机（手册 §10）。

    States (simplified from manual §10.1 six-phase model):
      ready     — 待命站立  (manual: READY)
      extending — 爆发伸展  (manual: CROUCH + LAUNCH, 合并)
      landing   — 着陆缓冲  (manual: LANDING + RECOVER, 合并)

    Public API:
      trigger(pitch, roll)  — 尝试从 ready 触发跳跃
      update(L_l, L_r)      — 每控制步推进状态机
      get_jump_force()       — 返回当前阶段脉冲力
      get_target_leg_length(L_stand) — 返回当前阶段目标腿长
    """

    def __init__(self, dt, cfg=None):
        if cfg is None:
            cfg = {}
        self.dt = dt

        geo = cfg.get("geometry", {})
        five = geo.get("five_link", {})
        self.L_min = float(five.get("min_length_m", 0.12))
        self.L_max = float(five.get("max_length_m", 0.35))
        self.L_stand = float(five.get("nominal_length_m", 0.194))

        jp = cfg.get("jump", {})
        if not jp:
            jp = cfg.get("control_initial", {}).get("jump", {})
        self.jump_force = float(jp.get("jump_force_N", 2000.0))
        self.extend_time = float(jp.get("extend_time_s", 0.30))
        self.cooldown_s = float(jp.get("cooldown_s", 0.5))

        self.state = "ready"
        self.jump_active = False
        self._cooldown_timer = 0.0
        self._phase_timer = 0.0

    @property
    def state_name(self):
        return self.state

    def trigger(self, pitch=0.0, roll=0.0):
        if self.state != "ready":
            return
        if self._cooldown_timer > 0.0:
            return
        if abs(pitch) > 1.0 or abs(roll) > 1.0:
            return
        self.state = "extending"
        self.jump_active = True
        self._phase_timer = 0.0

    def release(self):
        pass

    def arm(self):
        pass

    def update(self, left_length, right_length):
        self._phase_timer += self.dt
        if self._cooldown_timer > 0.0:
            self._cooldown_timer = max(0.0, self._cooldown_timer - self.dt)

        mx = max(left_length, right_length)
        mn = min(left_length, right_length)
        avg = 0.5 * (left_length + right_length)

        if self.state == "extending":
            if mn >= self.L_max or self._phase_timer > self.extend_time:
                self.state = "landing"
                self._phase_timer = 0.0

        elif self.state == "landing":
            if self._phase_timer > 0.5 and abs(avg - self.L_stand) < 0.05:
                self.state = "ready"
                self.jump_active = False
                self._cooldown_timer = self.cooldown_s
                self._phase_timer = 0.0

    def get_target_leg_length(self, stand_length):
        if self.state == "landing":
            return self.L_stand
        return stand_length  # extending: don't change target, pulse does the work

    def get_jump_force(self):
        if self.state == "extending":
            return self.jump_force
        return 0.0

    def bypass_pos_pid(self):
        """Keep position PID active — natural tapering of force as leg extends."""
        return False

    def bypass_vel_pid(self):
        """Bypass during extending: velocity damping would fight the extension."""
        return self.state == "extending"

    def bypass_roll_pid(self):
        return False  # always active — maintain roll

    def zero_gravity_comp(self):
        return False  # always on — support during impulse and landing
