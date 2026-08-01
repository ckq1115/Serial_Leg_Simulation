"""Finite-state jump scheduler for the wheel-leg controller.

The state machine only plans leg-length targets and extra axial force.
The controller remains responsible for PID force synthesis, LQR feedback,
VMC mapping, and actuator saturation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JumpCommand:
    active: bool
    state: str
    target_leg_length: float | None
    extra_force: float
    position_gain_scale: float
    velocity_gain_scale: float
    force_airborne: bool
    zero_wheels: bool


class JumpStateMachine:
    READY = "READY"
    CROUCH = "CROUCH"
    LAUNCH = "LAUNCH"
    TUCK = "TUCK"
    LANDING = "LANDING"
    RECOVER = "RECOVER"

    def __init__(self, dt, cfg, leg_length_min, leg_length_max):
        self.dt = float(dt)
        self.enabled = bool(cfg.get("enabled", False))
        self.leg_length_min = float(leg_length_min)
        self.leg_length_max = float(leg_length_max)

        self.l_crouch = self._clip_length(cfg.get("l_crouch_m", self.leg_length_min))
        self.l_launch = self._clip_length(cfg.get("l_launch_m", self.leg_length_max))
        self.l_tuck = self._clip_length(cfg.get("l_tuck_m", self.l_crouch))
        self.l_land = self._clip_length(cfg.get("l_land_m", 0.5 * (self.l_crouch + self.l_launch)))

        self.l_crouch_threshold = self._clip_length(
            cfg.get("l_crouch_threshold_m", self.l_crouch + 0.01)
        )
        self.launch_complete_tolerance = abs(float(
            cfg.get("launch_complete_tolerance_m", 0.01)
        ))
        launch_complete_length = self._clip_length(
            self.l_launch - self.launch_complete_tolerance
        )
        self.l_extend_threshold = self._clip_length(
            cfg.get("l_extend_threshold_m", launch_complete_length)
        )
        self.l_extend_threshold = max(self.l_extend_threshold, launch_complete_length)

        self.crouch_timeout_s = float(cfg.get("crouch_timeout_s", 0.25))
        self.launch_timeout_s = float(cfg.get("launch_timeout_s", 0.18))
        self.tuck_time_s = float(cfg.get("tuck_time_s", 0.12))
        self.landing_min_time_s = float(cfg.get("landing_min_time_s", 0.08))
        self.landing_timeout_s = float(cfg.get("landing_timeout_s", 0.60))
        self.recover_time_s = float(cfg.get("recover_time_s", 0.20))
        self.cooldown_s = float(cfg.get("cooldown_s", 0.50))

        self.liftoff_force_n = float(cfg.get("liftoff_force_n", 8.0))
        self.touchdown_force_n = float(cfg.get("touchdown_force_n", 30.0))

        self.position_gain_scale = float(cfg.get("position_gain_scale", 2.0))
        self.velocity_gain_scale = float(cfg.get("velocity_gain_scale", 1.5))
        self.f_crouch_n = float(cfg.get("f_crouch_n", -30.0))
        self.f_launch_n = float(cfg.get("f_launch_n", 0.0))
        self.f_tuck_n = float(cfg.get("f_tuck_n", -20.0))
        self.f_land_n = float(cfg.get("f_land_n", 0.0))

        self.reset()

    def reset(self):
        self.state = self.READY
        self.state_time = 0.0
        self.cooldown_time = 0.0
        self.landing_contact_time = 0.0
        self.return_length = self.l_land

    @property
    def is_active(self):
        return self.state != self.READY

    def update(self, trigger, current_target_length, left_length, right_length, fn_l, fn_r):
        current_target_length = self._clip_length(current_target_length)
        if not self.enabled:
            return self._ready_command(None)

        if self.cooldown_time > 0.0:
            self.cooldown_time = max(0.0, self.cooldown_time - self.dt)

        if self.state == self.READY:
            if trigger and self.cooldown_time <= 0.0:
                self.return_length = current_target_length
                self._enter(self.CROUCH)
            else:
                return self._ready_command(None)

        self._advance(left_length, right_length, fn_l, fn_r)
        command = self._command_for_state()
        self.state_time += self.dt
        return command

    def _advance(self, left_length, right_length, fn_l, fn_r):
        both_crouched = (left_length <= self.l_crouch_threshold
                         and right_length <= self.l_crouch_threshold)
        both_extended = (left_length >= self.l_extend_threshold
                         and right_length >= self.l_extend_threshold)
        both_touchdown = fn_l > self.touchdown_force_n and fn_r > self.touchdown_force_n

        if self.state == self.CROUCH:
            if both_crouched or self.state_time >= self.crouch_timeout_s:
                self._enter(self.LAUNCH)
        elif self.state == self.LAUNCH:
            if both_extended or self.state_time >= self.launch_timeout_s:
                self._enter(self.TUCK)
        elif self.state == self.TUCK:
            if self.state_time >= self.tuck_time_s:
                self._enter(self.LANDING)
        elif self.state == self.LANDING:
            if both_touchdown:
                self.landing_contact_time += self.dt
            else:
                self.landing_contact_time = 0.0
            if (self.landing_contact_time >= self.landing_min_time_s
                    or self.state_time >= self.landing_timeout_s):
                self._enter(self.RECOVER)
        elif self.state == self.RECOVER:
            if self.state_time >= self.recover_time_s:
                self._enter(self.READY)
                self.cooldown_time = self.cooldown_s

    def _enter(self, state):
        self.state = state
        self.state_time = 0.0
        if state == self.LANDING:
            self.landing_contact_time = 0.0

    def _command_for_state(self):
        if self.state == self.CROUCH:
            return self._active_command(self.l_crouch, self.f_crouch_n,
                                        force_airborne=False, zero_wheels=False)
        if self.state == self.LAUNCH:
            return self._active_command(self.l_launch, self.f_launch_n,
                                        force_airborne=False, zero_wheels=False)
        if self.state == self.TUCK:
            return self._active_command(self.l_tuck, self.f_tuck_n,
                                        force_airborne=True, zero_wheels=True)
        if self.state == self.LANDING:
            return self._active_command(self.l_land, self.f_land_n,
                                        force_airborne=True, zero_wheels=True)
        if self.state == self.RECOVER:
            return JumpCommand(True, self.state, self.return_length, 0.0,
                               1.0, 1.0, False, False)
        return self._ready_command(None)

    def _active_command(self, target_length, extra_force, force_airborne, zero_wheels):
        return JumpCommand(True, self.state, target_length, extra_force,
                           self.position_gain_scale, self.velocity_gain_scale,
                           force_airborne, zero_wheels)

    def _ready_command(self, target_length):
        return JumpCommand(False, self.READY, target_length, 0.0,
                           1.0, 1.0, False, False)

    def _clip_length(self, value):
        return float(np.clip(float(value), self.leg_length_min, self.leg_length_max))
