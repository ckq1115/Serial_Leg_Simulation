"""Serial wheel-leg simulation with an input-disturbance LESO.

This launcher leaves the original main.py and controller.py unchanged.  It
uses the scheduled 10-state, 4-input models produced by tools/linearize.py and
augments them with four constant matched disturbances.
"""

from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.linalg import solve_discrete_are

import main as base_main
from controller import StandController
from data.lqr_gain_fit import get_e3
from utils import load_config, resolve_project_path, require_actuator_order, PROJECT_ROOT


MODEL_TABLE = Path(__file__).resolve().parents[1] / "data" / "AB_sampling_points.npz"
DISTURBANCE_LIMIT = np.array([20.0, 20.0, 4.0, 4.0])
STATE_PROCESS_NOISE = 1e-5
DISTURBANCE_PROCESS_NOISE = 2e-3
MEASUREMENT_NOISE = 2e-3
DISTURBANCE_DECAY = 0.995


class LesoStandController(StandController):
    """Existing standing controller plus a gain-scheduled matched LESO."""

    def __init__(self, model, data, cfg, args):
        super().__init__(model, data, cfg, args)
        table = np.load(MODEL_TABLE)
        self.leso_lengths_left = np.asarray(table["L_l"], dtype=float)
        self.leso_lengths_right = np.asarray(table["L_r"], dtype=float)
        self.leso_ad = np.asarray(table["A_d"], dtype=float)
        self.leso_bd = np.asarray(table["B_d"], dtype=float)
        if self.leso_ad.shape[1:] != (10, 10) or self.leso_bd.shape[1:] != (10, 4):
            raise ValueError("LESO model table must contain 10-state, 4-input models")

        self.leso_state = np.zeros(14)
        self.leso_disturbance = np.zeros(4)
        self.leso_last_input = np.zeros(4)
        self.leso_initialized = False
        self.leso_gain_cache = {}

    def reset_internal_state(self):
        super().reset_internal_state()
        if hasattr(self, "leso_state"):
            self.leso_state[:] = 0.0
            self.leso_disturbance[:] = 0.0
            self.leso_last_input[:] = 0.0
            self.leso_initialized = False

    def _scheduled_model(self):
        left_length = self.left_leg.length
        right_length = self.right_leg.length
        distance = (
            (self.leso_lengths_left - left_length) ** 2
            + (self.leso_lengths_right - right_length) ** 2
        )
        index = int(np.argmin(distance))
        return index, self.leso_ad[index], self.leso_bd[index]

    def _observer_gain(self, index, ad, bd):
        cached = self.leso_gain_cache.get(index)
        if cached is not None:
            return cached

        ae = np.block([
            [ad, bd],
            [np.zeros((4, 10)), np.eye(4)],
        ])
        ce = np.hstack([np.eye(10), np.zeros((10, 4))])
        qe = np.diag(np.concatenate([
            np.full(10, STATE_PROCESS_NOISE),
            np.full(4, DISTURBANCE_PROCESS_NOISE),
        ]))
        re = np.eye(10) * MEASUREMENT_NOISE
        covariance = solve_discrete_are(ae.T, ce.T, qe, re)
        gain = ae @ covariance @ ce.T @ np.linalg.inv(ce @ covariance @ ce.T + re)
        self.leso_gain_cache[index] = (ae, ce, gain)
        return ae, ce, gain

    def _update_leso(self, x_lqr):
        e3_left, e3_right = get_e3(self.left_leg.length, self.right_leg.length)
        measurement = np.asarray(x_lqr, dtype=float).reshape(10).copy()
        measurement[3] -= e3_left
        measurement[4] -= e3_right

        if not self.leso_initialized:
            self.leso_state[:10] = measurement
            self.leso_initialized = True
            return

        left_contact, right_contact = self._wheel_contact_flags()
        if self.jump_state != "idle" or not (left_contact and right_contact):
            self.leso_state[:10] = measurement
            self.leso_state[10:] *= DISTURBANCE_DECAY
            self.leso_disturbance[:] = self.leso_state[10:]
            return

        index, ad, bd = self._scheduled_model()
        ae, ce, gain = self._observer_gain(index, ad, bd)
        be = np.vstack([bd, np.zeros((4, 4))])
        innovation = measurement - ce @ self.leso_state
        self.leso_state = ae @ self.leso_state + be @ self.leso_last_input + gain @ innovation
        self.leso_state[10:] = np.clip(
            self.leso_state[10:], -DISTURBANCE_LIMIT, DISTURBANCE_LIMIT
        )
        self.leso_disturbance[:] = self.leso_state[10:]

    def _map_to_actuators(self, force_left, force_right, x_lqr, gain_raw,
                          e3_left, e3_right, equilibrium_input, state):
        self._update_leso(x_lqr)

        gain = gain_raw.copy()
        adjusted_e3_left = e3_left
        adjusted_e3_right = e3_right
        suppress_wheel_left, suppress_wheel_right, suppress_leg_left, suppress_leg_right = \
            self.gc.get_K_suppression_mask()
        if suppress_wheel_left:
            gain[2, :] = 0.0
        if suppress_wheel_right:
            gain[3, :] = 0.0
        if suppress_leg_left:
            gain[0, :] = 0.0
            adjusted_e3_left = -e3_left
        if suppress_leg_right:
            gain[1, :] = 0.0
            adjusted_e3_right = -e3_right

        equilibrium = np.asarray(equilibrium_input, dtype=float).reshape(4, 1).copy()
        if suppress_leg_left:
            equilibrium[0, 0] = 0.0
        if suppress_leg_right:
            equilibrium[1, 0] = 0.0
        if suppress_wheel_left:
            equilibrium[2, 0] = 0.0
        if suppress_wheel_right:
            equilibrium[3, 0] = 0.0

        disturbance = self.leso_disturbance.reshape(4, 1).copy()
        if suppress_leg_left:
            disturbance[0, 0] = 0.0
        if suppress_leg_right:
            disturbance[1, 0] = 0.0
        if suppress_wheel_left:
            disturbance[2, 0] = 0.0
        if suppress_wheel_right:
            disturbance[3, 0] = 0.0

        target = self.build_lqr_target(adjusted_e3_left, adjusted_e3_right)
        generalized_input = equilibrium - gain @ (x_lqr - target) - disturbance
        self.U_lqr = generalized_input
        self.leso_last_input[:] = (generalized_input - equilibrium).reshape(4)

        left_front, left_back = self.left_leg.vmc(
            force_left, generalized_input[0, 0],
            state["left_front_pos"], state["left_back_pos"]
        )
        right_front, right_back = self.right_leg.vmc(
            force_right, generalized_input[1, 0],
            state["right_front_pos"], state["right_back_pos"]
        )
        actuator_input = np.array([
            left_front, right_front, left_back, right_back,
            generalized_input[2, 0], generalized_input[3, 0],
        ], dtype=float)
        actuator_input[:4] -= state["gas_hip_torque_nm"]
        actuator_input[:4] = np.clip(
            actuator_input[:4], -self.args.joint_torque_limit, self.args.joint_torque_limit
        )
        actuator_input[4:] = np.clip(
            actuator_input[4:], -self.args.wheel_torque_limit, self.args.wheel_torque_limit
        )
        left_contact, right_contact = self._wheel_contact_flags()
        if not left_contact:
            actuator_input[4] = 0.0
        if not right_contact:
            actuator_input[5] = 0.0
        return actuator_input


def main():
    args = base_main.parse_args()
    cfg = load_config()
    if args.fall_angle is None:
        args.fall_angle = cfg["control_initial"]["fall_angle_rad"]

    xml_path = resolve_project_path(cfg["mujoco"]["xml_path"])
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    base_main.apply_initial_disturbance(data, args)
    mj.mj_forward(model, data)

    actuator_order = require_actuator_order(model, cfg["mujoco"]["actuator_order"])
    controller = LesoStandController(model, data, cfg, args)
    print(f"Loaded {xml_path.relative_to(PROJECT_ROOT)}")
    print(f"actuator_order={actuator_order}")
    print("LESO enabled: 10 plant states + 4 matched input disturbances")
    print(f"LESO disturbance limits={DISTURBANCE_LIMIT.tolist()}")

    if args.viewer:
        ok = base_main.run_with_viewer(controller, args.duration, args.camera_mode)
    else:
        ok = base_main.run_headless(controller, args.duration)
    if ok:
        print(f"Finished {data.time:.3f}s without exceeding fall angle.")


if __name__ == "__main__":
    main()
