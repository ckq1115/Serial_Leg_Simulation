import argparse
import numpy as np
import mujoco as mj
from utils import load_config, resolve_project_path, require_actuator_order, PROJECT_ROOT
from controller import StandController
from keyboard_reader import KeyboardHoldReader

def parse_args():
    parser = argparse.ArgumentParser(description="Serial five-link wheel-leg standing simulation (upgraded).")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds. 0 for infinite.")
    parser.add_argument("--scene", type=str, default=None, help="Scene XML path relative to the project root.")
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--viewer", dest="viewer", action="store_true", help="Launch MuJoCo passive viewer.")
    viewer_group.add_argument("--headless", dest="viewer", action="store_false", help="Run without viewer.")
    parser.set_defaults(viewer=True)
    parser.add_argument("--camera-mode", choices=["base", "free"], default="free",
                        help="Initial viewer camera. 'base' follows the robot body; 'free' uses MuJoCo free camera.")
    parser.add_argument("--control-mode", choices=["lqr", "keyboard"], default="keyboard")
    parser.add_argument("--target-leg-length", type=float, default=None,
                        help="Target leg length in meters. Defaults to 0.22 m if omitted.")
    parser.add_argument("--support-force", type=float, default=None)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-yaw", type=float, default=None)
    parser.add_argument("--target-pitch", type=float, default=None)
    parser.add_argument("--target-left-leg-theta", type=float, default=None)
    parser.add_argument("--target-right-leg-theta", type=float, default=None)
    parser.add_argument("--target-x-dot", type=float, default=None)
    parser.add_argument("--target-yaw-dot", type=float, default=None)
    parser.add_argument("--lqr-gain-table", type=str, default=None)
    parser.add_argument("--max-keyboard-speed", type=float, default=None)
    parser.add_argument("--max-keyboard-yaw-rate", type=float, default=None)
    parser.add_argument("--keyboard-speed-accel", type=float, default=None)
    parser.add_argument("--keyboard-speed-release-accel", type=float, default=None)
    parser.add_argument("--keyboard-yaw-rate-accel", type=float, default=None)
    parser.add_argument("--keyboard-yaw-rate-release-accel", type=float, default=None)
    parser.add_argument("--initial-pitch", type=float, default=0.0)
    parser.add_argument("--initial-x-velocity", type=float, default=0.0)
    parser.add_argument("--push-body", type=str, default="base")
    parser.add_argument("--push-time", type=float, default=1.0)
    parser.add_argument("--push-duration", type=float, default=0.0)
    parser.add_argument("--push-force-x", type=float, default=0.0)
    parser.add_argument("--push-force-y", type=float, default=0.0)
    parser.add_argument("--push-force-z", type=float, default=0.0)
    parser.add_argument("--push-torque-x", type=float, default=0.0)
    parser.add_argument("--push-torque-y", type=float, default=0.0)
    parser.add_argument("--push-torque-z", type=float, default=0.0)
    parser.add_argument("--joint-torque-limit", type=float, default=40.0)
    parser.add_argument("--wheel-torque-limit", type=float, default=5.0)
    parser.add_argument("--length-position-force-limit", type=float, default=1500.0)
    parser.add_argument("--length-velocity-force-limit", type=float, default=1000.0)
    parser.add_argument("--roll-force-limit", type=float, default=1000.0)
    parser.add_argument("--fall-angle", type=float, default=None)
    parser.add_argument("--log-every", type=float, default=0.05)
    return parser.parse_args()

def apply_initial_disturbance(data, args):
    if args.initial_pitch != 0.0:
        half_pitch = 0.5 * args.initial_pitch
        data.qpos[3:7] = np.array([np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0])
    if args.initial_x_velocity != 0.0:
        data.qvel[0] = args.initial_x_velocity

def run_headless(controller, duration):
    unlimited = duration <= 0.0
    while unlimited or controller.data.time < duration:
        if not controller.step():
            print(f"Stopped: fall angle exceeded at t={controller.data.time:.3f}s")
            return False
    return True

def run_with_viewer(controller, duration, camera_mode):
    import mujoco.viewer
    keyboard = KeyboardHoldReader(consume_viewer_arrows=True) if controller.control_mode == "keyboard" else None
    controller.native_managed = True
    last_time = [None]
    cached = np.zeros(controller.model.nu)
    def control(model, data):
        controller.model, controller.data = model, data
        if last_time[0] is None or abs(data.time - last_time[0]) > 1e-12:
            axes = keyboard.read_axes() if keyboard is not None else None
            controller.step(axes)
            cached[:] = data.ctrl
            last_time[0] = data.time
        else:
            data.ctrl[:] = cached
    print("MuJoCo native viewer: native Pause/Run, Reset, camera, rendering groups and mouse perturbation.")
    print("J jumps once per press; Space pauses/runs. Collision groups use the native Rendering panel.")
    mj.set_mjcb_control(control)
    try:
        mujoco.viewer.launch(controller.model, controller.data)
    finally:
        mj.set_mjcb_control(None)
        controller.recovery_log.finish("viewer_closed")
        if keyboard is not None: keyboard.stop()
    return True


def main():
    args = parse_args()
    cfg = load_config()
    if args.fall_angle is None:
        args.fall_angle = cfg["control_initial"]["fall_angle_rad"]

    xml_path = resolve_project_path(args.scene or cfg["mujoco"]["xml_path"])
    expected_actuator_order = cfg["mujoco"]["actuator_order"]

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    apply_initial_disturbance(data, args)
    mj.mj_forward(model, data)

    actuator_order = require_actuator_order(model, expected_actuator_order)
    controller = StandController(model, data, cfg, args)

    print(f"Loaded {xml_path.relative_to(PROJECT_ROOT)}")
    print(f"actuator_order={actuator_order}")
    print(f"control_mode={controller.control_mode}")
    print(f"target_leg_length={controller.target_leg_length:.6f} m")
    print(f"lqr_target: x={controller.target_x:.3f}, x_dot={controller.target_x_dot:.3f}, "
          f"yaw_dot={controller.target_yaw_dot:.3f}, pitch={controller.target_pitch:.3f}, "
          f"left_theta={controller.target_left_leg_theta:.3f}, right_theta={controller.target_right_leg_theta:.3f}")
    print(f"support_force_per_leg={controller.support_force:.3f} N")

    if args.viewer:
        ok = run_with_viewer(controller, args.duration, args.camera_mode)
    else:
        try:
            ok = run_headless(controller, args.duration)
        finally:
            controller.recovery_log.finish("simulation_stopped")

    if ok:
        print(f"Finished {data.time:.3f}s without exceeding fall angle.")

if __name__ == "__main__":
    main()
