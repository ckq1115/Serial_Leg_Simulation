import argparse
import time
import numpy as np
import mujoco as mj
from utils import load_config, resolve_project_path, require_actuator_order, PROJECT_ROOT
from controller import StandController
from keyboard_reader import KeyboardHoldReader

DEFAULT_VIEWER_CAMERA = "base_follow_camera"
# 视角切换键。避开 MuJoCo viewer 自带的 F/T/Q/S/V/R/H/P、Tab、Space、Esc、方向键等。
CAMERA_TOGGLE_KEY = ord("B")

def parse_args():
    parser = argparse.ArgumentParser(description="Serial five-link wheel-leg standing simulation (upgraded).")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds. 0 for infinite.")
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--viewer", dest="viewer", action="store_true", help="Launch MuJoCo passive viewer.")
    viewer_group.add_argument("--headless", dest="viewer", action="store_false", help="Run without viewer.")
    parser.set_defaults(viewer=True)
    parser.add_argument("--camera-mode", choices=["base", "free"], default="base",
                        help="Initial viewer camera. 'base' follows the robot body; 'free' uses MuJoCo free camera.")
    parser.add_argument("--control-mode", choices=["lqr", "keyboard"], default="keyboard")
    parser.add_argument("--target-leg-length", type=float, default=None)
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

def set_viewer_camera(viewer, model, mode, announce=True):
    if mode == "free":
        viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        if announce:
            print("Viewer camera: free MuJoCo view. Drag/zoom with the mouse; press B to switch to base camera.")
        return "free"

    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, DEFAULT_VIEWER_CAMERA)
    if camera_id < 0:
        print(f"Viewer camera not found: {DEFAULT_VIEWER_CAMERA}; falling back to free MuJoCo view.")
        return set_viewer_camera(viewer, model, "free", announce)

    viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id
    if announce:
        print("Viewer camera: base first-person view. Press B to switch to free MuJoCo view.")
    return "base"

def run_with_viewer(controller, duration, camera_mode):
    import mujoco.viewer
    unlimited = duration <= 0.0
    keyboard_reader = None
    viewer_state = {"viewer": None, "camera_mode": camera_mode}

    def key_callback(key):
        if key != CAMERA_TOGGLE_KEY or viewer_state["viewer"] is None:
            return False
        next_mode = "free" if viewer_state["camera_mode"] == "base" else "base"
        viewer_state["camera_mode"] = set_viewer_camera(
            viewer_state["viewer"], controller.model, next_mode
        )
        return True

    if controller.control_mode == "keyboard":
        keyboard_reader = KeyboardHoldReader()
        print("Keyboard controls: Arrow keys for speed/yaw, left/right Shift for leg length.")
    try:
        with mujoco.viewer.launch_passive(
            controller.model, controller.data, key_callback=key_callback
        ) as viewer:
            viewer_state["viewer"] = viewer
            viewer_state["camera_mode"] = set_viewer_camera(viewer, controller.model, camera_mode)

            while viewer.is_running() and (unlimited or controller.data.time < duration):
                step_start = time.time()
                axes = keyboard_reader.read_axes() if keyboard_reader is not None else None
                if not controller.step(axes):
                    print(f"Stopped: fall angle exceeded at t={controller.data.time:.3f}s")
                    break
                viewer.sync()
                sleep_time = controller.dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        viewer_state["viewer"] = None
        if keyboard_reader is not None:
            keyboard_reader.stop()

def main():
    args = parse_args()
    cfg = load_config()
    if args.fall_angle is None:
        args.fall_angle = cfg["control_initial"]["fall_angle_rad"]

    xml_path = resolve_project_path(cfg["mujoco"]["xml_path"])
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
        ok = run_headless(controller, args.duration)

    if ok:
        print(f"Finished {data.time:.3f}s without exceeding fall angle.")

if __name__ == "__main__":
    main()
