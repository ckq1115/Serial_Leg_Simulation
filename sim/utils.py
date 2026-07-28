from pathlib import Path
import yaml
import numpy as np
import mujoco as mj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "robot_params.yaml"

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_project_path(relative_path):
    return PROJECT_ROOT / relative_path

def sensor_value(data, name):
    return data.sensor(name).data.copy()

def quat_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def normalize_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

def clip_symmetric(value, limit):
    return float(np.clip(value, -limit, limit))

def approach_value(current, target, max_delta):
    if max_delta <= 0.0:
        return float(target)
    return float(current + np.clip(target - current, -max_delta, max_delta))

def mj_names(model, obj_type, count):
    names = []
    for i in range(count):
        name = mj.mj_id2name(model, obj_type, i)
        names.append(name if name else f"<unnamed_{i}>")
    return names

def require_actuator_order(model, expected):
    actual = mj_names(model, mj.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actual != expected:
        raise RuntimeError(f"Unexpected actuator order: expected={expected}, actual={actual}")
    return actual


class ContinuousAngle:
    """角度连续化解包器，用于将从 IMU 读到的 ±π 包裹角度展开为无跳变的连续角度。"""

    def __init__(self):
        self.last_wrapped = None
        self.value = 0.0

    def reset(self, wrapped_angle=0.0):
        self.last_wrapped = normalize_angle(wrapped_angle)
        self.value = self.last_wrapped

    def update(self, wrapped_angle):
        wrapped_angle = normalize_angle(wrapped_angle)
        if self.last_wrapped is None:
            self.reset(wrapped_angle)
            return self.value
        self.value += normalize_angle(wrapped_angle - self.last_wrapped)
        self.last_wrapped = wrapped_angle
        return self.value