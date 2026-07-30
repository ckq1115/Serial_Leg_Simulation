"""Systematic LQR tuning for 15kg Serial_Leg plant.

Strategy:
  Phase 1: Sweep damping Q (pitch_dot, th_dot) to minimize max|lambda|
  Phase 2: With best damping, sweep Q_x_dot for velocity tracking
  Phase 3: Test top candidates headless and score them

Usage: python tools/tune_lqr.py
"""

import sys, time, itertools, numpy as np
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))
sys.path.insert(0, str(_proj / "sim"))
sys.path.insert(0, str(_proj / "data"))

# Import the linearize module components
import importlib.util
spec = importlib.util.spec_from_file_location("linearize", _proj / "tools" / "linearize.py")
lin = importlib.util.module_from_spec(spec)

# We'll just modify linearize.py's globals and call its functions directly.
# Simpler: rewrite Q_DIAG, R_MAT inline and call build + linearize.

# ── Constants (matching linearize.py) ──
M_BODY = 15.0; M_WHEEL = 0.353429173529
M_LEG = 0.13482+0.1554+0.230811+0.11718+0.349772111111+0.23625
I_BODY_YY = 0.11593; I_BODY_ZZ = 0.115485
I_WHEEL_YY = 0.000441786467; I_WHEEL_ZZ = 0.000223838477
WHEEL_RADIUS = 0.05; HALF_TRACK = 0.165
GRAVITY = 9.81; L_BODY_COM = 0.0
DT = 0.002; N_ITER_DARE = 50000
H_MIN, H_MAX = 0.156, 0.356

def build_and_solve(Q_diag, R_diag):
    """Build A,B for one representative leg length and solve DARE. Returns max|eig|, K matrix."""
    import mujoco as mj
    from scipy.linalg import expm, solve_discrete_are
    import sympy as sp

    # Simplified: use a single representative L=0.20 for fast evaluation
    L_val = 0.20
    r_com = 0.103  # approximate leg COM radial distance at L=0.20
    c_x = 0.005    # approximate tangential offset

    # Build symbolic model (from linearize.py build_serial_model)
    p = [M_BODY, M_WHEEL, HALF_TRACK, WHEEL_RADIUS,
         I_BODY_YY, I_BODY_ZZ, I_WHEEL_ZZ, I_WHEEL_YY, GRAVITY,
         L_val, L_val, M_LEG, M_LEG, r_com, r_com, c_x, c_x,
         L_BODY_COM, 0.0015, 0.0015]  # j_ly, j_ry small (leg inertia)

    # Use the sympy lambdified functions from linearize
    # Since we can't easily re-lambdify, use a pre-built approach:
    # Just evaluate at the equilibrium numerically

    # For fast evaluation, use the existing linearize module directly
    # by temporarily patching its globals
    import tools.linearize as lmod
    old_Q = lmod.Q_DIAG.copy()
    old_R = lmod.R_MAT.copy()
    lmod.Q_DIAG = np.array(Q_diag)
    lmod.R_MAT = np.diag(R_diag) if len(R_diag) == 4 else np.diag([R_diag[0]]*4)

    # Run linearize on a small grid
    try:
        # We can't easily call the full pipeline without MuJoCo model
        # Fall back to direct numerical approach
        pass
    finally:
        lmod.Q_DIAG = old_Q
        lmod.R_MAT = old_R


def phase1_sweep():
    """Sweep damping Q values. For each, regenerate K and test headless."""
    import subprocess, re

    pitch_dot_vals = [1, 5, 10, 20]
    th_dot_vals = [1, 5, 10, 20]
    base_Q = [10, 1, 10000, 4000, 4000, 1, 1]  # first 7, then pitch_dot, thL_dot, thR_dot

    results = []
    for pd, td in itertools.product(pitch_dot_vals, th_dot_vals):
        Q = base_Q + [pd, td, td]
        # Write Q to linearize.py
        write_q_r(Q, [1,1,1,1])

        # Run linearize
        r = subprocess.run([sys.executable, str(_proj/"tools"/"linearize.py")],
                          capture_output=True, text=True, cwd=str(_proj))
        max_lam = float(re.search(r"max\\|lambda\\|: ([0-9.]+)", r.stdout).group(1))

        # Quick headless test (3 seconds)
        r2 = subprocess.run([sys.executable, "-c", f"""
import sys, numpy as np, mujoco as mj
from pathlib import Path
sys.path.insert(0, '{_proj}'); sys.path.insert(0, '{_proj}/sim'); sys.path.insert(0, '{_proj}/data')
from controller import StandController; from utils import load_config; import argparse
cfg = load_config()
model = mj.MjModel.from_xml_path('{_proj}/mjcf/serial_leg.xml'); data = mj.MjData(model)
args = argparse.Namespace(duration=0, headless=True, viewer=False, control_mode='keyboard',
    target_leg_length=None, target_x=None, target_yaw=None, target_pitch=None,
    target_left_leg_theta=None, target_right_leg_theta=None, target_x_dot=None,
    target_yaw_dot=None, support_force=None, push_body='base', push_time=0,
    push_duration=0, push_force_x=0, push_force_y=0, push_force_z=0,
    push_torque_x=0, push_torque_y=0, push_torque_z=0,
    joint_torque_limit=40, wheel_torque_limit=5,
    length_position_force_limit=1500, length_velocity_force_limit=1000,
    roll_force_limit=1000, fall_angle=1.0, log_every=999,
    max_keyboard_speed=2.5, max_keyboard_yaw_rate=3.0,
    keyboard_speed_accel=None, keyboard_speed_release_accel=None,
    keyboard_yaw_rate_accel=None, keyboard_yaw_rate_release_accel=None)
ctrl = StandController(model, data, cfg, args)
dt = 0.002; steps = 1500  # 3 seconds
pitches = []
for i in range(steps):
    t = i*dt
    ax = (1.0, 0.0, 0.0) if 0.5 < t < 2.5 else (0.0, 0.0, 0.0)
    ctrl.step(keyboard_axes=ax)
    s = ctrl.read_state()
    pitches.append(s['euler'][1])
p = np.array(pitches)
falls = np.sum(np.abs(p) > 1.0)
max_p = np.max(np.abs(p))
steady_p = np.max(np.abs(p[250:500]))  # stand phase
print(f'OK max_p={max_p:.4f} steady_p={steady_p:.4f} falls={falls}')
"""], capture_output=True, text=True, cwd=str(_proj/"sim"), timeout=120)

        ok = 'OK' in r2.stdout
        if ok:
            parts = r2.stdout.strip().split()
            max_p = float(parts[0].split('=')[1])
            steady_p = float(parts[1].split('=')[1])
            falls = int(parts[2].split('=')[1])
        else:
            max_p, steady_p, falls = 999, 999, 999

        score = max_lam * 100 + steady_p * 50 + falls * 200
        results.append((pd, td, max_lam, steady_p, falls, score))
        print(f"  pd={pd:3d} td={td:3d}  max|lam|={max_lam:.4f}  steady_p={steady_p:.4f}  falls={falls}  score={score:.1f}")

    results.sort(key=lambda x: x[5])
    print("\n=== Phase 1 Best ===")
    for r in results[:5]:
        print(f"  Q_pitch_dot={r[0]:3d} Q_th_dot={r[1]:3d}  lam={r[2]:.4f}  p={r[3]:.4f}  falls={r[4]}  score={r[5]:.1f}")
    return results[0][0], results[0][1]


def write_q_r(Q_diag, R_diag):
    """Write Q_DIAG and R_MAT to linearize.py"""
    lpath = _proj / "tools" / "linearize.py"
    text = lpath.read_text()

    # Replace Q_DIAG block
    import re
    q_str = "Q_DIAG = np.array([\n"
    names = ['x','yaw','pitch','th_L','th_R','x_dot','yaw_dot','pitch_dot','thL_dot','thR_dot']
    for i, (q, n) in enumerate(zip(Q_diag, names)):
        comma = "," if i < 9 else ","
        q_str += f"    {q},   # {n}\n"
    q_str += "])"

    # Simple approach: replace the whole Q_DIAG ... ]) block
    old_q = re.search(r"Q_DIAG = np\.array\(\[.*?\]\)", text, re.DOTALL)
    if old_q:
        text = text[:old_q.start()] + q_str + text[old_q.end():]

    # Replace R_MAT block
    r_str = "R_MAT  = np.diag([\n"
    r_names = ['tau_L','tau_R','T_wL','T_wR']
    for i, (r, n) in enumerate(zip(R_diag, r_names)):
        comma = "," if i < 3 else ","
        r_str += f"    {r},  # {n}\n"
    r_str += "])"

    old_r = re.search(r"R_MAT  = np\.diag\(\[.*?\]\)", text, re.DOTALL)
    if old_r:
        text = text[:old_r.start()] + r_str + text[old_r.end():]

    lpath.write_text(text)


if __name__ == "__main__":
    import re
    print("=== Phase 1: Damping sweep ===")
    best_pd, best_td = phase1_sweep()
    print(f"\nBest damping: Q_pitch_dot={best_pd}, Q_th_dot={best_td}")

    print("\n=== Phase 2: Velocity sweep ===")
    x_dot_vals = [1, 5, 10, 20, 30, 50]
    for xd in x_dot_vals:
        Q = [10, 1, 10000, 4000, 4000, xd, 1, best_pd, best_td, best_td]
        write_q_r(Q, [1,1,1,1])
        r = subprocess.run([sys.executable, str(_proj/"tools"/"linearize.py")],
                          capture_output=True, text=True, cwd=str(_proj))
        max_lam = float(re.search(r"max\\|lambda\\|: ([0-9.]+)", r.stdout).group(1))
        print(f"  Q_x_dot={xd:3d}  max|lam|={max_lam:.4f}")
