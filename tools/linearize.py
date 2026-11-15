"""
Serial Lagrangian + MuJoCo geometry → 2D LQR gain table.

Uses SymPy Lagrangian dynamics + MuJoCo LegPoseSolver for leg COM geometry.
Outputs 2D Chebyshev gain fits to data/lqr_gain_fit.py and data/k_table_2d.h.

Usage:
  python tools/linearize.py                    # 2D Chebyshev table
  python tools/linearize.py --grid-size 27     # finer grid
  python tools/linearize.py --order 8  --grid-size 10
  python tools/linearize.py --leso-check-grid 81  # denser optional pole check
  python tools/linearize.py --wheel-load-only    # 检查左右轮静态法向载荷
"""

import sys, time, argparse, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import fsolve, brentq, least_squares, minimize
import mujoco as mj
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from utils import load_config, resolve_project_path  # type: ignore[import-untyped] 

M_BODY = 20.8775740
M_WHEEL = 0.4904
M_LEG   = 1.1132+0.306873+0.058433+0.042999+0.065877+0.010945
I_BODY_XX = 0.3595155; I_BODY_YY = 0.35249; I_BODY_ZZ = 0.45655
I_WHEEL_YY = 0.0005817; I_WHEEL_ZZ = 0.00033958
WHEEL_RADIUS = 0.06; HALF_TRACK = 0.1877
GRAVITY = 9.81; BODY_COM = np.zeros(3)
DT = 0.002; N_ITER_DARE = 100000
H_MIN, H_MAX = 0.10, 0.356
LESO_PLANT_STATES = 10
LESO_INPUTS = 4
LESO_AUG_STATES = LESO_PLANT_STATES + LESO_INPUTS
LESO_STATE_PROCESS_NOISE = 1e-5
LESO_DISTURBANCE_PROCESS_NOISE = 1e-5

# LESO 的测量标准差按下方状态顺序排列，平方后作为测量协方差的对角线。
LESO_MEASUREMENT_STD = np.array([
    2.0e-3,  # x [m], odometry/Kalman position
    2.0e-3,  # yaw [rad], IMU integrated yaw
    1.5e-3,  # pitch [rad], IMU attitude
    1.0e-2,  # theta_L [rad], leg kinematics
    1.0e-2,  # theta_R [rad], leg kinematics
    2.0e-2,  # x_dot [m/s], wheel/IMU fused velocity
    1.0e-2,  # yaw_dot [rad/s], gyroscope
    1.0e-2,  # pitch_dot [rad/s], gyroscope
    1.0e-2,  # theta_L_dot [rad/s], differentiated leg angle
    1.0e-2,  # theta_R_dot [rad/s], differentiated leg angle
], dtype=float)
LESO_MEASUREMENT_VARIANCE = LESO_MEASUREMENT_STD ** 2
LESO_DENSE_CHECK_GRID = 17

# 离线求解容差应高于数值差分的噪声底。
POSE_SOLVE_TOL = 1e-7
POSE_MAX_NFEV = 300
POSE_RESIDUAL_TOL_M = 1e-4  # 0.1 mm closure/leg-length error
EQUILIBRIUM_SOLVE_TOL = 1e-8
EQUILIBRIUM_MAX_ITER = 100
EQUILIBRIUM_RESIDUAL_TOL = 1e-5


# 降阶模型的广义阻尼，顺序为位移、偏航、俯仰、左腿角、右腿角。
DAMPING_Q = np.array([0.1, 0.1, 0.0, 0.1, 0.1])
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Q 的对角线按位移、偏航、俯仰、左右腿角及对应速度排列。
Q_DIAG = np.array([
    36.4960615194,      # 位移
    280.2831482354,     # 偏航角
    48363.1968069981,   # 俯仰角
    3016.0253765672,    # 左腿角
    3016.0253765672,    # 右腿角
    18.9416149683,      # 纵向速度
    9.9043106757,       # 偏航角速度
    3.5156691657,       # 俯仰角速度
    0.6061526386,       # 左腿角速度
    0.6061526386,       # 右腿角速度
])

# R 的输入顺序为左右虚拟腿摆矩、左右轮矩。
R_MAT = np.array([
    [9.7885694276, 9.4525021292, 0.0,          0.0],
    [9.4525021292, 9.7885694276, 0.0,          0.0],
    [0.0,          0.0,          4.5336677586, 0.7377569067],
    [0.0,          0.0,          0.7377569067, 4.5336677586],
], dtype=float)


class LegPoseSolver:
    def __init__(self, model, cfg):
        self.model = model
        self.leg_geom = cfg["geometry"]["five_link"]
        self.wheel_radius = cfg["geometry"]["wheel"]["radius_m"]
        # 从 MuJoCo 的 connect 约束读取局部铰点，保留左右安装板偏置。
        self._closures = [(int(model.eq_obj1id[i]), int(model.eq_obj2id[i]),
                           model.eq_data[i,:3].copy(),model.eq_data[i,3:6].copy())
                          for i in range(model.neq)
                          if model.eq_type[i] == mj.mjtEq.mjEQ_CONNECT]
        if not self._closures:
            raise ValueError('The robot needs native connect constraints for its closed chains')
        self._leg_body_ids = {}
        for side in ['left','right']:
            roots = {int(model.jnt_bodyid[model.joint(cfg['mujoco']['active_joints'][side+'_'+role]).id])
                     for role in ['front','back']}
            wheel = int(model.jnt_bodyid[model.joint(cfg['mujoco']['wheel_joints'][side]).id])
            ids=[]
            for bid in range(1,model.nbody):
                parent=bid
                while parent and parent not in roots:
                    parent=int(model.body_parentid[parent])
                if parent in roots and bid != wheel:ids.append(bid)
            if not ids:raise ValueError(f'No moving leg bodies found for {side}')
            self._leg_body_ids[side]=ids
        self._wheel_qpos = np.array([
            self._joint_qpos(cfg["mujoco"]["wheel_joints"]["left"]),
            self._joint_qpos(cfg["mujoco"]["wheel_joints"]["right"]),
        ], dtype=int)
        leg_bodies = set(self._leg_body_ids["left"] + self._leg_body_ids["right"])
        self._pose_qpos = np.array([
            2,
            *[int(model.jnt_qposadr[joint_id])
              for joint_id in range(1, model.njnt)
              if model.jnt_bodyid[joint_id] in leg_bodies],
        ], dtype=int)
        self._active_joints = cfg["mujoco"]["active_joints"]
        self._wheel_joints = cfg["mujoco"]["wheel_joints"]

    def _joint_qpos(self, joint_name):
        joint_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"joint not found: {joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_anchor(self, data, joint_name):
        return joint_anchor_world(self.model, data, joint_name)

    def _hip_midpoint(self, data, side):
        front = self._joint_anchor(
            data, self._active_joints[f"{side}_front"])
        back = self._joint_anchor(
            data, self._active_joints[f"{side}_back"])
        return 0.5 * (front + back)

    def _wheel_center(self, data, side):
        return self._joint_anchor(data, self._wheel_joints[side])

    def solve(self, data, h, seed=None):
        qpos = self.model.qpos0.copy() if seed is None else seed.copy()
        wz = self.wheel_radius-0.001
        qpos[:3]=[0,0,h+wz]; qpos[3:7]=[1,0,0,0]
        qpos[self._wheel_qpos] = 0.0
        y0 = qpos[self._pose_qpos]
        lo = np.r_[0.04, np.full(self._pose_qpos.size-1, -np.pi)]
        hi = np.r_[0.60, np.full(self._pose_qpos.size-1, np.pi)]

        def res(y):
            q=qpos.copy(); q[self._pose_qpos]=y
            data.qpos[:]=q; data.qvel[:]=0; data.ctrl[:]=0
            mj.mj_kinematics(self.model,data)
            r=[]
            for b1,b2,a1,a2 in self._closures:
                p1=data.xpos[b1]+data.xmat[b1].reshape(3,3)@a1
                p2=data.xpos[b2]+data.xmat[b2].reshape(3,3)@a2
                r.extend(1000*(p1-p2)[[0,2]])
            for side in ("left", "right"):
                hip = self._hip_midpoint(data, side)
                wheel = self._wheel_center(data, side)
                length = np.hypot(wheel[0]-hip[0], wheel[2]-hip[2])
                r.append(1000*(length-h))
                r.append(1000*(wheel[0]-hip[0]))
                r.append(1000*(wheel[2]-wz))
            return np.asarray(r)

        candidates = [y0]
        if seed is not None:
            candidates.append(self.model.qpos0[self._pose_qpos].copy())
            candidates[-1][0] = h+wz
        # Continue from the previous length first. A second cold solve is only
        # needed when that pose misses the physical residual tolerance.
        # LSMR regularizes the underdetermined pose fit's trust-region steps.
        solutions = []
        for candidate in candidates:
            result = least_squares(
                res, candidate, bounds=(lo, hi),
                xtol=POSE_SOLVE_TOL, ftol=POSE_SOLVE_TOL, gtol=POSE_SOLVE_TOL,
                max_nfev=POSE_MAX_NFEV, tr_solver="lsmr")
            solutions.append(result)
            if np.all(np.isfinite(result.fun)) and (
                    np.max(np.abs(result.fun)) / 1000.0 <= POSE_RESIDUAL_TOL_M):
                break
        sol = min(solutions, key=lambda item: np.max(np.abs(item.fun)))
        max_residual_m = float(np.max(np.abs(sol.fun))) / 1000.0
        if not np.isfinite(max_residual_m) or max_residual_m > POSE_RESIDUAL_TOL_M:
            raise RuntimeError(
                f"pose solve failed at L={h:.6f} m: "
                f"max residual={max_residual_m:.3e} m, {sol.message}")

        qpos[self._pose_qpos]=sol.x
        data.qpos[:]=qpos; data.qvel[:]=0; data.ctrl[:]=0
        mj.mj_kinematics(self.model,data)
        return qpos.copy()

    def leg_com(self, data, side):
        """Return the legacy reduced-model COM parameters (r_com, c_x)."""
        hip = self._hip_midpoint(data, side)
        tot=0.0; com=np.zeros(3)
        for bid in self._leg_body_ids[side]:
            m=self.model.body_mass[bid]
            # xipos is already the world-space inertial-frame COM.
            tot+=m; com+=m*data.xipos[bid]
        com/=tot; d=com-hip
        return float(-d[2]), float(d[0])

    def leg_inertia_diagonal(self, data, side):
        ids=self._leg_body_ids[side]
        mass=self.model.body_mass[ids]
        center=np.sum(mass[:,None]*data.xipos[ids],axis=0)/np.sum(mass)
        inertia=np.zeros((3,3))
        for bid in ids:
            r=data.ximat[bid].reshape(3,3)
            tensor=r@np.diag(self.model.body_inertia[bid])@r.T
            delta=data.xipos[bid]-center
            inertia+=tensor+self.model.body_mass[bid]*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
        return np.diag(inertia).copy()

    def leg_inertia(self, data, side):
        return float(self.leg_inertia_diagonal(data,side)[1])

    def leg_lateral_offset(self,data,side):
        ids=self._leg_body_ids[side];mass=self.model.body_mass[ids]
        com=np.sum(mass[:,None]*data.xipos[ids],axis=0)/np.sum(mass)
        midpoint=.5*(self._hip_midpoint(data,'left')+self._hip_midpoint(data,'right'))
        return float((com-midpoint)[1])


def chassis_body_ids(model, base):
    return [base, *[int(body) for body in np.flatnonzero(model.body_parentid == base)
                    if model.body(int(body)).name.startswith('guide_wheel_')]]


def load_model_parameters(model,data,cfg,solver):
    """Use the imported robot's physical data instead of the old CAD constants."""
    global M_BODY,M_WHEEL,M_LEG,I_BODY_XX,I_BODY_YY,I_BODY_ZZ,I_WHEEL_YY,I_WHEEL_ZZ
    global WHEEL_RADIUS,HALF_TRACK,GRAVITY,BODY_COM,DT
    base=model.body(cfg['mujoco']['base_body']).id
    base_r=data.xmat[base].reshape(3,3)
    chassis = chassis_body_ids(model, base)
    mass = model.body_mass[chassis]
    M_BODY = float(np.sum(mass))
    center = np.average(data.xipos[chassis], axis=0, weights=mass)
    inertia = np.zeros((3, 3))
    for bid, body_mass in zip(chassis, mass):
        rotation = base_r.T @ data.ximat[bid].reshape(3, 3)
        offset = base_r.T @ (data.xipos[bid] - center)
        inertia += rotation @ np.diag(model.body_inertia[bid]) @ rotation.T
        inertia += body_mass * (np.dot(offset, offset)*np.eye(3) - np.outer(offset, offset))
    I_BODY_XX,I_BODY_YY,I_BODY_ZZ=np.diag(inertia)
    wheels=[int(model.jnt_bodyid[model.joint(cfg['mujoco']['wheel_joints'][s]).id]) for s in ['left','right']]
    M_WHEEL=float(np.mean(model.body_mass[wheels]))
    wi=[]
    for bid in wheels:
        r=base_r.T@data.ximat[bid].reshape(3,3)
        wi.append(r@np.diag(model.body_inertia[bid])@r.T)
    I_WHEEL_YY=float(np.mean([a[1,1] for a in wi]));I_WHEEL_ZZ=float(np.mean([a[2,2] for a in wi]))
    M_LEG=float(np.mean([np.sum(model.body_mass[ids]) for ids in solver._leg_body_ids.values()]))
    # Hip plates are staggered; wheel centers define the vehicle track width.
    HALF_TRACK=float(abs((base_r.T@(solver._wheel_center(data,'left')-solver._wheel_center(data,'right')))[1])/2)
    WHEEL_RADIUS=float(cfg['geometry']['wheel']['radius_m'])
    GRAVITY=float(np.linalg.norm(model.opt.gravity));DT=float(cfg['simulation']['control_period_s'])
    offset=body_com_offset_from_hip(model,data,cfg)
    BODY_COM=offset.copy()
    print(f'  Robot parameters: body={M_BODY:.6f} kg, leg={M_LEG:.6f} kg/side, wheel={M_WHEEL:.6f} kg, track={2*HALF_TRACK:.6f} m, dt={DT:.6f} s')
    return offset


def joint_anchor_world(model, data, joint_name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"joint not found: {joint_name}")
    bid = model.jnt_bodyid[jid]
    return data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ model.jnt_pos[jid]


def body_com_offset_from_hip(model, data, cfg):
    """Body COM offset in the base frame, measured from the four hip anchors."""
    base_name = cfg["mujoco"].get("base_body", "base")
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
    if base_id < 0:
        raise ValueError(f"base body not found: {base_name}")

    active_joints = cfg["mujoco"]["active_joints"]
    hip_points = np.array([
        joint_anchor_world(model, data, active_joints["left_front"]),
        joint_anchor_world(model, data, active_joints["left_back"]),
        joint_anchor_world(model, data, active_joints["right_front"]),
        joint_anchor_world(model, data, active_joints["right_back"]),
    ])
    hip_mid = np.mean(hip_points, axis=0)
    base_rot = data.xmat[base_id].reshape(3, 3)
    chassis = chassis_body_ids(model, base_id)
    center = np.average(data.xipos[chassis], axis=0, weights=model.body_mass[chassis])
    return base_rot.T @ (center - hip_mid)


def build_serial_model(with_wheel_loads=False):
    """Build the local bilateral-support model with coupled Yaw/leg inertia."""

    x_s,w_s,th_s,thl_s,thr_s = sp.symbols('x w theta theta_l theta_r')
    xd_s,wd_s,thd_s,thld_s,thrd_s = sp.symbols('xd wd thetad thetald thetard')
    xdd_s,wdd_s,thdd_s,thldd_s,thrdd_s = sp.symbols('xdd wdd thetadd thetaldd thetardd')
    Ml2,Mr2,Ml3,Mr3 = sp.symbols('Ml2 Mr2 Ml3 Mr3')

    m,m_w,d,r = sp.symbols('m m_w d r')
    g = sp.symbols('g')
    bx,by,bz = sp.symbols('body_cx body_cy body_cz')
    j_y,j_z,j_wy,j_wz = sp.symbols('j_y j_z j_wy j_wz')
    j_x,j_lx,j_lz,j_rx,j_rz=sp.symbols('j_x j_lx j_lz j_rx j_rz')
    y_l,y_r=sp.symbols('leg_y_l leg_y_r')
    L_l,L_r = sp.symbols('L_l L_r'); ml,mr = sp.symbols('ml mr')
    c_l,c_r = sp.symbols('c_l c_r'); cx_l,cx_r = sp.symbols('cx_l cx_r')
    j_ly,j_ry = sp.symbols('j_ly j_ry')

    # x is the HIP midpoint position, matching the controller's body odometry.
    # theta_i > 0 means the hip is ahead of its wheel; pitch is +Y rotation.
    h_l=L_l*sp.cos(thl_s);h_r=L_r*sp.cos(thr_s)
    # 双轮同高支撑时，两腿竖直投影之差决定横滚几何。
    bank=sp.atan((h_l-h_r)/(2*d))
    bank_rate=sp.diff(bank,thl_s)*thld_s+sp.diff(bank,thr_s)*thrd_s
    hip_z=(h_l+h_r)/2*sp.cos(bank)
    # 求接触反力时，髋轴高度和横滚先作为独立坐标。
    height, roll = sp.symbols('height roll')
    height_rate, roll_rate = sp.symbols('height_rate roll_rate')
    q_full = sp.Matrix([x_s, w_s, th_s, thl_s, thr_s, height, roll])
    qd_full = sp.Matrix([xd_s, wd_s, thd_s, thld_s, thrd_s, height_rate, roll_rate])
    support = {height: hip_z, roll: bank, height_rate:
               sp.diff(hip_z, thl_s)*thld_s + sp.diff(hip_z, thr_s)*thrd_s,
               roll_rate: bank_rate}
    wheel_gap = sp.Matrix([height + d*sp.sin(roll) - h_l*sp.cos(roll),
                           height - d*sp.sin(roll) - h_r*sp.cos(roll)])
    axle_x=-(L_l*sp.sin(thl_s)+L_r*sp.sin(thr_s))/2
    wheel_terms=[];leg_terms=[]
    for side,L,angle,rate,mass,c,cx,j,y in [
        (1,L_l,thl_s,thld_s,ml,c_l,cx_l,j_ly,y_l),
        (-1,L_r,thr_s,thrd_s,mr,c_r,cx_r,j_ry,y_r)]:
        wvx=xd_s-side*d*wd_s-L*sp.cos(angle)*rate
        wheel_terms.append(sp.Rational(1,2)*(m_w+j_wy/r**2)*wvx**2)
        rx=cx*sp.cos(angle)-c*sp.sin(angle)
        rz=-cx*sp.sin(angle)-c*sp.cos(angle)
        vx=xd_s-y*wd_s+sp.diff(rx,angle)*rate
        point_z=height+y*sp.sin(roll)+rz*sp.cos(roll)
        vz=sp.Matrix([point_z]).jacobian(q_full).dot(qd_full)
        leg_terms.append(sp.Rational(1,2)*mass*(vx**2+vz**2)+sp.Rational(1,2)*j*rate**2)
    body_rx=bx*sp.cos(th_s)+bz*sp.sin(th_s)
    body_rz=-bx*sp.sin(th_s)+bz*sp.cos(th_s)
    body_vx=xd_s-by*wd_s+sp.diff(body_rx,th_s)*thd_s
    body_z=height+by*sp.sin(roll)+body_rz*sp.cos(roll)
    body_vz=sp.Matrix([body_z]).jacobian(q_full).dot(qd_full)
    # 在冻结腿长的直立模型中计入约束诱导的横滚动能。
    roll_inertia=j_x+m*(height+body_rz)**2
    # Forward velocity squares above already contain lateral-offset inertia
    # and Yaw/leg-rate cross terms. Add only lateral translation and local
    # rotational inertia here, without counting the wheel track twice.
    yaw_inertia=j_x*sp.sin(th_s)**2+j_z*sp.cos(th_s)**2+m*(body_rx-axle_x)**2
    for angle,mass,c,cx,jx,jz,y in [(thl_s,ml,c_l,cx_l,j_lx,j_lz,y_l),(thr_s,mr,c_r,cx_r,j_rx,j_rz,y_r)]:
        rx=cx*sp.cos(angle)-c*sp.sin(angle)
        rz=-cx*sp.sin(angle)-c*sp.cos(angle)
        roll_inertia+=jx*sp.cos(angle)**2+jz*sp.sin(angle)**2+mass*(height+rz)**2
        yaw_inertia+=jx*sp.sin(angle)**2+jz*sp.cos(angle)**2+mass*(rx-axle_x)**2
    yaw_inertia+=2*j_wz
    # 偏航惯量以支撑轮轴中点为基准。
    T_full=(sum(wheel_terms)+sum(leg_terms)
       +sp.Rational(1,2)*m*(body_vx**2+body_vz**2)
       +sp.Rational(1,2)*j_y*thd_s**2
       +sp.Rational(1,2)*roll_inertia*roll_rate**2
       +sp.Rational(1,2)*yaw_inertia*wd_s**2)
    rz_l=-cx_l*sp.sin(thl_s)-c_l*sp.cos(thl_s)
    rz_r=-cx_r*sp.sin(thr_s)-c_r*sp.cos(thr_s)
    V_full=g*(m*body_z
              + ml*(height+y_l*sp.sin(roll)+rz_l*sp.cos(roll))
              + mr*(height+y_r*sp.sin(roll)+rz_r*sp.cos(roll)))
    # 代入双轮支撑约束，得到原有五自由度的能量表达式。
    T = T_full.xreplace(support)
    V = V_full.xreplace(support)

    L_expr = T - V
    def ddL(Li,qi,qdi,qddi):
        dL=sp.diff(Li,qdi); ddt=0
        for qj,qdj,qddj in [(x_s,xd_s,xdd_s),(w_s,wd_s,wdd_s),(th_s,thd_s,thdd_s),
                              (thl_s,thld_s,thldd_s),(thr_s,thrd_s,thrdd_s)]:
            ddt+=sp.diff(dL,qj)*qdj+sp.diff(dL,qdj)*qddj
        return ddt

    # E-L - Q = 0  (so linear_eq_to_matrix gives M>0, Bq with correct sign)
    eq1 = (ddL(L_expr,x_s,xd_s,xdd_s)-sp.diff(L_expr,x_s)) - (Ml3+Mr3)/r
    eq2 = (ddL(L_expr,w_s,wd_s,wdd_s)-sp.diff(L_expr,w_s)) - (-Ml3+Mr3)*d/r
    eq3 = (ddL(L_expr,th_s,thd_s,thdd_s)-sp.diff(L_expr,th_s)) - (-(Ml2+Mr2))
    eq4 = (ddL(L_expr,thl_s,thld_s,thldd_s)-sp.diff(L_expr,thl_s)) - (Ml2-(1+L_l*sp.cos(thl_s)/r)*Ml3)
    eq5 = (ddL(L_expr,thr_s,thrd_s,thrdd_s)-sp.diff(L_expr,thr_s)) - (Mr2-(1+L_r*sp.cos(thr_s)/r)*Mr3)

    accel = [xdd_s,wdd_s,thdd_s,thldd_s,thrdd_s]
    M_sym, rhs_sym = sp.linear_eq_to_matrix([eq1,eq2,eq3,eq4,eq5], accel)

    u_vars = [Ml2,Mr2,Ml3,Mr3]
    Bq_sym = sp.zeros(5,4)
    for i in range(5):
        for j,uv in enumerate(u_vars):
            Bq_sym[i,j] = sp.diff(rhs_sym[i], uv)

    q_vars  = [x_s,w_s,th_s,thl_s,thr_s]
    qd_vars = [xd_s,wd_s,thd_s,thld_s,thrd_s]
    state_vars = q_vars+qd_vars
    param_vars = [m,m_w,d,r,j_y,j_z,j_wy,j_wz,g,
                  L_l,L_r,ml,mr,c_l,c_r,cx_l,cx_r,bx,by,bz,j_ly,j_ry,j_x,j_lx,j_lz,j_rx,j_rz,y_l,y_r]

    print("  Lambdifying M, Bq, V ...")
    M_func  = sp.lambdify(param_vars+state_vars, M_sym, 'numpy')
    Bq_func = sp.lambdify(param_vars+state_vars, Bq_sym, 'numpy')
    V_func  = sp.lambdify(param_vars+q_vars, V, 'numpy')
    if not with_wheel_loads:
        return M_func, Bq_func, V_func

    # 轮心高度以支撑面上的轮心高度为零。轮子的竖向动能和势能
    # 在约束流形上均为零，但势能的法向导数给出轮子自重。
    J_normal = wheel_gap.jacobian(q_full)
    wheel_vertical_speed = J_normal @ qd_full
    T_full += m_w * wheel_vertical_speed.dot(wheel_vertical_speed) / 2
    V_full += g * m_w * sum(wheel_gap)
    momentum = sp.Matrix([T_full]).jacobian(qd_full).T
    mass_full = momentum.jacobian(qd_full)
    bias_full = (momentum.jacobian(q_full) @ qd_full
                 - sp.Matrix([T_full - V_full]).jacobian(q_full).T)
    contact_bias = wheel_vertical_speed.jacobian(q_full) @ qd_full
    # 在约束流形上取值，但保留对独立高度和横滚坐标求导的结果。
    contact_terms = sp.lambdify(
        param_vars + state_vars,
        tuple(expr.xreplace(support) for expr in (mass_full, bias_full, J_normal, contact_bias)),
        'numpy', cse=True)

    def wheel_loads(parameters, q, qd, u):
        """返回双轮同高接触所需的左、右法向力，单位 N。"""
        mass, bias, jacobian, contact_accel = contact_terms(*parameters, *q, *qd)
        generalized_force = np.zeros(7)
        generalized_force[:5] = (
            Bq_func(*parameters, *q, *qd) @ u - DAMPING_Q * qd)
        # 法向向上。负反力表示该侧需要地面拉住轮子，双轮支撑不成立。
        system = np.block([[mass, -jacobian.T],
                           [jacobian, np.zeros((2, 2))]])
        rhs = np.r_[generalized_force - bias.ravel(), -contact_accel.ravel()]
        return np.linalg.solve(system, rhs)[-2:]

    return M_func, Bq_func, V_func, wheel_loads


def linearize_at_length(L_l, L_r, r_com_l, c_x_l, r_com_r, c_x_r,
                        M_func, Bq_func, V_func, eq_stats=None,
                        leg_inertia_l=0.003,leg_inertia_r=0.003,
                        leg_axes_l=(.003,.003),leg_axes_r=(.003,.003),
                        leg_y_l=None,leg_y_r=None, wheel_load_check=None):
    p = [M_BODY,M_WHEEL,HALF_TRACK,WHEEL_RADIUS,
         I_BODY_YY,I_BODY_ZZ,I_WHEEL_YY,I_WHEEL_ZZ,GRAVITY,
         L_l,L_r, M_LEG,M_LEG, r_com_l,r_com_r,c_x_l,c_x_r,
         *BODY_COM, leg_inertia_l,leg_inertia_r,I_BODY_XX,*leg_axes_l,*leg_axes_r,
         HALF_TRACK if leg_y_l is None else leg_y_l,-HALF_TRACK if leg_y_r is None else leg_y_r]

    def potential_gradient(q, epsilon=1.0e-6):
        q = np.asarray(q, dtype=float)
        gradient = np.zeros(5)
        for index in range(5):
            delta = np.zeros(5)
            delta[index] = epsilon
            gradient[index] = (
                V_func(*p, *(q+delta))-V_func(*p, *(q-delta))
            ) / (2.0*epsilon)
        return gradient

    def equilibrium_terms(variables):
        q = np.array([0.0, 0.0, 0.0, variables[0], variables[1]])
        state = np.r_[q, np.zeros(5)]
        gravity = potential_gradient(q)
        input_matrix = np.asarray(Bq_func(*p, *state), dtype=float)
        return gravity, input_matrix

    def equilibrium_residual(variables):
        gravity, input_matrix = equilibrium_terms(variables)
        return gravity-input_matrix@variables[2:]

    def equilibrium_objective(variables):
        return float(variables[2:]@variables[2:]
                     + 1.0e-4*variables[:2]@variables[:2])

    initial_angles = np.zeros(2)
    initial_gravity, initial_input = equilibrium_terms(
        np.r_[initial_angles, np.zeros(4)])
    initial_input_value = np.linalg.lstsq(
        initial_input, initial_gravity, rcond=None)[0]
    initial = np.r_[initial_angles, initial_input_value]
    equilibrium = minimize(
        equilibrium_objective, initial, method="SLSQP",
        bounds=[(-1.2, 1.2), (-1.2, 1.2),
                (-100.0, 100.0), (-100.0, 100.0),
                (-100.0, 100.0), (-100.0, 100.0)],
        constraints={"type": "eq", "fun": equilibrium_residual},
        options={"ftol": EQUILIBRIUM_SOLVE_TOL,
                 "maxiter": EQUILIBRIUM_MAX_ITER, "disp": False})
    residual = equilibrium_residual(equilibrium.x)
    res_norm = float(np.linalg.norm(residual, ord=np.inf))
    converged = np.isfinite(res_norm) and res_norm <= EQUILIBRIUM_RESIDUAL_TOL
    if eq_stats is not None:
        eq_stats["count"] += 1
        if res_norm > eq_stats["max_residual"]:
            eq_stats["max_residual"] = res_norm
            eq_stats["worst"] = (
                float(L_l), float(L_r), int(equilibrium.status),
                str(equilibrium.message).strip(), res_norm)
        if not converged:
            eq_stats["nonconverged"] += 1
    if not converged:
        raise RuntimeError(
            f"equilibrium solve failed at L=({L_l:.6f},{L_r:.6f}): "
            f"max|G-Bu|={res_norm:.3e}, {equilibrium.message}")

    th_eq = 0.0
    thl_eq, thr_eq = map(float, equilibrium.x[:2])
    u_eq = np.asarray(equilibrium.x[2:], dtype=float)

    q_eq=[0,0,th_eq,thl_eq,thr_eq]; qd_eq=[0,0,0,0,0]; se=q_eq+qd_eq
    if wheel_load_check is not None:
        wheel_loads, samples = wheel_load_check
        samples.append(wheel_loads(p, q_eq, qd_eq, u_eq))
    M =np.array(M_func(*p,*se),dtype=float)
    Bq=np.array(Bq_func(*p,*se),dtype=float)

    eps=1e-5; Kg=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            qpp=list(q_eq);qpp[i]+=eps;qpp[j]+=eps; qpm=list(q_eq);qpm[i]+=eps;qpm[j]-=eps
            qmp=list(q_eq);qmp[i]-=eps;qmp[j]+=eps; qmm=list(q_eq);qmm[i]-=eps;qmm[j]-=eps
            Kg[i,j]=(V_func(*p,*qpp)-V_func(*p,*qpm)-V_func(*p,*qmp)+V_func(*p,*qmm))/(4*eps*eps)

    Mi=np.linalg.inv(M)
    A=np.zeros((10,10)); A[0:5,5:10]=np.eye(5)
    A[5:10,0:5]=-Mi@Kg
    A[5:10,5:10]=-Mi@np.diag(DAMPING_Q)
    B=np.zeros((10,4)); B[5:10,:]=Mi@Bq

    return A, B, u_eq, th_eq, thl_eq, thr_eq


def discretize_zoh(A,B,dt):
    n,m=A.shape[0],B.shape[1]; M=np.zeros((n+m,n+m)); M[:n,:n]=A; M[:n,n:]=B; M*=dt; E=expm(M)
    return E[:n,:n],E[:n,n:]


def design_supported_gain(Ad,Bd,Q,R):
    """Joint ten-state LQR with four physical torque inputs.

    Hip cooperation is determined by the coupled plant, with no online
    steering-reaction feedforward or manually zeroed gain columns.
    """
    return solve_dare(Ad, Bd, Q, R)[0]


def solve_dare(Ad, Bd, Q, R, S=None, n_iter=N_ITER_DARE):
    """Solve discrete ARE via Schur decomposition (numerically stable).
    Falls back to backward iteration if Schur solver fails.
    """
    try:
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.solve(Bd.T @ P @ Bd + R, Bd.T @ P @ Ad)
        return K, P
    except (np.linalg.LinAlgError, ValueError):
        # fallback: backward iteration with warm start from Q
        if S is None:
            S = Q
        P = S.copy()
        for _ in range(max(n_iter, 50000)):
            F = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
            Pn = (Ad - Bd @ F).T @ P @ (Ad - Bd @ F) + F.T @ R @ F + Q
            if np.linalg.norm(Pn - P, 'fro') / max(np.linalg.norm(P, 'fro'), 1e-15) < 1e-10:
                P = Pn
                break
            P = Pn
        return np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad), P


def design_leso_gain(Ad, Bd):
    """Design the predictor-form gain for a matched-disturbance LESO.

    The augmented state is [x(10), d(4)], with d(k+1)=d(k) and the plant
    driven by Bd @ (u + d). All ten plant states are measured.
    """
    if Ad.shape != (LESO_PLANT_STATES, LESO_PLANT_STATES):
        raise ValueError(f"Expected Ad shape (10, 10), got {Ad.shape}")
    if Bd.shape != (LESO_PLANT_STATES, LESO_INPUTS):
        raise ValueError(f"Expected Bd shape (10, 4), got {Bd.shape}")

    Ae = np.block([
        [Ad, Bd],
        [np.zeros((LESO_INPUTS, LESO_PLANT_STATES)), np.eye(LESO_INPUTS)],
    ])
    Ce = np.hstack([
        np.eye(LESO_PLANT_STATES),
        np.zeros((LESO_PLANT_STATES, LESO_INPUTS)),
    ])
    Qe = np.diag(np.concatenate([
        np.full(LESO_PLANT_STATES, LESO_STATE_PROCESS_NOISE),
        np.full(LESO_INPUTS, LESO_DISTURBANCE_PROCESS_NOISE),
    ]))
    if LESO_MEASUREMENT_STD.shape != (LESO_PLANT_STATES,):
        raise ValueError(
            "LESO_MEASUREMENT_STD must contain one value per plant state"
        )
    if np.any(~np.isfinite(LESO_MEASUREMENT_STD)) or np.any(LESO_MEASUREMENT_STD <= 0.0):
        raise ValueError("LESO measurement standard deviations must be finite and positive")
    Re = np.diag(LESO_MEASUREMENT_VARIANCE)
    covariance = solve_discrete_are(Ae.T, Ce.T, Qe, Re)
    innovation_covariance = Ce @ covariance @ Ce.T + Re
    gain_numerator = Ae @ covariance @ Ce.T
    gain = np.linalg.solve(innovation_covariance, gain_numerator.T).T
    return gain


def normalize_length(L):
    """Map leg length from [H_MIN, H_MAX] to the Chebyshev domain [-1, 1]."""
    return (2.0 * L - H_MIN - H_MAX) / (H_MAX - H_MIN)


def cheb_values(p, x):
    """Evaluate T_0(x) ... T_p(x)."""
    vals = np.empty(p + 1, dtype=float)
    vals[0] = 1.0
    if p >= 1:
        vals[1] = x
    for k in range(2, p + 1):
        vals[k] = 2.0 * x * vals[k - 1] - vals[k - 2]
    return vals


def poly2d_terms(p, L_l, L_r):
    """Evaluate triangular 2D Chebyshev terms up to total order p."""
    T_l = cheb_values(p, normalize_length(L_l))
    T_r = cheb_values(p, normalize_length(L_r))
    out = []
    for d in range(p + 1):
        for i in range(d + 1):
            out.append(T_l[d - i] * T_r[i])
    return np.array(out, dtype=float)


def poly2d_design_matrix(p, L_l_vec, L_r_vec):
    """Build Chebyshev design matrix where each row is poly2d_terms(...)."""
    N = len(L_l_vec)
    n_terms = (p + 1) * (p + 2) // 2
    A = np.zeros((N, n_terms))
    for k in range(N):
        A[k, :] = poly2d_terms(p, L_l_vec[k], L_r_vec[k])
    return A


def fit_poly_2d(p, L_l_vec, L_r_vec, K_2d):
    """Fit p-th order triangular 2D Chebyshev basis to each K[i,j].

    K_2d shape: (n_samples, n_out, n_state)
    Returns coeff shape: (n_out, n_state, n_terms)
    """
    n_samples = len(L_l_vec)
    n_terms = (p + 1) * (p + 2) // 2
    no, ns = K_2d.shape[1], K_2d.shape[2]
    coeff = np.zeros((no, ns, n_terms))
    A = poly2d_design_matrix(p, L_l_vec, L_r_vec)
    for i in range(no):
        for j in range(ns):
            coeff[i, j, :] = np.linalg.lstsq(A, K_2d[:, i, j], rcond=None)[0]
    return coeff


def eval_poly_2d(p, coeff, L_l, L_r):
    """Evaluate 2D Chebyshev fit at (L_l, L_r). coeff shape: (n_out, n_state, n_terms)."""
    terms = poly2d_terms(p, L_l, L_r)
    no, ns = coeff.shape[0], coeff.shape[1]
    K = np.zeros((no, ns))
    for i in range(no):
        for j in range(ns):
            K[i, j] = np.dot(coeff[i, j, :], terms)
    return K


def assess_2d_fit(p, coeff, L_l_vec, L_r_vec, K_ref):
    """Compute max relative RMS error across all K elements."""
    no, ns = coeff.shape[0], coeff.shape[1]
    n_samples = len(L_l_vec)
    errors = np.zeros((no, ns, n_samples))
    rms_ref = np.zeros((no, ns))
    for k in range(n_samples):
        K_fit = eval_poly_2d(p, coeff, L_l_vec[k], L_r_vec[k])
        for i in range(no):
            for j in range(ns):
                errors[i, j, k] = (K_fit[i, j] - K_ref[k, i, j])
                rms_ref[i, j] += K_ref[k, i, j] ** 2
    rms_ref = np.sqrt(rms_ref / n_samples)
    rms_err = np.sqrt(np.mean(errors ** 2, axis=2))
    rel_err = rms_err / np.maximum(rms_ref, 1e-12)
    return rel_err


def export_c_header_2d(p, coeff, filepath):
    """Export 2D Chebyshev gains as C header for MCU.

    Format: flat float32 array, runtime eval function included.
    """
    no, ns, n_terms = coeff.shape
    flat = coeff.ravel()  # row-major: coeff[i,j,k] → index = i*ns*n_terms + j*n_terms + k
    n_floats = len(flat)
    n_kb = n_floats * 4 / 1024

    lines = []
    lines.append(f"// Auto-generated {p}th-order 2D Chebyshev LQR gain table for asymmetric leg lengths.")
    lines.append(f"// {n_floats} float32 coefficients")
    lines.append("// clang-format off")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define K2D_ORDER {p}")
    lines.append(f"#define K2D_N_TERMS {n_terms}")
    lines.append(f"#define K2D_N_OUTPUTS {no}")
    lines.append(f"#define K2D_N_STATES  {ns}")
    lines.append(f"#define K2D_H_MIN {H_MIN:.8f}f")
    lines.append(f"#define K2D_H_MAX {H_MAX:.8f}f")
    lines.append("")
    lines.append(f"static const float k_table_2d[{n_floats}] = {{")

    chunk = 8
    for start in range(0, n_floats, chunk):
        vals = flat[start:start + chunk]
        line = "    " + ", ".join(f"{v:12.8f}f" for v in vals)
        if start + chunk < n_floats:
            line += ","
        lines.append(line)
    lines.append("};")
    lines.append("")

    lines.append("// Evaluate K_2d(L_left, L_right) -> K[4][10] stored row-major in k_out[40]")
    lines.append("static inline void k_table_2d_eval(float L_left, float L_right, float *k_out) {")
    lines.append("    float xl = (2.0f * L_left - K2D_H_MIN - K2D_H_MAX) / (K2D_H_MAX - K2D_H_MIN);")
    lines.append("    float xr = (2.0f * L_right - K2D_H_MIN - K2D_H_MAX) / (K2D_H_MAX - K2D_H_MIN);")
    lines.append("    float Tl[K2D_ORDER + 1], Tr[K2D_ORDER + 1];")
    lines.append("    Tl[0] = 1.0f; Tr[0] = 1.0f;")
    lines.append("    if (K2D_ORDER >= 1) {")
    lines.append("        Tl[1] = xl; Tr[1] = xr;")
    lines.append("    }")
    lines.append("    for (int i = 2; i <= K2D_ORDER; i++) {")
    lines.append("        Tl[i] = 2.0f * xl * Tl[i - 1] - Tl[i - 2];")
    lines.append("        Tr[i] = 2.0f * xr * Tr[i - 1] - Tr[i - 2];")
    lines.append("    }")
    lines.append(f"    // Triangular Chebyshev terms: T_left[d-a] * T_right[a] for d=0..{p}")
    lines.append(f"    float terms[{n_terms}];")
    idx = 0
    for d in range(p + 1):
        for a in range(d + 1):
            b = d - a
            lines.append(f"    terms[{idx}] = Tl[{b}] * Tr[{a}];")
            idx += 1
    lines.append("")
    lines.append(f"    for (int i = 0; i < {no}; i++) {{")
    lines.append(f"        for (int j = 0; j < {ns}; j++) {{")
    lines.append(f"            float sum = 0.0f;")
    lines.append(f"            int base = (i * {ns} + j) * {n_terms};")
    lines.append(f"            for (int t = 0; t < {n_terms}; t++) {{")
    lines.append(f"                sum += k_table_2d[base + t] * terms[t];")
    lines.append(f"            }}")
    lines.append(f"            k_out[i * {ns} + j] = sum;")
    lines.append(f"        }}")
    lines.append(f"    }}")
    lines.append("}")
    lines.append("// clang-format on")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  C header -> {filepath}  ({n_floats} floats = {n_kb:.2f} KB)")


def export_c_header_ueq(p, coeff, filepath):
    """Export 2D Chebyshev equilibrium input table (4 outputs) for MCU."""
    no, _, n_terms = coeff.shape
    flat = coeff[:, 0, :].ravel()
    lines = ["// Auto-generated equilibrium input table; do not edit.",
             "#pragma once", "#include \"k_table_2d.h\"", "",
             f"#define UEQ_N_OUTPUTS {no}", f"#define UEQ_N_TERMS {n_terms}",
             f"static const float ueq_table_2d[{no * n_terms}] = {{"]
    for start in range(0, len(flat), 8):
        vals = flat[start:start + 8]
        suffix = "," if start + 8 < len(flat) else ""
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in vals) + suffix)
    lines += ["};", "", "static inline void ueq_table_2d_eval(float L_left, float L_right, float *u_eq) {",
              "    float xl = (2.0f * L_left - K2D_H_MIN - K2D_H_MAX) / (K2D_H_MAX - K2D_H_MIN);",
              "    float xr = (2.0f * L_right - K2D_H_MIN - K2D_H_MAX) / (K2D_H_MAX - K2D_H_MIN);",
              "    float Tl[K2D_ORDER + 1], Tr[K2D_ORDER + 1];",
              "    Tl[0] = 1.0f; Tr[0] = 1.0f;",
              "    for (int i = 1; i <= K2D_ORDER; i++) {",
              "        if (i == 1) { Tl[i] = xl; Tr[i] = xr; }",
              "        else { Tl[i] = 2.0f * xl * Tl[i - 1] - Tl[i - 2]; Tr[i] = 2.0f * xr * Tr[i - 1] - Tr[i - 2]; }",
              "    }", f"    float terms[{n_terms}];"]
    idx = 0
    for degree in range(p + 1):
        for right_degree in range(degree + 1):
            lines.append(f"    terms[{idx}] = Tl[{degree - right_degree}] * Tr[{right_degree}];")
            idx += 1
    lines += [f"    for (int i = 0; i < {no}; i++) {{", "        float sum = 0.0f;",
              f"        for (int t = 0; t < {n_terms}; t++) sum += ueq_table_2d[i * {n_terms} + t] * terms[t];",
              "        u_eq[i] = sum;", "    }", "}"]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  C header -> {filepath}  ({len(flat)} floats)")


def export_c_header_e3(coeff, filepath):
    """Export the legacy quartic leg-angle equilibrium offset."""
    values = np.asarray(coeff, dtype=float).ravel()
    if values.size != 5:
        raise ValueError(f"Expected five quartic e3 coefficients, got {values.size}")
    lines = ["// Auto-generated quartic leg-angle equilibrium offset; do not edit.",
             "#pragma once", "", "static const float e3_coeff[5] = {",
             "    " + ", ".join(f"{value:.8f}f" for value in values),
             "};", "", "static inline float e3_eval(float length) {",
             "    return (((e3_coeff[0] * length + e3_coeff[1]) * length + e3_coeff[2]) * length + e3_coeff[3]) * length + e3_coeff[4];",
             "}"]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  C header -> {filepath}  (5 floats)")


def export_c_header_leso(p, coeff_ad, coeff_bd, coeff_gain, filepath):
    """Export scheduled Ad, Bd and 14x10 LESO gain tables for MCU use."""
    n_terms = (p + 1) * (p + 2) // 2
    expected_shapes = {
        "Ad": (LESO_PLANT_STATES, LESO_PLANT_STATES, n_terms),
        "Bd": (LESO_PLANT_STATES, LESO_INPUTS, n_terms),
        "gain": (LESO_AUG_STATES, LESO_PLANT_STATES, n_terms),
    }
    coefficients = {"Ad": coeff_ad, "Bd": coeff_bd, "gain": coeff_gain}
    for name, expected in expected_shapes.items():
        if coefficients[name].shape != expected:
            raise ValueError(
                f"Expected {name} coefficient shape {expected}, got {coefficients[name].shape}"
            )

    def append_array(lines, c_name, values):
        flat = np.asarray(values, dtype=float).ravel()
        lines.append(f"static const float {c_name}[{flat.size}] = {{")
        for start in range(0, flat.size, 6):
            chunk = flat[start:start + 6]
            suffix = "," if start + 6 < flat.size else ""
            lines.append("    " + ", ".join(f"{value:.9e}f" for value in chunk) + suffix)
        lines.append("};")
        lines.append("")

    total_floats = coeff_ad.size + coeff_bd.size + coeff_gain.size
    lines = [
        "// Auto-generated gain-scheduled LESO tables; do not edit.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define LESO_TABLE_ORDER {p}",
        f"#define LESO_TABLE_N_TERMS {n_terms}",
        f"#define LESO_PLANT_STATES {LESO_PLANT_STATES}",
        f"#define LESO_INPUTS {LESO_INPUTS}",
        f"#define LESO_AUG_STATES {LESO_AUG_STATES}",
        f"#define LESO_TABLE_H_MIN {H_MIN:.8f}f",
        f"#define LESO_TABLE_H_MAX {H_MAX:.8f}f",
        f"#define LESO_TABLE_DT {DT:.8f}f",
        f"#define LESO_STATE_PROCESS_NOISE {LESO_STATE_PROCESS_NOISE:.9e}f",
        f"#define LESO_DISTURBANCE_PROCESS_NOISE {LESO_DISTURBANCE_PROCESS_NOISE:.9e}f",
        "// Measurement-noise covariance is diagonal and state dependent.",
        f"#define LESO_MEASUREMENT_NOISE_COUNT {LESO_PLANT_STATES}",
        "static const float leso_measurement_std[LESO_MEASUREMENT_NOISE_COUNT] = {",
        "    " + ", ".join(f"{value:.9e}f" for value in LESO_MEASUREMENT_STD),
        "};",
        "static const float leso_measurement_variance[LESO_MEASUREMENT_NOISE_COUNT] = {",
        "    " + ", ".join(f"{value:.9e}f" for value in LESO_MEASUREMENT_VARIANCE),
        "};",
        "",
    ]
    append_array(lines, "leso_ad_coeff", coeff_ad)
    append_array(lines, "leso_bd_coeff", coeff_bd)
    append_array(lines, "leso_gain_coeff", coeff_gain)

    lines += [
        "static inline void leso_table_2d_eval(float left_length, float right_length,",
        "                                      float *ad_out, float *bd_out,",
        "                                      float *gain_out) {",
        "    float xl = (2.0f * left_length - LESO_TABLE_H_MIN - LESO_TABLE_H_MAX)",
        "             / (LESO_TABLE_H_MAX - LESO_TABLE_H_MIN);",
        "    float xr = (2.0f * right_length - LESO_TABLE_H_MIN - LESO_TABLE_H_MAX)",
        "             / (LESO_TABLE_H_MAX - LESO_TABLE_H_MIN);",
        "    if (xl < -1.0f) xl = -1.0f; else if (xl > 1.0f) xl = 1.0f;",
        "    if (xr < -1.0f) xr = -1.0f; else if (xr > 1.0f) xr = 1.0f;",
        "    float tl[LESO_TABLE_ORDER + 1];",
        "    float tr[LESO_TABLE_ORDER + 1];",
        "    tl[0] = 1.0f; tr[0] = 1.0f;",
        "    if (LESO_TABLE_ORDER >= 1) { tl[1] = xl; tr[1] = xr; }",
        "    for (int i = 2; i <= LESO_TABLE_ORDER; i++) {",
        "        tl[i] = 2.0f * xl * tl[i - 1] - tl[i - 2];",
        "        tr[i] = 2.0f * xr * tr[i - 1] - tr[i - 2];",
        "    }",
        "    float terms[LESO_TABLE_N_TERMS];",
    ]
    term_index = 0
    for degree in range(p + 1):
        for right_degree in range(degree + 1):
            lines.append(
                f"    terms[{term_index}] = tl[{degree - right_degree}] * tr[{right_degree}];"
            )
            term_index += 1
    lines += [
        "    for (int element = 0; element < LESO_PLANT_STATES * LESO_PLANT_STATES; element++) {",
        "        float value = 0.0f;",
        "        for (int term = 0; term < LESO_TABLE_N_TERMS; term++)",
        "            value += leso_ad_coeff[element * LESO_TABLE_N_TERMS + term] * terms[term];",
        "        ad_out[element] = value;",
        "    }",
        "    for (int element = 0; element < LESO_PLANT_STATES * LESO_INPUTS; element++) {",
        "        float value = 0.0f;",
        "        for (int term = 0; term < LESO_TABLE_N_TERMS; term++)",
        "            value += leso_bd_coeff[element * LESO_TABLE_N_TERMS + term] * terms[term];",
        "        bd_out[element] = value;",
        "    }",
        "    for (int element = 0; element < LESO_AUG_STATES * LESO_PLANT_STATES; element++) {",
        "        float value = 0.0f;",
        "        for (int term = 0; term < LESO_TABLE_N_TERMS; term++)",
        "            value += leso_gain_coeff[element * LESO_TABLE_N_TERMS + term] * terms[term];",
        "        gain_out[element] = value;",
        "    }",
        "}",
    ]
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(
        f"  C header -> {filepath}  "
        f"({total_floats} floats = {total_floats * 4 / 1024:.2f} KB)"
    )


def fit_and_export_leso_tables(p, left_lengths, right_lengths, ad_set, bd_set,
                               gain_set=None,
                               dense_check_size=LESO_DENSE_CHECK_GRID):
    """Fit scheduled LESO matrices, validate them, and export MCU tables."""
    if gain_set is None:
        gain_set = np.stack([
            design_leso_gain(ad, bd) for ad, bd in zip(ad_set, bd_set)
        ])

    coeff_ad = fit_poly_2d(p, left_lengths, right_lengths, ad_set)
    coeff_bd = fit_poly_2d(p, left_lengths, right_lengths, bd_set)
    coeff_gain = fit_poly_2d(p, left_lengths, right_lengths, gain_set)
    np.save(str(_DATA_DIR / "_LESO_AD2D_COEFF.npy"), coeff_ad)
    np.save(str(_DATA_DIR / "_LESO_BD2D_COEFF.npy"), coeff_bd)
    np.save(str(_DATA_DIR / "_LESO_GAIN2D_COEFF.npy"), coeff_gain)

    Ce = np.hstack([
        np.eye(LESO_PLANT_STATES),
        np.zeros((LESO_PLANT_STATES, LESO_INPUTS)),
    ])
    fitted_observer_radius = np.zeros(len(left_lengths))
    for index, (left_length, right_length) in enumerate(
            zip(left_lengths, right_lengths)):
        fitted_ad = eval_poly_2d(p, coeff_ad, left_length, right_length)
        fitted_bd = eval_poly_2d(p, coeff_bd, left_length, right_length)
        fitted_gain = eval_poly_2d(p, coeff_gain, left_length, right_length)
        fitted_ae = np.block([
            [fitted_ad, fitted_bd],
            [np.zeros((LESO_INPUTS, LESO_PLANT_STATES)), np.eye(LESO_INPUTS)],
        ])
        fitted_observer_radius[index] = np.max(
            np.abs(np.linalg.eigvals(fitted_ae - fitted_gain @ Ce))
        )

    worst_index = int(np.argmax(fitted_observer_radius))
    print(
        f"  LESO fitted max|lambda|={fitted_observer_radius[worst_index]:.6f} "
        f"at L_l={left_lengths[worst_index]:.4f}, "
        f"L_r={right_lengths[worst_index]:.4f}"
    )
    if fitted_observer_radius[worst_index] >= 1.0:
        raise RuntimeError("Fitted LESO table is unstable at a sampling point")

    if dense_check_size < 2:
        raise ValueError("dense_check_size must be at least 2")
    dense_lengths = np.linspace(H_MIN, H_MAX, dense_check_size)
    dense_worst_radius = -np.inf
    dense_worst_lengths = (float("nan"), float("nan"))
    for left_length in dense_lengths:
        for right_length in dense_lengths:
            fitted_ad = eval_poly_2d(p, coeff_ad, left_length, right_length)
            fitted_bd = eval_poly_2d(p, coeff_bd, left_length, right_length)
            fitted_gain = eval_poly_2d(p, coeff_gain, left_length, right_length)
            fitted_ae = np.block([
                [fitted_ad, fitted_bd],
                [np.zeros((LESO_INPUTS, LESO_PLANT_STATES)),
                 np.eye(LESO_INPUTS)],
            ])
            radius = float(np.max(np.abs(
                np.linalg.eigvals(fitted_ae - fitted_gain @ Ce)
            )))
            if not np.isfinite(radius):
                raise RuntimeError(
                    "Non-finite fitted LESO pole at "
                    f"L_l={left_length:.6f}, L_r={right_length:.6f}"
                )
            if radius > dense_worst_radius:
                dense_worst_radius = radius
                dense_worst_lengths = (float(left_length), float(right_length))

    print(
        f"  LESO dense {dense_check_size}x{dense_check_size} max|lambda|="
        f"{dense_worst_radius:.6f} at L_l={dense_worst_lengths[0]:.4f}, "
        f"L_r={dense_worst_lengths[1]:.4f}; "
        f"stability margin={1.0 - dense_worst_radius:.6f}"
    )
    if dense_worst_radius >= 1.0:
        raise RuntimeError(
            "Fitted LESO table is unstable between sampling points: "
            f"max|lambda|={dense_worst_radius:.6f} at "
            f"L_l={dense_worst_lengths[0]:.6f}, "
            f"L_r={dense_worst_lengths[1]:.6f}"
        )

    export_c_header_leso(
        p, coeff_ad, coeff_bd, coeff_gain,
        str(_DATA_DIR / "leso_table_2d.h"),
    )
    return gain_set


def main():
    parser = argparse.ArgumentParser(
        description="Serial Lagrangian LQR gain table generator")
    parser.add_argument("--grid-size", type=int, default=14,
                        help="2D grid N (N x N points). Default: 14")
    parser.add_argument("--order", type=int, default=8,
                        help="2D triangular Chebyshev order. Default: 8")
    parser.add_argument("--leso-only", action="store_true",
                        help="Generate LESO tables from the existing AB sampling file")
    parser.add_argument("--leso-check-grid", type=int,
                        default=LESO_DENSE_CHECK_GRID,
                        help=f"Dense N x N grid used to verify fitted LESO poles. "
                             f"Default: {LESO_DENSE_CHECK_GRID}; use 81 for a denser check")
    parser.add_argument("--wheel-load-only", action="store_true",
                        help="检查静态工作点的左右轮法向载荷")
    args = parser.parse_args()
    if args.wheel_load_only and args.leso_only:
        parser.error("--wheel-load-only 与 --leso-only 不能同时使用")

    t0 = time.perf_counter()
    if args.leso_only:
        sample_path = _DATA_DIR / "AB_sampling_points.npz"
        samples = np.load(sample_path)
        # Recompute the observer gain from Ad/Bd so changes to Qe/Re are not
        # silently ignored by a cached leso_gain stored in the sampling file.
        fit_and_export_leso_tables(
            args.order,
            samples["L_l"], samples["L_r"],
            samples["A_d"], samples["B_d"], None,
            args.leso_check_grid,
        )
        print(f"LESO tables generated from {sample_path} in {time.perf_counter()-t0:.1f}s")
        return

    print("Serial Lagrangian + MuJoCo geometry")
    dynamics = build_serial_model(with_wheel_loads=args.wheel_load_only)
    M_func, Bq_func, V_func = dynamics[:3]
    wheel_load_samples = []
    wheel_load_check = (dynamics[3], wheel_load_samples) if args.wheel_load_only else None
    print(f"  Model built in {time.perf_counter()-t0:.1f}s")

    cfg = load_config()
    xml_path = resolve_project_path(cfg["mujoco"]["xml_path"])
    # Offline geometry solving does not need the RMUC field or its contacts.
    robot_only=xml_path.parent/'bigeye_robot.xml'
    if robot_only.is_file():xml_path=robot_only
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    solver = LegPoseSolver(model, cfg)
    body_com=load_model_parameters(model,data,cfg,solver)

    ORDER_2D = args.order
    N_2D = args.grid_size
    h_2d = np.linspace(H_MIN, H_MAX, N_2D)
    L_l_grid, L_r_grid = np.meshgrid(h_2d, h_2d)
    L_l_flat = L_l_grid.ravel()
    L_r_flat = L_r_grid.ravel()
    n_2d = len(L_l_flat)
    K_2d = np.zeros((n_2d, 4, 10))
    u_eq_set = np.zeros((n_2d, 4))
    A_d_set = np.zeros((n_2d, 10, 10))
    B_d_set = np.zeros((n_2d, 10, 4))
    leso_gain_set = np.zeros((n_2d, LESO_AUG_STATES, LESO_PLANT_STATES))
    e2_set = np.zeros(n_2d)
    e3_l_set = np.zeros(n_2d)
    e3_r_set = np.zeros(n_2d)
    S0 = np.diag(Q_DIAG)
    eig_max_track = np.zeros(n_2d)
    K_abs_max_track = np.zeros(n_2d)

    print(f"\n=== {ORDER_2D}th-order 2D Chebyshev fit (asymmetric legs) ===")
    print(f"  Grid: {N_2D}x{N_2D} = {n_2d} points, order={ORDER_2D}")
    t_fit = time.perf_counter()
    eq_stats = {
        "count": 0,
        "nonconverged": 0,
        "exceptions": 0,
        "max_residual": 0.0,
        "worst": None,
    }

    # 左右腿分别按各自长度取质心和惯量，保留非对称构型。
    unique_L = np.sort(np.unique(L_l_flat))
    com_by_L = {}
    seed_com = None
    for L in unique_L:
        seed_com = solver.solve(data, float(L), seed_com)
        com_by_L[float(L)] = {
            "left": solver.leg_com(data, "left"),
            "right": solver.leg_com(data, "right"),
            "left_inertia": solver.leg_inertia(data,"left"),
            "right_inertia": solver.leg_inertia(data,"right"),
            "left_axes":solver.leg_inertia_diagonal(data,"left")[[0,2]],
            "right_axes":solver.leg_inertia_diagonal(data,"right")[[0,2]],
            "left_y":solver.leg_lateral_offset(data,"left"),
            "right_y":solver.leg_lateral_offset(data,"right"),
        }

    for k in range(n_2d):
        L_l, L_r = float(L_l_flat[k]), float(L_r_flat[k])
        r_com_l, c_x_l = com_by_L[L_l]["left"]
        r_com_r, c_x_r = com_by_L[L_r]["right"]
        A, B, u_eq, th_eq, thl_eq, thr_eq = linearize_at_length(
            L_l, L_r, r_com_l, c_x_l, r_com_r, c_x_r,
            M_func, Bq_func, V_func, eq_stats,
            com_by_L[L_l]['left_inertia'],com_by_L[L_r]['right_inertia'],
            com_by_L[L_l]['left_axes'],com_by_L[L_r]['right_axes'],
            com_by_L[L_l]['left_y'],com_by_L[L_r]['right_y'],
            wheel_load_check=wheel_load_check)
        if args.wheel_load_only:
            continue
        u_eq_set[k] = u_eq
        Ad, Bd = discretize_zoh(A, B, DT)
        A_d_set[k], B_d_set[k] = Ad, Bd
        leso_gain_set[k] = design_leso_gain(Ad, Bd)
        e2_set[k] = th_eq
        e3_l_set[k] = thl_eq
        e3_r_set[k] = thr_eq
        F = design_supported_gain(Ad,Bd,S0,R_MAT)
        K_2d[k] = F
        eig_max_track[k] = np.max(np.abs(np.linalg.eigvals(Ad - Bd @ F)))
        K_abs_max_track[k] = np.max(np.abs(F))
    if args.wheel_load_only:
        loads = np.asarray(wheel_load_samples)
        weight = (M_BODY + 2*M_LEG + 2*M_WHEEL) * GRAVITY
        print(f"静态总重 {weight:.3f} N，双轮载荷和的最大误差 "
              f"{np.max(np.abs(loads.sum(axis=1)-weight)):.3e} N")
        for side, label in enumerate(("左轮", "右轮")):
            index = np.argmin(loads[:, side])
            print(f"{label}法向力 {loads[index, side]:.3f} 至 {loads[:, side].max():.3f} N，"
                  f"最小值对应腿长 ({L_l_flat[index]:.3f}, {L_r_flat[index]:.3f}) m")
        negative = np.any(loads < -1e-6, axis=1)
        print(f"出现负反力的工作点 {np.count_nonzero(negative)}/{n_2d}")
        return

    print(f"  Solved in {time.perf_counter()-t_fit:.1f}s")
    eig_worst_idx = int(np.argmax(eig_max_track))
    eig_max_dyn_track = np.zeros(n_2d)
    for k in range(n_2d):
        evals, evecs = np.linalg.eig(A_d_set[k] - B_d_set[k] @ K_2d[k])
        ev_abs = np.abs(evals)
        # 单独统计位移或偏航主导的慢模态，便于识别姿态稳定性问题。
        integrator_mask = np.zeros(len(evals), dtype=bool)
        for j in range(len(evals)):
            w = np.abs(evecs[:, j])
            if w[0] > 0.5 or w[1] > 0.5:
                integrator_mask[j] = True
        dyn_ev = ev_abs[~integrator_mask]
        eig_max_dyn_track[k] = np.max(dyn_ev) if len(dyn_ev) > 0 else 0.0
    eig_dyn_worst = int(np.argmax(eig_max_dyn_track))
    print(f"  Closed-loop max|lambda|: {eig_max_track[eig_worst_idx]:.4f}  "
          f"(all states, worst at L_l={L_l_flat[eig_worst_idx]:.3f}, L_r={L_r_flat[eig_worst_idx]:.3f})")
    print(f"  Closed-loop max|lambda| (dynamic): {eig_max_dyn_track[eig_dyn_worst]:.4f}  "
          f"(excl. x/psi, worst at L_l={L_l_flat[eig_dyn_worst]:.3f}, L_r={L_r_flat[eig_dyn_worst]:.3f})")
    if eig_max_dyn_track[eig_dyn_worst] >= 1.0:
        print(f"  ** ERROR: unstable dynamic pole (pitch/leg modes) **")
    elif eig_max_dyn_track[eig_dyn_worst] > 0.99:
        print(f"  ** WARNING: dynamic pole near unit circle (max|lambda|_dyn > 0.99) **")
    if eig_max_track[eig_worst_idx] >= 1.0:
        print(f"  ** ERROR: unstable closed-loop pole (incl. integrators) **")
    elif eig_max_track[eig_worst_idx] > 0.99 and eig_max_dyn_track[eig_dyn_worst] < 0.98:
        print(f"  NOTE: max|lambda| near 1.0 is from x/psi integrators — "
              f"increase Q[0]/Q[1] for position feedback, or ignore if velocity-only control")
    K_worst_idx = int(np.argmax(K_abs_max_track))
    print(f"  K matrix |element| range: "
          f"min={np.min(K_2d):.3e}  max={np.max(K_2d):.3e}  "
          f"median={np.median(K_2d):.3e}")
    if np.max(K_2d) > 5000:
        print(f"  ** WARNING: K element > 5000 at L_l={L_l_flat[K_worst_idx]:.3f}, "
              f"L_r={L_r_flat[K_worst_idx]:.3f} — check Q/R scaling **")
    if np.max(K_2d) > 1e5:
        print(f"  ** ERROR: K element magnitude explosion — Q/R ratio likely wrong **")
    bad_mask = eig_max_dyn_track > 0.99
    n_bad = int(np.sum(bad_mask))
    if n_bad > 0:
        bad_idx = np.where(bad_mask)[0]
        show_n = min(n_bad, 10)
        print(f"  Near-unstable dynamic poles (max|lambda|_dyn > 0.99): {n_bad}/{n_2d}"
              + (f"  (showing first {show_n})" if n_bad > 10 else ""))
        for idx in bad_idx[:show_n]:
            print(f"    L_l={L_l_flat[idx]:.4f}  L_r={L_r_flat[idx]:.4f}  "
                  f"max|lambda|_dyn={eig_max_dyn_track[idx]:.4f}")
        if n_bad > show_n:
            print(f"    ... and {n_bad - show_n} more")
    if eq_stats["worst"] is not None:
        Ll_w, Lr_w, ier_w, msg_w, res_w = eq_stats["worst"]
        print(
            f"  Equilibrium SLSQP: nonconverged={eq_stats['nonconverged']}/"
            f"{eq_stats['count']}, exceptions={eq_stats['exceptions']}, "
            f"max|G-Bu|_inf={res_w:.3e} at L=({Ll_w:.4f},{Lr_w:.4f}), status={ier_w}")
        if eq_stats["nonconverged"] > 0:
            print(f"    worst message: {msg_w}")

    ab_path = _DATA_DIR / "AB_sampling_points.npz"
    ab_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(ab_path),
             L_l=L_l_flat, L_r=L_r_flat,
             A_d=A_d_set, B_d=B_d_set,
             leso_gain=leso_gain_set,
             e2=e2_set, e3_l=e3_l_set, e3_r=e3_r_set,
             u_eq=u_eq_set, K=K_2d,body_com=body_com,state_frame="hip_midpoint",leg_com_frame="hip_down_x")
    print(f"  A_d, B_d, LESO gain, K, u_eq, e2, e3 -> {ab_path}")

    coeff_2d = fit_poly_2d(ORDER_2D, L_l_flat, L_r_flat, K_2d)
    cond_2d = float(np.linalg.cond(poly2d_design_matrix(ORDER_2D, L_l_flat, L_r_flat)))
    rel_err = assess_2d_fit(ORDER_2D, coeff_2d, L_l_flat, L_r_flat, K_2d)
    max_err_pct = float(np.max(rel_err)) * 100
    mean_err_pct = float(np.mean(rel_err)) * 100
    max_idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    print(f"  Fit accuracy: max={max_err_pct:.2f}%  mean={mean_err_pct:.2f}%  RMS relative error")
    print(f"  Fit matrix condition: cond={cond_2d:.3e}; worst K index={max_idx}")
    if max_err_pct < 5.0:
        print("  OK: Fit accepted (max error < 5%)")
    else:
        print(f"  WARNING: max error = {max_err_pct:.2f}% > 5%")

    h_path = _DATA_DIR / "k_table_2d.h"
    export_c_header_2d(ORDER_2D, coeff_2d, str(h_path))

    fit_and_export_leso_tables(
        ORDER_2D, L_l_flat, L_r_flat,
        A_d_set, B_d_set, leso_gain_set,
        args.leso_check_grid,
    )

    coeff_u_eq = fit_poly_2d(ORDER_2D, L_l_flat, L_r_flat,
                              u_eq_set.reshape(n_2d, 4, 1))
    np.save(str(_DATA_DIR / "_U_EQ_COEFF.npy"), coeff_u_eq)
    export_c_header_ueq(ORDER_2D, coeff_u_eq,
                        str(_DATA_DIR / "u_eq_table_2d.h"))

    # 一元四次曲线保留给旧接口，左右平衡腿角另做二维拟合。
    avg_L = 0.5 * (L_l_flat + L_r_flat)
    A_e = np.column_stack([avg_L**4, avg_L**3, avg_L**2, avg_L, np.ones_like(avg_L)])
    E2_COEFF = np.linalg.lstsq(A_e, e2_set, rcond=None)[0]
    E3_COEFF = np.linalg.lstsq(A_e, 0.5*(e3_l_set + e3_r_set), rcond=None)[0]

    np.save(str(_DATA_DIR / "_E2_COEFF.npy"), E2_COEFF)
    np.save(str(_DATA_DIR / "_E3_COEFF.npy"), E3_COEFF)
    # The Python controller loads this 2D table when body_com is recorded in
    # AB_sampling_points, keeping offsets and normalization bounds in sync.
    equilibrium_states=np.zeros((n_2d,10,1))
    equilibrium_states[:,2,0]=e2_set
    equilibrium_states[:,3,0]=e3_l_set;equilibrium_states[:,4,0]=e3_r_set
    coeff_equilibrium=fit_poly_2d(ORDER_2D,L_l_flat,L_r_flat,equilibrium_states)[:,0,:]
    np.save(str(_DATA_DIR/'_EQUILIBRIUM2D_COEFF.npy'),coeff_equilibrium)
    export_c_header_e3(E3_COEFF, str(_DATA_DIR / "e3_table.h"))
    # The MCU must use the same asymmetric equilibrium as the Python runtime.
    e3_path=_DATA_DIR/'e3_table.h'
    extra=['', '// Exact left/right equilibrium on the same 2D grid as K.',
           'static const float e3_2d_coeff[2][%d] = {' % coeff_equilibrium.shape[1]]
    for row in coeff_equilibrium[3:5]:
        extra.append('    {'+', '.join(f'{float(x):.9e}f' for x in row)+'},')
    extra += ['};', 'static inline void e3_table_2d_eval(float ll, float lr, float *el, float *er) {',
              f'    float x=(2.0f*ll-{H_MIN:.9f}f-{H_MAX:.9f}f)/{H_MAX-H_MIN:.9f}f;',
              f'    float y=(2.0f*lr-{H_MIN:.9f}f-{H_MAX:.9f}f)/{H_MAX-H_MIN:.9f}f;',
              '    x=x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);',
              '    y=y < -1.0f ? -1.0f : (y > 1.0f ? 1.0f : y);',
              f'    float a[{ORDER_2D+1}],b[{ORDER_2D+1}]; a[0]=b[0]=1.0f; a[1]=x; b[1]=y;',
              f'    for(int i=2;i<={ORDER_2D};++i) {{a[i]=2*x*a[i-1]-a[i-2];b[i]=2*y*b[i-1]-b[i-2];}}',
              '    *el=0.0f;*er=0.0f;int n=0;',
              f'    for(int d=0;d<={ORDER_2D};++d) for(int j=0;j<=d;++j,++n) {{',
              '        float t=a[d-j]*b[j];*el+=e3_2d_coeff[0][n]*t;*er+=e3_2d_coeff[1][n]*t;',
              '    }','}']
    with e3_path.open('a',encoding='utf8') as f:f.write('\n'.join(extra)+'\n')
    np.save(str(_DATA_DIR / "_K2D_COEFF.npy"), coeff_2d)
    print(f"  .npy coefficients -> {_DATA_DIR}/_*_COEFF.npy")

    (_DATA_DIR/'linearization_metadata.json').write_text(json.dumps(dict(
        state_position='hip midpoint horizontal displacement',body_com_m=body_com.tolist(),
        body_mass_kg=M_BODY,leg_mass_kg=M_LEG,wheel_mass_kg=M_WHEEL,
        body_com_in_dynamics=['x','y','z'],leg_com_convention='x offset and vertical distance below hip',
        support_model='simultaneous wheel support; induced small-bank inertia; locally upright',
        yaw_model='coupled Yaw/leg inertia; joint ten-state four-input LQR',
        Q_diagonal=Q_DIAG.tolist(),R_diagonal=np.diag(R_MAT).tolist(),R_matrix=R_MAT.tolist(),
        max_closed_loop_pole=float(np.max(eig_max_track)),fit_max_relative_error_pct=max_err_pct,
        grid_size=N_2D,order=ORDER_2D),indent=2)+'\n',encoding='utf8')
    print(f"\nTotal: {time.perf_counter()-t0:.1f}s")
    print('MCU: include k_table_2d.h, u_eq_table_2d.h, e3_table.h, leso_table_2d.h')


if __name__ == "__main__":
    main()
