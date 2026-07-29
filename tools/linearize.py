"""
Serial Lagrangian + MuJoCo geometry → 2D LQR gain table.

Uses SymPy Lagrangian dynamics + MuJoCo LegPoseSolver for leg COM geometry.
Outputs 2D Chebyshev gain fits to data/lqr_gain_fit.py and data/k_table_2d.h.

Usage:
  python tools/linearize.py                    # 2D Chebyshev table (default 14×14)
  python tools/linearize.py --grid-size 27     # finer grid
"""

import sys, time, argparse
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq, least_squares
import mujoco as mj
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from kinematics import FiveLinkLeg  # type: ignore[import-untyped]                                                                                 
from utils import load_config, resolve_project_path  # type: ignore[import-untyped] 

# ══════════════════════════════════════════════════════════════════════
#  Serial_Leg_Simulation physical constants
# ══════════════════════════════════════════════════════════════════════
M_BODY = 15.0
M_WHEEL = 0.353429173529
M_LEG   = 0.13482+0.1554+0.230811+0.11718+0.349772111111+0.23625
I_BODY_YY = 0.156700286822; I_BODY_ZZ = 0.156098776548
I_WHEEL_YY = 0.000441786467; I_WHEEL_ZZ = 0.000223838477
WHEEL_RADIUS = 0.05; HALF_TRACK = 0.165
GRAVITY = 9.81; L_BODY_COM = 0.0
DT = 0.002; N_ITER_DARE = 100000
H_MIN, H_MAX = 0.106, 0.366

# LQR 状态权重 Q（10×10 对角）
# 状态:      x    yaw   pitch  th_L   th_R   x_dot yaw_dot pitch_dot th_L_dot th_R_dot
Q_DIAG = np.array([
    10,   # x
    5,     # yaw
    15000, # pitch
    2000,   # theta_L
    2000,   # theta_R
    80,   # x_dot
    10,     # yaw_dot
    8,     # pitch_dot
    10,    # theta_L_dot
    10,    # theta_R_dot
])

# LQR 控制权重 R（4×4 对角）
# 控制:    tau_L  tau_R  T_wL   T_wR
R_MAT  = np.diag([
    0.8,  # tau_L
    0.8,  # tau_R
    1.0,  # T_wL
    1.0,  # T_wR
])
# python tools/linearize.py --grid-size 27


# ══════════════════════════════════════════════════════════════════════
#  MuJoCo LegPoseSolver — actual Serial 5-link leg COM geometry
# ══════════════════════════════════════════════════════════════════════

class LegPoseSolver:
    def __init__(self, model, cfg):
        self.model = model
        self.leg_geom = cfg["geometry"]["five_link"]
        self.wheel_radius = cfg["geometry"]["wheel"]["radius_m"]
        self.probe = FiveLinkLeg("probe", self.leg_geom, 0.002)
        self._pqv = np.array([8,9,10,11,15,16,17,18], dtype=int)
        self._sites = [
            ("left_rear_to_bridge_connect","left_bridge_rear_connect"),
            ("left_rear_to_lower_connect","left_lower_leg_rear_connect"),
            ("right_rear_to_bridge_connect","right_bridge_rear_connect"),
            ("right_rear_to_lower_connect","right_lower_leg_rear_connect")]
        ln = ["left_front_upper_link","left_front_lower_link",
              "left_lower_bridge_link","left_wheel_carrier_link",
              "left_lower_leg_link","left_rear_upper_link","left_wheel"]
        self._bids = [mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,n) for n in ln]

    def _front_for(self, h):
        lo,hi=np.deg2rad(-10),np.deg2rad(55)
        f=lambda a: self.probe.forward(a,np.pi-a,0)[0]-h
        if f(lo)*f(hi)>0: lo=np.deg2rad(-15)
        return brentq(f, lo, hi)

    def solve(self, data, h, seed=None):
        fa = self._front_for(h)
        qpos = self.model.qpos0.copy() if seed is None else seed.copy()
        wz = self.wheel_radius-0.001
        qpos[:3]=[0,0,h+wz]; qpos[3:7]=[1,0,0,0]
        qpos[7]=fa; qpos[13]=np.pi-fa; qpos[14]=fa; qpos[20]=np.pi-fa
        qpos[12]=0; qpos[19]=0
        y0=np.r_[qpos[2],qpos[self._pqv]]
        lo=np.r_[0.04,np.full(self._pqv.size,-np.pi)]
        hi=np.r_[0.60,np.full(self._pqv.size,np.pi)]
        def res(y):
            q=qpos.copy(); q[2]=y[0]; q[self._pqv]=y[1:]
            data.qpos[:]=q; data.qvel[:]=0; data.ctrl[:]=0
            mj.mj_forward(self.model,data)
            r=[]
            for a,b in self._sites:
                r.extend(100*(data.site(a).xpos-data.site(b).xpos))
            r.append(20*(data.body("left_wheel").xpos[2]-wz))
            r.append(20*(data.body("right_wheel").xpos[2]-wz))
            return np.asarray(r)
        sol=least_squares(res,y0,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=500)
        qpos[2]=sol.x[0]; qpos[self._pqv]=sol.x[1:]
        data.qpos[:]=qpos; data.qvel[:]=0; data.ctrl[:]=0
        mj.mj_forward(self.model,data)
        return qpos.copy()

    def leg_com(self, data):
        """Returns (r_com, c_x): radial distance and tangential offset."""
        hip = data.body("left_front_upper_link").xpos.copy()
        tot=0.0; com=np.zeros(3)
        for bid in self._bids:
            m=self.model.body_mass[bid]
            # xipos is already the world-space inertial-frame COM.
            tot+=m; com+=m*data.xipos[bid]
        com/=tot; d=com-hip
        return float(np.hypot(d[0],d[2])), float(d[0])


# ══════════════════════════════════════════════════════════════════════
#  Serial-specific SymPy Lagrangian (matching MATLAB structure)
# ══════════════════════════════════════════════════════════════════════

def build_serial_model():
    """SymPy Lagrangian with Serial's physical parameters.
    Uses exact sqrt terms, same Euler-Lagrange structure as MATLAB.
    """

    x_s,w_s,th_s,thl_s,thr_s = sp.symbols('x w theta theta_l theta_r')
    xd_s,wd_s,thd_s,thld_s,thrd_s = sp.symbols('xd wd thetad thetald thetard')
    xdd_s,wdd_s,thdd_s,thldd_s,thrdd_s = sp.symbols('xdd wdd thetadd thetaldd thetardd')
    Ml2,Mr2,Ml3,Mr3 = sp.symbols('Ml2 Mr2 Ml3 Mr3')

    m,m_w,d,r = sp.symbols('m m_w d r')
    g,l_body = sp.symbols('g l_body')
    j_y,j_z,j_wy,j_wz = sp.symbols('j_y j_z j_wy j_wz')
    L_l,L_r = sp.symbols('L_l L_r'); ml,mr = sp.symbols('ml mr')
    c_l,c_r = sp.symbols('c_l c_r'); cx_l,cx_r = sp.symbols('cx_l cx_r')
    j_ly,j_ry = sp.symbols('j_ly j_ry')

    # Exact sqrt terms (matching MATLAB)
    swL = sp.sqrt(d**2+sp.sin(thl_s)**2*L_l**2)
    swR = sp.sqrt(d**2+sp.sin(thr_s)**2*L_r**2)
    slL = sp.sqrt(d**2+sp.sin(thl_s)**2*(L_l-c_l)**2)
    slR = sp.sqrt(d**2+sp.sin(thr_s)**2*(L_r-c_r)**2)

    # Kinetic Energy
    T = (sp.Rational(1,2)*m_w*(xd_s-swL*wd_s-sp.cos(thl_s)*L_l*thld_s)**2
         +sp.Rational(1,2)*m_w*(xd_s+swR*wd_s-sp.cos(thr_s)*L_r*thrd_s)**2
         +sp.Rational(1,2)*j_wy*((xd_s-swL*wd_s-sp.cos(thl_s)*L_l*thld_s)/r)**2
         +sp.Rational(1,2)*j_wy*((xd_s+swR*wd_s-sp.cos(thr_s)*L_r*thrd_s)/r)**2
         +sp.Rational(1,2)*j_wz*wd_s**2*2
         +sp.Rational(1,2)*j_ly*thld_s**2
         +sp.Rational(1,2)*ml*((xd_s-slL*wd_s-sp.cos(thl_s)*(L_l-c_l)*thld_s)**2
                                +(c_l*thld_s*sp.sin(thl_s))**2)
         +sp.Rational(1,2)*j_ry*thrd_s**2
         +sp.Rational(1,2)*mr*((xd_s+slR*wd_s-sp.cos(thr_s)*(L_r-c_r)*thrd_s)**2
                                +(c_r*thrd_s*sp.sin(thr_s))**2)
         +sp.Rational(1,2)*j_y*thd_s**2
         +sp.Rational(1,2)*m*(xd_s**2
                               +(l_body*thd_s*sp.sin(th_s)
                                  +(L_l+L_r)/2*(thld_s+thrd_s)/2*sp.sin((thl_s+thr_s)/2))**2)
         +sp.Rational(1,2)*j_z*wd_s**2)

    # Potential Energy
    th_p_L = sp.atan2(cx_l,c_l); th_p_R = sp.atan2(cx_r,c_r)
    V = (ml*g*sp.sqrt(c_l**2+cx_l**2)*sp.cos(thl_s-th_p_L)
         +mr*g*sp.sqrt(c_r**2+cx_r**2)*sp.cos(thr_s-th_p_R)
         +m*g*((L_l+L_r)/2*sp.cos((thl_s+thr_s)/2)+l_body*sp.cos(th_s)))

    # Euler-Lagrange (same generalized forces as MATLAB eq1-eq5)
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
    eq4 = (ddL(L_expr,thl_s,thld_s,thldd_s)-sp.diff(L_expr,thl_s)) - (Ml2-Ml3)
    eq5 = (ddL(L_expr,thr_s,thrd_s,thrdd_s)-sp.diff(L_expr,thr_s)) - (Mr2-Mr3)

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
                  L_l,L_r,ml,mr,c_l,c_r,cx_l,cx_r,l_body,j_ly,j_ry]

    print("  Lambdifying M, Bq, V ...")
    M_func  = sp.lambdify(param_vars+state_vars, M_sym, 'numpy')
    Bq_func = sp.lambdify(param_vars+state_vars, Bq_sym, 'numpy')
    V_func  = sp.lambdify(param_vars+q_vars, V, 'numpy')
    return M_func, Bq_func, V_func


# ══════════════════════════════════════════════════════════════════════
#  Numerical linearization
# ══════════════════════════════════════════════════════════════════════

def linearize_at_length(L_l, L_r, r_com, c_x, M_func, Bq_func, V_func, eq_stats=None):
    p = [M_BODY,M_WHEEL,HALF_TRACK,WHEEL_RADIUS,
         I_BODY_YY,I_BODY_ZZ,I_WHEEL_ZZ,I_WHEEL_YY,GRAVITY,
         L_l,L_r, M_LEG,M_LEG, r_com,r_com, c_x,c_x,
         L_BODY_COM, 0.003,0.003]   # j_ly=0.003 calibrated for leg inertia

    def dV(q_a):
        qf=[0,0,q_a[0],q_a[1],q_a[2]]; eps=1e-6; grad=np.zeros(3)
        V0=V_func(*p,*qf)
        for i in range(3):
            qq=qf.copy(); qq[i+2]+=eps; grad[i]=(V_func(*p,*qq)-V0)/eps
        return grad
    try:
        sol, info, ier, msg = fsolve(dV,[0,0,0],maxfev=200,xtol=1e-12,full_output=True)
        res = np.asarray(info.get("fvec", dV(sol)), dtype=float)
        res_norm = float(np.linalg.norm(res, ord=np.inf))
        if eq_stats is not None:
            eq_stats["count"] += 1
            if res_norm > eq_stats["max_residual"]:
                eq_stats["max_residual"] = res_norm
                eq_stats["worst"] = (float(L_l), float(L_r), int(ier), str(msg).strip(), res_norm)
            if ier != 1:
                eq_stats["nonconverged"] += 1
        elif ier != 1:
            print(f"  WARNING: fsolve ier={ier} at L=({L_l:.4f},{L_r:.4f}), max|dV|={res_norm:.3e}: {msg}")
        th_eq,thl_eq,thr_eq=float(sol[0]),float(sol[1]),float(sol[2])
    except Exception as exc:
        if eq_stats is not None:
            eq_stats["count"] += 1
            eq_stats["nonconverged"] += 1
            eq_stats["exceptions"] += 1
            eq_stats["worst"] = (float(L_l), float(L_r), -1, repr(exc), float("inf"))
            eq_stats["max_residual"] = float("inf")
        th_eq=thl_eq=thr_eq=0.0

    q_eq=[0,0,th_eq,thl_eq,thr_eq]; qd_eq=[0,0,0,0,0]; se=q_eq+qd_eq
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
    Dd=np.diag([0.1,0.1,0.15,0.06,0.06]); A[5:10,5:10]=-Mi@Dd
    B=np.zeros((10,4)); B[5:10,:]=Mi@Bq
    return A,B,th_eq,thl_eq,thr_eq


# ══════════════════════════════════════════════════════════════════════
#  ZOH + DARE
# ══════════════════════════════════════════════════════════════════════

def discretize_zoh(A,B,dt):
    n,m=A.shape[0],B.shape[1]; M=np.zeros((n+m,n+m)); M[:n,:n]=A; M[:n,n:]=B; M*=dt; E=expm(M)
    return E[:n,:n],E[:n,n:]

def solve_dare(Ad,Bd,Q,R,S,n_iter=N_ITER_DARE):
    P=S.copy()
    for _ in range(n_iter):
        F=np.linalg.inv(Bd.T@P@Bd+R)@(Bd.T@P@Ad)
        Pn=(Ad-Bd@F).T@P@(Ad-Bd@F)+F.T@R@F+Q
        if np.linalg.norm(Pn-P,'fro')/max(np.linalg.norm(P,'fro'),1e-15)<1e-8: P=Pn; break
        P=Pn
    return np.linalg.inv(Bd.T@P@Bd+R)@(Bd.T@P@Ad),P

# ══════════════════════════════════════════════════════════════════════
#  2D Chebyshev polynomial fit + C header export（手册 §5.6, §12）
# ══════════════════════════════════════════════════════════════════════

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

    K_2d shape: (n_samples, 4, 10)
    Returns coeff shape: (4, 10, n_terms)
    """
    n_samples = len(L_l_vec)
    n_terms = (p + 1) * (p + 2) // 2
    no, ns = 4, 10
    coeff = np.zeros((no, ns, n_terms))
    A = poly2d_design_matrix(p, L_l_vec, L_r_vec)
    for i in range(no):
        for j in range(ns):
            coeff[i, j, :] = np.linalg.lstsq(A, K_2d[:, i, j], rcond=None)[0]
    return coeff


def eval_poly_2d(p, coeff, L_l, L_r):
    """Evaluate 2D Chebyshev fit at (L_l, L_r). coeff shape: (4, 10, n_terms)."""
    terms = poly2d_terms(p, L_l, L_r)
    no, ns = 4, 10
    K = np.zeros((no, ns))
    for i in range(no):
        for j in range(ns):
            K[i, j] = np.dot(coeff[i, j, :], terms)
    return K


def assess_2d_fit(p, coeff, L_l_vec, L_r_vec, K_ref):
    """Compute max relative RMS error across all K elements."""
    no, ns = 4, 10
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
    lines.append(f"// {n_floats} float32 coefficients = {n_kb:.2f} KB")
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

    # Print 8 values per line
    chunk = 8
    for start in range(0, n_floats, chunk):
        vals = flat[start:start + chunk]
        line = "    " + ", ".join(f"{v:12.8f}" for v in vals)
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


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Serial Lagrangian LQR gain table generator")
    parser.add_argument("--grid-size", type=int, default=14,
                        help="2D grid N (N x N points). Default: 14")
    parser.add_argument("--order", type=int, default=8,
                        help="2D triangular Chebyshev order. Default: 8")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("Serial Lagrangian + MuJoCo geometry")
    M_func, Bq_func, V_func = build_serial_model()
    print(f"  Model built in {time.perf_counter()-t0:.1f}s")

    cfg = load_config()
    xml_path = resolve_project_path(cfg["mujoco"]["xml_path"])
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    solver = LegPoseSolver(model, cfg)

    # ════════════════════════════════════════════════════════════════
    #  2D: asymmetric leg lengths + Chebyshev fit + C header
    # ════════════════════════════════════════════════════════════════
    ORDER_2D = args.order
    N_2D = args.grid_size
    h_2d = np.linspace(H_MIN, H_MAX, N_2D)
    L_l_grid, L_r_grid = np.meshgrid(h_2d, h_2d)
    L_l_flat = L_l_grid.ravel()
    L_r_flat = L_r_grid.ravel()
    n_2d = len(L_l_flat)
    K_2d = np.zeros((n_2d, 4, 10))
    # Store A_d, B_d, equilibrium offsets at each grid point
    A_d_set = np.zeros((n_2d, 10, 10))
    B_d_set = np.zeros((n_2d, 10, 4))
    e2_set = np.zeros(n_2d)
    e3_l_set = np.zeros(n_2d)
    e3_r_set = np.zeros(n_2d)
    S0 = np.diag(Q_DIAG)

    print(f"\n=== {ORDER_2D}th-order 2D Chebyshev fit (asymmetric legs) ===")
    print(f"  Grid: {N_2D}x{N_2D} = {n_2d} points, order={ORDER_2D}")
    t_fit = time.perf_counter(); seed_2d = None
    eq_stats = {
        "count": 0,
        "nonconverged": 0,
        "exceptions": 0,
        "max_residual": 0.0,
        "worst": None,
    }
    for k in range(n_2d):
        L_l, L_r = float(L_l_flat[k]), float(L_r_flat[k])
        seed_2d = solver.solve(data, 0.5 * (L_l + L_r), seed_2d)
        r_com, c_x = solver.leg_com(data)
        A, B, th_eq, thl_eq, thr_eq = linearize_at_length(
            L_l, L_r, r_com, c_x, M_func, Bq_func, V_func, eq_stats)
        Ad, Bd = discretize_zoh(A, B, DT)
        A_d_set[k], B_d_set[k] = Ad, Bd
        e2_set[k] = th_eq
        e3_l_set[k] = thl_eq
        e3_r_set[k] = thr_eq
        F, _ = solve_dare(Ad, Bd, S0, R_MAT, S0)
        K_2d[k] = F
    print(f"  Solved in {time.perf_counter()-t_fit:.1f}s")
    if eq_stats["worst"] is not None:
        Ll_w, Lr_w, ier_w, msg_w, res_w = eq_stats["worst"]
        print(
            f"  Equilibrium fsolve: nonconverged={eq_stats['nonconverged']}/"
            f"{eq_stats['count']}, exceptions={eq_stats['exceptions']}, "
            f"max|dV|_inf={res_w:.3e} at L=({Ll_w:.4f},{Lr_w:.4f}), ier={ier_w}")
        if eq_stats["nonconverged"] > 0:
            print(f"    worst message: {msg_w}")

    # ── Save A_d, B_d + equilibrium offsets ──
    ab_path = Path("data/AB_sampling_points.npz")
    ab_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(ab_path),
             L_l=L_l_flat, L_r=L_r_flat,
             A_d=A_d_set, B_d=B_d_set,
             e2=e2_set, e3_l=e3_l_set, e3_r=e3_r_set,
             K=K_2d)
    print(f"  A_d, B_d, K, e2, e3 -> {ab_path}")

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

    # ── Export C header ──
    h_path = Path("data/k_table_2d.h")
    export_c_header_2d(ORDER_2D, coeff_2d, str(h_path))

    # ── Fit e2(L) and e3(L) 1D polynomials + save all coefficients ──
    avg_L = 0.5 * (L_l_flat + L_r_flat)
    A_e = np.column_stack([avg_L**4, avg_L**3, avg_L**2, avg_L, np.ones_like(avg_L)])
    E2_COEFF = np.linalg.lstsq(A_e, e2_set, rcond=None)[0]
    E3_COEFF = np.linalg.lstsq(A_e, 0.5*(e3_l_set + e3_r_set), rcond=None)[0]

    np.save("data/_E2_COEFF.npy", E2_COEFF)
    np.save("data/_E3_COEFF.npy", E3_COEFF)
    np.save("data/_K2D_COEFF.npy", coeff_2d)
    print(f"  .npy coefficients -> data/_*_COEFF.npy")

    print(f"\nTotal: {time.perf_counter()-t0:.1f}s")
    print('MCU:  #include "data/k_table_2d.h"  ->  k_table_2d_eval(L_l, L_r, k_out)')


if __name__ == "__main__":
    main()
