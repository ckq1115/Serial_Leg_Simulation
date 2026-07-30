"""Fast LQR tuning: match K matrix ratios to legwheel. No simulation."""
import sys, re, itertools, numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
LPATH = PROJ / "tools" / "linearize.py"
ORIG = LPATH.read_text(encoding="utf-8")

sys.path.insert(0, str(PROJ / "data"))

def legwheel_K(L=0.20):
    """legwheel K matrix at given leg length."""
    kc = np.array([
[[6.2174,-11.2883,8.3893,-3.4140,0.7898],[1.4340,-1.8905,1.1834,-0.5364,-0.4896],[923.8710,-1092.6,502.6034,-110.6934,-57.0662],[364.5066,-420.9464,197.7163,-61.6707,29.2380],[347.2482,-494.2305,293.6293,-84.8201,-2.8636],[172.8698,-205.0662,97.0404,-23.7418,3.2421],[-2.1572,2.0367,-0.4234,-0.3492,-0.5612],[39.0661,-46.6079,21.7275,-4.8873,-2.8456],[16.1460,-16.5183,6.9372,-0.5876,0.6249],[13.1891,-10.1115,2.2276,-1.2081,-0.0720]],
[[6.2174,-11.2883,8.3893,-3.4140,0.7898],[-1.4340,1.8905,-1.1834,0.5364,0.4896],[923.8710,-1092.6,502.6034,-110.6934,-57.0662],[347.2482,-494.2305,293.6293,-84.8201,-2.8636],[364.5066,-420.9464,197.7163,-61.6707,29.2380],[172.8698,-205.0662,97.0404,-23.7418,3.2421],[2.1572,-2.0367,0.4234,0.3492,0.5612],[39.0661,-46.6079,21.7275,-4.8873,-2.8456],[13.1891,-10.1115,2.2276,-1.2081,-0.0720],[16.1460,-16.5183,6.9372,-0.5876,0.6249]],
[[17.2035,-25.3140,16.7064,-5.8122,-1.1309],[2.1946,-2.3090,0.7162,0.2570,-0.4818],[-1240.9,1624.2,-872.0751,248.8905,-39.5076],[2912.8,-2879.2,1171.0,-271.5191,-23.8006],[685.0781,-969.6809,519.8358,-124.8295,4.6944],[-243.6033,258.0985,-101.7795,17.4144,-6.2708],[2.2857,-2.4649,0.8261,0.2417,-0.5325],[-104.5821,129.5426,-64.9653,17.0979,-2.5537],[-71.1397,67.6541,-23.1166,-1.4280,-0.8085],[-65.3237,58.0976,-17.3916,-0.6455,-0.1529]],
[[17.2035,-25.3140,16.7064,-5.8122,-1.1309],[-2.1946,2.3090,-0.7162,-0.2570,0.4818],[-1240.9,1624.2,-872.0751,248.8905,-39.5076],[685.0781,-969.6809,519.8358,-124.8295,4.6944],[2912.8,-2879.2,1171.0,-271.5191,-23.8006],[-243.6033,258.0985,-101.7795,17.4144,-6.2708],[2.2857,-2.4649,0.8261,0.2417,-0.5325],[-104.5821,129.5426,-64.9653,17.0979,-2.5537],[-65.3237,58.0976,-17.3916,-0.6455,-0.1529],[-71.1397,67.6541,-23.1166,-1.4280,-0.8085]]])
    K = np.zeros((4,10))
    for i in range(4):
        for j in range(10):
            c = kc[i,j]
            K[i,j] = c[0]*L**4 + c[1]*L**3 + c[2]*L**2 + c[3]*L + c[4]
    return K


def apply_q(Q_diag, R_diag):
    """Modify linearize.py with new Q/R."""
    txt = ORIG
    names = ['x','yaw','pitch','th_L','th_R','x_dot','yaw_dot','pitch_dot','thL_dot','thR_dot']
    qb = "Q_DIAG = np.array([\n"
    for v,n in zip(Q_diag, names): qb += f"    {v},   # {n}\n"
    qb += "])"
    s = txt.find("Q_DIAG = np.array([")
    e = txt.find("])", s) + 2
    txt = txt[:s] + qb + txt[e:]

    rnames = ['tau_L','tau_R','T_wL','T_wR']
    rb = "R_MAT  = np.diag([\n"
    for v,n in zip(R_diag, rnames): rb += f"    {float(v)},  # {n}\n"
    rb += "])"
    s = txt.find("R_MAT  = np.diag([")
    e = txt.find("])", s) + 2
    txt = txt[:s] + rb + txt[e:]

    txt = txt.replace("H_MIN, H_MAX = 0.106, 0.366", "H_MIN, H_MAX = 0.156, 0.356")
    LPATH.write_text(txt, encoding="utf-8")


def regenerate_and_get_K():
    """Run linearize.py, load resulting K matrix from .npz."""
    import subprocess
    r = subprocess.run([sys.executable, str(PROJ/"tools"/"linearize.py")],
                      capture_output=True, text=True, cwd=str(PROJ), timeout=120)
    m = re.search(r"max\\\|lambda\\\|: ([0-9.]+)", r.stdout)
    lam = float(m.group(1)) if m else 0.999

    dat = np.load(PROJ / "data" / "AB_sampling_points.npz", allow_pickle=True)
    K_all = dat['K']  # (n_grid, 4, 10)
    L_l = dat['L_l']
    # Find K at L=0.20 (closest to typical operating point)
    idx = np.argmin(np.abs(L_l - 0.20))
    K = K_all[idx]
    return lam, K


def score_K(K_sl, K_lw):
    """Compute similarity score between Serial_Leg K and legwheel K. Lower = more similar."""
    # Key ratios we care about (from legwheel):
    # R1: K[2,5]/abs(K[2,2]) — wheel: velocity-to-pitch ratio
    # R2: K[0,2]/abs(K[0,3]) — leg: pitch-to-self-leg ratio
    # R3: K[2,5] absolute — wheel velocity gain magnitude
    # R4: K[2,2] — wheel pitch coupling (should be modest, not dominant)
    # R5: K[0,2] — leg pitch gain (primary pitch control)

    def safe_ratio(a, b): return a / b if abs(b) > 1e-6 else 999

    # Target ratios from legwheel
    r1_lw = safe_ratio(K_lw[2,5], abs(K_lw[2,2]))  # -5.18/13.60 = -0.38
    r2_lw = safe_ratio(K_lw[0,2], abs(K_lw[0,3]))  # -66.36/22.03 = -3.01
    r3_lw = safe_ratio(K_lw[0,2], abs(K_lw[2,2]))  # pitch: leg vs wheel -66.36/13.60 = -4.88

    r1_sl = safe_ratio(K_sl[2,5], abs(K_sl[2,2]))
    r2_sl = safe_ratio(K_sl[0,2], abs(K_sl[0,3]))
    r3_sl = safe_ratio(K_sl[0,2], abs(K_sl[2,2]))

    # Score: weighted sum of ratio differences + penalty for unstable
    score = 0.0
    score += abs(r1_sl - r1_lw) * 5.0      # velocity/pitch in wheel
    score += abs(r2_sl - r2_lw) * 1.0      # pitch/self-leg in leg torque
    score += abs(r3_sl - r3_lw) * 2.0      # pitch leg/wheel distribution
    score += abs(K_sl[2,5] - K_lw[2,5]) * 0.2  # absolute velocity gain
    score += abs(K_sl[2,4]) * 0.05  # penalize large cross-coupling (th_R->T_wL)
    return score


if __name__ == "__main__":
    Klw = legwheel_K(0.20)
    print(f"Target (legwheel K @ L=0.20):")
    print(f"  T_wL: pitch={Klw[2,2]:7.2f} x_dot={Klw[2,5]:7.2f} th_L={Klw[2,3]:7.2f} th_R={Klw[2,4]:7.2f}")
    print(f"  tau_L: pitch={Klw[0,2]:7.2f} th_L={Klw[0,3]:7.2f} x_dot={Klw[0,5]:7.2f} th_R={Klw[0,4]:7.2f}")
    print(f"  Ratios: x_dot/pitch_w={Klw[2,5]/abs(Klw[2,2]):.3f} pitch_leg/wheel={Klw[0,2]/abs(Klw[2,2]):.2f}")
    print()

    # Search grid: key Q parameters
    Q_base = [10, 1, 10000, 4000, 4000]  # x, yaw, pitch, th_L, th_R (fixed)
    xd_vals = [5, 10, 15, 20, 30]
    pd_vals = [3, 5, 10, 15]
    td_vals = [3, 5, 10, 15]
    R_w_vals = [1.0]

    print(f"Scanning {len(xd_vals)*len(pd_vals)*len(td_vals)*len(R_w_vals)} combos...")
    print(f"{'Q_xd':>5} {'Q_pd':>5} {'Q_td':>5} {'R_w':>5} {'lam':>8} {'T_wL[p]':>9} {'T_wL[v]':>9} {'T_wL[thR]':>10} {'score':>8}")
    print("-" * 75)

    results = []
    total = len(xd_vals) * len(pd_vals) * len(td_vals) * len(R_w_vals)
    count = 0
    best_score = float('inf')

    for xd, pd, td, rw in itertools.product(xd_vals, pd_vals, td_vals, R_w_vals):
        Q = Q_base + [xd, 1, pd, td, td]
        R = [1.0, 1.0, rw, rw]
        apply_q(Q, R)
        lam, Ksl = regenerate_and_get_K()

        sc = score_K(Ksl, Klw)
        if lam >= 1.0:
            sc += 100  # heavy penalty for unstable
        results.append((xd, pd, td, rw, lam, Ksl[2,2], Ksl[2,5], Ksl[2,4], sc))

        count += 1
        if sc < best_score:
            best_score = sc
            print(f"{xd:5d} {pd:5d} {td:5d} {rw:5.1f} {lam:8.4f} {Ksl[2,2]:9.2f} {Ksl[2,5]:9.2f} {Ksl[2,4]:10.2f} {sc:8.2f} ★")

    results.sort(key=lambda x: x[8])
    print(f"\n=== TOP 10 (lowest score = closest to legwheel) ===")
    print(f"{'Rank':<5} {'Q_xd':>5} {'Q_pd':>5} {'Q_td':>5} {'R_w':>5} {'lam':>8} {'T_wL[p]':>9} {'T_wL[v]':>9} {'T_wL[thR]':>10} {'score':>8}")
    for i, r in enumerate(results[:10]):
        print(f"#{i+1:<4} {r[0]:5d} {r[1]:5d} {r[2]:5d} {r[3]:5.1f} {r[4]:8.4f} {r[5]:9.2f} {r[6]:9.2f} {r[7]:10.2f} {r[8]:8.2f}")

    # Apply best
    best = results[0]
    print(f"\nApplying best: Q_x_dot={best[0]} Q_pd={best[1]} Q_td={best[2]} R_w={best[3]}")
    apply_q([10,1,10000,4000,4000,best[0],1,best[1],best[2],best[2]], [1,1,best[3],best[3]])
    import subprocess
    subprocess.run([sys.executable, str(PROJ/"tools"/"linearize.py")], cwd=str(PROJ))
    print("Done. Best config applied and K regenerated.")
