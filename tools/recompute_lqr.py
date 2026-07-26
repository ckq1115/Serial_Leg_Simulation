"""
Recompute LQR gain table for the Serial five-link wheel-leg robot.

Usage:
    1.  Export system matrices from MATLAB as a .mat file containing:
        A_set  : (n_lengths, 10, 10)  state matrix at each leg length
        B_set  : (n_lengths, 10, 4)   input matrix at each leg length
        h_vec  : (n_lengths,)         leg lengths [m]

    2.  Run:  python recompute_lqr.py sys_matrices_serial.mat

    3.  Copy the output file to data/lqr_gain_table.txt and/or update
        sim/change_length_fit.py with the new polynomial coefficients.
"""

import sys
import numpy as np
from scipy.io import loadmat
from scipy.linalg import solve_discrete_are
from pathlib import Path

DT = 0.002  # timestep [s]
N_ITER = 50000  # DARE iteration count

# ── Q / R cost matrices (tune these!) ──────────────────────────
# Higher value → stronger correction on that state/input.
# State order: [x, yaw, pitch, θL, θR, ẋ, ẏaw, pitcḣ, θL̇, θṘ]
# Input order: [τL_virtual, τR_virtual, Tw_L, Tw_R]
q_diag = np.array([
    10,     # [0] x   位置
    1,      # [1] yaw 偏航
    10000,  # [2] pitch
    4000,   # [3] θL  左腿角
    4000,   # [4] θR  右腿角
    50,    # [5] ẋ   速度
    1,      # [6] ẏaw 偏航速率
    1,      # [7] pitcḣ 俯仰阻尼
    1,     # [8] θL̇  腿速阻尼
    1,     # [9] θṘ
])
R = 1.0 * np.eye(4)  # [τL, τR, Tw_L, Tw_R] 控制代价

#python tools/recompute_lqr.py data/sys_matrices_serial.npz
# ────────────────────────────────────────────────────────────────


def solve_dare_iterative(Ad, Bd, Q, R, S, n_iter=N_ITER):
    """Iterative DARE solver (matches legwheel approach)."""
    P = S.copy()
    for _ in range(n_iter):
        F = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
        P_new = (Ad - Bd @ F).T @ P @ (Ad - Bd @ F) + F.T @ R @ F + Q
        if np.linalg.norm(P_new - P, 'fro') / max(np.linalg.norm(P, 'fro'), 1e-15) < 1e-8:
            P = P_new
            break
        P = P_new
    F = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return F, P


def fit_poly_4th(h_vec, K_set):
    """Fit 4th-order polynomial to each K[i,j] vs leg length.

    Returns
    -------
    coeff : (4, 10, 5)  polynomial coefficients [c4, c3, c2, c1, c0]
    """
    n_len = len(h_vec)
    n_out, n_state = 4, 10
    coeff = np.zeros((n_out, n_state, 5))
    A_poly = np.column_stack([h_vec**4, h_vec**3, h_vec**2, h_vec, np.ones(n_len)])
    for i in range(n_out):
        for j in range(n_state):
            coeff[i, j, :] = np.linalg.lstsq(A_poly, K_set[:, i, j], rcond=None)[0]
    return coeff


def write_gain_table(path, h_vec, K_set, theta_balance=0.0):
    """Write gain table in the same format as Serial's lqr_gain_table.txt."""
    with open(path, 'w') as f:
        for k in range(len(h_vec)):
            f.write(f"{h_vec[k]:.2f}\n")
            f.write(f"theta_balance = {theta_balance}\n")
            f.write("F_convergence:\n")
            np.savetxt(f, K_set[k], fmt='%12.8f')
            f.write("\n")


def write_polynomial_fit(path, coeff, theta_coeff, e1_coeff, e2_coeff, e3_coeff):
    """Write Python file with polynomial coefficients."""
    with open(path, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write("class ChangeLengthFit:\n")
        f.write("    def __init__(self):\n")
        f.write(f"        self.k_coeff = np.array({coeff.tolist()})\n")
        f.write(f"        self.theta_coeff = np.array({theta_coeff.tolist()})\n")
        f.write(f"        self.e1_coeff = np.array({e1_coeff.tolist()})\n")
        f.write(f"        self.e2_coeff = np.array({e2_coeff.tolist()})\n")
        f.write(f"        self.e3_coeff = np.array({e3_coeff.tolist()})\n")
        f.write("\n")
        f.write("    def _poly5(self, coeff, L):\n")
        f.write("        return coeff[0]*L**4 + coeff[1]*L**3 + coeff[2]*L**2 + coeff[3]*L + coeff[4]\n")
        f.write("\n")
        f.write("    def get_K(self, L):\n")
        f.write("        K = np.zeros((4, 10))\n")
        f.write("        for i in range(4):\n")
        f.write("            for j in range(10):\n")
        f.write("                K[i, j] = self._poly5(self.k_coeff[i, j, :], L)\n")
        f.write("        return K\n")
        f.write("\n")
        f.write("    def get_theta_target(self, L):\n")
        f.write("        return self._poly5(self.theta_coeff, L)\n")
        f.write("\n")
        f.write("    def get_e1(self, L):\n")
        f.write("        return self._poly5(self.e1_coeff, L)\n")
        f.write("\n")
        f.write("    def get_e2(self, L):\n")
        f.write("        return self._poly5(self.e2_coeff, L)\n")
        f.write("\n")
        f.write("    def get_e3(self, L):\n")
        f.write("        return self._poly5(self.e3_coeff, L)\n")
    print(f"Wrote {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python recompute_lqr.py <sys_matrices>")
        print("Accepts .npz (from linearize_model.py) or .mat (from MATLAB).")
        sys.exit(1)

    fpath = sys.argv[1]
    matrices_are_discrete = False
    if fpath.endswith('.npz'):
        data_npz = np.load(fpath)
        A_set = data_npz['A_set']
        B_set = data_npz['B_set']
        h_vec = data_npz['h_vec']
        matrices_are_discrete = (
            'discrete' in data_npz.files and bool(np.asarray(data_npz['discrete']).item())
        )
    elif fpath.endswith('.mat'):
        data = loadmat(fpath)
        A_set = data['A_set']
        B_set = data['B_set']
        h_vec = data['h_vec'].ravel()
        # MATLAB saves A_set/B_set as (state, input/state, length).
        # Python code below expects the length dimension first.
        if A_set.shape == (10, 10, len(h_vec)):
            A_set = np.moveaxis(A_set, 2, 0)
        if B_set.shape == (10, 4, len(h_vec)):
            B_set = np.moveaxis(B_set, 2, 0)
    else:
        print("Error: file must be .npz or .mat")
        sys.exit(1)

    n_len = len(h_vec)
    print(f"Loaded {n_len} linearisation points, h = {h_vec[0]:.2f} ~ {h_vec[-1]:.2f} m")
    print("Matrix type:", "discrete" if matrices_are_discrete else "continuous")

    S = np.diag(q_diag)
    Q = S
    K_set = np.zeros((n_len, 4, 10))

    # ── discretise if needed + solve DARE at each leg length ──
    from scipy.signal import cont2discrete
    for k in range(n_len):
        A, B = A_set[k], B_set[k]
        if matrices_are_discrete:
            Ad, Bd = A, B
        else:
            Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(10), np.zeros((10, 4))), DT, method='zoh')
        # Solve
        F, _ = solve_dare_iterative(Ad, Bd, Q, R, S)
        K_set[k] = F

        # Sanity check: closed-loop eigenvalues should be inside unit circle
        eigs = np.linalg.eigvals(Ad - Bd @ F)
        max_eig = np.max(np.abs(eigs))
        if max_eig >= 1.0:
            print(f"  WARNING: h={h_vec[k]:.2f} unstable! max|lambda|={max_eig:.4f}")
        else:
            print(f"  h={h_vec[k]:.2f}  max|lambda|={max_eig:.4f}  ok")

    # ── output 1: gain table ──
    table_path = Path("data/lqr_gain_table_serial.txt")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_gain_table(table_path, h_vec, K_set)
    print(f"\nGain table written to {table_path}")

    # ── output 2: polynomial fit ──
    coeff = fit_poly_4th(h_vec, K_set)
    # placeholder e1/e2/e3/theta — recompute when you have leg balance data
    theta_coeff = np.zeros(5)
    e1_coeff = np.zeros(5)
    e2_coeff = np.zeros(5)
    e3_coeff = np.zeros(5)

    poly_path = Path("data/change_length_fit_serial.py")
    write_polynomial_fit(poly_path, coeff, theta_coeff, e1_coeff, e2_coeff, e3_coeff)

    print("\nDone.  Remember to update robot_params.yaml:")
    print("  lqr_gain_table: data/lqr_gain_table_serial.txt")
    print("  change_length_fit: data/change_length_fit_serial.py")


if __name__ == "__main__":
    main()