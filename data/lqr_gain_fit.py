"""Auto-generated LQR gain fits for the Serial wheel-leg robot.

Provides:
  get_K(L_l, L_r)  → 4×10 K matrix  (2D Chebyshev)
  get_e2(L_l, L_r) → pitch equilibrium offset  (2D)
  get_e3(L_l, L_r) → (e3_l, e3_r) leg-theta equilibrium offsets  (2D)

Regenerate:  python tools/linearize.py
"""

import numpy as np
from pathlib import Path

_data_dir = Path(__file__).resolve().parent

E2_COEFF = np.load(_data_dir / "_E2_COEFF.npy")
E3_COEFF = np.load(_data_dir / "_E3_COEFF.npy")
_K2D_COEFF = np.load(_data_dir / "_K2D_COEFF.npy")  # (4, 10, n_terms)
_K2D_N_TERMS = _K2D_COEFF.shape[2]
_K2D_ORDER = int((np.sqrt(8 * _K2D_N_TERMS + 1) - 3) / 2)
if (_K2D_ORDER + 1) * (_K2D_ORDER + 2) // 2 != _K2D_N_TERMS:
    raise ValueError(f"Invalid triangular K coefficient count: {_K2D_N_TERMS}")

H_MIN = 0.106
H_MAX = 0.366


def _poly4(c, L):
    return c[0]*L**4 + c[1]*L**3 + c[2]*L**2 + c[3]*L + c[4]


def get_e2(L_l, L_r):
    """Pitch equilibrium offset at (L_l, L_r) [rad]."""
    L = 0.5 * (L_l + L_r)
    return _poly4(E2_COEFF, L)


def get_e3(L_l, L_r):
    """Leg-theta equilibrium offsets at (L_l, L_r) [rad].  Returns (e3_l, e3_r)."""
    return _poly4(E3_COEFF, L_l), _poly4(E3_COEFF, L_r)


def _normalize_length(L):
    return (2.0 * L - H_MIN - H_MAX) / (H_MAX - H_MIN)


def _cheb_values(p, x):
    vals = np.empty(p + 1, dtype=float)
    vals[0] = 1.0
    if p >= 1:
        vals[1] = x
    for k in range(2, p + 1):
        vals[k] = 2.0 * x * vals[k - 1] - vals[k - 2]
    return vals


def _poly2d_terms(p, L_l, L_r):
    """Evaluate triangular 2D Chebyshev terms up to total order p."""
    T_l = _cheb_values(p, _normalize_length(L_l))
    T_r = _cheb_values(p, _normalize_length(L_r))
    out = []
    for d in range(p + 1):
        for i in range(d + 1):
            out.append(T_l[d - i] * T_r[i])
    return np.array(out, dtype=float)


def get_K(L_l, L_r):
    """Return 4x10 K matrix (2D triangular Chebyshev basis)."""
    terms = _poly2d_terms(_K2D_ORDER, L_l, L_r)
    c = _K2D_COEFF  # (4, 10, n_terms)
    K = np.zeros((4, 10))
    for i in range(4):
        for j in range(10):
            K[i, j] = np.dot(c[i, j, :], terms)
    return K
