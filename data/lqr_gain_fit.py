"""Auto-generated LQR gain fits for the Serial wheel-leg robot.

Provides:
  get_K(L_l, L_r)  → 4×10 K matrix  (2D monomial)
  get_e2(L_l, L_r) → pitch equilibrium offset  (2D)
  get_e3(L_l, L_r) → (e3_l, e3_r) leg-theta equilibrium offsets  (2D)

Regenerate:  python tools/linearize.py
"""

import numpy as np
from pathlib import Path

_data_dir = Path(__file__).resolve().parent

E2_COEFF = np.load(_data_dir / "_E2_COEFF.npy")
E3_COEFF = np.load(_data_dir / "_E3_COEFF.npy")
_K2D_COEFF = np.load(_data_dir / "_K2D_COEFF.npy")  # (4, 10, 28)
_K2D_ORDER = 6
_K2D_N_TERMS = _K2D_COEFF.shape[2]


def _poly4(c, L):
    return c[0]*L**4 + c[1]*L**3 + c[2]*L**2 + c[3]*L + c[4]


def get_e2(L_l, L_r):
    """Pitch equilibrium offset at (L_l, L_r) [m]."""
    L = 0.5 * (L_l + L_r)
    return _poly4(E2_COEFF, L)


def get_e3(L_l, L_r):
    """Leg-theta equilibrium offsets at (L_l, L_r) [m].  Returns (e3_l, e3_r)."""
    return _poly4(E3_COEFF, L_l), _poly4(E3_COEFF, L_r)


def _poly2d_terms(p, L_l, L_r):
    """Evaluate all 2D monomials up to order p at (L_l, L_r)."""
    out = []
    for d in range(p + 1):
        for i in range(d + 1):
            out.append((L_l ** (d - i)) * (L_r ** i))
    return np.array(out, dtype=float)


def get_K(L_l, L_r):
    """Return 4x10 K matrix (2D monomial basis)."""
    terms = _poly2d_terms(_K2D_ORDER, L_l, L_r)
    c = _K2D_COEFF  # (4, 10, 28)
    K = np.zeros((4, 10))
    for i in range(4):
        for j in range(10):
            K[i, j] = np.dot(c[i, j, :], terms)
    return K
