"""Auto-generated LQR gain fits. Regenerate: python tools/linearize.py.

States: x_hip, yaw, pitch, theta_L, theta_R, then their rates.
Inputs: tau_L, tau_R, T_wL, T_wR. Leg equilibrium angles are fitted separately.
E3_COEFF retains the legacy common quartic; get_e3 uses the full equilibrium fit.
"""
from pathlib import Path
import numpy as np

_data_dir = Path(__file__).resolve().parent
E2_COEFF = np.load(_data_dir / "_E2_COEFF.npy", allow_pickle=False)
E3_COEFF = np.load(_data_dir / "_E3_COEFF.npy", allow_pickle=False)
_K2D_COEFF = np.load(_data_dir / "_K2D_COEFF.npy", allow_pickle=False)
_U_EQ_COEFF = np.load(_data_dir / "_U_EQ_COEFF.npy", allow_pickle=False)
_K2D_N_TERMS = _K2D_COEFF.shape[2]
_K2D_ORDER = int((np.sqrt(8 * _K2D_N_TERMS + 1) - 3) / 2)
if (_K2D_ORDER + 1) * (_K2D_ORDER + 2) // 2 != _K2D_N_TERMS:
    raise ValueError("Invalid triangular K coefficient count")

H_MIN = 0.1
H_MAX = 0.34
_EQUILIBRIUM_ORDER = 8
_EQUILIBRIUM2D_COEFF = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06697342544794083, -0.05571267753839493, 1.1001932280873796e-17, 0.012830909341573715, 2.5832987014832126e-18, -1.514533456307234e-17, -0.003948062192648649, 5.8141416834331494e-18, -3.25193286156565e-17, -5.613299749163437e-18, 0.0009086227510124445, 5.759681766263883e-17, 1.7300710813395638e-17, 5.4891051976335e-18, 7.761467120448336e-18, -0.0002597580896690488, -4.795865144023631e-17, 1.2959703354664302e-17, -1.9069241161007488e-17, 3.904705088939832e-17, -1.7786654607852166e-17, 4.738175994134508e-05, 3.176090656434133e-17, 1.9176147637735066e-18, 1.0372329547893347e-17, 1.037085964794484e-17, -2.9450898824668595e-17, -5.444115257707047e-18, -1.253533991985023e-05, 5.004716916318879e-18, -3.3718551493088423e-18, -7.663884623585455e-18, 1.9778563425522735e-17, 1.1899074175275333e-17, 5.436282601768721e-17, -2.3915409474186184e-17, 5.497134907272994e-07, 2.159187236841453e-17, 5.637851296924623e-18, 2.926833675475569e-17, 7.508100044462118e-18, 1.1377304361308522e-17, 9.107298248878237e-18, 1.84876008414847e-17, 2.0816681711721685e-17], [0.06454012542963028, 8.673617379884035e-19, -0.053506702184677124, -2.0377030253066076e-17, 1.438767020919827e-17, 0.012342146597802639, -1.3389584155924435e-17, -2.3535993307736686e-18, -1.463214178487709e-17, -0.003789702896028757, 2.1519214477769097e-17, -6.480416145382157e-17, 1.1963940851731129e-17, 4.8460301740164e-18, 0.0008723928476683795, -2.0286366294846588e-17, 2.456341118553165e-17, -1.3278961156704656e-17, -6.813224902935416e-18, -7.817441109010819e-17, -0.000248809956246987, 1.5090788412324215e-18, -3.577401301081175e-17, -4.93066233120915e-18, 9.756324837002657e-18, 7.268440178406517e-18, 8.011068208541722e-17, 4.540384543361142e-05, -2.2393544695430377e-19, -4.0147502197836817e-17, -6.603240968535767e-18, 1.2296475729783572e-17, -2.356008080717423e-17, -2.703833888701276e-17, -9.893238283067331e-17, -1.1974593689956237e-05, 1.1275702593849246e-17, -2.606901224927617e-17, 6.451002926288751e-18, -4.37451191313443e-17, 7.96549783597944e-18, -2.1921123339435137e-17, -1.235990476633475e-17, 3.0550972938747767e-18, 5.296251401887275e-07], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)


_equilibrium_path = _data_dir / "_EQUILIBRIUM2D_COEFF.npy"
if _equilibrium_path.is_file():
    with np.load(_data_dir / "AB_sampling_points.npz", allow_pickle=False) as _samples:
        if "body_com" in _samples:
            _EQUILIBRIUM2D_COEFF = np.load(_equilibrium_path, allow_pickle=False)
            _equilibrium_terms = _EQUILIBRIUM2D_COEFF.shape[1]
            _EQUILIBRIUM_ORDER = int((np.sqrt(8*_equilibrium_terms+1)-3)/2)
            if (_EQUILIBRIUM2D_COEFF.shape[0] != 10 or
                    (_EQUILIBRIUM_ORDER+1)*(_EQUILIBRIUM_ORDER+2)//2 != _equilibrium_terms):
                raise ValueError("Invalid equilibrium coefficient shape")
            H_MIN = float(min(np.min(_samples["L_l"]), np.min(_samples["L_r"])))
            H_MAX = float(max(np.max(_samples["L_l"]), np.max(_samples["L_r"])))


def _poly4(coefficients, length):
    return np.polyval(coefficients, length)


def _normalize_length(length):
    if not np.isfinite(length):
        raise ValueError("Leg lengths must be finite")
    length = np.clip(length, H_MIN, H_MAX)
    return (2.0 * length - H_MIN - H_MAX) / (H_MAX - H_MIN)


def _cheb_values(order, normalized):
    values = np.empty(order + 1, dtype=float)
    values[0] = 1.0
    if order:
        values[1] = normalized
    for degree in range(2, order + 1):
        values[degree] = 2.0 * normalized * values[degree - 1] - values[degree - 2]
    return values


def _poly2d_terms(order, L_l, L_r):
    left = _cheb_values(order, _normalize_length(L_l))
    right = _cheb_values(order, _normalize_length(L_r))
    return np.array([left[degree-index] * right[index]
                     for degree in range(order+1) for index in range(degree+1)])


def get_K(L_l, L_r):
    """Return row-major K[4,10]."""
    return _K2D_COEFF @ _poly2d_terms(_K2D_ORDER, L_l, L_r)


def get_e2(L_l, L_r):
    """Return the pitch equilibrium offset in radians."""
    return float(_EQUILIBRIUM2D_COEFF[2] @ _poly2d_terms(_EQUILIBRIUM_ORDER, L_l, L_r))


def get_e3(L_l, L_r):
    """Return separate (left, right) leg equilibrium offsets in radians."""
    angles = _EQUILIBRIUM2D_COEFF[3:5] @ _poly2d_terms(_EQUILIBRIUM_ORDER, L_l, L_r)
    return float(angles[0]), float(angles[1])


def get_u_eq(L_l, L_r):
    """Return physical gravity-balancing input [tau_L,tau_R,T_wL,T_wR]."""
    return (_U_EQ_COEFF @ _poly2d_terms(_K2D_ORDER, L_l, L_r)).reshape(4)
