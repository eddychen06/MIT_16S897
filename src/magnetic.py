import numpy as np
from src.dynamics import quaternion_to_matrix, full_dyn_controlled

# L14 p.1: Earth's field as a tilted dipole, m_hat fixed in ECEF.
# B(r) = (mu_0 m_E) / (4 pi r^3) * (3 (m_hat . r_hat) r_hat - m_hat)
MU_0 = 4.0 * np.pi * 1e-7
M_EARTH = 7.94e22
TILT_DEG = 11.0  # L14 p.1: "Tilt is ~11 deg"
OMEGA_EARTH = 7.2921159e-5


def m_hat_eci(t):
    tilt = np.radians(TILT_DEG)
    c = np.cos(OMEGA_EARTH * t)
    s = np.sin(OMEGA_EARTH * t)
    return np.array([c * np.sin(tilt), s * np.sin(tilt), -np.cos(tilt)])


def B_eci(r_eci_km, t):
    r_m = np.asarray(r_eci_km, dtype=float) * 1000.0
    r_mag = np.linalg.norm(r_m)
    r_hat = r_m / r_mag
    m_hat = m_hat_eci(t)
    coeff = (MU_0 * M_EARTH) / (4.0 * np.pi * r_mag**3)
    return coeff * (3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat)


def B_body(q, r_eci_km, t):
    R_body_to_eci = quaternion_to_matrix(q)
    return R_body_to_eci.T @ B_eci(r_eci_km, t)


def full_dyn_magnetic(t, x, J, mu, surfaces, env, tau_cmd, m_cmd):
    # Sample-hold m_cmd over the control interval, but recompute B_body(q, r, t)
    # at each RK4 substep so the magnetic torque tracks the evolving state.
    B_b = B_body(x[0:4], x[10:13], t)
    tau_mag = np.cross(m_cmd, B_b)
    return full_dyn_controlled(t, x, J, mu, surfaces, env, tau_cmd, tau_ext=tau_mag)
