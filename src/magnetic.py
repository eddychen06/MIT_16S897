import numpy as np
from src.dynamics import quaternion_to_matrix, full_dyn_controlled

MU_0 = 4.0 * np.pi * 1e-7
M_EARTH = 7.94e22
TILT_DEG = 11.0  
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
    B_b = B_body(x[0:4], x[10:13], t)
    tau_mag = np.cross(m_cmd, B_b)
    return full_dyn_controlled(t, x, J, mu, surfaces, env, tau_cmd, tau_ext=tau_mag)
