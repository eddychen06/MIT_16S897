import numpy as np

def orbit_dyn(t, state, mu):
    r = state[0:3]
    v = state[3:6]
    r_mag = np.linalg.norm(r)
    a = (-mu / r_mag**3) * r
    return np.concatenate((v, a))

def attitude_dyn(t, omega, I, p=None, p_dot=None, tau=None):
    if p is None:
        p = np.zeros(3)
    if p_dot is None:
        p_dot = np.zeros(3)
    if tau is None:
        tau = np.zeros(3)

    I_inv = np.linalg.inv(I)
    return I_inv @ (tau - p_dot - np.cross(omega, I @ omega + p))

def quaternion_kinematics(q, omega):
    s = q[0]
    v = q[1:]
    ds = -0.5 * np.dot(v, omega)
    dv = 0.5 * (s*omega + np.cross(v, omega))

    return np.concatenate(([ds], dv))

def full_dyn(t, x, J, mu):
    q = x[0:4]
    w = x[4:7]
    p = x[7:10]
    r = x[10:13]
    v = x[13:16]

    r_mag = np.linalg.norm(r)
    dr = v
    dv = (-mu / r_mag**3) * r
    dq = quaternion_kinematics(q / np.linalg.norm(q), w)
    dp = np.zeros(3)
    I_inv = np.linalg.inv(J)
    dw = I_inv @ (-dp - np.cross(w, J @ w + p)) 


    return np.concatenate((dq, dw, dp, dr, dv))