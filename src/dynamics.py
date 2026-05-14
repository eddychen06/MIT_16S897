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


def quaternion_to_matrix(q):
    q = q / np.linalg.norm(q)
    s, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - s*z), 2.0 * (x*z + s*y)],
        [2.0 * (x*y + s*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - s*x)],
        [2.0 * (x*z - s*y), 2.0 * (y*z + s*x), 1.0 - 2.0 * (x*x + y*y)],
    ])


def gravity_gradient_torque(q, r, J, mu):
    r_mag_m = np.linalg.norm(r) * 1000.0
    r_hat_eci = r / np.linalg.norm(r)
    R_body_to_eci = quaternion_to_matrix(q)
    r_hat_body = R_body_to_eci.T @ r_hat_eci
    mu_m = mu * 1e9
    return 3.0 * mu_m / r_mag_m**3 * np.cross(r_hat_body, J @ r_hat_body)


def drag_torque(q, r, v, surfaces, rho=4.76e-13, cd=2.2):
    omega_earth = np.array([0.0, 0.0, 7.2921159e-5])
    r_m = 1000.0 * r
    v_rel_eci = 1000.0 * v - np.cross(omega_earth, r_m)

    R_body_to_eci = quaternion_to_matrix(q)
    v_rel_body = R_body_to_eci.T @ v_rel_eci
    speed = np.linalg.norm(v_rel_body)
    if speed == 0.0:
        return np.zeros(3)

    v_hat = v_rel_body / speed
    dynamic_pressure = 0.5 * rho * speed**2
    tau = np.zeros(3)

    for surface in surfaces.values():
        n = surface["n"]
        projected_area = surface["A"] * max(0.0, np.dot(n, v_hat))
        if projected_area == 0.0:
            continue
        force = -cd * dynamic_pressure * projected_area * v_hat
        tau += np.cross(surface["r_c"], force)

    return tau


def environmental_torque(q, r, v, J, mu, surfaces=None, include_gg=True, include_drag=True, rho=4.76e-13, cd=2.2):
    tau = np.zeros(3)
    if include_gg:
        tau += gravity_gradient_torque(q, r, J, mu)
    if include_drag and surfaces is not None:
        tau += drag_torque(q, r, v, surfaces, rho=rho, cd=cd)
    return tau


def full_dyn_env(t, x, J, mu, surfaces, env):
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
    tau = environmental_torque(
        q,
        r,
        v,
        J,
        mu,
        surfaces=surfaces,
        include_gg=env.get("gravity_gradient", True),
        include_drag=env.get("drag", True),
        rho=env.get("rho", 4.76e-13),
        cd=env.get("cd", 2.2),
    )
    I_inv = np.linalg.inv(J)
    dw = I_inv @ (tau - dp - np.cross(w, J @ w + p))

    return np.concatenate((dq, dw, dp, dr, dv))


def full_dyn_controlled(t, x, J, mu, surfaces, env, tau_cmd, tau_ext=None):
    if tau_ext is None:
        tau_ext = np.zeros(3)
    q = x[0:4]
    w = x[4:7]
    p = x[7:10]
    r = x[10:13]
    v = x[13:16]

    r_mag = np.linalg.norm(r)
    dr = v
    dv = (-mu / r_mag**3) * r
    dq = quaternion_kinematics(q / np.linalg.norm(q), w)
    dp = -tau_cmd
    tau_env = environmental_torque(
        q,
        r,
        v,
        J,
        mu,
        surfaces=surfaces,
        include_gg=env.get("gravity_gradient", True),
        include_drag=env.get("drag", True),
        rho=env.get("rho", 4.76e-13),
        cd=env.get("cd", 2.2),
    )
    I_inv = np.linalg.inv(J)
    dw = I_inv @ (tau_env + tau_ext - dp - np.cross(w, J @ w + p))

    return np.concatenate((dq, dw, dp, dr, dv))
