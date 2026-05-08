import numpy as np
import scipy.linalg
from src.utils import expq, quat_error, quat_left_matrix, quat_multiply, hat


def attitude_error_state(q, omega, q_des, omega_des=None):
    if omega_des is None:
        omega_des = np.zeros(3)
    dq = quat_error(q, q_des)
    return np.concatenate((dq[1:4], omega - omega_des))


def eigen_axis_slew_trajectory(q0, qf, t, maneuver_time, J):
    q0 = q0 / np.linalg.norm(q0)
    qf = qf / np.linalg.norm(qf)
    dq_total = quat_left_matrix(q0).T @ qf
    if dq_total[0] < 0.0:
        dq_total = -dq_total
    dq_total /= np.linalg.norm(dq_total)

    sin_half = np.linalg.norm(dq_total[1:4])
    theta = 2.0 * np.arctan2(sin_half, dq_total[0])
    if sin_half < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = dq_total[1:4] / sin_half

    q_ref = np.zeros((len(t), 4))
    omega_ref = np.zeros((len(t), 3))
    alpha_ref = np.zeros((len(t), 3))
    torque_ref = np.zeros((len(t), 3))
    angle_ref = np.zeros(len(t))

    for k, tk in enumerate(t):
        u = np.clip(tk / maneuver_time, 0.0, 1.0)
        if tk <= maneuver_time:
            angle = 0.5 * theta * (1.0 - np.cos(np.pi * u))
            angle_dot = 0.5 * np.pi * theta / maneuver_time * np.sin(np.pi * u)
            angle_ddot = 0.5 * np.pi**2 * theta / maneuver_time**2 * np.cos(np.pi * u)
        else:
            angle = theta
            angle_dot = 0.0
            angle_ddot = 0.0

        q_ref[k] = quat_multiply(q0, expq(0.5 * angle * axis))
        q_ref[k] /= np.linalg.norm(q_ref[k])
        omega_ref[k] = angle_dot * axis
        alpha_ref[k] = angle_ddot * axis
        torque_ref[k] = J @ alpha_ref[k] + np.cross(omega_ref[k], J @ omega_ref[k])
        angle_ref[k] = angle

    return {
        "q": q_ref,
        "omega": omega_ref,
        "alpha": alpha_ref,
        "torque": torque_ref,
        "angle": angle_ref,
        "axis": axis,
        "total_angle": theta,
    }


def attitude_regulation_linearization(J):
    A = np.zeros((6, 6))
    A[:3, 3:] = 0.5 * np.eye(3)

    B = np.zeros((6, 3))
    B[3:, :] = np.linalg.inv(J)
    return A, B


def attitude_error_linearization(J, omega_ref, p_ref=None):
    omega_ref = np.zeros(3) if omega_ref is None else np.asarray(omega_ref, dtype=float)
    p_ref = np.zeros(3) if p_ref is None else np.asarray(p_ref, dtype=float)
    A = np.zeros((6, 6))
    A[:3, :3] = -hat(omega_ref)
    A[:3, 3:] = 0.5 * np.eye(3)
    h_total = J @ omega_ref + p_ref
    A[3:, 3:] = np.linalg.solve(J, hat(h_total) - hat(omega_ref) @ J)
    return A


def tvlqr(A, B, Q, R, Qf, t):
    t = np.asarray(t, dtype=float)
    N = len(t)
    nx = Q.shape[0]
    nu = R.shape[0]

    if A.ndim == 2:
        A_seq = np.repeat(A[None, :, :], N, axis=0)
    else:
        A_seq = A

    if B.ndim == 2:
        B_seq = np.repeat(B[None, :, :], N, axis=0)
    else:
        B_seq = B

    S = np.zeros((N, nx, nx))
    K = np.zeros((N, nu, nx))
    S[-1] = Qf.copy()

    for k in range(N - 2, -1, -1):
        dt_k = t[k + 1] - t[k]
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = A_seq[k]
        M[:nx, nx:] = B_seq[k]
        eM = scipy.linalg.expm(M * dt_k)
        Ad = eM[:nx, :nx]
        Bd = eM[:nx, nx:]

        S_next = S[k + 1]
        Phi = R + Bd.T @ S_next @ Bd
        K[k] = np.linalg.solve(Phi, Bd.T @ S_next @ Ad)
        S[k] = Q + Ad.T @ S_next @ Ad - Ad.T @ S_next @ Bd @ K[k]
        S[k] = 0.5 * (S[k] + S[k].T)

    if N > 1:
        K[-1] = K[-2]

    return K, S


class TVLQRAttitudeRegulator:
    def __init__(self, J, q_des, t, Q, R, Qf=None, torque_limit=None, omega_ref_traj=None, p_ref_traj=None):
        self.J = J
        self.q_des = q_des / np.linalg.norm(q_des)
        self.t = np.asarray(t, dtype=float)
        self.torque_limit = torque_limit

        B = np.zeros((6, 3))
        B[3:, :] = np.linalg.inv(J)

        if omega_ref_traj is None:
            A, _ = attitude_regulation_linearization(J)
        else:
            omega_ref_traj = np.asarray(omega_ref_traj, dtype=float)
            if omega_ref_traj.shape != (len(self.t), 3):
                raise ValueError("omega_ref_traj must have shape (len(t), 3)")
            if p_ref_traj is None:
                p_ref_traj = np.zeros_like(omega_ref_traj)
            else:
                p_ref_traj = np.asarray(p_ref_traj, dtype=float)
                if p_ref_traj.shape != (len(self.t), 3):
                    raise ValueError("p_ref_traj must have shape (len(t), 3)")
            A = np.zeros((len(self.t), 6, 6))
            for k in range(len(self.t)):
                A[k] = attitude_error_linearization(J, omega_ref_traj[k], p_ref_traj[k])

        if Qf is None:
            Qf = Q
        self.K, self.S = tvlqr(A, B, Q, R, Qf, self.t)

    def gain(self, t_now):
        idx = np.searchsorted(self.t, t_now, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.t) - 1))
        return self.K[idx]

    def torque(self, t_now, q_est, omega_est):
        x_err = attitude_error_state(q_est, omega_est, self.q_des)
        tau = -self.gain(t_now) @ x_err

        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        return tau
