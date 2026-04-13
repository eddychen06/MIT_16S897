import numpy as np
from src.estimation import L, R, Q, H, T
from src.sensors import expq


def _G(q):
    return L(q) @ H


class MEKF:
    def __init__(self, q0, beta0, P0, sigma_w, sigma_beta):
        self.q = q0 / np.linalg.norm(q0)
        self.beta = np.array(beta0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.sigma_w = sigma_w
        self.sigma_beta = sigma_beta

    def predict(self, gyro_meas, dt):
        omega = gyro_meas - self.beta
        dq = expq(0.5 * dt * omega)
        q_pred = L(self.q) @ dq
        q_pred /= np.linalg.norm(q_pred)

        A = self._prediction_jacobian(self.q, q_pred, dq, dt)
        V = self._process_noise(dt)

        self.q = q_pred
        self.P = A @ self.P @ A.T + V

    def _prediction_jacobian(self, q, q_pred, dq, dt):
        Gq = _G(q)
        Gqn = _G(q_pred)
        A11 = Gqn.T @ R(dq) @ Gq
        A = np.zeros((6, 6))
        A[:3, :3] = A11
        A[:3, 3:] = -0.5 * dt * np.eye(3)
        A[3:, 3:] = np.eye(3)
        return A

    def _process_noise(self, dt):
        V = np.zeros((6, 6))
        V[:3, :3] = self.sigma_w**2 * np.eye(3)
        V[3:, 3:] = self.sigma_beta**2 * dt * np.eye(3)
        return V

    def _apply_update(self, z, C_mat, W):
        S = C_mat @ self.P @ C_mat.T + W
        K = self.P @ C_mat.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T

        dx = K @ z
        phi = dx[:3]
        d_beta = dx[3:]

        phi_sq = np.dot(phi, phi)
        if phi_sq < 1.0:
            dq = np.concatenate(([np.sqrt(1.0 - phi_sq)], phi))
        else:
            dq = expq(phi)
        self.q = L(self.q) @ dq
        self.q /= np.linalg.norm(self.q)

        self.beta = self.beta + d_beta

        I6 = np.eye(6)
        IKC = I6 - K @ C_mat
        self.P = IKC @ self.P @ IKC.T + K @ W @ K.T

    def update_vector(self, y_meas, r_eci, W_vec):
        y_pred = Q(self.q).T @ r_eci
        z = y_meas - y_pred
        C = self._vector_jacobian(self.q, r_eci)
        self._apply_update(z, C, W_vec)

    def _vector_jacobian(self, q, r_N):
        Hr = H @ r_N
        Gq = _G(q)
        C_att = H.T @ (L(q).T @ L(Hr) + R(q) @ R(Hr) @ T) @ Gq
        C = np.zeros((3, 6))
        C[:, :3] = C_att
        return C

    def update_star_tracker(self, q_meas, W_st):
        dq = L(self.q).T @ q_meas
        if dq[0] < 0:
            dq = -dq
        z = dq[1:4]

        C = np.zeros((3, 6))
        C[:3, :3] = np.eye(3)
        self._apply_update(z, C, W_st)
