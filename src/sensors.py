import numpy as np
from src.estimation import L


def expq(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate(([np.cos(theta)], phi * np.sinc(theta / np.pi)))


class Sensor:
    def __init__(self, name, sigma_deg):
        self.name = name
        self.sigma_rad = np.radians(sigma_deg)
        self.R_cov = np.eye(3) * (self.sigma_rad ** 2)

    def measure(self, true_vector_body):
        w = np.random.multivariate_normal(np.zeros(3), self.R_cov)
        dq = expq(w / 2.0)
        s, e = dq[0], dq[1:4]
        v = true_vector_body
        rotated = v - 2 * s * np.cross(e, v) + 2 * np.cross(e, np.cross(e, v))
        return rotated / np.linalg.norm(rotated)


class VectorSensor:
    def __init__(self, name, M, b, sigma_deg):
        self.name = name
        self.M = np.asarray(M, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.sigma_rad = np.radians(sigma_deg)
        self.W = np.eye(3) * self.sigma_rad**2

    def measure(self, true_vector_body):
        w = np.random.multivariate_normal(np.zeros(3), self.W)
        y = self.M @ true_vector_body + self.b + w
        return y / np.linalg.norm(y)


class StarTracker:
    def __init__(self, name, sigma_cross_arcsec, sigma_bore_arcsec, boresight_axis=2):
        self.name = name
        self.boresight_axis = boresight_axis

        sigma_cross_rad = np.radians(sigma_cross_arcsec / 3600.0)
        sigma_bore_rad = np.radians(sigma_bore_arcsec / 3600.0)

        sigmas = np.array([sigma_cross_rad, sigma_cross_rad, sigma_cross_rad])
        sigmas[boresight_axis] = sigma_bore_rad
        self.W_st = np.diag(sigmas**2)

        self.sigma_cross_rad = sigma_cross_rad
        self.sigma_bore_rad = sigma_bore_rad

    def measure(self, q_true):
        delta_phi = np.random.multivariate_normal(np.zeros(3), self.W_st)
        dq = expq(delta_phi)
        q_meas = L(q_true) @ dq
        return q_meas / np.linalg.norm(q_meas)


class Gyroscope:
    def __init__(self, name, M, sigma_w_deg, sigma_beta_deg, b0=None):
        self.name = name
        self.M = np.asarray(M, dtype=float)
        self.sigma_w = np.radians(sigma_w_deg)
        self.sigma_beta = np.radians(sigma_beta_deg)
        self.bias = np.zeros(3) if b0 is None else np.asarray(b0, dtype=float).copy()

    def reset(self, b0=None):
        self.bias = np.zeros(3) if b0 is None else np.asarray(b0, dtype=float).copy()

    def measure(self, omega_true, dt):
        eta = np.random.normal(0, self.sigma_beta * np.sqrt(dt), size=3)
        self.bias += eta
        w = np.random.normal(0, self.sigma_w, size=3)
        return self.M @ omega_true + self.bias + w
