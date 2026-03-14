import numpy as np

class Sensor:
    def __init__(self, name, sigma_deg):
        self.name = name
        self.sigma_rad = np.radians(sigma_deg)
        self.R = np.eye(3) * (self.sigma_rad**2)

    def measure(self, true_vector_body):
        noise = np.random.multivariate_normal([0, 0, 0], self.R)
        noisy_vector = true_vector_body + noise
        return noisy_vector / np.linalg.norm(noisy_vector)
