import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M = 4.015 # kg 
LX = 0.100 # m
LY = 0.100 # m 
LZ = 0.3405 # m

CM = np.array([0.0, 0.0, 0.0])

IXX = M / 12.0 * (LY**2 + LZ**2)
IYY = M / 12.0 * (LX**2 + LZ**2)
IZZ = M / 12.0 * (LX**2 + LY**2)
IB = np.diag([IXX, IYY, IZZ])
II = np.linalg.inv(IB)

surfaces = {
    "+X": {"normal": np.array([1, 0, 0]),  "area": 0.034, "centroid": np.array([0.05, 0, 0])},
    "-X": {"normal": np.array([-1, 0, 0]), "area": 0.034, "centroid": np.array([-0.05, 0, 0])},
    "+Y": {"normal": np.array([0, 1, 0]),  "area": 0.034, "centroid": np.array([0, 0.05, 0])},
    "-Y": {"normal": np.array([0, -1, 0]), "area": 0.034, "centroid": np.array([0, -0.05, 0])},
    "+Z": {"normal": np.array([0, 0, 1]),  "area": 0.010, "centroid": np.array([0, 0, 0.17])},
    "-Z": {"normal": np.array([0, 0, -1]), "area": 0.010, "centroid": np.array([0, 0, -0.17])}
}

evals, evecs = np.linalg.eigh(IB) 
sort_idx = np.argsort(evals)
IP = evals[sort_idx]
RBP = evecs[:, sort_idx]

def rk4(func, y0, t_eval):
    N = len(t_eval)
    Y = np.zeros((N, len(y0)))
    Y[0] = y0

    for i in range(N-1):
        dt = t_eval[i+1] - t_eval[i]
        y_curr = Y[i]

        k1 = func(y_curr)
        k2 = func(y_curr + 0.5 * dt * k1)
        k3 = func(y_curr + 0.5 * dt * k2)
        k4 = func(y_curr + dt * k3)
        
        Y[i+1] = y_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return Y
        

mu = 3.98600e5
R_EARTH = 6378

def dynamics(state):
    r = state[0:3]
    v = state[3:6]
    a = (-mu / np.linalg.norm(r)**3) * r
    return np.concatenate((v, a))

R0 = np.array([R_EARTH + 400, 0.0, 0.0])
V0 = np.array([0.0, 0.0, np.sqrt(mu / (R_EARTH + 400))])

T = 2 * np.pi * np.sqrt((R_EARTH + 400)**3 / mu)
t_eval = np.linspace(0, 2*T, 1000)
Y0 = np.concatenate((R0, V0))
Y = rk4(dynamics, Y0, t_eval)

X_SAT = Y[:, 0]
Y_SAT = Y[:, 1]
Z_SAT = Y[:, 2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(X_SAT, Y_SAT, Z_SAT, color="b")
ax.plot([X_SAT[0]], [Y_SAT[0]], [Z_SAT[0]], color="r", marker="o")


u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
X_EARTH = R_EARTH * np.outer(np.cos(u), np.sin(v))
Y_EARTH = R_EARTH * np.outer(np.sin(u), np.sin(v))
Z_EARTH = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(X_EARTH, Y_EARTH, Z_EARTH, color="g", alpha=0.2)

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')

max_val = np.max(np.abs(Y[:, 0:3]))
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])
ax.set_box_aspect([1, 1, 1]) 

plt.show()