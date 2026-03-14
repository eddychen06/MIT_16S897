import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_orbit(sol_orbit, planet_radius):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol_orbit[:, 0], sol_orbit[:, 1], sol_orbit[:, 2], color="b", label="Trajectory")
    ax.plot([sol_orbit[0, 0]], [sol_orbit[0, 1]], [sol_orbit[0, 2]], color="r", marker="o", label="Start")

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X_e = planet_radius * np.outer(np.cos(u), np.sin(v))
    Y_e = planet_radius * np.outer(np.sin(u), np.sin(v))
    Z_e = planet_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(X_e, Y_e, Z_e, color="g", alpha=0.2)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title("Orbital Simulation")
    ax.set_box_aspect([1, 1, 1])
    plt.legend()
    plt.show()

def plot_attitude_stability(t_att, solutions):
    num_cases = len(solutions)
    fig, axes = plt.subplots(num_cases, 1, figsize=(10, 4 * num_cases))
    
    if num_cases == 1:
        axes = [axes]

    for ax, (name, sol) in zip(axes, solutions.items()):
        ax.plot(t_att, sol)
        ax.set_title(f"{name} Stability")
        ax.legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_momentum_sphere(h_mag, I_p, trajectories, filename=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X_h = h_mag * np.outer(np.cos(u), np.sin(v))
    Y_h = h_mag * np.outer(np.sin(u), np.sin(v))
    Z_h = h_mag * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(X_h, Y_h, Z_h, color='g', alpha=0.1)

    h_stable = np.array([[h_mag, 0, 0], [-h_mag, 0, 0], [0, 0, h_mag], [0, 0, -h_mag]])
    h_unstable = np.array([[0, h_mag, 0], [0, -h_mag, 0]])

    ax.scatter(h_stable[:, 0], h_stable[:, 1], h_stable[:, 2], color='yellow', s=100, label='Stable')
    ax.scatter(h_unstable[:, 0], h_unstable[:, 1], h_unstable[:, 2], color='red', s=100, label='Unstable')

    I_p_mat = np.diag(I_p)
    for w_sol in trajectories:
        h_sol = (I_p_mat @ w_sol.T).T
        h_sol_norm = np.zeros_like(h_sol)
        for i in range(len(h_sol)):
            norm = np.linalg.norm(h_sol[i, :])
            if norm > 0:
                h_sol_norm[i, :] = h_mag * h_sol[i, :] / norm
            else:
                h_sol_norm[i, :] = h_sol[i, :]
        
        ax.plot(h_sol_norm[:, 0], h_sol_norm[:, 1], h_sol_norm[:, 2], color='blue', alpha=0.6, linewidth=1)

    ax.set_xlabel('Minor Axis (H1)')
    ax.set_ylabel('Intermediate Axis (H2)')
    ax.set_zlabel('Major Axis (H3)')
    ax.set_title('Momentum Sphere Trajectories')
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def plot_full_dyn(t, sol, errors):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, sol[:, 0], label='q0 (scalar)')
    ax1.plot(t, sol[:, 1], label='q1 (x)')
    ax1.plot(t, sol[:, 2], label='q2 (y)')
    ax1.plot(t, sol[:, 3], label='q3 (z)')
    ax1.set_title('Attitude Quaternion Components (ECI to Body)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Quaternion Value')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, errors)
    ax2.set_title('Solar Panel Pointing Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (degrees)')
    ax2.grid(True)
    plt.show()
