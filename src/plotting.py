import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def _save_fig(fig, basename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / basename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

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
    _save_fig(fig, "orbit.png")

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
    stem = "_".join(k.replace(" ", "_") for k in solutions.keys())
    _save_fig(fig, f"{stem}.png")

def plot_momentum_sphere(h_mag, I_p, trajectories):
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
    axis_labels = ['Major Axis', 'Intermediate Axis', 'Minor Axis']
    axis_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for idx, w_sol in enumerate(trajectories):
        h_sol = (I_p_mat @ w_sol.T).T
        h_sol_norm = np.zeros_like(h_sol)
        for i in range(len(h_sol)):
            norm = np.linalg.norm(h_sol[i, :])
            if norm > 0:
                h_sol_norm[i, :] = h_mag * h_sol[i, :] / norm
            else:
                h_sol_norm[i, :] = h_sol[i, :]
        label = axis_labels[idx] if idx < len(axis_labels) else None
        color = axis_colors[idx] if idx < len(axis_colors) else 'blue'
        ax.plot(h_sol_norm[:, 0], h_sol_norm[:, 1], h_sol_norm[:, 2],
                color=color, alpha=0.7, linewidth=1.5, label=label)

    ax.set_xlabel('Minor Axis (H1)')
    ax.set_ylabel('Intermediate Axis (H2)')
    ax.set_zlabel('Major Axis (H3)')
    ax.set_title('Momentum Sphere Trajectories')
    ax.legend(loc='upper right')
    ax.set_box_aspect([1, 1, 1])
    _save_fig(fig, "momentum_sphere.png")

def plot_full_dyn(t, sol, errors):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, sol[:, 0], label='q0 (scalar)')
    ax1.plot(t, sol[:, 1], label='q1 (x)')
    ax1.plot(t, sol[:, 2], label='q2 (y)')
    ax1.plot(t, sol[:, 3], label='q3 (z)')
    ax1.set_title('Attitude Quaternion Components')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Quaternion Value')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, errors)
    ax2.set_title('Solar Panel Pointing Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (degrees)')
    ax2.grid(True)
    plt.tight_layout()
    _save_fig(fig, "full_dyn.png")


def plot_mekf_errors(t, att_errors, bias_errors, P_hist):
    att_deg = np.degrees(att_errors)
    bias_deg = np.degrees(bias_errors)
    sigma_att = np.degrees(np.sqrt(np.array([P_hist[:, i, i] for i in range(3)])).T)
    sigma_bias = np.degrees(np.sqrt(np.array([P_hist[:, i+3, i+3] for i in range(3)])).T)

    labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, att_deg[:, i], 'b', linewidth=0.8, label=r'$\phi_{%s}$' % labels[i])
        ax.plot(t, 3 * sigma_att[:, i], 'r--', linewidth=0.6, label=r'$3\sigma$')
        ax.plot(t, -3 * sigma_att[:, i], 'r--', linewidth=0.6)
        ax.set_ylabel('deg')
        ax.set_title(f'Attitude error ({labels[i]})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, bias_deg[:, i], 'b', linewidth=0.8,
                label=r'$\delta\beta_{%s}$' % labels[i])
        ax.plot(t, 3 * sigma_bias[:, i], 'r--', linewidth=0.6, label=r'$3\sigma$')
        ax.plot(t, -3 * sigma_bias[:, i], 'r--', linewidth=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('deg/s')
        ax.set_title(f'Bias error ({labels[i]})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    _save_fig(fig, "mekf_errors.png")


def plot_mc_attitude_errors(t, mc_err_norms_deg):
    N_mc = mc_err_norms_deg.shape[0]
    err_deg = mc_err_norms_deg

    median = np.median(err_deg, axis=0)
    p5 = np.maximum(np.percentile(err_deg, 5, axis=0), 1e-6)
    p95 = np.percentile(err_deg, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(N_mc):
        ax.semilogy(t, np.maximum(err_deg[i], 1e-6),
                    color='steelblue', alpha=0.15, linewidth=0.5)

    ax.semilogy(t, np.maximum(median, 1e-6), 'navy', linewidth=2,
                label='Median')
    ax.fill_between(t, p5, p95, color='steelblue', alpha=0.3,
                    label='5th\u201395th percentile')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Attitude error (deg)')
    ax.set_title(f'MEKF Attitude Error ({N_mc} Monte Carlo runs, random ICs)')
    ax.legend()
    ax.grid(True, which='both')
    plt.tight_layout()
    _save_fig(fig, "mekf_mc_errors.png")


def plot_mc_convergence(conv_results):
    err_vals = sorted(set(v['err_deg'] for v in conv_results.values()))
    p_vals = sorted(set(v['p_scale'] for v in conv_results.values()))

    fig, axes = plt.subplots(1, len(p_vals), figsize=(6 * len(p_vals), 5),
                             sharey=True)
    if len(p_vals) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(err_vals)))

    for j, ps in enumerate(p_vals):
        ax = axes[j]
        for i, ed in enumerate(err_vals):
            label = f"{ed}deg_P{ps}"
            if label in conv_results:
                d = conv_results[label]
                traces = d['traces']
                t = d['t']
                med = np.maximum(np.median(traces, axis=0), 1e-6)
                p10 = np.maximum(np.percentile(traces, 10, axis=0), 1e-6)
                p90 = np.percentile(traces, 90, axis=0)

                ax.semilogy(t, med, color=colors[i], linewidth=1.5,
                            label=f'{ed}\u00b0')
                ax.fill_between(t, p10, p90, color=colors[i], alpha=0.2)

        ax.set_xlabel('Time (s)')
        if j == 0:
            ax.set_ylabel('Attitude error (deg)')
        ax.set_title(f'$P_0$ scale = {ps}')
        ax.legend(title='Init error', fontsize=8)
        ax.grid(True, which='both')

    fig.suptitle('MEKF Convergence Study (10 MC trials per config)',
                 fontsize=14)
    plt.tight_layout()
    _save_fig(fig, "mekf_convergence.png")
