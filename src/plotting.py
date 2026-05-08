import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def _cumulative_trapezoid(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    out = np.zeros_like(y)
    if len(x) > 1:
        dx = np.diff(x)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx[:, None], axis=0)
    return out


def plot_environmental_torques(t, T_orbit, gg_torque, drag_torque, combined_torque=None):
    t_orbits = t / T_orbit
    torque_cases = [
        ("Gravity gradient", gg_torque),
        ("Atmospheric drag", drag_torque),
    ]
    if combined_torque is not None:
        torque_cases.append(("Combined", combined_torque))

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    floor = 1e-10
    for name, torque in torque_cases:
        norms = np.linalg.norm(torque, axis=1)
        axes[0].semilogy(t_orbits, np.maximum(norms, floor), label=name)
        mean_norm = np.trapezoid(norms, t) / (t[-1] - t[0])
        axes[0].axhline(mean_norm, linewidth=0.8, linestyle='--', alpha=0.7,
                        color=axes[0].lines[-1].get_color(),
                        label=f'{name} mean')
    axes[0].set_ylim(bottom=floor)
    axes[0].set_ylabel('Torque magnitude (N m)')
    axes[0].set_title('Environmental Torque History and Momentum Accumulation')
    axes[0].grid(True, which='both')
    axes[0].legend(ncol=2, fontsize=8, loc='lower right')

    component_labels = [r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']
    for i, label in enumerate(component_labels):
        axes[1].plot(t_orbits, gg_torque[:, i], label=label)
        axes[2].plot(t_orbits, drag_torque[:, i], label=label)
    axes[1].set_ylabel('GG torque (N m)')
    axes[1].grid(True)
    axes[1].legend(ncol=3, fontsize=8, loc='upper right')

    axes[2].set_ylabel('Drag torque (N m)')
    axes[2].grid(True)
    axes[2].legend(ncol=3, fontsize=8, loc='upper right')

    for name, torque in torque_cases:
        delta_h = _cumulative_trapezoid(torque, t)
        axes[3].plot(t_orbits, np.linalg.norm(delta_h, axis=1), label=name)
    axes[3].set_xlabel('Time (orbits)')
    axes[3].set_ylabel(r'$|\Delta H|$ (N m s)')
    axes[3].grid(True)
    axes[3].legend(loc='upper left')

    plt.tight_layout()
    _save_fig(fig, "environmental_torques.png")

    labels = [name for name, _ in torque_cases]
    mean_abs_day = []
    net_day = []
    mean_vectors = []
    duration = t[-1] - t[0]
    for _, torque in torque_cases:
        norms = np.linalg.norm(torque, axis=1)
        mean_norm = np.trapezoid(norms, t) / duration
        mean_vec = np.trapezoid(torque, t, axis=0) / duration
        mean_abs_day.append(mean_norm * 86400.0)
        net_day.append(np.linalg.norm(mean_vec) * 86400.0)
        mean_vectors.append(mean_vec)

    x = np.arange(len(labels))
    width = 0.36
    mean_vectors = np.asarray(mean_vectors)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    floor_h = 1e-7
    scalar_plot = np.maximum(mean_abs_day, floor_h)
    net_plot = np.maximum(net_day, floor_h)
    axes[0].bar(x - width / 2, scalar_plot, width, label=r'$\int |\tau| dt$ per day')
    axes[0].bar(x + width / 2, net_plot, width, label=r'$|\int \tau dt|$ per day')
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=floor_h)
    axes[0].axhline(1.5e-3, color='0.35', linewidth=0.8, linestyle=':', label='RW storage 1.5e-3')
    for xi, val in zip(x - width / 2, mean_abs_day):
        axes[0].text(xi, val * 1.15, f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    for xi, val in zip(x + width / 2, net_day):
        axes[0].text(xi, max(val, floor_h) * 1.15, f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    axes[0].set_ylabel('Angular momentum (N m s/day)')
    axes[0].set_title('Environmental Momentum Budget')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis='y', which='both')
    axes[0].legend(loc='lower right', fontsize=9)

    for i, label in enumerate(component_labels):
        axes[1].bar(x + (i - 1) * width / 1.5, mean_vectors[:, i], width / 1.5, label=label)
    linthresh = 1e-11
    axes[1].set_yscale('symlog', linthresh=linthresh)
    axes[1].axhline(0, color='0.5', linewidth=0.6)
    axes[1].set_ylabel('Orbit-average torque (N m, symlog)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis='y', which='both')
    axes[1].legend(ncol=3, loc='upper right')

    plt.tight_layout()
    _save_fig(fig, "environmental_momentum_budget.png")


def plot_environmental_response(t, T_orbit, case_solutions):
    t_orbits = t / T_orbit
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, sol in case_solutions.items():
        omega_norm_rad = np.linalg.norm(sol[:, 4:7], axis=1)
        cumulative_rad = np.zeros_like(omega_norm_rad)
        if len(t) > 1:
            cumulative_rad[1:] = np.cumsum(0.5 * (omega_norm_rad[1:] + omega_norm_rad[:-1]) * np.diff(t))
        omega_norm_deg = np.degrees(omega_norm_rad)
        axes[0].plot(t_orbits, np.degrees(cumulative_rad), label=name)
        axes[1].plot(t_orbits, omega_norm_deg, label=name)

    axes[0].set_ylabel(r'Cumulative rotation $\int|\vec{\omega}|\,dt$ (deg)')
    axes[0].set_title('Uncontrolled Attitude Response to Environmental Torques')
    axes[0].grid(True)
    axes[0].legend(loc='upper left')

    axes[1].set_xlabel('Time (orbits)')
    axes[1].set_ylabel('Angular-rate norm (deg/s)')
    axes[1].grid(True)
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    _save_fig(fig, "environmental_response.png")


def plot_attitude_regulation(t, cases, basename, torque_limit=None, momentum_limit=None):
    has_h = any("wheel_momentum" in v for v in cases.values())
    n_panels = 4 if has_h else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)

    floor_att = 1e-3
    for name, data in cases.items():
        axes[0].semilogy(t, np.maximum(data["att_err_deg"], floor_att), label=name)
        axes[1].plot(t, np.degrees(np.linalg.norm(data["omega"], axis=1)), label=name)
        axes[2].plot(t, np.max(np.abs(data["torque"]), axis=1), label=name)
        if has_h and "wheel_momentum" in data:
            axes[3].plot(t, np.max(np.abs(data["wheel_momentum"]), axis=1), label=name)

    axes[0].set_ylabel("Attitude error (deg)")
    axes[0].set_title("TVLQR Attitude Regulation")
    axes[0].grid(True, which='both')
    axes[0].legend(loc='upper right')

    axes[1].set_ylabel("Rate norm (deg/s)")
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

    if torque_limit is not None and torque_limit > 0.0:
        axes[2].axhline(torque_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Cmd limit {torque_limit:.0e}')
    axes[2].set_ylabel(r"$\max_i\,|\tau_i|$ (N m)")
    axes[2].grid(True)
    axes[2].legend(loc='upper right')

    if has_h:
        if momentum_limit is not None and momentum_limit > 0.0:
            axes[3].axhline(momentum_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Storage {momentum_limit:.1e}')
        axes[3].set_ylabel(r"$\max_i\,|h_{rw,i}|$ (N m s)")
        axes[3].grid(True)
        axes[3].legend(loc='upper right')

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    _save_fig(fig, basename)


def plot_attitude_regulation_orbit(t, T_orbit, att_err_deg, torque, estimate_err_deg, basename, wheel_momentum=None, torque_limit=None, momentum_limit=None):
    t_orbits = t / T_orbit
    n_panels = 4 if wheel_momentum is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)

    floor_att = 1e-3
    axes[0].semilogy(t_orbits, np.maximum(att_err_deg, floor_att))
    axes[0].set_ylabel("Pointing error (deg)")
    axes[0].set_title("TVLQR Regulation with Disturbances and MEKF Estimates")
    axes[0].grid(True, which='both')

    axes[1].plot(t_orbits, np.max(np.abs(torque), axis=1))
    if torque_limit is not None and torque_limit > 0.0:
        axes[1].axhline(torque_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Cmd limit {torque_limit:.0e}')
        axes[1].legend(loc='upper right')
    axes[1].set_ylabel(r"$\max_i\,|\tau_i|$ (N m)")
    axes[1].grid(True)

    if wheel_momentum is not None:
        component_labels = ['x', 'y', 'z']
        component_colors = ['tab:blue', 'tab:orange', 'tab:green']
        for i, (label, color) in enumerate(zip(component_labels, component_colors)):
            axes[2].plot(t_orbits, wheel_momentum[:, i], color=color, label=fr'$h_{{rw,{label}}}$')
        if momentum_limit is not None and momentum_limit > 0.0:
            axes[2].axhline(momentum_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Storage $\\pm${momentum_limit:.1e}')
            axes[2].axhline(-momentum_limit, color='0.35', linewidth=0.8, linestyle=':')
        axes[2].set_ylabel("Wheel momentum (N m s)")
        axes[2].grid(True)
        axes[2].legend(ncol=4, fontsize=8, loc='upper right')

    floor_mekf = 1e-3
    axes[-1].semilogy(t_orbits, np.maximum(estimate_err_deg, floor_mekf))
    axes[-1].set_xlabel("Time (orbits)")
    axes[-1].set_ylabel("MEKF attitude error (deg)")
    axes[-1].grid(True, which='both')

    plt.tight_layout()
    _save_fig(fig, basename)


def plot_eigen_axis_slew(t, nominal, closed_loop, tracking_err_deg, torque_cmd, basename, torque_limit=None, wheel_momentum=None, momentum_limit=None):
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(t, np.degrees(nominal["angle"]), label="Nominal")
    axes[0].plot(t, np.degrees(closed_loop["angle"]), label="Closed-loop")
    axes[0].set_ylabel("Slew angle (deg)")
    axes[0].set_title("Eigen-Axis Slew Tracking")
    axes[0].grid(True)
    axes[0].legend(loc='lower right')

    component_labels = ['x', 'y', 'z']
    component_colors = ['tab:blue', 'tab:orange', 'tab:green']
    omega_nom = np.degrees(nominal["omega"])
    omega_cl = np.degrees(closed_loop["omega"])
    for i, (label, color) in enumerate(zip(component_labels, component_colors)):
        axes[1].plot(t, omega_nom[:, i], color=color, label=fr'$\omega_{label}$ nominal')
        axes[1].plot(t, omega_cl[:, i], '--', color=color, alpha=0.7, label=fr'$\omega_{label}$ closed-loop')
    axes[1].set_ylabel("Body rate (deg/s)")
    axes[1].grid(True)
    axes[1].legend(ncol=3, fontsize=8, loc='upper right')

    for i, (label, color) in enumerate(zip(component_labels, component_colors)):
        axes[2].plot(t, nominal["torque"][:, i], color=color, label=fr'$\tau_{label}$ nominal')
        axes[2].plot(t, torque_cmd[:, i], '--', color=color, alpha=0.7, label=fr'$\tau_{label}$ commanded')
    cmd_max = float(np.max(np.abs(torque_cmd))) if torque_cmd.size else 0.0
    nom_max = float(np.max(np.abs(nominal["torque"]))) if nominal["torque"].size else 0.0
    data_max = max(cmd_max, nom_max)
    if data_max > 0.0:
        axes[2].set_ylim(-1.4 * data_max, 1.4 * data_max)
    if torque_limit is not None and torque_limit > 0.0:
        axes[2].axhline(torque_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Cmd limit $\\pm${torque_limit:.0e}')
        axes[2].axhline(-torque_limit, color='0.35', linewidth=0.8, linestyle=':')
    axes[2].set_ylabel("Torque (N m)")
    axes[2].grid(True)
    axes[2].legend(ncol=3, fontsize=8, loc='upper right')

    if wheel_momentum is not None:
        for i, (label, color) in enumerate(zip(component_labels, component_colors)):
            axes[3].plot(t, wheel_momentum[:, i], color=color, label=fr'$h_{{rw,{label}}}$')
        if momentum_limit is not None and momentum_limit > 0.0:
            axes[3].axhline(momentum_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Storage $\\pm${momentum_limit:.1e}')
            axes[3].axhline(-momentum_limit, color='0.35', linewidth=0.8, linestyle=':')
        axes[3].set_ylabel("Wheel momentum (N m s)")
        axes[3].grid(True)
        axes[3].legend(ncol=4, fontsize=8, loc='upper right')
    else:
        axes[3].set_visible(False)

    err_floor = max(1e-3, 0.5 * float(np.min(tracking_err_deg[tracking_err_deg > 0])) if np.any(tracking_err_deg > 0) else 1e-3)
    axes[4].semilogy(t, np.maximum(tracking_err_deg, err_floor))
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Tracking error (deg)")
    axes[4].grid(True, which='both')

    plt.tight_layout()
    _save_fig(fig, basename)


def plot_versine_profile(t, nominal, basename, maneuver_time=None):
    angle_deg = np.degrees(nominal["angle"])
    omega_norm = np.degrees(np.linalg.norm(nominal["omega"], axis=1))
    alpha_norm = np.degrees(np.linalg.norm(nominal["alpha"], axis=1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, angle_deg)
    axes[0].set_ylabel(r"$\theta$ (deg)")
    axes[0].set_title("Position-Versine Reference Profile")
    axes[0].grid(True)

    axes[1].plot(t, omega_norm)
    axes[1].set_ylabel(r"$\dot\theta$ (deg/s)")
    axes[1].grid(True)

    axes[2].plot(t, alpha_norm)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(r"$|\ddot\theta|$ (deg/s$^2$)")
    axes[2].grid(True)

    if maneuver_time is not None:
        for ax in axes:
            ax.axvline(maneuver_time, color='0.5', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    _save_fig(fig, basename)


def plot_slew_vs_regulator(t, slew, regulator, basename, torque_limit=None, momentum_limit=None):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t, slew["att_err_deg"], label="Eigen-axis slew", color='tab:blue')
    axes[0].plot(t, regulator["att_err_deg"], label="Regulator", color='tab:red')
    axes[0].set_ylabel("Attitude error (deg)")
    axes[0].set_title(r"$180^\circ$ Maneuver: Eigen-Axis Slew vs. Regulator")
    axes[0].grid(True)
    axes[0].legend(loc='upper right')

    axes[1].plot(t, np.max(np.abs(slew["torque"]), axis=1), label="Eigen-axis slew", color='tab:blue')
    axes[1].plot(t, np.max(np.abs(regulator["torque"]), axis=1), label="Regulator", color='tab:red')
    if torque_limit is not None and torque_limit > 0.0:
        axes[1].axhline(torque_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Cmd limit {torque_limit:.0e}')
    axes[1].set_ylabel(r"$\max_i\,|\tau_i|$ (N m)")
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

    slew_h = np.max(np.abs(slew["wheel_momentum"]), axis=1)
    reg_h = np.max(np.abs(regulator["wheel_momentum"]), axis=1)
    axes[2].plot(t, slew_h, label="Eigen-axis slew", color='tab:blue')
    axes[2].plot(t, reg_h, label="Regulator", color='tab:red')
    if momentum_limit is not None and momentum_limit > 0.0:
        axes[2].axhline(momentum_limit, color='0.35', linewidth=0.8, linestyle=':', label=f'Storage {momentum_limit:.1e}')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(r"$\max_i\,|h_{rw,i}|$ (N m s)")
    axes[2].grid(True)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    _save_fig(fig, basename)
