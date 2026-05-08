import numpy as np
from src.utils import rk4, box_inertia, parallel_axis_theorem, perturb_inertia, hat, expq, quat_error
from src.dynamics import orbit_dyn, attitude_dyn, full_dyn, full_dyn_env, full_dyn_controlled, gravity_gradient_torque, drag_torque, quaternion_to_matrix
from src.spacecraft import Spacecraft
from src.plotting import plot_orbit, plot_attitude_stability, plot_momentum_sphere, plot_full_dyn, plot_environmental_torques, plot_environmental_response, plot_attitude_regulation, plot_attitude_regulation_orbit, plot_eigen_axis_slew, plot_mekf_errors, plot_mc_attitude_errors, plot_mc_convergence, plot_versine_profile, plot_slew_vs_regulator
from src.sensors import Sensor, VectorSensor, StarTracker, Gyroscope
from src.estimation import solve_wahba_svd, qmethod, triad, Q, L as L_q
from src.mekf import MEKF
from src.control import TVLQRAttitudeRegulator, attitude_error_state, eigen_axis_slew_trajectory
import time
import copy


def sample_affine_error(M_error_scale, b_scale):
    M_error = np.random.normal(0.0, np.abs(M_error_scale))
    b = np.random.normal(0.0, np.abs(b_scale))
    return np.eye(3) + M_error, b


def make_vector_sensor(name, M_error_scale, b_scale, sigma_deg):
    M, b = sample_affine_error(M_error_scale, b_scale)
    return VectorSensor(name, M, b, sigma_deg=sigma_deg)


def make_gyro(name, M_error_scale, b0_scale, sigma_w_deg, sigma_beta_deg):
    M, _ = sample_affine_error(M_error_scale, np.zeros(3))
    b0 = np.random.normal(0.0, np.abs(b0_scale))
    return Gyroscope(name, M, sigma_w_deg=sigma_w_deg, sigma_beta_deg=sigma_beta_deg, b0=b0)


def main(run_mekf_studies=True, run_convergence_study=True, seed=16):
    if seed is not None:
        np.random.seed(seed)

    m1 = 3.015
    m2 = 1.0
    m = m1+m2

    w = 0.1
    d = 0.14
    h1 = 0.2
    h2 = 0.1405
    h = h1 + h2

    r1 = np.array([0.0, 0.0, h1/2.0])
    r2 = np.array([0.0, 0.0, h1+h2/2.0])

    I1 = box_inertia(m1, w, d, h1)
    I2 = box_inertia(m2, w, d, h2)

    r_com = (m1*r1 + m2*r2)/m

    I_body = (parallel_axis_theorem(I1, m1, r1-r_com)+parallel_axis_theorem(I2, m2, r2-r_com))

    evals, _ = np.linalg.eigh(I_body)
    I_principal = np.sort(evals)

    z_com = r_com[2]
    surfaces = {
        "+X": {"n": np.array([1, 0, 0]),  "A": d*h, "r_c": np.array([w/2, 0, h/2 - z_com])},
        "-X": {"n": np.array([-1, 0, 0]), "A": d*h, "r_c": np.array([-w/2, 0, h/2 - z_com])},
        "+Y": {"n": np.array([0, 1, 0]),  "A": w*h, "r_c": np.array([0, d/2, h/2 - z_com])},
        "-Y": {"n": np.array([0, -1, 0]), "A": w*h, "r_c": np.array([0, -d/2, h/2 - z_com])},
        "+Z": {"n": np.array([0, 0, 1]),  "A": w*d, "r_c": np.array([0, 0, h - z_com])},
        "-Z": {"n": np.array([0, 0, -1]), "A": w*d, "r_c": np.array([0, 0, -z_com])}
    }

    aalto = Spacecraft(m, r_com, I_body, I_principal, surfaces)

    mu = 3.98600e5
    R_e = 6378
    h_alt = 500

    r0 = np.array([R_e + h_alt, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, np.sqrt(mu / (R_e + h_alt))])
    y0_orbit = np.concatenate((r0, v0))

    T_orbit = 2 * np.pi * np.sqrt((R_e + h_alt)**3 / mu)
    t_orbit = np.linspace(0, 5 * T_orbit, 1000)

    sol_orbit = rk4(orbit_dyn, y0_orbit, t_orbit, args=(mu,))
    plot_orbit(sol_orbit, R_e)

    w_mag_rpm = 10
    w_mag_rad = w_mag_rpm * 2 * np.pi / 60
    h_mag = aalto.I_principal[2] * w_mag_rad

    w_major = np.array([0.0, 0.0, h_mag / aalto.I_principal[2]])
    w_inter = np.array([0.0, h_mag / aalto.I_principal[1], 0.0])
    w_minor = np.array([h_mag / aalto.I_principal[0], 0.0, 0.0])

    w_pert = np.array([0.01, 0.01, 0.01])
    t_span = np.linspace(0, 500, 2000)
    I_p_mat = np.diag(aalto.I_principal)

    sol_major = rk4(attitude_dyn, w_major + w_pert, t_span, args=(I_p_mat,))
    sol_inter = rk4(attitude_dyn, w_inter + w_pert, t_span, args=(I_p_mat,))
    sol_minor = rk4(attitude_dyn, w_minor + w_pert, t_span, args=(I_p_mat,))

    plot_attitude_stability(t_span, {"Major Axis": sol_major,"Intermediate Axis": sol_inter,"Minor Axis": sol_minor})

    sphere_trajectories = []
    t_sphere = np.linspace(0, 200, 2000)
    w_pert_sphere = 0.3 * w_mag_rad * np.array([0.01, 0.01, 0.01]) / np.linalg.norm([0.01, 0.01, 0.01])
    for w0_axis in [w_major, w_inter, w_minor]:
        w_sol = rk4(attitude_dyn, w0_axis + w_pert_sphere, t_sphere, args=(I_p_mat,))
        sphere_trajectories.append(w_sol)

    plot_momentum_sphere(h_mag, aalto.I_principal, sphere_trajectories)

    spin_rate_rpm = 10.0
    omega_mag = spin_rate_rpm * (2 * np.pi / 60)
    n_solar = np.array([1, 0, 0])
    omega_desired = omega_mag * n_solar
    print(f"desired omega: {omega_desired}")

    J_tilde = perturb_inertia(aalto.I_body)
    J_max = np.max(np.linalg.eigvalsh(J_tilde))
    Jeff = 1.2 * J_max
    Js = (omega_desired.T @ J_tilde @ omega_desired) / (omega_mag**2)
    rho_s = (Jeff - Js) * omega_mag
    A = np.vstack([omega_desired.reshape(1, 3), hat(omega_desired)])
    b = np.concatenate([[omega_mag * rho_s], -np.cross(omega_desired, J_tilde @ omega_desired)])
    rotor_momentum = np.linalg.pinv(A) @ b
    print(f"rotor momentum: {rotor_momentum}")

    t_stable_spin = np.linspace(0, 500, 2000)
    omega0 = omega_desired + np.array([0.05, 0.05, 0.05])
    sol_stable_spin = rk4(attitude_dyn, omega0, t_stable_spin, args=(J_tilde, rotor_momentum, np.zeros(3), np.zeros(3)))
    plot_attitude_stability(t_stable_spin, {"Stable Spin": sol_stable_spin})

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = np.concatenate((q0, omega0, rotor_momentum, r0, v0))

    t_sim = np.linspace(0, 200, 5000)
    sol_full = rk4(full_dyn, x0, t_sim, args=(J_tilde, mu))

    sun_eci = np.array([1.0, 0.0, 0.0])
    panel_body = np.array([1.0, 0.0, 0.0])
    errors = []

    for i in range(len(t_sim)):
        q = sol_full[i, 0:4]
        R_mat = quaternion_to_matrix(q)
        sun_body = R_mat.T @ sun_eci
        cos_theta = np.dot(sun_body, panel_body)
        errors.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    plot_full_dyn(t_sim, sol_full, np.array(errors))

    rho_500km = 4.76e-13
    cd_drag = 2.2
    t_env = np.linspace(0, 5 * T_orbit, 2500)
    x0_env = np.concatenate((q0, np.zeros(3), np.zeros(3), r0, v0))
    env_cases = {
        "None": {"gravity_gradient": False, "drag": False, "rho": rho_500km, "cd": cd_drag},
        "Gravity gradient": {"gravity_gradient": True, "drag": False, "rho": rho_500km, "cd": cd_drag},
        "Drag": {"gravity_gradient": False, "drag": True, "rho": rho_500km, "cd": cd_drag},
        "Gravity gradient + drag": {"gravity_gradient": True, "drag": True, "rho": rho_500km, "cd": cd_drag},
    }

    env_solutions = {}
    for case_name, env in env_cases.items():
        sol_env = rk4(full_dyn_env, x0_env, t_env, args=(I_body, mu, surfaces, env))
        sol_env[:, 0:4] /= np.linalg.norm(sol_env[:, 0:4], axis=1, keepdims=True)
        env_solutions[case_name] = sol_env

    gg_torque_hist = np.array([
        gravity_gradient_torque(x[0:4], x[10:13], I_body, mu)
        for x in env_solutions["Gravity gradient"]]
    )
    drag_torque_hist = np.array([
        drag_torque(x[0:4], x[10:13], x[13:16], surfaces, rho=rho_500km, cd=cd_drag)
        for x in env_solutions["Drag"]]
    )
    combined_torque_hist = np.array([
        gravity_gradient_torque(x[0:4], x[10:13], I_body, mu)
        + drag_torque(x[0:4], x[10:13], x[13:16], surfaces, rho=rho_500km, cd=cd_drag)
        for x in env_solutions["Gravity gradient + drag"]]
    )

    plot_environmental_torques(t_env, T_orbit, gg_torque_hist, drag_torque_hist, combined_torque_hist)
    plot_environmental_response(t_env, T_orbit, env_solutions)

    def print_environment_response_stats(case_solutions):
        print("Environmental response over five orbits")
        for name, sol in case_solutions.items():
            q0_abs = np.clip(np.abs(sol[:, 0]), -1.0, 1.0)
            attitude_change_deg = np.degrees(2.0 * np.arccos(q0_abs))
            rate_norm_deg_s = np.degrees(np.linalg.norm(sol[:, 4:7], axis=1))
            print(
                f"{name}: final_att_change={attitude_change_deg[-1]:.3f} deg, "
                f"max_att_change={np.max(attitude_change_deg):.3f} deg, "
                f"final_rate={rate_norm_deg_s[-1]:.5f} deg/s, "
                f"max_rate={np.max(rate_norm_deg_s):.5f} deg/s"
            )

    def print_torque_stats(name, torques):
        norms = np.linalg.norm(torques, axis=1)
        mean_vec = np.trapezoid(torques, t_env, axis=0) / (t_env[-1] - t_env[0])
        mean_norm = np.trapezoid(norms, t_env) / (t_env[-1] - t_env[0])
        scalar_day = mean_norm * 86400.0
        net_day = np.linalg.norm(mean_vec) * 86400.0
        wheel_capacity = 1.5e-3
        dump_interval_hr = wheel_capacity / mean_norm / 3600.0 if mean_norm > 0 else np.inf
        print(
            f"{name}: max={np.max(norms):.3e} N m, "
            f"orbit_avg_vec={mean_vec}, mean|tau|={mean_norm:.3e} N m, "
            f"scalar_day={scalar_day:.3e} N m s/day, "
            f"net_day={net_day:.3e} N m s/day, "
            f"wheel_dump_interval={dump_interval_hr:.2f} hr"
        )

    print("Environmental torque stats")
    print_torque_stats("Gravity gradient", gg_torque_hist)
    print_torque_stats("Drag", drag_torque_hist)
    print_torque_stats("Gravity gradient + drag", combined_torque_hist)
    print_environment_response_stats(env_solutions)

    st = StarTracker("ST-200", sigma_cross_arcsec=10.0, sigma_bore_arcsec=70.0, boresight_axis=2)

    M_ss_scale = np.array([
        [ 0.003,  0.001, -0.002],
        [-0.001, -0.005,  0.001],
        [ 0.002, -0.001,  0.004]
    ])
    b_ss_scale = np.array([0.001, -0.0005, 0.0008])
    ss = make_vector_sensor("FSS-100 Sun Sensor", M_ss_scale, b_ss_scale, sigma_deg=0.033)

    M_mag_scale = np.array([
        [ 0.008,  0.005, -0.007],
        [-0.004, -0.012,  0.006],
        [ 0.003, -0.005,  0.010]
    ])
    b_mag_scale = np.array([0.005, -0.003, 0.004])
    mag = make_vector_sensor("Magnetometer", M_mag_scale, b_mag_scale, sigma_deg=0.667)

    M_gyro_scale = np.array([
        [ 0.003,  0.001, -0.002],
        [-0.001, -0.004,  0.001],
        [ 0.002, -0.001,  0.005]
    ])
    gyro_b0_scale = np.radians(np.array([0.05, -0.03, 0.04]))
    gyro = make_gyro("BMI160 Gyro", M_gyro_scale, gyro_b0_scale, sigma_w_deg=0.007, sigma_beta_deg=0.0005)

    mag_eci = np.array([0.5, 0.5, 0.0])
    mag_eci /= np.linalg.norm(mag_eci)

    ss_errors = []
    mag_errors = []

    for i in range(len(t_sim)):
        q = sol_full[i, 0:4]
        R_mat = quaternion_to_matrix(q)

        sun_meas = ss.measure(R_mat.T @ sun_eci)
        mag_meas = mag.measure(R_mat.T @ mag_eci)

        ss_errors.append(np.degrees(np.arccos(np.clip(np.dot(sun_meas, R_mat.T @ sun_eci), -1, 1))))
        mag_errors.append(np.degrees(np.arccos(np.clip(np.dot(mag_meas, R_mat.T @ mag_eci), -1, 1))))

    for name, errs in [("Sun Sensor", ss_errors), ("Magnetometer", mag_errors)]:
        print(f"{name}: mean={np.mean(errs):.4f} deg, std={np.std(errs):.4f} deg")

    st_errors = []
    for i in range(len(t_sim)):
        q_true = sol_full[i, 0:4]
        q_meas = st.measure(q_true)
        R_true = quaternion_to_matrix(q_true)
        R_meas = quaternion_to_matrix(q_meas)
        cos_th = (np.trace(R_meas @ R_true.T) - 1.0) / 2.0
        st_errors.append(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))

    print(f"Star Tracker: mean={np.mean(st_errors):.6f} deg, std={np.std(st_errors):.6f} deg")
    print(f"Expected: cross-bore 1sig={10/3600:.6f} deg, bore 1sig={70/3600:.6f} deg")

    dt_sim = t_sim[1] - t_sim[0]
    gyro_errors = []
    gyro_bias_norms = []

    for i in range(len(t_sim)):
        omega_true = sol_full[i, 4:7]
        omega_meas = gyro.measure(omega_true, dt_sim)
        gyro_errors.append(np.degrees(np.linalg.norm(omega_meas - omega_true)))
        gyro_bias_norms.append(np.degrees(np.linalg.norm(gyro.bias)))

    print(f"Gyro: mean rate err={np.mean(gyro_errors):.4f} deg/s, std={np.std(gyro_errors):.4f} deg/s")
    print(f"Gyro bias norm: initial={gyro_bias_norms[0]:.4f} deg/s, final={gyro_bias_norms[-1]:.4f} deg/s")

    ss_hw2 = Sensor("Sun Sensor (HW2)", sigma_deg=0.033)
    mag_hw2 = Sensor("Magnetometer (HW2)", sigma_deg=0.667)

    num_trials = 1000

    sigma_ss_rad = np.radians(0.033)
    sigma_mag_rad = np.radians(0.667)
    weights = np.array([1.0 / sigma_ss_rad**2, 1.0 / sigma_mag_rad**2])
    weights /= np.sum(weights)

    r_eci = [sun_eci, mag_eci]

    errors_svd = []
    errors_q = []
    errors_triad = []

    time_svd = 0
    time_q = 0
    time_triad = 0

    def get_err(R_est, R_true):
        cos_theta = (np.trace(R_est @ R_true.T) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    for _ in range(num_trials):
        q_true_vec = np.random.randn(4)
        q_true_vec /= np.linalg.norm(q_true_vec)
        R_true = quaternion_to_matrix(q_true_vec)

        r_body_true = [R_true @ v for v in r_eci]
        r_body_meas = [ss_hw2.measure(r_body_true[0]), mag_hw2.measure(r_body_true[1])]

        start = time.perf_counter()
        R_svd = solve_wahba_svd(weights, r_body_meas, r_eci)
        time_svd += time.perf_counter() - start

        start = time.perf_counter()
        q_est = qmethod(weights, r_body_meas, r_eci)
        R_q = Q(q_est).T
        time_q += time.perf_counter() - start

        start = time.perf_counter()
        R_triad = triad(r_body_meas[0], r_body_meas[1], r_eci[0], r_eci[1])
        time_triad += time.perf_counter() - start

        errors_svd.append(get_err(R_svd, R_true))
        errors_q.append(get_err(R_q, R_true))
        errors_triad.append(get_err(R_triad, R_true))

    print(f"SVD: {np.mean(errors_svd):.4f} deg, {time_svd / num_trials * 1e6:.1f} us")
    print(f"q-method: {np.mean(errors_q):.4f} deg, {time_q / num_trials * 1e6:.1f} us")
    print(f"TRIAD: {np.mean(errors_triad):.4f} deg, {time_triad / num_trials * 1e6:.1f} us")

    wahba_mean_err = np.mean(errors_q)

    def run_mekf_trial(q_true_traj, omega_true_traj, t_filt, dt_filt, ss, mag, st, gyro_template, sun_eci, mag_eci, q0_est, beta0_est, P0, sigma_w, sigma_beta, W_ss_filt, W_mag_filt):
        N = len(t_filt)
        gyro_local = copy.deepcopy(gyro_template)
        filt = MEKF(q0_est, beta0_est, P0.copy(), sigma_w, sigma_beta)

        att_errors = np.zeros((N, 3))
        bias_errors = np.zeros((N, 3))
        P_hist = np.zeros((N, 6, 6))
        P_hist[0] = filt.P.copy()

        for k in range(N):
            q_true = q_true_traj[k]
            omega_true = omega_true_traj[k]

            Rot = quaternion_to_matrix(q_true)
            sun_body = Rot.T @ sun_eci
            mag_body = Rot.T @ mag_eci

            if k > 0:
                u_gyro = gyro_local.measure(omega_true, dt_filt)
                filt.predict(u_gyro, dt_filt)

                y_ss = ss.measure(sun_body)
                y_mag = mag.measure(mag_body)
                q_st = st.measure(q_true)

                filt.update_vector(y_ss, sun_eci, W_ss_filt)
                filt.update_vector(y_mag, mag_eci, W_mag_filt)
                filt.update_star_tracker(q_st, st.W_st)

                P_hist[k] = filt.P.copy()

            dq = L_q(filt.q).T @ q_true
            if dq[0] < 0:
                dq = -dq
            att_errors[k] = dq[1:4]
            bias_errors[k] = gyro_local.bias - filt.beta

        return att_errors, bias_errors, P_hist

    def generate_tumble_trajectory(omega0_body, q0_true, J_body, mu_val, r0_vec, v0_vec, t_high, t_filt):
        x0 = np.concatenate([q0_true, omega0_body, np.zeros(3), r0_vec, v0_vec])
        sol = rk4(full_dyn, x0, t_high, args=(J_body, mu_val))
        for i in range(len(t_high)):
            qn = sol[i, 0:4]
            sol[i, 0:4] = qn / np.linalg.norm(qn)

        idx = np.searchsorted(t_high, t_filt)
        idx = np.clip(idx, 0, len(t_high) - 1)
        q_traj = sol[idx, 0:4]
        omega_traj = sol[idx, 4:7]
        return q_traj, omega_traj

    def random_axis_angle_quat(angle_rad):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        return expq(axis * angle_rad)

    def random_uniform_quat():
        q = np.random.randn(4)
        return q / np.linalg.norm(q)

    ss_rms_rad = np.radians(np.sqrt(np.mean(np.array(ss_errors)**2)))
    mag_rms_rad = np.radians(np.sqrt(np.mean(np.array(mag_errors)**2)))
    W_ss_eff = ss_rms_rad**2 * np.eye(3)
    W_mag_eff = mag_rms_rad**2 * np.eye(3)
    sigma_w = gyro.sigma_w
    sigma_beta = gyro.sigma_beta
    print(f"MEKF effective W: ss_rms={np.degrees(ss_rms_rad):.4f} deg, mag_rms={np.degrees(mag_rms_rad):.4f} deg", flush=True)

    rw_torque_limit = 1.0e-4
    rw_momentum_limit = 1.5e-3
    q_hold = np.array([1.0, 0.0, 0.0, 0.0])
    Q_lqr = np.diag([500.0, 500.0, 500.0, 10.0, 10.0, 10.0])
    R_lqr = 1.0e8 * np.eye(3)
    t_lqr = np.arange(0.0, 3.0 * T_orbit + 0.5, 0.5)
    tvlqr_regulator = TVLQRAttitudeRegulator(
        I_body,
        q_hold,
        t_lqr,
        Q_lqr,
        R_lqr,
        Qf=20.0 * Q_lqr,
        torque_limit=rw_torque_limit,
    )

    def attitude_error_deg(q, q_des):
        dq = quat_error(q, q_des)
        return np.degrees(2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0)))

    def reaction_wheel_torque_command(tau_des, wheel_momentum, dt):
        tau_cmd = np.clip(tau_des, -rw_torque_limit, rw_torque_limit)
        hdot_cmd = -tau_cmd

        if dt > 0.0:
            h_next = wheel_momentum + hdot_cmd * dt
            for i in range(3):
                if abs(h_next[i]) > rw_momentum_limit:
                    h_lim = np.sign(h_next[i]) * rw_momentum_limit
                    hdot_cmd[i] = (h_lim - wheel_momentum[i]) / dt

        hdot_cmd = np.clip(hdot_cmd, -rw_torque_limit, rw_torque_limit)
        return -hdot_cmd

    def rk4_control_step(x, t_now, dt, tau_cmd, env):
        x_next = rk4(
            full_dyn_controlled,
            x,
            np.array([t_now, t_now + dt]),
            args=(I_body, mu, surfaces, env, tau_cmd),
        )[-1]
        x_next[0:4] /= np.linalg.norm(x_next[0:4])
        return x_next

    def simulate_attitude_regulation(q0_true, omega0_true, t_ctrl, env):
        x = np.concatenate((q0_true, omega0_true, np.zeros(3), r0, v0))
        torque_hist = np.zeros((len(t_ctrl), 3))
        att_err_hist = np.zeros(len(t_ctrl))
        estimate_err_hist = np.zeros(len(t_ctrl))
        state_hist = np.zeros((len(t_ctrl), len(x)))
        state_hist[0] = x

        gyro_local = make_gyro("BMI160 Gyro", M_gyro_scale, gyro_b0_scale, sigma_w_deg=0.007, sigma_beta_deg=0.0005)
        init_err = expq(0.5 * np.radians(2.0) * np.array([1.0, -0.4, 0.2]) / np.linalg.norm([1.0, -0.4, 0.2]))
        q0_est = L_q(q0_true) @ init_err
        q0_est /= np.linalg.norm(q0_est)
        P0_ctrl = np.zeros((6, 6))
        P0_ctrl[:3, :3] = np.radians(2.0)**2 * np.eye(3)
        P0_ctrl[3:, 3:] = np.radians(0.5)**2 * np.eye(3)
        filt = MEKF(q0_est, np.zeros(3), P0_ctrl, sigma_w, sigma_beta)

        for k in range(len(t_ctrl) - 1):
            dt = t_ctrl[k + 1] - t_ctrl[k]
            q_true = x[0:4]
            omega_true = x[4:7]

            gyro_meas = gyro_local.measure(omega_true, dt)
            q_ctrl = filt.q
            omega_ctrl = gyro_meas - filt.beta
            estimate_err_hist[k] = attitude_error_deg(filt.q, q_true)

            tau_des = tvlqr_regulator.torque(t_ctrl[k], q_ctrl, omega_ctrl)
            tau_cmd = reaction_wheel_torque_command(tau_des, x[7:10], dt)
            torque_hist[k] = tau_cmd
            att_err_hist[k] = attitude_error_deg(q_true, q_hold)
            x = rk4_control_step(x, t_ctrl[k], dt, tau_cmd, env)
            state_hist[k + 1] = x

            filt.predict(gyro_meas, dt)
            Rot = quaternion_to_matrix(x[0:4])
            sun_body = Rot.T @ sun_eci
            mag_body = Rot.T @ mag_eci
            filt.update_vector(ss.measure(sun_body), sun_eci, W_ss_eff)
            filt.update_vector(mag.measure(mag_body), mag_eci, W_mag_eff)
            filt.update_star_tracker(st.measure(x[0:4]), st.W_st)
            estimate_err_hist[k + 1] = attitude_error_deg(filt.q, x[0:4])

        att_err_hist[-1] = attitude_error_deg(x[0:4], q_hold)
        return state_hist, torque_hist, att_err_hist, estimate_err_hist

    print("TVLQR attitude regulation tests")
    t_reg = np.arange(0.0, 300.0 + 0.5, 0.5)
    rng = np.random.default_rng(16)
    reg_cases = {}
    for case_idx, err_deg in enumerate([15.0, 35.0, 60.0, 90.0]):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q0_reg = expq(0.5 * np.radians(err_deg) * axis)
        omega0_reg = np.radians(rng.uniform(-0.4, 0.4, size=3))
        state_h, torque_h, att_h, _ = simulate_attitude_regulation(
            q0_reg,
            omega0_reg,
            t_reg,
            {"gravity_gradient": False, "drag": False, "rho": rho_500km, "cd": cd_drag},
        )
        reg_cases[f"{err_deg:.0f} deg"] = {
            "att_err_deg": att_h,
            "omega": state_h[:, 4:7],
            "torque": torque_h,
            "wheel_momentum": state_h[:, 7:10],
        }
        settle_idx = np.where(att_h < 1.0)[0]
        settle_time = t_reg[settle_idx[0]] if len(settle_idx) > 0 else np.nan
        print(
            f"TVLQR IC {err_deg:.0f} deg: final_err={att_h[-1]:.3f} deg, "
            f"settle_1deg={settle_time:.1f}s, max_tau={np.max(np.abs(torque_h)):.2e} N m, "
            f"max_h_rw={np.max(np.abs(state_h[:, 7:10])):.2e} N m s"
        )
    plot_attitude_regulation(
        t_reg,
        reg_cases,
        "attitude_regulation_tvlqr_random_ics.png",
        torque_limit=rw_torque_limit,
        momentum_limit=rw_momentum_limit,
    )

    t_ctrl_orbit = np.arange(0.0, 3.0 * T_orbit + 1.0, 1.0)
    state_orbit, torque_orbit, att_orbit, est_orbit = simulate_attitude_regulation(
        q_hold.copy(),
        np.zeros(3),
        t_ctrl_orbit,
        {"gravity_gradient": True, "drag": True, "rho": rho_500km, "cd": cd_drag},
    )
    orbit_rms = np.sqrt(np.mean(att_orbit**2))
    steady_start_time = T_orbit
    steady_mask = t_ctrl_orbit >= steady_start_time
    steady_orbit_rms = np.sqrt(np.mean(att_orbit[steady_mask]**2))
    steady_mekf_rms = np.sqrt(np.mean(est_orbit[steady_mask]**2))
    print(
        f"TVLQR disturbed orbit: rms_pointing={orbit_rms:.4f} deg, "
        f"steady_state_rms_pointing={steady_orbit_rms:.4f} deg, "
        f"max_tau={np.max(np.abs(torque_orbit)):.2e} N m, "
        f"max_h_rw={np.max(np.abs(state_orbit[:, 7:10])):.2e} N m s, "
        f"mekf_rms={np.sqrt(np.mean(est_orbit**2)):.4f} deg, "
        f"steady_state_mekf_rms={steady_mekf_rms:.4f} deg"
    )
    plot_attitude_regulation_orbit(
        t_ctrl_orbit,
        T_orbit,
        att_orbit,
        torque_orbit,
        est_orbit,
        "attitude_regulation_tvlqr_orbit.png",
        wheel_momentum=state_orbit[:, 7:10],
        torque_limit=rw_torque_limit,
        momentum_limit=rw_momentum_limit,
    )

    print("Eigen-axis slew test")
    slew_time = 420.0
    t_slew = np.arange(0.0, slew_time + 120.0 + 1.0, 1.0)
    q_slew_0 = np.array([1.0, 0.0, 0.0, 0.0])
    q_slew_f = expq(0.5 * np.pi * np.array([1.0, 0.0, 0.0]))
    slew_nom = eigen_axis_slew_trajectory(q_slew_0, q_slew_f, t_slew, slew_time, I_body)

    nominal_torque = slew_nom["torque"]
    p_ref_traj = np.zeros_like(nominal_torque)
    for k in range(1, len(t_slew)):
        dt_k = t_slew[k] - t_slew[k - 1]
        p_ref_traj[k] = p_ref_traj[k - 1] - 0.5 * (nominal_torque[k] + nominal_torque[k - 1]) * dt_k

    slew_tracker = TVLQRAttitudeRegulator(
        I_body,
        q_slew_f,
        t_slew,
        Q_lqr,
        R_lqr,
        Qf=20.0 * Q_lqr,
        torque_limit=rw_torque_limit,
        omega_ref_traj=slew_nom["omega"],
        p_ref_traj=p_ref_traj,
    )

    def simulate_eigen_axis_slew():
        x = np.concatenate((q_slew_0, np.zeros(3), np.zeros(3), r0, v0))
        state_hist = np.zeros((len(t_slew), len(x)))
        torque_hist = np.zeros((len(t_slew), 3))
        tracking_err = np.zeros(len(t_slew))
        estimate_err = np.zeros(len(t_slew))
        state_hist[0] = x

        gyro_local = make_gyro("BMI160 Gyro", M_gyro_scale, gyro_b0_scale, sigma_w_deg=0.007, sigma_beta_deg=0.0005)
        init_err = expq(0.5 * np.radians(2.0) * np.array([-0.3, 0.7, 1.0]) / np.linalg.norm([-0.3, 0.7, 1.0]))
        q0_est = L_q(q_slew_0) @ init_err
        q0_est /= np.linalg.norm(q0_est)
        P0_slew = np.zeros((6, 6))
        P0_slew[:3, :3] = np.radians(2.0)**2 * np.eye(3)
        P0_slew[3:, 3:] = np.radians(0.5)**2 * np.eye(3)
        filt = MEKF(q0_est, np.zeros(3), P0_slew, sigma_w, sigma_beta)

        env_slew = {"gravity_gradient": True, "drag": True, "rho": rho_500km, "cd": cd_drag}
        for k in range(len(t_slew) - 1):
            dt = t_slew[k + 1] - t_slew[k]
            q_true = x[0:4]
            omega_true = x[4:7]
            q_ref = slew_nom["q"][k]
            omega_ref = slew_nom["omega"][k]
            tau_ff = slew_nom["torque"][k]

            gyro_meas = gyro_local.measure(omega_true, dt)
            q_ctrl = filt.q
            omega_ctrl = gyro_meas - filt.beta
            estimate_err[k] = attitude_error_deg(filt.q, q_true)

            x_err = attitude_error_state(q_ctrl, omega_ctrl, q_ref, omega_ref)
            tau_des = tau_ff - slew_tracker.gain(t_slew[k]) @ x_err
            tau_cmd = reaction_wheel_torque_command(tau_des, x[7:10], dt)
            torque_hist[k] = tau_cmd
            tracking_err[k] = attitude_error_deg(q_true, q_ref)
            x = rk4_control_step(x, t_slew[k], dt, tau_cmd, env_slew)
            state_hist[k + 1] = x

            filt.predict(gyro_meas, dt)
            Rot = quaternion_to_matrix(x[0:4])
            sun_body = Rot.T @ sun_eci
            mag_body = Rot.T @ mag_eci
            filt.update_vector(ss.measure(sun_body), sun_eci, W_ss_eff)
            filt.update_vector(mag.measure(mag_body), mag_eci, W_mag_eff)
            filt.update_star_tracker(st.measure(x[0:4]), st.W_st)
            estimate_err[k + 1] = attitude_error_deg(filt.q, x[0:4])

        tracking_err[-1] = attitude_error_deg(x[0:4], slew_nom["q"][-1])
        return state_hist, torque_hist, tracking_err, estimate_err

    slew_state, slew_torque, slew_tracking_err, slew_est_err = simulate_eigen_axis_slew()
    closed_loop_angle = np.array([attitude_error_deg(q_slew_0, q) for q in slew_state[:, 0:4]])
    closed_loop = {
        "angle": np.radians(closed_loop_angle),
        "q": slew_state[:, 0:4],
        "omega": slew_state[:, 4:7],
    }
    plot_eigen_axis_slew(
        t_slew,
        slew_nom,
        closed_loop,
        slew_tracking_err,
        slew_torque,
        "eigen_axis_slew_tvlqr.png",
        torque_limit=rw_torque_limit,
        wheel_momentum=slew_state[:, 7:10],
        momentum_limit=rw_momentum_limit,
    )

    plot_versine_profile(t_slew, slew_nom, "versine_profile.png", maneuver_time=slew_time)

    slew_final_err = attitude_error_deg(slew_state[-1, 0:4], q_slew_f)
    slew_rms_tracking = np.sqrt(np.mean(slew_tracking_err**2))
    print(
        f"Eigen-axis 180 deg slew: final_err={slew_final_err:.4f} deg, "
        f"rms_tracking={slew_rms_tracking:.4f} deg, "
        f"max_nominal_tau={np.max(np.abs(slew_nom['torque'])):.2e} N m, "
        f"max_commanded_tau={np.max(np.abs(slew_torque)):.2e} N m, "
        f"max_h_rw={np.max(np.abs(slew_state[:, 7:10])):.2e} N m s, "
        f"mekf_rms={np.sqrt(np.mean(slew_est_err**2)):.4f} deg"
    )

    regulator_to_final = TVLQRAttitudeRegulator(
        I_body,
        q_slew_f,
        t_slew,
        Q_lqr,
        R_lqr,
        Qf=20.0 * Q_lqr,
        torque_limit=rw_torque_limit,
    )
    x_reg_compare = np.concatenate((q_slew_0, np.zeros(3), np.zeros(3), r0, v0))
    reg_compare_err = np.zeros(len(t_slew))
    reg_compare_torque = np.zeros((len(t_slew), 3))
    reg_compare_h = np.zeros((len(t_slew), 3))
    reg_compare_h[0] = x_reg_compare[7:10]
    reg_compare_max_h = 0.0
    reg_gyro = make_gyro("BMI160 Gyro", M_gyro_scale, gyro_b0_scale, sigma_w_deg=0.007, sigma_beta_deg=0.0005)
    reg_init_err = expq(0.5 * np.radians(2.0) * np.array([0.8, -0.2, 0.5]) / np.linalg.norm([0.8, -0.2, 0.5]))
    reg_q0_est = L_q(q_slew_0) @ reg_init_err
    reg_q0_est /= np.linalg.norm(reg_q0_est)
    reg_P0 = np.zeros((6, 6))
    reg_P0[:3, :3] = np.radians(2.0)**2 * np.eye(3)
    reg_P0[3:, 3:] = np.radians(0.5)**2 * np.eye(3)
    reg_filt = MEKF(reg_q0_est, np.zeros(3), reg_P0, sigma_w, sigma_beta)
    env_compare = {"gravity_gradient": True, "drag": True, "rho": rho_500km, "cd": cd_drag}
    for k in range(len(t_slew) - 1):
        dt = t_slew[k + 1] - t_slew[k]
        q_true = x_reg_compare[0:4]
        omega_true = x_reg_compare[4:7]
        gyro_meas = reg_gyro.measure(omega_true, dt)
        tau_des = regulator_to_final.torque(t_slew[k], reg_filt.q, gyro_meas - reg_filt.beta)
        tau_cmd = reaction_wheel_torque_command(tau_des, x_reg_compare[7:10], dt)
        reg_compare_torque[k] = tau_cmd
        reg_compare_err[k] = attitude_error_deg(q_true, q_slew_f)
        x_reg_compare = rk4_control_step(x_reg_compare, t_slew[k], dt, tau_cmd, env_compare)
        reg_compare_h[k + 1] = x_reg_compare[7:10]
        reg_compare_max_h = max(reg_compare_max_h, np.max(np.abs(x_reg_compare[7:10])))
        reg_filt.predict(gyro_meas, dt)
        Rot = quaternion_to_matrix(x_reg_compare[0:4])
        sun_body = Rot.T @ sun_eci
        mag_body = Rot.T @ mag_eci
        reg_filt.update_vector(ss.measure(sun_body), sun_eci, W_ss_eff)
        reg_filt.update_vector(mag.measure(mag_body), mag_eci, W_mag_eff)
        reg_filt.update_star_tracker(st.measure(x_reg_compare[0:4]), st.W_st)
    reg_compare_err[-1] = attitude_error_deg(x_reg_compare[0:4], q_slew_f)
    reg_settle_idx = np.where(reg_compare_err < 1.0)[0]
    reg_settle_time = t_slew[reg_settle_idx[0]] if len(reg_settle_idx) > 0 else np.nan
    print(
        f"180 deg comparison: eigen_axis_maneuver_time={slew_time:.1f}s, "
        f"eigen_axis_max_tracking={np.max(slew_tracking_err):.4f} deg, "
        f"regulator_1deg={reg_settle_time:.1f}s, "
        f"regulator_final_err={reg_compare_err[-1]:.4f} deg, "
        f"regulator_max_tau={np.max(np.abs(reg_compare_torque)):.2e} N m, "
        f"regulator_max_h_rw={reg_compare_max_h:.2e} N m s"
    )

    slew_compare_err = np.array([attitude_error_deg(q, q_slew_f) for q in slew_state[:, 0:4]])
    plot_slew_vs_regulator(
        t_slew,
        slew={
            "att_err_deg": slew_compare_err,
            "torque": slew_torque,
            "wheel_momentum": slew_state[:, 7:10],
        },
        regulator={
            "att_err_deg": reg_compare_err,
            "torque": reg_compare_torque,
            "wheel_momentum": reg_compare_h,
        },
        basename="slew_vs_regulator_180deg.png",
        torque_limit=rw_torque_limit,
        momentum_limit=rw_momentum_limit,
    )

    T_sim_mekf = 120.0
    dt_high = 0.01
    t_high = np.arange(0, T_sim_mekf + dt_high, dt_high)

    if not run_mekf_studies:
        print("Skipping MEKF Monte Carlo validation and convergence study")
        return

    print("MEKF Monte Carlo validation")

    N_mc = 50
    dt_mc = 1.0 / 5.0
    t_mc = np.arange(0, T_sim_mekf + dt_mc, dt_mc)
    N_filt = len(t_mc)

    mc_att_err_norms = np.zeros((N_mc, N_filt))
    mc_ss_err_deg = np.zeros(N_mc)
    rep_att_err = rep_bias_err = rep_P = None

    for mc in range(N_mc):
        q0_true_mc = random_uniform_quat()
        omega_mag_mc = np.radians(np.random.uniform(1.0, 4.0))
        omega_dir_mc = np.random.randn(3)
        omega_dir_mc /= np.linalg.norm(omega_dir_mc)
        omega0_mc = omega_mag_mc * omega_dir_mc

        q_traj_mc, omega_traj_mc = generate_tumble_trajectory(omega0_mc, q0_true_mc, J_tilde, mu, r0, v0, t_high, t_mc)

        init_err_rad = np.radians(np.random.uniform(5.0, 30.0))
        dq_err = random_axis_angle_quat(init_err_rad)
        q0_est_mc = L_q(q0_true_mc) @ dq_err
        q0_est_mc /= np.linalg.norm(q0_est_mc)

        P0_mc = np.zeros((6, 6))
        P0_mc[:3, :3] = init_err_rad**2 * np.eye(3)
        P0_mc[3:, 3:] = np.radians(0.5)**2 * np.eye(3)

        ss_mc = make_vector_sensor("FSS-100 Sun Sensor", M_ss_scale, b_ss_scale, sigma_deg=0.033)
        mag_mc = make_vector_sensor("Magnetometer", M_mag_scale, b_mag_scale, sigma_deg=0.667)
        gyro_mc = make_gyro("BMI160 Gyro", M_gyro_scale, gyro_b0_scale, sigma_w_deg=0.007, sigma_beta_deg=0.0005)

        att_err, bias_err, P_h = run_mekf_trial(q_traj_mc, omega_traj_mc, t_mc, dt_mc, ss_mc, mag_mc, st, gyro_mc, sun_eci, mag_eci, q0_est_mc, np.zeros(3), P0_mc, sigma_w, sigma_beta, W_ss_eff, W_mag_eff)

        mc_att_err_norms[mc] = np.degrees(np.linalg.norm(att_err, axis=1))
        mc_ss_err_deg[mc] = np.mean(mc_att_err_norms[mc, N_filt // 2:])

        if mc == 0:
            rep_att_err = att_err
            rep_bias_err = bias_err
            rep_P = P_h

        print(f"{mc+1}/{N_mc}: init_err={np.degrees(init_err_rad):.1f} deg, |omega|={np.degrees(omega_mag_mc):.1f} deg/s, ss_err={mc_ss_err_deg[mc]:.4f} deg")

    print(f"SS error: median={np.median(mc_ss_err_deg):.4f}, mean={np.mean(mc_ss_err_deg):.4f}, 95th={np.percentile(mc_ss_err_deg, 95):.4f} deg")
    print(f"MEKF median={np.median(mc_ss_err_deg):.4f} deg, Wahba={wahba_mean_err:.4f} deg, improvement={wahba_mean_err / max(np.median(mc_ss_err_deg), 1e-12):.1f}x")

    plot_mc_attitude_errors(t_mc, mc_att_err_norms)
    plot_mekf_errors(t_mc, rep_att_err, rep_bias_err, rep_P)

    print("MEKF Convergence study")

    init_errors_deg = [5, 15, 30, 60]
    P_scales = [0.1, 1.0, 10.0]
    N_conv = 10
    T_conv = 30.0
    t_high_conv = np.arange(0, T_conv + dt_high, dt_high)
    dt_conv = 1.0 / 5.0
    t_conv = np.arange(0, T_conv + dt_conv, dt_conv)
    N_conv_filt = len(t_conv)

    conv_results = {}
    for err_deg in init_errors_deg:
        for p_scale in P_scales:
            traces = np.zeros((N_conv, N_conv_filt))
            c_times = []

            for trial in range(N_conv):
                q0_true_c = random_uniform_quat()
                omega_mag_c = np.radians(np.random.uniform(1.0, 4.0))
                omega_dir_c = np.random.randn(3)
                omega_dir_c /= np.linalg.norm(omega_dir_c)
                omega0_c = omega_mag_c * omega_dir_c

                q_traj_c, omega_traj_c = generate_tumble_trajectory(omega0_c, q0_true_c, J_tilde, mu, r0, v0, t_high_conv, t_conv)

                dq_c = random_axis_angle_quat(np.radians(err_deg))
                q0_est_c = L_q(q0_true_c) @ dq_c
                q0_est_c /= np.linalg.norm(q0_est_c)

                P0_c = np.zeros((6, 6))
                P0_c[:3, :3] = p_scale * np.radians(err_deg)**2 * np.eye(3)
                P0_c[3:, 3:] = np.radians(0.5)**2 * np.eye(3)

                ss_c = make_vector_sensor("FSS-100 Sun Sensor", M_ss_scale, b_ss_scale, sigma_deg=0.033)
                mag_c = make_vector_sensor("Magnetometer", M_mag_scale, b_mag_scale, sigma_deg=0.667)
                gyro_c = make_gyro("BMI160 Gyro", M_gyro_scale, gyro_b0_scale, sigma_w_deg=0.007, sigma_beta_deg=0.0005)

                att_e, _, _ = run_mekf_trial(q_traj_c, omega_traj_c, t_conv, dt_conv, ss_c, mag_c, st, gyro_c, sun_eci, mag_eci, q0_est_c, np.zeros(3), P0_c, sigma_w, sigma_beta, W_ss_eff, W_mag_eff)

                att_err_deg_c = np.degrees(np.linalg.norm(att_e, axis=1))
                traces[trial] = att_err_deg_c

                ss_part = att_err_deg_c[N_conv_filt // 2:]
                threshold = 2.0 * np.mean(ss_part)
                if np.any(att_err_deg_c < threshold):
                    c_times.append(t_conv[np.argmax(att_err_deg_c < threshold)])
                else:
                    c_times.append(T_conv)

            label = f"{err_deg}deg_P{p_scale}"
            conv_results[label] = {
                't': t_conv, 'traces': traces,
                'err_deg': err_deg, 'p_scale': p_scale,
                'conv_times': c_times,
            }

            print(f"init_err={err_deg} deg, P_scale={p_scale}: conv={np.median(c_times):.2f}s, final_err={np.median(traces[:, -1]):.4f} deg")

    plot_mc_convergence(conv_results)

if __name__ == "__main__":
    main()
