import numpy as np
from src.utils import rk4, box_inertia, parallel_axis_theorem, perturb_inertia, hat
from src.dynamics import orbit_dyn, attitude_dyn, full_dyn 
from src.spacecraft import Spacecraft
from src.plotting import plot_orbit, plot_attitude_stability, plot_momentum_sphere, plot_full_dyn, plot_mekf_errors, plot_mc_attitude_errors, plot_mc_convergence
from src.sensors import Sensor, VectorSensor, StarTracker, Gyroscope, expq
from scipy.spatial.transform import Rotation as R
from src.estimation import solve_wahba_svd, qmethod, triad, Q, L as L_q
from src.mekf import MEKF
import time
import copy

def main():
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
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        sun_body = R_mat.T @ sun_eci
        cos_theta = np.dot(sun_body, panel_body)
        errors.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    plot_full_dyn(t_sim, sol_full, np.array(errors))

    st = StarTracker("ST-200", sigma_cross_arcsec=10.0, sigma_bore_arcsec=70.0, boresight_axis=2)

    M_ss = np.eye(3) + np.array([
        [ 0.003,  0.001, -0.002],
        [-0.001, -0.005,  0.001],
        [ 0.002, -0.001,  0.004]
    ])
    b_ss = np.array([0.001, -0.0005, 0.0008])
    ss = VectorSensor("FSS-100 Sun Sensor", M_ss, b_ss, sigma_deg=0.033)

    M_mag = np.eye(3) + np.array([
        [ 0.008,  0.005, -0.007],
        [-0.004, -0.012,  0.006],
        [ 0.003, -0.005,  0.010]
    ])
    b_mag = np.array([0.005, -0.003, 0.004])
    mag = VectorSensor("Magnetometer", M_mag, b_mag, sigma_deg=0.667)

    M_gyro = np.eye(3) + np.array([
        [ 0.003,  0.001, -0.002],
        [-0.001, -0.004,  0.001],
        [ 0.002, -0.001,  0.005]
    ])
    gyro = Gyroscope("BMI160 Gyro", M_gyro, sigma_w_deg=0.007, sigma_beta_deg=0.0005, b0=np.radians(np.array([0.05, -0.03, 0.04])))

    mag_eci = np.array([0.5, 0.5, 0.0])
    mag_eci /= np.linalg.norm(mag_eci)

    ss_errors = []
    mag_errors = []

    for i in range(len(t_sim)):
        q = sol_full[i, 0:4]
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

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
        R_true = R.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]]).as_matrix()
        R_meas = R.from_quat([q_meas[1], q_meas[2], q_meas[3], q_meas[0]]).as_matrix()
        cos_th = (np.trace(R_meas @ R_true.T) - 1.0) / 2.0
        st_errors.append(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))

    print(f"Star Tracker: mean={np.mean(st_errors):.6f} deg, std={np.std(st_errors):.6f} deg")
    print(f"Expected: cross-bore 1sig={10/3600:.6f} deg, bore 1sig={70/3600:.6f} deg")

    gyro.reset(b0=np.radians(np.array([0.05, -0.03, 0.04])))
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
        R_true = R.from_quat([q_true_vec[1], q_true_vec[2], q_true_vec[3], q_true_vec[0]]).as_matrix()

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

            Rot = R.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]]).as_matrix()
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
    print(f"MEKF effective W: ss_rms={np.degrees(ss_rms_rad):.4f} deg, mag_rms={np.degrees(mag_rms_rad):.4f} deg", flush=True)

    T_sim_mekf = 120.0
    dt_high = 0.01
    t_high = np.arange(0, T_sim_mekf + dt_high, dt_high)

    sigma_w = gyro.sigma_w
    sigma_beta = gyro.sigma_beta

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

        gyro_mc = Gyroscope("BMI160 Gyro", M_gyro, sigma_w_deg=0.007, sigma_beta_deg=0.0005, b0=np.radians(np.array([0.05, -0.03, 0.04])))

        att_err, bias_err, P_h = run_mekf_trial(q_traj_mc, omega_traj_mc, t_mc, dt_mc, ss, mag, st, gyro_mc, sun_eci, mag_eci, q0_est_mc, np.zeros(3), P0_mc, sigma_w, sigma_beta, W_ss_eff, W_mag_eff)

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

                gyro_c = Gyroscope("BMI160 Gyro", M_gyro, sigma_w_deg=0.007, sigma_beta_deg=0.0005, b0=np.radians(np.array([0.05, -0.03, 0.04])))

                att_e, _, _ = run_mekf_trial(q_traj_c, omega_traj_c, t_conv, dt_conv, ss, mag, st, gyro_c, sun_eci, mag_eci, q0_est_c, np.zeros(3), P0_c, sigma_w, sigma_beta, W_ss_eff, W_mag_eff)

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

    print("Done")


if __name__ == "__main__":
    main()
