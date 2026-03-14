import numpy as np
from src.utils import rk4, box_inertia, parallel_axis_theorem, perturb_inertia, hat
from src.dynamics import orbit_dyn, attitude_dyn, full_dyn 
from src.spacecraft import Spacecraft
from src.plotting import plot_orbit, plot_attitude_stability, plot_momentum_sphere, plot_full_dyn
from src.sensors import Sensor
from scipy.spatial.transform import Rotation as R
from src.estimation import solve_wahba_svd, solve_wahba_q_method, triad
import time

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
    
    evals, evecs = np.linalg.eigh(I_body)
    I_principal = np.sort(evals)
    
    z_com = r_com[2]
    surfaces = {
        "+X": {"n": np.array([1, 0, 0]),  "A": w*h, "r_c": np.array([w/2, 0, h/2 - z_com])},
        "-X": {"n": np.array([-1, 0, 0]), "A": w*h, "r_c": np.array([-w/2, 0, h/2 - z_com])},
        "+Y": {"n": np.array([0, 1, 0]),  "A": d*h, "r_c": np.array([0, d/2, h/2 - z_com])},
        "-Y": {"n": np.array([0, -1, 0]), "A": d*h, "r_c": np.array([0, -d/2, h/2 - z_com])},
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

    sin2_theta_sep = (1/aalto.I_principal[1] - 1/aalto.I_principal[2]) / (1/aalto.I_principal[0] - 1/aalto.I_principal[2])
    theta_sep = np.arcsin(np.sqrt(sin2_theta_sep))

    starts = [
        (0.1, 0), (0.2, 0), (0.4, 0), (0.6, 0),       
        (np.pi - 0.1, 0), (np.pi - 0.4, 0),           
        (1.1, 0), (1.3, 0), (1.5, 0),                
        (theta_sep, 0.005), (theta_sep, -0.005),      
        (np.pi/2, np.pi/2),                          
    ]

    sphere_trajectories = []
    t_sphere = np.linspace(0, 200, 2000)
    for theta, phi in starts:
        h0_vec = np.array([
            h_mag * np.sin(theta) * np.cos(phi),
            h_mag * np.sin(theta) * np.sin(phi),
            h_mag * np.cos(theta)
        ])
        w0_vec = np.linalg.inv(I_p_mat) @ h0_vec
        w_sol = rk4(attitude_dyn, w0_vec, t_sphere, args=(I_p_mat,))
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
        sun_body = R_mat @ sun_eci
        cos_theta = np.dot(sun_body, panel_body)
        errors.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    plot_full_dyn(t_sim, sol_full, np.array(errors))

    st_sigma = 10/3600
    st = Sensor("Star Tracker", st_sigma)

    ss_sigma = 0.1 / 3.0
    ss = Sensor("Fine Sun Sensor", ss_sigma)

    mag_sigma = 2.0 / 3.0
    mag = Sensor("Magnetometer", mag_sigma)
    
    star_eci = np.array([0, 0, 1])
    
    mag_eci = np.array([0.5, 0.5, 0])
    mag_eci /= np.linalg.norm(mag_eci)

    st_errors = []
    sun_errors = []
    mag_errors = []

    for i in range(len(t_sim)):
        q = sol_full[i, 0:4]
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        
        star_body_true = R_mat @ star_eci
        sun_body_true = R_mat @ sun_eci
        mag_body_true = R_mat @ mag_eci

        st_meas = st.measure(star_body_true)
        sun_meas = ss.measure(sun_body_true)
        mag_meas = mag.measure(mag_body_true)

        st_err = np.degrees(np.arccos(np.clip(np.dot(st_meas, star_body_true), -1, 1)))
        sun_err = np.degrees(np.arccos(np.clip(np.dot(sun_meas, sun_body_true), -1, 1)))
        mag_err = np.degrees(np.arccos(np.clip(np.dot(mag_meas, mag_body_true), -1, 1)))

        st_errors.append(st_err)
        sun_errors.append(sun_err)
        mag_errors.append(mag_err)

    for s_name, errs, s_obj in zip(["Star Tracker", "Sun Sensor", "Magnetometer"], 
                                   [st_errors, sun_errors, mag_errors], 
                                   [st, ss, mag]):
        mean_err = np.mean(errs)
        std_err = np.std(errs)
        expected_std = np.degrees(s_obj.sigma_rad)
        print(f"{s_name}:")
        print(f"observed mean angular error: {mean_err}")
        print(f"observed std angular error: {std_err}")
        print(f"input sigma: {expected_std}")

        
    num_trials = 1000
    
    weights = np.array([1.0/st.sigma_rad**2, 1.0/ss.sigma_rad**2, 1.0/mag.sigma_rad**2])
    weights /= np.sum(weights) 
    
    r_eci = [star_eci, sun_eci, mag_eci]
    
    errors_svd = []
    errors_q = []
    errors_triad = []
    
    time_svd = 0
    time_q = 0
    time_triad = 0
    
    for _ in range(num_trials):
        q_true_vec = np.random.randn(4)
        q_true_vec /= np.linalg.norm(q_true_vec)
        R_true = R.from_quat([q_true_vec[1], q_true_vec[2], q_true_vec[3], q_true_vec[0]]).as_matrix()
        
        r_body_true = [R_true @ v for v in r_eci]
        
        r_body_meas = [st.measure(r_body_true[0]),ss.measure(r_body_true[1]),mag.measure(r_body_true[2])]
        
        start = time.perf_counter()
        R_svd = solve_wahba_svd(weights, r_body_meas, r_eci)
        time_svd += time.perf_counter() - start
        
        start = time.perf_counter()
        q_est = solve_wahba_q_method(weights, r_body_meas, r_eci)
        R_q = R.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]]).as_matrix()
        time_q += time.perf_counter() - start

        start = time.perf_counter()
        R_triad = triad(r_body_meas[0], r_body_meas[1], r_eci[0], r_eci[1])
        time_triad += time.perf_counter() - start
        
        def get_err(R_est, R_true):
            R_err = R_est @ R_true.T
            cos_theta = (np.trace(R_err) - 1.0) / 2.0
            return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        errors_svd.append(get_err(R_svd, R_true))
        errors_q.append(get_err(R_q, R_true))
        errors_triad.append(get_err(R_triad, R_true))

    print(f"SVD")
    print(f"mean angular error: {np.mean(errors_svd)}")
    print(f"avg compute time: {time_svd/num_trials*1e6}")
    
    print(f"q-Method:")
    print(f"mean angular error: {np.mean(errors_q)}")
    print(f"avg compute error: {time_q/num_trials}")

    print(f"TRIAD (ST + Sun):")
    print(f"  Mean Angular Error: {np.mean(errors_triad)}")
    print(f"  Avg Compute Time:   {time_triad/num_trials*1e6}")

if __name__ == "__main__":
    main()
