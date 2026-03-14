import numpy as np
from scipy.linalg import expm

def hat(v):
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],  0.0,   -v[0]],
        [-v[1], v[0],   0.0]
    ])

def rk4(func, y0, t_eval, args=()):
    N = len(t_eval)
    Y = np.zeros((N, len(y0)))
    Y[0] = y0

    for i in range(N - 1):
        t = t_eval[i]
        dt = t_eval[i+1] - t_eval[i]
        y_curr = Y[i]

        k1 = func(t, y_curr, *args)
        k2 = func(t + 0.5 * dt, y_curr + 0.5 * dt * k1, *args)
        k3 = func(t + 0.5 * dt, y_curr + 0.5 * dt * k2, *args)
        k4 = func(t + dt, y_curr + dt * k3, *args)
        
        Y[i+1] = y_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return Y

def box_inertia(m, dx, dy, dz):
    return (m / 12.0) * np.diag([dy**2 + dz**2, dx**2 + dz**2, dx**2 + dy**2])

def parallel_axis_theorem(I_cm, m, r_offset):
    r_sq = np.dot(r_offset, r_offset)
    outer_product = np.outer(r_offset, r_offset)
    return I_cm + m * (r_sq * np.eye(3) - outer_product)

def perturb_inertia(J): 
    D_vals, V = np.linalg.eigh(J)
    D = np.diag(D_vals)

    d = np.random.normal(0, 0.05, 3)
    v = np.random.normal(0, np.radians(5), 3)   
    
    D_tilde = D @ (np.eye(3) + np.diag(d))
    V_tilde = V @ expm(hat(v))
    J_tilde = V_tilde @ D_tilde @ V_tilde.T 

    return J_tilde