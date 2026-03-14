import numpy as np

def solve_wahba_svd(weights, body_vectors, inertial_vectors):
    B = np.zeros((3, 3))
    for w, r_B, r_N in zip(weights, body_vectors, inertial_vectors):
        B += w * np.outer(r_B, r_N)
    
    U, S, Vt = np.linalg.svd(B)
    
    d = np.linalg.det(U) * np.linalg.det(Vt)
    M = np.diag([1, 1, d])
    
    R = U @ M @ Vt
    return R
    
def solve_wahba_q_method(weights, body_vectors, inertial_vectors):
    B = np.zeros((3, 3))
    for w, r_B, r_N in zip(weights, body_vectors, inertial_vectors):
        B += w * np.outer(r_B, r_N)
    
    S = B + B.T
    z = np.array([B[2, 1] - B[1, 2],
                  B[0, 2] - B[2, 0],
                  B[1, 0] - B[0, 1]])
    sigma = np.trace(B)
    
    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = z
    K[1:, 0] = z
    K[1:, 1:] = S - np.eye(3) * sigma
    
    eigvals, eigvecs = np.linalg.eigh(K)
    q_opt = eigvecs[:, np.argmax(eigvals)]
    
    return q_opt

def triad(r1_B, r2_B, r1_N, r2_N):
    t1_B = r1_B / np.linalg.norm(r1_B)
    t2_B = np.cross(t1_B, r2_B) 
    t2_B /= np.linalg.norm(t2_B)
    t3_B = np.cross(t1_B, t2_B)
    
    t1_N = r1_N / np.linalg.norm(r1_N)
    t2_N = np.cross(t1_N, r2_N)
    t2_N /= np.linalg.norm(t2_N)
    t3_N = np.cross(t1_N, t2_N)
    
    T_B = np.column_stack((t1_B, t2_B, t3_B))
    T_N = np.column_stack((t1_N, t2_N, t3_N))
    
    return T_B @ T_N.T
