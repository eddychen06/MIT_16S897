import numpy as np
from src.utils import hat

H = np.vstack([np.zeros((1, 3)), np.eye(3)])
T = np.diag([1.0, -1.0, -1.0, -1.0])


def L(q):
    s = q[0]
    v = q[1:4]
    M = np.empty((4, 4))
    M[0, 0] = s
    M[0, 1:] = -v
    M[1:, 0] = v
    M[1:, 1:] = s * np.eye(3) + hat(v)
    return M


def R(q):
    s = q[0]
    v = q[1:4]
    M = np.empty((4, 4))
    M[0, 0] = s
    M[0, 1:] = -v
    M[1:, 0] = v
    M[1:, 1:] = s * np.eye(3) - hat(v)
    return M


def Q(q):
    return H.T @ R(q).T @ L(q) @ H


def solve_wahba_svd(weights, body_vectors, inertial_vectors):
    B = np.zeros((3, 3))
    for w, r_B, r_N in zip(weights, body_vectors, inertial_vectors):
        B += w * np.outer(r_B, r_N)
    
    U, S, Vt = np.linalg.svd(B)
    
    d = np.linalg.det(U) * np.linalg.det(Vt)
    M = np.diag([1, 1, d])
    
    R_opt = U @ M @ Vt
    return R_opt


def qmethod(weights, body_vectors, inertial_vectors):
    D = np.zeros((4, 4))
    for w, r_B, r_N in zip(weights, body_vectors, inertial_vectors):
        D += w * L(H @ r_N).T @ R(H @ r_B)

    eigvals, eigvecs = np.linalg.eigh(D)
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
