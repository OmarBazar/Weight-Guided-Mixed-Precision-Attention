import torch
import os
import time
import call_cspmm
import call_dummy

import sys


# D = torch.load("one.pt")

# Q, k, V = D

# Q = Q[0]
# k = k[0]
# V = V[0]

# # Q = torch.randn_like(Q)
# # k = torch.randn_like(k)
# # V = torch.randn_like(V)

# # print(Q.dtype, Q.device)

# A = Q @ (torch.transpose(k, 1, 2))
# A = torch.nn.functional.softmax(A, dim = 0)

# N = Q.shape[1]
# M = Q.shape[2]
# K = 3
# H = Q.shape[0]

# print(N, M, K, H)


# A = A.to('cuda')
# V = V.to('cuda')

# FOR RANDOM INPUT
N = 2048
M = 400
K = 3
H = 6

A = torch.rand((H, N, N), device="cuda", dtype=torch.float32)
V = torch.rand((H, N, M), device="cuda", dtype=torch.float32)

# Profiling parameters
num_warmup = 10
num_trials = 500

def cspmm(A, V):
    N = A.shape[1]
    M = V.shape[2]
    K = 3
    H = A.shape[0]
    top_k_values, top_k_indices = torch.topk(A, K, dim=2)
    # print(top_k_values.shape)
    R_H = call_cspmm.launch_sparseDenseMult_cpp(top_k_indices, top_k_values, V, N, M, K, H)
    # A_H = torch.zeros_like(A)
    A_L = A
    A_L = A_L.scatter(2, top_k_indices, 0)
    # A_L = A - A_H
    # print(A.shape)
    # print(top_k_indices.shape)
    # print(top_k_indices)
    # depth_indices = torch.arange(A_L.shape[0]).view(-1, 1, 1)  # Shape [D, 1, 1]
    # row_indices = torch.arange(A_L.shape[1]).view(1, -1, 1)
    # A_L = A
    # A_L[:, :, top_k_indices] = 0
    R_L = torch.matmul(A_L.half(), V.half())
    R = R_H + R_L
    # R = R_H
    return R

def cspmm2(A, V):
    N = A.shape[1]
    M = V.shape[2]
    K = 3
    H = A.shape[0]
    top_k_values, top_k_indices = torch.topk(A, K, dim=2)
    # print(top_k_values.shape)
    R_H = call_dummy.launch_dummy_cpp(top_k_indices, top_k_values, V, N, M, K, H)
    # A_H = torch.zeros_like(A)
    A_L = A
    A_L = A_L.scatter(2, top_k_indices, 0)
    # A_L = A - A_H
    # print(A.shape)
    # print(top_k_indices.shape)
    # print(top_k_indices)
    # depth_indices = torch.arange(A_L.shape[0]).view(-1, 1, 1)  # Shape [D, 1, 1]
    # row_indices = torch.arange(A_L.shape[1]).view(1, -1, 1)
    # A_L = A
    # A_L[:, :, top_k_indices] = 0
    R_L = torch.matmul(A_L.half(), V.half())
    R = R_H + R_L
    # R = R_H
    return R

def profile_low_precision_mm():
    for _ in range(num_warmup):
        torch.matmul(A.half(), V.half())
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        start_time = time.time()
        torch.matmul(A.half(), V.half())
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    return sum(times) / len(times)

def profile_high_precision_mm():
    for _ in range(num_warmup):
        torch.matmul(A, V)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        start_time = time.time()
        torch.matmul(A, V)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    return sum(times) / len(times)

def profile_cspmm():
    for _ in range(num_warmup):
        R_sparse = cspmm(A, V)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        start_time = time.time()
        R_sparse = cspmm(A, V)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    return sum(times) / len(times)

def profile_dummy():
    for _ in range(num_warmup):
        R_sparse = cspmm2(A, V)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        start_time = time.time()
        R_sparse = cspmm2(A, V)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    return sum(times) / len(times)


# Run profiling
low_precision_mm_time = profile_low_precision_mm()
print("Done Low")
high_precision_mm_time = profile_high_precision_mm()
print("Done High")
cspmm_time = profile_cspmm()
print("Done Cspmm")
dummy_time = profile_dummy()
print("Done Dummy")

# Print results
print(f"Average Time for Low Precision Matrix Multiplication: {low_precision_mm_time:.6f} seconds")
print(f"Average Time for High Precision Matrix Multiplication: {high_precision_mm_time:.6f} seconds")
print(f"Average Time for sparseDenseMult: {cspmm_time:.6f} seconds")
print(f"Average Time for dummy: {dummy_time:.6f} seconds")

R_dense = torch.matmul(A, V)
R_sparse = cspmm(A, V)

R_equal = torch.isclose(R_sparse, R_dense, rtol=1e-02)
equal = True
for h in range(H):
    print(h)
    for n in range(N):
        for m in range(M):
            if R_equal[h][n][m]==False:
                print(h,n,m)
                print(R_sparse[h][n][m], R_dense[h][n][m])
                equal=False
                break
        if not equal:
            break
    if not equal:
        break
print("R_sparse = R_dense: ", equal)
