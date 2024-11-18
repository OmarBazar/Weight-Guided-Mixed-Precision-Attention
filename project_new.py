import torch
import os
import time
import call_cspmm
import call_dummy

import sys


# D = torch.load("one.pt")

# Q, k, V = D

# Q = Q
# k = k
# V = V

# # # Q = torch.randn_like(Q)
# # # k = torch.randn_like(k)
# # # V = torch.randn_like(V)

# # # print(Q.dtype, Q.device)

# A = Q @ (torch.transpose(k, 2, 3))
# A = torch.nn.functional.softmax(A, dim = -1)

# N = Q.shape[2]
# M = Q.shape[3]
# K = 3
# H = Q.shape[1]

# # print(N, M, K, H)


# A = A.to('cuda')
# V = V.to('cuda')

# # FOR RANDOM INPUT




# # # # Profiling parameters
# num_warmup = 10
# num_trials = 5

# A = torch.load('tensorw.pt').to('cuda')
# V = torch.load('tensorv.pt').to('cuda')

# N = A.shape[2]
# M = V.shape[3]
# K = 3
# H = A.shape[1]

def cspmm(A, V):
    N = A.shape[2]
    M = V.shape[3]
    F = V.shape[2]
    K = 3
    H = A.shape[1]
    top_k_values, top_k_indices = torch.topk(A, K, dim=3)
    R_H = call_cspmm.launch_sparseDenseMult_cpp(top_k_indices, top_k_values, V, N, M, F, K, H)
    A_L = A.scatter(3, top_k_indices, 0)
    R_L = A_L.half() @ V.half()
    R = R_H + R_L
    return R

def cspmm2(A, V):
    N = A.shape[2]
    M = V.shape[3]
    F = V.shape[2]
    K = 3
    H = A.shape[1]
    top_k_values, top_k_indices = torch.topk(A, K, dim=3)
    R_H = call_dummy.launch_dummy_cpp(top_k_indices, top_k_values, V, N, M, F, K, H)
    # print("R_H", R_H)
    A_L = A.scatter(3, top_k_indices, 0)
    R_L = torch.matmul(A_L.half(), V.half())
    R = R_H + R_L
    return R

# def profile_low_precision_mm():
#     for _ in range(num_warmup):
#         torch.matmul(A.half(), V.half())
#     torch.cuda.synchronize()

#     times = []
#     for _ in range(num_trials):
#         start_time = time.time()
#         torch.matmul(A.half(), V.half())
#         torch.cuda.synchronize()
#         times.append(time.time() - start_time)

#     return sum(times) / len(times)

# def profile_high_precision_mm():
#     for _ in range(num_warmup):
#         torch.matmul(A, V)
#     torch.cuda.synchronize()

#     times = []
#     for _ in range(num_trials):
#         start_time = time.time()
#         torch.matmul(A, V)
#         torch.cuda.synchronize()
#         times.append(time.time() - start_time)

#     return sum(times) / len(times)

# def profile_cspmm():
#     for _ in range(num_warmup):
#         cspmm(A, V)
#     torch.cuda.synchronize()

#     times = []
#     for _ in range(num_trials):
#         start_time = time.time()
#         cspmm(A, V)
#         torch.cuda.synchronize()
#         times.append(time.time() - start_time)

#     return sum(times) / len(times)

# def profile_dummy():
#     for _ in range(num_warmup):
#         cspmm2(A, V)
#     torch.cuda.synchronize()

#     times = []
#     for _ in range(num_trials):
#         start_time = time.time()
#         cspmm2(A, V)
#         torch.cuda.synchronize()
#         times.append(time.time() - start_time)

#     return sum(times) / len(times)


# # Run profiling
# low_precision_mm_time = profile_low_precision_mm()
# print("Done Low")
# high_precision_mm_time = profile_high_precision_mm()
# print("Done High")
# cspmm_time = profile_cspmm()
# print("Done Cspmm")
# # dummy_time = profile_dummy()
# # print("Done Dummy")

# # Print results
# print(f"Average Time for Low Precision Matrix Multiplication: {low_precision_mm_time:.6f} seconds")
# print(f"Average Time for High Precision Matrix Multiplication: {high_precision_mm_time:.6f} seconds")
# print(f"Average Time for sparseDenseMult: {cspmm_time:.6f} seconds")
# # print(f"Average Time for dummy: {dummy_time:.6f} seconds")
# N = 16
# M = 4
# K = 3
# H = 4
# F = N

# for i in range(1):
#     A = torch.rand((1, H, N, F), device="cuda", dtype=torch.float32)
#     # A = torch.nn.functional.softmax(A, dim = -1)
#     # print(A.shape)
#     V = torch.ones((1, H, F, M), device="cuda", dtype=torch.float32) #- torch.ones((1, H, F, M), device="cuda", dtype=torch.float32)/2
#     # V = torch.zeros((1, H, F, M), device="cuda", dtype=torch.float32)
#     # A = torch.zeros((1, H, N, F), device="cuda", dtype=torch.float32)

#     # print(V)

#     # A = torch.load('tensorw.pt').to('cuda')
#     # V = torch.load('tensorv.pt').to('cuda')
#     # print(A)
#     # print(V)
#     R_dense = A @ V
#     R_sparse = cspmm2(A, V)

#     # print(A)
#     # print(V)

#     # print(R_dense)
#     print(R_sparse.shape)
#     # print(A[0, 0, 1, :], V[0, 0, :, 0])
#     # print(R_dense[0, 0, 1, 0])

#     R_equal = torch.isclose(R_sparse, R_dense, atol=1e-01)
#     equal = torch.all(R_equal)
#     equal = True
#     for h in range(H):
#         for n in range(N):
#             for m in range(M):
#                 if R_equal[0][h][n][m]==False:
#                     print(h,n,m)
#                     print(R_sparse[0][h][n][m], R_dense[0][h][n][m])
#                     equal=False
#                     break
#             if not equal:
#                 break
#         if not equal:
#             break
#     print("R_sparse = R_dense: ", equal)
