#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sparseDenseMult(long* g_mat1cols, float* g_mat1values, float* g_mat2, int N, int M, int K, int H, float* g_out) {

    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    extern __shared__ unsigned char sharedMem[];

    long *A_cols = (long*)sharedMem;
    float *A_values = (float*)(sharedMem + K * H * sizeof(long));

    if(threadIdx.x < K*H){
        int head = threadIdx.x / K;
        A_cols[threadIdx.x] = g_mat1cols[N * K * head + row * K + threadIdx.x % K];
        A_values[threadIdx.x] = g_mat1values[N * K * head + row * K + threadIdx.x % K];
    }

    __syncthreads();
    for (int head = 0; head < H; head++){
        col = threadIdx.x;
        for (int col = threadIdx.x; col < M; col += blockDim.x) {
            if (col < M) {
                float sum = 0;
                for (int k = 0; k < K; k++){
                    sum += A_values[head * K + k] * g_mat2[N * M * head + A_cols[head * K + k]*M + col];
                }
                g_out[N * M * head + row * M + col] = sum;
            }
        }
    }
}

__global__ void dummy(long* g_mat1cols, float* g_mat1values, float* g_mat2, int N, int M, int K, int H, float* g_out) {

    unsigned int row = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int col = threadIdx.x;
    
    extern __shared__ unsigned char sharedMem[];

    long *A_cols = (long*)sharedMem;
    float *A_values = (float*)(sharedMem + K * H * sizeof(long));


    if(threadIdx.x < K*H){
        int head = threadIdx.x / K;
        A_cols[threadIdx.x] = g_mat1cols[N * K * head + row * K + threadIdx.x % K];
        A_values[threadIdx.x] = g_mat1values[N * K * head + row * K + threadIdx.x % K];
    }

    __syncthreads();
    col = threadIdx.x;
    for (int col = threadIdx.x; col < M; col += blockDim.x) {
        if (col < M) {
            float sum = 0;
            for (int k = 0; k < K; k++){
                sum += A_values[head * K + k] * g_mat2[N * M * head + A_cols[head * K + k]*M + col];
            }
            g_out[N * M * head + row * M + col] = sum;
        }
    }
}

void launch_sparseDenseMult(torch::Tensor g_mat1cols, torch::Tensor g_mat1values, torch::Tensor g_mat2, int N, int M, int K, int H, torch::Tensor g_out){
    int threads = 64;
    int blocks = N;
    sparseDenseMult<<<blocks, threads>>>(g_mat1cols.data_ptr<long>(), g_mat1values.data_ptr<float>(), g_mat2.data_ptr<float>(), N, M, K, H, g_out.data_ptr<float>());
}

void launch_dummy(torch::Tensor g_mat1cols, torch::Tensor g_mat1values, torch::Tensor g_mat2, int N, int M, int K, int H, torch::Tensor g_out){
    int threads = 64;
    dim3 blocks = (N, H);
    dummy<<<blocks, threads>>>(g_mat1cols.data_ptr<long>(), g_mat1values.data_ptr<float>(), g_mat2.data_ptr<float>(), N, M, K, H, g_out.data_ptr<float>());
}

