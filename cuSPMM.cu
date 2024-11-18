#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sparseDenseMult(long* g_mat1cols, float* g_mat1values, float* g_mat2, int N, int M, int F, int K, int H, float* g_out) {

    unsigned int row = blockIdx.x;

    extern __shared__ unsigned char sharedMem[];

    long *A_cols = (long*)sharedMem;
    float *A_values = (float*)(sharedMem + K * H * sizeof(long));

    if(threadIdx.x < K*H){
        int h = threadIdx.x / K;
        A_cols[threadIdx.x] = g_mat1cols[N * K * h + row * K + threadIdx.x % K];
        A_values[threadIdx.x] = g_mat1values[N * K * h + row * K + threadIdx.x % K];
        // printf("A_cols[%d] = g_mat1cols[%d][%d][%d]: %ld\n", threadIdx.x, h , row , threadIdx.x % K, g_mat1cols[N * K * h + row * K + threadIdx.x % K]);
        // printf("threadIdx.x: %d and h: %d and row: %d and final index: %d and g_mat1values %f and A_values: %f and col: %ld and g_mat1cols: %ld and A_cols: %ld\n", threadIdx.x, h, row, N * K * h + row * K + threadIdx.x % K, g_mat1values[N * K * h + row * K + threadIdx.x % K], A_values[threadIdx.x], threadIdx.x % K, g_mat1cols[N * K * h + row * K + threadIdx.x % K], A_cols[threadIdx.x]);
    }

    __syncthreads();
    for (int head = 0; head < H; head++){
        for (int col = threadIdx.x; col < M; col += blockDim.x) {
            float sum = 0;
            for (int k = 0; k < K; k++){
                sum += A_values[head * K + k] * g_mat2[F * M * head + A_cols[head * K + k]*M + col];
                // if (col<2){
                //     printf("threadIdx.x: %d and head: %d and row: %d and col: %d and A_cols index: %d and g_mat2: %f and A_values: %f and gmat2 final index: %ld and A_cols: %ld\n", threadIdx.x, head, row, col, head * K + k, g_mat2[N * M * head + A_cols[head * K + k]*M + col], A_values[head * K + k], N * M * head + A_cols[head * K + k]*M + col, A_cols[head * K + k]);
                // }
            }
            g_out[N * M * head + row * M + col] = sum;
        }
    }
}

__global__ void dummy(long* g_mat1cols, float* g_mat1values, float* g_mat2, int N, int M, int F, int K, int H, float* g_out) {

    unsigned int row = blockIdx.x;
    unsigned int head = blockIdx.y;
    // printf("row: %d, head %d, test %d\n", row, head, test);
    
    extern __shared__ unsigned char sharedMem[];

    long *A_cols = (long*)sharedMem;
    float *A_values = (float*)(sharedMem + K * sizeof(long));


    if(threadIdx.x < K){
        A_cols[threadIdx.x] = g_mat1cols[N * K * head + row * K + threadIdx.x];
        A_values[threadIdx.x] = g_mat1values[N * K * head + row * K + threadIdx.x];
    }

    __syncthreads();
    for (int col = threadIdx.x; col < M; col += blockDim.x) {
        if (col < M) {
            float sum = 0;
            for (int k = 0; k < K; k++){
                sum += A_values[k] * g_mat2[F * M * head + A_cols[k]*M + col];
            }
            g_out[N * M * head + row * M + col] = sum;
        }
    }
}

void launch_sparseDenseMult(torch::Tensor g_mat1cols, torch::Tensor g_mat1values, torch::Tensor g_mat2, int N, int M, int F, int K, int H, torch::Tensor g_out){
    int threads = 64;
    int blocks = N;
    sparseDenseMult<<<blocks, threads>>>(g_mat1cols.data_ptr<long>(), g_mat1values.data_ptr<float>(), g_mat2.data_ptr<float>(), N, M, F, K, H, g_out.data_ptr<float>());
}

void launch_dummy(torch::Tensor g_mat1cols, torch::Tensor g_mat1values, torch::Tensor g_mat2, int N, int M, int F, int K, int H, torch::Tensor g_out){
    int threads = 64;
    dim3 blocks(N, H);
    dummy<<<blocks, threads>>>(g_mat1cols.data_ptr<long>(), g_mat1values.data_ptr<float>(), g_mat2.data_ptr<float>(), N, M, F, K, H, g_out.data_ptr<float>());
}

