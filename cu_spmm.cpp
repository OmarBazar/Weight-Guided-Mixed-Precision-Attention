#include <torch/extension.h>
#include <pybind11/pybind11.h>

void launch_sparseDenseMult(torch::Tensor g_mat1cols, torch::Tensor g_mat1values, torch::Tensor g_mat2, int N, int M, int F, int K, int H, torch::Tensor g_out);

torch::Tensor launch_sparseDenseMult_cpp(torch::Tensor g_mat1cols, torch::Tensor g_mat1values, torch::Tensor g_mat2, int N, int M, int F, int K, int H) {
    // Ensure the tensors are on the GPU
    // TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same device");
    // TORCH_CHECK(a.device().is_cuda(), "Tensors must be on a CUDA device");

    // Allocate the output tensor
    auto g_out = torch::zeros({1, H, N, M}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    // Call the CUDA function to perform the addition
    launch_sparseDenseMult(g_mat1cols, g_mat1values, g_mat2, N, M, F, K, H, g_out);

    return g_out;
}


PYBIND11_MODULE(call_cspmm, m) {
    m.def("launch_sparseDenseMult_cpp", &launch_sparseDenseMult_cpp, "Sparse Matrix-Matrix Multiply (CUDA)");
}
