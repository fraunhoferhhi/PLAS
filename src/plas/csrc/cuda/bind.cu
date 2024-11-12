#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "plas.cuh"
#include <iostream>

namespace plas
{
    at::Tensor random_philox_permutation_torch_cuda_wrapper(
        const int64_t n,
        const int64_t num_rounds,
        at::Tensor &dummy)
    {
        int *permutation = random_philox_permutation_cuda(n, num_rounds);
        cudaDeviceSynchronize();
        std::function<void(void *)> deleter = [](void *ptr) {
            cudaFree(ptr);
        };
        return torch::from_blob(permutation, {n}, deleter, at::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
    }

    TORCH_LIBRARY_IMPL(plas, CUDA, m)
    {
        m.impl("random_philox_permutation", &random_philox_permutation_torch_cuda_wrapper);
    }
}
