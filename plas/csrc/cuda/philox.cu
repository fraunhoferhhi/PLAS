#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include "stream_compaction.cuh"

#include <vector>
#include <random>
#include <limits>
#include <iostream>

namespace plas {

static const int64_t M0 = UINT64_C(0xD2B74407B1CE6E93);

__device__ int32_t mulhilo(int64_t a, int32_t b, int32_t &hip)
{
  int64_t product = a * int64_t(b);
  hip = product >> 32;
  return int32_t(product);
}

__global__ void random_philox_bijection(
    const int64_t n,
    const int64_t num_rounds,
    const int64_t right_side_bits,
    const int64_t left_side_bits,
    const int64_t right_side_mask,
    const int64_t left_side_mask,
    const int32_t *keys,
    int64_t *output)
{
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
  {
    return;
  }
  int32_t state[2] = {int32_t(i >> right_side_bits), int32_t(i & right_side_mask)};
  for (int i = 0; i < num_rounds; i++)
  {
    int32_t hi;
    int32_t lo = mulhilo(M0, state[0], hi);
    lo = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
    state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
    state[1] = lo & right_side_mask;
  }
  // Combine the left and right sides together to get result
  int64_t result = (int64_t)state[0] << right_side_bits | (int64_t)state[1];
  output[i] = result;
}

int64_t getCipherBits(int64_t capacity)
{
  if (capacity == 0)
    return 0;
  int64_t i = 0;
  capacity--;
  while (capacity != 0)
  {
    i++;
    capacity >>= 1;
  }

  return std::max(i, int64_t(4));
}

at::Tensor random_philox_bijection_cuda(
    const int64_t n,
    const int64_t num_rounds,
    at::Tensor &dummy)
{
  // prepare masks
  int64_t total_bits = getCipherBits(n);
  int64_t left_side_bits = total_bits / 2;
  int64_t left_side_mask = (1ULL << left_side_bits) - 1;
  int64_t right_side_bits = total_bits - left_side_bits;
  int64_t right_side_mask = (1ULL << right_side_bits) - 1;

  // generate random keys
  at::Tensor keys_tensor = torch::empty(static_cast<int64_t>(num_rounds), torch::kInt32);
  auto keys_accessor = keys_tensor.accessor<int32_t, 1>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dis(
    std::numeric_limits<int32_t>::min(), 
    std::numeric_limits<int32_t>::max()
  );
  for (int i = 0; i < num_rounds; i++) {
    keys_accessor[i] = dis(gen);
  }
  keys_tensor = keys_tensor.to(torch::kCUDA);

  // allocate output  
  at::Tensor output = torch::empty(1LL << total_bits, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

  // launch kernel
  const int64_t threads_per_block = 1024;  // maximum threads per block for many modern GPUs
  const int64_t blocks = (n + threads_per_block - 1) / threads_per_block;
  random_philox_bijection<<<blocks, threads_per_block>>>(
    1LL << total_bits,
    static_cast<int64_t>(num_rounds),
    static_cast<int64_t>(right_side_bits),
    static_cast<int64_t>(left_side_bits),
    static_cast<int64_t>(right_side_mask),
    static_cast<int64_t>(left_side_mask),
    keys_tensor.data_ptr<int32_t>(),
    output.data_ptr<int64_t>()
  );

  // check for any CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  return output;
}

TORCH_LIBRARY_IMPL(plas, CUDA, m) {
  m.impl("random_philox_bijection", &random_philox_bijection_cuda);
}

} // namespace plas
