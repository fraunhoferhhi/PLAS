#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "filter_less_than.cuh"

#include <iostream>
#include <limits>
#include <random>
#include <vector>

static const int64_t M0 = UINT64_C(0xD2B74407B1CE6E93);

__device__ int32_t mulhilo(int64_t a, int32_t b, int32_t &hip) {
  int64_t product = a * int64_t(b);
  hip = product >> 32;
  return int32_t(product);
}

__global__ void random_philox_permutation(const int n, const int num_rounds,
                                          const int right_side_bits,
                                          const int left_side_bits,
                                          const int right_side_mask,
                                          const int left_side_mask,
                                          const int *keys, int *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int state[2] = {int(i >> right_side_bits), int(i & right_side_mask)};
  for (int i = 0; i < num_rounds; i++) {
    int hi;
    int lo = mulhilo(M0, state[0], hi);
    lo =
        (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
    state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
    state[1] = lo & right_side_mask;
  }
  // Combine the left and right sides together to get result
  int result = state[0] << right_side_bits | state[1];
  output[i] = result;
}

int getCipherBits(int capacity) {
  if (capacity == 0)
    return 0;
  int i = 0;
  capacity--;
  while (capacity != 0) {
    i++;
    capacity >>= 1;
  }

  return i;
}

std::vector<int> get_random_philox_keys(int num_rounds) {
  std::vector<int> keys(num_rounds);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dis(
      std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
  for (int i = 0; i < num_rounds; i++) {
    keys[i] = dis(gen);
  }
  return keys;
}

int *philox_permutation_cuda(const int n, const int num_rounds, const std::vector<int> &keys) {
  // Ceil n to the next power of 2
  int N = int(1) << (32 - __builtin_clzll(n - 1));
  N = max(N, 2048);

  // prepare masks
  int total_bits = getCipherBits(N);
  int left_side_bits = total_bits / 2;
  int left_side_mask = (1ULL << left_side_bits) - 1;
  int right_side_bits = total_bits - left_side_bits;
  int right_side_mask = (1ULL << right_side_bits) - 1;

  // Copy keys to GPU memory
  int *d_keys;
  cudaMalloc(&d_keys, num_rounds * sizeof(int));
  cudaMemcpy(d_keys, keys.data(), num_rounds * sizeof(int), cudaMemcpyHostToDevice);

  // allocate output
  int *outer_permutation;
  cudaMalloc(&outer_permutation, N * sizeof(int));

  // launch kernel
  const int threads_per_block =
      1024; // maximum threads per block for many modern GPUs
  const int blocks = (N + threads_per_block - 1) / threads_per_block;

  random_philox_permutation<<<blocks, threads_per_block>>>(
      N, num_rounds, right_side_bits, left_side_bits, right_side_mask,
      left_side_mask, d_keys, outer_permutation);
  cudaDeviceSynchronize();
  cudaFree(d_keys);

  int *inner_permutation = filter_less_than(n, outer_permutation, N);
  cudaDeviceSynchronize();
  cudaFree(outer_permutation);

  return inner_permutation;
}

int *random_philox_permutation_cuda(int n, int num_rounds) {
  std::vector<int> keys = get_random_philox_keys(num_rounds);
  return philox_permutation_cuda(n, num_rounds, keys);
}
