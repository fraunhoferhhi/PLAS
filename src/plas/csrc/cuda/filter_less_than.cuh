#include <iostream>

#ifndef _STREAM_COMPACTION_KERNEL_H_
#define _STREAM_COMPACTION_KERNEL_H_

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define MAX_BLOCK_SIZE 1024

// Define this to more rigorously avoid bank conflicts, even at the lower (root)
// levels of the tree #define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index)                                            \
  ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index) CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index) temp[index]
#endif

__global__ void exclusive_scan_blockwise(int n, int *g_output,
                                         const int *g_input, int *sums) {
  // n must be a multiple of 2 * blockDim.x

  // Dynamically allocated shared memory for scan kernels
  extern __shared__ int temp[];

  int local_n = 2 * blockDim.x;

  int ai = threadIdx.x;
  int bi = threadIdx.x + (local_n / 2);

  // compute spacing to avoid bank conflicts
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Cache the computational window in shared memory
  TEMP(ai + bankOffsetA) = g_input[ai + blockIdx.x * local_n];
  TEMP(bi + bankOffsetB) = g_input[bi + blockIdx.x * local_n];

  int offset = 1;

  // build the sum in place up the tree
  for (int d = local_n / 2; d > 0; d >>= 1) {
    __syncthreads();

    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      TEMP(bi) += TEMP(ai);
    }

    offset *= 2;
  }

  // scan back down the tree
  __syncthreads();
  // clear the last element of the block and store the sum in the sums array
  if (threadIdx.x == 0) {
    int index = local_n - 1;
    index += CONFLICT_FREE_OFFSET(index);
    sums[blockIdx.x] = TEMP(index);
    TEMP(index) = 0;
  }

  // traverse down the tree building the scan in place
  __syncthreads();
  for (int d = 1; d < local_n; d *= 2) {
    offset /= 2;

    __syncthreads();

    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = TEMP(ai);
      TEMP(ai) = TEMP(bi);
      TEMP(bi) += t;
    }
  }

  // write results to global memory
  __syncthreads();
  g_output[ai + blockIdx.x * local_n] = TEMP(ai + bankOffsetA);
  g_output[bi + blockIdx.x * local_n] = TEMP(bi + bankOffsetB);
}

__global__ void increment_indices(int N, int *indices, int *incr) {
  int index = threadIdx.x + blockIdx.x * 2 * blockDim.x;
  if (index < N) {
    indices[index] += incr[blockIdx.x];
    indices[index + blockDim.x] += incr[blockIdx.x];
  }
}

__global__ void scatter(int N, int *input, int *output, int *scatter_indices,
                        int *mask) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N && mask[index]) {
    output[scatter_indices[index]] = input[index];
  }
}

int *exclusive_scan(int N, const int *input) {
  int num_blocks = (N + 2 * MAX_BLOCK_SIZE - 1) / (2 * MAX_BLOCK_SIZE);
  int *output, *sums;
  cudaMalloc(&output, N * sizeof(int));
  cudaMalloc(&sums, num_blocks * sizeof(int));
  exclusive_scan_blockwise<<<num_blocks, min(N, MAX_BLOCK_SIZE),
                             4 * MAX_BLOCK_SIZE * sizeof(int)>>>(N, output,
                                                                 input, sums);
  if (num_blocks > 1) {
    int *incr = exclusive_scan(num_blocks, sums);
    increment_indices<<<num_blocks, MAX_BLOCK_SIZE>>>(N, output, incr);
    cudaDeviceSynchronize();
    cudaFree(incr);
  }
  cudaDeviceSynchronize();
  cudaFree(sums);
  return output;
}

__global__ void fill_less_than_mask(int n, int *values, int *mask, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N && values[index] < n) {
    mask[index] = 1;
  } else if (index < N) {
    mask[index] = 0;
  }
}

int *filter_less_than(int n, int *values, int N) {
  int num_blocks;

  // Fill mask
  int *mask;
  cudaMalloc(&mask, N * sizeof(int));
  num_blocks = (N + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
  fill_less_than_mask<<<num_blocks, MAX_BLOCK_SIZE>>>(n, values, mask, N);

  // Compute exclusive prefix sum of mask
  int *indices = exclusive_scan(N, mask);

  // Scatter
  int *permutation;
  cudaMalloc(&permutation, n * sizeof(int));
  num_blocks = (N + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
  scatter<<<num_blocks, MAX_BLOCK_SIZE>>>(N, values, permutation, indices,
                                          mask);

  cudaDeviceSynchronize();
  cudaFree(mask);
  cudaFree(indices);
  return permutation;
}

#endif // _STREAM_COMPACTION_KERNEL_H_
