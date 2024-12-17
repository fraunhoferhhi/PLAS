#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <thrust/reduce.h>

#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "blur.cuh"
#include "plas.cuh"

std::vector<int> get_filter_side_lengths(int larger_grid_side,
                                         int min_filter_side_length,
                                         float decrease_factor) {
  std::vector<int> filter_side_lengths;
  float filter_side_length = larger_grid_side * decrease_factor;
  while (filter_side_length >= min_filter_side_length) {
    filter_side_lengths.push_back(int(filter_side_length));
    filter_side_length *= decrease_factor;
  }
  return filter_side_lengths;
}

template <typename T> T *get_3d_tensor_device_memory(int H, int W, int D) {
  T *tensor;
  size_t size = H * W * D * sizeof(T);
  cudaMalloc(&tensor, size);
  return tensor;
}

template <typename T> T *get_2d_tensor_device_memory(int H, int W) {
  T *tensor;
  size_t size = H * W * sizeof(T);
  cudaMalloc(&tensor, size);
  return tensor;
}

template <typename T> T *get_1d_tensor_device_memory(int size) {
  T *tensor;
  size_t size_bytes = size * sizeof(T);
  cudaMalloc(&tensor, size_bytes);
  return tensor;
}

__global__ void set_identity_permutation_kernel(int *tensor, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < size) {
    tensor[x] = x;
  }
}

void set_identity_permutation(int *tensor, int size) {
  int threads_per_block = 1024;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  set_identity_permutation_kernel<<<num_blocks, threads_per_block>>>(tensor,
                                                                     size);
  cudaDeviceSynchronize();
}

__global__ void gather_kernel(const float *input, float *output,
                              int *permutation, int H, int W, int D) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < H * W) {
    for (int d = 0; d < D; d++) {
      output[index * D + d] = input[permutation[index] * D + d];
    }
  }
}

void gather(const float *input, float *output, int *permutation, int H, int W,
            int D) {
  int threads_per_block = 1024;
  int num_blocks = (H * W + threads_per_block - 1) / threads_per_block;
  gather_kernel<<<num_blocks, threads_per_block>>>(input, output, permutation,
                                                   H, W, D);
  cudaDeviceSynchronize();
}

// scatter version of blockify | TODO: write a gather version
__global__ void blockify_kernel(const float *grid, float *blockified_grid,
                                const float *target_grid,
                                float *blockified_target_grid, const int *index,
                                int *blockified_index, int H, int W, int C,
                                int block_len_x, int block_len_y, int tx,
                                int ty) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= (H - ty) * (W - tx))
    return;

  // Calculate read start position
  int x = thread_idx % (W - tx);
  int y = thread_idx / (W - tx);
  int read_start = y * W * C + x * C;
  read_start += (y + 1) * tx * C; // accomodate for tx
  read_start += ty * W * C;       // accomodate for ty

  // Calculate write start position
  int num_blocks_x = (W - tx) / block_len_x;
  int block_x = x / block_len_x;
  int block_y = y / block_len_y;
  int block_idx = block_y * num_blocks_x + block_x;
  int write_start = block_idx * block_len_x * block_len_y * C;
  write_start += (y - block_y * block_len_y) * block_len_x * C;
  write_start += (x - block_x * block_len_x) * C;

  // Copy data
  for (int c = 0; c < C; c++) {
    blockified_grid[write_start + c] = grid[read_start + c];
    blockified_target_grid[write_start + c] = target_grid[read_start + c];
  }
  blockified_index[write_start / C] = index[read_start / C];
}

void blockify(const float *grid, float *blockified_grid,
              const float *target_grid, float *blockified_target_grid,
              const int *index, int *blockified_index, int H, int W, int C,
              int block_len_x, int block_len_y, int tx, int ty) {
  // Calculate number of blocks needed in output
  int num_blocks_x = (W - tx) / block_len_x;
  int num_blocks_y = (H - ty) / block_len_y;
  int total_blocks = num_blocks_x * num_blocks_y;
  int block_size = block_len_x * block_len_y;

  // Launch kernel with one thread per output position
  int threads_per_block = 1024;
  int num_blocks =
      (total_blocks * block_size + threads_per_block - 1) / threads_per_block;
  blockify_kernel<<<num_blocks, threads_per_block>>>(
      grid, blockified_grid, target_grid, blockified_target_grid, index,
      blockified_index, H, W, C, block_len_x, block_len_y, tx, ty);
  cudaDeviceSynchronize();
}

// scatter version of unblockify | TODO: write a gather version
__global__ void unblockify_kernel(float *grid, const float *blockified_grid,
                                  float *target_grid,
                                  const float *blockified_target_grid,
                                  int *index, const int *blockified_index,
                                  int H, int W, int C, int block_len_x,
                                  int block_len_y, int tx, int ty) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= (H - ty) * (W - tx))
    return;

  // Calculate write start position
  int x = thread_idx % (W - tx);
  int y = thread_idx / (W - tx);
  int write_start = y * W * C + x * C;
  write_start += (y + 1) * tx * C; // accomodate for tx
  write_start += ty * W * C;       // accomodate for ty

  // Calculate read start position
  int num_blocks_x = (W - tx) / block_len_x;
  int block_x = x / block_len_x;
  int block_y = y / block_len_y;
  int block_idx = block_y * num_blocks_x + block_x;
  int read_start = block_idx * block_len_x * block_len_y * C;
  read_start += (y - block_y * block_len_y) * block_len_x * C;
  read_start += (x - block_x * block_len_x) * C;

  // Copy data
  for (int c = 0; c < C; c++) {
    grid[write_start + c] = blockified_grid[read_start + c];
    target_grid[write_start + c] = blockified_target_grid[read_start + c];
  }
  index[write_start / C] = blockified_index[read_start / C];
}

void unblockify(float *grid, const float *blockified_grid, float *target_grid,
                const float *blockified_target_grid, int *index,
                const int *blockified_index, int H, int W, int C,
                int block_len_x, int block_len_y, int tx, int ty) {
  // Calculate number of blocks needed in output
  int num_blocks_x = (W - tx) / block_len_x;
  int num_blocks_y = (H - ty) / block_len_y;
  int total_blocks = num_blocks_x * num_blocks_y;
  int block_size = block_len_x * block_len_y;

  // Launch kernel with one thread per output position
  int threads_per_block = 1024;
  int num_blocks =
      (total_blocks * block_size + threads_per_block - 1) / threads_per_block;
  unblockify_kernel<<<num_blocks, threads_per_block>>>(
      grid, blockified_grid, target_grid, blockified_target_grid, index,
      blockified_index, H, W, C, block_len_x, block_len_y, tx, ty);
  cudaDeviceSynchronize();
}

__global__ void group_kernel(float *blockified_grid[2],
                             float *blockified_target_grid[2],
                             int *blockified_index[2], int *permutation,
                             int block_size, int num_blocks, int C, int turn) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= num_blocks * block_size)
    return;

  int block_idx = thread_idx / block_size;
  int block_offset = thread_idx % block_size;
  int read_start = (block_idx * block_size + permutation[block_offset]) * C;
  int write_start = (block_idx * block_size + block_offset) * C;
  for (int c = 0; c < C; c++) {
    blockified_grid[1 - turn][write_start + c] =
        blockified_grid[turn][read_start + c];
    blockified_target_grid[1 - turn][write_start + c] =
        blockified_target_grid[turn][read_start + c];
  }
  blockified_index[1 - turn][write_start / C] =
      blockified_index[turn][read_start / C];
}

void group(float *blockified_grid[2], float *blockified_target_grid[2],
           int *blockified_index[2], int *permutation, int block_size,
           int num_blocks, int C, int turn) {
  int threads_per_block = 1024;
  int num_cuda_blocks =
      (num_blocks * block_size + threads_per_block - 1) / threads_per_block;
  group_kernel<<<num_cuda_blocks, threads_per_block>>>(
      blockified_grid, blockified_target_grid, blockified_index, permutation,
      block_size, num_blocks, C, turn);
  cudaDeviceSynchronize();
}

void swap_within_group(float *blockified_grid, float *blockified_target_grid,
                       float *dist_to_target, int *blockified_index,
                       int *permutation, int block_size, int num_blocks) {}

std::pair<float *, int *> sort_with_plas(
    const float *grid_input, // of size H x W x C, memory layout:
                             // grid_input[h * W * C + w * C + c]
    int H, int W, int C, int min_block_side = 4, int min_filter_side_length = 1,
    float fitler_decrease_factor = 0.9, float improvement_break = 1e-5,
    int seed = 1337, int min_group_configs = 1, int max_group_configs = 10,
    int border_type_x = BorderType::REFLECT,
    int border_type_y = BorderType::REFLECT, bool verbose = false,
    RandomPermuter *permuter = new LCGPermuter()) {

  // grid is the current reordered input grid
  float *grid = get_3d_tensor_device_memory<float>(H, W, C);
  float *blockified_grid[2];
  blockified_grid[0] = get_2d_tensor_device_memory<float>(H * W, C);
  blockified_grid[1] = get_2d_tensor_device_memory<float>(H * W, C);
  // Permuting data across CUDA different CUDA blocks prohibits in-place permuting.
  // Therefore we need to use two buffers to store the data.
  // At index $turn is the data to be processed next. 1 - $turn is the location
  // where the processed data is stored.
  bool turn = 0;

  // target_grid is the idealized target grid
  float *target_grid = get_3d_tensor_device_memory<float>(H, W, C);
  float *blockified_target_grid[2];
  blockified_target_grid[0] = get_2d_tensor_device_memory<float>(H * W, C);
  blockified_target_grid[1] = get_2d_tensor_device_memory<float>(H * W, C);
  float *dist_to_target =
      get_1d_tensor_device_memory<float>(H * W); // also blockified

  // index maintains how pixels are reordered in reference to input_grid
  int *index = get_2d_tensor_device_memory<int>(H, W);
  int *blockified_index[2];
  blockified_index[0] = get_1d_tensor_device_memory<int>(H * W);
  blockified_index[1] = get_1d_tensor_device_memory<int>(H * W);
  set_identity_permutation(index, H * W);

  // start with an initial random reordering to not end up in a bad local
  // minimum
  int *permutation = permuter->get_new_permutation(H * W);
  gather(grid_input, grid, permutation, H, W, C);

  std::vector<int> filter_side_lengths = get_filter_side_lengths(
      min(H, W), min_filter_side_length, fitler_decrease_factor);
  for (int filter_side_length : filter_side_lengths) {

    // Blur the grid to obtain an idealized target.
    int filter_size = filter_side_length + (filter_side_length % 2);
    apply_filter(grid, target_grid, W, H, C, filter_size, filter_size);

    int block_len_x = filter_size - 1;
    int block_len_y = filter_size - 1;
    int block_size = block_len_x * block_len_y;
    int num_blocks = (W / block_len_x) * (H / block_len_y);
    // Draw a random alignment of the blocks in the grid.
    bool is_left_aligned_x = rand() % 2;
    bool is_left_aligned_y = rand() % 2;
    int tx = is_left_aligned_x ? 0 : (W % block_len_x);
    int ty = is_left_aligned_y ? 0 : (H % block_len_y);

    // Write the blocks sequentially to the blockified grid for more efficient
    // memory access patterns.
    blockify(grid, blockified_grid[turn], target_grid,
             blockified_target_grid[turn], index, blockified_index[turn], H, W, C,
             block_len_x, block_len_y, tx, ty);

    // Try a few random group configurations until the improvement is too small,
    // or we have tried enough configurations. But try at least
    // min_group_configs.
    float previous_dist_to_target = std::numeric_limits<float>::max();
    float improvement = std::numeric_limits<float>::max();
    for (int j = 0; j < max_group_configs &&
                    (improvement > improvement_break || j < min_group_configs);
         j++) {
      // Group all pixels inside a block into groups of 4 pixels based on a
      // random permutation.
      int *permutation = permuter->get_new_permutation(block_size);
      group(blockified_grid, blockified_target_grid, blockified_index,
            permutation, block_size, num_blocks, C, turn);
      turn = 1 - turn;

      // For each group, find the best permutation out of the 24 possible
      // ones judged by the idealized target grid.
      swap_within_group(blockified_grid[turn], blockified_target_grid[turn],
                        dist_to_target, blockified_index[turn], permutation,
                        block_size, num_blocks);

      // Imediately apply the inverse permutation if the permuter does not
      // support inverse fusion.
      if (!permuter->supports_inverse_fusion()) {
        int *inverse_permutation =
            permuter->get_inverse_permutation(block_size);
        group(blockified_grid, blockified_target_grid, blockified_index,
              inverse_permutation, block_size, num_blocks, C, turn);
        turn = 1 - turn;
      }

      // Compute the improvement.
      float current_dist_to_target =
          thrust::reduce(dist_to_target,
                         dist_to_target + num_blocks * block_size) /
          (num_blocks * block_size);
      improvement = previous_dist_to_target - current_dist_to_target;
      previous_dist_to_target = current_dist_to_target;
    }

    // Apply the inverse permutation if the permuter supports inverse fusion.
    if (permuter->supports_inverse_fusion()) {
      int *inverse_permutation = permuter->get_inverse_permutation(block_size);
      group(blockified_grid, blockified_target_grid, blockified_index,
            inverse_permutation, block_size, num_blocks, C, turn);
      turn = 1 - turn;
    }

    // Write the blocks back to the grid.
    unblockify(grid, blockified_grid[turn], target_grid,
               blockified_target_grid[turn], index, blockified_index[turn], H, W,
               C, block_len_x, block_len_y, tx, ty);
  }

  // Free all intermediate tensors.
  cudaFree(blockified_grid);
  cudaFree(target_grid);
  cudaFree(blockified_target_grid);
  cudaFree(blockified_index);

  return std::make_pair(grid, index);
}