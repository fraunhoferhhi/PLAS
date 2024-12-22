// Debuggible test for random_philox_permutation (CUDA-gdb)

#include "permuters.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <random>
#include <vector>

bool test_LCG_fused_inverse(int num_permutations = 3, int n = 10) {
  LCGPermuter permuter;
  int *original = new int[n];
  for (int i = 0; i < n; i++) {
    original[i] = i;
  }

  int *result = new int[n];
  std::copy(original, original + n, result);
  int *temp = new int[n];

  // Apply permutations in forward order
  for (int p = 0; p < num_permutations; p++) {
    int *d_perm = permuter.get_new_permutation(n);
    int *h_perm = new int[n];
    cudaMemcpy(h_perm, d_perm, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Apply permutation
    for (int i = 0; i < n; i++) {
      temp[i] = result[h_perm[i]];
    }
    std::copy(temp, temp + n, result);
  }

  // Get and apply single fused inverse permutation
  int *d_fused_inv = permuter.get_inverse_permutation(n);
  int *h_fused_inv = new int[n];
  cudaMemcpy(h_fused_inv, d_fused_inv, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Apply fused inverse permutation
  for (int i = 0; i < n; i++) {
    temp[i] = result[h_fused_inv[i]];
  }
  std::copy(temp, temp + n, result);

  // Check if we got back the original array
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (result[i] != original[i]) {
      success = false;
      break;
    }
  }

  // Cleanup
  delete[] original;
  delete[] result;
  delete[] temp;
  delete[] h_fused_inv;

  return success;
}

void test_LCG_permuter() {
  bool success = true;
  for (int num_permutations = 1; num_permutations < 20; num_permutations++) {
    std::vector<int> possible_ns = {2,          5,          10,      100,
                                    1'000,      10'000,     100'000, 1'000'000,
                                    10'000'000, 100'000'000};
    success = test_LCG_fused_inverse(num_permutations,
                                     possible_ns[rand() % possible_ns.size()]);
  }
  if (!success) {
    std::cout << "failed test_LCG_permuter" << std::endl;
  } else {
    std::cout << "passed test_LCG_permuter" << std::endl;
  }
}

int main(int argc, char **argv) { test_LCG_permuter(); }
