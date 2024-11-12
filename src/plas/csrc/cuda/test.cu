// Debuggible test for random_philox_permutation (CUDA-gdb)

#include "plas.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <random>
#include <vector>

bool device_array_is_permutation(int *d_array, int n) {
  int *host_array = new int[n];
  cudaMemcpy(host_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::vector<bool> seen(n, false);
  for (int i = 0; i < n; i++) {
    if (host_array[i] < 0 || host_array[i] >= n || seen[host_array[i]]) {
      return false;
    }
    seen[host_array[i]] = true;
  }
  return true;
}

void test_random_philox_permutation() {
  std::vector<int> sizes = {
      1,         2,         3,         4,          5,          6,      7,
      8,         9,         10,        20,         70,       1024,   1025,
      2048,      2049,      4096,      4097,       10'000,     50'000, 100'000,
      1'000'000, 4'000'000, 5'000'000, 10'000'000, 100'000'000};
  int num_philox_rounds = 5;
  int samples_per_size = 5;
  for (int n : sizes) {
    for (int sample = 0; sample < samples_per_size; sample++) {
      int *result = random_philox_permutation_cuda(n, num_philox_rounds);
      cudaDeviceSynchronize();
      if (!device_array_is_permutation(result, n)) {
        std::cout << "test_random_philox_permutation failed for size " << n
                  << " at sample " << sample << std::endl;
        return;
      }
      cudaFree(result);
    }
  }
  std::cout << "test_random_philox_permutation passed" << std::endl;
}

void test_multiple_random_philox_permutations_calls_without_free_inbetween() {
  std::vector<int> sizes = {6, 85, 2049, 4097, 10'000, 50'000};
  int num_philox_rounds = 5;
  int num_iterations = 10;

  std::vector<int*> results;
  for (int i = 0; i < num_iterations; i++) {
    for (int n : sizes) {
      int *result = random_philox_permutation_cuda(n, num_philox_rounds);
      results.push_back(result);
      cudaDeviceSynchronize();
      if (!device_array_is_permutation(result, n)) {
        std::cout << "test_multiple_calls passed for size " << n 
                 << " at iteration " << i << std::endl;
        return;
      }
      // Deliberately not freeing result to test memory handling
    }
  }
  for (int *result : results) {
    cudaFree(result);
  }
  std::cout << "test_multiple_random_philox_permutations_calls_without_free_inbetween passed" << std::endl;
}

int main(int argc, char **argv) {
  test_random_philox_permutation();
  test_multiple_random_philox_permutations_calls_without_free_inbetween();
}
