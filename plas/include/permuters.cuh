#include <vector>

#include <cassert>
#include <cuda.h>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

class RandomPermuter {
protected:
  int *current_permutation = nullptr; // maintain a pointer for memory management

public:
  ~RandomPermuter() {
    if (current_permutation) {
      cudaFree(current_permutation);
    }
  }
  virtual int *get_new_permutation(int n) = 0;
  virtual int *get_inverse_permutation(int n) = 0;
  virtual bool supports_inverse_fusion() = 0;
};

// // --------- Radix Sort Permuter --------- //
// class RadixSortPermuter : public RandomPermuter
// {};

// // --------- LCG Permuter --------- //

std::pair<int, int> get_random_lcg_generator_and_offset(int n);

int *lcg_permutation_cuda(int n, int generator, int offset, int *output);

class LCGPermuter : public RandomPermuter {
  int fused_gen = 1;
  int fused_offset = 0;
  int last_n = -1;

  inline int mod(int64_t a, int n) {
    return ((a % n) + n) % n;
  }

  std::pair<int, int> get_inverse_parameters(int generator, int offset, int n) {
    int inv_gen = multiplicative_inverse(generator, n);
    int inv_offset = mod(int64_t(inv_gen) * (-offset), n);
    return {inv_gen, inv_offset};
  }

  int multiplicative_inverse(int generator, int n) {
    int n0 = n, n1 = generator;
    int b[3] = {0, 1, 0};
    int c = 0;
    int i = 2;
    while (n1 > 1) {
      c = n0 / n1;
      int r = n0 - c * n1;
      n0 = n1;
      n1 = r;
      b[i % 3] = b[(i + 1) % 3] - c * b[(i + 2) % 3];
      i++;
    }
    return mod(b[(i + 2) % 3], n);
  }

public:
  int *get_new_permutation(int n) {
    auto [gen, offset] = get_random_lcg_generator_and_offset(n);
    // maintain the parameters of the composition of all generated permutations
    // unto a call to get_inverse_permutation 
    fused_offset = mod(int64_t(fused_gen) * offset + fused_offset, n);
    fused_gen = mod(int64_t(fused_gen) * gen, n);
    if (n != last_n) {
      if (current_permutation) {
        cudaFree(current_permutation);
        cudaDeviceSynchronize();
      }
      cudaMalloc(&current_permutation, n * sizeof(int));
      assert(current_permutation != nullptr);
      lcg_permutation_cuda(n, gen, offset, current_permutation);
      last_n = n;
    } else {
      lcg_permutation_cuda(n, gen, offset, current_permutation);
    }
    return current_permutation;
  }
  int *get_inverse_permutation(int n) {
    auto [inv_gen, inv_offset] = get_inverse_parameters(fused_gen, fused_offset, n);
    lcg_permutation_cuda(n, inv_gen, inv_offset,
                         current_permutation);


    // reset the parameters for the inverse permutation
    fused_gen = 1;
    fused_offset = 0;
    return current_permutation;
  }
  bool supports_inverse_fusion() { return true; }
};

// --------- Philox Permuter --------- //
std::vector<int> get_random_philox_keys(int num_rounds);

int *philox_permutation_cuda(int n, int num_rounds,
                             const std::vector<int> &keys);
int *philox_inverse_permutation_cuda(int n, int num_rounds,
                                     const std::vector<int> &keys);

class PhiloxPermuter : public RandomPermuter {
  int num_rounds;
  std::vector<int> keys;

public:
  PhiloxPermuter(int num_rounds) : num_rounds(num_rounds) {}
  int *get_new_permutation(int n) {
    keys = get_random_philox_keys(num_rounds);
    cudaFree(current_permutation);
    current_permutation = philox_permutation_cuda(n, num_rounds, keys);
    return current_permutation;
  };
  int *get_inverse_permutation(int n) {
    cudaFree(current_permutation);
    current_permutation = philox_inverse_permutation_cuda(n, num_rounds, keys);
    return current_permutation;
  }
  bool supports_inverse_fusion() { return false; }
};
