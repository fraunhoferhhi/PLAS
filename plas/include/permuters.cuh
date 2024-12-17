#include <vector>

#include <thread>
#include <mutex>
#include <random>

class RandomPermuter {
protected:
  int *current_permutation; // maintain a pointer for memory management

public:
  virtual ~RandomPermuter() = default;
  virtual int *get_new_permutation(int n) = 0;
  virtual int *get_inverse_permutation(int n) = 0;
  virtual bool supports_inverse_fusion() = 0;
};

// // --------- Radix Sort Permuter --------- //
// class RadixSortPermuter : public RandomPermuter
// {};

// // --------- LCG Permuter --------- //

std::pair<int, int> get_random_lcg_generator_and_offset(int n); 

int *lcg_permutation_cuda(int n, int generator, int offset, int* output);

class LCGPermuter : public RandomPermuter {
  int fused_inv_gen = 1;
  int fused_inv_offset = 0;
  int last_n = 0;

  std::pair<int, int> get_inverse_parameters(int generator, int offset) {
    return {1, 0};
  }

public:
  int *get_new_permutation(int n) {
    auto [gen, offset] = get_random_lcg_generator_and_offset(n);

    // maintain the parameters for the inverse permutation of all generated
    // permutations unto a call to get_inverse_permutation
    auto [inv_gen, inv_offset] = get_inverse_parameters(gen, offset);
    fused_inv_offset = (fused_inv_gen * inv_offset + fused_inv_offset) % n;
    fused_inv_gen = (inv_gen * fused_inv_gen) % n;

    if (n != last_n) {
      cudaFree(current_permutation);
      current_permutation = (int *)malloc(n * sizeof(int));
      lcg_permutation_cuda(n, gen, offset, current_permutation);
      last_n = n;
    } else {
      lcg_permutation_cuda(n, gen, offset, current_permutation);
    }
    return current_permutation;
  }
  int *get_inverse_permutation(int n) {
    cudaFree(current_permutation);
    current_permutation = (int *)malloc(n * sizeof(int));
    lcg_permutation_cuda(n, fused_inv_gen, fused_inv_offset, current_permutation);

    // reset the parameters for the inverse permutation
    fused_inv_gen = 1;
    fused_inv_offset = 0;
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
