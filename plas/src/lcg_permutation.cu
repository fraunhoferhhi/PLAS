#include <random>
#include <numeric>

std::pair<int, int> get_random_lcg_generator_and_offset(int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, n - 1);
  bool found_coprime = false;
  int generator = 0;
  while (!found_coprime) {
    generator = dis(gen);
    if (std::gcd(generator, n) == 1) {
      found_coprime = true;
    }
  }
  std::uniform_int_distribution<> dis_offset(0, n - 1);
  int offset = dis_offset(gen);
  return {generator, offset};
}

__global__ void lcg_permutation_kernel(int n, int generator, int offset, int* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = ((int64_t)generator * index + offset) % n;
  }
}

int *lcg_permutation_cuda(int n, int generator, int offset, int* output) {
  int threads = 1024;
  int blocks = (n + threads - 1) / threads;
  lcg_permutation_kernel<<<blocks, threads>>>(n, generator, offset, output);
  return output;
}
