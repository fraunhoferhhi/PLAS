#include<stdint.h>
#include<iostream>
static const uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);

__device__ uint32_t mulhilo(uint64_t a, uint32_t b, uint32_t& hip) {
  uint64_t product = a * uint64_t(b);
  hip = product >> 32;
  return uint32_t(product);
}

extern "C" __global__ void random_philox_bijection(
  const uint64_t n,
  const uint64_t num_rounds,
  const uint64_t right_side_bits,
  const uint64_t left_side_bits,
  const uint64_t right_side_mask,
  const uint64_t left_side_mask,
  const uint64_t* keys,
  uint64_t* output
) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint32_t state[2] = { uint32_t(i >> right_side_bits), uint32_t(i & right_side_mask) };
  for (int i = 0; i < num_rounds; i++) {
    uint32_t hi;
    uint32_t lo = mulhilo(M0, state[0], hi);
    lo = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
    state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
    state[1] = lo & right_side_mask;
  }
  // Combine the left and right sides together to get result
  uint64_t result = (uint64_t)state[0] << right_side_bits | (uint64_t)state[1];
  output[i] = result;
}
