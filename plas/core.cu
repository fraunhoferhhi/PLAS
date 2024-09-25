#include <vector>
#include <iostream>
#include <random>

struct Tensor {
  float f1;
};

__global__ void group_data(Tensor* data, Tensor* grouped_data, int64_t n, int64_t g, int64_t offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = (i * g + offset) % n;
  grouped_data[i] = data[j];
}

__global__ void swap_within_groups(Tensor* data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  Tensor tmp = data[2 * i];
  data[2 * i] = data[2 * i + 1];
  data[2 * i + 1] = tmp;
}

int64_t inverse(int64_t N, int64_t g) {
  int64_t n[2] = { N, g };
  int64_t b[3] = { 0, 1, 0 };
  int i = 2;
  while (n[1] > 1) {
    int64_t c = n[0] / n[1];
    int64_t r = n[0] - c * n[1];
    n[0] = n[1];
    n[1] = r;
    b[i % 3] = b[(i - 2) % 3] - c * b[(i - 1) % 3];
    i += 1;
  }
  return (b[(i - 1) % 3] % N + N) % N;
}

int main(void) {
  int N = 8, g = 5, offset = 4;

  std::vector<Tensor> data_on_host = { {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} };

  Tensor* data_on_device, * grouped_data_on_device;
  cudaMalloc(&data_on_device, N * sizeof(Tensor));
  cudaMalloc(&grouped_data_on_device, N * sizeof(Tensor));
  cudaMemcpy(data_on_device, data_on_host.data(), N * sizeof(Tensor), cudaMemcpyHostToDevice);

  group_data << <1, N >> > (data_on_device, grouped_data_on_device, N, g, offset);

  std::vector<Tensor> grouped_data_on_host(N);
  cudaMemcpy(grouped_data_on_host.data(), grouped_data_on_device, N * sizeof(Tensor), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    std::cout << grouped_data_on_host[i].f1 << " ";
  }
  std::cout << std::endl;

  swap_within_groups << <1, N / 2 >> > (grouped_data_on_device);
  cudaMemcpy(grouped_data_on_host.data(), grouped_data_on_device, N * sizeof(Tensor), cudaMemcpyDeviceToHost);
  for (int i = 0;i < N; i++) {
    std::cout << grouped_data_on_host[i].f1 << " ";
  }
  std::cout << std::endl;

  int64_t g_inv = inverse(N, g);
  int64_t offset_inv = -(offset * g_inv % N) + N;
  std::cout << g_inv << " " << offset_inv << std::endl;
  group_data << <1, N >> > (grouped_data_on_device, data_on_device, N, g_inv, offset_inv);

  cudaMemcpy(data_on_host.data(), data_on_device, N * sizeof(Tensor), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    std::cout << data_on_host[i].f1 << " ";
  }
  std::cout << std::endl;



  cudaFree(data_on_device);
  cudaFree(grouped_data_on_device);
}
