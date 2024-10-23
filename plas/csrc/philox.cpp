#include <torch/extension.h>
#include <random>
namespace plas
{

    at::Tensor random_philox_bijection_cpu(int64_t n, int64_t num_rounds, at::Tensor &dummy)
    {
        // Create a Mersenne Twister random number generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create a tensor with integers from 0 to n-1
        auto permutation = torch::arange(n, torch::kInt64);

        // Fisher-Yates shuffle algorithm
        for (int64_t i = n - 1; i > 0; --i)
        {
            // Generate a random index between 0 and i (inclusive)
            std::uniform_int_distribution<int64_t> dis(0, i);
            int64_t j = dis(gen);

            // Swap elements at positions i and j
            auto temp = permutation[i].item<int64_t>();
            permutation[i] = permutation[j];
            permutation[j] = temp;
        }

        return permutation;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(plas, m)
    {
        m.def("random_philox_bijection(int n, int num_rounds, Tensor dummy) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(plas, CPU, m)
    {
        m.impl("random_philox_bijection", &random_philox_bijection_cpu);
    }
} // namespace plas
