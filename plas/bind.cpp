#include <torch/extension.h>
#include <random>
namespace plas
{

    at::Tensor random_philox_permutation_cpu(int64_t n, int64_t num_rounds, at::Tensor &dummy)
    {
        // TODO: Implement the CPU version of the random philox permutation

        // Create a Mersenne Twister random number generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create a tensor with integers from 0 to n-1
        auto permutation = torch::arange(n, torch::kInt32);

        // Fisher-Yates shuffle algorithm
        for (int i = n - 1; i > 0; --i)
        {
            // Generate a random index between 0 and i (inclusive)
            std::uniform_int_distribution<int> dis(0, i);
            int j = dis(gen);

            // Swap elements at positions i and j
            auto temp = permutation[i].item<int>();
            permutation[i] = permutation[j];
            permutation[j] = temp;
        }

        return permutation;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(plas, m)
    {
        m.def("random_philox_permutation(int n, int num_rounds, Tensor dummy) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(plas, CPU, m)
    {
        m.impl("random_philox_permutation", &random_philox_permutation_cpu);
    }
} // namespace plas
