import plas
import torch
import unittest


def is_permutation(permutation: torch.Tensor) -> torch.Tensor:
    sorted = torch.sort(permutation)[0]
    expected = torch.arange(permutation.size(0), device=permutation.device)
    return torch.all(sorted == expected)


class TestIsValidPermutation(unittest.TestCase):
    def test_cuda_one_phase(self):
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 65, 1024, 1025, 2048]:
            dummy = torch.randn(n, device="cuda")
            result = plas.random_philox_permutation(n=n, num_rounds=5)
            self.assertTrue(is_permutation(result))

    def test_cuda_two_phases(self):
        for n in [2049, 4097, 10_000, 50_000, 100_000, 1_000_000, 2**22]:
            dummy = torch.randn(n, device="cuda")
            result = plas.random_philox_permutation(n=n, num_rounds=5)
            self.assertTrue(is_permutation(result))

    def test_cuda_three_phases(self):
        for n in [2**22 + 1, 2**23]:
            dummy = torch.randn(n, device="cuda")
            result = plas.random_philox_permutation(n=n, num_rounds=5)
            self.assertTrue(is_permutation(result))

    def test_cpu(self):
        for n in [1, 2, 3, 4, 5, 10, 100, 1000, 10_000, 100_000, 1_000_000]:
            dummy = torch.randn(n, device="cpu")
            result = plas.random_philox_permutation(n=n, num_rounds=5)
            self.assertTrue(is_permutation(result))
