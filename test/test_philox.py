import torch
import plas.ops as ops

def test_philox():
    n = 16
    num_rounds = 20

    # Test on CUDA
    dummy_cuda = torch.randn(1, device='cuda')
    permutation_cuda = ops.random_philox_bijection(n, num_rounds, dummy_cuda)
    assert permutation_cuda.shape == torch.Size([n])
    assert torch.all(permutation_cuda.sort().values == torch.arange(n, device='cuda'))
    print("CUDA permutation:", permutation_cuda)

    # Test on CPU
    dummy_cpu = torch.randn(1, device='cpu')
    permutation_cpu = ops.random_philox_bijection(n, num_rounds, dummy_cpu)
    assert permutation_cpu.shape == torch.Size([n])
    assert torch.all(permutation_cpu.sort().values == torch.arange(n, device='cpu'))
    print("CPU permutation:", permutation_cpu)

test_philox()
