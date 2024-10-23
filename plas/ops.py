import torch
from torch import Tensor

__all__ = ["random_philox_bijection"]

def random_philox_bijection(n: int, num_rounds: int, dummy: Tensor) -> Tensor:
    return torch.ops.plas.random_philox_bijection.default(n, num_rounds, dummy)
