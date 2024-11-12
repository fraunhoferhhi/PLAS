from .core import sort_with_plas
from .util import avg_L2_dist_between_neighbors
from .util import tensor_to_png
import torch
from torch import Tensor
from . import _C


### ------ Wrappers for custom torch CUDA/C++ operators ------ ###


def random_philox_permutation(
    n: int, num_rounds: int, dummy: Tensor = torch.randn(1, device="cuda")
) -> Tensor:
    return torch.ops.plas.random_philox_permutation.default(n, num_rounds, dummy)
