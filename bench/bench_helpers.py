from plas import sort_with_plas
from plas import (
    avg_L2_dist_between_neighbors,
    tensor_to_png,
)
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import timeit


def bench_sort_with_plas(grid, config: DictConfig = OmegaConf.create()):
    bench_log = OmegaConf.create()

    start_time = timeit.default_timer()
    sorted_coords, sorted_grid_indices = sort_with_plas(grid, **config)
    end_time = timeit.default_timer()
    time_taken = end_time - start_time
    bench_log.duration = float(time_taken)

    png_size_sum = 0
    for i in range(int(np.ceil(grid.shape[0] / 3))):
        png_size = len(
            tensor_to_png(sorted_coords[3 * i : min(3 * (i + 1), grid.shape[0]), :, :])
        )
        png_size_sum += png_size
    compression_factor = (np.prod(grid.shape) * grid.element_size()) / png_size_sum
    bench_log.png_compression_factor = float(compression_factor)

    and_unsorted = avg_L2_dist_between_neighbors(grid)
    and_sorted = avg_L2_dist_between_neighbors(sorted_coords)
    bench_log.avg_l2_dist_reduction_factor = float(and_unsorted / and_sorted)

    return bench_log


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    return device
