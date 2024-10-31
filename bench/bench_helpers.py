from plas import sort_with_plas
from plas import (
    compute_vad,
    avg_L2_dist_between_neighbors,
    tensor_to_png,
)
import numpy as np
import torch
from easydict import EasyDict

import timeit


def timed_execution(func, *args, num_rounds: int = 1, **kwargs):
    times = []
    for _ in range(num_rounds):
        start_time = timeit.default_timer()  # Start the timer
        result = func(*args, **kwargs)  # Call the function
        end_time = timeit.default_timer()  # End the timer
        execution_time = end_time - start_time
        times.append(execution_time)
    return result, np.mean(times)


def bench_sort_with_plas(grid, config: EasyDict = EasyDict()):
    bench_log = EasyDict()

    (sorted_coords, sorted_grid_indices), time_taken = timed_execution(
        sort_with_plas, grid, **config
    )
    bench_log.time_taken = float(time_taken)

    png_size_sum = 0
    for i in range(int(np.ceil(grid.shape[0] / 3))):
        png_size = len(tensor_to_png(sorted_coords[3*i:min(3*(i+1), grid.shape[0]), :, :]))
        png_size_sum += png_size
    compression_factor = (np.prod(grid.shape) * grid.element_size()) / png_size_sum
    bench_log.compression_factor = float(compression_factor)

    and_unsorted = avg_L2_dist_between_neighbors(grid)
    and_sorted = avg_L2_dist_between_neighbors(sorted_coords)
    bench_log.avg_l2_dist_reduction_factor = float(and_unsorted / and_sorted)

    vad_unsorted = compute_vad(grid)
    vad_sorted = compute_vad(sorted_coords)
    bench_log.vad_reduction_factor = float(vad_unsorted / vad_sorted)

    return bench_log

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    return device