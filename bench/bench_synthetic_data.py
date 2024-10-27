import torch
import math
import numpy as np
from easydict import EasyDict
from plas import sort_with_plas
from bench_helpers import bench_sort_with_plas, get_device
import tqdm

def bench(n_d_pairs: list[tuple[int, int]], config: EasyDict = EasyDict(), samples=1):
    
    bench_log = EasyDict()
    for n, d in tqdm.tqdm(n_d_pairs, desc="Benchmarking on synthetic data"):
        logs = []
        for _ in range(samples):
            grid = torch.randn(d, int(math.sqrt(n)), int(math.sqrt(n)), device=get_device())
            logs.append(bench_sort_with_plas(grid, config))
        bench_log[f"n={n}_d={d}"] = {
            key: float(np.mean([log[key] for log in logs])) for key in logs[0].keys()
        }
    return bench_log
