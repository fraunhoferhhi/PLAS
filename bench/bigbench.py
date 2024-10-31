import bench_3dgs
import bench_2d_images
import bench_synthetic_data
from easydict import EasyDict
import json

print("Enter the version of PLAS you are benchmarking: ")
version = input()

benchmark_log = EasyDict({"version": version})

# Synthetic data
n_d_pairs = []
for side_length in [10**2, 10**3]:
    for d in [3, 4, 10, 60]:
        n_d_pairs.append((side_length ** 2, d))
benchmark_log.synthetic_data = bench_synthetic_data.bench(n_d_pairs)

# 2D images
#TODO

# 3DGS plys
#TODO

# write benchmark log to json file
with open(f"bench/measurements/bigbench_{version}.json", "w") as f:
    json.dump(benchmark_log, f, indent=4)
