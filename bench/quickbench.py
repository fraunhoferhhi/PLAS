from bench3dgs import bench_3dgs

import bench_synthetic_data
import bench_2d_images

from easydict import EasyDict
import json

print("Enter the version of PLAS you are benchmarking: ")
version = input()

benchmark_log = EasyDict({"version": version})

# Synthetic data
n_d_pairs = []
for side_length in [3 * 10**2]:
    for d in [3, 9, 15, 21]:
        n_d_pairs.append((side_length ** 2, d))
benchmark_log.synthetic_data = bench_synthetic_data.bench(n_d_pairs)

# 2D images
# TODO

# 3DGS plys
benchmark_log.gaussian_splatting = EasyDict()
data_folder = "/home/helms/Data/compressed_3d_scenes_with_sogs/"
scenes = ["360_bicycle", "360_kitchen", "blender_lego", "db_playroom", "tandt_truck"]
for scene in scenes:
    print(f"Benchmarking PLAS on {scene}")
    compressed_folder = data_folder + scene
    benchmark_log.gaussian_splatting[scene] = bench_3dgs.bench(compressed_folder)

with open(f"bench/measurements/quickbench_{version}.json", "w") as f:
    json.dump(benchmark_log, f, indent=4)
