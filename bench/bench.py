from bench3dgs import bench_3dgs
from bench_helpers import (
    get_device,
    bench_sort_with_plas,
)

import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    benchmark_log = OmegaConf.create({"version": cfg.version})
    print(f'Benchmarking PLAS version {cfg.version} in mode "{cfg.mode}"\n')

    benchmark_3dgs_plys(cfg, benchmark_log)
    benchmark_2d_images(cfg, benchmark_log)
    benchmark_random_data(cfg, benchmark_log)

    with open(f"bench/measurements/{cfg.mode}/bench_{cfg.version}.json", "w") as f:
        json.dump(OmegaConf.to_container(benchmark_log), f, indent=4)


def benchmark_3dgs_plys(cfg: DictConfig, benchmark_log: DictConfig):
    benchmark_log["gaussian_splatting"] = {}
    folder = cfg.compressed_3dgs_folder
    folder = os.path.join(folder, cfg.mode)
    for scene in os.listdir(folder):
        compressed_folder = os.path.join(folder, scene)
        print(f"Benchmarking PLAS on 3DGS scene {scene}")
        benchmark_log.gaussian_splatting[scene] = bench_3dgs.bench(compressed_folder, cfg.plas_config)
        print()


def benchmark_2d_images(cfg: DictConfig, benchmark_log: DictConfig):
    benchmark_log["images"] = OmegaConf.create()
    folder = cfg.image_folder
    folder = os.path.join(folder, cfg.mode)
    for image_name in os.listdir(folder):
        print(f"Benchmarking PLAS on 2D image {image_name}")
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path).convert("RGB")
        transform = transforms.ToTensor()
        image_tensor = transform(image).to(get_device())
        benchmark_log.images[image_name] = bench_sort_with_plas(image_tensor, cfg.plas_config)
        print()


def benchmark_random_data(cfg: DictConfig, benchmark_log: DictConfig):
    benchmark_log["random_data"] = OmegaConf.create()
    for side_length in cfg.random_data.side_lengths[cfg.mode]:
        for d in cfg.random_data.dimensions[cfg.mode]:
            print(
                f"Benchmarking PLAS on random data with side length {side_length} and depth {d}"
            )
            bench_log = OmegaConf.create()
            logs = []
            for _ in range(cfg.random_data.samples):
                grid = torch.randn(d, side_length, side_length, device=get_device())
                logs.append(bench_sort_with_plas(grid, cfg.plas_config))
            benchmark_log.random_data[f"{side_length}x{side_length}x{d}"] = {
                key: float(np.mean([log[key] for log in logs]))
                for key in logs[0].keys()
            }
            print()


if __name__ == "__main__":
    main()
