import os
import yaml
from bench3dgs.compression.compression_exp import (
    run_single_compression,
    run_single_decompression,
)
from bench3dgs.gaussian_model import GaussianModel
from easydict import EasyDict as edict
from omegaconf import DictConfig
compression_cfg = edict({
    "enabled": True,
    "normalize": True,
    "activated": True,
    "shuffle": True,
    "weights": {
        "xyz": 1.0,
        "features_dc": 1.0,
        "features_rest": 0.0,
        "opacity": 0.0,
        "scaling": 1.0,
        "rotation": 0.0
    }
})

def bench(compressed_folder, plas_config: DictConfig):
    gaussians = run_single_decompression(compressed_folder)
    gaussians.prune_to_square_shape(sort_by_opacity=True, verbose=False)
    
    # Calculate uncompressed size in bytes
    uncompressed_size = 0
    attributes = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']
    for attr in attributes:
        tensor = getattr(gaussians, attr)
        uncompressed_size += tensor.numel() * tensor.element_size()
    
    # Sort with PLAS and measure time
    duration = gaussians.sort_into_grid(compression_cfg, plas_config)

    # Compress again and measure size
    with open(os.path.join(compressed_folder, "compression_config.yml"), "r") as stream:
        experiment_config = yaml.safe_load(stream)
    compressed_size = run_single_compression(gaussians, experiment_out_path=compressed_folder, experiment_config=experiment_config)

    compression_factor = uncompressed_size / compressed_size

    benchmark_log = {
        "duration": duration,
        "compression_factor": compression_factor
    }
    return benchmark_log