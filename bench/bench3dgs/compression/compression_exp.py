import numpy as np
import os
from tqdm import tqdm
from bench3dgs.gaussian_model import GaussianModel

import yaml
from dataclasses import dataclass, asdict
import pandas as pd

from bench3dgs.compression.jpeg_xl import JpegXlCodec
from bench3dgs.compression.npz import NpzCodec
from bench3dgs.compression.exr import EXRCodec
from bench3dgs.compression.png import PNGCodec

codecs = {
    "jpeg-xl": JpegXlCodec,
    "npz": NpzCodec,
    "exr": EXRCodec,
    "png": PNGCodec,
}

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


@dataclass
class QuantEval:
    psnr: float
    ssim: float
    lpips: float


@dataclass
class Measurement:
    name: str
    path: str
    size_bytes: int
    quant_eval: QuantEval = None

    @property
    def human_readable_byte_size(self):
        if self.size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(np.floor(np.log(self.size_bytes) / np.log(1000)))
        p = np.power(1000, i)
        s = round(self.size_bytes / p, 2)
        return f"{s}{size_name[i]}"

    def to_dict(self):
        d = asdict(self)
        d.pop("quant_eval")
        if self.quant_eval is not None:
            d.update(self.quant_eval.__dict__)
        d["size"] = self.human_readable_byte_size
        return d


def log_transform(coords):
    positive = coords > 0
    negative = coords < 0
    zero = coords == 0

    transformed_coords = np.zeros_like(coords)
    transformed_coords[positive] = np.log1p(coords[positive])
    transformed_coords[negative] = -np.log1p(-coords[negative])
    # For zero, no change is needed as transformed_coords is already initialized to zeros

    return transformed_coords


def inverse_log_transform(transformed_coords):
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    zero = transformed_coords == 0

    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    # For zero, no change is needed as original_coords is already initialized to zeros

    return original_coords


def get_attr_numpy(gaussians, attr_name):
    attr_tensor = gaussians.attr_as_grid_img(attr_name)
    attr_numpy = attr_tensor.detach().cpu().numpy()
    return attr_numpy


def compress_attr(attr_config, gaussians, out_folder):
    attr_name = attr_config["name"]
    attr_method = attr_config["method"]
    attr_params = attr_config.get("params", {})

    if not attr_params:
        attr_params = {}

    codec = codecs[attr_method]()
    attr_np = get_attr_numpy(gaussians, attr_name)

    file_name = f"{attr_name}.{codec.file_ending()}"
    out_file = os.path.join(out_folder, file_name)

    if attr_config.get("contract", False):
        # sc = SceneContraction()
        # TODO take the original cuda array
        # attr = torch.tensor(attr_np, device="cuda")
        # attr_contracted = sc(attr)
        # attr_np = attr_contracted.cpu().numpy()
        attr_np = log_transform(attr_np)

    if "quantize" in attr_config:
        quantization = attr_config["quantize"]
        min_val = attr_np.min()
        max_val = attr_np.max()
        val_range = max_val - min_val
        # no division by zero
        if val_range == 0:
            val_range = 1
        attr_np_norm = (attr_np - min_val) / (val_range)
        qpow = 2**quantization
        attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
        attr_np = attr_np_quantized * (val_range) + min_val
        attr_np = attr_np.astype(np.float32)

    if attr_config.get("normalize", False):
        min_val, max_val = codec.encode_with_normalization(
            attr_np, attr_name, out_file, **attr_params
        )
        return file_name, min_val, max_val
    else:
        codec.encode(attr_np, out_file, **attr_params)
        return file_name, None, None


def decompress_attr(gaussians, attr_config, compressed_file, min_val, max_val):
    attr_name = attr_config["name"]
    attr_method = attr_config["method"]

    codec = codecs[attr_method]()

    if attr_config.get("normalize", False):
        decompressed_attr = codec.decode_with_normalization(
            compressed_file, min_val, max_val
        )
    else:
        decompressed_attr = codec.decode(compressed_file)

    if attr_config.get("contract", False):
        decompressed_attr = inverse_log_transform(decompressed_attr)

    # TODO dtype?
    # TODO to device?
    # TODO add grad?
    gaussians.set_attr_from_grid_img(attr_name, decompressed_attr)


def run_single_compression(gaussians, experiment_out_path, experiment_config):
    compressed_min_vals = {}
    compressed_max_vals = {}

    compressed_files = {}

    total_size_bytes = 0

    for attribute in experiment_config["attributes"]:
        compressed_file, min_val, max_mal = compress_attr(
            attribute, gaussians, experiment_out_path
        )
        attr_name = attribute["name"]
        compressed_files[attr_name] = compressed_file
        compressed_min_vals[attr_name] = min_val
        compressed_max_vals[attr_name] = max_mal
        total_size_bytes += os.path.getsize(
            os.path.join(experiment_out_path, compressed_file)
        )

    compr_info = pd.DataFrame(
        [compressed_min_vals, compressed_max_vals, compressed_files],
        index=["min", "max", "file"],
    ).T
    compr_info.to_csv(os.path.join(experiment_out_path, "compression_info.csv"))

    experiment_config["max_sh_degree"] = gaussians.max_sh_degree
    experiment_config["active_sh_degree"] = gaussians.active_sh_degree
    experiment_config["disable_xyz_log_activation"] = (
        gaussians.disable_xyz_log_activation
    )
    with open(
        os.path.join(experiment_out_path, "compression_config.yml"), "w"
    ) as stream:
        yaml.dump(experiment_config, stream)

    return total_size_bytes


def run_compressions(gaussians, out_path, compr_exp_config):

    # TODO some code duplciation with run_experiments / run_roundtrip

    results = {}

    for experiment in compr_exp_config["experiments"]:

        experiment_name = experiment["name"]
        experiment_out_path = os.path.join(out_path, experiment_name)
        os.makedirs(experiment_out_path, exist_ok=True)

        size_bytes = run_single_compression(gaussians, experiment_out_path, experiment)
        results[f"size_bytes/cmpr_{experiment['name']}"] = size_bytes

    return results


def run_single_decompression(compressed_dir):

    compr_info = pd.read_csv(
        os.path.join(compressed_dir, "compression_info.csv"), index_col=0
    )

    with open(os.path.join(compressed_dir, "compression_config.yml"), "r") as stream:
        experiment_config = yaml.safe_load(stream)

    decompressed_gaussians = GaussianModel(
        experiment_config["max_sh_degree"],
        experiment_config["disable_xyz_log_activation"],
    )
    decompressed_gaussians.active_sh_degree = experiment_config["active_sh_degree"]

    for attribute in experiment_config["attributes"]:
        attr_name = attribute["name"]
        # compressed_bytes = compressed_attrs[attr_name]
        compressed_file = os.path.join(
            compressed_dir, compr_info.loc[attr_name, "file"]
        )

        decompress_attr(
            decompressed_gaussians,
            attribute,
            compressed_file,
            compr_info.loc[attr_name, "min"],
            compr_info.loc[attr_name, "max"],
        )

    return decompressed_gaussians


def run_decompressions(compressions_dir):

    for compressed_dir in os.listdir(compressions_dir):
        compressed_dir_path = os.path.join(compressions_dir, compressed_dir)
        if not os.path.isdir(compressed_dir_path):
            continue
        yield os.path.basename(compressed_dir_path), run_single_decompression(
            compressed_dir_path
        )
