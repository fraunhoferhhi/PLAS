# A script to sort Gaussians from a .ply file with PLAS.

import numpy as np
import pandas as pd
import sys
import torch
import trimesh as tm
from plyfile import PlyData, PlyElement

from point_filter import filter_points
from blockyssm import sort_with_blocky

from torch.profiler import profile, record_function, ProfilerActivity


SHUFFLE_INPUT = True

# process fewer elements for development testing
# DEBUG_TRUNCATE_ELEMENTS = 1_000_000
DEBUG_TRUNCATE_ELEMENTS = None

COORDS_SCALE = 255
RGB_SCALE = 255

C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5

def df_to_ply(df, ply_file):
    # TODO code duplicaten jxl_to_ply

    ply_columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
       'f_rest_0', 'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4', 'f_rest_5',
       'f_rest_6', 'f_rest_7', 'f_rest_8', 'f_rest_9', 'f_rest_10',
       'f_rest_11', 'f_rest_12', 'f_rest_13', 'f_rest_14', 'f_rest_15',
       'f_rest_16', 'f_rest_17', 'f_rest_18', 'f_rest_19', 'f_rest_20',
       'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24', 'f_rest_25',
       'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29', 'f_rest_30',
       'f_rest_31', 'f_rest_32', 'f_rest_33', 'f_rest_34', 'f_rest_35',
       'f_rest_36', 'f_rest_37', 'f_rest_38', 'f_rest_39', 'f_rest_40',
       'f_rest_41', 'f_rest_42', 'f_rest_43', 'f_rest_44', 'opacity',
       'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    df_ply = df[ply_columns]

    df_ply = df_ply.astype(np.float32)
    vertex_el = PlyElement.describe(df_ply.to_records(index=False), "vertex")
    ply_data = PlyData([vertex_el], text=False)
    ply_data.write(ply_file)


def df_to_rgb_ply(df, ply_file):

    xyz = df[['x', 'y', 'z']].values

    dc_vals = df.loc[:, df.columns.str.startswith("f_dc")].values
    rgb = np.clip(SH2RGB(dc_vals), 0, 1)

    pcl = tm.PointCloud(xyz, colors=rgb)
    pcl.export(ply_file)


def data_from_ply(ply_file, min_block_size):
    gaussian_ply = tm.load(ply_file)
    ply_data = gaussian_ply.metadata["_ply_raw"]["vertex"]["data"]
    df = pd.DataFrame(ply_data)
    # df.describe().drop("count")

    if DEBUG_TRUNCATE_ELEMENTS is not None:
        df = df.iloc[:DEBUG_TRUNCATE_ELEMENTS]

    num_gaussians = len(df)

    sidelen = int(np.sqrt(num_gaussians))

    sidelen = sidelen // min_block_size * min_block_size

    df = filter_points(df, sidelen * sidelen)

    return df, sidelen

def sort_gaussians(in_ply_file, out_ply_file, out_rgb_point_cloud_ply_file):

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    print(f"Using device: {device}")

    min_block_size = 16

    df, sidelen = data_from_ply(in_ply_file, min_block_size)

    if SHUFFLE_INPUT:
        df = df.sample(frac=1)

    # coords
    coords_xyz = df[["x", "y", "z"]].values
    coords_xyz_min = coords_xyz.min()
    coords_xyz_range = coords_xyz.max() - coords_xyz_min
    coords_xyz_norm = (coords_xyz - coords_xyz_min) / coords_xyz_range
    coords_xyz_norm *= COORDS_SCALE
    coords_torch = torch.from_numpy(coords_xyz_norm).float().to(device)

    # colors
    dc_vals = df.loc[:, df.columns.str.startswith("f_dc")].values
    rgb = np.clip(SH2RGB(dc_vals), 0, 1) * RGB_SCALE
    rgb_torch = torch.from_numpy(rgb).float().to(device)

    params = torch.cat([coords_torch, rgb_torch], dim=1)

    params_torch_grid = params.permute(1, 0).reshape(-1, sidelen, sidelen)

    sorted_coords, sorted_grid_indices, duration = sort_with_blocky(params_torch_grid, min_block_size, improvement_break=1e-3)

    sorted_indices = sorted_grid_indices.flatten().cpu().numpy()

    sorted_df = df.iloc[sorted_indices]
    
    # full gaussians ply
    df_to_ply(sorted_df, out_ply_file)

    # xyz + rgb colors point cloud ply
    df_to_rgb_ply(sorted_df, out_rgb_point_cloud_ply_file)


if __name__ == "__main__":
    in_ply_file = sys.argv[1]
    out_gaussian_ply_file = sys.argv[2]
    out_rgb_point_cloud_ply_file = sys.argv[3]

    # this only produces one CUDA conv3d?!
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     with_stack=True,
    #     experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    # ) as prof:

    sort_gaussians(in_ply_file, out_gaussian_ply_file, out_rgb_point_cloud_ply_file)

    # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))


