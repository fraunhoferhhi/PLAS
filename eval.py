import cv2
import functools
import kornia
import numpy as np
import random
import sys
import time
import torch
import torchvision

from flas import (
    generate_random_colors,
    sort_with_flas,
    distance_preservation_quality,
)

from plas import sort_with_plas

from vad import compute_vad

# DEBUG: show dist maps?

# TODO perf
# - try randperm and access instead of shuffle?

# TODO
# - SOM as target
# - max swap radius?

# EXPERIMENTS offset stuff
# - tried kernel_size -= 2 every round of all offsets, result was worse DPQ, and took much longer
# - tried random shuffle of offsets every round, result was worse
# -- with all offsets, it can be slightly better
# - gauss filter with size / 2 is better than gauss filter with size / 3
# - box filter of same size is a little better than gauss filter

# EXPERIMENTS 1D
# - taking only half the elements or a quarter is *slower* than taking all (random) elements, and DPQ is much worse
# - taking only elements with distance over mean is *slower* than taking all (random) elements, and DPQ worse

SHOW_VIS = True

COMPUTE_SOM_PSSM = False
COMPUTE_PSSM = False
COMPUTE_PFLAS = False
COMPUTE_BLOCKY = True
COMPUTE_FLAS = True


def imshow_torch(name, img):
    if not SHOW_VIS:
        return

    img_cv2 = img.permute(1, 2, 0).to(torch.uint8).cpu().numpy()[..., ::-1]
    imshow_cv2(name, img_cv2)


def imshow_cv2(name, img):
    if not SHOW_VIS:
        return

    if img.shape[0] < 512:
        img = cv2.resize(
            img,
            (512, 512),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow(name, img)


@functools.cache
def create_indices(sidelength, target_device):
    device = torch.device("cpu")

    indices = torch.arange(sidelength, device=device)
    y_coords, x_coords = torch.meshgrid(indices, indices, indexing="ij")

    coordinates = torch.stack((y_coords, x_coords), dim=-1)
    coordinates_flat = coordinates.reshape(-1, 2)

    return coordinates_flat.to(target_device)


def blur_img_gauss(img, kernel_size):
    blur_kernel_2d = torchvision.transforms.GaussianBlur(
        kernel_size=(kernel_size, kernel_size),
        sigma=(kernel_size / 2.0, kernel_size / 2.0),
    )
    blurred_img = blur_kernel_2d(img.unsqueeze(0)).squeeze(0)
    return blurred_img


def blur_img(img, kernel_size):
    img = kornia.filters.box_blur(
        img.unsqueeze(0), kernel_size=(kernel_size, kernel_size)
    ).squeeze(0)
    return img


def debug_show_candidates(cand_coords_np):
    if not SHOW_VIS:
        return

    np.random.seed(42)

    # cand_coords_np: (samples, pqrs=4, yx=2)
    num_candidates = cand_coords_np.shape[0]

    # (samples, _, rgb=3)
    cand_colors = np.random.uniform(0, 255, size=(num_candidates, 1, 3)).astype(int)

    # (height, width, rgb=3)
    cand_img = np.zeros(
        (cand_coords_np[..., 0].max() + 1, cand_coords_np[..., 1].max() + 1, 3),
        dtype=np.uint8,
    )

    cand_img[cand_coords_np[..., 0], cand_coords_np[..., 1]] = cand_colors

    imshow_cv2("cand_img", cand_img[..., ::-1])
    cv2.waitKey(1)


def shuffle_candidates(cand_coords, device):
    shuffled_indices = torch.randperm(cand_coords.shape[0], device=device)

    cand_coords_shuffled = cand_coords[shuffled_indices]

    # drop last elements if not divisible by 4
    cand_coords_shuffled = cand_coords_shuffled[
        : cand_coords_shuffled.shape[0] // 4 * 4
    ]

    cand_coords_shuffled = cand_coords_shuffled.reshape(-1, 4)

    return cand_coords_shuffled


@functools.cache
def coord_range(length, device):
    return torch.arange(length, device=device)


def distance(a, b):
    return torch.pow(a - b, 2).sum(dim=0)


def squared_l2_distance(q, p):
    return distance(q.unsqueeze(-1), p.unsqueeze(-2))


def squared_l2_distance_flas(q, p):
    """Faster on macOS CPU, much slower on CUDA"""
    ps = torch.sum(p * p, dim=0)
    qs = torch.sum(q * q, dim=0)

    distance = (
        ps.unsqueeze(-1) - 2 * torch.einsum("cbn,cbm->bnm", p, q) + qs.unsqueeze(-2)
    )

    distance.clamp_(min=0)
    return distance


def reorder_params(params, target, n_iters, filter_by_dist, device):
    target_flat = target.flatten(start_dim=1)

    # params: (channels, height, width)

    # (channels, elements=height * width)
    params_flat = params.flatten(start_dim=1)

    all_cand_coords = coord_range(params_flat.shape[-1], device)

    if filter_by_dist:
        current_dist = distance(params_flat, target_flat)
        current_dist_mean = current_dist.mean().item()

        # cand_coords where items have distance over mean
        # (samples, pqrs=4)
        selected_coords = current_dist > (current_dist_mean / 2)

        # on MPS this has the same performance, so the selection isn't doing anything smarter
        # selected_coords_idx = selected_coords.nonzero().flatten()
        # selected_cand_coords = all_cand_coords[selected_coords_idx]

        selected_cand_coords = all_cand_coords[selected_coords]

    else:
        selected_cand_coords = all_cand_coords

    for loop in range(n_iters):
        # pqrs are the four candidate positions

        # debug_show_candidates(valid_cand_coords.cpu().numpy())

        # (channels, samples, pqrs=4)
        cand_coords = shuffle_candidates(selected_cand_coords, device)

        # use half of the candidates
        # cand_coords = cand_coords[: params_flat.shape[-1] // 4 // 2]

        # the values of the candidates are ABCD at positions pqrs

        # (channels, samples, ABCD=4)
        cand_values = params_flat[:, cand_coords]

        # (channels, samples, pqrs=4)
        target_cand_values = target_flat[:, cand_coords]

        # Compute the squared differences
        # Sum along the channels dimension to get the squared Euclidean distances
        # (channels, samples, pqrs=4, ABCD=4)
        cand_dists = squared_l2_distance(target_cand_values, cand_values)

        # # permutations of assigning 4 elements to 4 positions
        # # perms: (perms=24, 4)
        # # perms_one_hot: (perms=24, 4, 4)
        # perms, perms_one_hot = get_permutations(device)

        # # (samples, perms=24)
        # perms_dist = torch.einsum("sce,pce->sp", cand_dists, perms_one_hot)

        # # (samples) [0...23]
        # best_perms = torch.argmin(perms_dist, dim=-1)

        # # (samples, 4) [0...3]
        # best_perms_idx = perms[best_perms]

        best_perms_idx = solve_assignments(cand_dists, device)

        # Use gather to index into cand_coords along the pqrs dimension
        # (samples, pqrs=4)
        best_coords = torch.gather(cand_coords, 1, best_perms_idx)

        # Assuming best_coords and cand_coords are already defined
        # Get the values from params at the best_coords positions
        # (channels, samples, ABCD=4)
        best_values_flat = params_flat[:, best_coords]

        # Use advanced indexing to set the values in params
        params_flat[:, cand_coords] = best_values_flat

    return params


def sort_with_pssm(params, som_target=None, device="cpu"):
    torch.manual_seed(42)

    size = params.shape[-1]

    radius = size // 2

    # max qual, 64x64 @ DPQ 0.952, 8.24 s
    # kernel_size_fn = lambda radius: radius * 1.5 // 2 * 2 + 1
    # loop_fn = lambda radius: 3 * size
    # radius_decay_fn = lambda radius: radius * 0.95

    # faster, 64x64 @ DPQ 0.946, 3.74 s
    kernel_size_fn = lambda radius: radius * 1.5 // 2 * 2 + 1
    loop_fn = lambda radius: 2 * size
    radius_decay_fn = lambda radius: radius * 0.9

    prev_params = params.clone()

    start_time = time.time()

    with torch.inference_mode():
        while True:
            radius = radius_decay_fn(radius)
            kernel_size = kernel_size_fn(radius)

            if som_target is not None:
                target = som_target
                som_target = None
            else:
                target = blur_img(params, kernel_size=kernel_size)
            imshow_torch("target", target)

            params = reorder_params(
                params,
                target,
                n_iters=loop_fn(radius),
                filter_by_dist=True,
                device=device,
            )

            imshow_torch("params", params)
            if SHOW_VIS and cv2.waitKey(1) == ord("q"):
                print(f"Quit manually after {time.time() - start_time:.2f}s")
                break

            num_el_changed = (~torch.eq(params, prev_params).all(dim=0)).sum()
            percent_changed = (
                num_el_changed.item() / params.shape[-2] / params.shape[-1] * 100
            )
            print(
                f"changed={percent_changed:.3f} % - {num_el_changed.item()} elements @ radius={radius:.2f}"
            )
            if radius <= 1 or num_el_changed == 0:
                print(f"Finished after {time.time() - start_time:.2f}s")
                break

            prev_params = params.clone()

    return params


def blocky_perf(device):

    import pandas as pd

    df = pd.DataFrame(columns=["i", "size", "duration", "blocky_var"])

    for i in [4, 4, 4]:

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        size = 2**i

        params_np = generate_random_colors(size, size)

        # (channels, height, width)
        params = torch.from_numpy(params_np).permute(2, 0, 1).float().to(device)
        assert params.shape[1] == params.shape[2]

        blocky_params, blocky_indices, duration = sort_with_plas(
            params.clone(), improvement_break=1e-4, verbose=False
        )

        assert (
            params.to(torch.int64).sum().item()
            == blocky_params.to(torch.int64).sum().item()
        )

        blocky_var = compute_vad(blocky_params.permute(1, 2, 0).cpu().numpy())

        print(f"{i=} {size=} {duration=:.2f} {blocky_var=:.4f}")

        df.loc[df.shape[0]] = [i, size, duration, blocky_var]

    print(df)
    df.to_csv("/tmp/blocky_perf_4.csv", index=False)

    sys.exit(0)


def blocky_vad(params):

    steps = 24

    import pandas as pd

    df = pd.DataFrame(columns=["ib_log", "imp", "duration", "blocky_var"])

    # warmup
    blocky_params, blocky_indices, duration = sort_with_plas(
        params.clone(), improvement_break=1e-2, verbose=False
    )

    # for i in range(steps):
    #     ib_log = - ((i / steps) * 8)

    for ib_log in [-2, -2.5, -3, -3.5, -4, -4.5, -5, -5.5, -6, -6.5, -7]:

        imp = 10**ib_log

        blocky_params, blocky_indices, duration = sort_with_plas(
            params.clone(), improvement_break=imp, verbose=False
        )

        assert (
            params.to(torch.int64).sum().item()
            == blocky_params.to(torch.int64).sum().item()
        )

        blocky_var = compute_vad(blocky_params.permute(1, 2, 0).cpu().numpy())

        print(f"{ib_log=} {imp=} {duration=:.2f} {blocky_var=:.4f}")

        df.loc[df.shape[0]] = [ib_log, imp, duration, blocky_var]

    print(df)
    df.to_csv("/tmp/blocky_vad.csv", index=False)

    import sys

    sys.exit(0)


def pssm():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    # device = "cpu"

    print(f"Using {device=}")

    # blocky_perf(device)

    if len(sys.argv) < 2:
        size = 256
    else:
        size = int(sys.argv[1])

    print(f"Size: {size} x {size}")

    params_np = generate_random_colors(size, size)

    # (channels, height, width)
    params = torch.from_numpy(params_np).permute(2, 0, 1).float().to(device)
    assert params.shape[1] == params.shape[2]

    # blocky_vad(params)

    org_params = params.clone()

    cv2.imwrite(
        f"/tmp/{size}_orig_pssm.png",
        params.permute(1, 2, 0).to(torch.uint8).cpu().numpy()[..., ::-1],
    )

    # --------------------------------------------
    # BLOCKY-SSM
    if COMPUTE_BLOCKY:
        blocky_params, blocky_indices = sort_with_plas(
            org_params.clone(), improvement_break=1e-4, verbose=True
        )
        imshow_torch(
            "blocky_sorted",
            blocky_params,
        )
        if SHOW_VIS:
            cv2.waitKey(20)

        assert (
            org_params.to(torch.int64).sum().item()
            == blocky_params.to(torch.int64).sum().item()
        )

        cv2.imwrite(
            f"/tmp/{size}_sorted_blocky.png",
            blocky_params.permute(1, 2, 0).to(torch.uint8).cpu().numpy()[..., ::-1],
        )

    # --------------------------------------------
    # FLAS
    if COMPUTE_FLAS:
        flas_sorted, flas_time = sort_with_flas(
            org_params.permute(1, 2, 0).cpu().numpy(),
            nc=100,
            radius_factor=0.9,
            return_time=True,
            randomize_X=False,
        )
        imshow_cv2(
            "flas_sorted",
            flas_sorted.astype(np.uint8)[..., ::-1],
        )
        if SHOW_VIS:
            cv2.waitKey(20)

        cv2.imwrite(f"/tmp/{size}_sorted_flas.png", flas_sorted.astype(np.uint8))

        print(f"FLAS took {flas_time:.2f}s")

    else:
        flas_sorted = params.permute(1, 2, 0).cpu().numpy()

    # --------------------------------------------
    # DPQ
    if size <= 64:
        if COMPUTE_BLOCKY:
            blocky_dist = distance_preservation_quality(
                blocky_params.permute(1, 2, 0).cpu().numpy(), p=16
            )
            print(f"BLOCKY DPQ: {blocky_dist:.4f}", end=", ")

        if COMPUTE_FLAS:
            flas_dist = distance_preservation_quality(flas_sorted, p=16)
            print(f"FLAS DPQ: {flas_dist:.4f}", end=", ")

        print("")

    if COMPUTE_BLOCKY:
        blocky_var = compute_vad(blocky_params.permute(1, 2, 0).cpu().numpy())
        print(f"BLOCKY var: {blocky_var:.4f}", end=", ")

    if COMPUTE_FLAS:
        flas_var = compute_vad(flas_sorted)
        print(f"FLAS var: {flas_var:.4f}", end=", ")

    org_var = compute_vad(params_np)
    print(f"Org var: {org_var:.4f}", end=", ")

    if SHOW_VIS:
        cv2.waitKey(0)


if __name__ == "__main__":
    pssm()
