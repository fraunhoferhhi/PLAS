import torch
import numpy as np
import random
import pandas as pd

from vad import compute_vad


def blocky_perf(device):

    df = pd.DataFrame(columns=["i", "size", "duration", "blocky_var"])

    for i in range(4, 14):

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        size = 2**i

        params_np = generate_random_colors(size, size)

        # (channels, height, width)
        params = torch.from_numpy(params_np).permute(2, 0, 1).float().to(device)
        assert params.shape[1] == params.shape[2]

        blocky_params, blocky_indices, duration = sort_with_blocky(
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


if __name__ == "__main__":
    device = torch.device("cuda")
    blocky_perf(device)
    device = torch.device("cpu")
    blocky_perf(device)
    print("Done")
