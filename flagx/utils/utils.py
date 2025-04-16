

import random
import torch
import numpy as np
import scanpy as sc

from .._legacy_typing import Dict


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)  # Sets the seed for generating random numbers on all devices.
    torch.cuda.manual_seed_all(seed)  # Set the seed for generating random numbers on all GPUs.
    # torch.cuda.manual_seed(seed)  # Set the seed for generating random numbers for the current GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def log10_trafo_w_cutoff_channel_wise(
        adata: sc.AnnData,
        channel_to_cutoff_dict: Dict[str, int],
):
    x = adata.X.copy()
    for channel, cutoff in channel_to_cutoff_dict.items():
        col_idx = np.where(adata.var_names == channel)[0][0]
        x_col = adata.X[:, col_idx].copy()
        mask = (x_col > cutoff)
        x_col[mask] = np.log10(x_col[mask])
        x_col[~mask] = np.log10(cutoff)
        x[:, col_idx] = x_col

    adata.X = x








