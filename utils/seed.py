"""
utils/seed.py
-------------
Reproducibility helpers.

Sets random seeds across Python, NumPy, PyTorch, and CUDA so that
results are deterministic when the same seed is used.

NOTE: Full determinism on GPU requires additional CUBLAS / cuDNN env
vars and may slow training.  We enable the lighter "best-effort"
determinism that is good enough for research reproducibility.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        The random seed (default 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Best-effort determinism — may slightly reduce GPU throughput
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
