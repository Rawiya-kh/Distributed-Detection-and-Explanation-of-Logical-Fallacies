"""
utils/device.py
---------------
Automatic hardware detection and device management.

The project is designed to run on CPU-only laptops but automatically
accelerates when a CUDA GPU (or Apple MPS) is detected.  This module
centralises all device logic so every script stays hardware-agnostic.
"""

import torch
from utils.logger import get_logger

log = get_logger(__name__)


def get_device() -> torch.device:
    """
    Detect and return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM) — using CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Apple MPS detected — using MPS")
    else:
        device = torch.device("cpu")
        cpu_count = torch.get_num_threads()
        log.info(f"No GPU found — running on CPU ({cpu_count} threads)")
    return device


def get_num_workers(default: int = 4) -> int:
    """
    Return a safe number of DataLoader workers.

    On Windows, multiprocessing with PyTorch requires `num_workers=0`
    unless scripts are guarded with `if __name__ == '__main__':`.
    We detect the OS and cap accordingly.
    """
    import os
    import platform
    if platform.system() == "Windows":
        # Windows fork-safety: use 0 by default unless user overrides
        return min(default, 0)
    cpu_count = os.cpu_count() or 1
    return min(default, cpu_count)


def can_use_fp16(device: torch.device) -> bool:
    """
    Return True if mixed-precision (FP16) is safe on this device.

    Mixed precision is only beneficial (and safe) on CUDA.  Enabling
    it on CPU or MPS causes errors or slowdowns.
    """
    return device.type == "cuda"


def wrap_model_for_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Wrap a model in DataParallel if multiple GPUs are available.

    Parameters
    ----------
    model  : The PyTorch model.
    device : The primary device.

    Returns
    -------
    Possibly wrapped model.
    """
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        gpu_count = torch.cuda.device_count()
        log.info(f"Multiple GPUs detected ({gpu_count}) — wrapping with DataParallel")
        model = torch.nn.DataParallel(model)
    return model
