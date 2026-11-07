"""
utils/checkpointing.py
-----------------------
Save and load training checkpoints.

A checkpoint contains:
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - current epoch
  - best metric value
  - training config

This lets training resume exactly where it stopped, which is critical
on free GPU services (Colab / Kaggle) that can disconnect without warning.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from utils.logger import get_logger

log = get_logger(__name__)


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metric: float,
    config: Dict,
    is_best: bool = False,
    filename: str = "checkpoint_last.pt",
) -> None:
    """
    Save a training checkpoint to disk.

    Parameters
    ----------
    save_dir  : Directory to save the checkpoint.
    epoch     : Current epoch number (0-indexed).
    model     : The model (may be wrapped in DataParallel).
    optimizer : The optimiser.
    scheduler : The LR scheduler.
    metric    : Current validation metric (e.g. macro F1).
    config    : Training config dict (for reproducibility).
    is_best   : If True, also save as 'checkpoint_best.pt'.
    filename  : Checkpoint filename.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Unwrap DataParallel before saving
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metric": metric,
        "config": config,
    }

    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    log.info(f"Checkpoint saved → {path}  (epoch={epoch}, metric={metric:.4f})")

    if is_best:
        best_path = os.path.join(save_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        log.info(f"New best checkpoint → {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a checkpoint and restore model/optimizer/scheduler state.

    Parameters
    ----------
    checkpoint_path : Path to the .pt checkpoint file.
    model           : Model to restore weights into.
    optimizer       : Optional — restored if provided.
    scheduler       : Optional — restored if provided.
    device          : Map tensors to this device.

    Returns
    -------
    dict with 'epoch' and 'metric' keys so the caller knows where to resume.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")

    # Unwrap DataParallel wrapper if present
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    log.info(
        f"Resumed from epoch {ckpt['epoch']} | best metric = {ckpt.get('metric', 'N/A'):.4f}"
    )
    return {"epoch": ckpt["epoch"], "metric": ckpt.get("metric", 0.0)}
