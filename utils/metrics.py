"""
utils/metrics.py
----------------
Shared metric utilities used during classifier training.

We track Macro F1 as the primary metric because our dataset has
imbalanced classes — macro averaging gives equal weight to every
class regardless of how many samples it has.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from utils.logger import get_logger

log = get_logger(__name__)


def compute_metrics(preds: List[int], labels: List[int], label_names: List[str]) -> Dict:
    """
    Compute classification metrics from raw integer predictions/labels.

    Parameters
    ----------
    preds       : Predicted class indices.
    labels      : True class indices.
    label_names : Human-readable label strings (for reporting).

    Returns
    -------
    dict with keys: accuracy, macro_f1, weighted_f1, report
    """
    accuracy  = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report    = classification_report(labels, preds, target_names=label_names, zero_division=0)

    log.info(f"Accuracy     : {accuracy:.4f}")
    log.info(f"Macro F1     : {macro_f1:.4f}")
    log.info(f"Weighted F1  : {weighted_f1:.4f}")
    log.info(f"\n{report}")

    return {
        "accuracy":    accuracy,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "report":      report,
    }
