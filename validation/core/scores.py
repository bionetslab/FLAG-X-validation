"""
Sample-wise scores from the per-class evaluation counts of one sample (evaluation.class_counts).

numpy only, Python 3.5-compatible (runs in every method env): no f-strings, no variable annotations.
Scoring rule D8 is already contained in the counts: an abstained event is an FN for its true class and an FP for no
class, so no extra handling is needed here.
"""

import numpy as np


def f1_from_counts(tp, fp, fn, binary=False):
    """
    F1 score of one sample from its per-class counts.

    Args:
        tp, fp, fn (array-like): Per-class counts of one sample, shape (K,), in label order (meta_labels.csv).
        binary (bool): If True (datasets with 2 labels), the F1 of the positive class (label 1, i.e. index 1);
            otherwise the macro F1.

    Returns:
        float: Macro F1 over the classes that occur in the sample or in its prediction (tp + fp + fn > 0, i.e.
        n_true + n_pred > 0; the sklearn / legacy convention, legacy/validation/utils/val_utils.py:53), or the F1 of
        label 1. NaN if no class (binary: label 1) occurs in the sample or its prediction.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    denom = 2 * tp + fp + fn
    if binary:
        if tp.shape[0] != 2:
            raise ValueError('binary=True needs exactly 2 classes, got {}'.format(tp.shape[0]))
        return float(2 * tp[1] / denom[1]) if denom[1] > 0 else float('nan')
    present = denom > 0
    if not present.any():
        return float('nan')
    return float(np.mean(2 * tp[present] / denom[present]))


def abstention_rate(n_abstained, n_true):
    """Share of the sample's events with abstention (-1): sum(n_abstained) / sum(n_true); NaN for an empty sample."""
    n = float(np.sum(n_true))
    return float(np.sum(n_abstained)) / n if n > 0 else float('nan')
