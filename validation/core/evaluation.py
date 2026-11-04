"""
Per-sample evaluation counts: confusion matrix incl. abstentions and the derived per-class counts.

numpy only, Python 3.5-compatible (runs in every method env): no f-strings, no variable annotations.
Scores are computed later from these counts (PLAN.md step 9, scoring rule D8).
"""

import numpy as np


ABSTENTION = -1


def confusion_counts(y_true, y_pred, labels):
    """
    Confusion matrix with an extra column for abstentions.

    Args:
        y_true (np.ndarray): True labels, shape (n_events,); every value must be in labels.
        y_pred (np.ndarray): Predicted labels, shape (n_events,); values in labels or -1 (abstention).
        labels (list of int): All classes of the dataset (meta_labels.csv), also those absent from the sample, so
            every sample gives the same shape.

    Returns:
        np.ndarray: int64 array of shape (K, K + 1); [i, j] = events of true class labels[i] predicted as labels[j],
        [i, K] = events of true class labels[i] with abstention (-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    labels = [int(l) for l in labels]
    k = len(labels)
    index = dict((l, i) for i, l in enumerate(labels))  # Map labels to consecutive integers
    index[ABSTENTION] = k

    unknown_true = np.setdiff1d(np.unique(y_true), labels)
    if unknown_true.size:
        raise ValueError('y_true contains labels not in labels: {}'.format(unknown_true.tolist()))
    unknown_pred = np.setdiff1d(np.unique(y_pred), labels + [ABSTENTION])
    if unknown_pred.size:
        raise ValueError('y_pred contains labels not in labels or -1: {}'.format(unknown_pred.tolist()))

    t = _map(y_true, index)  # Map ground truth labels to int labels
    p = _map(y_pred, index)  # Map predicted labels to int labels
    # Build a confusion matrix of shape (k, k + 1): axis 0 (rows) represents the k true classes,
    # axis 1 (columns) the k predicted classes plus abstention (p = k).
    # Encode each (true, predicted) pair (t, p) as its row-major flat index
    # t * (k + 1) + p. np.bincount counts occurrences of each index; minlength
    # ensures zero counts for absent pairs and allows reshaping to (k, k + 1).
    # Thus, cm[t, p] counts events with true class t and prediction p.
    cm = np.bincount(t * (k + 1) + p, minlength=k * (k + 1)).reshape(k, k + 1)
    return cm.astype(np.int64)


def class_counts(cm):
    """
    Per-class counts derived from a confusion matrix of confusion_counts.

    Scoring rule D8: an abstained event counts as FN for its true class and as FP for no class.

    Args:
        cm (np.ndarray): Array of shape (K, K + 1) from confusion_counts.

    Returns:
        dict of np.ndarray (each of shape (K,), int64): 'n_true' (events of the class), 'n_pred' (events predicted
        as the class), 'n_abstained' (events of the class with abstention), 'tp', 'fp', 'fn', 'tn'.
    """
    cm = np.asarray(cm, dtype=np.int64)
    k = cm.shape[0]  # Number of classes
    n = cm.sum()  # Total number of events
    tp = np.diag(cm[:, :k])  # TPs are on diagonal (abstention column excluded)
    n_true = cm.sum(axis=1)  # Number of events belonging to each class
    n_pred = cm[:, :k].sum(axis=0)  # Number of events predicted per class (abstention column excluded)
    fn = n_true - tp  # FN per class
    fp = n_pred - tp  # FP per class
    tn = n - tp - fp - fn  # TN per class
    n_abstained = cm[:, k].copy()  # Number of abstentions per class
    return {
        'n_true': n_true, 'n_pred': n_pred, 'n_abstained': n_abstained, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def _map(y, index):
    # Map label values to row/column indices
    values = np.unique(y)
    out = np.empty(y.shape[0], dtype=np.int64)
    for v in values:
        out[y == v] = index[int(v)]
    return out
