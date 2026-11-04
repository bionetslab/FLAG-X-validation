"""
Downsampling of one sample and joining of several samples, reproducible via passed-in seeds.

numpy + stdlib only, Python 3.5-compatible (runs in every method env): no f-strings, no variable annotations.
"""

import numpy as np


def downsample_indices(y, n=None, frac=None, stratified=False, seed=None):
    """
    Indices of the events kept when downsampling one sample.

    Port of get_downsampling_bool (legacy/validation/utils/val_utils.py L750-796) with a local RandomState instead
    of the global numpy seed, and with a fraction as alternative to the number of events. Stratified: per class
    round(n * n_class / n_events) events (at most n_class), then the total is corrected to exactly n by randomly
    dropping selected / adding unselected events (as in the legacy code; a very rare class may get 0 events).

    Args:
        y (np.ndarray): Labels of the sample, shape (n_events,).
        n (int or None): Number of events to keep. Exactly one of n and frac must be given.
        frac (float or None): Fraction of events to keep, n = round(frac * n_events).
        stratified (bool): Keep the class proportions (see above); otherwise a simple random subset.
        seed (int or None): Seed of the np.random.RandomState.

    Returns:
        np.ndarray: Sorted int64 indices into y. All indices if n >= n_events or frac >= 1.
    """
    if (n is None) == (frac is None):
        raise ValueError('give exactly one of n and frac')
    y = np.asarray(y)
    n_events = y.shape[0]
    if frac is not None:
        if frac <= 0:
            raise ValueError('frac must be > 0, got {}'.format(frac))
        n = int(round(frac * n_events))
    n = int(n)
    if n < 0:
        raise ValueError('n must be >= 0, got {}'.format(n))

    # --- All events
    if n >= n_events:
        return np.arange(n_events)

    # --- Random subset
    rng = np.random.RandomState(seed)
    if not stratified:
        return np.sort(rng.choice(n_events, n, replace=False))

    # --- Stratified random subset
    selected = []
    classes, counts = np.unique(y, return_counts=True)
    for c, count in zip(classes, counts):
        n_c = min(int(round(n * (count / float(n_events)))), int(count))
        if n_c > 0:
            selected.append(rng.choice(np.flatnonzero(y == c), n_c, replace=False))
    selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)

    # Correct rounding errors to exactly n events
    if selected.shape[0] > n:
        selected = rng.choice(selected, n, replace=False)
    elif selected.shape[0] < n:
        rest = np.setdiff1d(np.arange(n_events), selected)
        selected = np.concatenate([selected, rng.choice(rest, n - selected.shape[0], replace=False)])
    return np.sort(selected).astype(np.int64)


def join_samples(Xs, ys, sample_ids, shuffle=True, seed=None):
    """
    Concatenates several samples and (optionally) shuffles the events.

    Args:
        Xs (list of np.ndarray): Per-sample data, each of shape (n_events_i, n_channels).
        ys (list of np.ndarray): Per-sample labels, each of shape (n_events_i,).
        sample_ids (list): One id per sample (e.g. the sample names), repeated for its events.
        shuffle (bool): Shuffle the events with a RandomState(seed) permutation.
        seed (int or None): Seed of the permutation.

    Returns:
        tuple: (X, y, sample_ids) with aligned rows; sample_ids is an array of shape (n_events,).
    """
    if not (len(Xs) == len(ys) == len(sample_ids)):
        raise ValueError('Xs, ys and sample_ids must have the same length')
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    ids = np.concatenate([np.repeat(np.asarray([s]), len(yi)) for s, yi in zip(sample_ids, ys)])
    if shuffle:
        perm = np.random.RandomState(seed).permutation(y.shape[0])
        X, y, ids = X[perm], y[perm], ids[perm]
    return X, y, ids
