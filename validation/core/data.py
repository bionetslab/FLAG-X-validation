"""
Loading of the curated data and the splits (outputs of curate_data.py and make_splits.py).

numpy + stdlib only, Python 3.5-compatible (runs in every method env): no f-strings, no variable annotations.

Files (see CLAUDE.md):
- data/curated/<dataset>/sample_<NNN>_X.npy (float32, events x channels in the order of config.CHANNELS[dataset]),
  sample_<NNN>_y.npy (int32, labels 0..K-1), meta_samples.csv, meta_labels.csv
- data/splits/<dataset>.csv (columns split_<i> = train/test), data/splits/imstat_inner.csv (columns
  split_<i>_inner_<j> = train/val/test)
"""

import csv
import os

import numpy as np

import config


def load_sample(dataset, sample):
    """
    Loads one curated sample.

    Args:
        dataset (str): Dataset name, e.g. 'imstat'.
        sample (str): Sample name, e.g. 'sample_001'.

    Returns:
        tuple: (X, y), X float32 array of shape (n_events, n_channels), y int32 array of shape (n_events,).
    """
    d = os.path.join(config.DATA_CURATED, dataset)
    X = np.load(os.path.join(d, sample + '_X.npy'))
    y = np.load(os.path.join(d, sample + '_y.npy'))
    return X, y


def load_meta_samples(dataset):
    """
    Returns the rows of meta_samples.csv as a list of dicts (all values str).
    """
    return _read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_samples.csv'))


def load_meta_labels(dataset):
    """
    Returns the rows of meta_labels.csv as a list of dicts, with 'label' converted to int (sorted by label).
    Column 'label' always contains harmonized integer labels starting from 0.
    """
    rows = _read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_labels.csv'))
    for r in rows:
        r['label'] = int(r['label'])
    return sorted(rows, key=lambda r: r['label'])


def load_labels(dataset):
    """
    Returns the list of all class labels (ints) of the dataset, from meta_labels.csv.
    """
    return [r['label'] for r in load_meta_labels(dataset)]


def load_split(dataset, column, inner=False):
    """
    Returns the sample names per role for one split column.

    Args:
        dataset (str): Dataset name, e.g. 'imstat'.
        column (str): Split column, e.g. 'split_0' (outer split, data/splits/<dataset>.csv) or 'split_0_inner_1'
            (inner split, data/splits/<dataset>_inner.csv, with inner=True).
        inner (bool): Read the inner split file <dataset>_inner.csv instead of <dataset>.csv.

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}, sample names in file order (val empty for outer
        splits).
    """
    fn = dataset + ('_inner.csv' if inner else '.csv')
    rows = _read_csv(os.path.join(config.DATA_SPLITS, fn))
    if rows and column not in rows[0]:
        raise ValueError('column {!r} not in {}'.format(column, fn))
    out = {'train': [], 'val': [], 'test': []}
    for r in rows:
        out[r[column]].append(r['sample'])
    return out


def _read_csv(path):
    """Loads a CSV file and returns the rows as a list of dicts.

    Returns:
        list: [{col1: val_row1, col2: val_row1, ...}, {col1: val_row2, col2: val_row2, ...}, ...]
    """
    with open(path, newline='') as f:
        return list(csv.DictReader(f))
