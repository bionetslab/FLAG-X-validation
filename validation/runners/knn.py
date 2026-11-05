"""Runner: sklearn KNeighborsClassifier (tabular baseline; deterministic). Env fx_sklearn."""
from sklearn.neighbors import KNeighborsClassifier

from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn']


def build(params, seed, dataset, n_jobs):
    return KNeighborsClassifier(n_jobs=n_jobs, **params)
