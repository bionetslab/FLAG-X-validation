"""Runner: sklearn RandomForestClassifier (tabular baseline). Env fx_sklearn."""
from sklearn.ensemble import RandomForestClassifier

from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn']


def build(params, seed, dataset, n_jobs):
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
