"""Runner: sklearn LinearDiscriminantAnalysis (tabular baseline; deterministic). Env fx_sklearn."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn']


def build(params, seed, dataset, n_jobs):
    return LinearDiscriminantAnalysis(**params)
