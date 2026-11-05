"""Runner: sklearn HistGradientBoostingClassifier (tabular baseline). Env fx_sklearn."""
from sklearn.ensemble import HistGradientBoostingClassifier

from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn']


def build(params, seed, dataset, n_jobs):
    # Threads via OpenMP (OMP_NUM_THREADS, set in the job scripts)
    return HistGradientBoostingClassifier(random_state=seed, **params)
