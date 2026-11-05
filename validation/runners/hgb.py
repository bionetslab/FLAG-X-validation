"""Runner: sklearn HistGradientBoostingClassifier (tabular baseline). Env fx_sklearn."""
from sklearn.ensemble import HistGradientBoostingClassifier

from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn']


def build(params, seed, dataset, n_jobs):
    # Threads via OpenMP (OMP_NUM_THREADS, set in the job scripts)
    return HistGradientBoostingClassifier(random_state=seed, **params)


def fit_info(model):
    # Boosting iterations actually used (< max_iter if early stopping ended the training)
    return {'n_iter': int(model.n_iter_), 'early_stopped': bool(model.n_iter_ < model.max_iter)}
