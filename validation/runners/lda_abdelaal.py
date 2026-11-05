"""
Runner: LDA of Abdelaal et al. 2019 (validation/gating/lda.py, official R code via rpy2). Env fx_lda (activated).

Grid param: transformation in {False, 'log', 'arcsinh'} = the input arm (PLAN.md D3):
    False -> shared input (log10 with cut-offs), the method's own transformation off;
    'log' / 'arcsinh' -> curated (untransformed) data + the method's own transformation (official flow example: 'log').
- rejection_threshold=0 (D8). Deterministic (seed unused).
- 'log' gives NaN on negative values and MASS::lda fails; such a crash is a result of the grid point.
"""
import numpy as np

from validation.gating.lda import LDAClassifier

MODULES = ['numpy', 'pandas', 'rpy2', 'sklearn']


def input_arm(params):
    return 'shared' if params.get('transformation', False) is False else 'native'


def build(params, seed, dataset, n_jobs):
    return LDAClassifier(rejection_threshold=0.0, verbosity=0, **params)


def fit(model, X, y, sample_ids):
    return model.fit(X, y)


def predict(model, X, proba=False):
    if not proba:
        return model.predict(X), None
    # One R call: posteriors (columns = model.og_classes_) + argmax (rejection is off)
    p = model.predict_proba(X)
    return np.asarray(model.og_classes_)[np.argmax(p, axis=1)], p
