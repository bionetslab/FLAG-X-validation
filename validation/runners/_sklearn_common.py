"""
Shared glue of the sklearn-like runners (fit / predict with optional probabilities). Python 3.5-compatible.
"""

import numpy as np


def input_arm(params):
    return 'shared'


def fit(model, X, y, sample_ids):
    return model.fit(X, y)


def predict(model, X, proba=False):
    """
    predict, or predict_proba + argmax (one pass) if proba=True and the model has predict_proba.
    Labels are mapped back via model.classes_ (sklearn convention: proba columns ordered as classes_).
    """
    if proba and hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        return np.asarray(model.classes_)[np.argmax(p, axis=1)], p
    return model.predict(X), None
