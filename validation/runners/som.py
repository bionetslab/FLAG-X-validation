"""
Runner: FLAG-X SOM classifier (flagx.gating.SOMClassifier, flagx 0.2.1). Env fx_flagx_cpu (Somoclu, CPU).

- flagx has no seed argument; with initialization='pca' the training is deterministic (the grid keeps 'pca'). numpy is
  seeded anyway.
- A fresh instance is built per fit: SOMClassifier.fit overwrites radius_0 (flagx som_classifier.py L147-151, L203).
- predict returns -1 for events whose best-matching unit has no labelled training events (built-in abstention of the
  method, kept and scored as abstention, D8).
- Threads: Somoclu uses OpenMP (OMP_NUM_THREADS, set in the job scripts).
"""
import numpy as np
from flagx.gating import SOMClassifier

from validation.runners._sklearn_common import fit, input_arm  # noqa: F401

MODULES = ['numpy', 'flagx', 'somoclu', 'sklearn']


def build(params, seed, dataset, n_jobs):
    np.random.seed(seed)
    params = dict(params)
    if 'som_dimensions' in params:
        params['som_dimensions'] = tuple(params['som_dimensions'])
    return SOMClassifier(verbosity=0, **params)


def predict(model, X, proba=False):
    y_pred = model.predict(X)
    # predict_proba: class distribution of the BMU (columns = model.og_classes_); labels from predict (keeps -1)
    return y_pred, (model.predict_proba(X) if proba else None)
