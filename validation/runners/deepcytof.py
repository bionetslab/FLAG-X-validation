"""
Runner: DeepCyTOF (validation/gating/deepcytof.py, Keras 1.2.2 / TF 1.0.1). Env fx_deepcytof (Python 3.5, CPU).
Keep this file Python 3.5-compatible.

Grid params: preprocess_log (= the input arm, PLAN.md D3: False -> shared input; True -> curated data + the
method's own log preprocessing), calibrate, hidden_layer_sizes (list of 3 int), lr_initial, and any other wrapper
argument.
- Fixed here: denoise=False (not used by the authors on flow data, working_notes/deepcytof.md),
  rejection_threshold=0 (D8), seed = the task seed, n_threads = the job's threads.
- fit gets the sample_ids of the training events: the classifier is trained on the reference sample chosen with the
  original criterion (classes absent from it can never be predicted).
- predict is called per val/test sample (sample_ids=None: X is one sample, one MMD-ResNet if calibrate=True).
  One predict_proba call (each call retrains the MMD-ResNet), labels = argmax (no rejection).
- The wrapper clears the Keras session on fit / predict.
"""
import numpy as np

from validation.core.runs import n_threads
from validation.gating.deepcytof import DeepCyTOFClassifier

MODULES = ['numpy', 'keras', 'tensorflow', 'sklearn']


def input_arm(params):
    return 'native' if params.get('preprocess_log', True) else 'shared'


def build(params, seed, dataset, n_jobs):
    p = dict(params)
    if 'hidden_layer_sizes' in p:
        p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
    p.setdefault('denoise', False)
    p.setdefault('rejection_threshold', 0.0)
    return DeepCyTOFClassifier(seed=seed, n_threads=max(1, n_jobs), verbosity=0, **p)


def fit(model, X, y, sample_ids):
    return model.fit(X, y, sample_ids=sample_ids)


def predict(model, X, proba=False):
    p = model.predict_proba(X)
    return np.asarray(model.og_classes_)[np.argmax(p, axis=1)], (p if proba else None)
