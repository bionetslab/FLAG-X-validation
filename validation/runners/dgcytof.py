"""
Runner: DGCyTOF (validation/gating/dgcytof.py). Env fx_flagx_gpu (CUDA if available), also runs in fx_flagx_cpu.

Grid params: val_size, layer_sizes (list of int), n_epochs, batch_size (-> train_params).
- random_state = the task seed (seeds torch and the train/val split of the wrapper).
- The calibration step's abstention (-1) is built into the method and kept (D8).
- predict_proba is the softmax before the calibration step, so labels always come from predict.
- The wrapper warns with the list of all abstained events; warnings are silenced here (logs would explode).
"""
import warnings

import torch

from validation.gating.dgcytof import DGCyTOFClassifier

MODULES = ['numpy', 'torch', 'sklearn', 'pandas']


def input_arm(params):
    return 'shared'


def build(params, seed, dataset, n_jobs):
    torch.set_num_threads(max(1, n_jobs))
    p = dict(params)
    return DGCyTOFClassifier(
        val_size=p.pop('val_size', 0.2),
        layer_sizes=tuple(p.pop('layer_sizes', (128, 64, 32))),
        n_epochs=p.pop('n_epochs', 20),
        # 1 data loader worker (training and the wrapper's validation step): the data are in-memory tensors, more
        # workers bring no speed-up but add process copies (inflated memory)
        train_params={'batch_size': p.pop('batch_size', 128), 'shuffle': True, 'num_workers': 1},
        random_state=seed,
        verbosity=0,
        **p
    )


def fit(model, X, y, sample_ids):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(X, y)


def predict(model, X, proba=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y_pred = model.predict(X)
        return y_pred, (model.predict_proba(X) if proba else None)
