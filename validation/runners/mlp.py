"""
Runner: FLAG-X MLP classifier (flagx.gating.MLPClassifier, flagx 0.2.1). Env fx_flagx_gpu (CUDA if available).

Grid params (mapped here to the flagx arguments):
    layer_sizes (list of int), n_epochs (int), batch_size (int) -> data_loader_params,
    lr, weight_decay (float; optional, torch default 0 if not given) -> optimizer_params (Adam),
    class_weight (None or 'balanced') -> loss_params={'weight': ...} of CrossEntropyLoss,
    early_stopping (bool), validation_fraction (float), n_iter_no_change (int), tol (float) -> passed unchanged.
- flagx has no seed argument: torch / numpy are seeded here before building and before fitting (flagx's internal
  train/val split is fixed to random_state=42). GPU runs are not bit-reproducible.
- class_weight='balanced': n_events / (K * n_events_class) (sklearn definition), computed from the training labels in
  fit, in the order of the sorted classes (= flagx's internal labels 0..K-1), as a tensor on the model's device
  (flagx does not move the loss to the device).
"""
import numpy as np
import torch
from flagx.gating import MLPClassifier

MODULES = ['numpy', 'torch', 'flagx', 'sklearn']


def input_arm(params):
    return 'shared'


def build(params, seed, dataset, n_jobs):
    _seed(seed)
    torch.set_num_threads(max(1, n_jobs))
    p = dict(params)
    class_weight = p.pop('class_weight', None)
    optimizer_params = {'lr': p.pop('lr', 1e-3)}
    if 'weight_decay' in p:  # Otherwise Adam's default (0)
        optimizer_params['weight_decay'] = p.pop('weight_decay')
    model = MLPClassifier(
        layer_sizes=tuple(p.pop('layer_sizes', (128, 64, 32))),
        n_epochs=p.pop('n_epochs', 20),
        optimizer_params=optimizer_params,
        # 1 data loader worker (flagx default; its validation loader also uses 1): in-memory tensors, more workers
        # bring no speed-up but add process copies (inflated memory)
        data_loader_params={'batch_size': p.pop('batch_size', 128), 'shuffle': True, 'num_workers': 1},
        verbosity=0,
        **p
    )
    # Runner state used in fit (not flagx arguments)
    model.runner_class_weight_ = class_weight
    model.runner_seed_ = seed
    return model


def fit(model, X, y, sample_ids):
    _seed(model.runner_seed_)
    if model.runner_class_weight_ == 'balanced':
        classes, counts = np.unique(y, return_counts=True)
        w = y.shape[0] / (classes.shape[0] * counts.astype(np.float64))
        model.loss_params = {'weight': torch.tensor(w, dtype=torch.float32, device=model.device)}
    elif model.runner_class_weight_ is not None:
        raise ValueError('class_weight must be None or "balanced", got {!r}'.format(model.runner_class_weight_))
    return model.fit(X, y)


def fit_info(model):
    """
    Training course from flagx's training_log_ (one entry per epoch actually trained): epochs trained, whether early
    stopping ended the training before n_epochs, the epoch whose weights were kept (lowest validation loss; flagx
    restores it with early stopping), and the per-epoch losses.
    """
    log = model.training_log_
    epochs = len(log['training_loss'])
    val = log.get('validation_loss')
    best = int(min(range(epochs), key=lambda i: val[i])) + 1 if (model.early_stopping and val) else epochs
    return {
        'epochs_trained': epochs,
        'early_stopped': bool(model.early_stopping and epochs < model.n_epochs),
        'best_epoch': best,  # 1-based; weights of this epoch are used
        'training_loss': [float(v) for v in log['training_loss']],
        'validation_loss': None if val is None else [float(v) for v in val],
    }


def predict(model, X, proba=False):
    # flagx predict = argmax of predict_proba; one pass for both
    p = model.predict_proba(X)
    return np.asarray(model.og_classes_)[np.argmax(p, axis=1)], (p if proba else None)


def _seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
