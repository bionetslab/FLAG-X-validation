"""Runner: GMM classifier of Mocking et al. 2024 (validation/gating/gmm_mocking.py). Env fx_sklearn."""
from validation.gating.gmm_mocking import GMMMockingClassifier
from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn']


def build(params, seed, dataset, n_jobs):
    return GMMMockingClassifier(random_state=seed, **params)


def fit_info(model):
    # Per class (in the order of model.classes_): EM converged before max_iter, and EM iterations used (of the best
    # initialisation)
    return {'converged': [bool(g.converged_) for g in model.gmms_], 'n_iter': [int(g.n_iter_) for g in model.gmms_]}
