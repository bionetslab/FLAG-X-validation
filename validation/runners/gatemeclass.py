"""
Runner: GateMeClass (validation/gating/gatemeclass.py, R via rpy2). Env fx_gatemeclass (must be activated).

Grid params: gmc_gmm_parameterization ('E' / 'V'), gmc_k, gmc_sampling, gmc_sampling_imp_vars.
- marker_names = config.CHANNELS[dataset]; gmc_reject_option=False (D8); gmc_seed = the task seed.
- No predict_proba (the method yields none).
- "V" may crash (Mclust on the many values at the log10 cut-off, legacy/gatemeclass_bug_report.md); a crash is a
  result of that grid point and is not worked around (PLAN.md step 6).
"""
import config
from validation.gating.gatemeclass import GateMeClassClassifier

MODULES = ['numpy', 'pandas', 'rpy2', 'sklearn']


def input_arm(params):
    return 'shared'


def build(params, seed, dataset, n_jobs):
    return GateMeClassClassifier(
        marker_names=list(config.CHANNELS[dataset]), gmc_reject_option=False, gmc_seed=seed, verbosity=0, **params)


def fit(model, X, y, sample_ids):
    return model.fit(X, y)


def predict(model, X, proba=False):
    return model.predict(X), None
