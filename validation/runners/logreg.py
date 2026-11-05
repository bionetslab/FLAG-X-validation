"""
Runner: sklearn LogisticRegression (tabular baseline; multinomial, lbfgs). Env fx_sklearn.

Grid param C: float, or None = no penalty (mapped to C=inf; sklearn >= 1.8 deprecates penalty=None in favour of
C=np.inf). Threads via BLAS / OpenMP (OMP_NUM_THREADS, set in the job scripts).
"""
from sklearn.linear_model import LogisticRegression

from validation.runners._sklearn_common import fit, input_arm, predict  # noqa: F401

MODULES = ['numpy', 'sklearn', 'scipy']


def build(params, seed, dataset, n_jobs):
    p = dict(params)
    # Scikit-learn doc: ``C=np.inf`` results in unpenalized logistic regression
    if p.get('C', 1.0) is None:
        p['C'] = float('inf')
    return LogisticRegression(random_state=seed, **p)
