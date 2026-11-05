"""
sklearn-like wrapper for the GMM classifier of
Mocking et al. 2024, "Computational assessment of measurable residual disease in acute myeloid leukemia using mixture
models", Communications Medicine, https://doi.org/10.1038/s43856-024-00700-x

Original code: https://github.com/AUMC-HEMA/cMRD-manuscript, GMMClassifier.py,
pinned to commit df1b26e6035f22af6b32f8881bad1636f04acabd
(https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py)

The method fits one sklearn GaussianMixture per class on the training events of that class and assigns each event
to the class whose mixture gives the higher log-likelihood (score_samples). The original is binary (classes 0/1,
n_components_class0 / n_components_class1, tuned by grid / randomized search over 1-20 / 1-10 components in
02-GMMclf.ipynb, covariance_type 'full'). It is generalized here to K classes:
- one GaussianMixture per class, all with the same hyperparameters (n_components, covariance_type, ...);
- prediction = argmax over the per-class log-likelihoods. For 2 classes this is exactly the original rule
  (prob_class1 > prob_class0).astype(int): ties go to the lower class in both.
Other changes to the copied code are marked with # MOD. Addition: the input is cast to float64 (float32 data with
many identical values, e.g. at the log10 cut-off, makes GaussianMixture fail with 'ill-defined empirical covariance';
checked on BLAST110: with float64 input, predictions are identical to the original class for the same random_state).

predict_proba is not part of the original: softmax of the per-class log-likelihoods, i.e. the class posteriors under
uniform class priors (the quantity the decision rule compares). Its argmax equals predict.

Env: fx_sklearn (envs/sklearn.yml).
"""

import os
import pickle
from typing import Any, Dict, Tuple, TypeVar, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


T = TypeVar('T', bound='GMMMockingClassifier')


class GMMMockingClassifier(BaseEstimator, ClassifierMixin):
    """
    Per-class Gaussian mixture classifier (Mocking et al. 2024), generalized to K classes.

    Parameters (all passed unchanged to every per-class sklearn.mixture.GaussianMixture; defaults = sklearn defaults,
    as in the original, which only sets n_components and covariance_type)
    ----------
    n_components : int
        Number of mixture components per class. # MOD: one shared value instead of n_components_class0 /
        n_components_class1 (original default 1 for both).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Covariance type of the mixtures (original default 'full').
    tol, reg_covar, max_iter, n_init, init_params, verbose, verbose_interval : see sklearn.mixture.GaussianMixture.
        Not exposed: weights_init / means_init / precisions_init (arrays specific to one mixture, cannot be shared by
        the per-class mixtures) and warm_start (fit creates new mixtures, so it would have no effect).
    random_state : int or None
        Seed of the mixtures. # MOD: actually used (the original stores None instead, L11).
    """
    # Copied from https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py
    # lines L5-L11 (constructor), MOD: shared n_components, further GaussianMixture parameters, random_state stored
    def __init__(
            self,
            n_components: int = 1,
            covariance_type: str = 'full',
            tol: float = 1e-3,
            reg_covar: float = 1e-6,
            max_iter: int = 100,
            n_init: int = 1,
            init_params: str = 'kmeans',
            random_state: Union[int, None] = None,
            verbose: int = 0,
            verbose_interval: int = 10,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state  # MOD: original L11 sets self.random_state = None
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.ndarray, y: np.ndarray) -> T:
        # Addition: float64 input (float32 data with many identical values, e.g. at a log10 cut-off, makes
        # GaussianMixture fail with 'ill-defined empirical covariance'; sklearn recommends float64)
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Rename classes to integers starting from 0 (wrapper contract)
        y, self.classes_internal_, self.class_priors_, self.new_to_og_classes_dict_, self.classes_ = \
            GMMMockingClassifier._process_class_labels(y=y)

        # Copied from https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py
        # lines L13-L22 (one GaussianMixture per class, fitted on the events of that class)
        # MOD: loop over K classes instead of class0 / class1; shared hyperparameters
        self.gmms_ = []
        for c in self.classes_internal_:
            gmm = GaussianMixture(
                n_components=self.n_components, covariance_type=self.covariance_type, tol=self.tol,
                reg_covar=self.reg_covar, max_iter=self.max_iter, n_init=self.n_init, init_params=self.init_params,
                random_state=self.random_state, verbose=self.verbose, verbose_interval=self.verbose_interval,
            )
            self.gmms_.append(gmm.fit(X[y == c]))
        # End of copied block

        self.is_fitted_ = True
        return self  # MOD: original fit returns None

    def _log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, dtype=np.float64)  # Addition: float64, see fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}')
        # Copied from https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py
        # lines L25-L26 (per-class log-likelihoods), MOD: K classes, stacked to (n_events, K)
        return np.column_stack([gmm.score_samples(X) for gmm in self.gmms_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        ll = self._log_likelihoods(X)
        # Copied from https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py
        # line L27: return (prob_class1 > prob_class0).astype(int)
        # MOD: argmax over K classes (first maximum = lower class on ties, identical to L27 for K = 2)
        y_pred = np.argmax(ll, axis=1)
        # Map back to the original labels
        return np.array([self.new_to_og_classes_dict_[k] for k in y_pred])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Not in the original: softmax of the per-class log-likelihoods = class posteriors under uniform class priors.
        Columns ordered as self.classes_.
        """
        ll = self._log_likelihoods(X)
        ll = ll - ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Union[np.ndarray, None] = None):
        y_pred = self.predict(X)
        # Abstentions (-1) get weight 0 (wrapper contract; this method never abstains)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
            sample_weight[y_pred == -1] = 0
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def save(self, filename: str = 'gmm_mocking_classifier.pkl', filepath: Union[str, None] = None) -> None:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str = 'gmm_mocking_classifier.pkl', filepath: Union[str, None] = None) -> T:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _process_class_labels(
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Any, Any], np.ndarray]:
        og_classes, counts = np.unique(y, return_counts=True)
        class_priors = counts / counts.sum()
        new_classes = np.arange(og_classes.shape[0])
        og_to_new = {key: value for key, value in zip(og_classes, new_classes)}
        y_new = np.vectorize(og_to_new.get)(y)
        new_to_og = {key: value for key, value in zip(new_classes, og_classes)}
        return y_new, new_classes, class_priors, new_to_og, og_classes
