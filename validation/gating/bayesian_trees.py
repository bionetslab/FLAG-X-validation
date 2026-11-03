"""
sklearn-style wrapper for "Bayesian trees" (informed Mondrian processes) for automated gating.

Paper:  D. Ji, E. Nalisnick, Y. Qian, R.H. Scheuermann, P. Smyth,
        "Bayesian Trees for Automated Cytometry Data Analysis", MLHC 2018 (PMLR 85:465-483),
        http://proceedings.mlr.press/v85/ji18a/ji18a.pdf
Code:   https://github.com/disiji/fc_mondrian  (MIT License, Copyright (c) 2019 Disi Ji)
        pinned commit 97420ff311242afe103c45130ada509e1e60a0ac

IMPORTANT - the method is NOT supervised:
    The model needs (1) an expert prior table T (cell types x markers, entries in {-1, 0, +1} =
    low / neutral(unknown) / high) and (2) UNLABELLED cells. Tree structures are sampled from a Mondrian
    process prior informed by T; cut locations are then refined by MCMC on the (unlabelled) data likelihood.
    Cells are assigned to cell types via the leaves' agreement with T (majority vote over posterior samples).
    The original code contains no mechanism to derive T from labelled cells, therefore ``prior_table`` is a
    REQUIRED hyperparameter.

The method is transductive: in the original notebooks the MCMC and the classification both run on the data
that is classified (run/flowMP-AML-run-v1.0.ipynb, run/flowMP-BMMC-run-v1.0.ipynb: MP_mcmc(data, ...) and
classify_cells_majority(data, ...) on the same data, all subjects pooled). The wrapper follows this: fit()
only validates the prior table and sets up the label space (nothing is learned), predict(X) runs the MCMC on X
and classifies X. Multiple samples are pooled if passed jointly; to classify each sample separately, call
predict once per sample. (The multi-sample random-effects variant MP-RE is not wrapped.)

All functions marked "COPIED" below are taken from the official repository (URL pinned to the commit
above, original file + line numbers in the comment directly above each block). Modifications to the
copied code are listed explicitly in each comment ("MODIFIED: ..."); everything else is verbatim
(apart from the function names being prefixed with an underscore).
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Tuple, Dict, Any, TypeVar, Optional
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from sklearn.metrics import f1_score


T = TypeVar("T", bound="BayesianTreesClassifier")

_FC_MONDRIAN_URL = "https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac"

# Default prior hyperparameters, hard-coded in the original code
# (src/flowMP_sample.py L61-69 and L98-102; identical copies in src/flowMP_compute.py L57-65 and L84-88).
# Beta(alpha, beta) parameters for the relative cut location, keyed by the set of prior-table values of the
# chosen marker within the current sub-table. Paper: phi0 = 5, phi1 = 2 (Table 2).
DEFAULT_CUT_BETA_PARAMS = {
    '-1': (2., 5.), '0': (2., 2.), '1': (5., 2.),
    '-1 0': (2., 5.), '-1 1': (5., 5.), '0 1': (5., 2.),
    '-1 0 1': (5., 5.), '': (2., 2.),
}
# Multiplicative weights for the dimension-selection probability (p_d ~ weight * |X_d|).
# Paper: gamma0 = 1000 ({-1,+1}), gamma1 = 100 ({-1,0,+1}), gamma2 = 1 ({-1,0} / {0,+1}) (Table 2).
DEFAULT_DIM_PRIORITY = {
    '-1': 0, '0': 0, '1': 0,
    '-1 0': 1, '0 1': 1,
    '-1 0 1': 100, '-1 1': 1000,
}


# ----------------------------------------------------------------------------------------------------------------------
# COPIED from https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_sample.py
# lines 50-73 (draw_informed_Mondrian)
# MODIFIED: the hard-coded Beta-prior dict (L61-69) and the hard-coded dimension-priority dict (L98-102 inside
#           draw_informed_Mondrian_at_t) are passed in as arguments (defaults = original values, see above);
#           unused argument ``budget`` (has no effect for the informed MP in the original code) dropped.
# ----------------------------------------------------------------------------------------------------------------------
def _draw_informed_Mondrian(theta_space, table, priors_dict, priority_dict):
    cut_history = [1] * theta_space.shape[0]

    return _draw_informed_Mondrian_at_t(theta_space, table, priors_dict, priority_dict, cut_history)


# ----------------------------------------------------------------------------------------------------------------------
# COPIED from https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_sample.py
# lines 76-154 (draw_informed_Mondrian_at_t)
# MODIFIED: ``priority_dict`` (hard-coded at L98-102) is an argument and passed on in the recursion.
# ----------------------------------------------------------------------------------------------------------------------
def _draw_informed_Mondrian_at_t(theta_space, table, priors_dict, priority_dict, cut_history):
    if sum(cut_history) == 0 or table.shape[0] == 1:
        return (theta_space, None, None)

    types_str = [' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])
                 for d in range(table.shape[1])]

    if set([types_str[d] for d in range(table.shape[1]) if cut_history[d] == 1]) \
            .issubset({'0', '1', '-1'}):
        return (theta_space, None, None)

    types = np.array([priority_dict[_] for _ in types_str])

    dists = (theta_space[:, 1] - theta_space[:, 0]) * types
    lin_dim = np.sum(dists)
    # draw dimension to cut
    dim_probs = ((dists / lin_dim) * np.array(cut_history))
    dim_probs /= np.sum(dim_probs)
    d = np.argmax(np.random.multinomial(n=1, pvals=dim_probs))
    cut_history[d] = 0

    prior_type_str = ' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])
    prior_params = priors_dict[prior_type_str]

    # make scaled cut
    x = (theta_space[d, 1] - theta_space[d, 0]) * np.random.beta(prior_params[0], prior_params[1]) \
        + theta_space[d, 0]

    cut_type = types_str[d]

    if cut_type in {"-1 0 1", '-1 1'}:
        idx_table_left = table[table.columns[d]] != 1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] != -1
        table_right = table.loc[idx_table_right]

    if cut_type == '-1 0':
        idx_table_left = table[table.columns[d]] == -1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 0
        table_right = table.loc[idx_table_right]

    if cut_type == '0 1':
        idx_table_left = table[table.columns[d]] == 0
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 1
        table_right = table.loc[idx_table_right]

    # make lower partition
    theta_left = np.copy(theta_space)
    theta_left[d][1] = x
    M_left = _draw_informed_Mondrian_at_t(theta_left, table_left, priors_dict, priority_dict, list(cut_history))

    # make upper partition
    theta_right = np.copy(theta_space)
    theta_right[d][0] = x
    M_right = _draw_informed_Mondrian_at_t(theta_right, table_right, priors_dict, priority_dict, list(cut_history))

    return (theta_space, M_left, M_right)


# ----------------------------------------------------------------------------------------------------------------------
# COPIED (verbatim) from
# https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_sample.py
# lines 157-189 (Mondrian_Gaussian_perturbation)
# ----------------------------------------------------------------------------------------------------------------------
def _Mondrian_Gaussian_perturbation(theta_space, old_sample, stepsize):
    """
    Input:
        theta_space: D*2 np array. [[low_i, high_i] for dimension i], the area to be partitioned.
        old_sample: A MP tree
        stepsize: std of the gaussian distribution from which additive noise of cut locations are sampled
    Output:
        A new MP tree by adding gaussian noise to the cut locations of the old samples.
    """
    if old_sample[1] == None and old_sample[2] == None:
        return (theta_space, None, None)

    # find the dimension and location of first cut in the old_sample
    for _ in range(old_sample[0].shape[0]):
        if old_sample[0][_, 1] > old_sample[1][0][_, 1]:
            break
    dim, pos = _, old_sample[1][0][_, 1]
    # propose position of new cut
    good_propose = False
    while good_propose == False:
        new_pos = pos + np.random.normal(0, (old_sample[0][dim, 1] - old_sample[0][dim, 0]) * stepsize, 1)[0]
        if new_pos < theta_space[dim, 1] and new_pos > theta_space[dim, 0]:
            good_propose = True

    theta_left = np.copy(theta_space)
    theta_left[dim, 1] = new_pos
    theta_right = np.copy(theta_space)
    theta_right[dim, 0] = new_pos

    new_M_left = _Mondrian_Gaussian_perturbation(theta_left, old_sample[1], stepsize)
    new_M_right = _Mondrian_Gaussian_perturbation(theta_right, old_sample[2], stepsize)

    return (theta_space, new_M_left, new_M_right)


# ----------------------------------------------------------------------------------------------------------------------
# COPIED from https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_compute.py
# lines 6-36 (comp_log_p_sample)
# MODIFIED: the hard-coded covariance regulariser 0.001 (L20) is the argument ``cov_reg`` (default 0.001);
#           the unused variables log_len_left / log_len_right (L34-35) are dropped.
# ----------------------------------------------------------------------------------------------------------------------
def _comp_log_p_sample(mp_tree, data, cov_reg=0.001):
    """
    INPUT:
        mp_tree: a mondrian tree
        data: data points (each data point should be in the range of mp_tree[0])
    OUTPUT:
        log P(data | mp_tree)
    """
    if mp_tree[1] == None and mp_tree[2] == None:
        if data.shape[0] == 0:
            return 0
        else:
            mu = np.mean(data, axis=0)
            residual = data - mu
            cov = np.dot(residual.T, residual) / data.shape[0] + np.identity(data.shape[1]) * cov_reg
            return np.log(multivariate_normal.pdf(data, mean=mu, cov=cov)).sum()

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break

    dim, pos = _, left_rec[_, 1]
    idx_left = data[:, dim] < pos
    idx_right = data[:, dim] >= pos
    return _comp_log_p_sample(mp_tree[1], data[idx_left], cov_reg) + \
        _comp_log_p_sample(mp_tree[2], data[idx_right], cov_reg)


# ----------------------------------------------------------------------------------------------------------------------
# COPIED from https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_sample.py
# lines 192-227 (MP_mcmc)
# MODIFIED: prior dicts and ``cov_reg`` are passed through to _draw_informed_Mondrian / _comp_log_p_sample.
# ----------------------------------------------------------------------------------------------------------------------
def _MP_mcmc(data, theta_space, table, random_seed, N_MCMC_SAMPLE=3000, MCMC_GAUSSIAN_STD=0.1,
             priors_dict=None, priority_dict=None, cov_reg=0.001):
    """
    INPUT:
        data: N*D np array of a subject
        theta_space: empirical bounds of data
        table: prior information table
        random_seed: random seed for this chain
    OUTPUT:
        a list of accepted MP trees
    """
    if priors_dict is None:
        priors_dict = DEFAULT_CUT_BETA_PARAMS
    if priority_dict is None:
        priority_dict = DEFAULT_DIM_PRIORITY

    np.random.seed(random_seed)
    n_mcmc_sample = N_MCMC_SAMPLE
    mcmc_gaussian_std = MCMC_GAUSSIAN_STD

    accepts = []

    sample = _draw_informed_Mondrian(theta_space, table, priors_dict, priority_dict)
    # structure of MP tree is fixed after this step
    log_p_sample = _comp_log_p_sample(sample, data, cov_reg)
    accepts.append(sample)

    for idx in range(n_mcmc_sample):
        if (idx + 1) % (n_mcmc_sample / 10) == 0:
            mcmc_gaussian_std /= 2

        new_sample = _Mondrian_Gaussian_perturbation(theta_space, sample, mcmc_gaussian_std)
        new_log_p_sample = _comp_log_p_sample(new_sample, data, cov_reg)

        if new_log_p_sample >= log_p_sample or \
                np.log(np.random.uniform(low=0, high=1.)) <= (new_log_p_sample - log_p_sample):
            sample = new_sample
            log_p_sample = new_log_p_sample
            accepts.append(sample)

    return accepts


# ----------------------------------------------------------------------------------------------------------------------
# COPIED (verbatim) from
# https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_classify.py
# lines 55-102 (classify_cells)
# ----------------------------------------------------------------------------------------------------------------------
def _classify_cells(data, mp_tree, table, cell_type_name2idx):
    """
    This function is for cell classification after a MP tree is learnt.
    INPUT:
        data: N*D np.array
        mp_tree: a data structure recursively defined as [theta_space, left_tree, right_tree]
        table: a data frame
        cell_type_name2idx: a dicitionary that maps the cell type names to ids
    OUTPUT:
        Y: np.array of length N. Eech entry takes value from [0,1,...,K-1]
    """
    Y = np.array([1] * data.shape[0])

    if data.shape[0] == 0:
        return Y

    if mp_tree[1] == None and mp_tree[2] == None:
        if table.shape[0] > 1:
            #            print("more than one clusters, number of data points:", data.shape[0])
            labels = [cell_type_name2idx[table.index[_]] for _ in range(table.shape[0])]
            return np.array(np.random.choice(labels, data.shape[0], replace=True))
        else:
            return Y * cell_type_name2idx[table.index[0]]

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break
    dim, pos = _, left_rec[_, 1]

    # find labels that match dim info from table
    idx_table_left = table[table.columns[dim]] != 1
    table_left = table.loc[idx_table_left]

    idx_table_right = table[table.columns[dim]] != -1
    table_right = table.loc[idx_table_right]

    # find data INDICIES that go high / low on cut position in dimension dim
    idx_left = data[:, dim] < pos
    idx_right = data[:, dim] >= pos

    Y[idx_left] = _classify_cells(data[idx_left], mp_tree[1], table_left, cell_type_name2idx)
    Y[idx_right] = _classify_cells(data[idx_right], mp_tree[2], table_right, cell_type_name2idx)

    return Y


# ----------------------------------------------------------------------------------------------------------------------
# COPIED from https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/src/flowMP_classify.py
# lines 105-125 (classify_cells_majority)
# MODIFIED: additionally returns the vote matrix (used for predict_proba); the per-cell python vote loop
#           (L121-123) is replaced by an equivalent vectorised np.add.at.
# ----------------------------------------------------------------------------------------------------------------------
def _classify_cells_majority(data, burnt_samples, table, cell_type_name2idx):
    """
    This function is an extension of "classify_cells". It extends to the case when you need to ensemble a list of MP trees by majority.
    INPUT:
        data: N*D np.array
        burnt_samples: A list of MP trees
        table: a data frame
        cell_type_name2idx: a dicitionary that maps the cell type names to ids
    OUTPUT:
        Y_predict_majority: np.array of length N. Eech entry takes value from [0,1,...,K-1]
    """
    burnt_predictions = [None for i in burnt_samples]
    for i in range(len(burnt_samples)):
        burnt_predictions[i] = _classify_cells(data, burnt_samples[i], \
                                               table, cell_type_name2idx)
    votes = np.zeros([data.shape[0], table.shape[0]])
    rows = np.arange(data.shape[0])
    for Y_predict in burnt_predictions:
        np.add.at(votes, (rows, Y_predict), 1)
    Y_predict_majority = np.argmax(votes, axis=1)
    return Y_predict_majority, votes


class BayesianTreesClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian trees / informed Mondrian process gating (Ji et al., MLHC 2018), single-sample variant ("MP").

    Parameters
    ----------
    prior_table : pd.DataFrame or dict, REQUIRED
        Expert prior knowledge table (cell types x markers). Rows = cell types, given by the ORIGINAL label
        values used in ``y`` (DataFrame index or dict keys); columns = markers, in the SAME ORDER as the columns
        of X (DataFrame with D columns, or dict values = length-D sequences). Entries: +1 (high), -1 (low),
        0 / NaN (neutral / unknown; NaN is filled with 0 as in the original notebooks). Cannot be derived
        from labelled data with the original code. If ``y`` is passed to ``fit``, the rows must be labels
        occurring in ``y``.
    n_mcmc_chains : int, default 50
        Number of MCMC chains = number of tree structures sampled from the informed MP prior (paper: 50,
        notebook run/flowMP-AML-run-v1.0.ipynb: n_mcmc_chain = 50).
    n_mcmc_samples : int, default 3000
        Number of MCMC iterations (proposals) per chain (MP_mcmc default N_MCMC_SAMPLE=3000; paper: 3000).
    mcmc_gaussian_std : float, default 0.1
        Initial std of the Gaussian cut-location proposal, relative to the range of the cut dimension
        (MP_mcmc default MCMC_GAUSSIAN_STD=0.1). Halved every n_mcmc_samples / 10 iterations (original behaviour).
    n_burnt_samples : int, default 20
        Number of last ACCEPTED samples per chain kept for the majority-vote ensemble (paper: last 20 per
        chain -> 1000 trees; AML notebook cell 5 uses chain[-20:]).
    cut_beta_params : dict or None, default None (= original values, see DEFAULT_CUT_BETA_PARAMS)
        Beta(alpha, beta) prior on the relative cut location keyed by the set of table values of the
        chosen marker (paper's phi0=5, phi1=2).
    dim_priority : dict or None, default None (= original values, see DEFAULT_DIM_PRIORITY)
        Weights gamma for the dimension-selection probability p_d ~ gamma * |X_d| (paper: 1000 / 100 / 1).
    cov_reg : float, default 0.001
        Diagonal regulariser added to each leaf's Gaussian covariance in the likelihood (hard-coded in
        original code).
    random_state : int, default 0
        Chain i uses seed ``random_state + i`` (original notebooks: seed = chain index i, i.e. random_state=0).
        Also used to seed the random tie-breaking of ambiguous leaves in ``predict``.
    n_jobs : int, default 1
        Number of parallel chains (joblib), as in the commented-out parallel run in the original notebooks.
    time_fit_pred : bool, default False
        Store predict runtimes in ``time_df_`` (fit learns nothing).
    verbosity : int, default 0
    """

    def __init__(
            self,
            prior_table: Union[pd.DataFrame, Dict[Any, Any], None] = None,
            n_mcmc_chains: int = 50,
            n_mcmc_samples: int = 3000,
            mcmc_gaussian_std: float = 0.1,
            n_burnt_samples: int = 20,
            cut_beta_params: Optional[Dict[str, Tuple[float, float]]] = None,
            dim_priority: Optional[Dict[str, float]] = None,
            cov_reg: float = 0.001,
            random_state: int = 0,
            n_jobs: int = 1,
            time_fit_pred: bool = False,
            verbosity: int = 0,
    ):
        super().__init__()

        self.prior_table = prior_table
        self.n_mcmc_chains = n_mcmc_chains
        self.n_mcmc_samples = n_mcmc_samples
        self.mcmc_gaussian_std = mcmc_gaussian_std
        self.n_burnt_samples = n_burnt_samples
        self.cut_beta_params = cut_beta_params
        self.dim_priority = dim_priority
        self.cov_reg = cov_reg
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.time_fit_pred = time_fit_pred
        self.verbosity = verbosity

        self.is_fitted_ = False
        self.classes_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.table_ = None
        self.ct2idx_ = None
        self.emp_bounds_ = None
        self.burnt_samples_ = None
        self.n_accepted_per_chain_ = None
        self.time_df_ = None

    def fit(
            self,
            X: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None
    ) -> T:
        """
        Nothing is learned (the method is transductive, see module docstring). Validates the prior table and
        sets up the label space: the labels of y if given (table rows must occur in y), else the table rows.
        X is only used to check the number of markers.
        """

        table = self._parse_table()
        if X is not None:
            X = check_array(X)
            if X.shape[1] != table.shape[1]:
                raise ValueError(f'prior_table has {table.shape[1]} marker columns, X has {X.shape[1]} features.')

        labels = column_or_1d(y) if y is not None else np.asarray(table.index)
        # Rename classes to integers starting from 0
        _, self.classes_, class_priors, self.new_to_og_classes_dict_, self.og_classes_ = \
            BayesianTreesClassifier._process_class_labels(y=labels)
        self.class_priors_ = class_priors if y is not None else None
        # Add label for unclassifiable cells to dict (never produced by the method, kept for interface parity)
        self.new_to_og_classes_dict_[-1] = -1

        # Prior table with rows indexed by the new (0..K-1) class ids
        self.table_ = self._index_table(table)
        # classify_cells / classify_cells_majority expect ids 0..n_rows-1 in table-row order
        self.ct2idx_ = {ct: i for i, ct in enumerate(self.table_.index)}

        self.is_fitted_ = True

        return self

    def _predict_votes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        if X.shape[1] != self.table_.shape[1]:
            raise ValueError(f'X has {X.shape[1]} features, prior table has {self.table_.shape[1]} markers.')

        # As in the original notebooks: run the MCMC on the (unlabelled) data that is to be classified
        self.emp_bounds_, self.burnt_samples_, self.n_accepted_per_chain_ = self._run_mcmc(X)

        # Seed the global RNG used by the random assignment in ambiguous leaves (classify_cells)
        np.random.seed(self.random_state)
        y_idx, votes = _classify_cells_majority(X, self.burnt_samples_, self.table_, self.ct2idx_)
        return y_idx, votes

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """Runs the MCMC on X and classifies X (majority vote over the posterior tree samples)."""

        if self.time_fit_pred:
            st = time.time()

        y_idx, _ = self._predict_votes(X)

        if self.time_fit_pred:
            et = time.time()
            if self.time_df_ is None:
                self.time_df_ = pd.DataFrame(index=['times', ])
            self.time_df_[f'pred_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'] = et - st

        # table-row id -> new class id -> original label
        new_ids = np.asarray(self.table_.index)[y_idx]
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in new_ids])

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Runs the MCMC on X (same seeds as ``predict``) and returns the fraction of posterior tree samples
        (ensemble of burnt MCMC samples) voting for each class, i.e. the vote matrix computed by the original
        classify_cells_majority, normalised. Columns follow ``self.classes_``;
        classes in y without a row in the prior table always get probability 0.
        """
        _, votes = self._predict_votes(X)
        proba = np.zeros((votes.shape[0], self.classes_.shape[0]))
        proba[:, np.asarray(self.table_.index, dtype=int)] = votes / votes.sum(axis=1, keepdims=True)
        return proba

    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Union[np.ndarray, None] = None,
    ):
        y_pred = self.predict(X)
        # If no sample weight was passed,
        # exclude events with predicted label -1 (unclassifiable) from the score calculation
        # by setting their sample weight to 0,
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
            sample_weight[y_pred == -1] = 0

        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def save(
            self,
            filename: str = 'bayesian_trees_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(
            cls: type(T),
            filename: str = 'bayesian_trees_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> T:
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------------------------------------------------------
    def _run_mcmc(self, X: np.ndarray):
        # Empirical bounds of the data = space partitioned by the MP.
        # Adapted from notebook run/flowMP-AML-run-v1.0.ipynb (same commit), code cell 3:
        #   emp_bounds = np.array([[data[:,d].min(), data[:,d].max()] for d in range(data.shape[1])])
        emp_bounds = np.array([[X[:, d].min(), X[:, d].max()] for d in range(X.shape[1])])

        priors_dict = DEFAULT_CUT_BETA_PARAMS if self.cut_beta_params is None else self.cut_beta_params
        priority_dict = DEFAULT_DIM_PRIORITY if self.dim_priority is None else self.dim_priority

        # One chain per sampled tree structure, seeded with the chain index.
        # Adapted from notebook run/flowMP-AML-run-v1.0.ipynb, code cell 5 (parallel version, commented out there):
        #   accepted_MP = Parallel(n_jobs=num_cores)(delayed(MP_mcmc)(data, emp_bounds, table, i, n_mcmc_samples)
        #                                            for i in chains)
        #   burnt_samples = [sample for chain in accepted_MP for sample in chain[-20:]]
        if self.verbosity >= 1:
            print(f'Running {self.n_mcmc_chains} MCMC chains x {self.n_mcmc_samples} iterations '
                  f'on {X.shape[0]} cells ...')
        accepted = Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbosity >= 2 else 0)(
            delayed(_MP_mcmc)(
                X, emp_bounds, self.table_, self.random_state + i,
                self.n_mcmc_samples, self.mcmc_gaussian_std, priors_dict, priority_dict, self.cov_reg
            )
            for i in range(self.n_mcmc_chains)
        )
        n_accepted = [len(chain) for chain in accepted]
        burnt_samples = [sample for chain in accepted for sample in chain[-self.n_burnt_samples:]]
        if self.verbosity >= 1:
            print(f'Accepted samples per chain: {n_accepted}; ensemble size: {len(burnt_samples)}')
        return emp_bounds, burnt_samples, n_accepted

    def _parse_table(self) -> pd.DataFrame:
        if self.prior_table is None:
            raise ValueError(
                'prior_table is required: the Bayesian-trees method needs an expert prior table '
                '(cell types x markers, entries -1/0/+1); it cannot be learned from labelled cells.'
            )
        if isinstance(self.prior_table, pd.DataFrame):
            table = self.prior_table.copy()
        elif isinstance(self.prior_table, dict):
            table = pd.DataFrame.from_dict(
                {k: list(v) for k, v in self.prior_table.items()}, orient='index'
            )
        else:
            raise TypeError('prior_table must be a pandas DataFrame or a dict {label: sequence of D values}.')

        # As in the original notebooks: table = table.fillna(0)
        table = table.fillna(0)
        values = table.values.astype(float)
        if not np.isin(values, [-1, 0, 1]).all():
            raise ValueError('prior_table entries must be in {-1, 0, +1} (or NaN = 0).')
        table = pd.DataFrame(values.astype(int), index=table.index, columns=[f'm{d}' for d in range(table.shape[1])])
        if table.index.has_duplicates:
            raise ValueError('prior_table has duplicate cell-type rows.')
        return table

    def _index_table(self, table: pd.DataFrame) -> pd.DataFrame:
        # Map row labels (original labels) to new class ids; tolerate str/int mismatches (e.g. YAML keys)
        og_to_new = {v: k for k, v in self.new_to_og_classes_dict_.items() if k != -1}
        og_str_to_new = {str(k): v for k, v in og_to_new.items()}
        new_index = []
        for lab in table.index:
            if lab in og_to_new:
                new_index.append(og_to_new[lab])
            elif str(lab) in og_str_to_new:
                new_index.append(og_str_to_new[str(lab)])
            else:
                raise ValueError(f'prior_table row {lab!r} is not a label occurring in y '
                                 f'(labels: {list(self.og_classes_)}).')
        table.index = new_index

        missing = [self.new_to_og_classes_dict_[c] for c in self.classes_ if c not in set(new_index)]
        if missing:
            warnings.warn(f'Labels {missing} occur in y but have no row in prior_table; '
                          f'they can never be predicted.', UserWarning)
        if table.shape[0] < 2:
            raise ValueError('prior_table needs at least two cell types.')
        return table

    @staticmethod
    def _process_class_labels(
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Any, float], np.ndarray]:

        og_classes, counts = np.unique(y, return_counts=True)

        class_priors = counts / counts.sum()
        new_classes = np.array(list(range(og_classes.shape[0])))
        og_to_new_classes_dict = {key: value for key, value in zip(og_classes, new_classes)}
        y_new = np.vectorize(og_to_new_classes_dict.get)(y)

        new_to_og_classes_dict = {key: value for key, value in zip(new_classes, og_classes)}

        return y_new, new_classes, class_priors, new_to_og_classes_dict, og_classes
