"""
sklearn-like wrapper for ACDC (Automated Cell-type Discovery and Classification)
Lee et al. 2017, Bioinformatics 33(11):1689-1695, https://doi.org/10.1093/bioinformatics/btx054

Original code:
    https://bitbucket.org/dudleylab/acdc  (master @ b6e98212cab060261f13760daf581ac299fc2fa3, 2017-02-08)
    The Bitbucket workspace is deactivated; the repo is archived by Software Heritage:
    https://archive.softwareheritage.org/browse/origin/directory/?origin_url=https://bitbucket.org/dudleylab/acdc
    (swh:1:rev:b6e98212cab060261f13760daf581ac299fc2fa3). No license file in the repository.
    The pipeline that glues the package functions together only exists in the tutorial notebooks:
    https://github.com/howchihlee/ACDC_tutorial (commit b76e01e6dac838456f1f51f0bff89295200d4d37)

ACDC is NOT a supervised method: its only "training" input is a cell-type x marker table (+1 / -1 / 0).
Labelled cells are never used by the original algorithm. Landmarks (cluster centres of cells matching a
table row, found by PhenoGraph) are computed from (pooled) data, then all events are classified by a
semi-supervised random walk on a kNN graph of events + landmarks.

The method is transductive: in the original code (tutorial notebooks) the marker GMMs, the landmarks and the
random walk are all computed on the data that is classified; there is no "fit on data A, apply to data B".
The wrapper follows this: fit() only validates the marker table and sets up the label space (nothing is
learned), predict(X) runs the full pipeline on X. Multiple samples:
    - sample_ids=None: all events of X are pooled and classified jointly (as in the tutorial notebooks, which
      pool all subjects).
    - sample_ids given: landmarks are computed on the pooled X, then each sample is classified separately with
      the same landmarks (paper Sec. 2.2.2: "we computed a common set of landmark points using pooled data ...
      and then classify each replication independently with the same landmark points").
To classify each sample completely independently, call predict once per sample.

All functions in the section "Code copied from the original ACDC repository" are copied (with minimal,
declared modifications marked by '# MOD:') from the original code. Everything else is wrapper glue.

Written to be importable with Python 3.5 (original ACDC env, see envs/acdc.yml), hence no f-strings.
"""

import os
import pickle
import time
import warnings
from datetime import datetime
from typing import Union, List, Tuple, Dict, Any, TypeVar

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import scipy.sparse.linalg  # MOD: explicit import, the original relies on an implicit import via sklearn
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import BallTree
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

import phenograph


T = TypeVar("T", bound="ACDCClassifier")

# Internal key of the 'unknown' cell type (the original uses the string 'unknown', see
# ACDC_tutorial/notebooks/AML_event_classification.ipynb, lines 55-56). An int key is used here so that the
# landmark label array stays numeric; it is mapped to the abstention label -1 of the benchmark.
UNKNOWN_KEY = -1


class ACDCClassifier(BaseEstimator, ClassifierMixin):
    """
    Parameters (defaults = values used in the original code / tutorial notebooks)
    ----------
    marker_table : pd.DataFrame, REQUIRED
        Cell type x marker table with entries +1 (present), -1 (absent), 0/NaN (do not consider). The index
        holds the cell type labels that predict() returns (should match the labels of y for evaluation),
        the columns the marker names (must be contained in ``marker_names``). This is the original input of
        ACDC. It cannot be derived from labelled cells (the original code has no mechanism for this).
    marker_names : list of str or None
        Names of the columns of X. Required if the columns of marker_table are not exactly the columns of X
        in that order. If None, the columns of X are assumed to be the columns of the table.
    n_neighbor : int, default 10
        Number of nearest neighbours of the kNN graph used by the random walk classifier
        (tutorial notebooks, 'n_neighbor = 10'; paper Sec. 2.2.1).
    thres : float, default 0.5
        Score threshold for assigning events to a cell type subset S_j = {w | f(w, c_j) > thres} used for
        landmark generation (tutorial notebooks, 'thres = 0.5'; paper Sec. 2.1.4).
    marker_model_thres : float, default 0.0
        Only events with marker intensity > marker_model_thres are used to fit the per-marker 2-component GMM
        (tutorial notebooks: 'compute_marker_model(..., table, 0.0)').
    gmm_n_init : int, default 10
        n_init of the per-marker GaussianMixture (hard-coded in compute_marker_model).
    min_events_landmark_clustering : int, default 60
        A cell type subset is clustered with PhenoGraph only if it contains more than this number of events,
        otherwise its mean is used as single landmark (hard-coded in get_landmarks).
    pg_k, pg_directed, pg_prune, pg_min_cluster_size, pg_jaccard, pg_primary_metric, pg_n_jobs, pg_q_tol :
        PhenoGraph parameters used for landmark generation (hard-coded in get_landmarks:
        k=30, directed=False, prune=False, min_cluster_size=10, jaccard=True, primary_metric='euclidean',
        n_jobs=-1, q_tol=1e-3).
    pg_louvain_time_limit, pg_nn_method :
        Remaining PhenoGraph parameters, not set by ACDC (PhenoGraph 1.5.2 defaults 2000, 'kdtree').
    rw_tol : float, default 1e-6
        Tolerance of the BiCG solver of the random walk (hard-coded in rm_classify).
    random_state : int or None, default None
        If not None, np.random.seed(random_state) is called at the start of each predict / predict_proba
        call (the original sets no seed; the GMMs use the global numpy RNG). The Louvain binary of PhenoGraph is not seedable.
    time_fit_pred : bool
        Store predict runtimes in ``time_df_`` (fit learns nothing).
    verbosity : int
    """

    def __init__(
            self,
            marker_table: Union[pd.DataFrame, None] = None,
            marker_names: Union[List[str], None] = None,
            n_neighbor: int = 10,
            thres: float = 0.5,
            marker_model_thres: float = 0.0,
            gmm_n_init: int = 10,
            min_events_landmark_clustering: int = 60,
            pg_k: int = 30,
            pg_directed: bool = False,
            pg_prune: bool = False,
            pg_min_cluster_size: int = 10,
            pg_jaccard: bool = True,
            pg_primary_metric: str = 'euclidean',
            pg_n_jobs: int = -1,
            pg_q_tol: float = 1e-3,
            pg_louvain_time_limit: int = 2000,
            pg_nn_method: str = 'kdtree',
            rw_tol: float = 1e-6,
            random_state: Union[int, None] = None,
            time_fit_pred: bool = False,
            verbosity: int = 0,
    ):
        super().__init__()

        self.marker_table = marker_table
        self.marker_names = marker_names
        self.n_neighbor = n_neighbor
        self.thres = thres
        self.marker_model_thres = marker_model_thres
        self.gmm_n_init = gmm_n_init
        self.min_events_landmark_clustering = min_events_landmark_clustering
        self.pg_k = pg_k
        self.pg_directed = pg_directed
        self.pg_prune = pg_prune
        self.pg_min_cluster_size = pg_min_cluster_size
        self.pg_jaccard = pg_jaccard
        self.pg_primary_metric = pg_primary_metric
        self.pg_n_jobs = pg_n_jobs
        self.pg_q_tol = pg_q_tol
        self.pg_louvain_time_limit = pg_louvain_time_limit
        self.pg_nn_method = pg_nn_method
        self.rw_tol = rw_tol
        self.random_state = random_state
        self.time_fit_pred = time_fit_pred
        self.verbosity = verbosity

        self.is_fitted_ = False
        self.classes_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.marker_table_ = None  # Table actually used (index = og labels), +1/-1/0
        self.feature_idx_ = None  # Columns of X (in table column order) used by ACDC
        self.mk_model_ = None
        self.landmark_mat_ = None
        self.landmark_label_ = None  # Internal int labels (0..n_types-1, UNKNOWN_KEY)
        self.time_df_ = None

    def fit(
            self,
            X: Union[np.ndarray, None] = None,
            y: Union[np.ndarray, None] = None
    ) -> T:
        """
        Nothing is learned (ACDC is transductive, see module docstring). Validates the marker table and sets up
        the label space (the rows of the table). X is only used to check the columns, y only to warn about
        labels without a table row.
        """

        if self.marker_table is None:
            raise ValueError('ACDC requires a marker_table (cell type x marker, +1/-1/0); it cannot be derived '
                             'from labelled cells.')

        # Original: 'table = table.fillna(0)' (AML_event_classification.ipynb, line 40)
        table = self.marker_table.copy().fillna(0)
        if X is not None:
            X = check_array(X)
            ACDCClassifier._get_feature_idx(
                table_markers=list(table.columns), marker_names=self.marker_names, n_features=X.shape[1]
            )
        self.marker_table_ = table.astype(int)

        # ### Class label bookkeeping: classes are the rows of the table
        self.og_classes_ = np.array(table.index.tolist())
        self.classes_ = np.arange(self.og_classes_.shape[0])
        self.new_to_og_classes_dict_ = {key: value for key, value in zip(self.classes_, self.og_classes_)}
        # Add label for unclassifiable cells ('unknown' cell type of ACDC) to dict
        self.new_to_og_classes_dict_[UNKNOWN_KEY] = -1
        if y is not None:
            y = column_or_1d(y)
            _, _, class_priors, _, og_classes_y = ACDCClassifier._process_class_labels(y=y)
            self.class_priors_ = dict(zip(og_classes_y, class_priors))
            missing = set(og_classes_y.tolist()) - set(self.og_classes_.tolist())
            if missing and self.verbosity >= 1:
                warnings.warn('Labels {} of y are not rows of the marker table, ACDC can only predict them as '
                              'unknown (-1).'.format(sorted(missing)), UserWarning)

        self.is_fitted_ = True

        return self

    def _run_acdc(
            self,
            X: np.ndarray,
            sample_ids: Union[np.ndarray, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full ACDC pipeline on X: marker GMMs + landmarks on the pooled X, then the random walk on all of X
        (sample_ids=None) or separately per sample (Sec. 2.2.2). The models of the last call are stored in
        ``mk_model_``, ``landmark_mat_``, ``landmark_label_``.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        if sample_ids is not None:
            sample_ids = column_or_1d(sample_ids)
            if sample_ids.shape[0] != X.shape[0]:
                raise ValueError('sample_ids must have one entry per row of X.')

        if self.random_state is not None:
            np.random.seed(self.random_state)

        table = self.marker_table_
        self.feature_idx_ = ACDCClassifier._get_feature_idx(
            table_markers=list(table.columns), marker_names=self.marker_names, n_features=X.shape[1]
        )
        x_sub = X[:, self.feature_idx_]

        # Per-marker 2-component GMMs (original function)
        mk_model = compute_marker_model(
            pd.DataFrame(x_sub, columns=table.columns), table, self.marker_model_thres, self.gmm_n_init
        )
        self.mk_model_ = mk_model

        # Internal table with int index 0..n_types-1 (glue, see UNKNOWN_KEY)
        table_int = table.copy()
        table_int.index = self.classes_
        idx2ct = self.classes_.tolist() + [UNKNOWN_KEY]

        # ### Landmark generation
        # Adopted from ACDC_tutorial/notebooks/AML_event_classification.ipynb, lines 287-311
        # (https://github.com/howchihlee/ACDC_tutorial/blob/b76e01e6dac838456f1f51f0bff89295200d4d37/notebooks/AML_event_classification.ipynb)
        score = get_score_mat(x_sub, [], table_int, [], mk_model)
        score = np.concatenate([score, 1.0 - score.max(axis=1)[:, np.newaxis]], axis=1)
        ct_index = get_unique_index(x_sub, score, table_int, self.thres)
        res_c = get_landmarks(
            x_sub, score, ct_index, idx2ct, phenograph, self.thres,
            min_events=self.min_events_landmark_clustering, pg_kwargs=self._pg_kwargs(),
        )
        landmark_mat, landmark_label = output_feature_matrix(res_c, [idx2ct[i] for i in range(len(idx2ct))])
        landmark_label = np.array(landmark_label)

        self.landmark_mat_ = landmark_mat
        self.landmark_label_ = landmark_label

        if self.verbosity >= 1:
            print('# ### ACDC: {} landmarks for cell types {}'.format(
                landmark_mat.shape[0],
                {self.new_to_og_classes_dict_[k]: int(v) for k, v in zip(*np.unique(landmark_label, return_counts=True))}
            ))
        no_lm = [self.new_to_og_classes_dict_[k] for k in idx2ct if k not in set(landmark_label.tolist())]
        if no_lm:
            warnings.warn('No landmarks found for cell types {}; they cannot be predicted.'.format(no_lm), UserWarning)

        # ### Random walk classification
        # Adopted from ACDC_tutorial/notebooks/AML_event_classification.ipynb, line 313
        if sample_ids is None:
            lp, y_pred = rm_classify(x_sub, landmark_mat, landmark_label, self.n_neighbor, tol=self.rw_tol)
        else:
            # Glue: same landmarks, one random walk per sample (paper Sec. 2.2.2)
            lp = np.zeros((X.shape[0], np.unique(landmark_label).shape[0]))
            y_pred = np.zeros(X.shape[0], dtype=landmark_label.dtype)
            for sid in np.unique(sample_ids):
                mask = sample_ids == sid
                lp[mask], y_pred[mask] = rm_classify(
                    x_sub[mask], landmark_mat, landmark_label, self.n_neighbor, tol=self.rw_tol
                )
        return lp, y_pred

    def predict(
            self,
            X: np.ndarray,
            sample_ids: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """Runs the full ACDC pipeline on X (see ``_run_acdc``)."""

        if self.time_fit_pred:
            st = time.time()

        _, y_pred = self._run_acdc(X, sample_ids=sample_ids)

        if self.time_fit_pred:
            et = time.time()
            if self.time_df_ is None:
                self.time_df_ = pd.DataFrame(index=['times', ])
            self.time_df_['pred_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))] = et - st

        # Get vector with the original labels (ACDC 'unknown' -> -1)
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        if np.any(y_pred == -1) and self.verbosity >= 1:
            warnings.warn(
                "{} events were classified as ACDC cell type 'unknown'. They are labelled -1 ~= 'new cell "
                "type'".format(int(np.sum(y_pred == -1))), UserWarning)

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray,
            sample_ids: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Runs the full ACDC pipeline on X (see ``_run_acdc``) and returns the random walk probabilities (native
        output 'lp' of rm_classify): probability that a random walker starting at an event first reaches a
        landmark of a given cell type. Columns: ``self.classes_`` (i.e. ``self.og_classes_``) followed by a last
        column for 'unknown' (-1). Cell types without landmarks get probability 0. Values are the raw BiCG
        solution (rows sum to ~1 up to solver tolerance).
        """
        lp, _ = self._run_acdc(X, sample_ids=sample_ids)
        idx2lpt = sorted(set(self.landmark_label_.tolist()))  # Column order of lp, as in rm_classify
        cols = self.classes_.tolist() + [UNKNOWN_KEY]
        proba = np.zeros((lp.shape[0], len(cols)))
        for j, key in enumerate(idx2lpt):
            proba[:, cols.index(key)] = lp[:, j]
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
        # by setting their sample weight to 0
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
            sample_weight[y_pred == -1] = 0

        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def save(
            self,
            filename: str = 'acdc_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(
            cls,
            filename: str = 'acdc_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> T:
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    def _pg_kwargs(self) -> Dict[str, Any]:
        return {
            'k': self.pg_k,
            'directed': self.pg_directed,
            'prune': self.pg_prune,
            'min_cluster_size': self.pg_min_cluster_size,
            'jaccard': self.pg_jaccard,
            'primary_metric': self.pg_primary_metric,
            'n_jobs': self.pg_n_jobs,
            'q_tol': self.pg_q_tol,
            'louvain_time_limit': self.pg_louvain_time_limit,
            'nn_method': self.pg_nn_method,
        }

    @staticmethod
    def _get_feature_idx(table_markers: List[str], marker_names: Union[List[str], None], n_features: int):
        if marker_names is None:
            if len(table_markers) != n_features:
                raise ValueError('marker_names must be given if the marker table does not have one column per '
                                 'column of X (in the same order).')
            return np.arange(n_features)
        marker_names = list(marker_names)
        missing = [m for m in table_markers if m not in marker_names]
        if missing:
            raise ValueError('Markers {} of the marker table are not in marker_names.'.format(missing))
        return np.array([marker_names.index(m) for m in table_markers])

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


# ######################################################################################################################
# Code copied from the original ACDC repository
# https://bitbucket.org/dudleylab/acdc @ b6e98212cab060261f13760daf581ac299fc2fa3
# (archived: swh:1:rev:b6e98212cab060261f13760daf581ac299fc2fa3), files ACDC/cell_type_annotation.py and
# ACDC/random_walk_classifier.py. Modifications are marked with '# MOD:'.
# ######################################################################################################################

# Copied from ACDC/cell_type_annotation.py, lines 26-37
# MOD: n_init (hard-coded 10) exposed as argument; '.as_matrix()' -> '.values' (identical, as_matrix removed in
# pandas 1.0)
def compute_marker_model(df, table, thres, n_init=10):
    ## compute 2-gaussian mixture model for each marker
    ## output mean, variance, and weight for each marker
    mk_model = {}
    for mk in table.axes[1]:
        gmm = GMM(n_components=2, n_init=n_init)
        tmp = df[mk].values
        gmm.fit(tmp[tmp > thres, np.newaxis])
        index = np.argsort(gmm.means_, axis = 0)
        mk_model[mk] = (gmm.means_[index], gmm.covariances_[index], gmm.weights_[index])

    return mk_model


# Copied from ACDC/cell_type_annotation.py, lines 39-43 (unmodified)
def appr_fun(x, xroot, slope):
    ## heuristic for axxproimating the decision boundary
    y = np.exp( (x - xroot) * slope)
    y = y / (1.0 + y)
    return np.squeeze(y)


# Copied from ACDC/cell_type_annotation.py, lines 45-64 (unmodified)
def compute_paras(mk_model, mk):
    ## compute the critical point of the decision bounary
    ## critical point is defined by P(theta = 1|x) = P(theta = 0|x)
    ## given a 1-D two-mixture model
    ##
    ## inputs:
    ## mk_model: precomputed 2-gassian mixture model for all markers
    ## mk: one selected marker
    ##
    ## outputs:
    ## xroot: location of the critical point
    ## slope: slope of the decision boundary at the critical point

    mus, sigmas, ws = mk_model[mk]
    a = (-0.5 / sigmas[0] + 0.5 /sigmas[1])
    b = mus[0] / sigmas[0] - mus[1] / sigmas[1]
    c = 0.5 * (-mus[0] **2 / sigmas[0] + mus[1] ** 2 / sigmas[1]) + np.log(ws[0] / ws[1]) + 0.5 * np.log(sigmas[1] / sigmas[0])
    xroot = (-b - np.sqrt(b ** 2 - 4.0 * a * c) ) / (2.0 * a)
    slope = 0.5 * (xroot - mus[0]) / sigmas[0] - 0.5 * (xroot - mus[1]) / sigmas[1]
    return xroot, slope


# Copied from ACDC/cell_type_annotation.py, lines 66-71 (unmodified)
def score_fun(x, mk_model, mk):
    ## compute the score function
    ## the score should roughly proportional to P(theta = + | x)

    xroot, slope = compute_paras(mk_model, mk)
    return appr_fun(x, xroot, slope)


# Copied from ACDC/cell_type_annotation.py, lines 75-85 (unmodified; the y_cluster branch references an undefined
# 'thres' in the original, it is never reached since ACDC always passes y_cluster=[])
def get_score_mat(X, weights, table, y_cluster, mk_model):
    ## compute the cluster x annotation score matrix
    ## X: sample x feature data matrix
    ## table: a list of (ind, sign)
    if y_cluster:
        clusters = np.unique(y_cluster)
        mu = np.vstack([np.mean(X[y_cluster==cl, :], axis = 0) for cl in clusters])
        score = get_score_mat_main(mu, weights, table, mk_model, thres)  # noqa: F821 (original bug, unreachable)
    else:
        score = get_score_mat_main(X, weights, table, mk_model)
    return score


# Copied from ACDC/cell_type_annotation.py, lines 87-107 (unmodified)
def get_score_mat_main(X, weights, table, mk_model):
    ## compute the cluster x annotation score matrix
    ## X: sample x feature data matrix
    ## table: a list of (ind, sign)
    score = np.zeros((X.shape[0], len(table.axes[0])))
    for i, ct in enumerate(table.index):
        #score_tmp = np.zeros(X.shape[0])
        #count = 0.0
        score_tmp = np.ones(X.shape[0])
        count = 1.0
        for j, mk in enumerate(table.columns):
            if table.loc[ct, mk] > 0:
                score_tmp =  np.min([score_tmp, score_fun(X[:, j], mk_model, mk)], axis = 0)
                #score_tmp +=  score_fun(X[:, j], mk_model, mk)
                #count += 1.0
            elif  table.loc[ct, mk] < 0:
                score_tmp =  np.min([score_tmp, 1.0 - score_fun(X[:, j], mk_model, mk)], axis = 0)
                #score_tmp +=  1.0 - score_fun(X[:, j], mk_model, mk)
                #count += 1.0
        score[:, i]= score_tmp / count
    return score


# Copied from ACDC/cell_type_annotation.py, lines 111-127
# MOD: 'table.as_matrix()' -> 'table.values' (identical, as_matrix removed in pandas 1.0)
def get_unique_index(X, score, table, thres):

    ct_score = np.abs(table.values).sum(axis = 1)
    ct_index = np.zeros((X.shape[0], len(table.index) + 1))

    for i, ct in enumerate(table.index):
        ct_index[:, i] = (score[:, i] > thres) * 1

    for i in np.where(ct_index.sum(axis = 1) > 1)[0]:
        mk_tmp = np.where(ct_index[i, :] == 1)[0]
        score_tmp = ct_score[mk_tmp]
        ct_index[i, :] = 0
        if np.sum(score_tmp == score_tmp.max()) == 1:
            ct_index[i, mk_tmp[np.argmax(score_tmp)]] = 1

    ct_index[:, -1] = (score[:, -1] > thres) * 1
    return ct_index


# Copied from ACDC/cell_type_annotation.py, lines 129-145
# MOD: the hard-coded subset size (60) and PhenoGraph arguments (k=30, directed=False, prune=False,
# min_cluster_size=10, jaccard=True, primary_metric='euclidean', n_jobs=-1, q_tol=1e-3) are exposed as arguments
# 'min_events' / 'pg_kwargs' (same defaults via the wrapper).
def get_landmarks(X, score, ct_index, idx2ct, obj, thres, min_events=60, pg_kwargs=None):
    if pg_kwargs is None:
        pg_kwargs = dict(k=30, directed=False, prune=False, min_cluster_size=10, jaccard=True,
                         primary_metric='euclidean', n_jobs=-1, q_tol=1e-3)

    res_c = {}
    for idx, ct in enumerate(idx2ct):
        if np.any(ct_index[:, idx] == 1):
            item = X[ct_index[:, idx] == 1, :]
        else:
            item = X[score[:, idx] > thres, :]

        if item.shape[0] > min_events:
            res = obj.cluster(item, **pg_kwargs)
            res_c[ct] = return_center(item, res[0])
        elif item.shape[0]> 0:
            res_c[ct] = item.mean(axis = 0)[np.newaxis, :]

    return res_c


# Copied from ACDC/cell_type_annotation.py, lines 154-159 (unmodified)
def return_center(X, labels):
    clusters = np.unique(labels)
    centers = []
    for i in clusters:
        centers.append(np.mean(X[labels == i, :], axis = 0))
    return np.vstack(centers)


# Copied from ACDC/cell_type_annotation.py, lines 183-205 (unmodified)
def output_feature_matrix(res, label=None):
    ## turn a dictionary data structure into a matrix
    ## input:
    ## res: a dictionary that map a key (cluster) to an (n_sample, n_features) array
    ## label: a list of keys. output will be arragned as the order of keys in this list
    ## output:
    ## feature_mat: a (n_sample, n_features) matrix
    ## ct_mat: a (n_sample,) that each element is a key (cluster)

    ct_mat = []
    feature_mat = []
    if not label:
        label = res.keys()

    for key in label:
        if key in res:
            item = res[key]
            if item.shape[0] > 0:
                feature_mat.append(item)
                ct_mat += [key] * item.shape[0]

    feature_mat = np.vstack(feature_mat)
    return feature_mat, ct_mat


# Copied from ACDC/random_walk_classifier.py, lines 7-10 (unmodified)
def add_value(cor, key, val):
    if key not in cor:
        cor[key] = 0.0
    cor[key] += val


# Copied from ACDC/random_walk_classifier.py, lines 12-21 (unmodified)
def dict_2_cs(cor):
    data, indices, indptr = [], [], [0]
    ptr_curr = 0
    for key, item in cor.items():
        indices += item.keys()
        data += item.values()
        ptr_curr += len(item)
        indptr.append(ptr_curr)

    return data, indices, indptr


# Copied from ACDC/random_walk_classifier.py, lines 23-27 (unmodified)
def knn_neighbors(data, lpts, n_neighbor):
    X_semi = np.concatenate([lpts, data], axis = 0)
    btree = BallTree(X_semi)
    nbr_id = btree.query(X_semi, k=n_neighbor + 1, return_distance=False)
    return nbr_id


# Copied from ACDC/random_walk_classifier.py, lines 29-53
# MOD: 'dtype = np.float' -> 'dtype = np.float64' (identical, np.float removed in numpy 1.24)
def compute_BTLu(y_lpt, nbr_id, n_sample, n_type, n_lpt):

    cor_Lu = {i:{} for i in range(n_sample)}
    cor_BT = {i:{} for i in range(n_type)}

    for i in range(n_lpt):
        for j in nbr_id[i, :]:
            if j >= n_lpt:
                add_value(cor_BT[y_lpt[i]], j - n_lpt, -1)
                add_value(cor_Lu[j - n_lpt], j - n_lpt, 1)

    for i in range(n_sample):
        for j in nbr_id[n_lpt + i, 1:]:
            add_value(cor_Lu[i], i, 1)
            if j < n_lpt:
                add_value(cor_BT[y_lpt[j]], i, -1)

            elif j >= n_lpt:
                add_value(cor_Lu[j - n_lpt], i, -1)
                add_value(cor_Lu[i], j - n_lpt, -1)
                add_value(cor_Lu[j - n_lpt], j - n_lpt, 1)

    BT = scipy.sparse.csc_matrix(dict_2_cs(cor_BT), shape=(n_sample, n_type), dtype = np.float64)
    Lu = scipy.sparse.csc_matrix(dict_2_cs(cor_Lu), shape=(n_sample, n_sample), dtype = np.float64)
    return BT, Lu


# Copied from ACDC/random_walk_classifier.py, lines 55-84
# MOD: the hard-coded BiCG tolerance (1e-6) is exposed as argument 'tol'; it is passed as 'rtol' for scipy >= 1.12
# (where 'tol' was renamed; same meaning).
def rm_classify(data, lpts, y_lpt, n_neighbor, tol=1e-6):
    ## data: (n_sample, n_feature) ndarray. unlabeled data
    ## lpts: (n_lpt, n_feature) ndarray. labeled data
    ## y_lpts: (n_lpt, ) ndarray, labels for the labeled data
    ## n_neighbors: number of neighbors
    ## labels: order of labels


    n_lpt = lpts.shape[0]
    n_type = len(set(y_lpt))
    n_sample = data.shape[0]

    idx2lpt =sorted(set(y_lpt))
    lpt2idx = {l:idx for idx, l in enumerate(idx2lpt)}

    y_lpy_tmp = np.array([lpt2idx[l] for l in y_lpt])

    nbr_id = knn_neighbors(data, lpts, n_neighbor)

    BT, Lu = compute_BTLu(y_lpy_tmp, nbr_id, n_sample, n_type, n_lpt)

    tol_kw = _bicg_tol_kwarg(tol)  # MOD
    lp = np.zeros((n_sample, n_type))
    for i in range(n_type):
        result = scipy.sparse.linalg.bicg(Lu, BT[:, i].toarray(), **tol_kw)  # MOD: tol = 1e-6 -> **tol_kw
        lp[:, i] = -result[0]


    y_pred = np.argmax(lp, axis = 1)
    y_pred = np.array([idx2lpt[l] for l in y_pred])
    return lp, y_pred


def _bicg_tol_kwarg(tol):
    # Glue (not from ACDC): scipy renamed bicg's 'tol' to 'rtol' in 1.12 and removed 'tol' in 1.14
    major, minor = [int(v) for v in scipy.__version__.split('.')[:2]]
    return {'rtol': tol} if (major, minor) >= (1, 12) else {'tol': tol}
