"""
sklearn-like wrapper for the LDA classifier of
Abdelaal et al. 2019, "Predicting cell populations in single cell mass cytometry data", Cytometry A,
https://doi.org/10.1002/cyto.a.23738

Official code (R + MATLAB, MIT license): https://github.com/tabdelaal/CyTOF-Linear-Classifier
pinned to commit 19c2f8f524ab01d99a46e32989be385f5de20a8c

The official R functions CyTOF_LDAtrain / CyTOF_LDApredict read FCS/CSV files from folders. The wrapper only
replaces this file I/O by passing in-memory matrices (via rpy2); the method itself (optional arcsinh/log
transformation, MASS::lda training with default arguments, prediction, max-posterior rejection) is copied
verbatim from the official R code, see the marked blocks in the R code strings below.

Contents:
    _R_LDA_FIT (R): CyTOF_LDAtrain without the file reading: marker selection, optional transformation,
        MASS::lda on all training events (the original pools all training files the same way), returns the Model
        list of the original (LDAclassifier, Transformation, markers).
    _R_LDA_PREDICT (R): CyTOF_LDApredict without the file reading, for one matrix of events: optional
        transformation, predict.lda, labels with max posterior < RejectionThreshold set to 'unknown'. Returns the
        labels and the posterior matrix instead of a per-file list of labels.
    LDAClassifier (Python):
        fit: maps the labels to 0..K-1, copies X / y to R, calls lda_fit and stores the R model serialized to
            bytes (R objects cannot be pickled).
        predict / predict_proba: unserialize the model, call lda_predict, map the labels back ('unknown' -> -1)
            and reorder the posterior columns to the order of the classes (R sorts factor levels as strings).
        score: macro F1 with abstentions (-1) given weight 0. save / load: pickle.

Env: fx_lda (envs/lda.yml). The env must be activated (conda activate fx_lda / conda run -n fx_lda ...),
otherwise rpy2 picks up the system R and crashes.
"""

import os
import pickle
import warnings
import time
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from typing import Union, List, Literal, Tuple, TypeVar, Dict, Any
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score


T = TypeVar("T", bound="LDAClassifier")


_R_LDA_FIT = '''
lda_fit <- function(exp_matrix, labels, RelevantMarkers, Transformation) {

    # exp_matrix: events x markers numeric matrix (numpy -> R)
    # labels: character vector with the class labels
    # RelevantMarkers: integer vector (1-based) with the markers used for classification
    # Transformation: 'arcsinh', 'log' or FALSE

    # Replaces the file reading of the original code
    # (CyTOF_LDAtrain.R L39-L67: Training.data = rbind(...as.data.frame(Temp)[,RelevantMarkers]))
    Training.data <- as.data.frame(exp_matrix)[,RelevantMarkers]
    Training.labels <- data.frame(labels, stringsAsFactors = FALSE)

    # Copied from https://github.com/tabdelaal/CyTOF-Linear-Classifier/blob/19c2f8f524ab01d99a46e32989be385f5de20a8c/CyTOF_LDAtrain.R
    # lines L69-L77 (transformation)
    if(Transformation != FALSE){
      if(Transformation == 'arcsinh'){
        Training.data = asinh(Training.data/5)
      }
      else if (Transformation == 'log'){
        Training.data = log(Training.data)
        Training.data[sapply(Training.data,is.infinite)] = 0
      }
    }
    # End of copied block

    # Copied from https://github.com/tabdelaal/CyTOF-Linear-Classifier/blob/19c2f8f524ab01d99a46e32989be385f5de20a8c/CyTOF_LDAtrain.R
    # lines L92-L95 (LDA training + model)
    LDAclassifier <- MASS::lda(Training.data,as.factor(Training.labels[,1]))

    Model <- list(LDAclassifier = LDAclassifier,Transformation = Transformation, markers = RelevantMarkers)
    return(Model)
    # End of copied block
}
'''

_R_LDA_PREDICT = '''
lda_predict <- function(Model, exp_matrix, RejectionThreshold) {

    # Model: list returned by lda_fit (== Model of CyTOF_LDAtrain)
    # exp_matrix: events x markers numeric matrix (numpy -> R)
    # RejectionThreshold: posterior probability lower threshold, 0 = no rejection

    # Addition (not in the original code): make sure the MASS namespace (S3 method predict.lda) is loaded,
    # needed when predicting with a loaded model in a fresh R session (original relies on MASS::lda having been
    # called in the same session)
    invisible(loadNamespace("MASS"))

    # Replaces the file reading of the original code
    # (CyTOF_LDApredict.R L62-L66: Testing.data = as.data.frame(Temp)[,Model$markers])
    Testing.data <- as.data.frame(exp_matrix)[,Model$markers]

    # Copied from https://github.com/tabdelaal/CyTOF-Linear-Classifier/blob/19c2f8f524ab01d99a46e32989be385f5de20a8c/CyTOF_LDApredict.R
    # lines L68-L81 (transformation, prediction, rejection)
    if(Model$Transformation != FALSE){
      if(Model$Transformation == 'arcsinh'){
        Testing.data = asinh(Testing.data/5)
      }
      else if (Model$Transformation == 'log'){
        Testing.data = log(Testing.data)
        Testing.data[sapply(Testing.data,is.infinite)] = 0
      }
    }
    Predictions <- predict(Model$LDAclassifier,Testing.data)
    Post.max <-apply(Predictions$posterior,1,max)
    Predictions$class <- factor(Predictions$class,levels = c(levels(Predictions$class),"unknown"))
    Predictions$class[Post.max < RejectionThreshold] <- "unknown"
    # End of copied block
    # (L82-L83 wrap the labels into a per-file list; replaced by returning the label vector + posteriors directly)

    out <- list(
        labels = as.character(Predictions$class),
        posterior = Predictions$posterior,
        posterior_classes = colnames(Predictions$posterior)
    )
    return(out)
}
'''


class LDAClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear discriminant analysis (LDA) classifier for cytometry data (Abdelaal et al. 2019), wrapping the official
    R implementation (CyTOF_LDAtrain.R / CyTOF_LDApredict.R, MASS::lda) via rpy2.

    Parameters
    ----------
    relevant_markers : list of int or None
        0-based column indices of the markers used for classification (``RelevantMarkers`` of the original code,
        which uses 1-based R indices; converted internally). ``None`` = use all columns.
        The original code has no default (required argument).
    transformation : {'arcsinh', 'log', False}
        ``Transformation`` of the original code: 'arcsinh' = asinh(x / 5), 'log' = log(x) with -Inf/Inf set to 0,
        False = no transformation. The original code has no default (required argument; the official examples use
        'arcsinh' for CyTOF and 'log' for flow cytometry). Default False here, because the benchmark data is
        already transformed.
    rejection_threshold : float
        ``RejectionThreshold`` of the original code, value in [0, 1]. Events whose maximal posterior probability is
        below the threshold are labelled 'unknown' (-1 here). 0 = no rejection. The original code has no default
        (required argument; the official example notebooks use 0, the function docstring example uses 0.7).
    time_fit_pred : bool
        Store fit/predict runtimes in ``time_df_``.
    verbosity : int
        Verbosity level.
    """
    def __init__(
            self,
            relevant_markers: Union[List[int], None] = None,
            transformation: Literal['arcsinh', 'log', False] = False,
            rejection_threshold: float = 0.0,
            time_fit_pred: bool = False,
            verbosity: int = 0,
    ):
        super().__init__()

        self.relevant_markers = relevant_markers
        self.transformation = transformation
        self.rejection_threshold = rejection_threshold
        self.time_fit_pred = time_fit_pred
        self.verbosity = verbosity

        self.is_fitted_ = False
        self.classes_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.n_features_in_ = None
        self.r_model_serialized_ = None  # R model object serialized to bytes (R objects are not picklable)
        self.time_df_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> T:

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.transformation not in ('arcsinh', 'log', False):
            raise ValueError(f"transformation must be 'arcsinh', 'log' or False, got {self.transformation}")

        # Rename classes to integers starting from 0
        y, self.classes_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_ = \
            LDAClassifier._process_class_labels(y=y)
        # Add label for unclassifiable cells to dict
        self.new_to_og_classes_dict_[-1] = -1

        ro.r(_R_LDA_FIT)
        r_lda_fit = ro.globalenv['lda_fit']

        if self.time_fit_pred:
            st = time.time()

        with localconverter(default_converter):
            r_x = ro.r['matrix'](ro.FloatVector(X.ravel(order='F')), nrow=X.shape[0], ncol=X.shape[1])
            r_y = ro.StrVector(y.astype(str))
            r_markers = ro.IntVector(self._r_relevant_markers())
            r_transformation = ro.BoolVector([False]) if self.transformation is False \
                else ro.StrVector([self.transformation])

            r_model = r_lda_fit(r_x, r_y, r_markers, r_transformation)

            # Serialize the R model so the Python object can be pickled
            self.r_model_serialized_ = bytes(ro.r['serialize'](r_model, ro.NULL))

        if self.time_fit_pred:
            et = time.time()
            if self.time_df_ is None:
                self.time_df_ = pd.DataFrame(index=['times', ])
            self.time_df_[f'fit_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'] = et - st

        self.is_fitted_ = True

        return self

    def _predict_r(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}')

        ro.r(_R_LDA_PREDICT)
        r_lda_predict = ro.globalenv['lda_predict']

        with localconverter(default_converter):
            r_model = ro.r['unserialize'](ro.vectors.ByteVector(self.r_model_serialized_))
            r_x = ro.r['matrix'](ro.FloatVector(X.ravel(order='F')), nrow=X.shape[0], ncol=X.shape[1])

            res = r_lda_predict(r_model, r_x, float(self.rejection_threshold))

            labels = np.array(list(res[0])).astype(str)
            # R matrices are stored column-major
            posterior = np.array(list(res[1]), dtype=float).reshape((-1, X.shape[0])).T
            posterior_classes = np.array(list(res[2])).astype(str)

        # Reorder posterior columns to the order of self.classes_ (R factor levels are sorted lexicographically)
        col_order = [int(np.flatnonzero(posterior_classes == str(c))[0]) for c in self.classes_]
        posterior = posterior[:, col_order]

        labels[labels == 'unknown'] = '-1'
        y_pred = labels.astype(int)

        return y_pred, posterior

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:

        if self.time_fit_pred:
            st = time.time()

        y_pred, _ = self._predict_r(X)

        if self.time_fit_pred:
            et = time.time()
            if self.time_df_ is None:
                self.time_df_ = pd.DataFrame(index=['times', ])
            self.time_df_[f'pred_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'] = et - st

        # Get vector with the original labels
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        if np.any(y_pred == -1) and self.verbosity >= 1:
            warnings.warn(
                f"{int(np.sum(y_pred == -1))} events had a maximal posterior probability below the rejection "
                f"threshold {self.rejection_threshold}. They are classified as label -1 ~= 'unknown'",
                UserWarning)

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        LDA posterior probabilities (MASS::predict.lda ``$posterior``), columns ordered as ``self.og_classes_``.
        """
        _, posterior = self._predict_r(X)
        return posterior

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
            filename: str = 'lda_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(
            cls: type(T),
            filename: str = 'lda_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> T:
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    def _r_relevant_markers(self) -> List[int]:
        # Convert 0-based Python indices to 1-based R indices
        if self.relevant_markers is None:
            return list(range(1, self.n_features_in_ + 1))
        return [int(i) + 1 for i in self.relevant_markers]

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
