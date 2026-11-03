"""
sklearn-like wrapper for GateMeClass (Caligola et al. 2024, "GateMeClass: Gate Mining and Classification of
cytometry data", Bioinformatics 40(5): btae322, https://doi.org/10.1093/bioinformatics/btae322), calling the official
R package via rpy2.

Official code: https://github.com/simo1c/GateMeClass (R, MIT), commit a93e0202336f7048324c7c8b9b60ee965104f4d5
(R/GateMeClass.R last changed in 21105f9, 2024-03-19). The package is installed in the env; nothing is copied.

Method: GateMeClass_train learns a marker table (cell type x marker, +/-) from the labelled reference: per-marker
2-component GMMs (mclust) give each cell a +/- signature, decision trees (caret rpart) select the informative markers
per pair of cell types. GateMeClass_annotate matches the GMM signatures of a random subset of the new cells
(`sampling`) to the marker table and labels the remaining / ambiguous cells by kNN (`k`); with `reject_option` cells
without mutual nearest neighbours among the annotated cells (batchelor::findMutualNN) stay 'Unclassified' (-1).
No transformation or scaling inside the package; the input is used as given.

Contents:
    GateMeClassClassifier:
        fit: maps labels to 0..K-1 (passed to R as strings), formats the marker names, calls GateMeClass_train,
            stores the marker table as a pandas DataFrame.
        predict: calls GateMeClass_annotate with the stored marker table, returns the original labels,
            'Unclassified' -> -1.
        score: macro F1 with unassigned cells (-1) given weight 0. save / load: pickle.
    No predict_proba (GateMeClass yields no probabilities).

Notes:
    - GMM_parameterization: "E" (equal variance) is the package default and the wrapper default (used by the authors
      on flow data), "V" the value in all README examples; the package couples it to ranked set sampling (RSS on for
      "V", off for "E", R/GateMeClass.R L605, L892).
    - With "V", two upstream crash paths exist: Mclust returns NULL on an almost constant RSS sample (e.g. many values
      at a log10 cut-off; legacy/gatemeclass_bug_report.md), and 'index_sel' is undefined if a marker subset has
      2-99 cells (R/GateMeClass.R L189 vs L235).
    - GateMeClass infers 3-level markers ("mid"/"hi") from the label names; the wrapper passes integer labels, so all
      markers are modelled with 2 levels (+/-).

Env: fx_gatemeclass (envs/gatemeclass.yml; GateMeClass installed from the pinned commit after creating the env, see
the header of the yml). The env must be activated (conda activate fx_gatemeclass / conda run -n fx_gatemeclass ...),
otherwise rpy2 picks up the system R.
"""
import os
import pickle
import warnings
import time
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri,  pandas2ri, StrVector
from typing import Union, List, Literal, Tuple, TypeVar, Dict, Any
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score


T = TypeVar("T", bound="GateMeClassClassifier")


class GateMeClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            marker_names: Union[List[str], None],
            gmc_gmm_parameterization: Literal["V", "E"] = "E",
            gmc_sampling_imp_vars: float = 0.05,
            gmc_k: int = 20,
            gmc_sampling: float = 0.1,
            gmc_reject_option: bool = False,
            gmc_seed: int = 1,
            time_fit_pred: bool = False,
            verbosity: int = 0,
    ):
        super().__init__()

        self.marker_names = marker_names
        self.gmc_gmm_parameterization = gmc_gmm_parameterization
        self.gmc_sampling_imp_vars = gmc_sampling_imp_vars
        self.gmc_k = gmc_k
        self.gmc_sampling = gmc_sampling
        self.gmc_reject_option = gmc_reject_option
        self.gmc_seed = gmc_seed
        self.time_fit_pred = time_fit_pred
        self.verbosity = verbosity

        self.is_fitted_ = False
        self.classes_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.marker_names_correct_format_ = None
        self.marker_table_ = None
        self.time_df_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> T:

        X, y = check_X_y(X, y)

        # Rename classes to integers starting from 0
        y, self.classes_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_ = \
            GateMeClassClassifier._process_class_labels(y=y)
        # Add label for unclassifiable cells to dict
        self.new_to_og_classes_dict_[-1] = -1

        # Change the marker names into the correct format
        self.marker_names_correct_format_ = GateMeClassClassifier._check_marker_name_format(
            marker_names=self.marker_names,
            n_markers=X.shape[1]
        )

        # Annotate the train data matrix with the marker names
        # x_df = pd.DataFrame(
        #     X.T,
        #     index=self.marker_names_correct_format_,
        # )

        # Load the GateMeClass R package
        ro.r('''
            library(GateMeClass)
            ''')

        # Define the R function for GateMeClass training
        ro.r('''
            gmc_fit <- function(exp_matrix, labels, marker_names, GMM_parameterization, sampling_imp_vars, verbose,
                                seed) {

                # print(exp_matrix[,1:20])            
                # print(type(exp_matrix))  # double
                # print(class(exp_matrix))  # data.frame
                # print(dim(exp_matrix))  # channels x events
                
                # Change dtype from "data.frame" to "matrix, array" andadd marker names
                exp_matrix <- as.matrix(exp_matrix)
                rownames(exp_matrix) <- marker_names
            
                # print(exp_matrix[,1:20])            
                # print(type(exp_matrix))  # double
                # print(class(exp_matrix))  # matrix, array
                # print(dim(exp_matrix))  # channels x events

                # print(labels[1:20])
                # print(type(labels))  # character
                # print(class(labels))  # array
                
                # Change dtype from "array" to "factor"
                labels <- factor(labels)

                # print(labels[1:20])
                # print(type(labels))  # character
                # print(class(labels))  # factor

                # GateMeClass training
                gate <- GateMeClass_train(
                    reference = exp_matrix,
                    labels = labels,
                    GMM_parameterization = GMM_parameterization,
                    sampling_imp_vars = sampling_imp_vars,
                    verbose = verbose,
                    seed = seed
                )

                # reference             : The expression matrix of the reference annotated dataset.
                # labels                : A character vector with the labels of the reference dataset.
                # GMM_parameterization  : A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
                # sampling_imp_vars     : Fraction of the cells per cell type sampled to determine the important markers.
                # verbose               : TRUE to show output information.
                # seed                  : Seed for randomization.

                # print(type(gate))  # character
                # print(class(gate))  # data.frame

                return(gate)

            }
            ''')

        r_gmc_fit = ro.globalenv['gmc_fit']

        # numpy-/pandas-to-R conversion inside a local converter context
        # (numpy2ri/pandas2ri.activate() raise an error in rpy2 >= 3.6)
        with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():

            # ### Convert the Python train data into R objects
            # r_x_df = pandas2ri.py2rpy(x_df.copy())  # Do not use pandas2ri here, leads to memory error
            r_x = numpy2ri.py2rpy(X.T)
            marker_names = StrVector(self.marker_names_correct_format_)
            r_y = numpy2ri.py2rpy(y.astype(str).copy())  # GMC needs character vector as label input

            # Define a dict with the GMC hyperparameters
            fit_params = {
                "GMM_parameterization": self.gmc_gmm_parameterization,
                "sampling_imp_vars": self.gmc_sampling_imp_vars,
                "verbose": self.verbosity >= 1,
                "seed": self.gmc_seed,
            }

            # Call the R fit function => marker table, store as class attribute (Pandas dataframe)
            if self.time_fit_pred:
                st = time.time()

            # The converter context returns the R data.frame as a pandas DataFrame (picklable)
            self.marker_table_ = r_gmc_fit(r_x, r_y, marker_names, **fit_params)

            if self.time_fit_pred:
                et = time.time()
                if self.time_df_ is None:
                    self.time_df_ = pd.DataFrame(index=['times', ])
                self.time_df_[f'fit_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'] = et - st


        self.is_fitted_ = True

        return self

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Annotate the train data matrix with the marker names
        # x_df = pd.DataFrame(
        #     X.T,
        #     index=self.marker_names_correct_format_,
        # )

        # Load the GateMeClass R package
        ro.r('''
            library(GateMeClass)
            ''')

        # Define the R function for GateMeClass training and annotation
        ro.r('''
            gmc_predict <- function(
                exp_matrix,
                marker_names, 
                marker_table, 
                reject_option, 
                GMM_parameterization, 
                k, 
                sampling, 
                verbose, 
                seed){

                # print(exp_matrix[,1:20])            
                # print(type(exp_matrix))  # double
                # print(class(exp_matrix))  # matrix, array
                # print(dim(exp_matrix))  # channels x events
                
                # Change dtype from "data.frame" to "matrix, array" and add marker names
                exp_matrix <- as.matrix(exp_matrix)
                rownames(exp_matrix) <- marker_names

                # print(exp_matrix[,1:20])            
                # print(type(exp_matrix))  # double
                # print(class(exp_matrix))  # matrix, array
                # print(dim(exp_matrix))  # channels x events

                # print(marker_table)
                # print(type(marker_table))  # character
                # print(class(marker_table))  # data.frame

                # ### GateMeClass annotation
                res <- GateMeClass_annotate(
                    exp_matrix = exp_matrix,
                    marker_table = marker_table,
                    reject_option = reject_option,
                    GMM_parameterization = GMM_parameterization,
                    k = k,
                    sampling = sampling,
                    verbose = verbose,
                    narrow_marker_table = TRUE,  # The fit function returns a narrow marker table
                    seed = seed
                )

                # exp_matrix          : An expression matrix.
                # marker_table        : A data.frame with a manually curated or extracted marker table.
                # reject_option       : If TRUE this parameter tries to detect cell types not defined in the marker table using MNN algorithm.
                # GMM_parameterization: A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
                # k                   : k parameter of k-NN (k-Nearest-Neighbour) used to refine uncertain labels to the most similar already annotated.
                # sampling            : Percentage of the cells used for the annotation.
                # narrow_marker_table : format of marker table.
                # verbose             : TRUE to show output information.
                # train_parameters    : A list with the parameters for the training function, i.e., reference and labels.
                # seed                : Seed for randomization.

                # print(type(res$labels))  # character
                # print(class(res$labels))  # character
                # print(type(res$marker_table))  # character
                # print(class(res$marker_table))  # data.frame
                # print(type(res$cell_signatures))  # character
                # print(class(res$cell_signatures))  # data.frame

                # Return results
                out <- list(
                    labels = res$labels,
                    marker_table = res$marker_table,
                    cell_signatures = res$cell_signatures
                )
                
                return(out)
            }
            ''')

        gmc_predict = ro.globalenv['gmc_predict']

        # numpy-/pandas-to-R conversion inside a local converter context
        # (numpy2ri/pandas2ri.activate() raise an error in rpy2 >= 3.6)
        with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():

            # Convert the Python train data into R objects
            # r_x_df = pandas2ri.py2rpy(x_df.copy())
            r_x = numpy2ri.py2rpy(X.T)
            marker_names = StrVector(self.marker_names_correct_format_)

            # Get the previously inferred marker_table and convert it into an R dataframe
            r_marker_table = pandas2ri.py2rpy(self.marker_table_)

            # Define a dict with the GMC hyperparameters
            predict_params = {
                "reject_option": self.gmc_reject_option,
                "GMM_parameterization": self.gmc_gmm_parameterization,
                "k": self.gmc_k,
                "sampling": self.gmc_sampling,
                "verbose": self.verbosity >= 1,
                "seed": self.gmc_seed,
            }

            # Call the R predict function => (y_pred, marker_table, cell signatures)
            if self.time_fit_pred:
                st = time.time()

            r_y_pred, _, _ = gmc_predict(r_x, marker_names, r_marker_table, **predict_params)

            if self.time_fit_pred:
                et = time.time()
                if self.time_df_ is None:
                    self.time_df_ = pd.DataFrame(index=['times', ])
                self.time_df_[f'pred_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'] = et - st

            y_pred = r_y_pred  # numpy2ri automatically converts back to numpy array
            y_pred[y_pred == 'Unclassified'] = '-1.0'
            y_pred = y_pred.astype(float).astype(int)

            # Get vector with the original labels
            y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

            if np.any(y_pred == -1) and self.verbosity >= 1:
                warnings.warn(
                    f"{int(np.sum(y_pred == -1))} events were not classifiable with the inferred marker table. "
                    f"They are classified as label -1 ~= 'new cell type'",
                    UserWarning)


        return y_pred

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
            filename: str = 'gatemeclass_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(
            cls: type(T),
            filename: str = 'gatemeclass_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> T:
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

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

    @staticmethod
    def _check_marker_name_format(marker_names: List[str], n_markers: int) -> List[str]:

        if marker_names is None:
            processed_marker_names = [f'Marker{i}' for i in range(1, n_markers + 1)]
        else:
            # Marker names should not contain space and start with a capital (not a number)
            processed_marker_names = []
            for mn in marker_names:
                # Remove spaces
                mn = mn.replace(" ", "")
                # Replace dashes with underscores
                mn = mn.replace("-", "_")
                # Capitalize the first character
                if mn and mn[0].isalpha():
                    mn = mn[0].capitalize() + mn[1:]
                else:
                    # Add 'Z' if the first character is not a letter
                    mn = "Z_" + mn
                processed_marker_names.append(mn)

        return processed_marker_names

