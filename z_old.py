
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Tuple, Union, Literal
from scipy.sparse import spmatrix


class FlowDataset(Dataset):
    def __init__(
            self,
            data: Union[str, sc.AnnData, np.ndarray],
            on_disk: bool = True,
    ):
        assert isinstance(data, str) or isinstance(data, sc.AnnData) or isinstance(data, np.ndarray), \
            "'data' must be path to data file (.h5ad or .npy) or AnnData object or Numpy array"

        self.on_disk = on_disk

        if isinstance(data, str):
            self.file_path = data
            # ### Set flag if working on .npy file or on .h5ad file
            assert self.file_path.endswith('.npy') or self.file_path.endswith('.h5ad'), \
                'Input file must be .npy or .h5ad'
            self.np_data = self.file_path.endswith('.npy')
            # ### Load data in previously defined mode
            if self.np_data:
                if self.on_disk:
                    self.data = np.load(self.file_path)
                else:
                    self.data = np.memmap(self.file_path, mode='r')
            else:
                if self.on_disk:
                    self.data = sc.read_h5ad(self.file_path, backed='r')
                else:
                    self.data = sc.read_h5ad(self.file_path)

        elif isinstance(data, sc.AnnData):
            self.np_data = False
            self.data = data
        elif isinstance(data, np.ndarray):
            self.np_data = True
            self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):

        # Retrieve the specific event (row) as a numpy array
        if self.np_data:
            event = self.data[idx, :]
        else:
            event = self.data.X[idx, :]
            # Convert sparse to dense if needed
            if isinstance(event, spmatrix):
                event = event.toarray()
            event = event.squeeze()

        return event

    def __getitems__(self, idxs: Sequence[int]):

        # Retrieve the specific events (rows) as a numpy array
        if self.np_data:
            events = self.data[idxs, :]
        else:
            events = self.ad.X[idxs, :].toarray()
            if isinstance(events, spmatrix):
                events = events.toarray()

        return events


# ### Unnecessary, would need to store majority vote etc as well
    def _load_som_codebook(
            self,
            filename: str,
            filepath: Union[str, None] = None,
    ) -> None:
        # Load codebook ~ numpy array
        if filepath is None:
            filepath = os.getcwd()
        self.initial_codebook = np.load(os.path.join(filepath, filename))

        # Initialize SOM
        self.som = Somoclu(
            n_columns=self.som_dimensions[0],
            n_rows=self.som_dimensions[1],
            gridtype=self.som_grid_type,
            maptype=self.som_topology,
            neighborhood=self.neighborhood,
            std_coeff=self.gaussian_neighborhood_sigma,
            initialization=self.initialization,
            initialcodebook=self.initial_codebook,
            kerneltype=self.kernel_type,
            verbose=self.verbosity,
        )

        # Reset all variables associated with a trained SOM classifier
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None

    def _save_som_codebook(
            self,
            filename: str = 'codebook.npy',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        np.save(os.path.join(filepath, filename), self.som.codebook)


num_classes = classes.shape[0]

# Precision, Recall, F1-Score for each class
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

# Averages
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# AUC-ROC for each class (one-vs-rest)
auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')

# Precision-Recall Curve & Average Precision Score for each class
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
    avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

# Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

import os
import warnings
import pickle
import copy
import time
import flowio
import numpy as np
import pandas as pd

from typing import Literal, Tuple, Union, List, Self, Dict, Callable, Iterable, Any
from math import log, exp
from somoclu import Somoclu
from umap import UMAP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from scipy.spatial.distance import cdist
from numba import njit, prange


class SomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            confidence_threshold: float = 0.0,
            som_topology: Literal['planar', 'toroid'] = 'planar',
            som_grid_type: Literal['rectangular', 'hexagonal'] = 'rectangular',
            som_dimensions: Tuple[int, int] = (10, 10),
            neighborhood: Literal['gaussian', 'bubble'] = 'gaussian',
            gaussian_neighborhood_sigma: Union[float, None] = 1.0,
            initialization: Literal['random', 'pca'] = 'pca',
            initial_codebook: Union[np.ndarray, None] = None,
            n_epochs: int = 100,
            radius_0: float = 0.0,
            radius_n: float = 1.0,
            radius_cooling: Literal['linear', 'exponential'] = 'linear',
            learning_rate_0: float = 0.1,
            learning_rate_n: float = 0.01,
            learning_rate_decay: Literal['linear', 'exponential'] = 'linear',
            kernel_type: int = 0,  # 0 ~= CPU, 1 ~= GPU
            # cores: Union[List[int], None] = None,
            unlabeled_label: Any = -999,
            verbosity: int = 0,
    ):
        super().__init__()
        # Initialize parameters
        self._confidence_threshold = confidence_threshold
        self.som_topology = som_topology
        self.som_grid_type = som_grid_type
        self.som_dimensions = som_dimensions
        self.neighborhood = neighborhood
        self.gaussian_neighborhood_sigma = gaussian_neighborhood_sigma
        self.initialization = initialization
        self.initial_codebook = initial_codebook
        self.kernel_type = kernel_type
        # self.cores = cores
        # if self.cores is None:
        #     # Use half of all available cores
        #     n_cpus = ceil(multiprocessing.cpu_count() / 2)
        #     self.cores = list(range(n_cpus))
        #     logger.info(f'# ### The cores 0-{multiprocessing.cpu_count() - 1} are available, '
        #                 f'using half of them (0-{n_cpus})')
        # Set CPU affinity
        # self.set_cpu_affinity()
        self.unlabeled_label = unlabeled_label
        self.verbosity = verbosity

        # Initialize training specific parameters:
        self.n_epochs = n_epochs
        self.radius_0 = radius_0
        self.radius_n = radius_n
        self.radius_cooling = radius_cooling
        self.learning_rate_0 = learning_rate_0
        self.learning_rate_n = learning_rate_n
        self.learning_rate_decay = learning_rate_decay
        # If radius_0 < 0 set radius_0 relative to the SOM dimensions

        # Initialize all variables associated with a trained SOM classifier
        self.is_fitted_ = False
        self.labeled_data_ = False
        self.som_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.class_counts_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None

        # Epoch wise training
        self.epoch_wise_som_training_metrics_ = None  # Only relevant if SOM is trained epoch wise

        # Hyperparameter tuning
        self.grid_search_ = None

    # ### _initialize_som(), _set_radius0() ############################################################################
    def _initialize_som(self):
        self.som_ = Somoclu(
            n_columns=self.som_dimensions[0],
            n_rows=self.som_dimensions[1],
            gridtype=self.som_grid_type,
            maptype=self.som_topology,
            neighborhood=self.neighborhood,
            std_coeff=self.gaussian_neighborhood_sigma,
            initialization=self.initialization,
            initialcodebook=copy.deepcopy(self.initial_codebook),
            kerneltype=self.kernel_type,
            verbose=self.verbosity,
        )

    def _set_radius_0(self):
        if self.radius_0 < 0:
            self.radius_0 = min(self.som_dimensions[0], self.som_dimensions[1]) * abs(self.radius_0)

    # ### fit(), predict(), predict_proba(), annotate_and_export() #####################################################
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> Self:

        # Check input data format
        X, y = check_X_y(X, y)

        # Check if .fit() was called already
        if self.is_fitted_:
            # Raise a ValueError if the input data does not match the dimension of the previous input data
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features as per previous training, but got {X.shape[1]}."
                )

            # Set parameters to train with the new data and the codebook from the previously trained SOM
            self.initialization = None
            self.initial_codebook = np.copy(self.som_.codebook)
            warnings.warn(
                "The .fit() method was called on an already trained SOM classifier. "
                "Training will continue with the new data and the existing codebook. "
                "To restart training from scratch, call .reset() before calling .fit().",
                UserWarning
            )

        self.n_features_in_ = X.shape[1]

        # Initialize the SOM
        self._initialize_som()

        # Set the initial radius parameter of the SOM (negative values are interpreted as fractions of the grid size)
        self._set_radius_0()

        # Train the SOM on all data (unsupervised)
        self.som_.train(
            data=X,
            epochs=self.n_epochs,
            radius0=self.radius_0,
            radiusN=self.radius_n,
            radiuscooling=self.radius_cooling,
            scale0=self.learning_rate_0,
            scaleN=self.learning_rate_n,
            scalecooling=self.learning_rate_decay,
        )

        # Get labeled data from the input data
        x_labeled, y_labeled = SomClassifier._get_unlabeled(
            X=X,
            y=y,
            nan_val=self.unlabeled_label,
            verbosity=self.verbosity
        )

        # Check whether labeled data was part of the input, if not raise UserWarning
        if y_labeled.shape[0] >= 1:

            # Todo: warning

            # Rename classes to integers starting from 0 and extract label information
            if not self.is_fitted_ or (self.is_fitted_ and not self.labeled_data_):
                (
                    y_labeled, self.classes_, self.class_counts_, self.class_priors_, self.new_to_og_classes_dict_,
                    self.og_classes_
                ) = SomClassifier._process_class_labels(y=y_labeled)

            # If the SOM classifier was trained previously on labeled data, extract and align label information
            else:
                y_labeled, n_new_classes = self._merge_new_class_label_info(y=y_labeled)

            # ### Determine the majority class for all SOM units
            # Get BMUs for train data, shape n_events x 2 = som_coordinates
            # Note: surface_state = activation_map = codebook * data_matrix
            bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=x_labeled))

            # Append label vector to BMUs
            bmus = np.column_stack((bmus, y_labeled))

            # If .fit() has not been called before or the SOM classifier was only trained on unlabeled data,
            # initialize the class counts array, shape: (somdim0, somdim1, n_classes)
            if not self.is_fitted_ or (self.is_fitted_ and not self.labeled_data_):
                self.class_counts_per_unit_ = np.zeros(self.som_dimensions + (self.classes_.shape[0],))

            # Else, extend the count array to accommodate any new classes
            else:
                extension = np.zeros(self.som_dimensions + (n_new_classes,))
                self.class_counts_per_unit_ = np.concatenate((self.class_counts_per_unit_, extension), axis=-1)

            # Count the occurrences of rows ~= How often is unit (i,j) BMU for a sample with label k
            np.add.at(
                self.class_counts_per_unit_,
                (bmus[:, 0], bmus[:, 1], bmus[:, 2]),
                1,
            )

            # Assign majority class labels per SOM unit
            self.som_unit_labels_ = np.argmax(self.class_counts_per_unit_, axis=2)

            # Account for the case that the support of some units is 0, raise UserWarning if this case occurs
            support = self.class_counts_per_unit_.sum(axis=2)
            zero_support_bool = support == 0
            if np.any(zero_support_bool):
                self.som_unit_labels_[zero_support_bool] = -1
                warnings.warn(
                    f"{zero_support_bool.sum()} SOM nodes are not BMU for any training data: "
                    f"{[(int(i), int(j)) for i, j in np.argwhere(zero_support_bool)]}",
                    UserWarning
                )

            # Set flag indicating that SOM classifier was trained on labeled data
            self.labeled_data_ = True

        elif self.is_fitted_ and self.labeled_data_:
            # Todo
            warnings.warn(
                'No labeled data provided for the current call of .fit(). Please call .fit() again withe the previously  '
                'To manually annotate labels in Kaluza, '
                'extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        else:
            # Todo: change warning
            warnings.warn(
                'No labeled data provided—label prediction is not possible. '
                'To manually annotate labels in Kaluza, '
                'extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        # Set flag indicating that SOM classifier is fitted
        self.is_fitted_ = True

        return self

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # If SOM Classifier was trained on labeled data compute prediction
        if self.labeled_data_:

            # Get BMU of events
            bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

            # Get prediction as label of BMU of event
            y_pred = self.som_unit_labels_[tuple(bmus[:, 0]), tuple(bmus[:, 1])]

            # Get original labels
            y_pred = np.vectorize(self.new_to_og_classes_dict_.get)(y_pred)

            if self._confidence_threshold == 0:
                # Raise UserWarning if label less-unit is BMU at prediction time
                if np.any(y_pred == -1):
                    warnings.warn(
                        f"For events {np.argwhere(y_pred == -1).flatten().tolist()} the BMU has no label "
                        f"(support of the unit during training was 0). "
                        f"Its predicted class is -1 ~= unknown/undeterminable from the training data",
                        UserWarning)
            else:
                # If the confidence is below the threshold change prediction to 'unknown' = -1
                # Calculate the fraction of each class at each SOM unit
                unit_wise_class_probabilities = np.divide(
                    self.class_counts_per_unit_,
                    self.class_counts_per_unit_.sum(axis=2, keepdims=True),
                    where=self.class_counts_per_unit_.sum(axis=2, keepdims=True) != 0
                )
                y_proba = unit_wise_class_probabilities[tuple(bmus[:, 0]), tuple(bmus[:, 1]), :]
                y_proba_max = y_proba.max(axis=1)
                low_confidence_bool = (y_proba_max <= self._confidence_threshold)
                y_pred[low_confidence_bool] = -1

        # No labeled training data, return a dummy prediction vector and raise UserWarning
        else:
            y_pred = np.full(X.shape[0], self.unlabeled_label)
            warnings.warn(
                'No labeled data provided—label prediction is not possible. '
                'To manually annotate labels in Kaluza, '
                'extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        #  Confidence in prediction based on class distribution of events with unit as a bmu

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # If SOM Classifier was trained on labeled data compute prediction probabilities
        if self.labeled_data_:
            # Get BMUs of events
            bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

            # Calculate the fraction of each class at each SOM unit
            unit_wise_class_probabilities = np.divide(
                self.class_counts_per_unit_,
                self.class_counts_per_unit_.sum(axis=2, keepdims=True),
                where=self.class_counts_per_unit_.sum(axis=2, keepdims=True) != 0
            )

            y_proba = unit_wise_class_probabilities[tuple(bmus[:, 0]), tuple(bmus[:, 1]), :]

        # No labeled training data, return a dummy prediction probability vector and raise UserWarning
        else:
            y_proba = np.full((X.shape[0], 1), self.unlabeled_label)
            warnings.warn(
                'No labeled data provided—label prediction is not possible. '
                'To manually annotate labels in Kaluza, '
                'extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        return y_proba

    def annotate_som(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ):

        return

    def annotate_and_export(
            self,
            X: Union[List[np.ndarray], np.ndarray],
            X_raw: Union[List[np.ndarray], np.ndarray, List[pd.DataFrame], pd.DataFrame, None] = None,
            val_range: Union[Tuple[float, float], None] = (0.0, 2 ** 20),
            keep_unscaled: bool = False,
            channel_names: Union[List[str], None] = None,
            sample_ids: Union[List[int], None] = None,
            compute_umap: bool = False,
            umap_kwargs: Union[Dict, None] = None,
            save_mode: Literal['fcs', 'csv', 'no_save'] = 'no_save',
            fcs_metadata_dict: Union[Dict, None] = None,
            save_path: Union[str, None] = None,
            filename: Union[str, None] = None,
    ):

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Turn array input into list
        if not isinstance(X, List):
            X = [X, ]

        if not isinstance(X_raw, List):
            if X_raw is not None:
                X_raw = [X_raw, ]
            else:
                X_raw = [None, ] * len(X)

        if X_raw is not None and len(X_raw) != len(X):
            raise ValueError("'X' and 'X_raw' must have the same length.")

        # Define sample ids
        if sample_ids is not None:
            if len(sample_ids) != len(X):
                raise ValueError("'sample_ids' and 'X' must have the same length.")
        else:
            sample_ids = [i for i in range(len(X))]

        # Define channel names
        num_channels = X[0].shape[1] if X_raw[0] is None else X_raw[0].shape[1]
        if channel_names is not None:
            if len(channel_names) != num_channels:
                raise ValueError(
                    "Length of 'channel_names' must match number of dimensions of 'X' in axis 1 "
                    "(or of 'X_raw' if passed)."
                )
        else:
            if X_raw[0] is not None:
                channel_names = X_raw[0].columns.tolist()
            else:
                channel_names = [f'channel{i}' for i in range(X[0].shape[1] if X_raw[0] is None else X_raw[0].shape[1])]

        # Annotate the individual data matrices
        fcs_dfs = [pd.DataFrame()] * len(X)
        for i, (x, x_raw, s_id) in enumerate(zip(X, X_raw, sample_ids)):
            fcs_dfs[i] = self._x_to_fcs_style_df(X=x, X_raw=x_raw, channel_names=channel_names, sample_id=s_id)

        # Concatenate the annotated matrices
        fcs_df = pd.concat(fcs_dfs, axis=0, ignore_index=True)

        # Add annotations for better visualization of the SOM in Kaluza
        fcs_df = self._add_visualization_annotations(fcs_df=fcs_df)

        # Compute UMAP embedding of the data
        if compute_umap:
            umap_reducer = UMAP(**umap_kwargs if umap_kwargs is not None else {})
            umap_embedding = umap_reducer.fit_transform(np.concatenate(X))  # Shape: n samples x 2
            fcs_df['umap1'] = umap_embedding[:, 0]
            fcs_df['umap2'] = umap_embedding[:, 1]

        # Scale all entries of the data matrix to a given interval
        if val_range is not None:
            scaled_data = SomClassifier._scale_column_wise(x=fcs_df.to_numpy(), val_range=val_range)
            if keep_unscaled:
                scaled_df = pd.DataFrame(
                    data=scaled_data,
                    index=fcs_df.index,
                    columns=[f'{col}_scaled' for col in fcs_df.columns.tolist()],
                )
                fcs_df = pd.concat([fcs_df, scaled_df], axis=1)
            else:
                fcs_df = pd.DataFrame(
                    data=scaled_data,
                    index=fcs_df.index,
                    columns=fcs_df.columns,
                )

        # Save to .fcs or .csv format
        if save_mode != 'no_save':
            if save_path is None:
                save_path = os.getcwd()

            if save_mode == 'fcs':
                if filename is None:
                    filename = 'som.fcs'
                with open(os.path.join(save_path, filename), 'wb') as f:
                    flowio.create_fcs(
                        file_handle=f,
                        event_data=fcs_df.to_numpy().flatten().tolist(),
                        channel_names=fcs_df.columns.tolist(),
                        opt_channel_names=fcs_df.columns.tolist(),
                        metadata_dict=fcs_metadata_dict,
                    )

            else:
                if filename is None:
                    filename = 'som.csv'

                fcs_df.to_csv(os.path.join(save_path, filename))

        return fcs_df

    # ### hyperparameter_tuning(), fit_epoch_wise() ####################################################################
    def hyperparameter_tuning(
            self,
            X: np.ndarray,
            y: np.ndarray,
            param_grid: Union[Dict, None] = None,
            cv: Union[int, BaseCrossValidator, Iterable, None] = 5,
            scoring: Union[str, Callable, List, Tuple, Dict, None] = 'internal',
            refit: Union[bool, str, Callable] = True,
            # None or 'internal' -> internal score (macro F1) is used
            # Other scores are:
            # https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring-parameter
            gridsearchcv_kwargs: Union[Dict, None] = None,
            # 'n_jobs', 'pre_dispatch', 'error_score', 'return_train_score'
    ) -> Self:

        # Set a default parameter grid if none is provided
        if param_grid is None:
            param_grid = {
                'som_dimensions': [(10, 10), (20, 20)],
                'n_epochs': [50, 100],
                'radius_cooling': ['linear', 'exponential'],
                'learning_rate_decay': ['linear', 'exponential']
            }

        # If no kwargs are passed set to empty dictionary
        if gridsearchcv_kwargs is None:
            gridsearchcv_kwargs = {}

        # Set scoring to None if internal score should be used
        if scoring == 'internal':
            scoring = None

        # Initialize GridSearchCV with self as the base estimator
        self.grid_search_ = GridSearchCV(
            estimator=self,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=refit,
            verbose=self.verbosity,
            **gridsearchcv_kwargs,
        )

        # Fit GridSearchCV on the data
        if self.verbosity >= 1:
            print('# ### Starting Gridsearch ...')
        self.grid_search_.fit(X, y)

        if self.verbosity >= 1:
            print(f'# ### Best parameters: {self.grid_search_.best_params_}')
            print(f'# ### Best score: {self.grid_search_.best_score_}')

        # Update the instance's parameters with the best found parameters
        if refit is not False:
            # Get the attributes of the best classifier and update the Som classifier instance with them
            best_classifier_attribute_dict = copy.deepcopy(self.grid_search_.best_estimator_.__dict__)
            # Remove the gridsearch attribute, such that it remains unchanged
            best_classifier_attribute_dict.pop('grid_search_')
            self.__dict__.update(best_classifier_attribute_dict)

        # Return self with updated parameters
        return self

    def fit_with_checkpoints(
            self,
            X: np.ndarray,
            y: np.ndarray,
            checkpoint_interval: Union[int, None] = 100,
            tracking_interval: Union[int, None] = None,  # 10
            track_losses: bool = False,  # som_metrics
            track_impurities: bool = False,
            X_y_val: Union[Tuple[np.ndarray, np.ndarray], None] = None,
            save_res_dict: bool = False,
            filename: Union[str, None] = None,
            filepath: Union[str, None] = None,
    ) -> Self:
        # ### Train two epochs at the time and set decaying parameters manually, calculate epoch-wise validation metrics

        # Todo:
        #  - rename to train with checkpoints (optional performance tracking options)
        #  - refactor to accommodate changes of other functions (unlabeled data etc, best use fit function with restart)
        #  - possibly also debug (maybe restart was false due to initialization parameter)
        #  - at the same time test purity etc (unlabeled data case)

        if checkpoint_interval < 2 or checkpoint_interval >= self.n_epochs:
            raise ValueError("'checkpoint_interval' must be greater or equal to 2 and smaller than 'n_epochs'")

        if tracking_interval < 2 or tracking_interval >= self.n_epochs:
            raise ValueError("'checkpoint_interval' must be greater or equal to 2 and smaller than 'n_epochs'")

        if self.radius_cooling != 'linear' or self.learning_rate_decay != 'linear':
            warnings.warn(
                "For checkpointed training it is recommended to use 'linear' decay functions for the radius "
                "and learning rate not 'exponential'", UserWarning)

        if track_losses or track_impurities and tracking_interval is None:
            tracking_interval = 10

        if save_intermediate_interval is not None:
            assert save_intermediate_interval % tracking_interval == 0, \
                "'save_intermediate_interval' must be an integer multiple of 'tracking_interval'"

        n_checkpoint_intervals = self.n_epochs // checkpoint_interval
        i_checkpoints = [(i + 1) * checkpoint_interval for i in range(n_checkpoint_intervals)]
        epochs_left_checkpoint = self.n_epochs % checkpoint_interval

        i_stop = np.array(i_checkpoints)  # Points where training is stopped for checkpointing or tracking

        if track_impurities is not None:
            n_tracking_intervals = self.n_epochs // tracking_interval
            i_tracking = [(i + 1) * tracking_interval for i in range(n_tracking_intervals)]
            epochs_left_tracking = self.n_epochs % tracking_interval
            i_stop = np.sort(np.unique(np.concatenate(np.array(i_tracking), i_stop)))

        if intervals_left == 1:
            print(f"# ### n_epochs % tracking_interval == 1, last epoch is omitted")

        n_total_tracks = n_tracking_intervals + 1 if intervals_left > 1 else n_tracking_intervals

        X, y = check_X_y(X, y)

        self._initialize_som()

        y_og = y.copy()
        self.n_features_in_ = X.shape[1]
        # Rename classes to integers starting from 0
        y, self.classes_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_ = \
            SomClassifier._process_class_labels(y=y)

        # Set initial radius
        if self.radius_0 == 0:
            self.radius_0 = min(self.som_dimensions[0], self.som_dimensions[1]) / 2

        # Get decay functions
        decay_fct_radius = SomClassifier._get_decay_function(
            val0=self.radius_0, val1=self.radius_n, n_epochs=self.n_epochs, strategy=self.radius_cooling)
        decay_fct_lr = SomClassifier._get_decay_function(
            val0=self.learning_rate_0, val1=self.learning_rate_n, n_epochs=self.n_epochs,
            strategy=self.learning_rate_decay)

        if track_losses:
            quantization_loss = np.zeros(n_total_tracks)
            topographical_loss = np.zeros(n_total_tracks)
        if track_impurities:
            entropies = np.zeros((self.som_dimensions[0], self.som_dimensions[1], n_total_tracks))
            ginies = np.zeros((self.som_dimensions[0], self.som_dimensions[1], n_total_tracks))
        if val_data is not None:
            X_val, y_val = check_X_y(val_data[0], val_data[1])
            if track_losses:
                quantization_loss_val = np.zeros(n_total_tracks)
                topographical_loss_val = np.zeros(n_total_tracks)
            f1_class_wise_train = np.zeros((self.classes_.shape[0], n_total_tracks))
            f1_class_wise_val = np.zeros((self.classes_.shape[0], n_total_tracks))

        for interval in range(n_total_tracks):
            st = time.time()
            current_epoch = interval * tracking_interval
            if self.verbosity >= 0:
                print(
                    f'# ### Epochs {current_epoch}--{current_epoch + tracking_interval - 1} / {self.n_epochs - 1} ### #')
            current_radius = decay_fct_radius(x=current_epoch)
            current_learning_rate = decay_fct_lr(x=current_epoch)
            self.som_.train(
                data=X,
                epochs=tracking_interval if interval <= n_total_tracks - 2 else intervals_left,
                radius0=current_radius,
                radiusN=current_radius,
                radiuscooling=self.radius_cooling,
                scale0=current_learning_rate,
                scaleN=current_learning_rate,
                scalecooling=self.learning_rate_decay,
            )

            # Only calculate majority classes in every step if necessary for evaluation metrics
            if track_impurities or val_data is not None:
                # ### Determine the majority class for vertices
                # Get BMUs for train data, shape n_events x 2 ~ som_coordinates
                # Note: surface_state = activation_map = codebook * data_matrix
                bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))
                # Append label vector to BMUs
                bmus = np.column_stack((bmus, y))
                # Initialize class counts array, shape (somdim0, somdim1, n_classes)
                self.class_counts_per_unit_ = np.zeros(self.som_dimensions + (self.classes_.shape[0],))
                # Count the occurrences of rows ~= How often is unit (i,j) BMU for a sample with label k
                np.add.at(
                    self.class_counts_per_unit_,
                    (bmus[:, 0], bmus[:, 1], bmus[:, 2]),
                    1,
                )
                # Assign majority class labels per SOM unit
                self.som_unit_labels_ = np.argmax(self.class_counts_per_unit_, axis=2)

            if track_losses:
                quantization_loss[interval] = self.calculate_quantization_error(x=X)
                topographical_loss[interval] = self.calculate_topographic_error(x=X)

            if track_impurities:
                entropies[:, :, interval] = self.calculate_unit_impurity(impurity_measure='entropy')
                ginies[:, :, interval] = self.calculate_unit_impurity(impurity_measure='gini')

            if val_data is not None:
                y_pred_train = self.predict(X=X)
                y_pred_val = self.predict(X=X_val)

                f1_class_wise_train[:, interval] = f1_score(y_og, y_pred_train, average=None)
                f1_class_wise_val[:, interval] = f1_score(y_val, y_pred_val, average=None)

                if track_losses:
                    quantization_loss_val[interval] = self.calculate_quantization_error(x=X_val)
                    topographical_loss_val[interval] = self.calculate_topographic_error(x=X_val)

            if save_intermediate_interval is not None \
                    and (current_epoch + tracking_interval) % save_intermediate_interval == 0 \
                    and interval < n_total_tracks - 1:
                res_dict_intermediate = {'tracking_interval': tracking_interval,
                                         'n_epochs': current_epoch + tracking_interval}
                if track_losses:
                    res_dict_intermediate['q_loss_train'] = quantization_loss[: interval + 1]
                    res_dict_intermediate['t_loss_train'] = topographical_loss[: interval + 1]
                    if val_data is not None:
                        res_dict_intermediate['q_loss_val'] = quantization_loss_val[: interval + 1]
                        res_dict_intermediate['t_loss_val'] = topographical_loss_val[: interval + 1]
                if val_data is not None:
                    res_dict_intermediate['f1_train'] = f1_class_wise_train[:, : interval + 1]
                    res_dict_intermediate['f1_val'] = f1_class_wise_val[:, : interval + 1]
                if track_impurities:
                    res_dict_intermediate['entropy'] = entropies[:, :, :interval + 1]
                    res_dict_intermediate['gini'] = ginies[:, :, :interval + 1]

                if filepath is None:
                    filepath = os.getcwd()
                filename_intermediate = f'res_dict_intermediate_epoch{current_epoch + tracking_interval - 1}.pkl'
                with open(os.path.join(filepath, filename_intermediate), 'wb') as f:
                    pickle.dump(res_dict_intermediate, f)

            et = time.time()
            if self.verbosity >= 1:
                print(f'# ### Epochs {current_epoch}--{current_epoch + tracking_interval - 1} took {et - st} sec')

        # ### Determine the majority class for vertices
        # Get BMUs for train data, shape n_events x 2 ~ som_coordinates
        # Note: surface_state = activation_map = codebook * data_matrix
        # bmus = self.som_.get_bmus(activation_map=self.som_.get_surface_state(data=X))
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        # Append label vector to BMUs
        bmus = np.column_stack((bmus, y))

        # Initialize class counts array, shape (somdim0, somdim1, n_classes)
        self.class_counts_per_unit_ = np.zeros(self.som_dimensions + (self.classes_.shape[0],))

        # Count the occurrences of rows ~= How often is unit (i,j) BMU for a sample with label k
        np.add.at(
            self.class_counts_per_unit_,
            (bmus[:, 0], bmus[:, 1], bmus[:, 2]),
            1,
        )
        # Assign majority class labels per SOM unit
        self.som_unit_labels_ = np.argmax(self.class_counts_per_unit_, axis=2)

        res_dict = {'tracking_interval': tracking_interval, 'n_epochs': self.n_epochs}
        if track_losses:
            res_dict['q_loss_train'] = quantization_loss
            res_dict['t_loss_train'] = topographical_loss
            if val_data is not None:
                res_dict['q_loss_val'] = quantization_loss_val
                res_dict['t_loss_val'] = topographical_loss_val
        if val_data is not None:
            res_dict['f1_train'] = f1_class_wise_train
            res_dict['f1_val'] = f1_class_wise_val
        if track_impurities:
            res_dict['entropy'] = entropies
            res_dict['gini'] = ginies

        self.epoch_wise_som_training_metrics_ = res_dict

        if save_res_dict:
            if filepath is None:
                filepath = os.getcwd()
            if filename is None:
                filename = 'res_dict.pkl'
            with open(os.path.join(filepath, filename), 'wb') as f:
                pickle.dump(res_dict, f)

        self.is_fitted_ = True

        return self

    # ### Scores and performance metrics ###############################################################################
    # score(), activation_frequencies(), quantization_error(), topographic_error(), unit_impurity(), mean_impurity()
    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Union[np.ndarray, None] = None,
    ):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def activation_frequencies(
            self,
            X: np.ndarray,
    ):
        # Get BMUs for the data
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        frequencies = np.zeros(self.som_dimensions, dtype=int)

        # Count the occurrences of rows ~= How often is unit (i,j) BMU for a sample of X
        np.add.at(
            frequencies,
            (bmus[:, 0], bmus[:, 1]),
            1,
        )

        frequencies = frequencies / X.shape[0]

        return frequencies

    def quantization_error(
            self,
            X: np.ndarray,
    ) -> float:
        # Note: mse(x - BMU_vec(x))

        # Get BMUs for the data
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        # Extract BMU codebook vectors
        bmu_vectors = self.som_.codebook[bmus[:, 0], bmus[:, 1]]  # Shape: (n_samples, n_features)

        # Compute Euclidean distances
        quantization_error = np.linalg.norm(X - bmu_vectors, axis=1)

        return quantization_error.mean()

    def topographic_error(
            self,
            X: np.ndarray,
    ) -> float:
        # Note: Count how often the 1st and 2nd BMUs are not adjacent in the trained SOM
        assert self.som_topology == 'planar' and self.som_grid_type == 'rectangular', \
            'Topographical error calculation is currently only implemented for planar and rectangular SOMs'
        surface_state = self.som_.get_surface_state(data=X)
        bmu1 = self.som_.get_bmus(surface_state)
        bmu2 = SomClassifier._get_ith_bmus(surface_state=surface_state, i=2, som_dim0=self.som_dimensions[0])

        adjacency_bool = SomClassifier._check_adjacency(bmus0=bmu1, bmus1=bmu2)

        return np.logical_not(adjacency_bool).mean()

    def unit_impurity(
            self,
            impurity_measure: Literal['entropy', 'gini'] = 'entropy',
    ) -> np.ndarray:

        check_is_fitted(self, 'is_fitted_')

        if self.labeled_data_:
            # Calculate the frequency with which a class occurs at each unit,
            # account for case where unit is not bmu for any by excluding from division
            # frequencies = self.class_counts_per_unit_ / self.class_counts_per_unit_.sum(axis=2, keepdims=True)
            frequencies = np.divide(
                self.class_counts_per_unit_,
                self.class_counts_per_unit_.sum(axis=2, keepdims=True),
                where=self.class_counts_per_unit_.sum(axis=2, keepdims=True) != 0
            )

            # Calculate impurity
            if impurity_measure == 'entropy':
                # Calculate entropy only on nonzero entries
                mask = frequencies > np.finfo(float).eps
                log_freq = np.zeros_like(frequencies)
                log_freq[mask] = np.log2(frequencies[mask])
                impurity = - np.sum(frequencies * log_freq, axis=2)
                # impurity = - np.sum(frequencies * np.where(frequencies > 0, np.log2(frequencies), 0), axis=2)
            else:
                impurity = 1 - np.sum(frequencies * frequencies, axis=2)
        else:
            impurity = np.full(self.som_dimensions, np.inf)
            warnings.warn(
                'No labeled data provided—label prediction is not possible. '
                'To manually annotate labels in Kaluza, '
                'extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        return impurity

    def mean_impurity(
            self,
            impurity_measure: Literal['entropy', 'gini'],
    ) -> float:
        return self.unit_impurity(impurity_measure=impurity_measure).mean()

    def unpredictable_classes(self) -> np.ndarray:

        check_is_fitted(self, 'is_fitted_')

        if self.labeled_data_:
            output_classes = np.sort(
                np.array([self.new_to_og_classes_dict_[key] for key in np.unique(self.som_unit_labels_)]))
            input_classes = np.sort(self.og_classes_)

            only_in_input = np.setdiff1d(input_classes, output_classes)

            if only_in_input.size != 0:
                warnings.warn(
                    f'Trained SOM classifier cannot predict classes {only_in_input} found in the input.'
                )
            else:
                if self.verbosity >= 1:
                    print('Trained SOM classifier can predict all input classes')
        else:
            only_in_input = np.array([])
            warnings.warn(
                'No labeled data provided—label prediction is not possible. '
                'To manually annotate labels in Kaluza, '
                'extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        return only_in_input

    # ### save(), load(), reset(), confidence_threshold() ##############################################################
    def save(
            self,
            filename: str = 'som_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(
            cls,
            filename: str = 'som_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> Self:
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    def reset(self):
        # Initialize all variables associated with a trained SOM classifier
        self.is_fitted_ = False
        self.labeled_data_ = False
        self.som_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.class_counts_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None
        self.epoch_wise_som_training_metrics_ = None
        self.grid_search_ = None

    # 'confidence_threshold' may be changed after training, define property and setter
    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, threshold: float) -> None:
        if threshold < 0 or threshold > 1:
            raise ValueError('confidence_threshold must be between 0 and 1')
        self._confidence_threshold = threshold

    # ### Auxiliary functions ##########################################################################################
    @staticmethod
    def _get_unlabeled(
            X: np.ndarray,
            y: np.ndarray,
            nan_val: Any = np.nan,
            verbosity: int = 0,
    ):
        nan_mask = (y == nan_val)

        x_labeled = X[~nan_mask, :]
        y_labeled = y[~nan_mask]

        if verbosity >= 1:
            n_total = y.shape[0]
            n_labeled = y_labeled.shape[0]
            n_unlabeled = n_total - n_labeled

            print(
                f'# ### Total of {n_total} training events, '
                f'labeled: {n_labeled} ({n_labeled / n_total * 100} %), '
                f'unlabeled: {n_unlabeled} ({n_unlabeled / n_total * 100} %).'
            )

        return x_labeled, y_labeled

    @staticmethod
    def _process_class_labels(
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Any, float], np.ndarray]:
        og_classes, counts = np.unique(y, return_counts=True)

        class_priors = counts / counts.sum()
        new_classes = np.array(list(range(og_classes.shape[0])))
        og_to_new_classes_dict = {key: value for key, value in zip(og_classes, new_classes)}
        y_new = np.vectorize(og_to_new_classes_dict.get)(y)

        new_to_og_classes_dict = {key: value for key, value in zip(new_classes, og_classes)}
        new_to_og_classes_dict[-1] = -1  # Predict label -1 for unclassifiable events

        return y_new, new_classes, counts, class_priors, new_to_og_classes_dict, og_classes

    def _merge_new_class_label_info(
            self,
            y: np.ndarray,
    ):
        # Get unique classes and their count
        classes, counts = np.unique(y, return_counts=True)

        # Identify new previously unseen classes
        unique_new_classes = np.setdiff1d(classes, self.og_classes_)

        if unique_new_classes.size > 0:
            # Update array with og class names, append at end
            self.og_classes_ = np.concatenate((self.og_classes_, unique_new_classes))

            # Update class counts array, append at end
            self.class_counts_ = np.concatenate((self.class_counts_, np.zeros(unique_new_classes.shape[0])))

        # Update the class counts
        for cl, cl_count in zip(classes, counts):
            # Add class count from new data to existing counts
            self.class_counts_[np.where(self.og_classes_ == cl)[0]] += cl_count

        # Update the class priors
        self.class_priors_ = self.class_counts_ / self.class_counts_.sum()

        # Enumerate the original classes starting from 0, this is sound since new classes are appended at the end
        new_classes = np.array(list(range(self.og_classes_.shape[0])))
        og_to_new_classes_dict = {key: value for key, value in zip(self.og_classes_, new_classes)}
        y_new = np.vectorize(og_to_new_classes_dict.get)(y)
        self.classes_ = new_classes

        # Regenerate the dict which is mapping the new class labels to the original ones
        self.new_to_og_classes_dict_ = {key: value for key, value in zip(new_classes, self.og_classes_)}
        self.new_to_og_classes_dict_[-1] = -1  # Predict label -1 for unclassifiable events

        return y_new, unique_new_classes.size

    def _custom_get_surface_state(
            self,
            data: np.ndarray,
    ):
        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> (somdim0 * somdim1, n_features)

        codebook_reshaped = self.som_.codebook.reshape(-1, self.som_.codebook.shape[2])

        if self.som_dimensions[0] <= 10 and self.som_dimensions[1] <= 10 and data.shape[0] <= 100000:
            # ### For small dimension, compute Euclidean distances in chunks for memory efficiency
            # Split data into 200 chunks along axis 0
            num_splits = 200
            chunks = np.array_split(data, num_splits, axis=0)
            # For each chunk compute the Euclidean distance to the codebook, stack results
            activation_map = np.vstack(
                [cdist(chunk, codebook_reshaped, metric='euclidean') for chunk in chunks]
            )
        else:
            # ### For larger dimension, compute Euclidean distances with numba parallelized loop
            activation_map = SomClassifier._compute_distances(data, codebook_reshaped)

        return activation_map

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _compute_distances(data: np.ndarray, codebook: np.ndarray):
        num_samples, num_codebook = data.shape[0], codebook.shape[0]
        distances = np.empty((num_samples, num_codebook), dtype=np.float64)

        for i in prange(num_samples):  # Parallel loop
            for j in range(num_codebook):
                diff = data[i] - codebook[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))

        return distances

    def _custom_get_bmus(
            self,
            activation_map: np.ndarray,
    ):
        # Shape activation map: (n_events, somdim0 * somdim1)

        # ### Find position in the SOM grid of the minimum value for each row
        bmu_indices = np.argmin(activation_map, axis=1)
        j_s, i_s = np.divmod(bmu_indices, self.som_dimensions[1])
        return np.column_stack((i_s, j_s))

    @staticmethod
    def _get_decay_function(
            val0: float,
            val1: float,
            n_epochs: int,
            strategy: Literal['linear', 'exponential'] = 'linear',
    ) -> Callable:
        # ### For train epoch wise, Todo ...
        if strategy == 'linear':
            m = (val1 - val0) / n_epochs
            b = val0

            def _decay_function(x: int):
                return m * x + b
        else:
            assert val0 != 0, "For exponential decay 'val0' cannot be zero"
            rate = log(val1 / val0) / n_epochs

            def _decay_function(x: int):
                return val0 * exp(rate * x)

        # Return decay fct witch params
        return _decay_function

    @staticmethod
    def _get_ith_bmus(
            surface_state: np.ndarray,
            i: int,
            som_dim0: int,
    ) -> np.ndarray:
        # Surface_map is of shape n_events x n_units, from somoclu source code:
        # codebookReshaped = self.codebook.reshape(
        # self.codebook.shape[0] * self.codebook.shape[1], self.codebook.shape[2])
        # From numpy.reshape doc: Index order by default is ‘C’:
        # means to read / write the elements using C-like index order, with the last axis index changing fastest, back
        # to the first axis index changing slowest
        # Let codebook dims be (n, m, k) => (i,j,l) ~= (i * n + j, l) in surface_state
        # Also for surface_state (r, l) => (r // n, r % n, l) in codebook

        # ### Argsort
        # st = time.time()

        # Get idx of ith largest value
        # sorted_indices = np.argsort(surface_state, axis=1)[:, ::-1]
        # ith_largest_indices = sorted_indices[:, i - 1]
        # Convert back to som unit positions
        # i_idxs = ith_largest_indices // som_dim0
        # j_idxs = ith_largest_indices % som_dim0
        # ibmus = np.stack((i_idxs, j_idxs), axis=1)

        # et = time.time()
        # print(f'# ### argsort: {et - st}')

        # ### Argpartition + argsort
        # st = time.time()

        # Extract the top i indices
        # partitioned_indices = np.argpartition(surface_state, -i, axis=1)[:, -i:]
        # Extract the corresponding values
        # top_i_values = surface_state[np.arange(surface_state.shape[0])[:, None], partitioned_indices]
        # Sort these values (descending order)
        # sorted_top_i_indices = np.argsort(-top_i_values, axis=1)
        # Use the sorted indices to get the i-th largest element
        # ith_indices = partitioned_indices[np.arange(surface_state.shape[0]), sorted_top_i_indices[:, i - 1]]
        # Step 4: Convert linear indices to 2D grid coordinates
        # rows = ith_indices // som_dim0
        # cols = ith_indices % som_dim0
        # Combine into BMU coordinates
        # ibmus = np.column_stack((rows, cols))

        # et = time.time()
        # print(f'# ### argpartition + argsort: {et - st}')

        # ### Argpartition
        # st = time.time()

        # ith_indices = np.argpartition(surface_state, -i, axis=1)[:, -i]
        # Step 4: Convert linear indices to 2D grid coordinates
        # rows = ith_indices // som_dim0
        # cols = ith_indices % som_dim0
        # Combine into BMU coordinates
        # ibmus = np.column_stack((rows, cols))

        # et = time.time()
        # print(f'# ### argpartition : {et - st}')

        # ### Iterative max
        # st = time.time()

        # Initialize original indices row-wise
        original_indices = np.tile(np.arange(surface_state.shape[1]), (surface_state.shape[0], 1))
        # Iteratively exclude the largest values row-wise
        for k in range(i - 1):
            # Find the max indices row-wise
            dummy_surface_state = surface_state.copy()

            max_indices = np.argmax(dummy_surface_state, axis=1)
            # Create a mask for all rows
            mask = np.ones_like(surface_state, dtype=bool)
            mask[np.arange(surface_state.shape[0]), max_indices] = False  # Exclude the max values
            # Update surface_state row-wise
            surface_state = surface_state[mask].reshape(surface_state.shape[0], surface_state.shape[1] - 1)
            # Update the original indices row-wise
            original_indices = original_indices[mask].reshape(original_indices.shape[0], original_indices.shape[1] - 1)

        # Get the indices of the i-th largest value row-wise
        ith_indices_new = np.argmax(surface_state, axis=1)
        # Convert to original indices row-wise
        ith_indices = original_indices[np.arange(original_indices.shape[0]), ith_indices_new]
        # Convert linear indices to 2D grid coordinates
        i_idxs = ith_indices // som_dim0
        j_idxs = ith_indices % som_dim0
        # Combine into BMU coordinates
        ibmus = np.column_stack((i_idxs, j_idxs))

        # et = time.time()
        # print(f'# ### Iterative max : {et - st}')

        return ibmus

    @staticmethod
    def _check_adjacency(
            bmus0: np.ndarray,
            bmus1: np.ndarray,
    ) -> np.ndarray:
        # (i0, j0), (i1, j1) adjacent in gris is if: abs(i0 - i1) + abs(j0 - j1) = 1
        return np.abs(bmus1 - bmus0).sum(axis=1) == 1

    def _x_to_fcs_style_df(
            self,
            X: np.ndarray,
            X_raw: Union[np.ndarray, pd.DataFrame, None],
            channel_names: List[str],
            sample_id: int,
    ) -> pd.DataFrame:

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # Generate dataframe from input data, either use X or X_raw
        if X_raw is None:
            x_df = pd.DataFrame(data=X)
        else:
            if isinstance(X_raw, pd.DataFrame):
                x_df = X_raw
            else:
                x_df = pd.DataFrame(data=X_raw)

        # Set the channel names as column names
        x_df.columns = channel_names

        # Annotate the sample id
        x_df['sample_id'] = np.full(x_df.shape[0], sample_id, dtype=int)

        # Annotate the bmu coordinates
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))
        x_df['bmu1'] = bmus[:, 0]
        x_df['bmu2'] = bmus[:, 1]

        # Annotate the SOM unit label of the bmus
        x_df['som_unit_label'] = self._get_row_major_positions(bmus=bmus, start_from=1)

        # Annotate labels
        if self.labeled_data_:
            x_df['label_predicted'] = self.predict(X=X)

        return x_df

    def _get_row_major_positions(
            self,
            bmus: np.ndarray,
            start_from: int = 0,
    ):
        pos = bmus[:, 0] * self.som_dimensions[0] + bmus[:, 1] + start_from
        return pos

    @staticmethod
    def _scale_column_wise(
            x: np.ndarray,
            val_range: Tuple[float, float],
    ) -> np.ndarray:

        # Get column-wise min and max
        col_min = x.min(axis=0)
        col_max = x.max(axis=0)

        # Get scale, avoid zero division in constant columns
        scale = col_max - col_min
        scale[scale == 0] = 1

        x_scaled = (x - col_min) / scale * (val_range[1] - val_range[0]) + val_range[0]

        return x_scaled

    @staticmethod
    def _add_visualization_annotations(
            fcs_df: pd.DataFrame,
    ) -> pd.DataFrame:

        # Get dataframe of structure: 'bmu1', 'bmu2', 'som_unit_label', 'count'
        unit_counts = fcs_df.groupby(['bmu1', 'bmu2', 'som_unit_label']).size().reset_index(name='count')

        # Compute radius for each unit that is proportional to count
        radii = unit_counts['count'].to_numpy()
        radii = (radii - radii.min()) / (radii.max() - radii.min())
        radii = np.sqrt(radii)  # Area should be proportional to count -> use square root
        radii = radii * 0.5  # Radius should be <= 0.5
        unit_counts['radius'] = radii

        fcs_df['bmu1_plot'] = np.zeros(fcs_df.shape[0])
        fcs_df['bmu2_plot'] = np.zeros(fcs_df.shape[0])
        fcs_df['radius'] = np.zeros(fcs_df.shape[0])

        for bmu1, bmu2, som_unit_label, count, radius in zip(
                unit_counts['bmu1'], unit_counts['bmu2'], unit_counts['som_unit_label'], unit_counts['count'],
                unit_counts['radius']
        ):
            x, y = SomClassifier._random_points_on_sphere(x_center=bmu1, y_center=bmu2, radius=radius, n=count)
            mask = fcs_df['som_unit_label'] == som_unit_label
            fcs_df.loc[mask, 'bmu1_plot'] = x
            fcs_df.loc[mask, 'bmu2_plot'] = y
            fcs_df.loc[mask, 'radius'] = radius

        return fcs_df

    @staticmethod
    def _random_points_on_sphere(
            x_center: float,
            y_center: float,
            radius: float,
            n: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Random angles between 0 and 2pi
        theta = np.random.uniform(0, 2 * np.pi, n)

        # Random radii, using square root ensures uniformity in the circle
        random_r = radius * np.sqrt(np.random.uniform(0, 1, n))

        # Convert polar coordinates to Cartesian (x, y)
        x_points = x_center + random_r * np.cos(theta)
        y_points = y_center + random_r * np.sin(theta)

        return x_points, y_points


import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import pytometry as pm
import copy
import os
import logging
import warnings
from typing import Sequence, Tuple, Union, Literal, List
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from math import floor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
# from anndata.experimental import concat_on_disk

# ### Todo:
# - Add more advanced preprocessing
# - Add down-/up-sampling of events per sample to mitigate sample bias
# - Add more functionality to _align_channel_names_helper to incorporate different orderings of channels
# - Add check_sample/patient_bias()


class FlowDataManager:
    def __init__(
            self,
            filename_list: Sequence[str],
            file_type: Union[Literal['fcs', 'csv'], None] = None,
            preprocessing_flavour: Union[Literal['logicle', 'arcsinh', 'biexp', 'autologicle'], None] = None,
            additional_preprocessing_kwargs: Union[dict, None] = None,
            channel_name_reference: Union[int, dict] = 0,
            data_file_path: Union[str, None] = None,
            save_path: Union[str, None] = None,
            base_path: Union[str, None] = None,
            filenames: Union[dict[str, str], None] = None,  # Used to store fns for saving of metadata
            # {'fn_og_channel_names': 'original_channel_names.csv', 'fn_data_split': ...}
            memory_saving: bool = False,
            verbosity: int = 0,
    ):

        # ### Set path variables for data loading and storage
        self.filename_list = filename_list
        self.file_type = file_type
        self.invalid_files = []
        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = os.getcwd()
        if data_file_path is not None:
            self.data_file_path = os.path.join(self.base_path, data_file_path)
        else:
            self.data_file_path = self.base_path
        if save_path is not None:
            self.save_path = os.path.join(self.base_path, save_path)
        else:
            self.save_path = os.path.join(self.base_path, 'fc_data_processed')

        Path(self.save_path).mkdir(exist_ok=True, parents=True)

        self.filenames = filenames if filenames is not None else dict()

        self.memory_saving = memory_saving
        self.verbosity = verbosity

        # ### Load AnnData objects from .fcs files, keep in memory or store on disk based on 'memory_saving'
        self.anndata_list = self._load_data_files_to_anndata()  # Either list of AnnData or list of .h5ad filenames

        # ### Rename channels based on a reference dataset or a passed mapping
        self.channel_name_reference = channel_name_reference
        self.og_channel_names = self.align_channel_names()
        self.check_og_channel_names_df()

        # ### Perform sample wise preprocessing
        self.preprocessing_flavour = preprocessing_flavour
        self.additional_preprocessing_kwargs = additional_preprocessing_kwargs
        if self.additional_preprocessing_kwargs is None:
            self.additional_preprocessing_kwargs = dict()
        if self.preprocessing_flavour is not None:
            self.sample_wise_preprocessing()

        # ### Instantiate lists for train-(val-)test-split of data
        self.train_data = []
        self.test_data = []
        self.val_data = []

    def _load_data_files_to_anndata(self):
        # ### Load AnnData objects into memory and store in list
        anndata_list = []
        if not self.memory_saving:
            for fn in self.filename_list:
                if self.file_type is None:
                    if fn.endswith('.fcs'):
                        self.file_type = 'fcs'
                    elif fn.endswith('.csv'):
                        self.file_type = 'csv'
                    else:
                        warnings.warn(f"Skipping invalid file: '{fn}'. Not a CSV or FCS file.", UserWarning)
                        self.invalid_files.append(fn)
                        continue

                if self.file_type == 'fcs':
                    adata = pm.io.read_fcs(os.path.join(self.data_file_path, fn))
                elif self.file_type == 'csv':
                    df = pd.read_csv(os.path.join(self.data_file_path, fn))
                    adata = sc.AnnData(X=df.to_numpy())
                    adata.var_names = df.columns.copy()
                else:
                    warnings.warn(f"Skipping invalid file: '{fn}'. Not a CSV or FCS file.", UserWarning)
                    self.invalid_files.append(fn)
                    continue

                adata.uns['filename'] = fn
                anndata_list.append(adata)
        # ### Load AnnData objects one by one and save them to disk
        else:
            for fn in self.filename_list:
                if self.file_type is None:
                    if fn.endswith('.fcs'):
                        self.file_type = 'fcs'
                    elif fn.endswith('.csv'):
                        self.file_type = 'csv'
                    else:
                        warnings.warn(f"Skipping invalid file: '{fn}'. Not a CSV or FCS file.", UserWarning)
                        self.invalid_files.append(fn)
                        continue
                if self.file_type == 'fcs':
                    adata = pm.io.read_fcs(os.path.join(self.data_file_path, fn))
                elif self.file_type == 'csv':
                    df = pd.read_csv(os.path.join(self.data_file_path, fn))
                    adata = sc.AnnData(X=df.to_numpy())
                    adata.var_names = df.columns.copy()
                else:
                    warnings.warn(f"Skipping invalid file: '{fn}'. Not a CSV or FCS file.", UserWarning)
                    self.invalid_files.append(fn)
                    continue

                adata.uns['filename'] = fn
                ad_fn = fn[:-4] + '.h5ad'
                adata.write_h5ad(filename=Path(os.path.join(self.save_path, ad_fn)))
                anndata_list.append(ad_fn)
                del adata
        return anndata_list

    def align_channel_names(self) -> pd.DataFrame:
        return FlowDataManager.align_channel_names_worker(
            data_list=self.anndata_list,
            reference=self.channel_name_reference,
            inplace=True,
            file_path=self.save_path,
            save_path=self.save_path,
            out_filename=self.filenames.get('fn_og_channel_names', None)
        )

    @staticmethod
    def align_channel_names_worker(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            reference: Union[int, dict],  # Either int for which file to use as reference or a
            # dictionary with possible_name: reference_name
            inplace: bool = False,
            file_path: Union[str, None] = None,
            save_path: Union[str, None] = None,
            out_filename: Union[str, None] = None,
    ) -> Union[Tuple[Union[Sequence[sc.AnnData], Sequence[str]], pd.DataFrame], pd.DataFrame]:
        # ### Function to unify the channel names across multiple fcs data objects,
        # assumes the same number of channels for all

        # ### Copy input if it should not be altered inplace
        if not inplace:
            data_list = copy.deepcopy(data_list)

        # ### Set flag whether to load data or not
        # (depending on list of AnnData objects or filenames of .h5ad files being passed)
        load_data = isinstance(data_list[0], str)

        if load_data:
            assert all(s.endswith('.h5ad') for s in data_list), "Data files must be .h5ad"

        # ### If idx to reference anndata / file is passed use it to create list of reference channel names
        if isinstance(reference, int):
            if load_data:
                # Load anndata object into memory create list of channel names
                fcdata = sc.read_h5ad(os.path.join(file_path, data_list[reference]))
                reference = fcdata.var_names.values.tolist()
                del fcdata
            else:
                # Create list of channel names on the basis of selected AnnData object
                reference = data_list[reference].var_names.values.tolist()

        # ### Create dataframe to store the original channel names
        if load_data:
            fcdata = sc.read_h5ad(os.path.join(file_path, data_list[0]))
            log_df = pd.DataFrame(columns=['filename'] + list(range(1, fcdata.n_vars + 1)))
            del fcdata
        else:
            log_df = pd.DataFrame(columns=['filename'] + list(range(1, data_list[0].n_vars + 1)))

        # ### Iterate over individual fcs samples and change their channel names
        for i in range(len(data_list)):
            if load_data:
                # Load AnnData object, change channel names, save again
                fldata = sc.read_h5ad(os.path.join(file_path, data_list[i]))
                fldata, log_df = FlowDataManager._align_channel_names_helper(
                    adata=fldata, reference=reference, log_df=log_df)
                log_df.loc[log_df.index[-1], 'filename'] = data_list[i]

                if save_path is not None:
                    fn = data_list[i]
                    fldata.write_h5ad(Path(os.path.join(save_path, fn)))
                del fldata

            else:
                # Change channel names of AnnData object
                _, log_df = FlowDataManager._align_channel_names_helper(
                    adata=data_list[i], reference=reference, log_df=log_df)

                try:
                    log_df.loc[log_df.index[-1], 'filename'] = data_list[i].uns['filename']
                except KeyError:
                    warnings.warn('# ### Key "filename" does not exist in .uns of the AnnData object', UserWarning)

        if save_path is not None:
            if out_filename is None:
                out_filename = 'original_channel_names.csv'
            log_df.to_csv(os.path.join(save_path, out_filename))

        if not inplace:
            return data_list, log_df
        else:
            return log_df

    @staticmethod
    def _align_channel_names_helper(
            adata: sc.AnnData,
            reference: Union[List[str], dict],
            log_df: pd.DataFrame
    ) -> Tuple[sc.AnnData, pd.DataFrame]:

        # Store original var_names to log_df
        log_df.loc[len(log_df)] = [None] + adata.var_names.values.tolist()
        # Store original var_names in separate .var annotation
        adata.var['og_var_names'] = adata.var_names.values.copy()
        if isinstance(reference, list):
            # Replace var_names with list of reference var_names
            adata.var_names = reference
        else:
            # Replace individual var_names with corresponding dict entries
            new_var_names = [None] * adata.n_vars
            for i, vn in enumerate(adata.var_names.values):
                new_var_names[i] = reference[vn]
            adata.var_names = new_var_names
        return adata, log_df

    def check_og_channel_names_df(self) -> None:
        FlowDataManager.check_og_channel_names_df_worker(self.og_channel_names)

    @staticmethod
    def check_og_channel_names_df_worker(og_filenames: pd.DataFrame) -> None:
        for i, col in enumerate(og_filenames.columns):
            if col == 'filename':
                continue

            value_counts = og_filenames[col].value_counts()

            if value_counts.size >= 2:
                msg = f'# ### The channel names for channel {i} were not consistent across samples\n'
                msg += f'# ### Channel: {i} ### #\n'
                for value, count in value_counts.items():
                    msg += f'# ### Name: {value}, Count: {count}\n'
                warnings.warn(msg, UserWarning)

    def sample_wise_preprocessing(
            self,
            save_og_to_layer: Union[str, None] = None,
    ) -> None:
        FlowDataManager.sample_wise_preprocessing_worker(
            data_list=self.anndata_list, flavour=self.preprocessing_flavour, file_path=self.save_path, inplace=True,
            save_og_to_layer=save_og_to_layer, **self.additional_preprocessing_kwargs.copy()
        )

    @staticmethod
    def sample_wise_preprocessing_worker(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            flavour: Literal['logicle', 'arcsinh', 'biexp', 'autologicle'],
            file_path: Union[str, None],
            inplace: bool = False,
            save_og_to_layer: Union[str, None] = None,
            **kwargs,
    ) -> Union[Sequence[sc.AnnData], Sequence[str], None]:

        assert flavour in {'logicle', 'arcsinh', 'biexp', 'custom'}, \
            "'flavour' must be in 'logicle', 'arcsinh', 'biexp', 'autologicle', or 'custom'"

        if not inplace:
            data_list = copy.deepcopy(data_list)

        load_data = isinstance(data_list[0], str)

        if flavour == 'logicle':
            trafo_fct = pm.tl.normalize_logicle
        elif flavour == 'arcsinh':
            trafo_fct = pm.tl.normalize_arcsinh
        elif flavour == 'biexp':
            trafo_fct = pm.tl.normalize_biExp
        elif flavour == 'custom':
            if 'preprocessing_method' not in kwargs:
                raise ValueError(
                    "'preprocessing_method' must be provided in kwargs when 'flavour' is 'custom'."
                )
            trafo_fct = kwargs.pop('preprocessing_method')
        else:
            raise ValueError(f'Unsupported flavour: {flavour}')

        for d in data_list:
            if load_data:
                dummydata = sc.read_h5ad(os.path.join(file_path, data_list[0]))
            else:
                dummydata = d

            # Store unprocessed data matrix in layer
            if save_og_to_layer is None:
                dummydata.layers['original'] = dummydata.X.copy()
            else:
                dummydata.layers[save_og_to_layer] = dummydata.X.copy()

            if kwargs:
                trafo_fct(adata=dummydata, **kwargs)
            else:
                trafo_fct(adata=dummydata)

            if load_data:
                dummydata.write_h5ad(os.path.join(file_path, data_list[0]))
                del dummydata

        if not inplace:
            return data_list

    def perform_data_split(
            self,
            data_split: Union[Tuple[float, float], Tuple[float, float, float], pd.DataFrame] = (0.75, 0.25),
            additional_split_kwargs: Union[dict, None] = None,
            save: bool = False,
    ) -> None:

        if additional_split_kwargs is None:
            additional_split_kwargs = dict()

        dummy_data_split = FlowDataManager.perform_data_split_worker(
            data_list=self.anndata_list,
            data_split=data_split,
            save=save,
            save_kwargs={'filename': self.filenames.get('fn_data_split', 'data_split.csv'), 'filepath': self.save_path},
            verbosity=self.verbosity,
            **additional_split_kwargs)

        if len(dummy_data_split) == 2:
            self.train_data, self.test_data = dummy_data_split
        else:
            self.train_data, self.val_data, self.test_data = dummy_data_split

    @staticmethod
    def perform_data_split_worker(
            data_list: Union[Sequence[str], Sequence[sc.AnnData]],
            data_split: Union[Tuple[float, float], Tuple[float, float, float], pd.DataFrame],
            save: bool = False,
            save_kwargs: Union[dict, None] = None,
            verbosity: int = 0,
            **kwargs  # kwargs for sklearn are: random_state, shuffle, stratify
    ) -> Tuple[Union[Sequence[str], Sequence[sc.AnnData]], ...]:

        # ### Split according to fractions passed as tuple
        if not isinstance(data_split, pd.DataFrame):
            assert len(data_split) == 2 or len(data_split) == 3, \
                "'data_split' must be tuple or triple corresponding with fractions for train- (val-) and test-data"
            assert sum(data_split) == 1 and all(x >= 0 for x in data_split), \
                'The train-(val-)test-split must be passed as a tuple of non negative decimals that sum to one'

            if len(data_split) == 2:
                train_data, test_data = train_test_split(
                    data_list, test_size=data_split[1], train_size=data_split[0], **kwargs)
                if save:
                    FlowDataManager._save_data_split_helper(data_tuple=(train_data, test_data), save_kwargs=save_kwargs)
                return train_data, test_data
            else:
                perc_val_test_data = data_split[1] + data_split[2]
                train_data, val_test_data = train_test_split(
                    data_list, test_size=perc_val_test_data, train_size=data_split[0], **kwargs)
                val_data, test_data = train_test_split(
                    val_test_data, test_size=data_split[2] / perc_val_test_data,
                    train_size=data_split[1] / perc_val_test_data, **kwargs)
                if save:
                    FlowDataManager._save_data_split_helper(
                        data_tuple=(train_data, val_data, test_data), save_kwargs=save_kwargs)
                return train_data, val_data, test_data
        # Split according to previously saved dataframe
        else:
            # 'data_list' is list of AnnData with filename annotated in .uns
            train_data = []
            val_data = []
            test_data = []
            for d in data_list:
                if isinstance(data_list[0], sc.AnnData):
                    mode = data_split.loc[d.uns['filename'][:-4], 'mode']
                else:
                    mode = data_split.loc[d[:-5], 'mode']
                if mode == 'train':
                    train_data.append(d)
                elif mode == 'val':
                    val_data.append(d)
                else:
                    test_data.append(d)
            if len(val_data) == 0:
                if verbosity >= 1:
                    print('# ### The passed data_split dataframe did not include validation data')
                return train_data, test_data
            else:
                return train_data, val_data, test_data

    @staticmethod
    def _save_data_split_helper(
            data_tuple: Tuple[Union[Sequence[str], Sequence[sc.AnnData]], ...],
            save_kwargs: Union[dict, None] = None
    ):
        if save_kwargs is None:
            save_kwargs = dict()
        filename = save_kwargs.get('filename', None)
        if filename is None:
            'data_split.csv'
        filepath = save_kwargs.get('filepath', os.getcwd())

        is_anndata = isinstance(data_tuple[0][0], sc.AnnData)
        fns = []
        modes = []

        def append_filenames_and_modes(
                data_list: Sequence[Union[sc.AnnData, str]],
                m: str,
                is_ad: bool):
            for d in data_list:
                if is_ad:
                    fns.append(d.uns['filename'][:-4])  # fn.fcs -> append fn
                else:
                    fns.append(d[:-5])  # fn.h5ad -> append fn
                modes.append(m)

        # Define modes based on the length of data_tuple
        mode_labels = ['train', 'val', 'test'] if len(data_tuple) == 3 else ['train', 'test']

        for i, mode in enumerate(mode_labels):
            append_filenames_and_modes(data_list=data_tuple[i], m=mode, is_ad=is_anndata)

        df = pd.DataFrame({
            'filename': fns,
            'mode': modes
        })

        df.to_csv(os.path.join(filepath, filename), index=False)

    '''@staticmethod
    def concatenate_anndata(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            data_path: Union[str, None] = None,
            file_name: str = 'concatenated.h5ad',
    ) -> Union[sc.AnnData, str]:
        # ### Set flag whether to load data or not
        # (depending on list of AnnData objects or filenames of .h5ad files being passed)
        load_data = isinstance(data_list[0], str)

        if load_data:
            if data_path is None:
                data_path = os.getcwd()
            # ### Annotate filename in obs dimension
            for d in data_list:
                try:
                    dummy = sc.read_h5ad(os.path.join(data_path, d))
                    dummy.obs['file_name'] = [dummy.uns['file_name']] * dummy.n_obs
                    dummy.write_h5ad(Path(os.path.join(data_path, d)))
                    del dummy
                except KeyError:
                    print('WARNING: Key "file_name" does not exist in .uns of the AnnData object, '
                          'can not annotate the original file for each cell when concatenating')
            concat_on_disk(
                in_files=[os.path.join(data_path, file) for file in data_list],
                out_file=os.path.join(data_path, file_name), axis=0, join='inner', merge='same', index_unique='-',
                keys=list(range(len(data_list)))
            )
            concatenated_data = file_name
        else:
            # ### Annotate filename in obs dimension
            for d in data_list:
                try:
                    d.obs['file_name'] = [d.uns['file_name']] * d.n_obs
                except KeyError:
                    print('WARNING: Key "file_name" does not exist in .uns of the AnnData object, '
                          'can not annotate the original file for each cell when concatenating')

            concatenated_data = ad.concat(
                data_list, axis=0, join='inner', merge='same', index_unique='-', keys=list(range(len(data_list)))
            )

        return concatenated_data'''

    def create_data_loader(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            balance: bool = False,
            label_layer_key: Union[str, None] = None,
            batch_size: int = -1,
            shuffle: bool = True,
            return_data_loader: Literal['np_array', 'torch_tensor'] = 'np_array',
            on_disk: bool = False,
            filename: str = 'data.npy',
            **kwargs,
    ) -> Union[DataLoader, None]:

        assert data_set in {'all', 'train', 'test', 'val'}, "'data_set' must be 'all', 'train', 'test' or 'val'"
        if data_set == 'all':
            data_list = self.anndata_list
        elif data_set == 'train':
            data_list = self.train_data
        elif data_set == 'test':
            data_list = self.test_data
        elif data_set == 'val':
            try:
                data_list = self.val_data
            except NameError:
                logger.warning('# ### No validation set was created when splitting the data')
                return
        else:
            data_list = []

        out = FlowDataManager.create_data_loader_worker(
            data_list=data_list,
            save_path=self.save_path,
            channels=channels,
            layer_key=layer_key,
            label_key=label_key,
            balance=balance,
            label_layer_key=label_layer_key,
            batch_size=batch_size,
            shuffle=shuffle,
            return_data_loader=return_data_loader,
            on_disk=on_disk,
            filename=filename,
            **kwargs,
        )

        return out

    @staticmethod
    def create_data_loader_worker(
            data_list: Union[Sequence[str], Sequence[sc.AnnData]],
            save_path: str,  # Where data was saved and where .npy files are to be saved if 'on_disk' is True
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            balance: bool = False,
            label_layer_key: Union[str, None] = None,
            batch_size: int = -1,
            shuffle: bool = True,
            return_data_loader: Literal['np_array', 'torch_tensor'] = 'np_array',
            on_disk: bool = False,
            filename: str = 'data.npy',
            **kwargs
    ) -> DataLoader:

        # Set all channels as data
        if channels is None:
            if isinstance(data_list[0], str):
                channels = list(range(sc.read_h5ad(os.path.join(save_path, data_list[0])).n_vars))
            else:
                channels = list(range(data_list[0].n_vars))

        data_array = FlowDataManager._get_numpy_data_matrix(
            data_list=data_list, channels=channels, layer_key=layer_key, data_path=save_path
        )

        if label_key is not None:
            label_array = FlowDataManager._get_numpy_label_vector(
                data_list=data_list, label_key=label_key, layer_key=label_layer_key, data_path=save_path
            )
            if balance:
                os_strategy, us_strategy = FlowDataManager._create_sampling_strategies(y=label_array)
                rus = RandomUnderSampler(random_state=0, sampling_strategy=us_strategy)
                data_array, label_array = rus.fit_resample(data_array, label_array)
                ros = RandomOverSampler(random_state=0, sampling_strategy=os_strategy)
                data_array, label_array = ros.fit_resample(data_array, label_array)

            data_array = np.concatenate((data_array, np.expand_dims(label_array, axis=1)), axis=1)

        if on_disk:
            np.save(os.path.join(save_path, filename), data_array)

        ds = FlowDataset(
            data=os.path.join(save_path, filename) if on_disk else data_array, on_disk=on_disk,
            includes_labels=True if label_key is not None else False
        )

        if batch_size == -1:
            batch_size = len(ds)

        flow_dataloader = FlowDataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

        if return_data_loader == 'np_array':
            out = flow_dataloader.pytorch_np_dataloader
        else:
            out = flow_dataloader.pytorch_dataloader

        return out

    @staticmethod
    def _get_numpy_data_matrix(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            channels: Union[Sequence[int], Sequence[str]],
            layer_key: Union[str, None] = None,
            data_path: Union[str, None] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        load_data = isinstance(data_list[0], str)

        array = data_list[0]
        if load_data:
            array = sc.read_h5ad(os.path.join(data_path, array))
        # Get data matrix from anndata
        if layer_key is None:
            array = array[:, channels].X.copy()
        else:
            array = array[:, channels].layers[layer_key].copy()

        for d in data_list[1:]:
            if load_data:
                d = sc.read_h5ad(os.path.join(data_path, d))
            if layer_key is None:
                x = d[:, channels].X.copy()
            else:
                x = d[:, channels].layers[layer_key].copy()
            if load_data:
                del d

            array = np.concatenate((array, x), axis=0)

        return array

    @staticmethod
    def _get_numpy_label_vector(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            label_key: Union[int, str, None],
            layer_key: Union[str, None] = None,
            data_path: Union[str, None] = None
    ) -> np.ndarray:
        load_data = isinstance(data_list[0], str)

        label_array = data_list[0]
        if load_data:
            label_array = sc.read_h5ad(os.path.join(data_path, label_array))

        label_array = FlowDataManager._get_labels(adata=label_array, label_key=label_key, layer_key=layer_key)

        for d in data_list[1:]:
            if load_data:
                d = sc.read_h5ad(os.path.join(data_path, d))
            l = FlowDataManager._get_labels(adata=d, label_key=label_key, layer_key=layer_key)
            if load_data:
                del d
            label_array = np.concatenate((label_array, l), axis=0)

        return label_array

    @staticmethod
    def _get_labels(
            adata: sc.AnnData,
            label_key: Union[str, int],
            layer_key: Union[str, None] = None,
    ) -> np.ndarray:
        if isinstance(label_key, int):
            try:
                labels = adata.X[:, label_key].copy()
            except IndexError:
                logger.warning("'label_key' index out of bounds in .X matrix.")
                raise ValueError("'label_key' index is out of bounds")
        else:
            try:
                label_idx = adata.var_names.get_loc(label_key)
                if layer_key is None:
                    labels = adata.X[:, label_idx].copy()
                else:
                    labels = adata.layers[layer_key][:, label_idx].copy()
            except KeyError:
                logger.warning("# ### 'label_key' not found in .var_names, trying .obs")
                try:
                    labels = adata.obs[label_key].copy()
                except KeyError:
                    logger.warning("# ### 'label_key' not found in .obs either")
                    raise ValueError("'label_key' not found in .obs or .var_names")
        return labels

    @staticmethod
    def _create_sampling_strategies(
            y: np.ndarray,
    ) -> Tuple[dict, dict]:
        y_series = pd.Series(y)
        count_series = y_series.value_counts()
        count_array = count_series.values

        median = floor(np.median(count_array))
        mad = np.median(np.abs(count_array - median))
        upper = int(median + mad)
        lower = int(median - mad)
        logger.info(f'# ### Class counts have: median: {median}, mad: {mad}, '
                    f'up-sampling to median - mad = {lower}, down-sampling to median + mad = {upper}')

        ds_dict = {}
        us_dict = {}
        for idx, val in count_series.items():
            if val <= lower:
                us_dict[idx] = lower
            if val >= upper:
                ds_dict[idx] = upper

        return us_dict, ds_dict

    @staticmethod
    def check_class_balance(
            dl: DataLoader,
            save: bool = False,
            save_kwargs: Union[dict, None] = None,
            plot: bool = False,
            plot_kwargs: Union[dict, None] = None,
    ):
        label_array = np.array([])
        for _, y in dl:
            label_array = np.concatenate((label_array, y), axis=0)
        label_series = pd.Series(label_array)
        class_counts = label_series.value_counts()
        class_fracs = label_series.value_counts(normalize=True)
        logger.info(f'# ### Absolute counts for the labels:\n{class_counts}')
        logger.info(f'# ### Relative frequencies for the labels:\n{class_fracs}')

        if save:
            if save_kwargs is None:
                save_kwargs = dict()
            filename = save_kwargs.get('filename', None)
            filepath = save_kwargs.get('filepath', None)
            if filename is None:
                filename = 'class_balances.csv'
            if filepath is None:
                filepath = os.getcwd()

            result_df = pd.DataFrame({
                'count': class_counts.astype(str),
                'fraction': class_fracs
            }).T

            result_df.to_csv(os.path.join(filepath, filename))

        if plot:
            if plot_kwargs is None:
                plot_kwargs = dict()
            ax = plot_kwargs.get('ax', None)
            filename = plot_kwargs.get('filename', None)
            filepath = plot_kwargs.get('filepath', None)
            show = plot_kwargs.get('show', None)
            if filepath is None:
                filepath = os.getcwd()
            if ax is None:
                fig, ax = plt.subplots()

            num_classes = len(class_fracs)
            color_map = plt.cm.get_cmap("tab10" if num_classes <= 10 else "tab20", num_classes)
            colors = [color_map(i) for i in range(num_classes)]

            class_fracs.plot(kind='bar', color=colors, ax=ax)
            ax.set_xlabel('Class')
            ax.set_ylabel('Frequency')
            ax.set_title('Class Balance')

            y_max = class_fracs.max() * 1.1
            ax.set_ylim(0, y_max)
            for idx, value in enumerate(class_fracs):
                text = f'total: {class_counts.iloc[idx]}, frac: {round(value, 4)}'
                text_offset = 0.05 * y_max
                y_text = value + text_offset
                if y_text + 6 * text_offset > y_max:
                    ax.text(
                        idx, y_text, text, ha='center', va='top', rotation=90)
                else:
                    ax.text(
                        idx, y_text, text, ha='center', va='bottom', rotation=90)

            if filename:
                plt.savefig(os.path.join(filepath, filename))

            if show:
                plt.show()


class FlowDataset(Dataset):
    def __init__(
            self,
            data: Union[str, np.ndarray],
            on_disk: bool = True,
            includes_labels: bool = False,
    ):
        assert isinstance(data, str) or isinstance(data, np.ndarray), \
            "'data' must be path to data file (.npy) or Numpy array"

        self.on_disk = on_disk

        if isinstance(data, str):
            self.file_path = data
            # ### Load data in previously defined mode

            if self.on_disk:
                self.data = np.load(self.file_path)
            else:
                self.data = np.load(self.file_path, mmap_mode='r')

        else:
            self.data = data

        # Slicing on memory-mapped arrays only works for contiguous slices, expect labels in last column
        self.includes_labels = includes_labels
        if self.includes_labels:
            assert self.data.ndim == 2 and self.data.shape[1] > 1, \
                "Data must have at least two dimensions with labels in the last column"
            self.label_idx = self.data.shape[1] - 1

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, int], np.ndarray]:

        if self.includes_labels:
            event = self.data[idx, :self.label_idx]
            label = int(self.data[idx, self.label_idx].item())

            return event, label
        else:
            event = self.data[idx, :]
            return event


class FlowDataLoader:
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
    ):
        self.dataset = dataset
        self.pytorch_dataloader = DataLoader(dataset, **kwargs)
        # Might be adding other custom Dataloaders later on
        self.pytorch_np_dataloader = DataLoader(dataset, collate_fn=FlowDataLoader.np_collate, **kwargs)

    @staticmethod
    def np_collate(batch: Union[Sequence[Tuple[np.ndarray, int]], Sequence[np.ndarray]]):
        # Check if the batch contains labels by looking at the first item
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # Separate data and labels
            batch_data = np.stack([item[0] for item in batch], axis=0)
            batch_labels = np.stack([item[1] for item in batch], axis=0)
            return batch_data, batch_labels
        else:
            # Only data without labels
            batch_data = np.stack(batch, axis=0)
            return batch_data


