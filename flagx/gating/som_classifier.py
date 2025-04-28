
import os
import warnings
import pickle
import copy
import time

import flowio
import numpy as np
import pandas as pd

from .._legacy_typing import Literal, Tuple, Union, List, Dict, Callable, Iterable, SelfSomClassifier, Any
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
            verbosity: int = 1,
    ):
        super().__init__()
        # ### Initialize parameters
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

        # ### Initialize training specific parameters:
        self.n_epochs = n_epochs
        self.radius_0 = radius_0
        self.radius_n = radius_n
        self.radius_cooling = radius_cooling
        self.learning_rate_0 = learning_rate_0
        self.learning_rate_n = learning_rate_n
        self.learning_rate_decay = learning_rate_decay
        # If radius_0 < 0 set radius_0 relative to the SOM dimensions

        # ### Initialize all variables associated with a trained SOM classifier
        self.is_fitted_ = False
        self.som_ = None
        self.n_features_in_ = None
        self.labeled_data_ = False
        # These are only relevant if trained on labeled data:
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
        elif self.radius_0 == 0:  # Default case from Somoclu
            self.radius_0 = min(self.som_dimensions[0], self.som_dimensions[1]) / 2

    # ### fit(), predict(), predict_proba(), annotate_and_export() #####################################################
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> SelfSomClassifier:

        # Check input data format
        X, y = check_X_y(X, y)

        # Check if .fit() was called already
        if self.is_fitted_:

            warnings.warn(
                "The `.fit()` method was called on an already trained SOM classifier. "
                "Training will continue with the new data and the existing codebook. "
                "To restart training from scratch, call `.reset()` before calling `.fit()`.",
                UserWarning
            )

            # Raise a ValueError if the input data does not match the dimension of the previous input data
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features as per previous training, but got {X.shape[1]}."
                )

            # Set parameters to train with the new data and the codebook from the previously trained SOM
            self.initialization = None
            self.initial_codebook = np.copy(self.som_.codebook)

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

        # If SOM classifier was already trained on labeled data (and now is trained again on labeled data),
        # raise warning regarding the label computation
        if self.is_fitted_ and self.labeled_data_:
            warnings.warn(
                "SOM unit annotations are based on current labeled data. "
                "To include previous training labels, call `.annotate_som()` with all labeled data.", UserWarning
            )

        # Annotate the trained SOM's units
        self.annotate_som(X=X, y=y)

        # If no labeled data passed to .fit(), raise UserWarning
        if not self.labeled_data_:
            warnings.warn(
                'No labeled data provided—label prediction is not possible. '
                'To manually annotate labels in Kaluza, call `.export_fcs()` '
                'to extract the training data annotated with SOM nodes as an .fcs file.',
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
                'To manually annotate labels in Kaluza, call `.export_fcs()` '
                'to extract the training data annotated with SOM nodes as an .fcs file.',
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
                'To manually annotate labels in Kaluza, call `.export_fcs()` '
                'to extract the training data annotated with SOM nodes as an .fcs file.',
                UserWarning
            )

        return y_proba

    def annotate_som(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> SelfSomClassifier:

        # Ensure SOM has been trained before annotation
        if not hasattr(self, 'som_') or self.som_ is None:
            raise RuntimeError('Cannot annotate SOM before training. Call `.fit()` before `.annotate_som()`.')

        # Check input data format
        X, y = check_X_y(X, y)

        # Get labeled data from the input data
        x_labeled, y_labeled = SomClassifier._get_labeled_data(
            X=X,
            y=y,
            nan_val=self.unlabeled_label,
            verbosity=self.verbosity
        )

        if x_labeled.shape[0] != 0:

            # Reset any previous annotations
            self._reset_to_unlabeled()

            # Rename classes to integers starting from 0 and extract label information
            (
                y_labeled, self.classes_, self.class_counts_, self.class_priors_, self.new_to_og_classes_dict_,
                self.og_classes_
            ) = SomClassifier._process_class_labels(y=y_labeled)

            # ### Determine the majority class for all SOM units
            # Get BMUs for train data, shape n_events x 2 = som_coordinates
            # Note: surface_state = activation_map = codebook * data_matrix
            bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=x_labeled))

            # Append label vector to BMUs
            bmus = np.column_stack((bmus, y_labeled))

            # If .fit() has not been called before or the SOM classifier was only trained on unlabeled data,
            # initialize the class counts array, shape: (somdim0, somdim1, n_classes)
            self.class_counts_per_unit_ = np.zeros(self.som_dimensions + (self.classes_.shape[0],))

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
        else:
            # Reset all class attributes that are associated with a SOM classifier trained on labeled data
            self._reset_to_unlabeled()

        return self

    def export_fcs(
            self,
            X: Union[List[np.ndarray], np.ndarray],
            channel_names_X: Union[List[str], None] = None,
            X_raw: Union[List[np.ndarray], np.ndarray, List[pd.DataFrame], pd.DataFrame, None] = None,
            channel_names_X_raw: Union[List[str], None] = None,
            keep_X: bool = False,
            val_range: Union[Tuple[float, float], None] = (0.0, 2**20),
            save_unscaled_data: bool = False,
            scale_X_raw_channels: Union[List[str], None] = None,
            sample_ids: Union[List[int], None] = None,
            compute_umap: bool = False,
            umap_kwargs: Union[Dict, None] = None,
            save_mode: Literal['fcs', 'csv', 'no_save'] = 'no_save',
            fcs_metadata_dict: Union[Dict, None] = None,
            save_path: Union[str, None] = None,
            filename: Union[str, None] = None,
    ) -> pd.DataFrame:

        # Behaviour:
        # - X: pass or annotate channel names
        # - X, X_raw: keep only X_raw, if channel names passed use, if df use columns, else annotate
        # - X, X_raw, keep_X: keep X, X_raw, if channel names passed use, if df use columns, else annotate (for both)

        # - All channels that are not in channel_names_X_raw are scaled

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Turn array input into list
        if not isinstance(X, List):
            X = [X, ]

        if not isinstance(X_raw, List):
            if X_raw is not None:
                X_raw = [X_raw, ]

        if X_raw is not None and len(X_raw) != len(X):
            raise ValueError("'X' and 'X_raw' must have the same length.")

        # Define sample ids
        if sample_ids is not None:
            if len(sample_ids) != len(X):
                raise ValueError("'sample_ids' and 'X' must have the same length.")
        else:
            sample_ids = [i for i in range(len(X))]

        # Define channel names for X if X_raw is None or X is to be kept
        if X_raw is None or keep_X:
            if channel_names_X is not None:
                if len(channel_names_X) != X[0].shape[1]:
                    raise ValueError(
                        "Length of 'channel_names_X' must match number of dimensions of 'X' in axis 1 "
                    )
            else:
                channel_names_X = [f'X_channel_{i}' for i in range(X[0].shape[1])]
        else:
            channel_names_X = None

        # Define channel names for X_raw if it is not None
        if X_raw is not None:
            if channel_names_X_raw is not None:
                if len(channel_names_X_raw) != X_raw[0].shape[1]:
                    raise ValueError(
                        "Length of 'channel_names_X_raw' must match number of dimensions of 'X_raw' in axis 1 "
                    )
            else:
                if isinstance(X_raw[0], pd.DataFrame):
                    channel_names_X_raw = X_raw[0].columns.tolist()
                else:
                    channel_names_X_raw = [f'X_raw_channel_{i}' for i in range(X[0].shape[1])]
        else:
            channel_names_X_raw = None

        # Annotate the individual data matrices
        fcs_dfs = [pd.DataFrame()] * len(X)
        for i, (x, x_raw, s_id) in enumerate(zip(X, X_raw if X_raw is not None else [None, ] * len(X), sample_ids)):
            fcs_dfs[i] = self._x_to_fcs_style_df(
                X=x,
                channel_names_X=channel_names_X,
                X_raw=x_raw,
                channel_names_X_raw=channel_names_X_raw,
                keep_X=keep_X,
                sample_id=s_id
            )

        # Concatenate the annotated matrices
        fcs_df = pd.concat(fcs_dfs, axis=0, ignore_index=True)

        # Add annotations for better visualization of the SOM in Kaluza
        fcs_df = self._add_visualization_annotations(fcs_df=fcs_df)

        # Compute UMAP embedding of the data
        if compute_umap:
            if self.verbosity >= 1:
                print('Computing UMAP map...')
            st = time.time()
            umap_reducer = UMAP(**umap_kwargs if umap_kwargs is not None else {})
            umap_embedding = umap_reducer.fit_transform(np.concatenate(X))  # Shape: n samples x 2
            et = time.time()
            fcs_df['umap1'] = umap_embedding[:, 0]
            fcs_df['umap2'] = umap_embedding[:, 1]

            if self.verbosity >= 1:
                print(f'UMAP map computation took {et-st:.2f} seconds')

        # Scale all entries of the data matrix to a given interval
        if val_range is not None:
            if X_raw is not None: # Raw data should not be scaled, except for certain channels
                if scale_X_raw_channels is None:
                    scale_X_raw_channels = []

                scale_bool = ~fcs_df.columns.isin(
                    [c for c in channel_names_X_raw if c not in scale_X_raw_channels]
                )

            else:  # No X_raw, scale everything
                scale_bool = np.ones(fcs_df.shape[1]).astype(bool)

            scaled_data = SomClassifier._scale_column_wise(x=fcs_df.loc[:, scale_bool].to_numpy(), val_range=val_range)

            if save_unscaled_data:
                # Create df with scaled data, mark channels as scaled
                scaled_df = pd.DataFrame(
                    data=scaled_data,
                    index=fcs_df.index,
                    columns=[f'{col}_scaled' for i, col in enumerate(fcs_df.columns.tolist()) if scale_bool[i]],
                )
                # Concatenate with original df
                fcs_df = pd.concat([fcs_df, scaled_df], axis=1)
            else:
                # Create df with scaled data, keep original channel names
                scaled_df = pd.DataFrame(
                    data=scaled_data,
                    index=fcs_df.index,
                    columns=fcs_df.columns[scale_bool],
                )
                # Concatenate with part of original df that was not scaled
                fcs_df = pd.concat([fcs_df.loc[:, ~scale_bool], scaled_df], axis=1)

        # Save to .fcs or .csv format
        if save_mode != 'no_save':
            if save_path is None:
                save_path = os.getcwd()

            if save_mode == 'fcs':
                if filename is None:
                    filename = 'som.fcs'

                # Define meta dict
                if fcs_metadata_dict is None:
                    fcs_metadata_dict = {}
                fcs_metadata_dict.update({f"P{i}R": str(val_range[1]) for i in range(1, fcs_df.shape[1] + 1)})

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
    ) -> SelfSomClassifier:

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
            checkpoint_dir: Union[str, None] = None,
            save_tracked: bool = False,
    ) -> SelfSomClassifier:

        if checkpoint_interval < 2 or checkpoint_interval >= self.n_epochs:
            raise ValueError("'checkpoint_interval' must be greater or equal to 2 and smaller than 'n_epochs'")

        if tracking_interval is not None:
            if tracking_interval < 2 or tracking_interval >= self.n_epochs:
                raise ValueError("'checkpoint_interval' must be greater or equal to 2 and smaller than 'n_epochs'")

        if self.radius_cooling != 'linear' or self.learning_rate_decay != 'linear':
            warnings.warn(
                "For checkpointed training it is recommended to use 'linear' decay functions for the radius "
                "and learning rate, not 'exponential'", UserWarning)

        if (track_losses or track_impurities or (X_y_val is not None)) and tracking_interval is None:
            tracking_interval = 10

        if checkpoint_dir is None:
            checkpoint_dir = os.getcwd()

        n_checkpoint_intervals = self.n_epochs // checkpoint_interval
        i_checkpoints = [(i + 1) * checkpoint_interval for i in range(n_checkpoint_intervals)]

        # Define points where training is stopped for checkpointing or tracking
        i_stop = np.unique(np.concatenate((np.array(i_checkpoints), np.array([self.n_epochs, ])), axis=0))

        if tracking_interval is not None:
            n_tracking_intervals = self.n_epochs // tracking_interval
            i_tracking = [(i + 1) * tracking_interval for i in range(n_tracking_intervals)]
            i_stop = np.sort(np.unique(np.concatenate((np.array(i_tracking), i_stop), axis=0)))
        else:
            i_tracking = []

        training_intervals = np.diff(np.concatenate((np.array([0,]), i_stop), axis=0))

        # Raise an error if any training interval is smaller than 2
        if (training_intervals <= 1).any():
            raise ValueError("One or more training interval is smaller than 2.")

        # Check input format
        X, y = check_X_y(X, y)

        # Get labeled input data
        X_labeled, y_labeled = SomClassifier._get_labeled_data(
            X=X,
            y=y,
            nan_val=self.unlabeled_label,
            verbosity=self.verbosity
        )

        # Check if .fit() was called already
        if self.is_fitted_:

            warnings.warn(
                "The `.fit_with_checkpoints()` method was called on an already trained SOM classifier. "
                "Training will continue with the new data and the existing codebook. "
                "To restart training from scratch, call `.reset()` before calling `.fit_with_checkpoints()`.",
                UserWarning
            )

            # Raise a ValueError if the input data does not match the dimension of the previous input data
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features as per previous training, but got {X.shape[1]}."
                )

            # Set parameters to train with the new data and the codebook from the previously trained SOM
            self.initialization = None
            self.initial_codebook = np.copy(self.som_.codebook)

        self.n_features_in_ = X.shape[1]

        # Initialize the SOM
        self._initialize_som()

        # Set the initial radius parameter of the SOM (negative values are interpreted as fractions of the grid size)
        self._set_radius_0()

        # Get decay functions for radius and learning rate
        decay_fct_radius = SomClassifier._get_decay_function(
            val0=self.radius_0, val1=self.radius_n, n_epochs=self.n_epochs, strategy=self.radius_cooling)
        decay_fct_lr = SomClassifier._get_decay_function(
            val0=self.learning_rate_0, val1=self.learning_rate_n, n_epochs=self.n_epochs,
            strategy=self.learning_rate_decay)

        # Initialize lists for tracking
        if track_losses:
            quantization_loss = []  # 1 dim
            topographical_loss = []  # 1 dim
        if track_impurities:
            entropies = []  # som dim
            ginies = []  # som dim
        if X_y_val is not None:
            X_val, y_val = check_X_y(X_y_val[0], X_y_val[1])
            if track_losses:
                quantization_loss_val = []  # 1 dim
                topographical_loss_val = []  # 1 dim

            f1_macro_train = []  # 1 dim
            f1_micro_train = []  # 1 dim
            f1_weighted_train = []  # 1 dim
            f1_macro_val = []  # 1 dim
            f1_micro_val = []  # 1 dim
            f1_weighted_val = []  # 1 dim
            f1_class_wise_train = []  # n classes dim
            f1_class_wise_val = []  # n classes dim

        # Define var that tracks how many epochs were already trained
        n_epochs_done = 0

        # Loop over intervals, stop for tracking or saving
        for j, (interval, i) in enumerate(zip(training_intervals, i_stop)):
            # Train the SOM on all data (unsupervised)
            self.som_.train(
                data=X,
                epochs=int(interval),
                radius0=decay_fct_radius(n_epochs_done),
                radiusN=decay_fct_radius(n_epochs_done + interval),
                radiuscooling='linear',  # Approximate the chosen decay function by piecewise linear function
                scale0=decay_fct_lr(n_epochs_done),
                scaleN=decay_fct_lr(n_epochs_done + interval),
                scalecooling='linear',  # Approximate the chosen decay function by piecewise linear function
            )

            # Only need to annotate som if it is to be saved, impurities are tracked or validation metrics are tracked
            if i in i_checkpoints or (i in i_tracking and (track_impurities or X_y_val is not None)) :

                # If SOM classifier was already trained on labeled data (and now is trained again on labeled data),
                # raise warning regarding the label computation
                if self.is_fitted_ and self.labeled_data_:
                    warnings.warn(
                        "SOM unit annotations are based on current labeled data. "
                        "To include previous training labels, call `.annotate_som()` with all labeled data.",
                        UserWarning
                    )

                # Annotate the trained SOM's units
                self.annotate_som(X=X, y=y)

                # If no labeled data passed to .fit(), raise UserWarning
                if not self.labeled_data_:
                    warnings.warn(
                        'No labeled data provided—label prediction is not possible. '
                        'To manually annotate labels in Kaluza, call `.export_fcs()` '
                        'to extract the training data annotated with SOM nodes as an .fcs file.',
                        UserWarning
                    )

                if track_impurities:
                    entropies.append(self.unit_impurity(impurity_measure='entropy'))
                    ginies.append(self.unit_impurity(impurity_measure='gini'))

                if X_y_val is not None:
                    if track_losses:
                        quantization_loss_val.append(self.quantization_error(X=X_val))
                        topographical_loss_val.append(self.topographic_error(X=X_val))

                    y_pred_train = self.predict(X=X_labeled)
                    y_pred_val = self.predict(X=X_val)

                    f1_macro_train.append(f1_score(y_labeled, y_pred_train, average='macro'))
                    f1_micro_train.append(f1_score(y_labeled, y_pred_train, average='micro'))
                    f1_weighted_train.append(f1_score(y_labeled, y_pred_train, average='weighted'))

                    f1_macro_val.append(f1_score(y_val, y_pred_val, average='macro'))
                    f1_micro_val.append(f1_score(y_val, y_pred_val, average='micro'))
                    f1_weighted_val.append(f1_score(y_val, y_pred_val, average='weighted'))

                    f1_class_wise_train.append(f1_score(y_labeled, y_pred_train, average=None))
                    f1_class_wise_val.append(f1_score(y_val, y_pred_val, average=None))

                if (i_checkpoints == i).any():
                    self.save(filename=f'checkpoint_nepochs{n_epochs_done + interval}.pkl', filepath=checkpoint_dir)

            if i in i_tracking and track_losses:
                quantization_loss.append(self.quantization_error(X=X))
                topographical_loss.append(self.topographic_error(X=X))

            n_epochs_done += interval

        if save_tracked:
            res_dict = {'n_epochs': i_tracking}
            if track_losses:
                res_dict['quantization_loss'] = quantization_loss
                res_dict['topographical_loss'] = topographical_loss
            if track_impurities:
                res_dict['entropies'] = entropies
                res_dict['ginies'] = ginies
            if X_y_val is not None:
                if track_losses:
                    res_dict['quantization_loss_val'] = quantization_loss_val
                    res_dict['topographical_loss_val'] = topographical_loss_val

                res_dict['f1_macro_train'] = f1_macro_train
                res_dict['f1_micro_train'] = f1_micro_train
                res_dict['f1_weighted_train'] = f1_weighted_train
                res_dict['f1_macro_val'] = f1_macro_val
                res_dict['f1_micro_val'] = f1_micro_val
                res_dict['f1_weighted_val'] = f1_weighted_val
                res_dict['f1_class_wise_train'] = f1_class_wise_train
                res_dict['f1_class_wise_val'] = f1_class_wise_val

            with open(os.path.join(checkpoint_dir, 'res_dict.pkl'), 'wb') as f:
                pickle.dump(res_dict, f)

        # Set flag indicating that SOM classifier is fitted
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
                'To manually annotate labels in Kaluza, call `.export_fcs()` '
                'to extract the training data annotated with SOM nodes as an .fcs file.',
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
                'To manually annotate labels in Kaluza, call `.export_fcs()` '
                'to extract the training data annotated with SOM nodes as an .fcs file.',
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
    ) -> SelfSomClassifier:
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
    def _reset_to_unlabeled(self):
        self.labeled_data_ = False
        self.classes_ = None
        self.class_counts_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None

    @staticmethod
    def _get_labeled_data(
            X: np.ndarray,
            y: np.ndarray,
            nan_val: Any = np.nan,
            verbosity: int = 0,
    ):
        nan_mask = (y == nan_val)

        x_labeled = X[~nan_mask, :]
        y_labeled = y[~nan_mask]

        if verbosity >= 2:
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
        elif strategy == 'exponential':
            if val0 == 0:
                raise ValueError("For exponential decay, val0 cannot be zero.")

            rate = log(val1 / val0) / n_epochs

            def _decay_function(x: int):
                return val0 * exp(rate * x)
        else:
            raise ValueError(f"Invalid decay strategy: {strategy}. Choose 'linear' or 'exponential'.")

        # Return decay fct with params
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

    def transform(
            self,
            X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        check_is_fitted(self, 'is_fitted_')

        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        som_unit_ids = self._get_row_major_positions(bmus=bmus, start_from=1)

        bmus_df = pd.DataFrame(
            data=np.hstack((bmus, som_unit_ids.reshape((-1, 1)))),
            columns=['bmu1', 'bmu2', 'unit_id']
        )

        # Get dataframe with columns: 'bmu1', 'bmu2', 'som_unit_id', 'count'
        unit_counts = bmus_df.groupby(['bmu1', 'bmu2', 'unit_id']).size().reset_index(name='count')

        # Compute radius for each unit that is proportional to count
        counts = unit_counts['count'].to_numpy()
        # Version 2:
        radii = 0.5 * np.sqrt(counts) / np.sqrt(counts.max())
        # Version 1:
        # radii = np.sqrt(counts / np.pi)
        # radii = 0.5 * radii / radii.max()
        # Version 0:
        # radii = (counts - counts.min()) / (counts.max() - counts.min())
        # radii = np.sqrt(radii)  # Area should be proportional to count -> use square root
        # radii = radii * 0.5  # Radius should be <= 0.5
        unit_counts['radius'] = radii

        bmus_df['bmu1_scattered'] = np.zeros(bmus.shape[0])
        bmus_df['bmu2_scattered'] = np.zeros(bmus.shape[0])
        bmus_df['radius'] = np.zeros(bmus.shape[0])

        for bmu1, bmu2, unit_id, count, radius in zip(
                unit_counts['bmu1'], unit_counts['bmu2'], unit_counts['unit_id'], unit_counts['count'],
                unit_counts['radius']
        ):
            x, y = SomClassifier._random_points_on_sphere(x_center=bmu1, y_center=bmu2, radius=radius, n=count)
            mask = bmus_df['unit_id'] == unit_id
            bmus_df.loc[mask, 'bmu1_scattered'] = x
            bmus_df.loc[mask, 'bmu2_scattered'] = y
            bmus_df.loc[mask, 'radius'] = radius

        bmus_scattered = bmus_df[['bmu1_scattered', 'bmu2_scattered']].to_numpy()

        radii_out = bmus_df['radius'].to_numpy()

        return bmus, bmus_scattered, som_unit_ids, radii_out

    def _x_to_fcs_style_df(
            self,
            X: np.ndarray,
            channel_names_X: Union[List[str], None],
            X_raw: Union[np.ndarray, pd.DataFrame, None],
            channel_names_X_raw: Union[List[str], None],
            keep_X: bool,
            sample_id: int,
    ) -> pd.DataFrame:

        # Check input data format
        X = check_array(X)

        # No X_raw, use just X, channel_names are not None by design
        if X_raw is None:
            fcs_df = pd.DataFrame(data=X, columns=channel_names_X)

        # X_raw not None  -> channel_names_X_raw cannot be None
        else:
            # X_raw is df
            if isinstance(X_raw, pd.DataFrame):
                fcs_df = X_raw
                fcs_df.columns = channel_names_X_raw

            # X_raw is numpy array
            else:
                fcs_df = pd.DataFrame(data=X_raw, columns=channel_names_X_raw)

            # keep_X -> channel_names_X cannot be None
            if keep_X:
                # Check if channel names overlap
                overlap = set(channel_names_X) & set(fcs_df.columns)
                if bool(overlap):
                    channel_names_X = [f'train_{cn}' for cn in channel_names_X]
                    warnings.warn(
                        f'The channels names {overlap} are shared between X and X_raw. '
                        f'Adding prefix "train_" to "channel_names_X"', UserWarning)

                x_df = pd.DataFrame(data=X, columns=channel_names_X)
                # Concatenate the dataframes
                fcs_df = pd.concat([fcs_df.reset_index(drop=True), x_df.reset_index(drop=True)], axis=1)

        # Annotate the sample id
        fcs_df['sample_id'] = np.full(fcs_df.shape[0], sample_id, dtype=int)

        # Annotate the bmu coordinates
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))
        fcs_df['bmu1'] = bmus[:, 0]
        fcs_df['bmu2'] = bmus[:, 1]

        # Annotate the SOM unit label of the bmus
        fcs_df['som_unit_label'] = self._get_row_major_positions(bmus=bmus, start_from=1)

        # Annotate labels
        if self.labeled_data_:
            fcs_df['label_predicted'] = self.predict(X=X)

        return fcs_df

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
