
# Needed for when an older version of the SOM classifier must be loaded



from typing import Literal, Tuple, Union, List, Self, Dict, Callable, Iterable, Any

import numpy as np
import os
import logging
import warnings
import pickle
import copy
import time

import pandas as pd
from math import log, exp
from somoclu import Somoclu
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from scipy.spatial.distance import cdist


# ### Set up logger
logger = logging.getLogger(__name__)


class SomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
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
            verbosity: int = 0,
    ):
        super().__init__()
        # Initialize parameters
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
        self.som_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None

        # Epoch wise training
        self.epoch_wise_som_training_metrics_ = None  # Only relevant if SOM is trained epoch wise

        # Hyperparameter tuning
        self.grid_search_ = None

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

    # ### Fitting of SOM classifier and predicting with fitted SOM classifier
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_epochs: Union[int, None] = None,
            radius_0: Union[float, None] = None,
            radius_n: Union[float, None] = None,
            radius_cooling: Union[Literal['linear', 'exponential'], None] = None,
            learning_rate_0: Union[float, None] = None,
            learning_rate_n: Union[float, None] = None,
            learning_rate_decay: Union[Literal['linear', 'exponential'], None] = None,
    ) -> Self:

        if n_epochs is not None:
            self.n_epochs = n_epochs
            warnings.warn(
                "Setting `n_epochs` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if radius_0 is not None:
            self.radius_0 = radius_0
            warnings.warn(
                "Setting `radius_0` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if radius_n is not None:
            self.radius_n = radius_n
            warnings.warn(
                "Setting `radius_n` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if radius_cooling is not None:
            self.radius_cooling = radius_cooling
            warnings.warn(
                "Setting `radius_cooling` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if learning_rate_0 is not None:
            self.learning_rate_0 = learning_rate_0
            warnings.warn(
                "Setting `learning_rate_0` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if learning_rate_n is not None:
            self.learning_rate_n = learning_rate_n
            warnings.warn(
                "Setting `learning_rate_n` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if learning_rate_decay is not None:
            self.learning_rate_decay = learning_rate_decay
            warnings.warn(
                "Setting `learning_rate_decay` in `fit` is not recommended. It should be set in `__init__`.",
                UserWarning)

        X, y = check_X_y(X, y)

        self._initialize_som()

        self._set_radius_0()

        self.n_features_in_ = X.shape[1]
        # Rename classes to integers starting from 0
        y, self.classes_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_ = \
            SomClassifier._process_class_labels(y=y)
        # self.classes_, counts = np.unique(y, return_counts=True)
        # self.class_priors_ = counts / counts.sum()

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

        # ### Determine the majority class for vertices
        # Get BMUs for train data, shape n_events x 2 ~ som_coordinates
        # Note: surface_state = activation_map = codebook * data_matrix

        # bmus = self.som_.get_bmus(activation_map=self.som_.get_surface_state(data=X))
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        # Append label vector to BMUs
        # bmus = np.concatenate((bmus, np.expand_dims(y, axis=1)), axis=1)
        # Get the unique rows and their count ~= How often is unit (i,j) BMU for a sample with label k
        # unique_rows, counts = np.unique(bmus, axis=0, return_counts=True)
        # Store info in one array
        # self.class_counts_per_unit_ = np.zeros(self.som_dimensions + (self.classes_.shape[0], ))
        # self.class_counts_per_unit_[tuple(unique_rows[:, 0]), tuple(unique_rows[:, 1]), tuple(unique_rows[:, 2])] = \
        #     counts
        # for i in range(counts.shape[0]):
        #     self.class_counts_per_unit_[tuple(unique_rows[i, :])] = counts[i]

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

        # Todo: (1.) Check if this leads to no unforeseen bugs
        # ### Account for the case that the support of some units is 0
        support = self.class_counts_per_unit_.sum(axis=2)
        zero_support_bool = support == 0
        if np.any(zero_support_bool):
            self.som_unit_labels_[zero_support_bool] = -1
            self.new_to_og_classes_dict_[-1] = -1
            warnings.warn(
                f"{zero_support_bool.sum()} SOM nodes are not BMU for any training data: "
                f"{[(int(i), int(j)) for i,j in np.argwhere(zero_support_bool)]}",
                UserWarning
            )

        self.is_fitted_ = True

        return self

    @staticmethod
    def _process_class_labels(
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Any, float], np.ndarray]:
        og_classes, counts = np.unique(y, return_counts=True)
        # Turn labels into integers, if possible
        try:
            og_classes = og_classes.astype(float)
            if np.any(og_classes % 1 != 0):
                og_classes = og_classes.astype(int)
        except ValueError:
            pass

        class_priors = counts / counts.sum()
        new_classes = np.array(list(range(og_classes.shape[0])))
        og_to_new_classes_dict = {key: value for key, value in zip(og_classes, new_classes)}
        y_new = np.vectorize(og_to_new_classes_dict.get)(y)

        new_to_og_classes_dict = {key: value for key, value in zip(new_classes, og_classes)}

        return y_new, new_classes, class_priors, new_to_og_classes_dict, og_classes

    def _custom_get_surface_state(
            self,
            data: np.ndarray,
    ):
        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> somdim0 * somdim1, n_features)
        # Todo: remove legacy case when not needed anymore
        try:
            codebook_reshaped = self.som_.codebook.reshape(-1, self.som_.codebook.shape[2])
        except AttributeError:
            # Legacy
            codebook_reshaped = self.som.codebook.reshape(-1, self.som.codebook.shape[2])

        # ### Compute Euclidean distances in chunks for memory efficiency
        # Split data into 200 chunks along axis 0
        num_splits = 200
        chunks = np.array_split(data, num_splits, axis=0)
        # Compute for each chunk the euclidean distance to the codebook, stack results
        activation_map = np.vstack(
            [cdist(chunk, codebook_reshaped, metric='euclidean') for chunk in chunks]
        )

        return activation_map

    def _custom_get_bmus(
            self,
            activation_map: np.ndarray,
    ):
        # Shape activation map: (n_events, somdim0 * somdim1)

        # ### Find position in the SOM grid of the minimum value for each row
        bmu_indices = np.argmin(activation_map, axis=1)
        j_s, i_s = np.divmod(bmu_indices, self.som_dimensions[1])
        return np.column_stack((i_s, j_s))

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Get BMU of event
        # bmus = self.som_.get_bmus(activation_map=self.som_.get_surface_state(data=X))
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        # Get prediction as label of BMU of event
        y_pred = self.som_unit_labels_[tuple(bmus[:, 0]), tuple(bmus[:, 1])]

        # Get original labels
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        # Todo: (2.) Check if this leads to no unforeseen bugs
        # Utter warning if label less-unit is BMU at prediction time
        if np.any(y_pred == -1):
            warnings.warn(
                f"For events {np.argwhere(y_pred == -1).flatten().tolist()} the BMU has no label "
                f"(support of the unit during training was 0). "
                f"Its predicted class is -1 ~= unknown/undeterminable from the training data",
                UserWarning)

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        # 1.) Confidence in prediction based on class dist of events with unit as a bmu
        # or Todo: 2.) Softmax on all event-unit similarities => dist across som nodes, pred via sampling from dist
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Get BMUs of events
        # bmus = self.som_.get_bmus(activation_map=self.som_.get_surface_state(data=X))
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        # Calculate the fraction of each class at each SOM unit
        unit_wise_class_probabilities = np.divide(
            self.class_counts_per_unit_,
            self.class_counts_per_unit_.sum(axis=2, keepdims=True),
            where=self.class_counts_per_unit_.sum(axis=2, keepdims=True) != 0
        )

        y_proba = unit_wise_class_probabilities[tuple(bmus[:, 0]), tuple(bmus[:, 1]), :]

        return y_proba

    def fit_epoch_wise(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_epochs: Union[int, None] = None,
            radius_0: Union[float, None] = None,
            radius_n: Union[float, None] = None,
            radius_cooling: Union[Literal['linear', 'exponential'], None] = None,
            learning_rate_0: Union[float, None] = None,
            learning_rate_n: Union[float, None] = None,
            learning_rate_decay: Union[Literal['linear', 'exponential'], None] = None,
            tracking_interval: int = 2,
            track_losses: bool = False,
            track_impurities: bool = False,
            save_intermediate_interval: Union[int, None] = None,
            val_data: Union[Tuple[np.ndarray, np.ndarray], None] = None,
            save_res_dict: bool = False,
            filename: Union[str, None] = None,
            filepath: Union[str, None] = None,
    ) -> Self:
        # ### Train two epochs at the time and set decaying parameters manually, calculate epoch-wise validation metrics

        if n_epochs is not None:
            self.n_epochs = n_epochs
            warnings.warn(
                "Setting `n_epochs` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if radius_0 is not None:
            self.radius_0 = radius_0
            warnings.warn(
                "Setting `radius_0` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if radius_n is not None:
            self.radius_n = radius_n
            warnings.warn(
                "Setting `radius_n` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if radius_cooling is not None:
            self.radius_cooling = radius_cooling
            warnings.warn(
                "Setting `radius_cooling` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if learning_rate_0 is not None:
            self.learning_rate_0 = learning_rate_0
            warnings.warn(
                "Setting `learning_rate_0` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if learning_rate_n is not None:
            self.learning_rate_n = learning_rate_n
            warnings.warn(
                "Setting `learning_rate_n` in `fit` is not recommended. It should be set in `__init__`.", UserWarning)
        if learning_rate_decay is not None:
            self.learning_rate_decay = learning_rate_decay
            warnings.warn(
                "Setting `learning_rate_decay` in `fit` is not recommended. It should be set in `__init__`.",
                UserWarning)

        assert self.n_epochs % 2 == 0, \
            "Somuclu's minimal number of epochs is 2, thus the SOM classifier must be trained in steps of 2 epochs"

        assert tracking_interval < self.n_epochs, "'tracking_interval' must be strictly smaller than 'n_epochs'"

        assert tracking_interval % 2 == 0, \
            "Somuclu's minimal number of epochs is 2, " \
            "thus the performance metrics can only be tracked in step sizes that are a multiple of 2"

        if save_intermediate_interval is not None:
            assert save_intermediate_interval % tracking_interval == 0, \
                "'save_intermediate_interval' must be an integer multiple of 'tracking_interval'"

        n_tracking_intervals = self.n_epochs // tracking_interval
        intervals_left = self.n_epochs % tracking_interval

        if intervals_left == 1:
            logger.info(f"# ### n_epochs % tracking_interval == 1, last epoch is omitted")

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
                logger.info(
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
                logger.info(f'# ### Epochs {current_epoch}--{current_epoch + tracking_interval - 1} took {et - st} sec')

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

    @staticmethod
    def _get_decay_function(
            val0: float,
            val1: float,
            n_epochs: int,
            strategy: Literal['linear', 'exponential'] = 'linear',
    ) -> Callable:
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

    # ### Internal performance metrics
    def calculate_quantization_error(
            self,
            x: np.ndarray,
    ) -> float:
        # Note: mse(x - BMU(x))
        # Get BMUs for the data
        # bmus = self.som_.get_bmus(activation_map=self.som_.get_surface_state(data=x))
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=x))

        # Iterate over data
        # n_events = x.shape[0]
        # quantization_error = np.zeros(n_events)
        # for i in range(n_events):
        #     quantization_error[i] = np.linalg.norm(x[i, :] - self.som_.codebook[bmus[i][0], bmus[i][1], :])

        # Extract BMU codebook vectors
        bmu_vectors = self.som_.codebook[bmus[:, 0], bmus[:, 1]]  # Shape: (n_samples, n_features)
        # Compute Euclidean distances
        quantization_error = np.linalg.norm(x - bmu_vectors, axis=1)

        return quantization_error.mean()

    def calculate_topographic_error(
            self,
            x: np.ndarray,
    ) -> float:
        # Note: Count how often the 1st and 2nd BMUs are not adjacent in the trained SOM
        assert self.som_topology == 'planar' and self.som_grid_type == 'rectangular', \
            'Topographical error calculation is currently only implemented for planar and rectangular SOMs'
        surface_state = self.som_.get_surface_state(data=x)
        bmu1 = self.som_.get_bmus(surface_state)
        bmu2 = SomClassifier._get_ith_bmus(surface_state=surface_state, i=2, som_dim0=self.som_dimensions[0])

        adjacency_bool = SomClassifier._check_adjacency(bmus0=bmu1, bmus1=bmu2)

        return np.logical_not(adjacency_bool).mean()

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

    def calculate_mean_impurity(
            self,
            impurity_measure: Literal['entropy', 'gini'],
    ) -> float:
        return self.calculate_unit_impurity(impurity_measure=impurity_measure).mean()

    def calculate_unit_impurity(
            self,
            impurity_measure: Literal['entropy', 'gini'] = 'entropy',
    ) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        # Calculate the frequency with which a class occurs at each unit,
        # account for case where unit is not bmu for any
        # frequencies = self.class_counts_per_unit_ / self.class_counts_per_unit_.sum(axis=2, keepdims=True)
        frequencies = np.divide(
            self.class_counts_per_unit_,
            self.class_counts_per_unit_.sum(axis=2, keepdims=True),
            where=self.class_counts_per_unit_.sum(axis=2, keepdims=True) != 0
        )
        # frequencies[self.class_counts_per_unit_.sum(axis=2, keepdims=True) == 0] = 0

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
        return impurity

    def check_possible_predictions(self) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        output_classes = np.sort(
            np.array([self.new_to_og_classes_dict_[key] for key in np.unique(self.som_unit_labels_)]))
        input_classes = np.sort(self.og_classes_)

        only_in_input = np.setdiff1d(input_classes, output_classes)

        if only_in_input.size != 0:
            logger.warning(
                '# ### The classes that are predicted by the trained SOM classifier do not include all classes that '
                'occur in the input')
            logger.info(f'# ### Classes {only_in_input} are not predicted')

        return only_in_input

    def get_activation_frequencies(
            self,
            x: np.ndarray,
    ):
        # Get BMUs for the data
        # bmus = self.som_.get_bmus(activation_map=self.som_.get_surface_state(data=x))
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=x))

        unique_rows, counts = np.unique(bmus, axis=0, return_counts=True)

        frequencies = np.zeros(self.som_dimensions)

        frequencies[tuple(unique_rows[:, 0]), tuple(unique_rows[:, 1])] = counts

        return frequencies / x.shape[0]

    # ### Performance evaluation and hyperparameter tuning
    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Union[np.ndarray, None] = None,
    ):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

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
            logger.info('# ### Starting Gridsearch with cross-validation for performance estimation')
        self.grid_search_.fit(X, y)

        if self.verbosity >= 1:
            logger.info(f'# ### Best parameters: {self.grid_search_.best_params_}')
            logger.info(f'# ### Best score: {self.grid_search_.best_score_}')

        # Update the instance's parameters with the best found parameters
        if refit is not False:
            # Get the attributes of the best classifier and update the Som classifier instance with them
            best_classifier_attribute_dict = copy.deepcopy(self.grid_search_.best_estimator_.__dict__)
            # Remove the gridsearch attribute, such that it remains unchanged
            best_classifier_attribute_dict.pop('grid_search_')
            self.__dict__.update(best_classifier_attribute_dict)

        # Return self with updated parameters
        return self

    # ### Saving and loading of a SOM classifier
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

    # ### General utilities
    # def set_cpu_affinity(self):
    #     p = psutil.Process(os.getpid())
    #     p.cpu_affinity(self.cores)

    #     os.environ["OMP_PROC_BIND"] = 'TRUE'
    #     os.environ["OMP_PLACES"] = ','.join(f'{{{core}}}' for core in self.cores)

    def reset(self):

        # Initialize all variables associated with a trained SOM classifier
        self.is_fitted_ = False
        self.som_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None
        self.epoch_wise_som_training_metrics_ = None
        self.grid_search_ = None
