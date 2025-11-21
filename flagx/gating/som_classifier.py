
import os
import warnings
import pickle
import copy

import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Union, Callable, Iterable, Any
from typing_extensions import Literal, Self
from somoclu import Somoclu
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from scipy.spatial.distance import cdist
from numba import njit, prange


class SomClassifier(BaseEstimator, ClassifierMixin):
    """
    Self-Organizing Map (SOM) classifier with scikit-learn–compatible API.

    This classifier uses Somoclu to train a 2D SOM grid in an unsupervised
    fashion and assigns class labels to SOM units by majority vote across labeled training samples.
    Predictions are computed using the BMU (best-matching unit) for each sample
    and the majority class associated with that unit.

    The classifier supports:
        • Unsupervised SOM training
        • Supervised unit annotation
        • Class probability estimation
        • Hyperparameter tuning (via GridSearchCV)
        • SOM quality metrics (quantization error, topographic error)
        • Visualization-oriented transformations
        • Model saving and loading

    Attributes:
        som_topology (Literal['planar', 'toroid']): SOM grid topology. Defaults to 'planar'.
        som_grid_type (Literal['rectangular', 'hexagonal']): Grid layout type. Defaults to 'rectangular'.
        som_dimensions (Tuple[int, int]): Dimensions of the SOM grid (n_columns, n_rows). Defaults to (10, 10).
        neighborhood (Literal['gaussian', 'bubble']): Neighborhood function type. Defaults to 'gaussian'.
        gaussian_neighborhood_sigma (float or None): Sigma for Gaussian neighborhood function. Defaults to 1.0.
        initialization (Literal['random', 'pca']): Codebook initialization method. Defaults to 'pca'.
        initial_codebook (np.ndarray or None): Custom initialization of SOM weights. Defaults to None.
        n_epochs (int): Number of SOM training epochs. Defaults to 100.
        radius_0 (float): Initial neighborhood radius. Negative values are interpreted as fractions of the grid size. Defaults to -0.5.
        radius_n (float): Final neighborhood radius. Defaults to 1.0.
        radius_cooling (Literal['linear', 'exponential']): Radius decay schedule. Defaults to 'linear'.
        learning_rate_0 (float): Initial learning rate. Defaults to 0.1.
        learning_rate_n (float): Final learning rate. Defaults to 0.01.
        learning_rate_decay (Literal['linear', 'exponential']): Learning rate decay schedule. Defaults to 'linear'.
        unlabeled_label (Any): Label indicating unlabeled samples. Defaults to -999.
        verbosity (int): Logging level. Defaults to 1.
        som_ (Somoclu): Trained SOM object.
        is_fitted_ (bool): Whether the model has been fitted.
        classes_ (np.ndarray | None): Class labels after re-indexing to integers starting from 0.
        class_counts_ (np.ndarray | None): Class counts from the training data.
        og_classes_ (np.ndarray | None): Original class labels before re-indexing.
        class_priors_ (np.ndarray | None): Empirical class priors.
        som_unit_labels_ (np.ndarray): Majority class per SOM unit.
        class_counts_per_unit_ (np.ndarray): Class histogram per SOM unit.
        grid_search_ (GridSearchCV or None): Grid search results if hyperparameter tuning was performed.
    """
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
            radius_0: float = -0.5,
            radius_n: float = 1.0,
            radius_cooling: Literal['linear', 'exponential'] = 'linear',
            learning_rate_0: float = 0.1,
            learning_rate_n: float = 0.01,
            learning_rate_decay: Literal['linear', 'exponential'] = 'linear',
            unlabeled_label: Any = -999,
            verbosity: int = 1,
    ):
        """
        Initializes the MLPClassifier.

        Parameters:
            som_topology (Literal['planar', 'toroid']): SOM grid topology. Defaults to 'planar'.
            som_grid_type (Literal['rectangular', 'hexagonal']): Grid layout type. Defaults to 'rectangular'.
            som_dimensions (Tuple[int, int]): Dimensions of the SOM grid (n_columns, n_rows). Defaults to (10, 10).
            neighborhood (Literal['gaussian', 'bubble']): Neighborhood function type. Defaults to 'gaussian'.
            gaussian_neighborhood_sigma (float or None): Sigma for Gaussian neighborhood function. Defaults to 1.0.
            initialization (Literal['random', 'pca']): Codebook initialization method. Defaults to 'pca'.
            initial_codebook (np.ndarray or None): Custom initialization of SOM weights. Defaults to None.
            n_epochs (int): Number of SOM training epochs. Defaults to 100.
            radius_0 (float): Initial neighborhood radius. Negative values are interpreted as fractions of the grid size. Defaults to -0.5.
            radius_n (float): Final neighborhood radius. Defaults to 1.0.
            radius_cooling (Literal['linear', 'exponential']): Radius decay schedule. Defaults to 'linear'.
            learning_rate_0 (float): Initial learning rate. Defaults to 0.1.
            learning_rate_n (float): Final learning rate. Defaults to 0.01.
            learning_rate_decay (Literal['linear', 'exponential']): Learning rate decay schedule. Defaults to 'linear'.
            unlabeled_label (Any): Label indicating unlabeled samples. Defaults to -999.
            verbosity (int): Logging level. Defaults to 1.
        """
        super().__init__()
        # ### Initialize parameters
        self.som_topology = som_topology
        self.som_grid_type = som_grid_type
        self.som_dimensions = som_dimensions
        self.neighborhood = neighborhood
        self.gaussian_neighborhood_sigma = gaussian_neighborhood_sigma
        self.initialization = initialization
        self.initial_codebook = initial_codebook
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

        # Internal flags
        self._is_fitted = False
        self._labeled_data = False

        # ### Initialize all variables associated with a trained SOM classifier
        self.som_ = None
        self.n_features_in_ = None
        # These are only relevant if trained on labeled data:
        self.classes_ = None
        self.class_counts_ = None
        self.class_priors_ = None
        self.og_classes_ = None
        self.new_to_og_classes_dict_ = None
        self.class_counts_per_unit_ = None
        self.som_unit_labels_ = None

        # Epoch wise training
        # self.epoch_wise_som_training_metrics_ = None  # Only relevant if SOM is trained epoch wise

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
            kerneltype=0,  # Only cpu training
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
    ) -> Self:
        """
        Train the SOM on input data and annotate units if labeled data is provided.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training labels. Unlabeled samples must be marked using `unlabeled_label`.

        Returns:
            Self: The fitted classifier instance.

        Raises:
            ValueError: If the feature dimension does not match a previous fit call.
            UserWarning: If fitting continues from an already-initialized SOM.
        """

        # Check input data format
        X, y = check_X_y(X, y)

        # Check if .fit() was called already
        if self._is_fitted:

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
        if self._is_fitted and self._labeled_data:
            warnings.warn(
                "SOM unit annotations are based on current labeled data. "
                "To include previous training labels, call `.annotate_som()` with all labeled data.", UserWarning
            )

        # Annotate the trained SOM's units
        self.annotate_som(X=X, y=y)

        # If no labeled data passed to .fit(), raise UserWarning
        if not self._labeled_data:
            warnings.warn(
                'No labeled data provided. Only the SOM component is trained in an unsupervised fashion.',
                UserWarning
            )

        # Set flag indicating that SOM classifier is fitted
        self._is_fitted = True
        self.is_fitted_ = True

        return self

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Predict labels for new samples using the BMU and unit annotations.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted labels in the original label space.

        Raises:
            NotFittedError: If the classifier has not been fitted.
            UserWarning: If units without labels are BMU for some samples.
        """

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # If SOM Classifier was trained on labeled data compute prediction
        if self._labeled_data:

            # Get BMU of events
            bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

            # Get prediction as label of BMU of event
            y_pred = self.som_unit_labels_[tuple(bmus[:, 0]), tuple(bmus[:, 1])]

            # Get original labels
            y_pred = np.vectorize(self.new_to_og_classes_dict_.get)(y_pred)

            # Raise UserWarning if label less-unit is BMU at prediction time
            if np.any(y_pred == -1):
                warnings.warn(
                    f"For events {np.argwhere(y_pred == -1).flatten().tolist()} the BMU has no label "
                    f"(support of the unit during training was 0). "
                    f"Its predicted class is -1 ~= unknown/undeterminable from the training data",
                    UserWarning)

        # No labeled training data, return a dummy prediction vector and raise UserWarning
        else:
            y_pred = np.full(X.shape[0], self.unlabeled_label)
            warnings.warn(
                'No labeled training data was provided -- label prediction is not possible.',
                UserWarning
            )

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Estimate class probabilities based on the class distribution of the BMU.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Class probabilities per sample.

        Raises:
            NotFittedError: If the classifier has not been fitted.
            UserWarning: If no labeled data was provided.
        """
        #  Confidence in prediction based on class distribution of events with unit as a bmu

        # Check whether the SOM classifier was fitted
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # If SOM Classifier was trained on labeled data compute prediction probabilities
        if self._labeled_data:
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
                'No labeled training data was provided. Prediction is not possible.',
                UserWarning
            )

        return y_proba

    def annotate_som(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> Self:
        """
        Assign class labels to SOM units by computing the majority class
        among samples for which the respective unit is the BMU.

        Args:
            X (np.ndarray): Input features for annotation.
            y (np.ndarray): Labels corresponding to X.

        Returns:
            Self: Updated classifier instance with unit annotations.

        Raises:
            RuntimeError: If SOM has not been trained prior to annotation.
            UserWarning: If some SOM units have no support from labeled samples.
        """

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
            self._labeled_data = True
        else:
            # Reset all class attributes that are associated with a SOM classifier trained on labeled data
            self._reset_to_unlabeled()

        return self

    # ### hyperparameter_tuning() ######################################################################################
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
        """
        Perform hyperparameter optimization using Scikit-learn's GridSearchCV.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            param_grid (dict or None): Hyperparameter search space.
            cv (int or CrossValidator): Number of folds or cross-validation strategy. Defaults to 5.
            scoring (str, callable, or None): Scoring metric. If 'internal', macro-F1 is used. Defaults to 'internal'.
            refit (bool or str or callable): Whether to refit using the best model. Defaults to True.
            gridsearchcv_kwargs (dict or None): Additional parameters for GridSearchCV. Defaults to None.

        Returns:
            Self: Classifier with updated best-found parameters.

        Notes:
            The method updates the instance with GridSearchCV stored in the grid_search_ attribute.
        """

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

    # ### Scores and performance metrics ###############################################################################
    # score(), activation_frequencies(), quantization_error(), topographic_error(), unit_impurity(), mean_impurity()
    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Union[np.ndarray, None] = None,
    ):
        """
        Compute macro F1 score on the provided data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.
            sample_weight (np.ndarray or None): Optional sample weights.

        Returns:
            float: Macro-averaged F1 score.
        """
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def activation_frequencies(
            self,
            X: np.ndarray,
    ):
        """
        Compute activation frequencies of each SOM unit on the given data.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Array of shape (som_dim0, som_dim1) with normalized activation counts per unit.
        """
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
        """
        Compute the SOM quantization error.

        Quantization error = mean Euclidean distance between samples and
        the codebook vector of their BMU.

        Args:
            X (np.ndarray): Input features.

        Returns:
            float: Mean quantization error.
        """
        # Note: mse(x - BMU_vec(x))

        # Get BMUs for the data
        bmus = self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))

        # Extract BMU codebook vectors
        bmu_vectors = self.som_.codebook[bmus[:, 1], bmus[:, 0], :]  # Shape: (n_samples, n_features)

        # Compute Euclidean distances
        quantization_error = np.linalg.norm(X - bmu_vectors, axis=1)

        return quantization_error.mean()

    def topographic_error(
            self,
            X: np.ndarray,
    ) -> float:
        """
        Compute the SOM topographic error.

        Topographic error = proportion of samples where the 1st and 2nd BMUs
        are not adjacent on the SOM grid.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            float: Topographic error.

        Raises:
            NotImplementedError:
                If SOM topology is not planar rectangular.
        """

        # Note: Count how often the 1st and 2nd BMUs are not adjacent in the trained SOM
        if self.som_topology != 'planar' or self.som_grid_type != 'rectangular':
           raise NotImplementedError(
               'Topographical error calculation is currently only implemented for planar and rectangular SOMs'
           )

        surface_state = self.som_.get_surface_state(data=X)
        bmu1 = self.som_.get_bmus(surface_state)
        bmu2 = SomClassifier._get_ith_bmus(surface_state=surface_state, i=2, som_dim0=self.som_dimensions[0])

        adjacency_bool = SomClassifier._check_adjacency(bmus0=bmu1, bmus1=bmu2)

        return np.logical_not(adjacency_bool).mean()

    def unit_impurity(
            self,
            impurity_measure: Literal['entropy', 'gini'] = 'entropy',
    ) -> np.ndarray:

        """
        Compute class impurity for each SOM unit.

        Args:
            impurity_measure (Literal['entropy', 'gini']): Impurity metric.

        Returns:
            np.ndarray: Impurity per SOM unit.

        Raises:
            UserWarning: If classifier was trained without labeled data.
        """

        check_is_fitted(self, 'is_fitted_')

        if self._labeled_data:
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
                'No labeled data provided for training -- calculating unit impurity is not possible.',
                UserWarning
            )

        return impurity

    def mean_impurity(
            self,
            impurity_measure: Literal['entropy', 'gini'],
    ) -> float:
        """
        Compute the mean impurity across all SOM units.

        Args:
            impurity_measure (Literal['entropy', 'gini']): Impurity metric.

        Returns:
            float: Mean impurity over all units.
        """
        return self.unit_impurity(impurity_measure=impurity_measure).mean()

    def unpredictable_classes(self) -> np.ndarray:
        """
        Identify classes that were seen during training but cannot be predicted
        because no SOM unit was annotated with those labels.

        Returns:
            np.ndarray: Array of missing/unpredictable classes.

        Raises:
            UserWarning: If no labeled data was provided.
        """

        check_is_fitted(self, 'is_fitted_')

        if self._labeled_data:
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
                'No labeled training data was provided',
                UserWarning
            )

        return only_in_input

    # ### save(), load(), reset(), confidence_threshold() ##############################################################
    def save(
            self,
            filename: str = 'som_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        """
        Save the trained classifier to disk using pickle.

        Args:
            filename (str): Output filename.
            filepath (str or None): Directory to save the file. Defaults to CWD.

        Returns:
            None
        """
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
        """
        Load a saved classifier instance from disk.

        Args:
            filename (str): File to load.
            filepath (str or None): Directory containing the file.

        Returns:
            Self: Loaded classifier instance.
        """
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    def reset(self):
        """
        Reset the classifier to its untrained state, clearing the trained SOM, class annotations, and metadata.

        Returns:
            None
        """

        # Initialize all variables associated with a trained SOM classifier
        self._is_fitted = False
        del self.is_fitted_
        self._labeled_data = False
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

    # ### Auxiliary functions ##########################################################################################
    def _reset_to_unlabeled(self):
        self._labeled_data = False
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
    def _get_ith_bmus(
            surface_state: np.ndarray,
            i: int,
            som_dim0: int,
    ) -> np.ndarray:

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
        """
        Project samples onto the SOM grid and generate visualization-friendly scattered BMU coordinates.

        Args:
            X (np.ndarray): Input data.

        Returns:
            Tuple:
                bmus (np.ndarray): BMU coordinates for each sample.
                bmus_scattered (np.ndarray): Scattered BMU coordinates for visualization for each sample.
                som_unit_ids (np.ndarray): Unit ID in row-major format for each sample.
                radii (np.ndarray): Radius proportional to activation frequency across input data of BMU for each sample.

        Raises:
            NotFittedError: If the classifier has not been trained.
        """

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
        radii = 0.5 * np.sqrt(counts) / np.sqrt(counts.max()) # Area should be proportional to count -> use square root
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

    def _get_row_major_positions(
            self,
            bmus: np.ndarray,
            start_from: int = 0,
    ):
        pos = bmus[:, 0] * self.som_dimensions[0] + bmus[:, 1] + start_from
        return pos

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
