
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple, List, Dict, Union, Any
from typing_extensions import Self
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .fcnn_model import FCNNModel


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    A three layer perceptron (MLP) classifier compatible with scikit-learn.

    This classifier wraps a fully connected neural network implemented in PyTorch
    while exposing a scikit-learn–style API.
    The model supports multi-class classification, automatic device selection
    (CPU/GPU), and provides the methods fit(), predict(), predict_proba(), score(), save(), and load().

    Attributes:
        layer_sizes (Tuple[int, int, int]): Sizes of the hidden layers in the fully connected neural network.
        n_epochs (int): Number of training epochs.
        data_loader_params (dict[str, Any] or None): Parameters passed to the PyTorch DataLoader. If None, defaults to ``{'batch_size': 128, 'shuffle': True, 'num_workers': 6}``.
        device (str or None): Device to use for training (e.g., ``'cpu'``, ``'cuda'``, ``'cuda:0'``). If None, CUDA is used when available, otherwise falls back to CPU.
        verbosity (int): Verbosity level for training logs.
        classes_ (np.ndarray or None): Class labels after re-indexing to integers starting from 0.
        class_counts_ (np.ndarray or None): Class counts from the training data.
        og_classes_ (np.ndarray or None): Original class labels before re-indexing.
        class_priors_ (np.ndarray or None): Empirical class priors.
        new_to_og_classes_dict_ (dict[int, Any] or None): Mapping from new integer labels back to original labels.
        data_set_ (TensorDataset or None): PyTorch tensor dataset constructed during fitting.
        data_loader_ (DataLoader or None): PyTorch DataLoader used for minibatch training.
        model_ (nn.Module or None): Neural network model.
        criterion_ (nn.Module or None): Loss function, PyTorch CrossEntropyLoss.
        optimizer_ (Optimizer or None): PyTorch Adam optimizer with learning rate 0.001.
        losses_ (list[float] or None): Training loss per epoch.
        n_corrects_ (list[int] or None): Number of correct predictions on the training data per epoch.
        is_fitted_ (bool): Whether the classifier has been fitted.
    """
    def __init__(
            self,
            layer_sizes: Tuple[int, int, int] = (128, 64, 32),
            n_epochs: int = 20,
            data_loader_params: Union[Dict[str, Any], None] = None,
            device: Union[str, None] = None,  # If None use gpu if available, else cpu
            verbosity: int = 1,
    ):
        """
        Initializes the MLPClassifier.

        Args:
            layer_sizes (Tuple[int, int, int]): Sizes of the hidden layers in the fully connected neural network.
            n_epochs (int): Number of training epochs.
            data_loader_params (dict[str, Any] or None): Parameters passed to the PyTorch DataLoader. If None, defaults to ``{'batch_size': 128, 'shuffle': True, 'num_workers': 6}``.
            device (str or None): Device to use for training (e.g., ``'cpu'``, ``'cuda'``, ``'cuda:0'``). If None, CUDA is used when available, otherwise falls back to CPU.
            verbosity (int): Verbosity level for training logs.
        """

        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_epochs = n_epochs
        self.data_loader_params = data_loader_params
        self.device = device
        self.verbosity = verbosity

        # Set device to default cuda device if available, else cpu
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)

        self.classes_ = None
        self.class_counts_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.data_set_ = None
        self.data_loader_ = None
        self.model_ = None
        self.criterion_ = None
        self.optimizer_ = None
        self.losses_ = None
        self.n_corrects_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Self:
        """
        Fit the MLP classifier to the provided training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            Self: The fitted classifier instance.

        Raises:
            ValueError: If X and y have incompatible shapes.
        """

        # ### Data processing and preparation
        # Check input data format
        X, y = check_X_y(X, y)

        # Rename classes to integers starting from 0 and extract label information
        (
            y, self.classes_, self.class_counts_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_
        ) = MLPClassifier._process_class_labels(y=y)

        # Get tensor dataset and data loader
        if self.data_loader_params is None:
            self.data_loader_params = {'batch_size': 128, 'shuffle': True, 'num_workers': 6}
            warnings.warn("No data_loader_params provided, using batch_size=128, shuffle=True, num_workers=6")

        self.data_set_, self.data_loader_ = MLPClassifier._get_data_set_data_loader(
            X=X,
            y=y,
            data_loader_params=self.data_loader_params
        )

        # ### Model training
        # Instantiate the softmax classifier
        self.model_ = FCNNModel(in_size=X.shape[1], out_size=self.classes_.shape[0])

        # Instantiate the loss function
        self.criterion_ = nn.CrossEntropyLoss()

        # Instantiate the optimizer
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=0.001)

        # Train the model
        self.losses_, self.n_corrects_ = MLPClassifier._train_loop(
            model=self.model_,
            data_loader=self.data_loader_,
            criterion=self.criterion_,
            optimizer=self.optimizer_,
            n_epochs=self.n_epochs,
            device=self.device,
            verbosity=self.verbosity,
        )

        self.is_fitted_ = True

        return self

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels for the given input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels using the original label encoding.

        Raises:
           NotFittedError: If `predict()` is used before calling `fit()`.
        """

        # Get softmax prediction
        y_proba = self.predict_proba(X)

        # Get class with max probability
        y_pred = y_proba.argmax(axis=1)

        # Get vector with the original labels
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
       Predict class probabilities for the given samples.

       Args:
           X (np.ndarray): Feature matrix of shape (n_samples, n_features).

       Returns:
           np.ndarray: Array of shape (n_samples, n_classes) containing class probabilities.

       Raises:
           NotFittedError: If `predict()` is used before calling `fit()`.
       """

        # Check whether fit was already called
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # Cast numpy array into torch Tensor
        x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)

        # Set model to eval mode and disable gradient tracking, forward pass,
        self.model_.eval()
        with torch.no_grad():
            model_out = self.model_(x_tensor)
            y_proba = torch.softmax(model_out, dim=1).numpy()

        return y_proba

    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Union[np.ndarray, None] = None,
    ):
        """
        Compute the macro F1 score of the classifier on the given dataset.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels.
            sample_weight (np.ndarray or None): Optional sample weights.

        Returns:
            float: Macro-averaged F1 score.

        Raises:
           NotFittedError: If `score()` is used before calling `fit()`.
        """

        y_pred = self.predict(X)

        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    @staticmethod
    def _train_loop(
            model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            n_epochs: int,
            device: torch.device,
            verbosity: int = 1,
    ) -> Tuple[List[float], List[int]]:

        # Adopted from:
        # (https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py)

        # Get number of events in train set
        n_events = len(data_loader.dataset)

        # Move model to device
        model.to(device=device)

        # Set model to train mode
        model.train()

        losses = []
        n_correct = []

        for epoch in range(n_epochs):

            loss_epoch = 0
            n_correct_epoch = 0

            for batch in data_loader:  # Get Batch

                # Extract data from current batch
                x, y = batch

                # Move data to device
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                model_out = model(x)

                # Calculate loss
                loss = criterion(model_out, y)

                # Clear old gradients
                optimizer.zero_grad()

                # Backward pass, gradient computation
                loss.backward()

                # Update parameters
                optimizer.step()

                # Track loss and number of correct predictions
                n_correct_epoch += MLPClassifier._get_num_correct(model_out, y)
                loss_epoch += loss.item()

            # Track loss and number of correct predictions
            losses.append(loss_epoch)
            n_correct.append(n_correct_epoch)

            if verbosity >= 2:
                print(f'# --- Epoch {epoch + 1}/{n_epochs} --- #')
                print(f'# Loss: {loss_epoch:.4f}')
                print(f'# Number of correct predictions: {n_correct_epoch}/{n_events}')

        # Move model to cpu
        model.cpu()

        # Set model to evaluation mode
        model.eval()

        return losses, n_correct

    def save(
            self,
            filename: str = 'mlp_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        """
        Save the fitted classifier to disk using `torch.save`.

        Args:
            filename (str): Name of the file to save the model to.
            filepath (str or None): Directory where the file will be saved. Defaults to current working directory.

        Returns:
            None
        """

        if filepath is None:
            filepath = os.getcwd()

        torch.save(self, os.path.join(filepath, filename))

    @classmethod
    def load(
            cls,
            filename: str = 'mlp_classifier.pkl',
            filepath: Union[str, None] = None,
            map_location: Union[str, torch.device] = 'cpu',
    ) -> Self:
        """
        Load a previously saved classifier from disk.

        Args:
            filename (str): Name of the saved file.
            filepath (str or None): Directory containing the saved file. Defaults to current working directory.
            map_location (str or torch.device): Device mapping for loading the model (e.g., 'cpu' or 'cuda').

        Returns:
            Self: The loaded classifier instance.
        """

        if filepath is None:
            filepath = os.getcwd()

        return torch.load(os.path.join(filepath, filename), map_location=map_location, weights_only=False)

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

    @staticmethod
    def _get_data_set_data_loader(
            X: np.ndarray,
            y: np.ndarray,
            data_loader_params: Dict[str, Any],
    ) -> Tuple[torch.utils.data.TensorDataset, DataLoader]:

        # Turn input data into tensors and create corresponding torch dataset
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)

        data_loader = DataLoader(dataset=dataset, **data_loader_params)

        return dataset, data_loader

    @staticmethod
    def _get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()



