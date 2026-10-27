
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from typing import Union, Tuple, Dict, Self, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
from torch.autograd import Variable
from scipy.stats import spearmanr


class DgcytofClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            val_size: float = 0.2,
            layer_sizes: Tuple[int, int, int] = (128, 64, 32),
            n_epochs: int = 20,
            train_params: Union[dict, None] = None,
            verbosity: int = 0,
    ):
        super().__init__()
        self.val_size = val_size
        self.layer_sizes = layer_sizes
        self.n_epochs = n_epochs
        self.verbosity = verbosity
        self.train_params = train_params
        if self.train_params is None:
            self.train_params = {'batch_size': 128, 'shuffle': True, 'num_workers': 6}

        self.is_fitted_ = False
        self.classes_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.device_ = None
        self.fcnn_model_ = None
        self.validation_results_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Self:

        X, y = check_X_y(X, y)

        # Rename classes to integers starting from 0
        y, self.classes_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_ = \
            DgcytofClassifier._process_class_labels(y=y)
        # Add label for unclassifiable cells to dict
        self.new_to_og_classes_dict_[-1] = -1

        # Split data into train and validation set
        x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=self.val_size, random_state=42)

        # Define device on which to train
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Turn into tensors and create corresponding torch datasets
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        dataset_val = torch.utils.data.TensorDataset(x_val, y_val)

        # ### Model training
        # Instantiate the softmax classifier
        self.fcnn_model_ = FCNNModel(in_size=x_train.shape[1], out_size=np.unique(y_train).shape[0])
        # Move model and tensors to device
        self.fcnn_model_.to(device=self.device_)
        # Use the function provided by dgcytof for training
        # (https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py)
        DgcytofClassifier.train_model(
            model_fc=self.fcnn_model_,
            X_train=dataset_train,
            max_epochs=self.n_epochs,
            params_train=self.train_params,
            device=self.device_,
        )

        # Move model back to cpu
        self.fcnn_model_ = self.fcnn_model_.to("cpu")

        # ### Validation step
        # Use the validation set to get an estimate of the prediction confidence
        # (for later correction of the predictions)
        # Use the function provided by dgcytof for the validation step
        # (https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py)
        self.validation_results_ = DgcytofClassifier.validate_model(
            model_fc=self.fcnn_model_,
            val_tensor=dataset_val,
            classes=np.unique(y_train).tolist(),
            params_val={'batch_size': 10000, 'shuffle': False, 'num_workers': 6}
        )

        self.is_fitted_ = True

        return self

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        x_df = pd.DataFrame(X)
        x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)

        # Predict and correct the prediction according to the Dgcytof method
        # Adopted the function provided by dgcytof for the calibration step
        # (https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py)
        _, y_pred = DgcytofClassifier._calibrate_data(
            model_fc=self.fcnn_model_,
            X_test=x_tensor,
            classes=self.classes_.tolist(),
            validation_results=self.validation_results_,
            unlabeled_data=x_df,
        )

        # Get vector with the original labels
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        if np.any(y_pred == -1):
            warnings.warn(
                f"The events:\n{np.argwhere(y_pred == -1).flatten().tolist()}\nwere not classifiable with a high "
                f"confidence. They are classified as label -1 ~= 'new cell type'",
                UserWarning)

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)

        # Get softmax probabilities
        with torch.no_grad():  # Ensure no gradients are computed
            y_proba = F.softmax(self.fcnn_model_(x_tensor), dim=1)

        # Convert tensor to NumPy array and return
        return y_proba.cpu().numpy()

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
            sample_weight[y_pred == -1] = 0  # Todo

        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def save(
            self,
            filename: str = 'dgcytof_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()

        torch.save(self, os.path.join(filepath, filename))

    @classmethod
    def load(
            cls,
            filename: str = 'dgcytof_classifier.pkl',
            filepath: Union[str, None] = None,
            map_location: Union[str, torch.device] = 'cpu'
    ) -> Self:
        if filepath is None:
            filepath = os.getcwd()

        return torch.load(os.path.join(filepath, filename), map_location=map_location, weights_only=False)

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
    def train_model(
            model_fc: nn.Module,
            X_train: torch.utils.data.TensorDataset,
            max_epochs: int = 20,
            params_train: Union[Dict[str, Any], None] = None,
            device: Union[torch.device, None] = None,
    ) -> None:
        """
        Train loop for the fully connected neural network at the core of the Dgcytof classifier.
        Function is an adapted version of train_model() from:
        https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py

        Trains the entered deep learning model using Pytorch. Criterion is CrossEntropyLoss and optimizer is Adam
        optimizer with learning rate 0.001. The input model is set to evaluation mode after training.

        Args:
            model_fc (nn.Module):
                A PyTorch model with a `forward()` method. It should perform
                classification using `argmax` in its design.
            X_train (torch.utils.data.TensorDataset):
                Training dataset wrapped in a TensorDataset.
            max_epochs (int, optional):
                Number of training epochs. Defaults to 20.
            params_train (dict, optional):
                Parameters for the DataLoader. Expected keys are:
                - 'batch_size' (int): Size of each batch (default: 128)
                - 'shuffle' (bool): Whether to shuffle the data (default: True)
                - 'num_workers' (int): Number of subprocesses for data loading (default: 6)
            device (torch.device, optional):
                Device to run the training on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            None
        """

        if params_train is None:
            params_train = {'batch_size': 128, 'shuffle': True, 'num_workers': 6}

        if device is None:
            device = torch.device('cpu')

        train_loader = data_utils.DataLoader(dataset=X_train, **params_train)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_fc.parameters(), lr=0.001)

        for epoch in range(max_epochs):  # loop over the dataset multiple times

            total_loss = 0
            total_correct = 0

            for data in train_loader:  # Get Batch
                samples, labels = data

                # ###### Start, My addition: ###### #
                # Move data to device
                samples = samples.to(device)
                labels = labels.to(device)
                # ###### End, My addition ###### #

                preds = model_fc(samples)  # Pass Batch
                loss = criterion(preds, labels)  # Calculate Loss

                optimizer.zero_grad()  # Zero Gradients
                loss.backward()  # Calculate Gradients
                optimizer.step()  # Update Weights

                # loss_outputs.append(outputs)
                total_loss += loss.item()
                total_correct += DgcytofClassifier.get_num_correct(preds, labels)

            print("epoch", epoch, "total_correct:", total_correct, "loss:", loss.item(), "total_loss:", total_loss)
        model_fc.eval()

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def _calibrate_data(model_fc, X_test, classes, validation_results, unlabeled_data):
        # Function adopted from
        # https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py
        # Added comments and slight changes such that an updated prediction is returned
        # (as well as the unclassifiable samples as before), unclassifiable samples are predicted to have label - 1
        # Also, did not include all events in calculation of correlation matrices, to not exceed the memory limit.
        # Previously this was only done for class 1

        """
        Function adapted with minor changes from:
        https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py

        Changes to the original function are highlighted by inline comments of the form:
        # ###### Start, My addition ###### #
        ...
        # ###### End, My addition ###### #

        Note:
        A limit for the dimension of the correlation matrices is introduced to avoid memory issues.
        In the original function this was only done for some classes of the input data.

        ################################################################################################################
        # ### Original function description:

        Calibrates the test set of the data based on the training model and validation results in performing over the test set.
        Test data either are classified more accurately or labeled as a new subtype based on the minimum threshold
        in classification. Minumum threshold is computed by obtaining the lowest correlation probability from validation results.
        Returns calibrated incorrect data.

        **Params**:

        * model_fc: Trained PyTorch model, must have a forward function and utilize argmax as classification in its design
        * X_test: CyTOF data for test set.
        * classes: List of types of cells
        * validation_results: zip created by validate_model. Contains the following keys:
            * pred: Predicted label of a data point
            * label: Actual label of a data point
            * out: Output value of the data running forward through model_fc

        **Returns**:

        * updated_incorrect_data: Calibrated incorrect data.
        """

        # ###### Start, My addition ###### #
        corr_mtrx_dim_limit = 12000
        # ###### End, My addition ###### #

        # Create list of instances from the val set where the prediction was correct,
        # save the predicted probability of the predicted class (i.e. the confidence of the class prediction)
        correct_pred_info = [(label, pred, np.max(F.softmax(out, dim=0).data.numpy())) for pred, label, out in
                             validation_results if (pred == label)]
        # Get a list of the prediction probs of the correct predictions
        corr_prob = [prob for label, pred, prob in correct_pred_info]

        # For the unlabeled data get the model output (forward pass), and the predicted class label (argmax of output)
        test_outputs = model_fc(Variable(X_test))
        _, test_predicted = torch.max(test_outputs.data, 1)  # Find the class index with the maximum value.

        # Get the predicted class probs for the unlabeled data
        output_logits = F.softmax(test_outputs, dim=1).data.numpy()
        output_labels = test_predicted.data.numpy()  # Why ???
        # Get the predicted probability of the predicted class (i.e. the confidence of the class prediction), and round
        probabilities = np.max(output_logits, axis=1)
        tem = [round(i, 4) for i in probabilities]
        # incorrect_index = [tem.index(a) for a in tem if a <= min(corr_prob)]
        # Accept the prediction for the new unlabelled data (correct_index),
        # if its confidence is larger than the min confidence on the validation set
        # otherwise reject the prediction (incorrect_index)
        correct_index = [i for i, v in enumerate(tem) if v > min(corr_prob)]
        incorrect_index = [i for i, v in enumerate(tem) if v <= min(corr_prob)]
        # Get the data vectors for the unlabeled events for which the prediction was rejected
        # (due to too low confidence)
        incorrect_data = pd.DataFrame([X_test[i].data.numpy() for i in incorrect_index])
        print("A total of " + str(len(incorrect_data)) + "  incorrect labeling as been found")

        # ###### Start, My addition: ###### #
        y_pred = np.full(len(X_test), -1)  # -1 indicates unassigned
        # Assign predicted labels where the prediction is deemed to be correct (high confidence)
        for i in correct_index:
            y_pred[i] = test_predicted[i].item()
        # ###### End, My addition: ###### #

        # Get copy of df with unlabeled data
        test_with_labels = unlabeled_data.copy()
        # Get the predicted labels for the unlabeled data, add as label column in df
        test_with_labels['label'] = test_predicted.numpy()
        test_with_labels = test_with_labels.reset_index(drop=True)
        # Subset the df to the instances where the prediction was deemed correct (high confidence)
        test_correct_with_labels = test_with_labels.iloc[correct_index]

        # Initialize dict
        test_correct_list = dict()
        # Iterate over the classes in the training set
        for subtype_no in range(0, len(classes)):
            # Dict entry with 'subtype_no' as key is the intensity vectors of the unlabeled events
            # that were predicted to be of class 'subtype_no'
            test_correct_list[subtype_no] = test_correct_with_labels[test_correct_with_labels.label == subtype_no].drop(
                ['label'],
                axis=1).values.tolist()
        # => {label: data matrix from unlabeled events predicted to be label}

        # Each row (intensity vector corresponding to an event vor which the prediction was rejected) becomes a list
        # => list of lists
        incorrect_list = incorrect_data.values.tolist()

        # Initialize dict
        rho_dict = dict()

        # Iterate over the classes in the training set
        for i in range(len(classes)):
            # Correlation of intensity vectors of unlabeled events that were deemed to be correctly classified as
            # class i and the intensity vectors of unlabeled events that were deemed to be incorrectly classified

            # print(np.array(test_correct_list[i] + incorrect_list))
            # print(np.array(test_correct_list[i] + incorrect_list).shape)
            # print(np.transpose(test_correct_list[i] + incorrect_list))
            # print(np.transpose(test_correct_list[i] + incorrect_list).shape)
            # np.transpose(test_correct_list[i] + incorrect_list) has dim: (channels, n_events)
            # spearmanr: column represents a variable, with observations in the rows
            # => Compute correlation between events (= variable)

            # ###### Start, My addition ###### #
            rho_matrix, _ = spearmanr(np.transpose(test_correct_list[i][-corr_mtrx_dim_limit:] + incorrect_list))
            if rho_matrix.ndim == 0:  # If it's a scalar, convert to 2D
                rho_matrix = np.array([[rho_matrix]])
            rho_dict[i] = rho_matrix
            # if i == 1:
            #     rho_dict[i], _ = spearmanr(np.transpose(test_correct_list[i][-12000:] + incorrect_list))
            # else:
            #     rho_dict[i], _ = spearmanr(np.transpose(test_correct_list[i] + incorrect_list))
                # print(np.transpose(test_correct_list[i] + incorrect_list).shape)
                # print(spearmanr(np.transpose(test_correct_list[i] + incorrect_list))[0])
                # print(spearmanr(np.transpose(test_correct_list[i] + incorrect_list))[0].shape)
            # ###### End, My addition ###### #

        # => {label: correlation matrix of size n_events x n_events}

        #################

        # Intitialize dicts
        wrongly_incorrect_index = dict()
        wrongly_incorrect_corr = dict()

        # Iterate over the classes in the training set
        for subtype_no in range(0, len(classes)):
            # Active learning,
            # Input:
            # - data matrix from unlabeled events predicted to be label 'subtype_no'
            # - correlation matrix of unlabeled events that were deemed to be correctly classified as class 'subtype_no'
            #   and unlabeled events that were deemed to be incorrectly classified
            # Output:
            # - List of indices of wrongly classified events that probably belong to class 'subtype_no'
            #   (high correlation)
            # - The corresponding average correlation value

            # ###### Start, My addition ###### #
            wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = \
                DgcytofClassifier._active_learning_index(
                    test_correct_list[subtype_no][-corr_mtrx_dim_limit:], rho_dict[subtype_no]
                )
            # if subtype_no == 1:
            #     wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = \
            #         DgcytofClassifier._active_learning_index(
            #             test_correct_list[subtype_no][-12000:], rho_dict[subtype_no])
            # else:
            #     wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = \
            #         DgcytofClassifier._active_learning_index(test_correct_list[subtype_no], rho_dict[subtype_no])
            # ###### End, My addition ###### #

        ###################

        # Initialize df of dim (n_incorrectly classified events, number of classes)
        temp = pd.DataFrame(np.zeros((len(incorrect_list), len(classes))))
        # Iterate over the classes in the training set
        for subtype_no in range(0, len(classes)):
            # For each class enter the previously computed average correlation value into the df
            temp[subtype_no][wrongly_incorrect_index[subtype_no]] = wrongly_incorrect_corr[subtype_no]
        # => Df with average correlations values for each wrongly classified event,
        #    they indicate the most probable class they actually belong to

        # Drop rows with all zeros
        temp_1 = temp.loc[(temp != 0).any(axis=1)]  # ### My addition: add axis keyword

        # Set highest value in each row to 1 and other entries to 0, store in df
        # => Df with the corrected cell type predictions
        m = np.zeros_like(temp_1.values)
        m[np.arange(len(temp_1)), temp_1.values.argmax(1)] = 1
        temp_2 = pd.DataFrame(m, columns=temp_1.columns, index=temp_1.index).astype(int)

        # Initialize dict
        updated_test = dict()
        # Iterate over the classes in the training set
        # indices (of cells) that should be added to respective celltype
        for subtype_no in range(0, len(classes)):
            # In column of temp_2 corresponding to 'subtype_no'
            # (-> holds info which cells are reassigned to 'subtype_no'),
            # Retrieve the indices (= events) where this columns has a 1 entry => np array with those indices
            temp_index = np.array(temp_2[subtype_no].index[temp_2[subtype_no].to_numpy().nonzero()])
            # Add dict entry:
            # - test_correct_list = {label: data matrix from unlabeled events predicted to be label, as a list of lists}
            # - incorrect_data.iloc[temp_index].values.tolist()
            #   = datamatrix from unlabeled events now also predicted to be label, as a list of lists
            updated_test[subtype_no] = test_correct_list[subtype_no] + incorrect_data.iloc[temp_index].values.tolist()

            # ###### Start, My addition ###### #
            # Iterate over the indices of events that are now assigned to 'subtype_no'
            # - 'temp_index' contains indices of rows in 'incorrect_data' that were reassigned to 'subtype_no',
            #   'incorrect_data' = df with the data vectors for the unlabeled events for which the prediction was
            #                      rejected
            # - 'incorrect_index' = indices of the rows in X_test for which the prediction was deemed wrong,
            #   correspond to the rows in 'incorrect_data'
            # => Each row in 'incorrect_data' corresponds to a row in 'X_test',
            #    which can be identified via 'incorrect_index'
            for idx in temp_index:
                y_pred[incorrect_index[idx]] = subtype_no
            # ###### End, My addition ###### #

        # => updated_test = {label: data matrix from unlabeled events now finally predicted to be label,
        #                           as a list of lists}

        # incorrect_data = data vectors for the unlabeled events for which the prediction was rejected, as df
        # Drop the events from the df that were reassigned with a new label
        # The remaining samples in updated_incorrect_data are those that could not be confidently assigned to any class
        # and might require further analysis.
        updated_incorrect_data = incorrect_data.drop(np.array(temp_2.index), axis=0)

        return updated_incorrect_data, y_pred

    @staticmethod
    def _active_learning_index(test_correct_list, rho, indices=True):
        """
        Helper function, identical to active_learning_index() from:
        https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py
        """
        wrongly_incorrect_index = []  # indices of wrongly predicted incorrect
        # Number of events that were deemed to be correctly classified as some label
        correct_size = len(test_correct_list)  # length of correct class
        # Subset the correlation matrix to those events
        rho_correct = rho[:correct_size, :correct_size]  # correlation matrix of correct class
        # Get the entries of the upper triangle matrix
        rho_correct_array = rho_correct[np.triu_indices(correct_size, k=1)]  # array of correlations of correct class
        # Compute the mean for those entries
        rho_avg = np.mean(rho_correct_array)
        # => 'rho_avg' = mean of the correlation values among events that were deemed as correctly classified

        if indices == True:
            # For samples i (beyond 'correct_size', range(correct_size + 1, len(rho) ) that were deemed to be
            # incorrectly classified do:
            # - Compute average correlation of sample i with all correctly classified samples
            #   (np.mean(rho[i, :correct_size]))
            # - If the mean is larger than the internal mean of correctly classified samples (mean > rho_avg),
            #   store the event in 'wrongly_incorrect_index',
            #   i.e. it likely belongs to the class of the correctly classified events
            # - Also store the average correlation value of the event with the correctly classified samples
            #   Needed later on as a tiebreaker
            wrongly_incorrect_index = [(i - correct_size) for i in range(correct_size + 1, len(rho))
                                       if np.mean(rho[i, :correct_size]) > rho_avg]
            wrongly_incorrect_corr = [np.mean(rho[i, :correct_size]) for i in range(correct_size + 1, len(rho))
                                      if np.mean(rho[i, :correct_size]) > rho_avg]
            return wrongly_incorrect_index, wrongly_incorrect_corr, rho_avg
        else:
            return rho_avg

    @staticmethod
    def validate_model(model_fc, val_tensor, classes,
                       params_val={'batch_size': 10000, 'shuffle': False, 'num_workers': 6}):
        """
        Function is identical to validate_model() from:
        https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py

        ################################################################################################################
        # ### Original function description:
        Runs validation on the validation dataset, print out the performance of the trained model for all cell types and returns
        them as a zip.

        **Params**:

        * model_fc: Trained PyTorch model, must have a forward function and utilize argmax as classification in its design
        * val_tensor: Validation dataset as a Torch tensor.
        * classes: List of types of cells
        * params_val: dictionary containing information for dataloader, requires at least a batch_size, shuffle, and num_workers
        keys.
            * batch_size: Number of data points in a single batch, default 128
            * shuffle: Shuffle the batches prior to training, default True
            * num_workers: Number of processes that will be used to load data, default 6

        **Returns**:

        * Zip of listed results. Each respective row contains pred,label,out in validation_results
            * pred: Predicted label of a data point
            * label: Actual label of a data point
            * out: Output value of the data running forward through model_fc

        """
        assert (len(set(classes)) > 1), "There must be at least 2 classes"

        labels = len(classes)

        model_fc.eval()

        val_loader = data_utils.DataLoader(dataset=val_tensor, **params_val)

        class_correct = list(0. for i in range(labels))
        class_total = list(0. for i in range(labels))

        val_correct = 0
        val_total = 0

        for data in val_loader:
            val_samples, val_labels = data
            val_outputs = model_fc(Variable(val_samples))
            _, val_predicted = torch.max(val_outputs.data, 1)  # Find the class index with the maximum value.
            c = (val_predicted == val_labels).squeeze()
            for i in range(val_labels.shape[0]):
                label = val_labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum()

        print("Accuracy:", round(100 * val_correct.item() / val_total, 4))
        print('-' * 100)
        for i in range(labels):
            print('Accuracy of {} : {}'.format(
                classes[i], round(100 * class_correct[i] / class_total[i], 3)))

        # Return. validation results
        return list(zip(val_predicted, val_labels, val_outputs))


# ### Define the FCNN Softmax model
# Dimensions chosen as described in Dgcytof paper and
# https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb
class FCNNModel(nn.Module):
    def __init__(self, in_size, out_size, layer_sizes: Tuple[int, int, int] = (128, 64, 32)):
        super(FCNNModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], out_size, bias=True)
        # self.softmax = nn.Softmax(dim=1)  # Softmax for the output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.softmax(x)  # Softmax activation for output
        # Do not apply softmax, CrossEntropyLoss does so internally
        return x
