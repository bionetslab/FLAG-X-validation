"""
sklearn-style wrapper for DeepCyTOF (Li et al. 2017, https://doi.org/10.1093/bioinformatics/btx448).

Official code: https://github.com/KlugerLab/deepcytof (Apache-2.0),
pinned to commit ac924b10a77cf2d5c7ea43ea39a172898d71abb1.
All DeepCyTOF building blocks below are copied from that repository (Keras 1.x code). Every copied block is marked
with the source URL + file + line numbers. Changes w.r.t. the original code are marked with '# ### Change:'.

NOTE: The original code (and therefore this module) requires the old Keras 1.x API
(W_regularizer, nb_epoch, merge, initializations, ...) => use the conda env 'fx_deepcytof' (envs/deepcytof.yml,
Python 3.5, Keras 1.2.2, TensorFlow 1.0.1). Hence, this module must stay Python 3.5 compatible (no f-strings,
no variable annotations, no typing.Self).
"""

import os
import math
import pickle
import warnings
import numpy as np
from typing import Union, Tuple, TypeVar, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
import sklearn.preprocessing as prep
from sklearn.cross_validation import train_test_split  # Original import (scikit-learn < 0.20)
from sklearn.neighbors import NearestNeighbors

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import tensorflow as tf
from keras import backend as K
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
from keras import initializations
from keras.layers import Input, Dense, merge, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.optimizers
from keras.regularizers import l2


T = TypeVar("T", bound="DeepCyTOFClassifier")

_SRC = 'https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/'


class DeepCyTOFClassifier(BaseEstimator, ClassifierMixin):
    """
    DeepCyTOF: (1) log(1 + |x|) preprocessing, (2) optional denoising autoencoder (DAE, trained unsupervised on all
    training cells), (3) z-scoring w.r.t. the reference sample, (4) feed-forward softmax classifier trained on the
    labelled reference sample, (5) optional per-sample MMD-ResNet calibration of each test sample onto the reference
    (domain adaptation, trained at prediction time without labels), (6) rejection of cells with max softmax
    probability < rejection_threshold (label -1).

    fit(X, y, sample_ids=None):
        - sample_ids=None: X is treated as the (single) hand-gated reference sample.
        - sample_ids given: the reference sample is chosen among the training samples with the original covariance
          criterion; the classifier is trained on the reference sample only, the DAE on all samples (as in the paper).
    predict(X, sample_ids=None):
        - sample_ids=None: X is treated as one sample (one MMD-ResNet is trained for it).
        - sample_ids given: one MMD-ResNet per sample (as in the paper).
    """
    def __init__(
            self,
            # Preprocessing
            preprocess_log: bool = True,
            unlabeled_label: Union[int, None] = None,
            # Denoising autoencoder
            denoise: bool = True,
            dae_keep_prob: float = 0.8,
            dae_num_zeros_ok: int = 1,
            dae_encoding_dim: int = 25,
            dae_l2_penalty: float = 1e-2,
            dae_lr: float = 1e-3,
            dae_n_epochs: int = 80,
            dae_batch_size: int = 128,
            dae_validation_split: float = 0.1,
            dae_patience: int = 25,
            # Feed-forward cell classifier
            hidden_layer_sizes: Tuple[int, int, int] = (12, 6, 3),
            activation: str = 'softplus',
            l2_penalty: float = 1e-4,
            n_epochs: int = 80,
            batch_size: int = 128,
            validation_split: float = 0.1,
            patience: int = 25,
            lr_initial: float = 1e-3,
            lr_drop: float = 0.5,
            lr_epochs_drop: float = 50.0,
            rejection_threshold: float = 0.4,
            # MMD-ResNet calibration
            calibrate: bool = True,
            mmd_layer_sizes: Tuple[int, int] = (25, 25),
            mmd_l2_penalty: float = 1e-2,
            mmd_init_scale: float = .1e-4,
            mmd_subsample_fraction: float = 0.2,
            mmd_n_epochs: int = 500,
            mmd_batch_size: int = 1000,
            mmd_validation_split: float = 0.1,
            mmd_patience: int = 20,
            mmd_lr_initial: float = 1e-5,
            mmd_lr_drop: float = 0.1,
            mmd_lr_epochs_drop: float = 15.0,
            mmd_target_validation_split: float = 0.1,
            mmd_target_sample_size: int = 1000,
            mmd_n_neighbors: int = 20,
            mmd_scales: Union[List[float], None] = None,
            mmd_weights: Union[List[float], None] = None,
            # Runtime (not part of the original method)
            seed: Union[int, None] = None,
            n_threads: Union[int, None] = None,
            verbosity: int = 0,
    ):
        super().__init__()

        self.preprocess_log = preprocess_log
        self.unlabeled_label = unlabeled_label
        self.denoise = denoise
        self.dae_keep_prob = dae_keep_prob
        self.dae_num_zeros_ok = dae_num_zeros_ok
        self.dae_encoding_dim = dae_encoding_dim
        self.dae_l2_penalty = dae_l2_penalty
        self.dae_lr = dae_lr
        self.dae_n_epochs = dae_n_epochs
        self.dae_batch_size = dae_batch_size
        self.dae_validation_split = dae_validation_split
        self.dae_patience = dae_patience
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.l2_penalty = l2_penalty
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.lr_initial = lr_initial
        self.lr_drop = lr_drop
        self.lr_epochs_drop = lr_epochs_drop
        self.rejection_threshold = rejection_threshold
        self.calibrate = calibrate
        self.mmd_layer_sizes = mmd_layer_sizes
        self.mmd_l2_penalty = mmd_l2_penalty
        self.mmd_init_scale = mmd_init_scale
        self.mmd_subsample_fraction = mmd_subsample_fraction
        self.mmd_n_epochs = mmd_n_epochs
        self.mmd_batch_size = mmd_batch_size
        self.mmd_validation_split = mmd_validation_split
        self.mmd_patience = mmd_patience
        self.mmd_lr_initial = mmd_lr_initial
        self.mmd_lr_drop = mmd_lr_drop
        self.mmd_lr_epochs_drop = mmd_lr_epochs_drop
        self.mmd_target_validation_split = mmd_target_validation_split
        self.mmd_target_sample_size = mmd_target_sample_size
        self.mmd_n_neighbors = mmd_n_neighbors
        self.mmd_scales = mmd_scales
        self.mmd_weights = mmd_weights
        self.seed = seed
        self.n_threads = n_threads
        self.verbosity = verbosity

        self.is_fitted_ = False
        self.classes_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.n_features_ = None
        self.reference_sample_id_ = None
        self.preprocessor_ = None  # StandardScaler fitted on the (denoised) reference sample
        self.target_x_ = None  # (Denoised, scaled) reference sample, the target of the MMD-ResNet calibration
        self.target_y_ = None
        self.dae_weights_ = None
        self.classifier_weights_ = None
        # Live Keras models (not pickled, rebuilt from the weights when needed)
        self.dae_ = None
        self.classifier_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_ids: Union[np.ndarray, None] = None,
    ) -> T:

        X, y = check_X_y(X, y)
        X = X.astype(np.float32)

        self._reset_keras_session()
        self._set_seed()

        # Cells with label 'unlabeled_label' are not used for training the classifier (original: label 0)
        labeled_mask = np.ones(y.shape[0], dtype=bool) if self.unlabeled_label is None else y != self.unlabeled_label

        # Rename classes to integers starting from 0
        y_new = np.full(y.shape[0], -1, dtype=int)
        y_new[labeled_mask], self.classes_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_ = \
            DeepCyTOFClassifier._process_class_labels(y=y[labeled_mask])
        y = y_new
        # Add label for unclassifiable cells to dict
        self.new_to_og_classes_dict_[-1] = -1
        self.n_features_ = X.shape[1]

        # ### Pre-processing (log(1 + |x|))
        if self.preprocess_log:
            X = _pre_process_samples_cytof_data(X)

        # ### Choose the reference sample
        if sample_ids is None:
            ref_mask = np.ones(X.shape[0], dtype=bool)
        else:
            sample_ids = np.asarray(sample_ids)
            unique_ids = np.unique(sample_ids)
            ref_sample_ind = _choose_reference_sample([X[sample_ids == sid] for sid in unique_ids])
            self.reference_sample_id_ = unique_ids[ref_sample_ind]
            ref_mask = sample_ids == self.reference_sample_id_
            if self.verbosity >= 1:
                print('Chose reference sample: {}'.format(self.reference_sample_id_))

        # ### Train the denoising autoencoder on all training cells (unsupervised)
        if self.denoise:
            self.dae_ = _train_dae(
                x_all=X,
                num_zeros_ok=self.dae_num_zeros_ok,
                keep_prob=self.dae_keep_prob,
                encoding_dim=self.dae_encoding_dim,
                l2_penalty_ae=self.dae_l2_penalty,
                lr=self.dae_lr,
                n_epochs=self.dae_n_epochs,
                batch_size=self.dae_batch_size,
                validation_split=self.dae_validation_split,
                patience=self.dae_patience,
                verbose=1 if self.verbosity >= 2 else 0,
            )
            self.dae_weights_ = self.dae_.get_weights()

        # ### Denoise + z-score the reference (target) sample
        target_x = _predict_dae(X[ref_mask], self.dae_, self.denoise)
        target_x, self.preprocessor_ = _standard_scale(target_x, preprocessor=None)
        target_y = y[ref_mask]

        # ### Train the feed-forward classifier on the labelled cells of the (denoised) reference sample
        self.classifier_ = _train_classifier(
            x_train=target_x[target_y != -1],
            y_train=target_y[target_y != -1],
            n_classes=self.classes_.shape[0],
            hidden_layers_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            l2_penalty=self.l2_penalty,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            patience=self.patience,
            lr_initial=self.lr_initial,
            lr_drop=self.lr_drop,
            lr_epochs_drop=self.lr_epochs_drop,
            verbose=1 if self.verbosity >= 2 else 0,
        )
        self.classifier_weights_ = self.classifier_.get_weights()

        # Store the reference sample (needed as target for the MMD-ResNet calibration at prediction time)
        self.target_x_ = target_x
        self.target_y_ = target_y

        self.is_fitted_ = True

        return self

    def predict_proba(
            self,
            X: np.ndarray,
            sample_ids: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Softmax output of the DeepCyTOF classifier (after DAE denoising and, if calibrate=True, MMD-ResNet
        calibration). NOTE: the calibration is stochastic and retrained on each call.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X).astype(np.float32)

        self._reset_keras_session()
        self._set_seed()
        self._rebuild_models()

        if self.preprocess_log:
            X = _pre_process_samples_cytof_data(X)

        if sample_ids is None:
            sample_ids = np.zeros(X.shape[0], dtype=int)
        sample_ids = np.asarray(sample_ids)

        y_proba = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float32)

        # Process each sample separately (one MMD-ResNet per sample, as in the original DeepCyTOF.py script)
        for sid in np.unique(sample_ids):
            mask = sample_ids == sid
            # Denoise + z-score w.r.t. the reference sample
            source_x = _predict_dae(X[mask], self.dae_, self.denoise)
            source_x, _ = _standard_scale(source_x, preprocessor=self.preprocessor_)

            # Prediction without calibration
            proba, pred_label = _prediction(source_x, self.classifier_, self.rejection_threshold)

            if self.calibrate:
                if self.verbosity >= 1:
                    print('Calibrating sample {} with MMD-ResNet'.format(sid))
                source_x = _calibrate(
                    target_x=self.target_x_,
                    target_y=self.target_y_,
                    source_x=source_x,
                    pred_label=pred_label,
                    mmd_net_layer_sizes=self.mmd_layer_sizes,
                    l2_penalty=self.mmd_l2_penalty,
                    init_scale=self.mmd_init_scale,
                    subsample_fraction=self.mmd_subsample_fraction,
                    n_epochs=self.mmd_n_epochs,
                    batch_size=self.mmd_batch_size,
                    validation_split=self.mmd_validation_split,
                    patience=self.mmd_patience,
                    lr_initial=self.mmd_lr_initial,
                    lr_drop=self.mmd_lr_drop,
                    lr_epochs_drop=self.mmd_lr_epochs_drop,
                    mmd_target_validation_split=self.mmd_target_validation_split,
                    mmd_target_sample_size=self.mmd_target_sample_size,
                    mmd_n_neighbors=self.mmd_n_neighbors,
                    mmd_scales=self.mmd_scales,
                    mmd_weights=self.mmd_weights,
                    verbose=1 if self.verbosity >= 2 else 0,
                )
                # Prediction with calibration
                proba, _ = _prediction(source_x, self.classifier_, self.rejection_threshold)

            y_proba[mask] = proba

        return y_proba

    def predict(
            self,
            X: np.ndarray,
            sample_ids: Union[np.ndarray, None] = None,
    ) -> np.ndarray:

        y_proba = self.predict_proba(X, sample_ids=sample_ids)

        # Original rejection rule (see _prediction()): argmax, reject if max probability < rejection_threshold
        y_pred = np.argmax(y_proba, axis=1)
        y_pred[np.max(y_proba, axis=1) < self.rejection_threshold] = -1

        # Get vector with the original labels
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        if np.any(y_pred == -1) and self.verbosity >= 1:
            warnings.warn(
                "{} events were classified with a max. probability < {}. "
                "They are classified as label -1 ~= 'unassigned'".format(
                    int(np.sum(y_pred == -1)), self.rejection_threshold),
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
        # by setting their sample weight to 0
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
            sample_weight[y_pred == -1] = 0

        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def save(
            self,
            filename: str = 'deepcytof_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        if filepath is None:
            filepath = os.getcwd()
        # Keras models are dropped in __getstate__, only their weights (numpy arrays) are pickled
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(
            cls,
            filename: str = 'deepcytof_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> T:
        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            return pickle.load(f)

    def __getstate__(self):
        # Keras 1.x / TF 1.x models are not picklable => drop them, they are rebuilt from the stored weights
        state = self.__dict__.copy()
        state['dae_'] = None
        state['classifier_'] = None
        return state

    def _rebuild_models(self):
        # (Re)build the Keras models in the current graph/session from the stored weights
        if self.denoise:
            self.dae_ = _build_dae(
                input_dim=self.n_features_, encoding_dim=self.dae_encoding_dim, l2_penalty_ae=self.dae_l2_penalty
            )
            self.dae_.set_weights(self.dae_weights_)
        self.classifier_ = _build_classifier(
            input_dim=self.n_features_,
            n_classes=self.classes_.shape[0],
            hidden_layers_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            l2_penalty=self.l2_penalty,
        )
        self.classifier_.set_weights(self.classifier_weights_)

    def _reset_keras_session(self):
        # Clear the TF graph (avoids accumulating one MMD-ResNet graph per predicted sample) and limit threads
        K.clear_session()
        if self.n_threads is not None:
            config = tf.ConfigProto(
                intra_op_parallelism_threads=self.n_threads,
                inter_op_parallelism_threads=min(2, self.n_threads),
            )
            K.set_session(tf.Session(config=config))

    def _set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

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
# ### DeepCyTOF building blocks (copied from the official repository, see source comments)
# ######################################################################################################################

# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py
# lines 25-27 (preProcessSamplesCyTOFData)
# ### Change: operates on a numpy array instead of a Sample object
def _pre_process_samples_cytof_data(x):
    x = np.log(1 + np.abs(x))
    return x


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py
# lines 29-34 (standard_scale)
# ### Change: operates on a numpy array instead of a Sample object, 'is None' instead of '== None'
def _standard_scale(x, preprocessor=None):
    if preprocessor is None:
        preprocessor = prep.StandardScaler().fit(x)
    x = preprocessor.transform(x)

    return x, preprocessor


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py
# lines 59-77 (chooseReferenceSample)
# ### Change: takes a list of (already log-preprocessed) numpy arrays instead of loading the samples from disk
#     (original lines 60-64), loop body unchanged
def _choose_reference_sample(samples):
    numSamples = len(samples)
    norms = np.zeros(shape=[numSamples, numSamples])
    for i in range(numSamples):
        cov_i = np.cov(samples[i], rowvar=False)
        for j in range(numSamples):
            cov_j = np.cov(samples[j], rowvar=False)
            covDiff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(covDiff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            refSampleInd = np.argmin(avg)

    return refSampleInd


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/denoisingAutoEncoder.py
# lines 57-65 (DAE architecture inside trainDAE)
# ### Change: factored out into a function, hard coded ae_encodingDim (25) / l2_penalty_ae (1e-2) (lines 42-43)
#     are exposed as arguments
def _build_dae(input_dim, encoding_dim=25, l2_penalty_ae=1e-2):
    input_cell = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu',
                    W_regularizer=l2(l2_penalty_ae))(input_cell)
    encoded1 = Dense(encoding_dim, activation='relu',
                     W_regularizer=l2(l2_penalty_ae))(encoded)
    decoded = Dense(input_dim, activation='linear',
                    W_regularizer=l2(l2_penalty_ae))(encoded1)

    autoencoder = Model(input=input_cell, output=decoded)
    return autoencoder


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/denoisingAutoEncoder.py
# lines 20-77 (trainDAE), relevant parts: lines 27-28 + 37-38 (zero filter), 52-55 (masking noise), 66-71 (fit)
# ### Changes:
#     - The original trains the DAE on the (log-preprocessed) non-reference training samples (loaded from disk,
#       lines 22-35) + the reference sample. Here all cells passed to fit() are used (= same set if sample_ids
#       are passed to fit()), the zero filter is applied once to all of them.
#     - Hard coded values exposed as arguments: numZerosOK=1, nb_epoch=80, batch_size=128, validation_split=0.1,
#       patience=25, optimizer 'rmsprop' (Keras default lr=0.001 -> argument 'lr')
#     - Removed: model loading/saving to disk, the plotting callback mn.monitor() (only plots the loss curves)
def _train_dae(x_all, num_zeros_ok=1, keep_prob=0.8, encoding_dim=25, l2_penalty_ae=1e-2, lr=1e-3, n_epochs=80,
               batch_size=128, validation_split=0.1, patience=25, verbose=0):
    numZerosOK = num_zeros_ok
    toKeepT = np.sum((x_all == 0), axis=1) <= numZerosOK

    inputDim = x_all.shape[1]

    # train de-noising auto encoder and save it.
    trainTarget_ae = x_all[toKeepT]
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keep_prob,
                                                       size=trainTarget_ae.shape)

    autoencoder = _build_dae(input_dim=inputDim, encoding_dim=encoding_dim, l2_penalty_ae=l2_penalty_ae)
    autoencoder.compile(optimizer=keras.optimizers.rmsprop(lr=lr), loss='mse')
    autoencoder.fit(trainData_ae, trainTarget_ae, nb_epoch=n_epochs,
                    batch_size=batch_size, shuffle=True,
                    validation_split=validation_split, verbose=verbose,
                    callbacks=[cb.EarlyStopping(
                        monitor='val_loss', patience=patience, mode='auto')])

    return autoencoder


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/denoisingAutoEncoder.py
# lines 79-86 (predictDAE)
# ### Change: operates on a numpy array instead of a Sample object
def _predict_dae(x, autoencoder, denoise=False):
    if denoise:
        # apply de-noising auto encoder to target.
        denoised_x = autoencoder.predict(x)
    else:
        denoised_x = x

    return denoised_x


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py
# lines 30-38 (step_decay)
# ### Change: wrapped in a factory so that initial_lrate, drop, epochs_drop (hard coded 1e-3, 0.5, 50) are arguments
def _make_step_decay(initial_lrate=1e-3, drop=0.5, epochs_drop=50.0):
    def step_decay(epoch):
        '''
        Learning rate schedule.
        '''
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    return step_decay


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py
# lines 95-106 (network architecture inside trainClassifier)
# ### Change: factored out into a function, numClasses passed as argument (original line 103 counts unique labels)
def _build_classifier(input_dim, n_classes, hidden_layers_sizes=(12, 6, 3), activation='softplus', l2_penalty=1e-4):
    # Construct a feed-forward neural network.
    inputLayer = Input(shape=(input_dim,))
    hidden1 = Dense(hidden_layers_sizes[0], activation=activation,
                    W_regularizer=l2(l2_penalty))(inputLayer)
    hidden2 = Dense(hidden_layers_sizes[1], activation=activation,
                    W_regularizer=l2(l2_penalty))(hidden1)
    hidden3 = Dense(hidden_layers_sizes[2], activation=activation,
                    W_regularizer=l2(l2_penalty))(hidden2)
    outputLayer = Dense(n_classes, activation='softmax')(hidden3)

    net = Model(input=inputLayer, output=outputLayer)
    return net


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py
# lines 75-125 (trainClassifier), relevant parts: lines 92-93 (label expansion), 108-117 (compile + fit)
# ### Changes:
#     - Removal of unlabeled cells (label 0) and label shift (lines 79-84) done by the wrapper (labels 0..K-1),
#       GvHD special case (lines 87-90) removed
#     - Hard coded values exposed as arguments: nb_epoch=80, batch_size=128, validation_split=0.1, patience=25,
#       learning rate schedule (see _make_step_decay)
#     - Removed: saving to disk, the plotting callback mn.monitor() (only plots the loss curves)
def _train_classifier(x_train, y_train, n_classes, hidden_layers_sizes=(12, 6, 3), activation='softplus',
                      l2_penalty=1e-4, n_epochs=80, batch_size=128, validation_split=0.1, patience=25,
                      lr_initial=1e-3, lr_drop=0.5, lr_epochs_drop=50.0, verbose=0):
    # Expand labels, to work with sparse categorical cross entropy.
    y_train = np.expand_dims(y_train, -1)

    net = _build_classifier(
        input_dim=x_train.shape[1], n_classes=n_classes, hidden_layers_sizes=hidden_layers_sizes,
        activation=activation, l2_penalty=l2_penalty
    )
    lrate = LearningRateScheduler(_make_step_decay(lr_initial, lr_drop, lr_epochs_drop))
    optimizer = keras.optimizers.rmsprop(lr=0.0)

    net.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy')
    net.fit(x_train, y_train, nb_epoch=n_epochs, batch_size=batch_size, shuffle=True,
            validation_split=validation_split, verbose=verbose,
            callbacks=[lrate,
                       cb.EarlyStopping(monitor='val_loss',
                                        patience=patience, mode='auto')])

    return net


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py
# lines 139-141 (inside prediction)
# ### Changes: labels 0..K-1 with -1 = rejected (original: 1..K with 0 = rejected), threshold .4 exposed as argument,
#     probabilities are returned as well, accuracy/F1 computation on test labels (lines 129-159) removed
def _prediction(x, net, rejection_threshold=.4):
    y_test_pred_prob = net.predict(x, verbose=0)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    y_test_pred[np.max(y_test_pred_prob, axis=1) < rejection_threshold] = -1
    return y_test_pred_prob, y_test_pred


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/MMDNet.py
# lines 27-32 (step_decay)
# ### Change: wrapped in a factory so that initial_lrate, drop, epochs_drop (hard coded 1e-5, 0.1, 15) are arguments
def _make_mmd_step_decay(initial_lrate=1e-5, drop=0.1, epochs_drop=15.0):
    def step_decay(epoch):
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    return step_decay


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/MMDNet.py
# lines 34-74 (constructMMD; identical to the architecture in calibrate(), lines 78-113)
# ### Change: hard coded mmdNetLayerSizes ([25, 25]), l2_penalty (1e-2), init scale (.1e-4) exposed as arguments,
#     takes the input dimension instead of the target Sample
def _construct_mmd(space_dim, mmd_net_layer_sizes=(25, 25), l2_penalty=1e-2, init_scale=.1e-4):

    mmdNetLayerSizes = mmd_net_layer_sizes
    init = lambda shape, name: initializations.normal(shape,
                                                      scale=init_scale, name=name)

    calibInput = Input(shape=(space_dim,))
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      W_regularizer=l2(l2_penalty), init=init)(block1_a1)
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init=init)(block1_a2)
    block1_output = merge([block1_w2, calibInput], mode='sum')
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init=init)(block2_a1)
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init=init)(block2_a2)
    block2_output = merge([block2_w2, block1_output], mode='sum')
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init=init)(block3_a1)
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init=init)(block3_a2)
    block3_output = merge([block3_w2, block2_output], mode='sum')

    calibMMDNet = Model(input=calibInput, output=block3_output)

    return calibMMDNet, block3_output


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/MMDNet.py
# lines 76-156 (calibrate), relevant parts: lines 115-153
# ### Changes:
#     - The architecture (lines 78-113) is built via _construct_mmd() (identical code, MMDNet.py lines 34-74)
#     - Operates on numpy arrays instead of Sample objects; rejected cells are -1 (original: 0)
#     - Hard coded values exposed as arguments: subsample fraction .2, nb_epoch=500, batch_size=1000,
#       validation_split=0.1, patience=20, MMD settings (MMDTargetValidation_split=0.1 + MMD() defaults),
#       learning rate schedule (see _make_mmd_step_decay)
#     - Removed: saving the weights to disk, the plotting callback mn.monitorMMD() (only plots a PCA embedding)
def _calibrate(target_x, target_y, source_x, pred_label, mmd_net_layer_sizes=(25, 25), l2_penalty=1e-2,
               init_scale=.1e-4, subsample_fraction=.2, n_epochs=500, batch_size=1000, validation_split=0.1,
               patience=20, lr_initial=1e-5, lr_drop=0.1, lr_epochs_drop=15.0, mmd_target_validation_split=0.1,
               mmd_target_sample_size=1000, mmd_n_neighbors=20, mmd_scales=None, mmd_weights=None, verbose=0):

    calibMMDNet, block3_output = _construct_mmd(
        space_dim=target_x.shape[1], mmd_net_layer_sizes=mmd_net_layer_sizes, l2_penalty=l2_penalty,
        init_scale=init_scale
    )

    n = target_x.shape[0]
    p = np.random.permutation(n)
    toTake = p[range(int(subsample_fraction * n))]
    targetXMMD = target_x[toTake]
    targetYMMD = target_y[toTake]

    targetXMMD = targetXMMD[targetYMMD != -1]
    targetYMMD = targetYMMD[targetYMMD != -1]

    targetYMMD = np.reshape(targetYMMD, (-1, 1))

    n = source_x.shape[0]
    p = np.random.permutation(n)
    toTake = p[range(int(subsample_fraction * n))]
    sourceXMMD = source_x[toTake]
    sourceYMMD = pred_label[toTake]

    sourceXMMD = sourceXMMD[sourceYMMD != -1]
    sourceYMMD = sourceYMMD[sourceYMMD != -1]

    sourceYMMD = np.reshape(sourceYMMD, (-1, 1))

    lrate = LearningRateScheduler(_make_mmd_step_decay(lr_initial, lr_drop, lr_epochs_drop))
    optimizer = keras.optimizers.rmsprop(lr=0.0)
    calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true, y_pred:
                        MMD(block3_output, targetXMMD,
                            MMDTargetValidation_split=mmd_target_validation_split,
                            MMDTargetSampleSize=mmd_target_sample_size,
                            n_neighbors=mmd_n_neighbors,
                            scales=mmd_scales,
                            weights=mmd_weights).KerasCost(y_true, y_pred))

    sourceLabels = np.zeros(sourceXMMD.shape[0])

    calibMMDNet.fit(sourceXMMD, sourceLabels, nb_epoch=n_epochs,
                    batch_size=batch_size, validation_split=validation_split, verbose=verbose,
                    callbacks=[lrate,
                               cb.EarlyStopping(monitor='val_loss', patience=patience, mode='auto')])
    calibrated_source_x = calibMMDNet.predict(source_x)
    calibMMDNet = None
    return calibrated_source_x


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/CostFunctions.py
# lines 11-19 (IntType, FloatType, squaredDistance), unchanged
IntType = 'int32'
FloatType = 'float32'


# calculate the squared distance between x and y
def squaredDistance(X, Y):
    # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
    # broadcasted subtraction, a square, and a sum.
    r = K.expand_dims(X, dim=1)
    return K.sum(K.square(r - Y), axis=-1)


# Copied from: https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/CostFunctions.py
# lines 21-109 (class MMD)
# ### Change: 'is None' instead of '== None' (lines 39, 52) to
#     allow numpy arrays as scales/weights; otherwise unchanged
class MMD:
    MMDTargetTrain = None
    MMDTargetTrainSize = None
    MMDTargetValidation = None
    MMDTargetValidationSize = None
    MMDTargetSampleSize = None
    kernel = None
    scales = None
    weights = None

    def __init__(self,
                 MMDLayer,
                 MMDTargetTrain,
                 MMDTargetValidation_split=0.1,
                 MMDTargetSampleSize=1000,
                 n_neighbors=20,
                 scales=None,
                 weights=None):
        if scales is None:
            print("setting scales using KNN")
            med = np.zeros(20)
            for ii in range(1, 20):
                sample = MMDTargetTrain[np.random.randint(MMDTargetTrain.shape[0], size=MMDTargetSampleSize), :]
                nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
                distances, dummy = nbrs.kneighbors(sample)
                # nearest neighbor is the point so we need to exclude it
                med[ii] = np.median(distances[:, 1:n_neighbors])
            med = np.median(med)
            scales = [med / 2, med, med * 2]
            print(scales)
        scales = K.variable(value=np.asarray(scales))
        if weights is None:
            print("setting all scale weights to 1")
            weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        self.MMDLayer = MMDLayer
        MMDTargetTrain, MMDTargetValidation = train_test_split(MMDTargetTrain, test_size=MMDTargetValidation_split,
                                                               random_state=42)
        self.MMDTargetTrain = K.variable(value=MMDTargetTrain)
        self.MMDTargetTrainSize = K.eval(K.shape(self.MMDTargetTrain)[0])
        self.MMDTargetValidation = K.variable(value=MMDTargetValidation)
        self.MMDTargetValidationSize = K.eval(K.shape(self.MMDTargetValidation)[0])
        self.MMDTargetSampleSize = MMDTargetSampleSize
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights

    # calculate the raphy kernel applied to all entries in a pairwise distance matrix
    def RaphyKernel(self, X, Y):
        # expand dist to a 1xnxm tensor where the 1 is broadcastable
        sQdist = K.expand_dims(squaredDistance(X, Y), 0)
        # expand scales into a px1x1 tensor so we can do an element wise exponential
        self.scales = K.expand_dims(K.expand_dims(self.scales, -1), -1)
        # expand scales into a px1x1 tensor so we can do an element wise exponential
        self.weights = K.expand_dims(K.expand_dims(self.weights, -1), -1)
        # calculated the kernal for each scale weight on the distance matrix and sum them up
        return K.sum(self.weights * K.exp(-sQdist / (K.pow(self.scales, 2))), 0)

    # Calculate the MMD cost
    def cost(self, source, target):
        # calculate the 3 MMD terms
        xx = self.kernel(source, source)
        xy = self.kernel(source, target)
        yy = self.kernel(target, target)
        # calculate the bias MMD estimater (cannot be less than 0)
        MMD = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
        # return the square root of the MMD because it optimizes better
        return K.sqrt(MMD)

    def KerasCost(self, y_true, y_pred):
        # create a random subsampling of the target instances for the test set
        # This is rarely going to hit the last entry
        sample = K.cast(K.round(K.random_uniform(shape=tuple([self.MMDTargetSampleSize]), low=0,
                                                 high=self.MMDTargetTrainSize - 1)), IntType)
        # this is a subset operation (not a very pretty way to do it)
        MMDTargetSampleTrain = K.gather(self.MMDTargetTrain, sample)
        # do the same for the validation set
        sample = K.cast(K.round(K.random_uniform(shape=tuple([self.MMDTargetSampleSize]), low=0,
                                                 high=self.MMDTargetValidationSize - 1)), IntType)
        # and the subset operation
        MMDTargetSampleValidation = K.gather(self.MMDTargetValidation, sample)
        # create the sample based on whether we are in training or validation steps
        MMDtargetSample = K.in_train_phase(MMDTargetSampleTrain, MMDTargetSampleValidation)
        # return the MMD cost for this subset
        ret = self.cost(self.MMDLayer, MMDtargetSample)
        # pretty dumb but y_treu has to be in the cost for keras to not barf when cleaning up
        ret = ret + 0 * K.sum(y_pred) + 0 * K.sum(y_true)
        return ret
