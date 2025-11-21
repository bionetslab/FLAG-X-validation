
import os
import warnings
import pickle
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from itertools import accumulate
from sklearn.exceptions import NotFittedError

from typing import List, Tuple, Dict, Union, Any
from typing_extensions import Literal

from .io import FlowDataManager, export_to_fcs
from .gating import SomClassifier, MLPClassifier
from .dimred import PCA, UMAP, TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding


class GatingPipeline:
    """
    End-to-end flow cytometry gating pipeline supporting preprocessing, downsampling, dimensionality reduction, and supervised or unsupervised gating.

    This class orchestrates the full workflow:

    1. **Load raw FCS/CSV files**
    2. **Align channel names**
    3. **(Optional) Relabel training data**
    4. **(Optional) Preprocess data sample-wise**
    5. **(Optional) Downsample data**
    6. **Train the gating module**
       - supervised: MLP classifier
       - supervised or unsupervised: SOM classifier
    7. **Inference on new samples**
    8. **Optional dimensionality reduction (UMAP, SOM, PCA, t-SNE, etc.)**
    9. **Export annotated FCS files**

    Attributes:
        train_data_file_path (str or None): Path to directory containing training data. Defaults to CWD.
        train_data_file_names (list[str] or None): Specific training filenames to load. If None, uses all files in directory.
        train_data_file_type (Literal['fcs','csv'] or None): Input file type. If None, inferred from first filename.
        save_path (str or None): Output directory for pipeline metadata and results. If None, defaults to CWD.
        channels (list[int] or list[str] or None): Indices or names of channels to train on.
        label_key (int, str, or None): Key to labels in `.X`, `.obs`, or `.layers`. If None, only unsupervised SOM training is available.
        channel_names_alignment_kwargs (dict or None): Arguments forwarded to channel alignment.
        relabel_data_kwargs (dict or None): Mapping for relabeling training data.
        preprocessing_kwargs (dict or None): Sample-wise preprocessing configuration.
        downsampling_kwargs (dict or None): Sample-wise downsampling configuration.
        gating_method (Literal['som','mlp']): Which model to train.
        gating_method_kwargs (dict or None): Additional arguments for SOM or MLP.
        prediction_threshold (float or None): Binary/abstention threshold; defaults chosen automatically.
        verbosity (int): Logging level.
        is_trained_ (bool): Whether the pipeline has been successfully trained.
        gating_module_ (SomClassifier or MLPClassifier or None): The fitted gating model.
        binary_classes_ (bool or None): Whether the task is binary classification.
    """
    def __init__(
            self,
            train_data_file_path: Union[str, None] = None,   # default: cwd
            train_data_file_names: Union[List[str], None] = None,  # default: listdir(path)
            train_data_file_type: Union[Literal['fcs', 'csv'], None] = None,

            save_path: Union[str, None] = None,

            channels: Union[List[int], List[str], None] = None,
            label_key: Union[int, str, None] = None,
            # .obs key or varname or var index, if none is passed -> unsupervised

            channel_names_alignment_kwargs: Union[Dict[str, Any], None] = None,  # {'reference_channel_names': int | dict | None}

            relabel_data_kwargs: Union[Dict[str, Any], None] = None,  # {'old_to_new_label_mapping': dict, !optional! 'new_label_key': str}

            preprocessing_kwargs: Union[Dict[str, Any], None] = None,
            # {'flavour': str, !optional! 'flavour_kwargs': dict, !optional! 'save_raw_to_layer': str}
            # if flavour == 'custom' then 'flavour_kwargs' must contain 'preprocessing_method'

            downsampling_kwargs: Union[Dict[str, Any], None] = None,

            gating_method: Literal['som', 'mlp'] = 'som',
            gating_method_kwargs: Union[Dict[str, Any], None] = None,
            prediction_threshold: Union[float, None] = None,

            verbosity: int = 1,
    ):
        """
        Initialize the full gating pipeline and configure optional steps.

        Args:
            train_data_file_path (str or None): Path to directory containing training data. Defaults to CWD.
            train_data_file_names (list[str] or None): Specific training filenames to load. If None, uses all files in directory.
            train_data_file_type (Literal['fcs','csv'] or None): Input file type. If None, inferred from first filename.
            save_path (str or None): Output directory for pipeline metadata and results. If None, defaults to CWD.
            channels (list[int] or list[str] or None): Indices or names of channels to train on.
            label_key (int, str, or None): Key to labels in `.X`, `.obs`, or `.layers`. If None, only unsupervised SOM training is available.
            channel_names_alignment_kwargs (dict or None): Arguments forwarded to channel alignment.
            relabel_data_kwargs (dict or None): Mapping for relabeling training data.
            preprocessing_kwargs (dict or None): Sample-wise preprocessing configuration.
            downsampling_kwargs (dict or None): Sample-wise downsampling configuration.
            gating_method (Literal['som','mlp']): Which model to train.
            gating_method_kwargs (dict or None): Additional arguments for SOM or MLP.
            prediction_threshold (float or None): Binary/abstention threshold; defaults chosen automatically.
            verbosity (int): Logging level.

        Returns:
            None
        """

        super().__init__()

        # Path to/ filenames of/ filetype of training data
        self.train_data_file_path = train_data_file_path
        self.train_data_file_names = train_data_file_names
        self.train_data_file_type = train_data_file_type

        # Initialize the save paths
        self.save_path = save_path
        self.train_data_manager_save_path = None
        self._init_save_paths()

        # Keyword arguments for aligning the channel names across samples
        self.channel_names_alignment_kwargs = channel_names_alignment_kwargs

        # Keyword arguments for preprocessing
        self.preprocessing_kwargs = preprocessing_kwargs

        self.downsampling_kwargs = downsampling_kwargs

        # Channels to train on and channel with labels
        self.channels = channels
        self.label_key = label_key

        # Keyword arguments for relabeling the training data
        self.relabel_data_kwargs = relabel_data_kwargs

        # Gating method
        self.gating_method = gating_method
        self.gating_method_kwargs = gating_method_kwargs
        self.prediction_threshold = prediction_threshold

        self.verbosity = verbosity

        self.is_trained_ = False
        self.gating_module_ = None
        self.binary_classes_ = None


    def _init_save_paths(self):

        if self.save_path is None:
            self.save_path = os.getcwd()

        os.makedirs(self.save_path, exist_ok=True)

        self.train_data_manager_save_path = os.path.join(self.save_path, 'train_data_manager_output')


    def train(self):
        """
        Train the full gating pipeline.

        This executes the full training workflow:
        - Load raw data
        - (Optional) Align channel names
        - (Optional) Relabel and preprocess
        - (Optional) Downsample
        - Construct training matrix from all samples
        - Train SOM or MLP gating module

        The gating module is stored in `self.gating_module_`.

        Raises:
            ValueError: If MLP is selected but `label_key` is None.
            ValueError: If binary labels are not exactly {0, 1}.
        """

        # Get the train data from the raw data
        train_fdm, x_train, y_train = self._data_pipeline(
            data_file_path=self.train_data_file_path,
            data_file_names=self.train_data_file_names,
            data_file_type=self.train_data_file_type,
            label_key=self.label_key,
            downsampling_kwargs=self.downsampling_kwargs,
            data_manager_save_path=self.train_data_manager_save_path,
            save_meta_info=True,
            fn_prefix_saving='train_'
        )

        # If file type was inferred from the filenames set it here
        if self.train_data_file_type is None:
            self.train_data_file_type = train_fdm._data_file_type

        unique_classes = np.unique(y_train)
        if unique_classes.shape[0] == 2:
            self.binary_classes_ = True

            if set(unique_classes) != {0,1}:
                raise ValueError(
                    f"Binary classification requires class labels to be 0 (negative class) or 1 (positive class). "
                    f"Found: {set(unique_classes)}"
                )

        else:
            self.binary_classes_ = False

        if self.prediction_threshold is None:
            if self.binary_classes_:
                self.prediction_threshold = 0.5
                if self.verbosity >= 2:
                    print('# ### Setting default prediction threshold to 0.5 for binary classification.')
            else:
                self.prediction_threshold = 0.0
                if self.verbosity >= 2:
                    print((
                        '# ### Setting default prediction threshold to 0.0 for multiclass classification. '
                        'The maximum over class probabilities will be used.'
                    ))

        # Instantiate the gating module of the pipeline
        if self.gating_method_kwargs is None:
            self.gating_method_kwargs = {}

        if self.gating_method == 'som':
            self.gating_module_ = SomClassifier(**self.gating_method_kwargs)
        elif self.gating_method == 'mlp':
            if self.label_key is None:
                raise ValueError(
                    "'label_key' is required when gating_method is 'mlp'. "
                    "Unsupervised training is not possible for a MLP."
                )
            self.gating_module_ = MLPClassifier(**self.gating_method_kwargs)
        else:
            raise NotImplementedError(
                f"Gating method '{self.gating_method}' is not implemented. "
                "Supported methods are: 'som', 'mlp'."
            )

        # Call the fit method of the gating module
        self.gating_module_.fit(X=x_train, y=y_train)

        self.is_trained_ = True


    def inference(
            self,
            data_file_path: Union[str, None] = None,  # default: cwd
            data_file_names: Union[List[str], None] = None,  # default: listdir(path)
            sample_wise: bool = False,
            gate: bool = True,
            dim_red_methods: Union[
                Tuple[
                    Literal[
                        'som', 'pca', 'umap', 'tsne', 'isomap', 'locallylinearembedding', 'mds', 'spectralembedding'
                    ],
                    ...
                ],
                None
            ] = ('umap', ),
            dim_red_method_kwargs: Union[Tuple[Union[Dict[str, Any], None], ...], None] = None,
            save_path: Union[str, None] = None,
            save_filename: Union[str, None] = None,
            scale_channels: Union[List[str], None] = None,
            val_range: Tuple[float, float] = (0.0, 2 ** 20),
            keep_unscaled: bool = False,
    ):
        """
        Apply the trained pipeline to new data for gating and/or dimensionality reduction.

        This performs:
        - Data loading + preprocessing (same as during training)
        - (Optional) Prediction using the trained model
        - (Optional) Dimensionality reduction using one or more methods
        - Export to FCS file(s) with annotations added in new channels

        Args:
            data_file_path (str or None): Directory containing inference data.
            data_file_names (list[str] or None): Specific inference filenames.
            sample_wise (bool): If True, run dimension reduction and export separately per sample.
            gate (bool): Whether to apply the trained gating model.
            dim_red_methods (tuple[str] or None): Dimensionality reduction methods to apply.
            dim_red_method_kwargs (tuple[dict] or None): One kwargs dict per method.
            save_path (str or None): Output directory for FCS export.
            save_filename (str or None): Base filename for exported FCS.
            scale_channels (list[str] or None): Additional channels to scale for FCS export (e.g., previously added integer labels).
            val_range (tuple[float,float]): Value range for scaling when writing FCS. (This is done for proper display of the added annotations in standard analysis software.)
            keep_unscaled (bool): Whether to also retain unscaled values in separate channels.

        Returns:
            None

        Raises:
            NotFittedError: If gating was requested but the model is not trained.
            ValueError: If dimensionality reduction kwargs do not match number of methods.
        """

        # Load and process the data
        fdm, _, _ = self._data_pipeline(
            data_file_path=data_file_path,
            data_file_names=data_file_names,
            data_file_type=self.train_data_file_type,
            label_key=None,  # No labels here
            data_manager_save_path=None,
            fn_prefix_saving=None,
            save_meta_info=False,
        )

        # Extract the processed data matrices (just channels used for training)
        xs = []
        for i, adata in enumerate(fdm.anndata_list_):
            dl = fdm.get_data_loader_worker(
                data_list=[adata, ],
                channels=self.channels,
                layer_key=None,
                label_key=None,
                label_layer_key=None,
                batch_size=-1,
                shuffle=False,
                return_data_loader='np_array',
                on_disk=False,
                filename_np=None,
                # **kwargs  # No data loader kwargs needed here
            )
            x = next(iter(dl))
            xs.append(x)

        if gate:
            if not self.is_trained_:
                raise NotFittedError("This pipeline instance is not trained yet. Call 'train' before gating.")

            if self.label_key is not None:  # Trained with labels -> gate
                y_preds = self._gating_helper(xs=xs)

            else:
                warnings.warn(
                    "Gating was requested (gate=True), but training was unsupervised (just SOM module). "
                    "Skipping gating step.",
                    UserWarning
                )
                y_preds = None
        else:
            y_preds = None

        if dim_red_methods is not None:

            if dim_red_method_kwargs is None:
                dim_red_method_kwargs = ({},) * len(dim_red_methods)
            else:
                if len(dim_red_methods) != len(dim_red_method_kwargs):
                    raise ValueError(
                        "Mismatch: 'dim_red_methods' and 'dim_red_method_kwargs' must have the same length."
                    )

            x_dimreds = []
            for dim_red_method, kwargs_dict in zip(dim_red_methods, dim_red_method_kwargs):

                x_dimred = self._reduce_dimension_helper(
                    xs=xs,
                    sample_wise=sample_wise,
                    dim_red_method=dim_red_method,
                    dim_red_method_kwargs=kwargs_dict,
                )
                x_dimreds.append(x_dimred)

        else:
            x_dimreds = None

        add_columns = []
        add_columns_names = []

        if y_preds is not None:
            add_columns.append(y_preds)
            add_columns_names.append(f'pred_{self.gating_method}')

        if x_dimreds is not None:
            for x_dimred, dimred_name in zip(x_dimreds, dim_red_methods):
                add_columns.append([x[:, 0] for x in x_dimred])
                add_columns.append([x[:, 1] for x in x_dimred])
                add_columns_names.append(dimred_name + '_1')
                add_columns_names.append(dimred_name + '_2')

        # Define columns to be scaled
        scale_columns = add_columns_names.copy()

        # Add additional columns to be scaled as well if present
        if scale_channels is not None:
            for channel in scale_channels:
                exists_channel = all(channel in adata.var_names for adata in fdm.anndata_list_)
                if exists_channel:
                    scale_columns.append(channel)

        if save_path is None:
            save_path = os.getcwd()

        if save_filename is None:
            save_filename = 'annotated_data.fcs'

        if sample_wise:
            save_filename = [f'{save_filename[:-4]}_sample_id_{i + 1}.fcs' for i in range(len(fdm.anndata_list_))]

        export_to_fcs(
            data_list=fdm.anndata_list_,
            layer_key='raw' if self.preprocessing_kwargs is not None else None,
            sample_wise=sample_wise,
            add_columns=add_columns,
            add_columns_names=add_columns_names,
            scale_columns=scale_columns,
            val_range=val_range,
            keep_unscaled=keep_unscaled,
            save_path=save_path,
            save_filenames=save_filename,
        )

    def _data_pipeline(
            self,
            data_file_path: Union[str, None] = None,  # default: cwd
            data_file_names: Union[List[str], None] = None,  # default: listdir(path)
            data_file_type: Union[Literal['fcs', 'csv'], None] = None,
            label_key: Union[int, str, None] = None,  # If None unlabeled case
            downsampling_kwargs: Union[Dict, None] = None,
            data_manager_save_path: Union[str, None] = None,  # default: cwd
            save_meta_info: bool = False,
            fn_prefix_saving: Union[str, None] = None,
    ) -> Tuple[FlowDataManager, np.ndarray, np.ndarray]:

        if fn_prefix_saving is None:
            fn_prefix_saving = ''

        # If no filenames are passed, get all filenames from data dir
        if data_file_names is None:
            data_file_names = os.listdir(data_file_path)

        if not save_meta_info:
            data_manager_save_path = None

        # For reproducibility and consistency
        data_file_names = sorted(data_file_names)

        # Instantiate the train data manager
        fdm = FlowDataManager(
            data_file_names=data_file_names,
            data_file_type=data_file_type,
            data_file_path=data_file_path,
            save_path=data_manager_save_path,
            verbosity=self.verbosity,
        )

        # Load train data files to anndata
        fdm.load_data_files_to_anndata()

        # Downsample first
        if downsampling_kwargs is not None:
            if label_key is not None:
                downsampling_kwargs['label_key'] = label_key
            fdm.sample_wise_downsampling(data_set='all', **downsampling_kwargs)

        # Check the number of events per sample
        if save_meta_info:
            fdm.check_sample_sizes(filename_sample_sizes_df=f'{fn_prefix_saving}sample_sizes.csv')
            fdm.plot_sample_size_df(sample_size_df=fdm.sample_sizes_, dpi=300)
            plt.tight_layout()
            plt.savefig(os.path.join(fdm.save_path, f'{fn_prefix_saving}sample_sizes.png'))
            plt.close('all')

        # Align channel names
        if self.channel_names_alignment_kwargs is not None:
            reference_channel_names = self.channel_names_alignment_kwargs.get('reference_channel_names', None)
            fdm.align_channel_names(
                reference_channel_names=reference_channel_names,  # None -> use 1st entry of train data list as reference
                filename_log_df=f'{fn_prefix_saving}og_channel_names.csv'  if save_meta_info else None,  # Save only during training
            )

        # Relabel data if relabel_data_kwargs is not None
        if self.relabel_data_kwargs is not None and label_key is not None:

            old_to_new_label_mapping = self.relabel_data_kwargs['old_to_new_label_mapping']
            new_label_key = self.relabel_data_kwargs['new_label_key']

            fdm.relabel_data(
                data_set='all',
                old_to_new_label_mapping=old_to_new_label_mapping,
                label_key=label_key,
                label_layer_key=None,  # No preprocessing done yet
                new_label_key=new_label_key,
            )

            # Update the label key
            label_key = new_label_key

        # Check the class balance
        if save_meta_info and label_key is not None:
            cb_df = fdm.check_class_balance(
                data_set='all',
                label_key=label_key,
                label_layer_key=None,
                filename_class_balance_df=f'{fn_prefix_saving}class_balance.csv',
            )
            fdm.plot_class_balance_df(class_balance_df=cb_df, dpi=300)
            plt.savefig(os.path.join(fdm.save_path, f'{fn_prefix_saving}class_balance.png'))
            plt.close('all')

        # Apply sample wise preprocessing transformation, if preprocessing_kwargs is not None
        if self.preprocessing_kwargs is not None:
            flavour = self.preprocessing_kwargs['flavour']
            flavour_kwargs = self.preprocessing_kwargs.get('flavour_kwargs', {})

            fdm.sample_wise_preprocessing(
                flavour=flavour,
                save_raw_to_layer='raw',  # Save raw data to layer 'raw'
                **flavour_kwargs,
            )

            label_layer_key = 'raw'

        else:

            label_layer_key = None

        # Create dataloader with batch size = all events
        dl = fdm.get_data_loader(
            data_set='all',
            channels=self.channels,
            layer_key=None,
            label_key=label_key,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key=label_layer_key,
            batch_size=-1,
            shuffle=True,
            return_data_loader='np_array',
            on_disk=False,
            filename_np=None,  # Filename of numpy data file if 'on_disk' is True
            # **kwargs  # No data loader kwargs needed here
        )

        # Get the data matrices from the data loader
        if label_key is not None:
            x_train, y_train = next(iter(dl))
        else:
            x_train = next(iter(dl))

            # Check if unlabeled_label in kwargs for SOM, use to create dummy y, if not use default
            unlabeled_label = self.gating_method_kwargs.get('unlabeled_label', None)
            if unlabeled_label is not None:
                y_train = np.full(x_train.shape[0], unlabeled_label)
            else:
                y_train = np.full(x_train.shape[0], -999)

        return fdm, x_train, y_train


    def _gating_helper(self, xs: List[np.ndarray]) -> List[np.ndarray]:

        # Iterate over the data list and gate
        y_preds = []
        for x in xs:

            y_proba = self.gating_module_.predict_proba(X=x)

            if self.binary_classes_:
                # Binary case: predict class 1 if prob >= threshold, else class 0
                y_pred = (y_proba[:, 1] >= self.prediction_threshold).astype(int)
            else:
                # Multiclass case: abstain (predict -1) if max prob < threshold
                max_probs = y_proba.max(axis=1)
                abstention_bool = max_probs < self.prediction_threshold

                y_pred = self.gating_module_.predict(X=x)

                y_pred[abstention_bool] = -1

            y_preds.append(y_pred)

        return y_preds


    def _reduce_dimension_helper(
            self,
            xs: List[np.ndarray],
            sample_wise: bool = False,
            dim_red_method: Literal[
                'som', 'pca', 'umap', 'tsne', 'isomap', 'locallylinearembedding', 'mds', 'spectralembedding'
            ] = 'umap',
            dim_red_method_kwargs: Union[Dict[str, Any], None] = None,
    ) -> List[np.ndarray]:

        if dim_red_method_kwargs is None:
            dim_red_method_kwargs = {}

        # --- Helper: build reducer ---
        def make_reducer(method: str):
            reducers = {
                'pca': PCA,
                'umap': UMAP,
                'tsne': TSNE,
                'isomap': Isomap,
                'locallylinearembedding': LocallyLinearEmbedding,
                'spectralembedding': SpectralEmbedding,
                'mds': MDS
            }
            if method not in reducers:
                raise NotImplementedError(f"Dimensionality reduction method '{method}' is not implemented.")
            return reducers[method](n_components=2, **dim_red_method_kwargs)

        # --- Helper: split arrays ---
        def split_arrays(a, a_references):
            """Split concatenated array back into original samples."""
            lengths = [a_reference.shape[0] for a_reference in a_references]
            starts = list(accumulate([0] + lengths[:-1]))
            return [a[start:start + length, :].copy() for start, length in zip(starts, lengths)]

        if dim_red_method == 'som':

            if self.gating_method != 'som':
                raise ValueError("The 'dimred_method' cannot be 'som' if 'gating_method' is not 'som'.")

            if self.gating_method == 'som':
                if not self.gating_module_.is_fitted_:
                    raise NotFittedError(
                        "The SOM must be fitted before it can be used for dimensionality reduction. "
                        "Call 'train()' beforehand."
                    )

            if sample_wise:
                return [self.gating_module_.transform(x)[1] for x in xs]  # Only return scattered coordinates
            else:
                x_all = np.concatenate(xs, axis=0)
                _, scatter, _, _ = self.gating_module_.transform(x_all)
                return split_arrays(a=scatter, a_references=xs)

        else:
            if sample_wise:
                return [make_reducer(method=dim_red_method).fit_transform(x) for x in xs]
            else:
                x_all = np.concatenate(xs, axis=0)
                reducer = make_reducer(method=dim_red_method)
                x_dimred = reducer.fit_transform(x_all)
                return split_arrays(a=x_dimred, a_references=xs)

    def save(self, filename: str ='gating_pipeline.pkl', filepath: Union[str, None] = None):
        """
        Save the pipeline to a pickle file, including the gating model.

        Args:
            filename (str): Output filename.
            filepath (str or None): Directory to save to. Defaults to pipeline `save_path`.

        Returns:
            None
        """

        if filepath is None:
            filepath = self.save_path

        gating_module_bytes = None

        # Save gating module separately if it has a custom save() method
        if self.gating_module_ is not None and hasattr(self.gating_module_, 'save'):

            # Save to temp file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = os.path.join(tmpdir, 'gating_module.pkl')
                self.gating_module_.save(filename='gating_module.pkl', filepath=tmpdir)

                # Read the raw bytes
                with open(tmp_file, 'rb') as f:
                    gating_module_bytes = f.read()

            # Temporarily remove module for pickling
            gating_module_backup = self.gating_module_
            self.gating_module_ = None
            classifier_saved_separately = True
        else:
            classifier_saved_separately = False
            gating_module_backup = None

        # Save pipeline + gating module bytes to one file
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump({
                'pipeline': self,
                'classifier_saved_separately': classifier_saved_separately,
                'gating_module_bytes': gating_module_bytes,
            }, f)

        # Restore gating module after saving
        if classifier_saved_separately:
            self.gating_module_ = gating_module_backup

    @classmethod
    def load(cls, filename: str = 'gating_pipeline.pkl', filepath: Union[str, None] = None):
        """
        Load a previously saved GatingPipeline.

        Args:
            filename (str): Pipeline pickle filename.
            filepath (str or None): Directory path for the file. Defaults to CWD.

        Returns:
            GatingPipeline: Fully restored pipeline instance.
        """

        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            obj = pickle.load(f)

        pipeline = obj['pipeline']
        classifier_saved_separately = obj.get('classifier_saved_separately', False)
        gating_module_bytes = obj.get('gating_module_bytes', None)

        # Reconstruct gating module using existing load() API
        if classifier_saved_separately and gating_module_bytes is not None:

            with tempfile.TemporaryDirectory() as tmpdir:

                tmp_file = os.path.join(tmpdir, 'gating_module.pkl')

                # Write bytes back to the same filename that `.load()` expects
                with open(tmp_file, 'wb') as f:
                    f.write(gating_module_bytes)

                # Ask gating module class to load normally
                if pipeline.gating_method == 'som':
                    pipeline.gating_module_ = SomClassifier.load(
                        filename='gating_module.pkl', filepath=tmpdir
                    )
                elif pipeline.gating_method == 'mlp':
                    pipeline.gating_module_ = MLPClassifier.load(
                        filename='gating_module.pkl', filepath=tmpdir
                    )
                else:
                    raise NotImplementedError(
                        f'No load method defined for {pipeline.gating_method}'
                    )

        return pipeline



