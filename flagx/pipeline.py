
import os
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt

from itertools import accumulate
from sklearn.exceptions import NotFittedError

from ._legacy_typing import List, Tuple, Dict, Union, Literal, Any
from .io import FlowDataManager, export_to_fcs
from .gating import SomClassifier, SoftmaxClassifier
from .dimred import PCA, UMAP, TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding


class GatingPipeline:
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

            gating_method: Literal['som', 'fcnn_softmax'] = 'som',
            gating_method_kwargs: Union[Dict[str, Any], None] = None,

            verbosity: int = 1,
    ):
        super().__init__()

        # Train data
        self.train_data_file_path = train_data_file_path
        self.train_data_file_names = train_data_file_names
        self.train_data_file_type = train_data_file_type

        # Initialize the save paths
        self.save_path = save_path
        self.train_data_manager_save_path = None
        self._init_save_paths()

        self.channel_names_alignment_kwargs = channel_names_alignment_kwargs
        self.preprocessing_kwargs = preprocessing_kwargs
        self.channels = channels
        self.label_key = label_key

        self.relabel_data_kwargs = relabel_data_kwargs

        # Gating method
        self.gating_method = gating_method
        self.gating_method_kwargs = gating_method_kwargs

        self.verbosity = verbosity

        self.is_trained_ = False
        self.gating_module_ = None


    def _init_save_paths(self):

        if self.save_path is None:
            self.save_path = os.getcwd()

        os.makedirs(self.save_path, exist_ok=True)

        self.train_data_manager_save_path = os.path.join(self.save_path, 'train_data_manager_output')


    def train(self):

        # Get the train data from the raw data
        train_fdm, x_train, y_train = self._data_pipeline(
            data_file_path=self.train_data_file_path,
            data_file_names=self.train_data_file_names,
            data_file_type=self.train_data_file_type,
            label_key=self.label_key,
            data_manager_save_path=self.train_data_manager_save_path,
            save_meta_info=True,
            fn_prefix_saving='train_'
        )

        # Instantiate the gating module of the pipeline
        if self.gating_method_kwargs is None:
            self.gating_method_kwargs = {}

        if self.gating_method == 'som':
            self.gating_module_ = SomClassifier(**self.gating_method_kwargs)
        elif self.gating_method == 'fcnn_softmax':
            if self.label_key is None:
                raise ValueError(
                    "'label_key' is required when gating_method is 'fcnn_softmax'. "
                    "Unsupervised training is not possible for a FCNN."
                )
            self.gating_module_ = SoftmaxClassifier(**self.gating_method_kwargs)
        else:
            raise NotImplementedError(
                f"Gating method '{self.gating_method}' is not implemented. "
                "Supported methods are: 'som', 'fcnn_softmax'."
            )

        # Call the fit method of the gating module
        self.gating_module_.fit(X=x_train, y=y_train)

        self.is_trained_ = True


    def inference(
            self,
            data_file_path: Union[str, None] = None,  # default: cwd
            data_file_names: Union[List[str], None] = None,  # default: listdir(path)
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
            save_sample_wise: bool = False,  # whether to save to individual or one fcs file
            save_path: Union[str, None] = None,
            save_filenames: Union[str, List[str], None] = None,
            val_range: Tuple[float, float] = (0.0, 2 ** 20),
            keep_unscaled: bool = False,
            fcs_metadata_dicts: Union[Dict, List[Dict], None] = None
    ):

        if dim_red_method_kwargs is None:
            dim_red_method_kwargs = ({}, ) * len(dim_red_methods)
        else:
            if len(dim_red_methods) != len(dim_red_method_kwargs):
                raise ValueError("Mismatch: 'dim_red_methods' and 'dim_red_method_kwargs' must have the same length.")

        # Load and process the data
        fdm_save_p = os.path.join(self.save_path, 'inference_data_manager_output')
        os.makedirs(fdm_save_p, exist_ok=True)

        # Load and process the data
        fdm, _, _ = self._data_pipeline(
            data_file_path=data_file_path,
            data_file_names=data_file_names,
            data_file_type=self.train_data_file_type,
            label_key=None,  # No labels here
            data_manager_save_path=fdm_save_p,
            fn_prefix_saving=None,
            save_meta_info=False,
        )

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

        if gate and self.label_key is not None:
            if not self.is_trained_:
                raise NotFittedError("This pipeline instance is not trained yet. Call 'train' before gating.")

            y_preds = self._gating_helper(xs=xs)

        else:
            if self.label_key is None:
                warnings.warn(
                    "Gating was requested (gate=True), but 'label_key' is None. Skipping gating step.",
                    UserWarning
                )

            y_preds = None

        if dim_red_methods is not None:
            x_dimreds = []
            other_annotations = []
            other_annotations_names = []
            for dim_red_method, kwargs_dict in zip(dim_red_methods, dim_red_method_kwargs):

                out = self._reduce_dimension_helper(
                    xs=xs,
                    dim_red_method=dim_red_method,
                    dim_red_method_kwargs=kwargs_dict,
                )

                if dim_red_method == 'som':
                    x_dimred, x_dimred_scattered, som_unit_ids, radii = out

                    x_dimreds.append(x_dimred_scattered)

                    other_annotations.append([x[:, 0] for x in x_dimred])
                    other_annotations_names.append('som_no_scatter_1')
                    other_annotations.append([x[:, 1] for x in x_dimred])
                    other_annotations_names.append('som_no_scatter_2')

                    other_annotations.append(som_unit_ids)
                    other_annotations_names.append('som_unit_id')

                else:
                    x_dimred = out
                    x_dimreds.append(x_dimred)

            # If 'som' was not in dim_red_methods, set other_annotations to None
            if not other_annotations:
                other_annotations = None
                other_annotations_names = None
        else:
            x_dimreds = None
            other_annotations = None
            other_annotations_names = None

        # Get the key of the layer where the raw data (no transformation) was stored
        if self.preprocessing_kwargs is not None:
            raw_layer_key = self.preprocessing_kwargs.get('save_raw_to_layer', 'raw')
        else:
            raw_layer_key = None

        export_to_fcs(
            data_list=fdm.anndata_list_,
            layer_key=raw_layer_key,
            val_range=val_range,
            keep_unscaled=keep_unscaled,
            sample_wise=save_sample_wise,
            y_preds=y_preds,
            dim_red_coords=x_dimreds,
            dim_red_names=dim_red_methods,
            other_annotations=other_annotations,
            other_annotations_names=other_annotations_names,
            save_path=save_path,
            save_filenames=save_filenames,
            fcs_metadata_dicts=fcs_metadata_dicts,
        )


    def _data_pipeline(
            self,
            data_file_path: Union[str, None] = None,  # default: cwd
            data_file_names: Union[List[str], None] = None,  # default: listdir(path)
            data_file_type: Union[Literal['fcs', 'csv'], None] = None,
            label_key: Union[int, str, None] = None,  # If None unlabeled case
            data_manager_save_path: Union[str, None] = None,  # default: cwd
            save_meta_info: bool = False,
            fn_prefix_saving: Union[str, None] = None,
    ) -> Tuple[FlowDataManager, np.ndarray, np.ndarray]:

        if fn_prefix_saving is None:
            fn_prefix_saving = ''

        # If no filenames are passed, get filenames from data dir
        if data_file_names is None:
            data_file_names = os.listdir(data_file_path)

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

        if save_meta_info:
            # Check the number of events per sample
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
                filename_log_df=f'{fn_prefix_saving}og_channel_names.csv',
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
            save_raw_to_layer = self.preprocessing_kwargs.get('save_raw_to_layer', 'raw')
            flavour_kwargs = self.preprocessing_kwargs.get('flavour_kwargs', {})

            fdm.sample_wise_preprocessing(
                flavour=flavour,
                save_raw_to_layer=save_raw_to_layer,
                **flavour_kwargs,
            )

        # Get the label_layer_key
        if self.preprocessing_kwargs is not None and label_key is not None:
            label_layer_key = self.preprocessing_kwargs.get('save_raw_to_layer', 'raw')
        else:
            label_layer_key = None

        # Create dataloader
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

        # Get the train data from the data loader and fit
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
            y_pred = self.gating_module_.predict(X=x)
            y_preds.append(y_pred)

        return y_preds


    def _reduce_dimension_helper(
            self,
            xs: List[np.ndarray],
            dim_red_method: Literal[
                'som', 'pca', 'umap', 'tsne', 'isomap', 'locallylinearembedding', 'mds', 'spectralembedding'
            ] = 'umap',
            dim_red_method_kwargs: Union[Dict[str, Any], None] = None,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:

        # Usually new data
        # If som is som for dim red, scatter coordinates

        if self.gating_method != 'som' and dim_red_method == 'som':
            raise ValueError("The 'dimred_method' cannot be 'som' if 'gating_method' is not 'som'.")

        if self.gating_method == 'som' and dim_red_method == 'som':
            if not self.gating_module_.is_fitted_:
                raise NotFittedError(
                    "The SOM must be fitted before it can be used for dimensionality reduction. "
                    "Call 'train' before using 'reduce_dimension'."
                )

        x_all = np.concatenate(xs, axis=0)

        if dim_red_method == 'som':

            x_dimred, x_dimred_scattered, som_unit_ids, radii = self.gating_module_.transform(x_all)

            lengths = [x.shape[0] for x in xs]
            starts = list(accumulate([0] + lengths[:-1]))
            x_dimred = [x_dimred[start:start + length, :].copy() for start, length in zip(starts, lengths)]
            x_dimred_scattered = [x_dimred_scattered[start:start + length, :].copy() for start, length in zip(starts, lengths)]
            som_unit_ids = [som_unit_ids[start:start + length].copy() for start, length in zip(starts, lengths)]
            radii = [radii[start:start + length].copy() for start, length in zip(starts, lengths)]

            return x_dimred, x_dimred_scattered, som_unit_ids, radii

        else:

            if dim_red_method_kwargs is None:
                dim_red_method_kwargs = {}

            if dim_red_method == 'pca':
                reducer = PCA(n_components=2, **dim_red_method_kwargs)
            elif dim_red_method == 'umap':
                reducer = UMAP(n_components=2, **dim_red_method_kwargs)
            elif dim_red_method == 'tsne':
                reducer = TSNE(n_components=2, **dim_red_method_kwargs)
            elif dim_red_method == 'isomap':
                reducer = Isomap(n_components=2, **dim_red_method_kwargs)
            elif dim_red_method == 'locallylinearembedding':
                reducer = LocallyLinearEmbedding(n_components=2, **dim_red_method_kwargs)
            elif dim_red_method == 'spectralembedding':
                reducer = SpectralEmbedding(n_components=2, **dim_red_method_kwargs)
            elif dim_red_method == 'mds':
                reducer = MDS(n_components=2, **dim_red_method_kwargs)
            else:
                raise NotImplementedError(f"Dimensionality reduction method '{dim_red_method}' is not implemented.")

            if self.verbosity >= 2:
                print(f'# ### Computing {dim_red_method} ...')

            x_dimred = reducer.fit_transform(x_all)

            lengths = [x.shape[0] for x in xs]
            starts = list(accumulate([0] + lengths[:-1]))
            x_dimred = [x_dimred[start:start + length, :].copy() for start, length in zip(starts, lengths)]

            return x_dimred

    def save(
            self,
            filename: str = 'gating_pipeline.pkl',
            filepath: Union[str, None] = None,
    ):

        """Save the full pipeline to a pickle file, handling gating module separately if needed."""

        if filepath is None:
            filepath = self.save_path

        # Save classifier separately if it has a custom save method
        if self.gating_module_ is not None and hasattr(self.gating_module_, 'save'):
            self.gating_module_.save(
                filename='gating_module_' + filename, filepath=filepath
            )
            classifier_saved_separately = True
            gating_module_backup = self.gating_module_
            self.gating_module_ = None  # Temporarily remove for pickling
        else:
            classifier_saved_separately = False
            gating_module_backup = None  # nothing to restore

        # Save the rest of the pipeline
        with open(os.path.join(filepath, filename), 'wb') as f:
            pickle.dump({
                'pipeline': self,
                'classifier_saved_separately': classifier_saved_separately
            }, f)

        # Restore gating module after saving
        if classifier_saved_separately:
            self.gating_module_ = gating_module_backup

    @classmethod
    def load(
            cls,
            filename: str = 'gating_pipeline.pkl',
            filepath: Union[str, None] = None,
    ):

        """Load the full pipeline from a pickle file. Load gating module separately if needed."""

        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'rb') as f:
            obj = pickle.load(f)

        pipeline = obj['pipeline']
        classifier_saved_separately = obj.get('classifier_saved_separately', False)

        # Load classifier if saved separately
        if classifier_saved_separately:
            if pipeline.gating_method == 'som':
                pipeline.gating_module_ = SomClassifier.load(
                    filename='gating_module_' + filename, filepath=filepath,
                )
            elif pipeline.gating_method == 'fcnn_softmax':
                pipeline.gating_module_ = SoftmaxClassifier.load(
                    filename='gating_module_' + filename, filepath=filepath,
                )
            else:
                raise NotImplementedError(f"No load method defined for {pipeline.gating_method}")

        return pipeline


