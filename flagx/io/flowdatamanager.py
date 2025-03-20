
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import pytometry as pm
import copy
import os
import warnings
import gc

from typing import Sequence, Tuple, Union, Literal, List, Dict, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .flowdataset import FlowDataset
from .flowdataloader import FlowDataLoader

# Todo:
#  - Add documentation

class FlowDataManager:
    def __init__(
            self,
            data_file_names: List[str],
            data_file_type: Union[Literal['fcs', 'csv'], None] = None,
            data_file_path: Union[str, None] = None,
            save_path: Union[str, None] = None,
            verbosity: int = 1,  # 0 = silent, 1 = warnings, 2 = info
    ):
        # ### Check input format
        if not isinstance(data_file_names, list) or any(not isinstance(x, str) for x in data_file_names):
            raise TypeError("'data_file_names' must be a list of strings")
        if not len(data_file_names) >= 1:
            raise ValueError("'data_file_names' must have at least one entry")

        if data_file_type not in ['fcs', 'csv', None]:
            raise ValueError("'data_file_type' must be either 'fcs', 'csv', or None")

        if not isinstance(data_file_path, (str, type(None))):
            raise TypeError("'data_file_path' must be a string or None")

        if not isinstance(save_path, (str, type(None))):
            raise TypeError("'save_path' must be a string or None")

        if not isinstance(verbosity, int) or verbosity < 0:
            raise ValueError("'verbosity' must be an integer >= 0")


        # ### Set path variables for data loading and storage
        self._data_file_names = data_file_names  # List of filenames that should be loaded
        self._data_file_type = data_file_type  # If None guessed from file ending of 1st filename, assume all have same type

        self._data_file_path = data_file_path if data_file_path is not None else os.getcwd()  # Path to .fcs/.csv
        self._save_path = save_path if save_path is not None else os.path.join(os.getcwd(), 'data_handling') # Path to save any results to
        os.makedirs(self._save_path, exist_ok=True)

        self._verbosity = verbosity

        # Conventions:
        # - Store only data as attribute if it does not concern one specific subset of the data (i.e. train, val, test)
        # - Assume integer labels

        # When load_data_files_to_anndata() was called
        self.invalid_files_ = None  # Files that are not .fcs or .csv
        self.anndata_list_ = None  # Either list of AnnData or list of .h5ad filenames (stored at save_path)

        # When check_sample_sizes() was called
        self.sample_sizes_ = None

        # When align_channel_names() was called
        self.og_channel_names_ = None

        # When perform_data_split() was called
        self.train_data_ = None
        self.test_data_ = None
        self.val_data_ = None

    # ### Add attributes as immutable properties #######################################################################
    @property
    def data_file_names(self):
        """Read-only property for data file names."""
        return self._data_file_names

    @property
    def data_file_type(self):
        """Read-only property for data file type."""
        return self._data_file_type

    @property
    def data_file_path(self):
        """Read-only property for data file path."""
        return self._data_file_path

    @property
    def save_path(self):
        """Mutable property for save path."""
        return self._save_path

    @save_path.setter
    def save_path(self, new_path: str):
        """Allows updating the save path and ensures the directory exists."""
        if not isinstance(new_path, str):
            raise TypeError("'new_path' must be a string")
        self._save_path = new_path
        os.makedirs(self._save_path, exist_ok=True)  # Ensure the new path exists

    @property
    def verbosity(self):
        """Mutable property for save path."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, new_verbosity: int):
        """Allows updating the verbosity level."""
        if not isinstance(new_verbosity, int) or new_verbosity < 0:
            raise ValueError("'new_verbosity' must be an integer >= 0")
        self._verbosity = new_verbosity

    # ### load_data_files_to_anndata() #################################################################################
    def load_data_files_to_anndata(self):

        # If no filetype is passed, determine from ending of 1st file
        if self._data_file_type is None:
            self._data_file_type = FlowDataManager._determine_filetype(filename=self._data_file_names[0])

            if self._data_file_type == "unknown":
                raise ValueError(f"Unsupported or unknown file type for {self._data_file_names[0]}. "
                                 f"Cannot use it as reference. Please remove it from 'data_filenames''")

        # Initialize list for saving anndatas (or their filenames) and invalid filenames
        self.invalid_files_ = []
        self.anndata_list_ = []

        for fn in self._data_file_names:

            # Check the filetype of the input file
            ft = FlowDataManager._determine_filetype(filename=fn)
            if ft != self._data_file_type:
                if self._verbosity >= 1:
                    warnings.warn(
                        f"Skipping invalid file '{fn}'. "
                        f"It is of type '{ft}' but should be '{self._data_file_type}'.",
                        UserWarning
                    )
                self.invalid_files_.append(fn)
                continue

            # Load data file to anndata
            if self._data_file_type == 'fcs':  # data_file_type is fcs
                adata = pm.io.read_fcs(os.path.join(self._data_file_path, fn))
            else: # data_file_type is csv
                df = pd.read_csv(os.path.join(self._data_file_path, fn), dtype=np.float32)
                adata = sc.AnnData(X=df.to_numpy())
                adata.var_names = df.columns.copy()

            # Annotate filename in uns of anndata
            adata.uns['filename'] = fn

            self.anndata_list_.append(adata)

    @staticmethod
    def _determine_filetype(filename: str) -> str:
        if filename.endswith(".fcs"):
            data_file_type = "fcs"
        elif filename.endswith(".csv"):
            data_file_type = "csv"
        else:
            data_file_type = "unknown"
        return data_file_type

    # check_sample_sizes() #############################################################################################
    def check_sample_sizes(
            self,
            filename_sample_sizes_df: Union[str, None] = None,
    ):
        self.sample_sizes_ = FlowDataManager.check_sample_sizes_worker(
            data_list=self.anndata_list_,
            save_path=self._save_path,
            filename_sample_sizes_df=filename_sample_sizes_df,
            verbosity=self._verbosity,
        )

    @staticmethod
    def check_sample_sizes_worker(
            data_list: List[sc.AnnData],
            save_path: Union[str, None] = None,
            filename_sample_sizes_df: Union[str, None] = None,
            verbosity: int = 0,
    ) -> pd.DataFrame:

        # ### Inspect the number of samples and their sample size
        sn = []
        ss = []
        for fldata in data_list:
            sn.append(fldata.uns['filename'])
            ss.append(fldata.X.shape[0])

        df = pd.DataFrame()
        df['sample'] = sn
        df['n_events'] = ss
        s = df['n_events'].sum()
        m = df['n_events'].mean()
        std = df['n_events'].std()
        df.loc[len(df)] = ['std', std]
        df.loc[len(df)] = ['mean', m]
        df.loc[len(df)] = ['total', s]

        if save_path is None:
            save_path = os.getcwd()

        if filename_sample_sizes_df is not None:
            df.to_csv(os.path.join(save_path, filename_sample_sizes_df))

        if verbosity >= 2:
            print(f'# ### Sample sizes:\n{df}')

        return df

    # ### align_channel_names(), check_og_channel_names_df() ###########################################################
    def align_channel_names(
            self,
            reference_channel_names: Union[int, dict, None] = None,
            filename_log_df: Union[str, None] = None,
    ) -> None:
        log_df = FlowDataManager.align_channel_names_worker(
            data_list=self.anndata_list_,  # Work on anndata_list
            reference=reference_channel_names,  # Int = idx of anndata_list or dict: {og_cn: new_cn}, None = 1st entry of list as reference
            inplace=True,  # Work inplace, change anndata_list
            filename_log_df=filename_log_df,  # Filename for log df, None then no saving
            save_path=self._save_path,  # Where to save log_df to
        )
        self.og_channel_names_ = log_df
        self.check_og_channel_names_df()

    @staticmethod
    def align_channel_names_worker(
            data_list: List[sc.AnnData],
            reference: Union[int, dict],  # Either int for which file to use as reference or a
            # dictionary with possible_name: reference_name
            inplace: bool = False,
            filename_log_df: Union[str, None] = None,  # Filename for log df, None then no saving
            save_path: Union[str, None] = None,  # Where to save log_df to, None then cwd
    ) -> Union[Tuple[List[sc.AnnData], pd.DataFrame], pd.DataFrame]:
        # ### Function to unify the channel names across multiple fcs data objects,
        # assumes the same number of channels for all

        # ### Copy input if it should not be altered inplace
        if not inplace:
            data_list = copy.deepcopy(data_list)

        # ### If idx to reference anndata / file is passed use it to create list of reference channel names
        if isinstance(reference, int):
            # Create list of channel names on the basis of selected AnnData object
            reference = data_list[reference].var_names.values.tolist()

        # ### Create dataframe to store the original channel names
        log_df = pd.DataFrame(columns=['filename'] + list(range(1, data_list[0].n_vars + 1)))

        # ### Iterate over individual fcs samples and change their channel names
        for adata in data_list:

            # Change channel names of AnnData object
            _, log_df = FlowDataManager._align_channel_names_helper(
                adata=adata,
                reference=reference,
                log_df=log_df
            )

            # Add filename key always exists in .uns since it is added in load_data_files_to_anndata()
            log_df.loc[log_df.index[-1], 'filename'] = adata.uns['filename']

        if save_path is None:
            save_path = os.getcwd()

        if filename_log_df is not None:
            log_df.to_csv(os.path.join(save_path, filename_log_df))

        if not inplace:
            return data_list, log_df
        else:
            return log_df

    @staticmethod
    def _align_channel_names_helper(
            adata: sc.AnnData,
            reference: Union[List[str], Dict],
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
        FlowDataManager.check_og_channel_names_df_worker(self.og_channel_names_)

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
            else:
                print(f'# ### Channel: {i}, Name: {value_counts.index[0]} is consistent across samples\n')

    # ### sample_wise_preprocessing() ##################################################################################
    def sample_wise_preprocessing(
            self,
            flavour: Literal['logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'custom'] = 'arcsinh',
            save_raw_to_layer: Union[str, None] = None,
            **kwargs
    ) -> None:
        FlowDataManager.sample_wise_preprocessing_worker(
            data_list=self.anndata_list_,
            flavour=flavour,
            inplace=True,
            save_raw_to_layer=save_raw_to_layer,
            **kwargs
        )

    @staticmethod
    def sample_wise_preprocessing_worker(
            data_list: List[sc.AnnData],
            flavour: Literal['logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'custom'],  # custom must work inplace
            inplace: bool = False,
            save_raw_to_layer: Union[str, None] = None,  # Key for layer where raw data is stored, if None no storage
            **kwargs,
    ) -> Union[List[sc.AnnData], None]:

        if flavour not in {'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'custom'}:
            raise ValueError(
                "'flavour' must be one of: 'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff' or 'custom'")

        if not inplace:
            data_list = copy.deepcopy(data_list)

        if flavour == 'logicle':
            trafo_fct = pm.tl.normalize_logicle
        elif flavour == 'arcsinh':
            trafo_fct = pm.tl.normalize_arcsinh
        elif flavour == 'biexp':
            trafo_fct = pm.tl.normalize_biExp
        elif flavour == 'log10_w_cutoff':
            trafo_fct = FlowDataManager.log10_w_cutoff
        else:
            if 'preprocessing_method' not in kwargs:
                raise ValueError(
                    "'preprocessing_method' must be provided in kwargs when 'flavour' is 'custom'."
                )
            trafo_fct = kwargs.pop('preprocessing_method')

        for adata in data_list:

            # Store unprocessed data matrix in layer
            if save_raw_to_layer is not None:
                adata.layers[save_raw_to_layer] = adata.X.copy()

            # Apply transformation
            if kwargs:
                trafo_fct(adata=adata, **kwargs)
            else:
                trafo_fct(adata=adata)

        if not inplace:
            return data_list

    @staticmethod
    def log10_w_cutoff(adata: sc.AnnData, cutoff: float = 100):
        x = adata.X
        x = np.log10(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
        adata.X = x

    # ### perform_data_split() #########################################################################################
    def perform_data_split(
            self,
            data_split: Union[Tuple[float, float], Tuple[float, float, float], pd.DataFrame] = (0.75, 0.25),
            filename_data_split: Union[str, None] = None,
            **kwargs,
    ) -> None:

        dummy_data_split = FlowDataManager.perform_data_split_worker(
            data_list=self.anndata_list_,
            data_split=data_split,
            filename_data_split=filename_data_split,
            save_path=self._save_path,
            verbosity=self._verbosity,
            **kwargs
        )

        if len(dummy_data_split) == 2:
            self.train_data_, self.test_data_ = dummy_data_split
        else:
            self.train_data_, self.val_data_, self.test_data_ = dummy_data_split

    @staticmethod
    def perform_data_split_worker(
            data_list: List[sc.AnnData],
            data_split: Union[Tuple[float, float], Tuple[float, float, float], pd.DataFrame],
            filename_data_split: Union[str, None] = None,
            save_path: Union[str, None] = None,
            verbosity: int = 0,
            **kwargs  # kwargs for sklearn are: random_state, shuffle, stratify
    ) -> Tuple[Union[Sequence[str], Sequence[sc.AnnData]], ...]:

        # ### Split according to fractions passed as tuple
        if not isinstance(data_split, pd.DataFrame):
            if len(data_split) not in {2, 3}:
                raise ValueError(
                    "'data_split' must be tuple or triple corresponding to fractions for train- (val-) and test-data")

            if sum(data_split) != 1 or any(x < 0 for x in data_split):
                raise ValueError(
                    'The train-(val-)test-split must be passed as a tuple of non negative decimals that sum to one')

            if save_path is None:
                save_path = os.getcwd()

            if len(data_split) == 2:
                # Split into train and test set
                train_data, test_data = train_test_split(
                    data_list,
                    test_size=data_split[1],
                    train_size=data_split[0],
                    **kwargs
                )

                # Save data split to .csv (filename and train, test information)
                if filename_data_split is not None:
                    FlowDataManager._save_data_split_helper(
                        data_tuple=(train_data, test_data),
                        filename_data_split=filename_data_split,
                        save_path=save_path,
                    )

                return train_data, test_data

            else:
                # Split into train and val-test set
                perc_val_test_data = data_split[1] + data_split[2]
                train_data, val_test_data = train_test_split(
                    data_list,
                    test_size=perc_val_test_data,
                    train_size=data_split[0], **kwargs
                )

                # Split val-test data into val and test set
                val_data, test_data = train_test_split(
                    val_test_data,
                    test_size=data_split[2] / perc_val_test_data,
                    train_size=data_split[1] / perc_val_test_data,
                    **kwargs
                )

                # Save data split to .csv (filename and train, val, test information)
                if filename_data_split is not None:
                    FlowDataManager._save_data_split_helper(
                        data_tuple=(train_data, val_data, test_data),
                        filename_data_split=filename_data_split,
                        save_path=save_path,
                    )

                return train_data, val_data, test_data

        # Split according to previously saved dataframe
        else:

            # Check format of data split dataframe and set filenames as index
            data_split = FlowDataManager._check_data_split_df_format(data_split=data_split)

            # 'data_list' is list of AnnData with filename annotated in .uns
            train_data = []
            val_data = []
            test_data = []

            # Iterate over the data-split dataframe
            for d in data_list:
                mode = data_split.loc[d.uns['filename'], 'mode']
                if mode == 'train':
                    train_data.append(d)
                elif mode == 'val':
                    val_data.append(d)
                else:
                    test_data.append(d)

            if len(val_data) == 0:
                if verbosity >= 2:
                    print('# ### The passed data_split dataframe did not include validation data')
                return train_data, test_data
            else:
                return train_data, val_data, test_data

    @staticmethod
    def _save_data_split_helper(
            data_tuple: Tuple[List[sc.AnnData], ...],
            filename_data_split: Union[str, None] = None,
            save_path: Union[str, None] = None,
    ):

        # Define modes based on the length of data_tuple
        mode_labels = ['train', 'val', 'test'] if len(data_tuple) == 3 else ['train', 'test']

        # Append the filename and the respective mode to lists, modes are: (train, val, test)
        fns = []
        modes = []
        for data_list, mode in zip(data_tuple, mode_labels):
            for adata in data_list:
                fns.append(adata.uns['filename'])
                modes.append(mode)

        # Create dataframe and save
        df = pd.DataFrame({
            'filename': fns,
            'mode': modes
        })

        df.to_csv(os.path.join(save_path, filename_data_split))

    @staticmethod
    def _check_data_split_df_format(data_split: pd.DataFrame):

        if 'filename' not in data_split.columns:
            raise ValueError("The column 'filename' is missing in the 'data_split' dataframe'")

        if 'mode' not in data_split.columns:
            raise ValueError("The column 'mode' is missing in the 'data_split' dataframe'")

        # Set filename columns as index
        data_split.set_index('filename', drop=True, inplace=True)

        return data_split

    # ### get_data_loader() ############################################################################################
    def get_data_loader(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            batch_size: int = -1,
            shuffle: bool = True,
            return_data_loader: Literal['np_array', 'torch_tensor'] = 'np_array',
            on_disk: bool = False,
            filename_np: Union[str, None] = None,  # Filename of numpy data file if 'on_disk' is True
            **kwargs,
    ) -> Union[DataLoader, None]:

        if data_set == 'all':
            data_list = self.anndata_list_
        elif data_set == 'train':
            data_list = self.train_data_
        elif data_set == 'test':
            data_list = self.test_data_
        elif data_set == 'val':
            data_list = self.val_data_

            if data_list is None:
                if self._verbosity >= 1:
                    warnings.warn(
                        'No validation set was created when splitting the data. '
                        'Options are "train", "test", "all". Returning None.' ,
                        UserWarning
                    )
                return
        else:
            raise ValueError("'data_set' must be 'all', 'train', 'test' or 'val'")

        out = FlowDataManager.get_data_loader_worker(
            data_list=data_list,
            channels=channels,
            layer_key=layer_key,
            label_key=label_key,
            label_layer_key=label_layer_key,
            batch_size=batch_size,
            shuffle=shuffle,
            return_data_loader=return_data_loader,
            on_disk=on_disk,
            save_path=self._save_path,
            filename_np=filename_np,
            verbosity=self._verbosity,
            **kwargs,
        )

        return out

    @staticmethod
    def get_data_loader_worker(
            data_list: List[sc.AnnData],
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            batch_size: int = -1,
            shuffle: bool = True,
            return_data_loader: Literal['np_array', 'torch_tensor'] = 'np_array',
            on_disk: bool = False,
            save_path: Union[str, None] = None,  # Where .npy files are saved if 'on_disk' is True
            filename_np: Union[str, None] = None,  # Filename of numpy data file if 'on_disk' is True
            verbosity: int = 1,
            **kwargs
    ) -> DataLoader:

        if on_disk and save_path is None:
            raise ValueError(
                "If 'on_disk' is True 'save_path' (= dir where the dataloader data file is stored) cannot be None"
            )

        # Set all channels as data if none are specified
        if channels is None:
                channels = list(range(data_list[0].n_vars))

        # Get single data matrix (concatenated from all samples)
        data_array = FlowDataManager._get_numpy_data_matrix(
            data_list=data_list,
            channels=channels,
            layer_key=layer_key,
        )

        # Add labels as last columns of data matrix
        if label_key is not None:
            label_array = FlowDataManager._get_numpy_label_vector(
                data_list=data_list,
                label_key=label_key,
                layer_key=label_layer_key,
                verbosity=verbosity,
            )

            data_array = np.concatenate((data_array, np.expand_dims(label_array, axis=1)), axis=1)

        # Save data array to disk
        if on_disk:

            if save_path is None:
                save_path = os.getcwd()

            if filename_np is None:
                filename_np = 'data.npy'

            np.save(os.path.join(save_path, filename_np), data_array)

            # Delete data array from memory
            del data_array
            gc.collect()

            data = os.path.join(save_path, filename_np)
        else:
            data = data_array

        # Instantiate FlowDataset
        ds = FlowDataset(
            data=data,
            on_disk=on_disk,
            includes_labels=True if label_key is not None else False,  # Assume labels in last column
        )

        # Set batch size to all data if batch_size == -1
        if batch_size == -1:
            batch_size = len(ds)

        # Instantiate the FlowDataLoader
        flow_dataloader = FlowDataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

        # Get Dataloader that returns np arrays or pytorch tensors
        if return_data_loader == 'np_array':
            out = flow_dataloader.pytorch_np_dataloader
        else:
            out = flow_dataloader.pytorch_dataloader

        return out

    @staticmethod
    def _get_numpy_data_matrix(
            data_list: List[sc.AnnData],
            channels: Union[Sequence[int], Sequence[str]],
            layer_key: Union[str, None] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        # Get data matrix from each anndata in data_list
        arrays = []
        for adata in data_list:

            if layer_key is None:
                arrays.append(adata[:, channels].X.copy())
            else:
                arrays.append(adata[:, channels].layers[layer_key].copy())

        # Concatenate to single data matrix
        array = np.concatenate(arrays, axis=0)

        return array

    @staticmethod
    def _get_numpy_label_vector(
            data_list: List[sc.AnnData],
            label_key: Union[int, str, None],
            layer_key: Union[str, None] = None,
            verbosity: int = 1,
    ) -> np.ndarray:

        # Get labels from each anndata in data_list
        labels = []
        for adata in data_list:

            labels.append(
                FlowDataManager._get_labels(
                    adata=adata,
                    label_key=label_key,
                    layer_key=layer_key,
                    verbosity=verbosity,
                )
            )

        label_array = np.concatenate(labels, axis=0)

        return label_array

    @staticmethod
    def _get_labels(
            adata: sc.AnnData,
            label_key: Union[str, int],
            layer_key: Union[str, None] = None,
            verbosity: int = 1,
    ) -> np.ndarray:

        # Label key is index of data matrix
        if isinstance(label_key, int):
            try:
                if layer_key is None:
                    labels = adata.X[:, label_key].copy()
                else:
                    labels = adata.layers[layer_key][:, label_key].copy()
            except IndexError:
                raise ValueError("'label_key' index is out of bounds in .X/.layers[layer_key] matrix")

        # Label key is var name or obs key
        else:
            try:
                label_idx = adata.var_names.get_loc(label_key)
                if layer_key is None:
                    labels = adata.X[:, label_idx].copy()
                else:
                    labels = adata.layers[layer_key][:, label_idx].copy()
            except KeyError:
                if verbosity >= 1:
                    warnings.warn(f"'label_key' not found in .var_names, trying .obs")
                try:
                    labels = adata.obs[label_key].to_numpy().copy()
                except KeyError:
                    raise ValueError("'label_key' not found in .obs or .var_names")
        return labels

    # ### sample_wise_downsampling() ###################################################################################
    def sample_wise_downsampling(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            fraction: float,
            stratified: bool = False,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
    ) -> None:

        if data_set == 'all':
            data_list = self.anndata_list_
        elif data_set == 'train':
            data_list = self.train_data_
        elif data_set == 'test':
            data_list = self.test_data_
        elif data_set == 'val':
            data_list = self.val_data_

            if data_list is None:
                if self._verbosity >= 1:
                    warnings.warn(
                        'No validation set was created when splitting the data. '
                        'Options are "train", "test", "all".',
                        UserWarning
                    )
                return
        else:
            raise ValueError("'data_set' must be 'all', 'train', 'test' or 'val'")

        # Downsample selected data list inplace, if og is to be kept use the worker
        FlowDataManager.sample_wise_downsampling_worker(
            data_list=data_list,
            fraction=fraction,
            stratified=stratified,
            label_key=label_key,
            label_layer_key=label_layer_key,
            inplace=True,
        )

    @staticmethod
    def sample_wise_downsampling_worker(
            data_list: List[sc.AnnData],
            fraction: float,
            stratified: bool = False,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            inplace : bool = False,
    ) -> Union[Sequence[sc.AnnData], None]:

        if fraction < 0 or fraction > 1:
            raise ValueError("'fraction' must be between 0 and 1")

        if stratified and label_key is None:
            raise ValueError("'stratified' is True but 'label_key' is None. Need labels for stratification.")

        if not inplace:
            data_list = copy.deepcopy(data_list)

        for i, adata in enumerate(data_list):
            # Get labels (by column index, column name, obs key)
            labels = FlowDataManager._get_labels(adata=adata, label_key=label_key, layer_key=label_layer_key)
            # Get bool indicating which events to keep
            ds_bool = FlowDataManager._get_downsampling_bool(y=labels, fraction=fraction, stratified=stratified)
            # Update data_list
            data_list[i] = adata[ds_bool, :].copy()

        if not inplace:
            return data_list

    @staticmethod
    def _get_downsampling_bool(y: np.ndarray, fraction: float, stratified: bool = False) -> np.ndarray:

        keep_mask = np.zeros_like(y, dtype=bool)

        if stratified:

            unique_labels, counts = np.unique(y, return_counts=True)

            for label, count in zip(unique_labels, counts):
                # Get indices where y == label
                label_indices = np.where(y == label)[0]
                # Keep fraction events with label, at least one
                num_events_to_keep = max(1, int(np.round(count * fraction)))
                # Choose num_events_to_keep random events with label
                selected_indices = np.random.choice(label_indices, num_events_to_keep, replace=False)
                # Set mask to True for selected events
                keep_mask[selected_indices] = True

        else:
            num_samples = y.shape[0]
            num_samples_to_keep = max(1, int(np.round(num_samples * fraction)))

            selected_indices = np.random.choice(np.arange(num_samples), num_samples_to_keep, replace=False)

            keep_mask[selected_indices] = True

        return keep_mask

    # ### check_class_balance() ########################################################################################
    def check_class_balance(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            label_key: Union[int, str],
            label_layer_key: Union[str, None] = None,
            filename_class_balance_df: Union[str, None] = None,
    ) -> Union[pd.DataFrame, None]:

        if data_set == 'all':
            data_list = self.anndata_list_
        elif data_set == 'train':
            data_list = self.train_data_
        elif data_set == 'test':
            data_list = self.test_data_
        elif data_set == 'val':
            data_list = self.val_data_

            if data_list is None:
                if self._verbosity >= 1:
                    warnings.warn(
                        'No validation set was created when splitting the data. '
                        'Options are "train", "test", "all". Returning None',
                        UserWarning
                    )
                return
        else:
            raise ValueError("'data_set' must be 'all', 'train', 'test' or 'val'")

        class_balance_df = FlowDataManager.check_class_balance_worker(
            data_list=data_list,
            label_key=label_key,
            label_layer_key=label_layer_key,
            save_path=self.save_path,
            filename_class_balance_df=filename_class_balance_df,
        )

        return class_balance_df

    @staticmethod
    def check_class_balance_worker(
            data_list: List[sc.AnnData],
            label_key: Union[int, str],
            label_layer_key: Union[str, None] = None,
            save_path: Union[str, None] = None,
            filename_class_balance_df: Union[str, None] = None,
            verbosity: int = 1,
    ) -> pd.DataFrame:
        # Extract labels from data list
        label_vec = FlowDataManager._get_numpy_label_vector(
            data_list=data_list,
            label_key=label_key,
            layer_key=label_layer_key,
            verbosity=verbosity,
        )

        unique_labels, class_counts = np.unique(label_vec, return_counts=True)
        class_fracs = class_counts / class_counts.sum()

        sorted_indices = np.argsort(class_counts)[::-1]
        unique_labels = unique_labels[sorted_indices]
        class_counts = class_counts[sorted_indices]
        class_fracs = class_fracs[sorted_indices]

        if verbosity >= 2:
            print(f'# ### Absolute counts for the labels:\n{class_counts}')
            print(f'# ### Relative frequencies for the labels:\n{class_fracs}')

        results_df = pd.DataFrame(
            {
                'count': class_counts.astype(int),
                'fraction': class_fracs
            },
            index=unique_labels.astype(int),
        ).T

        if filename_class_balance_df is not None:
            if save_path is None:
                save_path = os.getcwd()

            results_df.to_csv(os.path.join(save_path, filename_class_balance_df))

        return results_df

    @staticmethod
    def plot_class_balance_df(
            class_balance_df: pd.DataFrame,
            dpi: int = 100,
            ax: Union[plt.Axes, None] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        num_classes = class_balance_df.shape[1]

        color_map = plt.cm.get_cmap("tab10" if num_classes <= 10 else "tab20", num_classes)
        colors = [color_map(i) for i in range(num_classes)]

        class_balance_df.loc['fraction'].plot(kind='bar', color=colors, ax=ax)

        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_title('Class Balance')

        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_balance_df.columns)

        class_fracs = class_balance_df.loc['fraction'].to_numpy()
        class_counts = class_balance_df.loc['count'].to_numpy()

        y_max = class_fracs.max() * 1.1
        ax.set_ylim(0, y_max)
        for idx, (frac, count) in enumerate(zip(class_fracs, class_counts)):
            text = f'total: {count}, frac: {round(frac, 4)}'
            text_offset = 0.05 * y_max
            y_text = frac + text_offset
            if y_text + 6 * text_offset > y_max:
                ax.text(
                    idx, y_text, text, ha='center', va='top', rotation=90)
            else:
                ax.text(
                    idx, y_text, text, ha='center', va='bottom', rotation=90)

        return ax

    # ### save_to_numpy_files() ########################################################################################
    def save_to_numpy_files(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            sample_wise: bool = False,
            save_path: Union[str, None] = None,
            filename_suffix: Union[str, None] = None,
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            shuffle: bool = True,
    ):

        if data_set == 'all':
            data_list = self.anndata_list_
        elif data_set == 'train':
            data_list = self.train_data_
        elif data_set == 'test':
            data_list = self.test_data_
        elif data_set == 'val':
            data_list = self.val_data_

            if data_list is None:
                if self._verbosity >= 1:
                    warnings.warn(
                        'No validation set was created when splitting the data. '
                        'Options are "train", "test", "all".',
                        UserWarning
                    )
                return
        else:
            raise ValueError("'data_set' must be 'all', 'train', 'test' or 'val'")

        FlowDataManager.save_to_numpy_files_worker(
            data_list=data_list,
            sample_wise=sample_wise,
            save_path=save_path,
            filename_suffix=filename_suffix,
            channels=channels,
            layer_key=layer_key,
            label_key=label_key,
            label_layer_key=label_layer_key,
            shuffle=shuffle,
        )

    @staticmethod
    def save_to_numpy_files_worker(
            data_list: List[sc.AnnData],
            sample_wise: bool = False,
            save_path: Union[str, None] = None,
            filename_suffix: Union[str, None] = None,
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            shuffle: bool = True,
    ):

        if save_path is None:
            save_path = os.getcwd()

        if filename_suffix is None:
            filename_suffix = ''

        # Save all data in one data matrix
        if not sample_wise:
            # Create dataloader
            dl = FlowDataManager.get_data_loader_worker(
                data_list=data_list,
                channels=channels,
                layer_key=layer_key,
                label_key=label_key,
                label_layer_key=label_layer_key,
                batch_size=-1,  # Return one matrix with all events
                shuffle=shuffle,
                return_data_loader='np_array',
                on_disk=False,  # No saving of np file on disk
                save_path=None,
                filename_np=None,
            )

            if label_key is not None:
                x, y = next(iter(dl))
                np.save(os.path.join(save_path, f'y{filename_suffix}.npy'), y.astype(int))
            else:
                x = next(iter(dl))

            np.save(os.path.join(save_path, f'x{filename_suffix}.npy'), x.astype(float))

        # Save data in sample-wise data matrices
        else:

            og_sample_names = [''] * len(data_list)
            new_sample_names = [''] * len(data_list)

            for i, adata in enumerate(data_list):

                # Save old and new filenames to list
                og_sample_names[i] = adata.uns['filename']

                new_fn = f'sample_{str(i).zfill(2)}{filename_suffix}.npy'
                new_sample_names[i] = new_fn

                # Create dataloader for just the current sample
                dummy_data_list = [adata, ]
                dummy_data_loader = FlowDataManager.get_data_loader_worker(
                    data_list=dummy_data_list,
                    channels=channels,
                    layer_key=layer_key,
                    label_key=label_key,
                    label_layer_key=label_layer_key,
                    batch_size=-1,
                    shuffle=shuffle,
                    return_data_loader='np_array',
                    on_disk=False,
                    save_path=None,
                    filename_np=None,
                )

                if label_key is not None:
                    x, y = next(iter(dummy_data_loader))
                    np.save(
                        os.path.join(save_path, 'y_' + new_fn),
                        y.astype(int)
                    )
                else:
                    x = next(iter(dummy_data_loader))
                np.save(
                    os.path.join(save_path, 'x_' + new_fn),
                    x.astype(float)
                )

            df = pd.DataFrame()
            df['og_sample_name'] = og_sample_names
            df['new_sample_name'] = new_sample_names
            df.to_csv(os.path.join(save_path, f'sample_names_mapping{filename_suffix}.csv'))

    # ### relabel_data() ###############################################################################################
    def relabel_data(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            old_to_new_label_mapping: Dict[Any, Any],  # Dict mapping old labels to new
            label_key: Union[int, str],
            label_layer_key: Union[str, None] = None,
            new_label_key: str = 'new_labels',  # New labels always added to .obs, this way no conflict with prepr
    ) -> None:

        if data_set == 'all':
            data_list = self.anndata_list_
        elif data_set == 'train':
            data_list = self.train_data_
        elif data_set == 'test':
            data_list = self.test_data_
        elif data_set == 'val':
            data_list = self.val_data_

            if data_list is None:
                if self._verbosity >= 1:
                    warnings.warn(
                        'No validation set was created when splitting the data. '
                        'Options are "train", "test", "all".',
                        UserWarning
                    )
                return
        else:
            raise ValueError("'data_set' must be 'all', 'train', 'test' or 'val'")

        FlowDataManager.relabel_data_worker(
            data_list=data_list,
            old_to_new_label_mapping=old_to_new_label_mapping,
            label_key=label_key,
            label_layer_key=label_layer_key,
            new_label_key=new_label_key,
            inplace=True,
        )

    @staticmethod
    def relabel_data_worker(
            data_list: List[sc.AnnData],
            old_to_new_label_mapping: Dict[Any, Any],  # Dict mapping old labels to new
            label_key: Union[int, str],
            label_layer_key: Union[str, None] = None,
            new_label_key: str = 'new_labels',  # New labels always added to .obs, this way no conflict with prepr
            inplace: bool = False,
    ) -> Union[List[sc.AnnData], None]:

        # Copy data_list if not inplace
        if not inplace:
            data_list = copy.deepcopy(data_list)

        for i, adata in enumerate(data_list):

            # Get labels (by column index, column name, obs key)
            labels = FlowDataManager._get_labels(adata=adata, label_key=label_key, layer_key=label_layer_key)

            # Map old to new labels
            labels_series = pd.Series(labels)
            new_labels = labels_series.map(old_to_new_label_mapping).to_numpy()

            # Add new labels to .obs
            adata.obs[new_label_key] = new_labels

        return None if inplace else data_list


