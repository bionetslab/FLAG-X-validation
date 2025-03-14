
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import pytometry as pm
import copy
import os
import warnings

from typing import Sequence, Tuple, Union, Literal, List, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from math import floor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from .flowdataset import FlowDataset
from .flowdataloader import FlowDataLoader

# Todo:
#  - Add documentation
#  - Add downsampling/balancing functionality -> save ds in separate data list

class FlowDataManager:
    def __init__(
            self,
            data_file_names: List[str],
            data_file_type: Union[Literal['fcs', 'csv'], None] = None,
            data_file_path: Union[str, None] = None,
            save_path: Union[str, None] = None,
            memory_saving: bool = False,
            verbosity: int = 0,
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

        if not isinstance(memory_saving, bool):
            raise ValueError("'memory_saving' must be a boolean value")

        if not isinstance(verbosity, int) or verbosity < 0:
            raise ValueError("'verbosity' must be an integer >= 0")


        # ### Set path variables for data loading and storage
        self._data_file_names = data_file_names  # List of filenames that should be loaded
        self._data_file_type = data_file_type  # If None guessed from file ending of 1st filename, assume all have same type

        self._data_file_path = data_file_path if data_file_path is not None else os.getcwd()  # Path to .fcs/.csv
        self._save_path = save_path if save_path is not None else os.path.join(os.getcwd(), 'data_handling') # Path to save any results to
        os.makedirs(self._save_path, exist_ok=True)
        
        self._memory_saving = memory_saving  # Whether to load and save to disk (.h5ad) one by one

        self._verbosity = verbosity

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
    def memory_saving(self):
        """Read-only property for memory saving."""
        return self._memory_saving

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
                warnings.warn(
                    f"Skipping invalid file '{fn}'. It is of type '{ft}' but should be '{self._data_file_type}'.",
                    UserWarning
                )
                self.invalid_files_.append(fn)
                continue

            # Load data file to anndata
            if self._data_file_type == 'fcs':  # data_file_type is fcs
                adata = pm.io.read_fcs(os.path.join(self._data_file_path, fn))
            else: # data_file_type is csv
                df = pd.read_csv(os.path.join(self._data_file_path, fn))
                adata = sc.AnnData(X=df.to_numpy())
                adata.var_names = df.columns.copy()

            # Annotate filename in uns of anndata
            adata.uns['filename'] = fn

            if not self._memory_saving:  # Keep all anndata objects in memory
                self.anndata_list_.append(adata)
            else:  # Save to .h5ad and store filename
                ad_fn = fn[:-4] + '.h5ad'
                adata.write_h5ad(filename=Path(os.path.join(self._save_path, ad_fn)))
                self.anndata_list_.append(ad_fn)
                del adata

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
            out_filename: Union[str, None] = None,
    ):
        self.sample_sizes_ = FlowDataManager.check_sample_sizes_worker(
            data_list=self.anndata_list_,
            file_path=self._save_path,
            save_path=self._save_path,
            out_filename=out_filename,
            verbosity=self._verbosity,
        )

    @staticmethod
    def check_sample_sizes_worker(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            file_path: Union[str, None] = None,
            save_path: Union[str, None] = None,
            out_filename: Union[str, None] = None,
            verbosity: int = 0,
    ):

        # ### Set flag whether to load data or not
        # (depending on list of AnnData objects or filenames of .h5ad files being passed)
        load_data = isinstance(data_list[0], str)

        # ### Inspect the number of samples and their sample size
        sn = []
        ss = []
        for i in range(len(data_list)):
            if load_data:
                # Load AnnData object, change channel names, save again
                fldata = sc.read_h5ad(os.path.join(file_path, data_list[i]))
            else:
                fldata = data_list[i]

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

        if out_filename is not None and save_path is not None:
            df.to_csv(os.path.join(save_path, out_filename))

        if verbosity >= 1:
            print(f'# ### Sample sizes:\n{df}')

        return df

    # ### align_channel_names(), check_og_channel_names_df() ###########################################################
    def align_channel_names(
            self,
            reference_channel_names: Union[int, dict, None] = None,
            out_filename: Union[str, None] = None,
    ) -> None:
        log_df = FlowDataManager.align_channel_names_worker(
            data_list=self.anndata_list_,  # Work on anndata_list
            reference=reference_channel_names,  # Int = idx of anndata_list or dict: {og_cn: new_cn}, None = 1st entry of list as reference
            inplace=True,  # Work inplace, change anndata_list
            file_path=self._save_path,  # If .h5ad files need to be loaded
            save_path=self._save_path,  # Where to save log_df to
            out_filename=out_filename,  # Filename for log df, None then no saving
        )
        self.og_channel_names_ = log_df
        self.check_og_channel_names_df()

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

        if out_filename is not None and save_path is not None:
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
            file_path=self._save_path,
            inplace=True,
            save_raw_to_layer=save_raw_to_layer,
            **kwargs
        )

    @staticmethod
    def sample_wise_preprocessing_worker(
            data_list: Union[Sequence[sc.AnnData], Sequence[str]],
            flavour: Literal['logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'custom'],
            file_path: Union[str, None] = None,  # Only necessary if .h5ad is to be loaded
            inplace: bool = False,
            save_raw_to_layer: Union[str, None] = None,  # Key for layer where raw data is to be stored
            **kwargs,
    ) -> Union[Sequence[sc.AnnData], Sequence[str], None]:

        if flavour not in {'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'custom'}:
            raise ValueError(
                "'flavour' must be one of: 'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff' or 'custom'")

        if not inplace:
            data_list = copy.deepcopy(data_list)

        load_data = isinstance(data_list[0], str)

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

        for d in data_list:
            if load_data:
                dummydata = sc.read_h5ad(os.path.join(file_path, d))
            else:
                dummydata = d

            # Store unprocessed data matrix in layer
            if save_raw_to_layer is None:
                dummydata.layers['original'] = dummydata.X.copy()
            else:
                dummydata.layers[save_raw_to_layer] = dummydata.X.copy()

            if kwargs:
                trafo_fct(adata=dummydata, **kwargs)
            else:
                trafo_fct(adata=dummydata)

            if load_data:
                dummydata.write_h5ad(os.path.join(file_path, d))
                del dummydata

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
            save: bool = False,
            **kwargs,
    ) -> None:

        dummy_data_split = FlowDataManager.perform_data_split_worker(
            data_list=self.anndata_list_,
            data_split=data_split,
            save=save,
            save_path=self._save_path,
            save_filename='data_split.csv',
            verbosity=self._verbosity,
            **kwargs
        )

        if len(dummy_data_split) == 2:
            self.train_data_, self.test_data_ = dummy_data_split
        else:
            self.train_data_, self.val_data_, self.test_data_ = dummy_data_split

    @staticmethod
    def perform_data_split_worker(
            data_list: Union[Sequence[str], Sequence[sc.AnnData]],
            data_split: Union[Tuple[float, float], Tuple[float, float, float], pd.DataFrame],
            save: bool = False,
            save_path: Union[str, None] = None,
            save_filename: Union[str, None] = None,
            verbosity: int = 0,
            **kwargs  # kwargs for sklearn are: random_state, shuffle, stratify
    ) -> Tuple[Union[Sequence[str], Sequence[sc.AnnData]], ...]:

        # ### Split according to fractions passed as tuple
        if not isinstance(data_split, pd.DataFrame):
            if len(data_split) not in {2, 3}:
                raise ValueError(
                    "'data_split' must be tuple or triple corresponding with fractions for train- (val-) and test-data")

            if sum(data_split) != 1 or any(x < 0 for x in data_split):
                raise ValueError(
                    'The train-(val-)test-split must be passed as a tuple of non negative decimals that sum to one')

            if len(data_split) == 2:
                train_data, test_data = train_test_split(
                    data_list, test_size=data_split[1], train_size=data_split[0], **kwargs)
                if save:
                    save_kwargs = {'filename': save_filename, 'filepath': save_path}
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
                    save_kwargs = {'filename': save_filename, 'filepath': save_path}
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
            save_kwargs: Dict,
    ):

        filename = save_kwargs.get('filename', None)
        if filename is None:
            'data_split.csv'

        filepath = save_kwargs.get('filepath', None)
        if filepath is None:
            filepath = os.getcwd()

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
            filename: str = 'data.npy',
            **kwargs,
    ) -> Union[DataLoader, None]:

        if data_set not in {'all', 'train', 'test', 'val'}:
            raise ValueError("'data_set' must be 'all', 'train', 'test' or 'val'")

        if data_set == 'all':
            data_list = self.anndata_list_
        elif data_set == 'train':
            data_list = self.train_data_
        elif data_set == 'test':
            data_list = self.test_data_
        elif data_set == 'val':
            try:
                data_list = self.val_data_
            except NameError:
                warnings.warn(
                    'No validation set was created when splitting the data. Options are "train", "test", "all"',
                    UserWarning
                )
                return
        else:
            data_list = []

        out = FlowDataManager.get_data_loader_worker(
            data_list=data_list,
            save_path=self._save_path,
            channels=channels,
            layer_key=layer_key,
            label_key=label_key,
            label_layer_key=label_layer_key,
            batch_size=batch_size,
            shuffle=shuffle,
            return_data_loader=return_data_loader,
            on_disk=on_disk,
            filename=filename,
            verbosity=self._verbosity,
            **kwargs,
        )

        return out

    @staticmethod
    def get_data_loader_worker(
            data_list: Union[Sequence[str], Sequence[sc.AnnData]],
            save_path: Union[str, None] = None,  # Where data was saved and where .npy files are to be saved if 'on_disk' is True
            channels: Union[Sequence[int], Sequence[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            batch_size: int = -1,
            shuffle: bool = True,
            return_data_loader: Literal['np_array', 'torch_tensor'] = 'np_array',
            on_disk: bool = False,
            filename: str = 'data.npy',
            verbosity: int = 0,
            **kwargs
    ) -> DataLoader:

        load_data = isinstance(data_list[0], str)

        if load_data and save_path is None:
            raise ValueError(
                "If 'data_list' is a list of filenames 'save_path' (= dir where files are stored) cannot be None"
            )

        if on_disk and save_path is None:
            raise ValueError(
                "If 'on_disk' is True 'save_path' (= dir where the dataloader data file is stored) cannot be None"
            )

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
                if layer_key is None:
                    labels = adata.X[:, label_key].copy()
                else:
                    labels = adata.layers[layer_key][:, label_key].copy()
            except IndexError:
                raise ValueError("'label_key' index is out of bounds in .X/.layers[layer_key] matrix")
        else:
            try:
                label_idx = adata.var_names.get_loc(label_key)
                if layer_key is None:
                    labels = adata.X[:, label_idx].copy()
                else:
                    labels = adata.layers[layer_key][:, label_idx].copy()
            except KeyError:
                warnings.warn(f"'label_key' not found in .var_names, trying .obs")
                try:
                    labels = adata.obs[label_key].copy()
                except KeyError:
                    raise ValueError("'label_key' not found in .obs or .var_names")
        return labels

    # ### over_under_sample_data_list() ################################################################################
    def sample_wise_stratified_downsampling(
            self
    ):
        # Todo
        return

    @staticmethod
    def sample_wise_stratified_downsampling_worker(
            sampling_strategy: str,
            data_list: Union[Sequence[str], Sequence[sc.AnnData]],
            data_path: Union[str, None] = None,  # Where data should be loaded from if data_list is list of filenames
            save_path: Union[str, None] = None, # Where data is saved to if data_list is list of filenames

    ) -> Union[Sequence[str], Sequence[sc.AnnData]]:

        load_data = isinstance(data_list[0], str)

        if load_data and data_path is None:
            raise ValueError(
                "If 'data_list' is a list of filenames 'data_path' (= dir where files are stored) cannot be None"
            )

        if load_data and save_path is None:
            save_path = data_path
            warnings.warn("'save_path' is None, saving to 'data_path', original data may be overwritten.", UserWarning)

        for d in data_list:
            if load_data:
                dummydata = sc.read_h5ad(os.path.join(data_path, d))
            else:
                dummydata = d


        return


    # ### check_class_balance() ########################################################################################
    @staticmethod
    def check_class_balance(
            data_loader: DataLoader,
            save: bool = False,
            plot: bool = False,
            save_path: Union[str, None] = None,
            filename_df: Union[str, None] = None,
            filename_plot: Union[str, None] = None,
            ax: Union[plt.Axes, None] = None,
            verbosity: int = 0,
    ):
        label_array = np.array([])
        for _, y in data_loader:
            label_array = np.concatenate((label_array, y), axis=0)
        label_series = pd.Series(label_array)
        class_counts = label_series.value_counts()
        class_fracs = label_series.value_counts(normalize=True)

        if verbosity >= 1:
            print(f'# ### Absolute counts for the labels:\n{class_counts}')
            print(f'# ### Relative frequencies for the labels:\n{class_fracs}')

        if save:
            if save_path is None:
                save_path = os.getcwd()
            if filename_df is None:
                filename_df = 'class_balances.csv'

            result_df = pd.DataFrame({
                'count': class_counts.astype(str),
                'fraction': class_fracs
            }).T

            result_df.to_csv(os.path.join(save_path, filename_df))

        if plot:
            if save_path is None:
                save_path = os.getcwd()
            if filename_plot is None:
                filename_plot = 'class_balances.png'
            if ax is None:
                fig, ax = plt.subplots()
                save_plot = True
            else:
                save_plot = False

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

            if save_plot:
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, filename_plot))

    # ### data_list_to_numpy() #########################################################################################
    @staticmethod
    def datalist_to_numpy(
            data_list: Union[Sequence[str], Sequence[sc.AnnData]],
            sample_wise: bool = False,
            save_path: Union[str, None] = None,
            filename_suffix: Union[str, None] = None,
            data_path: Union[str, None] = None,  # If anndata are to be loaded
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

        if not sample_wise:
            # Create dataloader
            dl = FlowDataManager.get_data_loader_worker(
                data_list=data_list,
                save_path=save_path,
                channels=channels,
                layer_key=layer_key,
                label_key=label_key,
                label_layer_key=label_layer_key,
                shuffle=shuffle,
                batch_size=-1,  # Return one matrix with all events
                return_data_loader='np_array',
                on_disk=False,
                verbosity=0,
            )

            if label_key is not None:
                x, y = next(iter(dl))
                np.save(os.path.join(save_path, f'y{filename_suffix}.npy'), y.astype(int))
            else:
                x = next(iter(dl))

            np.save(os.path.join(save_path, f'x{filename_suffix}.npy'), x.astype(float))

        else:

            # Check whether data needs to be loaded
            load_data = isinstance(data_list[0], str)
            if load_data and save_path is None:
                raise ValueError(
                    "If 'data_list' is a list of filenames 'save_path' (= dir where files are stored) cannot be None"
                )

            og_sample_names = [''] * len(data_list)
            new_sample_names = [''] * len(data_list)

            for i, d in enumerate(data_list):

                # Load the anndata if necessary
                if load_data:
                    d = sc.read_h5ad(os.path.join(data_path, d))

                og_sample_names[i] = d.uns['filename']
                new_sample_names[i] = f'sample_{str(i).zfill(2)}.npy'

                # Create dataloader for just the current sample
                dummy_data_list = [d, ]
                dummy_data_loader = FlowDataManager.get_data_loader_worker(
                    data_list=dummy_data_list,
                    save_path=save_path,
                    layer_key=layer_key,
                    label_key=label_key,
                    label_layer_key=label_layer_key,
                    shuffle=shuffle,
                    channels=channels,
                    batch_size=-1,
                    return_data_loader='np_array',
                    on_disk=False,
                    verbosity=0,
                )

                if label_key is not None:
                    x, y = next(iter(dummy_data_loader))
                    np.save(
                        os.path.join(save_path, f'y{filename_suffix}_sample_{str(i).zfill(2)}.npy'),
                        y.astype(int))
                else:
                    x = next(iter(dummy_data_loader))
                np.save(
                    os.path.join(save_path, f'x{filename_suffix}_sample_{str(i).zfill(2)}.npy'),
                    x.astype(float))

                if load_data:
                    del d

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
            try:
                data_list = self.val_data_
            except NameError:
                warnings.warn(
                    'No validation set was created when splitting the data. Options are "train", "test", "all"',
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
            data_path=self._save_path
        )

    @staticmethod
    def relabel_data_worker(
            data_list: Union[List[str], List[sc.AnnData]],
            old_to_new_label_mapping: Dict[Any, Any],  # Dict mapping old labels to new
            label_key: Union[int, str],
            label_layer_key: Union[str, None] = None,
            new_label_key: str = 'new_labels',  # New labels always added to .obs, this way no conflict with prepr
            inplace: bool = False,
            data_path: Union[str, None] = None,  # If anndata are to be loaded
    ) -> Union[Union[List[str], List[sc.AnnData]], None]:

        # Copy data_list if not inplace
        if not inplace:
            data_list = copy.deepcopy(data_list)

        # Check whether data needs to be loaded
        load_data = isinstance(data_list[0], str)
        if load_data and data_path is None:
            raise ValueError(
                "If 'data_list' is a list of filenames 'save_path' (= dir where files are stored) cannot be None"
            )

        for i, adata in enumerate(data_list):

            # Load the anndata if necessary
            if load_data:
                adata = sc.read_h5ad(os.path.join(data_path, adata))

            # Get labels (by column index, column name, obs key)
            labels = FlowDataManager._get_labels(adata=adata, label_key=label_key, layer_key=label_layer_key)

            # Map old to new labels
            labels_series = pd.Series(labels)
            new_labels = labels_series.map(old_to_new_label_mapping).to_numpy()

            # Add new labels to .obs
            adata.obs[new_label_key] = new_labels

            # Save relabeled and add filename to data_list
            if load_data:
                if inplace:
                    ad_fn = adata.uns['filename'][:-4] + '.h5ad'
                else:
                    ad_fn = 'relabeled_' + adata.uns['filename'][:-4] + '.h5ad'

                adata.write_h5ad(filename=Path(os.path.join(data_path, ad_fn)))
                data_list[i] = ad_fn
                del adata

        return None if inplace else data_list


