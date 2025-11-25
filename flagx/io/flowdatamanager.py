
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import pytometry as pm
import copy
import os
import warnings
import gc

from typing import Tuple, List, Dict, Union, Any
from typing_extensions import Literal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from matplotlib import colormaps
from .flowdataset import FlowDataset
from .flowdataloaders import FlowDataLoaders


class FlowDataManager:
    """
    Manager class for loading, preprocessing, organizing, and exporting
    flow cytometry datasets stored as FCS or CSV files.

    The class wraps a complete data-management pipeline for cytometry workflows:
    - Load raw files into AnnData objects: load_data_files_to_anndata()
    - Inspect sample sizes: check_sample_sizes(), plot_sample_size_df()
    - Check class balance: check_class_balance(), plot_class_balance_df(
    - Relabel datasets: relabel_data()
    - Align channel names across samples: align_channel_names(), check_og_channel_names_df()
    - Normalize/transform channels: sample_wise_preprocessing()
    - Perform train/val/test splitting: perform_data_split()
    - Downsample samples (optional stratification): sample_wise_downsampling()
    - Create PyTorch/NumPy dataloaders: get_data_loader()
    - Export datasets to disk: save_to_numpy_files()

    Parameters:
            data_file_names (List[str]): List of input filenames to load.
            data_file_type (Literal['fcs', 'csv'] or None): Type of input files. If None, inferred from extension of the first file.
            data_file_path (str or None): Directory containing the raw files. Defaults to CWD.
            save_path (str or None): Output directory for any exported files. Defaults to CWD.
            verbosity (int): Logging level. 0: silent, 1: warnings, 2+: info/debug.

    Attributes:
        invalid_files_ (list or None): Filenames skipped due to incompatible type.
        anndata_list_ (list or None): List of loaded AnnData objects.
        sample_sizes_ (pd.DataFrame or None): Summary of sample sizes.
        og_channel_names_ (pd.DataFrame or None): Original channel names per file before alignment.
        train_data_ (list or None): Train split as a list of AnnData objects.
        val_data_ (list or None): Validation split as a list of AnnData objects.
        test_data_ (list or None): Test split as a list of AnnData objects.
    """
    def __init__(
            self,
            data_file_names: List[str],
            data_file_type: Union[Literal['fcs', 'csv'], None] = None,
            data_file_path: Union[str, None] = None,
            save_path: Union[str, None] = None,
            verbosity: int = 1,  # 0 = silent, 1 = warnings, 2 = info
    ):
        """
        Initializes the FlowDataManager.

        Parameters:
            data_file_names (List[str]): List of input filenames to load.
            data_file_type (Literal['fcs', 'csv'] or None): Type of input files. If None, inferred from extension of the first file.
            data_file_path (str or None): Directory containing the raw files. Defaults to CWD.
            save_path (str or None): Output directory for any exported files. Defaults to CWD.
            verbosity (int): Logging level. 0: silent, 1: warnings, 2+: info/debug.
        """
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
        self._save_path = save_path if save_path is not None else os.getcwd() # Path to save any results to
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
    def load_data_files_to_anndata(self) -> None:
        """
        Load all provided data files into AnnData objects.

        FCS files are read using `pytometry`, and CSV files are read with pandas
        before being wrapped into AnnData. Invalid files are skipped and recorded.

        Raises:
            ValueError: If file type cannot be inferred for the first file.
            UserWarning: When skipping incompatible file types.

        Returns:
            None
        """
        # Note: Fcd data is stored as float32 according to the Flow Cytometry Standard
        # - read_fcs() loads as float32
        # - read in .csv also as float32

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
        """
        Compute and optionally save a summary table with the number of events
        per dataset.

        Args:
            filename_sample_sizes_df (str or None): If provided, the summary dataframe is saved to this filename inside `save_path`.

        Returns:
            None: Results stored in `sample_sizes_`.
        """
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
        df.sort_values(by='n_events', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

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

    @staticmethod
    def plot_sample_size_df(
            sample_size_df: pd.DataFrame,
            dpi: int = 100,
            ax: Union[plt.Axes, None] = None,
    ) -> plt.Axes:
        """
        Plot a bar chart of sample sizes from a summary dataframe.

        Args:
            sample_size_df (pd.DataFrame): Output of `check_sample_sizes_worker`.
            dpi (int): Plot resolution if a new figure is created.
            ax (matplotlib.axes.Axes or None): Existing axes to plot into, or None to create new axes.

        Returns:
            matplotlib.axes.Axes: Axes containing the bar plot.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        # Extract info from dataframe
        y_vals = sample_size_df.loc[~sample_size_df['sample'].isin(['mean', 'std', 'total']), 'n_events'].to_numpy()
        x_vals = list(range(y_vals.shape[0]))

        m = sample_size_df.loc[sample_size_df['sample'] == 'mean', 'n_events'].squeeze()
        std = sample_size_df.loc[sample_size_df['sample'] == 'std', 'n_events'].squeeze()
        total = sample_size_df.loc[sample_size_df['sample'] == 'total', 'n_events'].squeeze()

        ax.bar(x_vals, y_vals, color='skyblue', edgecolor='black')

        ax.set_xlabel('Sample id')
        ax.set_ylabel('n events per sample')

        xtick_vals = np.linspace(min(x_vals), max(x_vals), 6, dtype=int)
        ax.set_xticks(xtick_vals)

        ax.axhline(m, color='red', linestyle='-', linewidth=2, label=f'Mean: {m:.2f}', alpha=0.6)

        ax.axhline(m - std, color='orange', linestyle='dashed', label=f'Std: {std:.2f}', linewidth=2, alpha=0.5)
        ax.axhline(m + std, color='orange', linestyle='dashed', linewidth=2, alpha=0.5)

        ax.legend()

        ax.set_title(f'n samples: {y_vals.shape[0]}, Total events: {int(total)}')

        return ax

    # ### align_channel_names(), check_og_channel_names_df() ###########################################################
    def align_channel_names(
            self,
            reference_channel_names: Union[int, dict, None] = None,
            filename_log_df: Union[str, None] = None,
    ) -> None:
        """
        Harmonize channel names across all samples by using a reference sample
        or a user-provided mapping.

        Channel names from each file are aligned so that all datasets have
        identical `var_names`. A log dataframe is stored to allow inspection of
        original channel names.

        Args:
            reference_channel_names (int, dict, or None):
                • int: index of the reference AnnData in `anndata_list_`.
                • dict: mapping {old_name: new_name}.
                • None: use the first sample as reference.
            filename_log_df (str or None): Filename to save the channel-name log dataframe. If None, no file is saved.

        Returns:
            None: Log dataframe stored in `og_channel_names_`.
        """
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
            reference: Union[int, dict, None] = None,  # Either int for which file to use as reference or a
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
        if reference is None:
            reference = 0

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
        """
        Validate consistency of original channel names before alignment.

        Checks whether each channel index had identical names across all samples.
        If inconsistencies are found, a warning is emitted.

        Returns:
            None
        """
        FlowDataManager.check_og_channel_names_df_worker(
            og_channel_names=self.og_channel_names_,
            verbosity=self.verbosity
        )

    @staticmethod
    def check_og_channel_names_df_worker(og_channel_names: pd.DataFrame, verbosity: int = 1) -> None:
        for i, col in enumerate(og_channel_names.columns):
            if col == 'filename':
                continue

            value_counts = og_channel_names[col].value_counts()

            if value_counts.size >= 2:
                msg = f'# ### The channel names for channel {i} were not consistent across samples\n'
                msg += f'# ### Channel: {i} ### #\n'
                for value, count in value_counts.items():
                    msg += f'# ### Name: {value}, Count: {count}\n'
                if verbosity >= 1:
                    warnings.warn(msg, UserWarning)
            else:
                if verbosity >= 2:
                    print(f'# ### Channel: {i}, Name: {value_counts.index[0]} is consistent across samples\n')

    # ### sample_wise_preprocessing() ##################################################################################
    def sample_wise_preprocessing(
            self,
            flavour: Literal[
                'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'log10_w_custom_cutoffs', 'custom'
            ] = 'arcsinh',
            save_raw_to_layer: Union[str, None] = None,
            **kwargs
    ) -> None:
        """
        Applies a per-sample preprocessing transformation to all AnnData objects.

        This method supports common cytometry transformations such as
        arcsinh, logicle, and biexponential scaling. Log10-based transformations
        require user-specified cutoffs, and fully custom preprocessing functions
        may also be supplied. For detailed documentation of the built-in
        transformation functions and their arguments, see:
        https://pytometry.netlify.app/api.

        Args:
            flavour (Literal['logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'log10_w_custom_cutoffs', 'custom']):
                The transformation type to apply. Options:
                - `'logicle'`, `'arcsinh'`, `'biexp'`: Apply the corresponding
                  cytometry scaling function. Parameters (e.g. cofactor of arcsinh) can be set via kwargs.
                - `'log10_w_cutoff'`: Requires a `cutoff` (float) passed via kwargs.
                - `'log10_w_custom_cutoffs'`: Requires `cutoffs` (dict mapping channel names to cutoff values) passed via kwargs.
                - `'custom'`: Expects a user-defined preprocessing callable passed as `preprocessing_method` via kwargs. The callable must modify the AnnData object in place.

            save_raw_to_layer (str or None):
                If provided, the raw (untransformed) data matrix of each AnnData object will be saved under `adata.layers[save_raw_to_layer]` before transformation.

            **kwargs:
                Additional arguments forwarded to the selected transformation function or to the custom preprocessing callable.

        Returns:
            None: The transformation is performed in place on each AnnData object.
        """

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
            flavour: Literal['logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'log10_w_custom_cutoffs', 'custom'],  # custom must work inplace
            inplace: bool = False,
            save_raw_to_layer: Union[str, None] = None,  # Key for layer where raw data is stored, if None no storage
            **kwargs,
    ) -> Union[List[sc.AnnData], None]:

        if flavour not in {'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'log10_w_custom_cutoffs', 'custom'}:
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
        elif flavour == 'log10_w_custom_cutoffs':
            if 'cutoffs' not in kwargs:
                raise ValueError(
                    "Missing required argument: 'cutoffs' (dict of {channel: cutoff}) must be provided in kwargs "
                    "for 'log10_w_custom_cutoffs' flavour."
                )
            trafo_fct = FlowDataManager.log10_w_custom_cutoffs
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
                trafo_fct(adata, **kwargs)
            else:
                trafo_fct(adata)

        if not inplace:
            return data_list

    @staticmethod
    def log10_w_cutoff(adata: sc.AnnData, cutoff: float = 100):
        """
        Apply a log10 transform to values above a cutoff and clamp smaller values.

        Args:
            adata (AnnData): Input AnnData object. Transformation is applied inplace.
            cutoff (float): Minimum value for the transform. Values ≤ cutoff are set to log10(cutoff).

        Returns:
            None
        """
        x = adata.X
        x = np.log10(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
        adata.X = x

    @staticmethod
    def log10_w_custom_cutoffs(
            adata: sc.AnnData,
            cutoffs: Dict[str, int],
    ):
        """
        Apply per-channel log10 transforms using custom cutoffs.

        Args:
            adata (AnnData): Input AnnData object modified inplace.
            cutoffs (dict): Mapping {channel_name: cutoff}. Values above the cutoff are log10-transformed; values below are clamped to log10(cutoff).

        Returns:
            None
        """
        x = adata.X.copy()
        for channel, cutoff in cutoffs.items():
            col_idx = np.where(adata.var_names == channel)[0][0]
            x_col = adata.X[:, col_idx].copy()
            mask = (x_col > cutoff)
            x_col[mask] = np.log10(x_col[mask])
            x_col[~mask] = np.log10(cutoff)
            x[:, col_idx] = x_col

        adata.X = x

    # ### perform_data_split() #########################################################################################
    def perform_data_split(
            self,
            data_split: Union[Tuple[float, float], Tuple[float, float, float], pd.DataFrame] = (0.75, 0.25),
            filename_data_split: Union[str, None] = None,
            **kwargs,
    ) -> None:
        """
        Split the dataset into train-test- or train-validation-test-sets.

        Splitting can be done in two ways:
            • By providing fractions (e.g., (0.7, 0.2, 0.1))
            • By passing a saved dataframe specifying each sample's assignment

        Args:
            data_split (tuple or pd.DataFrame): Fractions for train/(val)/test or a dataframe with columns `'filename'` and `'mode'`.
            filename_data_split (str or None): If provided, the split assignment is saved to this CSV inside `save_path`.
            **kwargs: Additional parameters passed to Sklearns's `train_test_split` such as `random_state`, `shuffle`, or `stratify`.

        Returns:
            None: Results stored in `train_data_`, `val_data_`, and `test_data_`.
        """

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
    ) -> Tuple[List[sc.AnnData], ...]:

        # ### Split according to fractions passed as tuple
        if not isinstance(data_split, pd.DataFrame):
            if len(data_split) not in {2, 3}:
                raise ValueError(
                    "'data_split' must be tuple or triple corresponding to fractions for train- (val-) and test-data")

            if not np.isclose(sum(data_split), 1.0) or any(x < 0 for x in data_split):
                raise ValueError(
                    'The train-(val-)test-split must be passed as a tuple of non negative decimals that sum to one')

            if filename_data_split is not None and save_path is None:
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

                # Compute number of samples in train, val and test set beforehand
                n = len(data_list)
                n_train = int(data_split[0] * n)
                n_val = int(data_split[1] * n)
                n_test = n - n_train - n_val

                # Check for stratification
                stratify = kwargs.pop('stratify', None)

                if stratify is not None:

                    # Split into train and val-test set
                    train_data, val_test_data, train_stratify, val_test_stratify = train_test_split(
                        data_list, stratify,
                        test_size=n_val + n_test,
                        train_size=n_train,
                        stratify=stratify,
                        **kwargs
                    )

                    # Split val-test data into val and test set
                    val_data, test_data = train_test_split(
                        val_test_data,
                        test_size=n_test / (n_val + n_test),
                        train_size=n_val / (n_val + n_test),
                        stratify=val_test_stratify,
                        **kwargs
                    )

                else:
                    # Split into train and val-test set
                    train_data, val_test_data = train_test_split(
                        data_list,
                        test_size=n_val + n_test,
                        train_size=n_train,
                        **kwargs
                    )

                    # Split val-test data into val and test set
                    val_data, test_data = train_test_split(
                        val_test_data,
                        test_size=n_test,
                        train_size=n_val,
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

    # ### sample_wise_downsampling() ###################################################################################
    def sample_wise_downsampling(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            target_num_events: Union[int, float],
            stratified: bool = False,
            label_key: Union[int, str, None] = None,
            # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
    ) -> None:
        """
        Downsample each sample in the specified dataset.

        Downsampling may be:
            • Uniform random (no stratification)
            • Stratified by class labels (requires `label_key`)

        Args:
            data_set (Literal['train', 'val', 'test', 'all']): Which subset to downsample.
            target_num_events (int or float): If ≥1: absolute number of events to retain. If <1: fraction of events to retain.
            stratified (bool): Whether to preserve class proportions via stratified sampling.
            label_key (int, str, or None): Label column for stratification (X column index, var name, or obs key).
            label_layer_key (str or None): Layer name if labels are stored in a layer instead of `.X`.

        Returns:
            None
        """

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
            target_num_events=target_num_events,
            stratified=stratified,
            label_key=label_key,
            label_layer_key=label_layer_key,
            inplace=True,
        )

    @staticmethod
    def sample_wise_downsampling_worker(
            data_list: List[sc.AnnData],
            target_num_events: Union[int, float],  # values < 1 will be interpreted as fractions
            stratified: bool = False,
            label_key: Union[int, str, None] = None,
            # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            inplace: bool = False,
    ) -> Union[List[sc.AnnData], None]:

        if target_num_events < 0:
            raise ValueError("'target_num_events' must be greater than 0")

        if target_num_events >= 1 and not isinstance(target_num_events, int):
            raise ValueError("'target_num_events' must be of type int if >= 1.")

        if stratified and label_key is None:
            raise ValueError("'stratified' is True but 'label_key' is None. Need labels for stratification.")

        if not inplace:
            data_list = copy.deepcopy(data_list)

        for i, adata in enumerate(data_list):
            # If target_num_events is < 1 interpret as fraction
            tne = target_num_events if target_num_events >= 1 else round(adata.n_obs * target_num_events)
            # Get labels or number of events
            if stratified:
                y = FlowDataManager._get_labels(adata=adata, label_key=label_key, layer_key=label_layer_key)
            else:
                y = adata.n_obs
            # Get bool indicating which events to keep
            ds_bool = FlowDataManager._get_downsampling_bool(y=y, target_num_events=tne, stratified=stratified)
            # Update data_list
            data_list[i] = adata[ds_bool, :].copy()

        if not inplace:
            return data_list

    @staticmethod
    def _get_downsampling_bool(
            y: Union[np.ndarray, int],
            target_num_events: int,
            stratified: bool = False
    ) -> np.ndarray:

        if not (isinstance(y, int) or isinstance(y, np.ndarray)):
            raise ValueError("'y' must be int (no stratification) or numpy array (stratification).")

        if isinstance(y, int) and stratified:
            raise ValueError('y must be an array of labels and not an integer')

        if isinstance(y, int) and not stratified:
            num_events = y
        else:
            num_events = y.shape[0]

        keep_mask = np.zeros(num_events, dtype=bool)

        if target_num_events >= num_events:
            keep_mask[:] = True

        elif stratified:

            unique_labels, counts = np.unique(y, return_counts=True)
            selected_indices = []

            for label, count in zip(unique_labels, counts):

                # Get the number of events of this class to keep
                target_num_events_class = int(round(target_num_events * (count / num_events)))
                target_num_events_class = min(target_num_events_class, count)

                # Get the indices where y == class
                class_indices = np.where(y == label)[0]

                # Randomly draw from the indices and append to list
                if target_num_events_class > 0:
                    selected = np.random.choice(class_indices, target_num_events_class, replace=False)
                    selected_indices.extend(selected)

            # Adjust the number of events to the exact desired number (account for rounding errors)
            if len(selected_indices) > target_num_events:
                selected_indices = np.random.choice(selected_indices, target_num_events, replace=False)
            elif len(selected_indices) < target_num_events:
                remaining_unselected_events = np.setdiff1d(np.arange(num_events), selected_indices)
                additional_events = np.random.choice(
                    remaining_unselected_events, target_num_events - len(selected_indices), replace=False
                )
                selected_indices.extend(additional_events)

            # Set mask to True for selected events
            keep_mask[selected_indices] = True


        else:

            # Select events by drawing uniform at random without replacement
            selected_indices = np.random.choice(np.arange(num_events), target_num_events, replace=False)

            # Set mask to True for selected events
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
        """
        Compute class frequency and counts for a dataset subset.

        Args:
            data_set (Literal['train', 'val', 'test', 'all']): Subset to analyze.
            label_key (int or str): Location of labels (X column index, var name, or obs key).
            label_layer_key (str or None): Layer key if labels are stored in a layer.
            filename_class_balance_df (str or None): Optional output file for saving the class-balance dataframe.

        Returns:
            pd.DataFrame or None: Dataframe with columns 'count' and 'fraction' and labels in index. None if the specified subset does not exist.
        """

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
        """
        Plot absolute and relative class frequencies as a bar chart.

        Args:
            class_balance_df (pd.DataFrame): Output from `check_class_balance` containing 'count' and 'fraction' and labels in index.
            dpi (int): Resolution of the figure when creating a new plot.
            ax (Axes or None): Matplotlib axis to plot into. Creates a new figure if None.

        Returns:
            matplotlib.axes.Axes: The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        num_classes = class_balance_df.shape[1]

        cmap_name = 'tab10' if num_classes <= 10 else 'tab20'
        color_map = colormaps[cmap_name].resampled(num_classes)
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

    # ### get_data_loader() ############################################################################################
    def get_data_loader(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            channels: Union[List[int], List[str], None] = None,
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
        """
        Construct a dataloader for the selected dataset split.

        This method concatenates samples, extracts requested channels, appends labels
        (optional), and returns a PyTorch dataloader that returns either PyTorch Tensors or Numpy arrays.

        Args:
            data_set (Literal['train', 'val', 'test', 'all']): Subset from which to load data.
            channels (list[int] or list[str] or None): Which channels (features) to extract. Defaults to all channels.
            layer_key (str or None): Layer key if data should come from a layer instead of `.X`.
            label_key (int, str, or None): Location of labels: X column index, var name, or obs key. If None, no labels are added.
            label_layer_key (str or None): Layer key if labels are stored in a layer instead of `.X`.
            batch_size (int): Batch size. -1 loads all data at once.
            shuffle (bool): Whether to shuffle samples each epoch.
            return_data_loader (Literal['np_array', 'torch_tensor']): Output format of the dataloader.
            on_disk (bool): If True, data is first saved to disk as a .npy file and loaded lazily in memory-mapped mode.
            filename_np (str or None): Filename for on-disk storage when `on_disk=True`.
            **kwargs: Additional arguments forwarded to `FlowDataLoaders` and in term PyTorch DataLoader.

        Returns:
            DataLoader or None:
                The prepared dataloader, or None if the chosen subset is unavailable.
        """

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
            channels: Union[List[int], List[str], None] = None,
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
        flow_dataloader = FlowDataLoaders(
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
            channels: Union[List[int], List[str]],
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

        return label_array.astype(int)

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
        return labels.astype(int)

    # ### save_to_numpy_files() ########################################################################################
    def save_to_numpy_files(
            self,
            data_set: Literal['train', 'val', 'test', 'all'],
            sample_wise: bool = False,
            save_path: Union[str, None] = None,
            filename_suffix: Union[str, None] = None,
            channels: Union[List[int], List[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            shuffle: bool = True,
            precision: Literal['16bit', '32bit', '64bit'] = '32bit',
    ):
        """
        Export data to .npy files in either combined or per-sample format.

        The exported data matrices may optionally include labels and may be stored
        with user-selected numeric precision. Files are placed in `save_path`.

        Args:
            data_set (Literal['train', 'val', 'test', 'all']): Which subset to export.
            sample_wise (bool): If False: save all data as a single matrix. If True: save one file per sample.
            save_path (str or None): Output directory. Defaults to the manager's `save_path`.
            filename_suffix (str or None): Optional suffix appended to output filenames.
            channels (list[int] or list[str] or None): Channels to export; defaults to all.
            layer_key (str or None): Which layer to export; defaults to `.X`.
            label_key (int, str, or None): If provided, labels are appended or saved separately.
            label_layer_key (str or None): Layer containing labels, if not `.X`.
            shuffle (bool): Whether to shuffle events before saving.
            precision (Literal['16bit', '32bit', '64bit']): Numeric precision for output arrays.

        Returns:
            None
        """

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
            precision=precision,
        )

    @staticmethod
    def save_to_numpy_files_worker(
            data_list: List[sc.AnnData],
            sample_wise: bool = False,
            save_path: Union[str, None] = None,
            filename_suffix: Union[str, None] = None,
            channels: Union[List[int], List[str], None] = None,
            layer_key: Union[str, None] = None,
            label_key: Union[int, str, None] = None,  # .obs key or varname or var index, if none is passed -> just data
            label_layer_key: Union[str, None] = None,
            shuffle: bool = True,
            precision: Literal['16bit', '32bit', '64bit'] = '32bit',
    ):

        if precision == '16bit':
            float_prec = np.float16
            int_prec = np.int16

        elif precision == '32bit':
            float_prec = np.float32
            int_prec = np.int32

        elif precision == '64bit':
            float_prec = np.float64
            int_prec = np.int64

        else:
            raise ValueError("'precision' must be '16bit', '32bit', '64bit'")


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
                np.save(os.path.join(save_path, f'y{filename_suffix}.npy'), y.astype(int_prec))
            else:
                x = next(iter(dl))

            np.save(os.path.join(save_path, f'x{filename_suffix}.npy'), x.astype(float_prec))

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
                        y.astype(int_prec)
                    )
                else:
                    x = next(iter(dummy_data_loader))
                np.save(
                    os.path.join(save_path, 'x_' + new_fn),
                    x.astype(float_prec)
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
        """
        Apply a mapping from old to new labels for all samples in a dataset.

        The new labels are always written to `.obs[new_label_key]` to avoid
        interference with existing preprocessing, layers, or var-based labels.

        Args:
            data_set (Literal['train', 'val', 'test', 'all']): Which data subset to relabel.
            old_to_new_label_mapping (dict): Dictionary mapping old labels to new labels.
            label_key (int or str): Location of original labels (X column index, var name, or obs key).
            label_layer_key (str or None): If labels are stored in a layer instead of `.X`.
            new_label_key (str): Name of the new label field added to `.obs`.

        Returns:
            None
        """

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


