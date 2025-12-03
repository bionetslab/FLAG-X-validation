

def main_data_processing():
    """
    Script for loading the .fcs files of the datasets, perform data split on sample level,
    apply transformation, subset to channels chosen for model training, save resulting data matrices to .npy files.
    Additionally, save data from training and test samples respectively concatenated in one matrix.

    Returns:
        None
    """

    import os
    import copy
    import re
    import random
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import List, Dict, Union, Tuple
    from flagx.io import FlowDataManager


    def data_processing_workflow(
            dataset_names: List[str],
            raw_data_paths: List[str],
            data_file_types: List[str],
            label_keys: List[str],
            relabel_dicts: List[Union[Dict[int, int], None]],
            cutoff_dicts: List[Union[Dict[str, int], None]],
            data_splits: List[Union[Tuple[float, float], pd.DataFrame]],
            channel_names: List[List[str]],
    ):

        for dn, rdp, dft, lk, rd, cd, ds, cn,   in zip(
                dataset_names, raw_data_paths, data_file_types,
                label_keys, relabel_dicts,
                cutoff_dicts,
                data_splits, channel_names,
        ):
            for pf in ['arcsinh', 'log10'] :

                # Set random seeds for random, numpy and torch
                seed = 42
                torch.manual_seed(seed)  # Sets the seed for generating random numbers on all devices.
                torch.cuda.manual_seed_all(seed)  # Set the seed for generating random numbers on all GPUs.
                # torch.cuda.manual_seed(seed)  # Set the seed for generating random numbers for the current GPU.
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False
                np.random.seed(seed)
                random.seed(seed)

                # Set label key
                current_lk = lk

                # Define path where results should be stored and create dir
                if pf == 'arcsinh':
                    pf_str = 'arcsinh_cofactor150'
                else:  # log10
                    if cd is not None:
                        pf_str = 'log10_w_custom_cutoffs'
                    else:
                        pf_str = 'log10_cutoff100'

                np_data_p = os.path.join(
                    os.getcwd(),
                    f'data/np_files/{dn}/{pf_str}'
                )
                os.makedirs(os.path.join(np_data_p, 'data_handling'), exist_ok=True)

                # ### Load and process the data set
                # Create a list of the filenames
                filename_list = sorted(os.listdir(rdp))

                # Instantiate the FlowDataManager
                fdm = FlowDataManager(
                    data_file_names=filename_list,
                    data_file_type=dft,
                    data_file_path=rdp,
                    save_path=os.path.join(np_data_p, 'data_handling'),
                    verbosity=2,
                )

                # Load data files to anndata
                fdm.load_data_files_to_anndata()

                # Check the number of events per sample
                fdm.check_sample_sizes(filename_sample_sizes_df='sample_sizes.csv')
                fdm.plot_sample_size_df(sample_size_df=fdm.sample_sizes_, dpi=300, ax=None)
                plt.tight_layout()
                plt.savefig(os.path.join(fdm.save_path, 'sample_sizes_df.png'))

                # Align channel names
                fdm.align_channel_names(
                    reference_channel_names=0,  # Use 1st anndata in data list as reference
                    filename_log_df='og_channel_names.csv',
                )

                # Relabel data
                if rd is not None:
                    new_label_key = 'new_labels'
                    fdm.relabel_data(
                        data_set='all',
                        old_to_new_label_mapping=rd,
                        label_key=current_lk,
                        label_layer_key=None,  # No preprocessing done yet
                        new_label_key=new_label_key,
                    )
                    current_lk = new_label_key

                # Apply sample wise preprocessing transformation
                if pf == 'arcsinh':
                    prepr_kwargs = {'cofactor': 150}
                else:  # log10
                    if cd is not None:  # If passed, apply channel wise cutoffs ...
                        pf = 'log10_w_custom_cutoffs'
                        prepr_kwargs = {
                            'cutoffs': cd,
                        }
                    else:  # ... otherwise log10 with cutoff 100
                        pf = 'log10_w_cutoff'
                        prepr_kwargs = {'cutoff': 100}

                fdm.sample_wise_preprocessing(flavour=pf, save_raw_to_layer='no_trafo', **prepr_kwargs)

                # Split samples into train and test split (stratified for lymphoma datasets)
                split_kwargs = {'shuffle': True, 'random_state': seed}
                # Extract lymphoma subtype from filename for stratification
                if dn not in {'imstat', 'flowcyt'}:
                    pattern = re.compile(r'^\d+_([A-Za-z]+)_')
                    lymphoma_subtypes = [pattern.search(f).group(1) for f in filename_list]
                    split_kwargs['stratify'] = lymphoma_subtypes

                fdm.perform_data_split(
                    data_split=copy.deepcopy(ds),
                    filename_data_split='data_split.csv',
                    **split_kwargs
                )

                # Check class balance of train and test set
                cb_df_train = fdm.check_class_balance(
                    data_set='train',
                    label_key=current_lk,
                    label_layer_key='no_trafo',
                    filename_class_balance_df='class_balance_train.csv',
                )
                fdm.plot_class_balance_df(class_balance_df=cb_df_train, dpi=300)
                plt.savefig(os.path.join(fdm.save_path, 'class_balance_train.png'))

                cb_df_test = fdm.check_class_balance(
                    data_set='test',
                    label_key=current_lk,
                    label_layer_key='no_trafo',
                    filename_class_balance_df='class_balance_test.csv',
                )
                fdm.plot_class_balance_df(class_balance_df=cb_df_test, dpi=300)
                plt.savefig(os.path.join(fdm.save_path, 'class_balance_test.png'))

                # Save data to numpy files
                for sw in [False, True]:
                    for mode in ['train', 'test']:
                        dummy_save_path = os.path.join(np_data_p, '' if not sw else f'sample_wise_{mode}')
                        os.makedirs(dummy_save_path, exist_ok=True)
                        fdm.save_to_numpy_files(
                            data_set=mode,
                            sample_wise=sw,
                            save_path=dummy_save_path,
                            filename_suffix=f'_{mode}',
                            channels=cn,
                            layer_key=None,  # Use adata.X, transformed data
                            label_key=current_lk,
                            label_layer_key='no_trafo',  # Use labels from original untransformed data
                            shuffle=True,  # Used pytorch dataloader under the hood, seed is set beforehand
                            precision='32bit',  # Same as FCS
                        )

    # ### Set flags and important variables  ###########################################################################

    # --- Immune status ---
    ds_name_imstat = 'imstat'
    raw_data_p_imstat = os.path.join(os.getcwd(), 'data/raw/imstat')
    data_file_type_imstat = 'fcs'
    data_split_imstat = pd.read_csv(
        os.path.join(os.getcwd(), 'data/raw/imstat_data_split_development.csv'),
    )
    data_split_imstat['filename'] = data_split_imstat['filename'].apply(lambda x: f'{x}.fcs')

    channels_imstat = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    cutoff_dict_imstat = {
        'FS INT': 100000, 'SS INT': 20000, '16-FITC': 250, '56-PE': 450, '3-ECD': 700,
        '4-PC7': 1200,
        '19-APC': 1700,
        '14-APC700': 900,
        '8-PB': 450, '45-CO': 500
    }
    label_key_imstat = 'population'
    relabel_dict_imstat = None

    # --- Lymphoma tube 1 ---
    ds_name_lt1 = 'lymphoma_tube1'
    raw_data_p_lt1 = os.path.join(
        os.getcwd(),
        'data/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels'
    )
    data_file_type_lt1 = 'fcs'
    data_split_lt1 = (0.6, 0.4)
    channels_lt1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7',
        'CD38_APC', 'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    cutoff_dict_lt1 = {
        'FS': 100000, 'SS': 20000, 'kappavCD8_FITC': 500, 'lambdavCD7_PE': 400, 'CD23_ECD': 500,
        'CD79bvCD4_PC5.5': 1200, 'CD5_PC7': 300, 'CD38_APC': 700,
        'CD19_APC_A700': 150,
        'CD20vCD3_APC_A750': 500, 'FMC7vCD2_PB': 500, 'CD45_KrOr': 1000
    }
    label_key_lt1 = 'population'
    relabel_dict_lt1 = None


    # --- Lymphoma tube 1, binary case ---
    ds_name_lt1_binary = 'lymphoma_tube1_binary'
    raw_data_p_lt1_binary = raw_data_p_lt1
    data_file_type_lt1_binary = data_file_type_lt1
    data_split_lt1_binary = data_split_lt1
    channels_lt1_binary = channels_lt1
    cutoff_dict_lt1_binary = cutoff_dict_lt1
    label_key_lt1_binary = label_key_lt1
    relabel_dict_lt1_binary = {1: 1, 9: 1, 7: 0, 10: 0}
    # Map:
    # - class 1, 9 (B cells, B cells outside FSSS gate (dying B cells)) to label 1 (positive)
    # - class 7, 10 (CD45 negative cells (mostly erythrocytes), other cells) to label 0 (negative)


    # --- Lymphoma tube 2 ---
    ds_name_lt2 = 'lymphoma_tube2'
    raw_data_p_lt2 = os.path.join(
        os.getcwd(),
        'data/raw/lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels'
    )
    data_file_type_lt2 = 'fcs'
    data_split_lt2 = (0.6, 0.4)
    channels_lt2 = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7',
        'CD52_APC', 'CD11c_APC_A700', 'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]
    cutoff_dict_lt2 = {
        'FS': 100000, 'SS': 20000, 'CD103_FITC': 300, 'CD43_PE': 1500, 'CD25_ECD': 1000,
        'CD10_PC5.5': 1000, 'CD200_PC7': 1000, 'CD52_APC': 150, 'CD11c_APC_A700': 200,
        'CD20_APC_A750': 300, 'IgM_PB': 400, 'CD19_KrOr': 200,
    }
    label_key_lt2 = 'population'
    relabel_dict_lt2 = None


    # --- Lymphoma tube 2, binary case ---
    ds_name_lt2_binary = 'lymphoma_tube2_binary'
    raw_data_p_lt2_binary = raw_data_p_lt2
    data_file_type_lt2_binary = data_file_type_lt2
    data_split_lt2_binary = data_split_lt2
    channels_lt2_binary = channels_lt2
    cutoff_dict_lt2_binary = cutoff_dict_lt2
    label_key_lt2_binary = label_key_lt2
    relabel_dict_lt2_binary = {1: 1, 9: 1, 7: 0, 10: 0}
    # Map:
    # - class 1, 9 (B cells, B cells outside FSSS gate (dying B cells)) to label 1 (positive)
    # - class 7, 10 (CD45 negative cells (mostly erythrocytes), other cells) to label 0 (negative)


    # --- Flowcyt ---
    ds_name_flowcyt = 'flowcyt'
    raw_data_p_flowcyt = os.path.join(os.getcwd(), 'data/raw/flowcyt/data_original')
    data_file_type_flowcyt = 'csv'
    data_split_flowcyt = (0.6, 0.4)
    channels_flowcyt = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]
    cutoff_dict_flowcyt = None
    label_key_flowcyt = 'label'
    relabel_dict_flowcyt = None

    # Define lists of parameters
    dataset_names_input = [
        ds_name_imstat,
        ds_name_lt1, ds_name_lt2,
        ds_name_lt1_binary, ds_name_lt2_binary,
        ds_name_flowcyt
    ]
    raw_data_paths_input = [
        raw_data_p_imstat,
        raw_data_p_lt1, raw_data_p_lt2,
        raw_data_p_lt1_binary, raw_data_p_lt2_binary,
        raw_data_p_flowcyt
    ]
    data_file_types_input = [
        data_file_type_imstat,
        data_file_type_lt1, data_file_type_lt2,
        data_file_type_lt1_binary, data_file_type_lt2_binary,
        data_file_type_flowcyt
    ]
    data_splits_input = [
        data_split_imstat,
        data_split_lt1, data_split_lt2,
        data_split_lt1_binary, data_split_lt2_binary,
        data_split_flowcyt
    ]
    channel_names_input = [
        channels_imstat,
        channels_lt1, channels_lt2,
        channels_lt1_binary, channels_lt2_binary,
        channels_flowcyt
    ]
    cutoff_dicts_input = [
        cutoff_dict_imstat,
        cutoff_dict_lt1, cutoff_dict_lt2,
        cutoff_dict_lt1_binary, cutoff_dict_lt2_binary,
        cutoff_dict_flowcyt
    ]
    label_keys_input = [
        label_key_imstat,
        label_key_lt1, label_key_lt2,
        label_key_lt1_binary, label_key_lt2_binary,
        label_key_flowcyt
    ]
    relabel_dicts_input = [
        relabel_dict_imstat,
        relabel_dict_lt1, relabel_dict_lt2,
        relabel_dict_lt1_binary, relabel_dict_lt2_binary,
        relabel_dict_flowcyt
    ]

    data_processing_workflow(
        dataset_names=dataset_names_input,
        raw_data_paths=raw_data_paths_input,
        data_file_types=data_file_types_input,
        label_keys=label_keys_input,
        relabel_dicts=relabel_dicts_input,
        cutoff_dicts=cutoff_dicts_input,
        data_splits=data_splits_input,
        channel_names=channel_names_input,
    )


def main_som_parameter_influence_study():
    """
    Script for running parameter-wise hyperparameter tuning on the immune status dataset.
    Its purpose is to determine the influence of individual parameters on the SOM classifier's performance
    and get an intuition of the magnitude.
    The workflow is as follows:
        - Preprocessed data is loaded.
        - Training data is downsampled to 10% of all events for faster training.
        - 3-fold cross validation is performed.
        - Results are saved and plotted.

    Note: Analysis should be run for n_epochs first to set a sensible value for all other analyses.

    The following flags are defined in the header and can be adjusted as needed:
    - inference (bool),  whether to do the training or just load and plot previously generated results.
    - test_n_epochs (bool), whether to run analysis for n_epochs or all other parameters.
    - random_seed (int)
    - downsampling_frac (float)
    - n_splits (int)
    - n_epochs (int)

    Returns:
        None
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import StratifiedKFold

    from flagx.gating import SomClassifier
    from validation.plt import plot_param_lineplot, plot_param_stripplot
    from validation.utils import set_pandas_print_options, get_downsampling_bool

    # ### Set flags and variables ######################################################################################
    inference = False  # Whether to do the hyperparameter tuning or just view the results
    test_n_epochs = False

    random_seed = 42
    downsampling_frac = 0.10
    n_splits = 3

    trafo = 'log10_w_custom_cutoffs'

    # Based on gridsearch for n_epochs, selected n_epochs such that performance is stable with default parameters
    n_epochs = 6000
    ####################################################################################################################

    # ### Load the train data
    data_p = os.path.join(os.getcwd(), f'data/np_files/imstat/{trafo}')

    x = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
    y = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int32)

    # ### Downsample for faster training
    np.random.seed(random_seed)
    target_num_events = int(x.shape[0] * downsampling_frac)
    print((
        f'# ### Downsampling training data from {x.shape[0]} events '
        f'to {target_num_events} events ({int(downsampling_frac * 100)} %)'
    ))
    downsampling_bool = get_downsampling_bool(y=y, target_num_events=target_num_events, stratified=True)
    x = x[downsampling_bool, :]
    y = y[downsampling_bool]

    # ### Define parameter grids for each individual parameter
    n_epochs_list = list(range(10, 101, 10)) + list(range(200, 1001, 100)) + list(range(2000, 15001, 1000))
    param_grid_nepochs = {
        'n_epochs': n_epochs_list,
    }

    param_grid_initialization = {
        'initialization': ['random', 'pca'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_gridtype = {
        'som_grid_type': ['rectangular', 'hexagonal'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_topology = {
        'som_topology': ['planar', 'toroid'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_dimension = {
        'som_dimensions': [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (35, 35), (40, 40)],
        'n_epochs': [n_epochs, ],
    }

    param_grid_neigh_fct = {
        'neighborhood': ['gaussian', 'bubble'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_neigh_sigma = {
        'gaussian_neighborhood_sigma': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        'n_epochs': [n_epochs, ],
    }

    param_grid_r0 = {
        'radius_0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        # 0.0 => min(n_columns, n_rows)/2, rn_default = 1.0
        'n_epochs': [n_epochs, ],
    }

    param_grid_rn = {
        'radius_n': [4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
        # r0_default = min(n_columns, n_rows)/2 = 5
        'n_epochs': [n_epochs, ],
    }

    param_grid_rcooling = {
        'radius_cooling': ['linear', 'exponential'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_lr0 = {
        'learning_rate_0': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0],
        # lr_n = 0.01 in default setting
        'n_epochs': [n_epochs, ],
    }

    param_grid_lrn = {
        'learning_rate_n': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001],
        # lr_0 = 0.1 in default setting
        'n_epochs': [n_epochs, ],
    }

    param_grid_lrdecay = {
        'learning_rate_decay': ['linear', 'exponential'],
        'n_epochs': [n_epochs, ],
    }

    if test_n_epochs:
        grids = [param_grid_nepochs, ]
        grid_names = ['n_epochs', ]
    else:
        grids = [
            param_grid_initialization,
            param_grid_gridtype, param_grid_topology, param_grid_dimension,
            param_grid_neigh_fct, param_grid_neigh_sigma, param_grid_r0, param_grid_rn, param_grid_rcooling,
            param_grid_lr0, param_grid_lrn, param_grid_lrdecay
        ]
        grid_names = [
            'initialization',
            'som_grid_type', 'som_topology', 'som_dimensions',
            'neighborhood', 'gaussian_neighborhood_sigma', 'radius_0', 'radius_n', 'radius_cooling',
            'learning_rate_0', 'learning_rate_n', 'learning_rate_decay'

        ]

    # ### Perform the parameter tuning
    # Define path where results will be stored
    save_p = os.path.join(os.getcwd(), f'results/som_parameter_influence_study/{trafo}')
    os.makedirs(save_p, exist_ok=True)

    for grid, grid_name in zip(grids, grid_names):

        print(f'# ### Grid name: {grid_name}')

        current_save_p = os.path.join(save_p, grid_name)
        os.makedirs(current_save_p, exist_ok=True)

        if inference:

            # Instantiate the SOM classifier
            som_clf = SomClassifier(verbosity=2)

            # Instantiate a stratified k-fold splitter
            cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

            # Perform cross-validated grid-search
            som_clf.hyperparameter_tuning(
                X=x.copy(),
                y=y.copy(),
                param_grid=grid,
                cv=cv_splitter,
                scoring='internal',
                refit=False,
            )

            # Save SOM classifier with results
            som_clf.save(filepath=current_save_p)

        else:
            # Load the previously trained SOM classifier
            som_clf = SomClassifier.load(filepath=current_save_p)

        # ### Evaluate the performance
        res_df = pd.DataFrame(som_clf.grid_search_.cv_results_)

        res_df.to_csv(os.path.join(current_save_p, f'{grid_name}.csv'))

        set_pandas_print_options()
        print('# ### Results:\n', res_df)

        if grid_name in {
            'initialization',
            'som_grid_type', 'som_topology', 'som_dimensions',
            'neighborhood', 'radius_cooling',
            'learning_rate_decay'
        }:
            plot_param_stripplot(
                res_df=res_df,
                id_var='param_' + grid_name,
                val_var= 'mean_test_score',
                val_name='Macro F1',
                jitter=True,
                xlabel=grid_name,
                dpi=300
            )
            plt.tight_layout()
            plt.savefig(os.path.join(current_save_p, f'{grid_name}.png'))
            plt.close('all')
        else:
            plot_param_lineplot(
                res_df=res_df,
                x_col='param_' + grid_name,
                y_col='mean_test_score',
                xlog10=False if grid_name != 'n_epochs' else True,
                xlog10plusone=False,
                custom_x_ticks=None if grid_name != 'n_epochs' else 'log10_scale',
                x_label=grid_name,
                y_label='Macro F1',
                x_axis_grid=True,
                dpi=300,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(current_save_p, f'{grid_name}.png'))
            plt.close('all')


def main_som_parameter_tuning():
    """
    Script for running hyperparameter tuning on the immune status dataset. The workflow is as follows:
        - Preprocessed data is loaded.
        - Data is downsampled to 10% of all events for faster training.
        - Data is split into train and val.
        - Results are printed and saved.

    The following flags are defined in the header and can be adjusted as needed:
    - inference (bool), whether to do the training or just load and print previously generated results.
    - random_seed (int)
    - downsampling_frac (float)
    - val_frac (float), relative size of the validation set
    - n_epochs (int)

    Returns:
        None
    """

    import os
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split, PredefinedSplit

    from flagx.gating import SomClassifier
    from validation.utils import set_pandas_print_options, get_downsampling_bool

    # ### Set flags and variables ######################################################################################
    inference = True  # Whether to do the hyperparameter tuning or just view the results
    trafo = 'log10_w_custom_cutoffs'

    random_seed = 42
    downsampling_frac = 0.10
    val_frac = 0.34

    grid = 'full_grid'  # 'full_grid', 'test'

    n_epochs = 6000
    ####################################################################################################################

    # Define dir for saving the results
    save_p = os.path.join(os.getcwd(), f'results/som_parameter_tuning/{trafo}/{grid}')
    os.makedirs(save_p, exist_ok=True)

    # ### Load the train data
    data_p_default = os.path.join(f'./data/np_files/imstat/{trafo}')
    data_p_hpc = f'/home/woody/iwbn/iwbn107h/data/np_files/imstat/{trafo}'

    if os.path.exists(data_p_hpc):
        data_p = data_p_hpc
    elif os.path.exists(data_p_default):
        data_p = data_p_default
    else:
        raise RuntimeError(f'\n# ### No data found at:\nhpc: "{data_p_hpc}"\ndefault: "{data_p_default}".')

    x = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
    y = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int32)

    # ### Downsample for faster training
    np.random.seed(random_seed)
    target_num_events = int(x.shape[0] * downsampling_frac)
    print((
        f'# ### Downsampling training data from {x.shape[0]} events '
        f'to {target_num_events} events ({int(downsampling_frac * 100)} %)'
    ))
    downsampling_bool = get_downsampling_bool(y=y, target_num_events=target_num_events, stratified=True)
    x = x[downsampling_bool, :]
    y = y[downsampling_bool]

    # ### Split data into train and val set (performance was observed to be stable across folds => No more cv here)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=val_frac,
        stratify=y,
        random_state=random_seed,
    )

    # Reconcatenate the data (first train, then val)
    x = np.vstack((x_train, x_val))
    y = np.hstack((y_train, y_val))

    # Create a PredefinedSplit according to the previous data split
    # (https://scikit-learn.org/1.5/modules/cross_validation.html#predefined-split)
    val_fold = [-1] * x_train.shape[0] + [0] * x_val.shape[0]
    cv = PredefinedSplit(test_fold=val_fold)

    # ### Define parameter grid, based on the previous experiments

    if grid == 'full_grid':
        param_grid = {
            'som_topology': ['planar', ],
            'som_grid_type': ['rectangular', ],
            'som_dimensions': [(15, 15), (20, 20), (25, 25)],
            'neighborhood': ['gaussian', ],
            'gaussian_neighborhood_sigma': [0.1, 0.5, 1.0],
            'initialization': ['pca', ],
            'n_epochs': [n_epochs, ],
            'radius_0': [-0.25, -0.5, -0.75],
            'radius_n': [0.1, ],
            'radius_cooling': ['exponential', ],
            'learning_rate_0': [0.1, 0.5, 1.0],
            'learning_rate_n': [0.001, 0.05, 0.1],
            'learning_rate_decay': ['exponential', ],
        }

    else:  # test
        param_grid = {
            'som_topology': ['planar', ],
            'som_grid_type': ['rectangular', ],
            'som_dimensions': [(15, 15), (20, 20), (25, 25)],
            'neighborhood': ['gaussian', ],
            'gaussian_neighborhood_sigma': [0.5, ],
            'initialization': ['pca', ],
            'n_epochs': [50, ],
            'radius_0': [-0.75, ],
            'radius_n': [0.75, ],
            'radius_cooling': ['linear', ],
            'learning_rate_0': [0.1, ],
            'learning_rate_n': [0.01, ],
            'learning_rate_decay': ['exponential', ],
        }

    # ### Inference
    if inference:
        # Instantiate the SOM classifier
        som_clf = SomClassifier(verbosity=2)

        # Perform the hyperparameter tuning
        som_clf.hyperparameter_tuning(
            X=x,
            y=y,
            param_grid=param_grid,
            cv=cv,
            scoring='internal',
            refit=False
        )

        # Save the SOM classifier
        som_clf.save(filepath=save_p)

    else:
        som_clf = SomClassifier.load(filepath=save_p)

    res_df = pd.DataFrame(som_clf.grid_search_.cv_results_)

    res_df.to_csv(os.path.join(save_p, f'res_df.csv'))

    set_pandas_print_options()
    print('# ### Results:\n', res_df)
    print('# ### Best parameters:\n', som_clf.grid_search_.best_params_)
    print('# ### Best score:\n', som_clf.grid_search_.best_score_)


def main_som_n_epochs_calibration():
    """
    Script for finding the optimal number of epochs to train for given a set of optimized parameters.
    The workflow is as follows:
        - Preprocessed data is loaded.
        - Data is split into train and val. (No downsampling!)
        - Results are printed and saved.

    The following flags are defined in the header and can be adjusted as needed:
        - inference (bool), whether to do the training or just load and print previously generated results.
        - random_seed (int)
        - val_frac (float), relative size of the validation set
        - n_epochs (int)

    Returns:
        None
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split, PredefinedSplit

    from flagx.gating import SomClassifier
    from validation.utils import set_pandas_print_options
    from validation.plt import plot_param_lineplot

    # ### Set flags and variables ######################################################################################
    inference = True  # Whether to do the hyperparameter tuning or just view the results
    trafo = 'log10_w_custom_cutoffs'

    random_seed = 42
    val_frac = 0.34

    # Gridsearch for n_epochs
    n_epochs = list(range(10, 101, 10)) + list(range(200, 2001, 100)) + list(range(2500, 6001, 500))
    ####################################################################################################################

    # ### Load the train data
    data_p = f'./data/np_files/imstat/{trafo}'

    x = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
    y = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int32)

    # ### Split data into train and val set (performance was observed to be stable across folds => No more cv here)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=val_frac,
        stratify=y,
        random_state=random_seed,
    )

    # Reconcatenate the data (first train, then val)
    x = np.vstack((x_train, x_val))
    y = np.hstack((y_train, y_val))

    # Create a PredefinedSplit according to the previous data split
    # (https://scikit-learn.org/1.5/modules/cross_validation.html#predefined-split)
    val_fold = [-1] * x_train.shape[0] + [0] * x_val.shape[0]
    cv = PredefinedSplit(test_fold=val_fold)

    # Load the previously optimized parameters and define a parameter grid with them
    som_clf_param_tuning = SomClassifier.load(filepath=f'./results/som_parameter_tuning/{trafo}/full_grid')
    best_params = som_clf_param_tuning.grid_search_.best_params_
    print('# --- Best parameters:')
    for key, value in best_params.items():
        print(f'# {key}: {value}')

    # Define dir for saving the results
    save_p = f'./results/som_n_epochs_calibration/{trafo}'
    os.makedirs(save_p, exist_ok=True)

    # ### Inference
    if inference:
        # Instantiate the SOM classifier with the best parameters
        som_clf = SomClassifier(verbosity=1, **best_params)

        # Perform the hyperparameter tuning
        som_clf.hyperparameter_tuning(
            X=x,
            y=y,
            param_grid={'n_epochs': n_epochs},
            cv=cv,
            scoring='internal',
            refit=False
        )

        # Save the SOM classifier
        som_clf.save(filepath=save_p)

    else:
        som_clf = SomClassifier.load(filepath=save_p)

    res_df = pd.DataFrame(som_clf.grid_search_.cv_results_)

    res_df.to_csv(os.path.join(save_p, f'res_df.csv'))

    set_pandas_print_options()
    print('# ### Results:\n', res_df)
    print('# ### Best parameters:\n', som_clf.grid_search_.best_params_)
    print('# ### Best score:\n', som_clf.grid_search_.best_score_)

    plot_param_lineplot(
        res_df=res_df,
        x_col='param_n_epochs',
        y_col='mean_test_score',
        xlog10=True,
        xlog10plusone=False,
        custom_x_ticks='log10_scale',
        x_label='n epochs',
        y_label='Macro F1',
        dpi=300,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'n_epochs.png'))
    plt.close('all')


def main_som_classifier():
    """
    Runs training, prediction, and evaluation for the SOM classifier across multiple datasets.

    This script loads preprocessed data, trains the SOM classifier with previously optimized parameters,
    performs sample-wise and concatenated predictions, computes evaluation metrics,
    tracks runtime performance, and stores all outputs in the results directory.
    The function supports HPC and local file paths and handles automatic loading of
    existing models and predictions when fitting or prediction is disabled.

    Saves:
        - Trained SOM classifier
        - Runtime statistics.
        - Predictions (full test set and sample-wise).
        - Evaluation metrics (average, class-wise, sample-wise).
        - Confusion matrices and abstention counts.
    """

    import os
    import numpy as np
    import pandas as pd

    from flagx.gating import SomClassifier
    from validation.utils import eval_wrapper, eval_wrapper_sample_wise, scalability_wrapper

    # ### Set flags and important variables here #######################################################################
    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    others_labels = [8, None, None, None, None, 5]
    pos_labels = [None, None, None, 1, 1, None]

    preprocessing_trafo = 'log10_w_custom_cutoffs'

    fit = True
    predict = True
    evaluate = True
    ####################################################################################################################

    for data_set, others_label, pos_label in zip(data_sets, others_labels, pos_labels):

        if preprocessing_trafo == 'log10_w_custom_cutoffs' and data_set == 'flowcyt':
            trafo = 'log10_cutoff100'
        else:
            trafo = preprocessing_trafo

        print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

        # Check whether train data exists for this dataset and trafo, if not continue
        data_p_default = os.path.join(f'./data/np_files/{data_set}/{trafo}')
        data_p_hpc = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

        if os.path.exists(data_p_hpc):
            data_p = data_p_hpc
        elif os.path.exists(data_p_default):
            data_p = data_p_default
        else:
            print(f'\n# ### No data found at:\nhpc: "{data_p_hpc}"\ndefault: "{data_p_default}".\nContinue.')
            continue

        # Define path where results will be saved to
        save_p = os.path.join(os.getcwd(), f'./results/gating_performance/som/{data_set}/{trafo}')
        os.makedirs(save_p, exist_ok=True)

        if fit:

            # Load training data
            x_train = np.load(os.path.join(data_p, 'x_train.npy'))
            y_train = np.load(os.path.join(data_p, 'y_train.npy'))

            # Instantiate the SOM classifier
            som_clf = SomClassifier(
                som_topology='planar',
                som_grid_type='rectangular',
                som_dimensions=(25, 25),
                neighborhood='gaussian',
                gaussian_neighborhood_sigma=0.1,
                initialization='pca',
                n_epochs=1000,
                radius_0=-0.5,
                radius_n=0.1,
                radius_cooling='exponential',
                learning_rate_0=0.1,
                learning_rate_n=0.001,
                learning_rate_decay='exponential',
                verbosity=2,
            )

            # Fit and track time
            print('# ### Starting fit ...')
            def dummy_fit():
                som_clf.fit(X=x_train, y=y_train)
                return som_clf
            fit_time_df, som_clf = scalability_wrapper(function=dummy_fit, function_params=None, track_gpu=False)
            fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))
            som_clf.save(filepath=save_p)

        else:
            # Load the SOM classifier
            som_clf = SomClassifier.load(filepath=save_p)

        if predict:
            # Load the test data
            x_test = np.load(os.path.join(data_p, 'x_test.npy'))

            print('# ### Starting prediction ...')
            def dummy_predict():
                return som_clf.predict(X=x_test)
            pred_time_df, y_pred = scalability_wrapper(function=dummy_predict, function_params=None, track_gpu=False)
            pred_time_df.to_csv(os.path.join(save_p, 'pred_time_df.csv'))
            np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)

            # Load the sample-wise test data
            samples_p = os.path.join(data_p, 'sample_wise_test')
            n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]

            samples_y_pred = []
            samples_pred_time_dfs = []

            print('# ### Starting sample-wise prediction ...')
            for x, sn in zip(samples_x_test, sample_names):
                def dummy_predict_sample():
                    return som_clf.predict(X=x)
                pred_time_df_sample, y_pred_sample = scalability_wrapper(
                    function=dummy_predict_sample, function_params=None, track_gpu=False
                )
                pred_time_df_sample['sample_name'] = sn
                samples_y_pred.append(y_pred_sample)
                samples_pred_time_dfs.append(pred_time_df_sample)

            samples_pred_times_df = pd.concat(samples_pred_time_dfs, ignore_index=True)
            samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

            os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
            for y, sn in zip(samples_y_pred, sample_names):
                np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y)

        else:
            # Load the predictions
            y_pred = np.load(os.path.join(save_p, 'y_pred.npy'))

            samples_y_pred_p = os.path.join(save_p, 'samples_y_pred')
            samples_y_pred_filenames = [
                f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(len(os.listdir(samples_y_pred_p)))
            ]
            samples_y_pred = [np.load(os.path.join(samples_y_pred_p, fn)) for fn in samples_y_pred_filenames]

        if evaluate:

            # Load the labels of the test data
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            # Compute evaluation metrics for samples concatenated to one
            out = eval_wrapper(
                y_true=y_test,
                y_pred=y_pred,
                abstention_label=-1,
                others_label=others_label,
                pos_label=pos_label,
                verbosity=2
            )

            out[0].to_csv(os.path.join(save_p, 'res_df_avg.csv'))
            out[1].to_csv(os.path.join(save_p, 'res_df_cw.csv'))
            out[2].to_csv(os.path.join(save_p, 'cf_mat.csv'))
            out[3].to_csv(os.path.join(save_p, 'abst_counts.csv'))

            # Load the labels of the sample-wise test data
            samples_p = os.path.join(data_p, 'sample_wise_test')
            n_samples = len([f for f in os.listdir(samples_p) if f.startswith('y_')])
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_y_test_filenames = [f'y_{sn}.npy' for sn in sample_names]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test_filenames]

            # Compute sample-wise evaluation metrics
            out_sw = eval_wrapper_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                abstention_label=-1,
                others_label=others_label,
                pos_label=pos_label,
                verbosity=2,
            )

            out_sw[0].to_csv(os.path.join(save_p, 'res_df_sw_avg_prec.csv'))
            out_sw[1].to_csv(os.path.join(save_p, 'res_df_sw_avg_rec.csv'))
            out_sw[2].to_csv(os.path.join(save_p, 'res_df_sw_avg_f1.csv'))

            out_sw[3].to_csv(os.path.join(save_p, 'res_df_sw_cw_prec.csv'))
            out_sw[4].to_csv(os.path.join(save_p, 'res_df_sw_cw_rec.csv'))
            out_sw[5].to_csv(os.path.join(save_p, 'res_df_sw_cw_f1.csv'))

            out_sw[7].to_csv(os.path.join(save_p, 'abst_counts.csv'))

            os.makedirs(os.path.join(save_p, 'confusion_matrices_sw'), exist_ok=True)
            for cf_df, sn in zip(out_sw[6], sample_names):
                cf_df.to_csv(os.path.join(save_p, 'confusion_matrices_sw', f'cf_mat_{sn}.csv'))


def main_gatemeclass():
    """
    Runs training, prediction, and evaluation for the GateMeClass classifier across multiple datasets.

    This script loads preprocessed data, trains the GMC classifier,
    performs both concatenated and sample-wise predictions, computes evaluation metrics,
    tracks runtime performance, tracks errors at the training or inference stage, and stores all outputs in the results directory.
    It supports HPC and local file paths and includes error handling for failed fits or predictions,
    logging all encountered issues for later inspection.

    Saves:
        - Trained GMC classifier.
        - Runtime statistics.
        - Predictions (full test set and sample-wise).
        - Evaluation metrics (average, class-wise, sample-wise).
        - Confusion matrices and abstention counts (if enabled).
        - Error logs for failed fits or predictions.
    """

    import os
    import numpy as np
    import pandas as pd
    from validation.gating.gatemeclass import GateMeClassClassifier
    from validation.utils import eval_wrapper, eval_wrapper_sample_wise, get_error_dataframe, scalability_wrapper

    # ### Set flags and important variables here #######################################################################
    fit = True
    predict = True
    evaluate = True

    allow_abstention = False
    abstention_label = -1 if allow_abstention else None

    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    others_labels = [8, None, None, None, None, 5]
    pos_labels = [None, None, None, 1, 1, None]

    marker_names_imstat = [
        'FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'
    ]
    marker_names_lymphoma_t1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7',
        'CD38_APC', 'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    marker_names_lymphoma_t2 = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7',
        'CD52_APC', 'CD11c_APC_A700', 'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]

    marker_names_lymphoma_t1_binary = marker_names_lymphoma_t1

    marker_names_lymphoma_t2_binary = marker_names_lymphoma_t2

    marker_names_flowcyt = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]

    marker_names_list = [
        marker_names_imstat,
        marker_names_lymphoma_t1, marker_names_lymphoma_t2,
        marker_names_lymphoma_t1_binary, marker_names_lymphoma_t2_binary,
        marker_names_flowcyt,
    ]

    preprocessing_trafo = 'arcsinh_cofactor150'  # 'log10_w_custom_cutoffs', 'arcsinh_cofactor150'

    ####################################################################################################################

    # Define lists to track errors
    failure_combinations = []
    failure_points = []
    error_types = []
    error_messages = []

    for data_set, others_label, pos_label, marker_names  in zip(
            data_sets, others_labels, pos_labels, marker_names_list
    ):


        if preprocessing_trafo == 'log10_w_custom_cutoffs' and data_set == 'flowcyt':
            trafo = 'log10_cutoff100'
        else:
            trafo = preprocessing_trafo

        print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

        # Check whether train data exists for this dataset and trafo, if not continue
        data_p_default = os.path.join(f'./data/np_files/{data_set}/{trafo}')
        data_p_hpc = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

        if os.path.exists(data_p_hpc):
            data_p = data_p_hpc
        elif os.path.exists(data_p_default):
            data_p = data_p_default
        else:
            print(f'\n# ### No data found at:\nhpc: "{data_p_hpc}"\ndefault: "{data_p_default}".\nContinue.')
            continue

        # Define path where results will be saved to
        abstention_str = '_w_abstention' if allow_abstention else ''
        save_p = f'./results/gating_performance/gatemeclass{abstention_str}/{data_set}/{trafo}'
        os.makedirs(save_p, exist_ok=True)

        # Get the number of samples
        samples_p = os.path.join(data_p, 'sample_wise_test')
        n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])

        if fit:

            # Load training data
            x_train = np.load(os.path.join(data_p, 'x_train.npy'))
            y_train = np.load(os.path.join(data_p, 'y_train.npy'))

            # Instantiate the GMC classifier with default parameters
            gmc_clf = GateMeClassClassifier(
                marker_names=marker_names,
                gmc_gmm_parameterization='V',
                gmc_k=20,
                gmc_sampling=0.1,
                gmc_reject_option=allow_abstention,
                gmc_seed=1,
                time_fit_pred=True,
                verbosity=1
            )

            try:
                # Fit and track time
                def dummy_fit():
                    gmc_clf.fit(X=x_train, y=y_train)
                    return gmc_clf
                fit_time_df, gmc_clf = scalability_wrapper(function=dummy_fit)
                fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))
                gmc_clf.save(filepath=save_p)

            except Exception as e:
                # Log errors
                failure_combinations.append(f'{data_set}_{trafo}')
                failure_points.append('fit')
                error_types.append(type(e).__name__)
                error_messages.append(str(e))

                error_df = get_error_dataframe(failure_combinations, failure_points, error_types ,error_messages)
                error_df.to_csv(os.path.join(save_p, 'errors.csv'))
                continue

        else:
            # Load the GMC classifier
            try:
                gmc_clf = GateMeClassClassifier.load(filepath=save_p)
            except FileNotFoundError:
                print(f'# ### Classifier could not be trained without error. Continue.\n')
                continue

        if predict:
            # Load the test data
            x_test = np.load(os.path.join(data_p, 'x_test.npy'))

            try:
                print('# ### Starting prediction ...')
                def dummy_predict():
                    return gmc_clf.predict(X=x_test)
                pred_time_df, y_pred = scalability_wrapper(function=dummy_predict)
                pred_time_df.to_csv(os.path.join(save_p, 'pred_time_df.csv'))
                np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)

            except Exception as e:
                failure_combinations.append(f'{data_set}_{trafo}')
                failure_points.append('predict_concatenated')
                error_types.append(type(e).__name__)
                error_messages.append(str(e))

                error_df = get_error_dataframe(failure_combinations, failure_points, error_types, error_messages)
                error_df.to_csv(os.path.join(save_p, 'errors.csv'))
                continue

            # Load the sample-wise test data
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]

            samples_y_pred = []
            samples_pred_time_dfs = []
            successful_fit_idx = []
            print('# ### Starting sample-wise prediction ...')
            for i, (x, sn) in enumerate(zip(samples_x_test, sample_names)):
                try:
                    def dummy_predict_sample():
                        return gmc_clf.predict(X=x)
                    pred_time_df_sample, y_pred_sample = scalability_wrapper(function=dummy_predict_sample)
                    pred_time_df_sample['sample_name'] = sn
                    samples_y_pred.append(y_pred_sample)
                    samples_pred_time_dfs.append(pred_time_df_sample)
                    successful_fit_idx.append(i)
                except Exception as e:
                    nan_row = pd.DataFrame([{
                        'sample_name': sn,
                        'wall_time': np.nan,
                        'mem_peak_cpu': np.nan,
                        'mem_avg_cpu': np.nan,
                        'samples_cpu': np.nan,
                        'mem_peak_gpu': np.nan,
                        'mem_avg_gpu': np.nan,
                        'samples_gpu': np.nan,
                    }])
                    samples_pred_time_dfs.append(nan_row)
                    failure_combinations.append(f'{data_set}_{trafo}')
                    failure_points.append(f'predict_{sn}')
                    error_types.append(type(e).__name__)
                    error_messages.append(str(e))

                    error_df = get_error_dataframe(
                        failure_combinations, failure_points, error_types, error_messages
                    )
                    error_df.to_csv(os.path.join(save_p, 'errors.csv'))

            samples_pred_times_df = pd.concat(samples_pred_time_dfs, ignore_index=True)
            samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

            os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
            for y, i in zip(samples_y_pred, successful_fit_idx):
                np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_sample_{str(i).zfill(2)}_test.npy'), y)

        else:
            # Load the predictions
            try:
                y_pred = np.load(os.path.join(save_p, 'y_pred.npy'))
            except FileNotFoundError:
                print(f'# ### Fit was not successful for dataset: {data_set}, trafo: {trafo}. Continue.\n')
                continue

            samples_y_pred_p = os.path.join(save_p, 'samples_y_pred')
            samples_y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]

            samples_y_pred = []
            successful_fit_idx = []
            for i, fn in enumerate(samples_y_pred_filenames):
                try:
                    samples_y_pred.append(np.load(os.path.join(samples_y_pred_p, fn)))
                    successful_fit_idx.append(i)
                except FileNotFoundError:
                    print(
                        f'# ### Fit was not successful for dataset: {data_set}, trafo: {trafo}, sample: {fn}. '
                        f'Cannot include it in evaluation.\n'
                    )
                    continue

        if evaluate:

            # Load the labels of the test data
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            # Compute evaluation metrics for samples concatenated to one
            out = eval_wrapper(
                y_true=y_test,
                y_pred=y_pred,
                abstention_label=abstention_label,
                others_label=others_label,
                pos_label=pos_label,
                verbosity=2
            )

            out[0].to_csv(os.path.join(save_p, 'res_df_avg.csv'))
            out[1].to_csv(os.path.join(save_p, 'res_df_cw.csv'))
            out[2].to_csv(os.path.join(save_p, 'cf_mat.csv'))
            if abstention_label is not None:
                out[3].to_csv(os.path.join(save_p, 'abst_counts.csv'))

            # Get the sample-wise test data for which the prediction was successful
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples) if i in successful_fit_idx]
            samples_y_test_filenames = [f'y_{sn}.npy' for sn in sample_names]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test_filenames]

            # Compute sample-wise evaluation metrics
            out_sw = eval_wrapper_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                abstention_label=abstention_label,
                others_label=others_label,
                pos_label=pos_label,
                verbosity=2,
            )

            out_sw[0].to_csv(os.path.join(save_p, 'res_df_sw_avg_prec.csv'))
            out_sw[1].to_csv(os.path.join(save_p, 'res_df_sw_avg_rec.csv'))
            out_sw[2].to_csv(os.path.join(save_p, 'res_df_sw_avg_f1.csv'))

            out_sw[3].to_csv(os.path.join(save_p, 'res_df_sw_cw_prec.csv'))
            out_sw[4].to_csv(os.path.join(save_p, 'res_df_sw_cw_rec.csv'))
            out_sw[5].to_csv(os.path.join(save_p, 'res_df_sw_cw_f1.csv'))

            if abstention_label is not None:
                out_sw[7].to_csv(os.path.join(save_p, 'abst_counts.csv'))

            os.makedirs(os.path.join(save_p, 'confusion_matrices_sw'), exist_ok=True)
            for cf_df, sn in zip(out_sw[6], sample_names):
                cf_df.to_csv(os.path.join(save_p, 'confusion_matrices_sw', f'cf_mat_{sn}.csv'))


def main_dgcytof():
    """
    Runs training, prediction, and evaluation for the DGCytof classifier across multiple datasets.

    This script loads preprocessed data, trains the DGCytof classifier,
    performs sample-wise predictions, computes evaluation metrics, tracks runtime performance, tracks errors,
    and stores all outputs in the results directory. It supports HPC and local file paths and
    includes extensive error handling for failed training or prediction steps, logging all issues
    for later inspection.

    Saves:
        - Trained DGCytof classifier.
        - Runtime statistics.
        - Sample-wise predictions.
        - Evaluation metrics (average, class-wise, sample-wise).
        - Confusion matrices and abstention counts.
        - Error logs for failed fits or predictions.
    """

    import os
    import numpy as np
    import pandas as pd
    from validation.gating.dgcytof import DgcytofClassifier
    from validation.utils import eval_wrapper, eval_wrapper_sample_wise, get_error_dataframe, scalability_wrapper

    # ### Set flags and important variables here #######################################################################
    fit = True
    predict = True
    evaluate = True

    data_sets = [
        'flowcyt', 'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary'
    ]
    others_labels = [5, 8, None, None, None, None]
    pos_labels = [None, None, None, None, 1, 1]

    preprocessing_trafo = 'log10_w_custom_cutoffs'

    ####################################################################################################################

    # Define lists to track errors
    failure_combinations = []
    failure_points = []
    error_types = []
    error_messages = []

    for data_set, others_label, pos_label in zip(data_sets, others_labels, pos_labels):

        if preprocessing_trafo == 'log10_w_custom_cutoffs' and data_set == 'flowcyt':
            trafo = 'log10_cutoff100'
        else:
            trafo = preprocessing_trafo

        print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

        # Check whether train data exists for this dataset and trafo, if not continue
        data_p_default = os.path.join(f'./data/np_files/{data_set}/{trafo}')
        data_p_hpc = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

        if os.path.exists(data_p_hpc):
            data_p = data_p_hpc
        elif os.path.exists(data_p_default):
            data_p = data_p_default
        else:
            print(f'\n# ### No data found at:\nhpc: "{data_p_hpc}"\ndefault: "{data_p_default}".\nContinue.')
            continue

        # Define path where results will be saved to
        save_p = f'./results/gating_performance/dgcytof/{data_set}/{trafo}'
        os.makedirs(save_p, exist_ok=True)

        # Get the number of samples
        samples_p = os.path.join(data_p, 'sample_wise_test')
        n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])

        if fit:

            # Load training data
            x_train = np.load(os.path.join(data_p, 'x_train.npy'))
            y_train = np.load(os.path.join(data_p, 'y_train.npy'))

            # Instantiate the Dgcytof classifier with default parameters
            dgcytof_clf = DgcytofClassifier(
                val_size=0.2,
                layer_sizes=(128, 64, 32),
                n_epochs=20,
                train_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
                verbosity=2,
            )

            try:
                # Fit and track time
                print('# ### Starting fit ...')
                def dummy_fit():
                    dgcytof_clf.fit(X=x_train, y=y_train)
                    return dgcytof_clf
                fit_time_df, dgcytof_clf = scalability_wrapper(function=dummy_fit, track_gpu=True)
                fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))
                dgcytof_clf.save(filepath=save_p)

            except Exception as e:
                # Log errors
                failure_combinations.append(f'{data_set}_{trafo}')
                failure_points.append('fit')
                error_types.append(type(e).__name__)
                error_messages.append(str(e))

                error_df = get_error_dataframe(failure_combinations, failure_points, error_types, error_messages)
                error_df.to_csv(os.path.join(save_p, 'errors.csv'))
                continue

        else:
            try:
                dgcytof_clf = DgcytofClassifier.load(filepath=save_p)
            except FileNotFoundError:
                print(f'# ### Classifier could not be trained without error. Continue.\n')
                continue

        if predict:

            # # Load the test data
            # x_test = np.load(os.path.join(data_p, 'x_test.npy'))
            # try:
            #     print('# ### Starting prediction ...')
            #     def dummy_predict():
            #         return dgcytof_clf.predict(X=x_test)
            #     pred_time_df, y_pred = scalability_wrapper(function=dummy_predict)
            #     pred_time_df.to_csv(os.path.join(save_p, 'pred_time_df.csv'))
            #     np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)
            #
            # except Exception as e:
            #     failure_combinations.append(f'{data_set}_{trafo}')
            #     failure_points.append('predict_concatenated')
            #     error_types.append(type(e).__name__)
            #     error_messages.append(str(e))
            #
            #     error_df = get_error_dataframe(failure_combinations, failure_points, error_types, error_messages)
            #     error_df.to_csv(os.path.join(save_p, 'errors.csv'))
            #     continue

            # Load the sample-wise test data
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]

            samples_y_pred = []
            samples_pred_time_dfs = []
            successful_pred_idx = []
            print('# ### Starting sample-wise prediction ...')
            for i, (x, sn) in enumerate(zip(samples_x_test, sample_names)):
                try:
                    def dummy_predict_sample():
                        return dgcytof_clf.predict(X=x)
                    pred_time_df_sample, y_pred_sample = scalability_wrapper(function=dummy_predict_sample)
                    pred_time_df_sample['sample_name'] = sn
                    samples_y_pred.append(y_pred_sample)
                    samples_pred_time_dfs.append(pred_time_df_sample)
                    successful_pred_idx.append(i)

                except Exception as e:
                    nan_row = pd.DataFrame([{
                        'sample_name': sn,
                        'wall_time': np.nan,
                        'mem_peak_cpu': np.nan,
                        'mem_avg_cpu': np.nan,
                        'samples_cpu': np.nan,
                        'mem_peak_gpu': np.nan,
                        'mem_avg_gpu': np.nan,
                        'samples_gpu': np.nan,
                    }])
                    samples_pred_time_dfs.append(nan_row)
                    failure_combinations.append(f'{data_set}_{trafo}')
                    failure_points.append(f'predict_{sn}')
                    error_types.append(type(e).__name__)
                    error_messages.append(str(e))

                    error_df = get_error_dataframe(
                        failure_combinations, failure_points, error_types, error_messages
                    )
                    error_df.to_csv(os.path.join(save_p, 'errors.csv'))

            samples_pred_times_df = pd.concat(samples_pred_time_dfs, ignore_index=True)
            samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

            os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
            for y, i in zip(samples_y_pred, successful_pred_idx):
                np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_sample_{str(i).zfill(2)}_test.npy'), y)

        else:
            # Load the predictions
            # try:
            #     y_pred = np.load(os.path.join(save_p, 'y_pred.npy'))
            # except FileNotFoundError:
            #     print(f'# ### Fit was not successful for dataset: {data_set}, trafo: {trafo}. Continue.\n')
            #     continue

            samples_y_pred_p = os.path.join(save_p, 'samples_y_pred')
            samples_y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
            samples_y_pred = []
            successful_pred_idx = []
            for i, fn in enumerate(samples_y_pred_filenames):
                try:
                    samples_y_pred.append(np.load(os.path.join(samples_y_pred_p, fn)))
                    successful_pred_idx.append(i)
                except FileNotFoundError:
                    print(
                        f'# ### Fit was not successful for dataset: {data_set}, trafo: {trafo}, sample: {fn}. '
                        f'Cannot include it in evaluation.\n'
                    )
                    continue

        if evaluate:

            # # Load the labels of the test data
            # y_test = np.load(os.path.join(data_p, 'y_test.npy'))
            #
            # # Compute evaluation metrics for samples concatenated to one
            # out = eval_wrapper(
            #     y_true=y_test,
            #     y_pred=y_pred,
            #     abstention_label=-1,  # Dgcytof can predict events to be of 'unknown'/-1 class
            #     others_label=others_label,
            #     pos_label=pos_label,
            #     verbosity=2
            # )
            #
            # out[0].to_csv(os.path.join(save_p, 'res_df_avg.csv'))
            # out[1].to_csv(os.path.join(save_p, 'res_df_cw.csv'))
            # out[2].to_csv(os.path.join(save_p, 'cf_mat.csv'))
            # out[3].to_csv(os.path.join(save_p, 'abst_counts.csv'))

            # Get the sample-wise test data for which the prediction was successful
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples) if i in successful_pred_idx]
            samples_y_test_filenames = [f'y_{sn}.npy' for sn in sample_names]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test_filenames]

            # Compute sample-wise evaluation metrics
            out_sw = eval_wrapper_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                abstention_label=-1,  # Dgcytof can predict events to be of 'unknown'/-1 class
                others_label=others_label,
                pos_label=pos_label,
                verbosity=2,
            )

            out_sw[0].to_csv(os.path.join(save_p, 'res_df_sw_avg_prec.csv'))
            out_sw[1].to_csv(os.path.join(save_p, 'res_df_sw_avg_rec.csv'))
            out_sw[2].to_csv(os.path.join(save_p, 'res_df_sw_avg_f1.csv'))

            out_sw[3].to_csv(os.path.join(save_p, 'res_df_sw_cw_prec.csv'))
            out_sw[4].to_csv(os.path.join(save_p, 'res_df_sw_cw_rec.csv'))
            out_sw[5].to_csv(os.path.join(save_p, 'res_df_sw_cw_f1.csv'))

            out_sw[7].to_csv(os.path.join(save_p, 'abst_counts.csv'))

            os.makedirs(os.path.join(save_p, 'confusion_matrices_sw'), exist_ok=True)
            for cf_df, sn in zip(out_sw[6], sample_names):
                cf_df.to_csv(os.path.join(save_p, 'confusion_matrices_sw', f'cf_mat_{sn}.csv'))


def main_fcnn():
    """
    Runs training, prediction, and evaluation for the MLP classifier across multiple datasets.

    This script loads preprocessed data, trains the MLP-based classifier using predefined
    parameters, performs both concatenated and sample-wise predictions, computes evaluation
    metrics, tracks runtime performance, and stores all outputs in the results directory.
    It supports HPC and local file paths and automatically loads existing models and
    predictions when fitting or prediction is disabled.

    Saves:
        - Trained FCNN classifier.
        - Runtime statistics.
        - Predictions (full test set and sample-wise).
        - Evaluation metrics (average, class-wise, sample-wise).
        - Confusion matrices.
    """


    import os
    import numpy as np
    import pandas as pd

    from flagx.gating import MLPClassifier
    from validation.utils import eval_wrapper, eval_wrapper_sample_wise, scalability_wrapper

    # ### Set flags and important variables here #######################################################################
    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    pos_labels = [None, None, None, 1, 1, None]

    preprocessing_trafo = 'log10_w_custom_cutoffs'

    fit = True
    predict = True
    evaluate = True
    ####################################################################################################################

    for data_set, pos_label in zip(data_sets, pos_labels):

        if preprocessing_trafo == 'log10_w_custom_cutoffs' and data_set == 'flowcyt':
            trafo = 'log10_cutoff100'
        else:
            trafo = preprocessing_trafo

        print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

        # Check whether train data exists for this dataset and trafo, if not continue
        data_p_default = os.path.join(f'./data/np_files/{data_set}/{trafo}')
        data_p_hpc = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

        if os.path.exists(data_p_hpc):
            data_p = data_p_hpc
        elif os.path.exists(data_p_default):
            data_p = data_p_default
        else:
            print(f'\n# ### No data found at:\nhpc: "{data_p_hpc}"\ndefault: "{data_p_default}".\nContinue.')
            continue

        # Define path where results will be saved to
        save_p = f'./results/gating_performance/fcnn/{data_set}/{trafo}'
        os.makedirs(save_p, exist_ok=True)

        if fit:

            # Load training data
            x_train = np.load(os.path.join(data_p, 'x_train.npy'))
            y_train = np.load(os.path.join(data_p, 'y_train.npy'))

            # Instantiate the Softmax classifier with default parameters
            fcnn_clf = MLPClassifier(
                layer_sizes=(128, 64, 32),
                n_epochs=20,
                data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
                device=None,  # Tries to use default cuda device, if none available cpu
                verbosity=2
            )

            print('# ### Starting fit ...')
            def dummy_fit():
                fcnn_clf.fit(X=x_train, y=y_train)
                return fcnn_clf
            fit_time_df, fcnn_clf = scalability_wrapper(function=dummy_fit, track_gpu=True)
            fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))
            fcnn_clf.save(filepath=save_p)

        else:
            # Load the SOM classifier
            fcnn_clf = MLPClassifier.load(filepath=save_p)

        if predict:
            # Load the test data
            x_test = np.load(os.path.join(data_p, 'x_test.npy'))

            print('# ### Starting prediction ...')
            def dummy_predict():
                return fcnn_clf.predict(X=x_test)
            pred_time_df, y_pred = scalability_wrapper(function=dummy_predict)
            pred_time_df.to_csv(os.path.join(save_p, 'pred_time_df.csv'))
            np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)

            # Load the sample-wise test data
            samples_p = os.path.join(data_p, 'sample_wise_test')
            n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]

            samples_y_pred = []
            samples_pred_time_dfs = []

            print('# ### Starting sample-wise prediction ...')
            for x, sn in zip(samples_x_test, sample_names):
                def dummy_predict_sample():
                    return fcnn_clf.predict(X=x)
                pred_time_df_sample, y_pred_sample = scalability_wrapper(function=dummy_predict_sample)
                pred_time_df_sample['sample_name'] = sn
                samples_y_pred.append(y_pred_sample)
                samples_pred_time_dfs.append(pred_time_df_sample)

            samples_pred_times_df = pd.concat(samples_pred_time_dfs, ignore_index=True)
            samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

            os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
            for y, sn in zip(samples_y_pred, sample_names):
                np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y)

        else:
            # Load the predictions
            y_pred = np.load(os.path.join(save_p, 'y_pred.npy'))

            samples_y_pred_p = os.path.join(save_p, 'samples_y_pred')
            samples_y_pred_filenames = [
                f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(len(os.listdir(samples_y_pred_p)))
            ]
            samples_y_pred = [np.load(os.path.join(samples_y_pred_p, fn)) for fn in samples_y_pred_filenames]

        if evaluate:

            # Load the labels of the test data
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            # Compute evaluation metrics for samples concatenated to one
            out = eval_wrapper(
                y_true=y_test,
                y_pred=y_pred,
                abstention_label=None,
                others_label=None,
                pos_label=pos_label,
                verbosity=2
            )

            out[0].to_csv(os.path.join(save_p, 'res_df_avg.csv'))
            out[1].to_csv(os.path.join(save_p, 'res_df_cw.csv'))
            out[2].to_csv(os.path.join(save_p, 'cf_mat.csv'))

            # Load the labels of the sample-wise test data
            samples_p = os.path.join(data_p, 'sample_wise_test')
            n_samples = len([f for f in os.listdir(samples_p) if f.startswith('y_')])
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_y_test_filenames = [f'y_{sn}.npy' for sn in sample_names]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test_filenames]

            # Compute sample-wise evaluation metrics
            out_sw = eval_wrapper_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                abstention_label=None,
                others_label=None,
                pos_label=pos_label,
                verbosity=2,
            )

            out_sw[0].to_csv(os.path.join(save_p, 'res_df_sw_avg_prec.csv'))
            out_sw[1].to_csv(os.path.join(save_p, 'res_df_sw_avg_rec.csv'))
            out_sw[2].to_csv(os.path.join(save_p, 'res_df_sw_avg_f1.csv'))

            out_sw[3].to_csv(os.path.join(save_p, 'res_df_sw_cw_prec.csv'))
            out_sw[4].to_csv(os.path.join(save_p, 'res_df_sw_cw_rec.csv'))
            out_sw[5].to_csv(os.path.join(save_p, 'res_df_sw_cw_f1.csv'))

            os.makedirs(os.path.join(save_p, 'confusion_matrices_sw'), exist_ok=True)
            for cf_df, sn in zip(out_sw[6], sample_names):
                cf_df.to_csv(os.path.join(save_p, 'confusion_matrices_sw', f'cf_mat_{sn}.csv'))


def main_num_samples_num_events_experiment():
    """
    Runs the core num-samples / num-events experiment for a given classifier.

    This helper function trains a classifier on varying numbers of samples and events,
    evaluates its performance, and stores detailed results for each configuration.
    For every combination of (n_samples, n_events), the function loads the corresponding
    training samples, downsamples events per sample, fits a fresh classifier instance,
    performs predictions, computes evaluation metrics, and tracks runtime performance.
    Results are structured into a grid of metrics which can later be aggregated or visualized.

    Flags:
        - data_set: Dataset on which to run the experiment.
        - classifier: Chooses 'som' or 'fcnn'.
        - inference: Whether to do the training or just load and print previously generated results.
        - preprocessing_trafo: Transformation applied to the input features.

    Saves:
        - Trained classifier for each (n_samples, n_events) setting.
        - Runtime statistics (fit and prediction).
        - Predictions (full test set and sample-wise).
        - Evaluation metrics:
            * Average, class-wise, and confusion matrices.
            * Sample-wise precision, recall, F1.
            * Abstention counts (if enabled).
        - Summary metric grids (`res_df_*_*.csv`) across all settings.
        - Confusion matrices for each test sample.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from validation.utils.val_utils import n_samples_experiment_helper
    from flagx.gating import SomClassifier, MLPClassifier
    from validation.plt import plot_n_samples_n_events

    # ### Set flags and important variables here #######################################################################
    data_set = 'lymphoma_tube2_binary'
    # 'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'

    random_sample_order = True

    classifier = 'som'  # 'som', 'fcnn'

    inference = True

    preprocessing_trafo = 'log10_w_custom_cutoffs'

    if data_set == 'flowcyt':
        n_samples = list(range(1,6)) + list(range(10, 16, 5)) + [18, ]
    elif data_set == 'imstat':
        n_samples = list(range(1,21)) + list(range(25, 76, 5))
    else:
        n_samples = list(range(1,21)) + list(range(25, 56, 5)) + [58, ]

    n_events = [100, 1000, 5000, 10000, 20000, 50000, 'all']

    ####################################################################################################################

    if preprocessing_trafo == 'log10_w_custom_cutoffs' and data_set == 'flowcyt':
        preprocessing_trafo = 'log10_cutoff100'

    np.random.seed(42)

    base_p = './results/num_samples_num_events'

    # Load the sample order file
    sample_order_file = None
    if not random_sample_order:
        if data_set in {'imstat', 'flowcyt'}:
            print(f'No sample order available for {data_set}.')
            quit()
        else:
            fn_str = 'lymphoma_tube1' if data_set in {'lymphoma_tube1', 'lymphoma_tube1_binary'} else 'lymphoma_tube2'
            sample_order_file = os.path.join(base_p, f'sample_order_{fn_str}.txt')

    # Set the others label
    if data_set == 'imstat':
        others_label = 8
    elif data_set == 'flowcyt':
        others_label = 5
    else:
        others_label = None

    # Set the positive label
    if data_set in {'lymphoma_tube1_binary', 'lymphoma_tube2_binary'}:
        pos_label = 1
    else:
        pos_label = None

    # Set the abstention label
    abstention_label = -1 if classifier == 'som' else None

    # Set the save_path
    sample_order_str = 'random' if random_sample_order else 'ordered'
    save_p = os.path.join(
        base_p, classifier, data_set, preprocessing_trafo, sample_order_str
    )
    os.makedirs(save_p, exist_ok=True)

    # Set the data path
    data_p = os.path.join('./data/np_files', data_set, preprocessing_trafo)

    if inference:
        if classifier == 'som':
            # Instantiate the SOM classifier
            clf = SomClassifier(
                som_topology='planar',
                som_grid_type='rectangular',
                som_dimensions=(25, 25),
                neighborhood='gaussian',
                gaussian_neighborhood_sigma=0.1,
                initialization='pca',
                n_epochs=1000,
                radius_0=-0.5,
                radius_n=0.1,
                radius_cooling='exponential',
                learning_rate_0=0.1,
                learning_rate_n=0.001,
                learning_rate_decay='exponential',
                verbosity=2,
            )
        else:  # FCNN
            clf = MLPClassifier(
                layer_sizes=(128, 64, 32),
                n_epochs=20,
                data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 3},
                device=None,  # Tries to use default cuda device, if none available cpu
                verbosity=2
            )

        n_samples_experiment_helper(
            classifier=clf,
            n_samples=n_samples,
            n_events=n_events,
            data_p=data_p,
            save_p=save_p,
            abstention_label=abstention_label,
            others_label=others_label,
            pos_label=pos_label,
            sample_order_file=sample_order_file,
            track_gpu=True if classifier == 'fcnn' else False,
        )

    res_df = pd.read_csv(os.path.join(save_p, 'res_df_f1_macro.csv'), index_col=0)

    print(res_df)

    plot_n_samples_n_events(res_df=res_df, cmap_name='magma')
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'macro_f1.png'), dpi=300)


def main_random_sample_order_trials():
    """
    Runs repeated random-sample-order training trials to estimate classifier performance with randomly ordered samples.

    For each trial, a random subset of training samples is drawn, incrementally enlarged, and used
    to train either a SOM or FCNN classifier. Each model is evaluated sample-wise on the test set,
    and precision, recall, and F1 metrics (micro, macro, weighted, and binary if applicable) are
    recorded across all configurations.

    Flags:
        - data_set: Dataset on which to run the experiment.
        - classifier: Chooses 'som' or 'fcnn'.
        - max_n_samples: Maximum number of training samples per trial.
        - n_trials: Number of random sample-order trials.
        - preprocessing_trafo: Transformation applied to the input features.

    Saves:
        - The training samples selected in each trial (`train_samples.csv`).
        - Performance matrices for all metrics and modes (`res_df_*_*.csv`).
        - Incremental results for each trial and sample-count setting.
    """

    import os
    import random
    import numpy as np
    import pandas as pd

    from flagx.gating import SomClassifier, MLPClassifier
    from validation.utils import eval_wrapper_sample_wise

    # ### Set flags and important variables here #######################################################################
    data_set = 'lymphoma_tube2_binary'
    # 'lymphoma_tube1', 'lymphoma_tube1_binary', 'lymphoma_tube2', 'lymphoma_tube2_binary'

    classifier = 'som'  # 'som', 'fcnn'

    max_n_samples = 20

    n_trials = 100

    ####################################################################################################################

    preprocessing_trafo = 'log10_w_custom_cutoffs'

    # Set the positive label
    if data_set in {'lymphoma_tube1_binary', 'lymphoma_tube2_binary'}:
        pos_label = 1
    else:
        pos_label = None

    # Set the abstention label
    abstention_label = -1 if classifier == 'som' else None

    # Set the save_path
    save_p = os.path.join('./results/random_sample_order_trials', classifier, data_set, preprocessing_trafo)
    os.makedirs(save_p, exist_ok=True)

    # Set the data path
    data_p = os.path.join('./data/np_files', data_set, preprocessing_trafo)

    # Set the random seeds
    random.seed(42)
    np.random.seed(42)

    # Load the sample-wise test data
    samples_p_test = os.path.join(data_p, 'sample_wise_test')
    n_samples_test = len([f for f in os.listdir(samples_p_test) if f.startswith('x_')])
    sample_names_test = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples_test)]
    samples_x_test = [np.load(os.path.join(samples_p_test, f'x_{sn}.npy')) for sn in sample_names_test]
    samples_y_test = [np.load(os.path.join(samples_p_test, f'y_{sn}.npy')) for sn in sample_names_test]

    # Get the sample names of the train data
    samples_p_train = os.path.join(data_p, 'sample_wise_train')
    n_samples_train = len([f for f in os.listdir(samples_p_train) if f.startswith('x_')])
    sample_names_train = [f'sample_{str(i).zfill(2)}_train' for i in range(n_samples_train)]

    # Init dfs to track performance, prec, rec, f1, micro, macro, weighted, binary (if available)
    dummy_df = pd.DataFrame(np.nan, index=list(range(n_trials)), columns=list(range(1, max_n_samples + 1)))
    n_modes = 3 if pos_label is None else 4
    res_dfs = [dummy_df.copy() for _ in range(n_modes * 3)]
    metrics = ['prec', ] * n_modes + ['rec', ] * n_modes + ['f1'] * n_modes
    modes = ['micro', 'macro', 'weighted'] * 3 if pos_label is None else ['micro', 'macro', 'weighted', 'binary'] * 3

    train_samples_df = dummy_df.copy().astype(str)

    for n in range(n_trials):
        # Pick random training samples and load them
        selected_train_samples = random.sample(sample_names_train, max_n_samples)
        samples_x_train = [np.load(os.path.join(samples_p_train, f'x_{sn}.npy')) for sn in selected_train_samples]
        samples_y_train = [np.load(os.path.join(samples_p_train, f'y_{sn}.npy')) for sn in selected_train_samples]

        train_samples_df.loc[n, :] = selected_train_samples
        train_samples_df.to_csv(os.path.join(save_p, 'train_samples.csv'))

        for i in range(1, max_n_samples + 1):

            # Select the samples to train with
            current_x_trains = samples_x_train[0:i]
            current_y_trains = samples_y_train[0:i]

            # Concatenate and shuffle rows
            x_train = np.concatenate(current_x_trains, axis=0)
            y_train = np.concatenate(current_y_trains, axis=0)
            shuffle_permutation = np.random.permutation(x_train.shape[0])
            x_train = x_train[shuffle_permutation, :]
            y_train = y_train[shuffle_permutation]

            # Instantiate classifier
            if classifier == 'som':
                clf = SomClassifier(
                    som_topology='planar',
                    som_grid_type='rectangular',
                    som_dimensions=(25, 25),
                    neighborhood='gaussian',
                    gaussian_neighborhood_sigma=0.1,
                    initialization='pca',
                    n_epochs=1000,
                    radius_0=-0.5,
                    radius_n=0.1,
                    radius_cooling='exponential',
                    learning_rate_0=0.1,
                    learning_rate_n=0.001,
                    learning_rate_decay='exponential',
                    verbosity=2,
                )
            else:
                clf = MLPClassifier(
                    layer_sizes=(128, 64, 32),
                    n_epochs=20,
                    data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 3},
                    device=None,  # Tries to use default cuda device, if none available cpu
                    verbosity=2
                )

            # ### Fit the classifier
            clf.fit(X=x_train, y=y_train)

            # ### Predict
            samples_y_pred = []
            for x, sn in zip(samples_x_test, sample_names_test):
                samples_y_pred.append(clf.predict(X=x))

            # ### Evaluate
            # Compute sample-wise evaluation metrics
            out_sw = eval_wrapper_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                abstention_label=abstention_label,
                others_label=None,
                pos_label=pos_label,
                verbosity=2,
            )

            # Save results
            for res_df, metric, mode in zip(res_dfs, metrics, modes):

                if metric == 'prec':
                    out_df = out_sw[0]
                elif metric == 'rec':
                    out_df = out_sw[1]
                else:  # f1
                    out_df = out_sw[2]

                res_df.loc[n, i] = out_df.loc['mean', mode]

                res_df.to_csv(os.path.join(save_p, f'res_df_{metric}_{mode}.csv'))


def main_local_training():
    """
    Runs local training experiments for SOM or FCNN classifiers across datasets.

    This script trains a classifier on a limited number of samples and downsampled number of events per sample.
    For each dataset, a subset of training samples is selected randomly, downsampled, and used for model training.
    Evaluation is carried out on the corresponding test sets. Runtime measurements, predictions, and evaluation
    metrics are stored for further inspection.

    Flags:
       - data_sets: List of datasets to evaluate.
       - gating_method: Chooses 'som' or 'fcnn'.
       - num_samples / num_events: Number of samples and events to use per dataset.
       - preprocessing_trafo: Preprocessing applied to the data.

    Saves:
       - Selected training sample names.
       - Trained classifier and runtime statistics.
       - Sample-wise predictions and prediction timings.
       - Evaluation metrics (average and class-wise precision, recall, F1).
       - Confusion matrices for each test sample.
       - Abstention counts (if SOM is used).
    """
    import os
    import random
    import numpy as np
    import pandas as pd

    from flagx.gating import SomClassifier, MLPClassifier
    from validation.utils import eval_wrapper_sample_wise, scalability_wrapper, get_downsampling_bool


    # ### Set flags and important variables here #######################################################################
    data_sets = [
        'flowcyt', 'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary'
    ]
    others_labels = [5, 8, None, None, None, None]
    pos_labels = [None, None, None, None, 1, 1]

    gating_method = 'som'  # 'som', 'fcnn'
    som_dims = (25, 25)  # (20, 20)

    if gating_method == 'som':
        num_samples = [5, 5, 5, 5, 5, 5]
        num_events = [20000, 5000, 5000, 5000, 5000, 5000]
    else:
        num_samples = [5, 10, 15, 15, 5, 5]
        num_events = [50000, 20000, 'all', 'all', 5000, 5000]

    ####################################################################################################################

    abstention_label = -1 if gating_method == 'som' else None

    for data_set, others_label, pos_label, ns, ne in zip(data_sets, others_labels, pos_labels, num_samples, num_events):

        preprocessing_trafo = 'log10_w_custom_cutoffs' if data_set != 'flowcyt' else 'log10_cutoff100'

        # Set random seed anew in each iteration
        random.seed(42)
        np.random.seed(42)

        print(f'# ###### Data set: {data_set} ###### #')

        # Define path where results will be saved to
        dim_str = '_' + str(som_dims[0]) if som_dims[0] != 25 and gating_method == 'som' else ''
        save_p = (f'./results/local_training/{gating_method + dim_str}/{data_set}/{preprocessing_trafo}')
        os.makedirs(save_p, exist_ok=True)

        # --- Model fitting
        # Load training data
        data_p = os.path.join(os.getcwd(), f'data/np_files/{data_set}/{preprocessing_trafo}')
        data_p_train = os.path.join(data_p, 'sample_wise_train')
        n_samples_train = len(
            [fn for fn in os.listdir(data_p_train) if fn.startswith('x_')])
        sample_names_train = [f'sample_{str(i).zfill(2)}_train' for i in range(n_samples_train)]

        # Sample n sample names from list
        sample_names_train = random.sample(sample_names_train, k=ns)

        with open(os.path.join(save_p, 'train_samples.txt'), 'w') as f:
            for s in sample_names_train:
                f.write(s + "\n")

        x_trains = [np.load(os.path.join(data_p_train, f'x_{sn}.npy')) for sn in sample_names_train]
        y_trains = [np.load(os.path.join(data_p_train, f'y_{sn}.npy')) for sn in sample_names_train]

        # Downsample sample-wise
        if ne != 'all':
            keep_bools = []
            for y in y_trains:

                ds_keep_bool = get_downsampling_bool(
                    y=y, target_num_events=ne, stratified=True
                )
                keep_bools.append(ds_keep_bool)

            x_trains = [x[kb, :] for x, kb in zip(x_trains, keep_bools)]
            y_trains = [y[kb] for y, kb in zip(y_trains, keep_bools)]

        x_train = np.concatenate(x_trains)
        y_train = np.concatenate(y_trains)

        permutation_indices = np.random.permutation(y_train.shape[0])
        x_train = x_train[permutation_indices, :]
        y_train = y_train[permutation_indices]

        if gating_method == 'som':
            # Instantiate the SOM classifier
            clf = SomClassifier(
                som_topology='planar',
                som_grid_type='rectangular',
                som_dimensions=som_dims,
                neighborhood='gaussian',
                gaussian_neighborhood_sigma=0.1,
                initialization='pca',
                n_epochs=1000,
                radius_0=-0.5,
                radius_n=0.1,
                radius_cooling='exponential',
                learning_rate_0=0.1,
                learning_rate_n=0.001,
                learning_rate_decay='exponential',
                verbosity=2,
            )
        else:  # fcnn
            clf = MLPClassifier(
                layer_sizes=(128, 64, 32),
                n_epochs=20,
                data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
                device=None,  # Tries to use default cuda device, if none available cpu
                verbosity=2
            )

        # Fit and track time
        print('# ### Starting fit ...')
        def dummy_fit():
            clf.fit(X=x_train, y=y_train)
            return clf
        fit_time_df, clf = scalability_wrapper(function=dummy_fit, track_gpu=True if gating_method == 'fcnn' else False)
        fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))
        clf.save(filepath=save_p)

        # --- Prediction
        # Load the sample-wise test data
        samples_p = os.path.join(data_p, 'sample_wise_test')
        n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
        sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
        samples_x_test = [np.load(os.path.join(samples_p, f'x_{sn}.npy')) for sn in sample_names]

        print('# ### Starting sample-wise prediction ...')
        samples_y_pred = []
        samples_pred_time_dfs = []
        for x, sn in zip(samples_x_test, sample_names):
            def dummy_predict_sample():
                return clf.predict(X=x)
            pred_time_df_sample, y_pred_sample = scalability_wrapper(function=dummy_predict_sample)
            pred_time_df_sample['sample_name'] = sn
            samples_y_pred.append(y_pred_sample)
            samples_pred_time_dfs.append(pred_time_df_sample)

        samples_pred_times_df = pd.concat(samples_pred_time_dfs, ignore_index=True)
        samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

        os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
        for y, sn in zip(samples_y_pred, sample_names):
            np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y)

        # --- Evaluation
        # Load the labels of the sample-wise test data
        samples_p = os.path.join(data_p, 'sample_wise_test')
        n_samples = len([f for f in os.listdir(samples_p) if f.startswith('y_')])
        sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
        samples_y_test = [np.load(os.path.join(samples_p, f'y_{sn}.npy')) for sn in sample_names]

        # Compute sample-wise evaluation metrics
        out_sw = eval_wrapper_sample_wise(
            y_trues=samples_y_test,
            y_preds=samples_y_pred,
            abstention_label=abstention_label,
            others_label=others_label,
            pos_label=pos_label,
            verbosity=2,
        )

        out_sw[0].to_csv(os.path.join(save_p, 'res_df_sw_avg_prec.csv'))
        out_sw[1].to_csv(os.path.join(save_p, 'res_df_sw_avg_rec.csv'))
        out_sw[2].to_csv(os.path.join(save_p, 'res_df_sw_avg_f1.csv'))

        out_sw[3].to_csv(os.path.join(save_p, 'res_df_sw_cw_prec.csv'))
        out_sw[4].to_csv(os.path.join(save_p, 'res_df_sw_cw_rec.csv'))
        out_sw[5].to_csv(os.path.join(save_p, 'res_df_sw_cw_f1.csv'))

        if gating_method == 'som':
            out_sw[7].to_csv(os.path.join(save_p, 'abst_counts.csv'))

        os.makedirs(os.path.join(save_p, 'confusion_matrices_sw'), exist_ok=True)
        for cf_df, sn in zip(out_sw[6], sample_names):
            cf_df.to_csv(os.path.join(save_p, 'confusion_matrices_sw', f'cf_mat_{sn}.csv'))


def main_probabilistic_predictions():
    """
    Generates probabilistic predictions for previously trained classifiers.

    This script loads SOM and FCNN models that were trained in the local training experiments and applies them to both
    concatenated and sample-wise test data for the binary versions of the lymphoma datasets (LT1b, LT2b).
    For each model, class probabilities and hard predictions are computed and saved for downstream analysis.

    Flags:
        - data_sets: Datasets for which probabilistic predictions are generated.
        - methods: Trained model variants to evaluate (e.g., 'som_20', 'fcnn').
        - trafo: Preprocessing transformation used during training.

    Saves:
        - Concatenated test-set probabilities and predictions.
        - Sample-wise probabilities and predictions for each test sample.
    """
    import os
    import numpy as np

    from flagx.gating import SomClassifier, MLPClassifier

    # ### Set flags and important variables here #######################################################################
    data_sets = ['lymphoma_tube1_binary', 'lymphoma_tube2_binary']
    methods = ['som_20', 'fcnn']
    trafo = 'log10_w_custom_cutoffs'
    ####################################################################################################################

    for data_set in data_sets:
        # Load the test data
        data_p = os.path.join(os.getcwd(), 'data/np_files', data_set, trafo)
        x_test = np.load(os.path.join(data_p, 'x_test.npy'))
        # Load the sample-wise test data
        samples_p = os.path.join(data_p, 'sample_wise_test')
        n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
        sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
        samples_x_test = [np.load(os.path.join(samples_p, f'x_{sn}.npy')) for sn in sample_names]

        for m in methods:
            print(f'# ### Dataset: {data_set}, method: {m}')

            # Load the previously trained classifier
            if m == 'som_20':
                clf = SomClassifier.load(filepath=os.path.join('results/local_training', m, data_set, trafo))
            else:  # 'fcnn'
                clf = MLPClassifier.load(filepath=os.path.join('results/local_training', m, data_set, trafo))

            # Predict probabilities and save predictions
            save_p = os.path.join(os.getcwd(), 'results/probabilistic_predictions', m, data_set, trafo)
            os.makedirs(save_p, exist_ok=True)

            y_proba = clf.predict_proba(X=x_test)
            y_pred = clf.predict(X=x_test)

            np.save(os.path.join(save_p, 'y_proba.npy'), y_proba)
            np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)

            samples_y_proba = []
            samples_y_pred = []
            for x in samples_x_test:
                samples_y_proba.append(clf.predict_proba(X=x))
                samples_y_pred.append(clf.predict(X=x))

            os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
            os.makedirs(os.path.join(save_p, 'samples_y_proba'), exist_ok=True)

            for y_proba, y_pred, sn in zip(samples_y_proba, samples_y_pred, sample_names):
                np.save(os.path.join(save_p, 'samples_y_proba', f'y_proba_{sn}.npy'), y_proba)
                np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y_pred)


def main_time_table_aggregation():
    """
    Aggregates runtime and memory usage statistics across methods and datasets.

    This script collects fit-time, prediction-time, and memory-consumption statistics
    from the results of different gating methods (GateMeClass, DGCyTOF, FCNN, SOM-classifier)
    applied to multiple datasets. It parses the output files generated during full-data
    and local-training experiments, computes summary statistics (means and standard deviations),
    converts selected fields into human-readable formats, and stores aggregated tables
    for further analysis.

    Saves:
        - Table with exact values (for local training and training on all data).
        - Table with rounded values in human-readable format (for local training and training on all data).
    """

    import os
    import numpy as np
    import pandas as pd

    from validation.utils import get_time_str, get_gb_mb_str


    # Convert dataset names to corresponding dir names
    dataset_to_dir = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    # Convert method names to corresponding dir names
    method_to_dir = {
        'GateMeClass': 'gatemeclass',
        'DGCyTOF': 'dgcytof', 'FCNN': 'fcnn',
        'SOM-classifier': 'som',
    }

    # --- All data results
    print('# --- All data --- #')
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT2', 'LT1b', 'LT2b']
    methods = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-classifier']
    base_path = './results/gating_performance'
    fit_times = []
    pred_times_sample_wise_avg = []
    pred_times_sample_wise_std = []
    fit_mem_peaks_cpu = []
    pred_mem_peaks_cpu_sample_wise_avg = []
    pred_mem_peaks_cpu_sample_wise_std = []
    fit_mem_peaks_gpu = []
    pred_mem_peaks_gpu_sample_wise_avg = []
    datasets_df = []
    methods_df = []

    for dataset in datasets:
        for method in methods:

            data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            if method == 'GateMeClass':
                data_trafo = 'arcsinh_cofactor150'

            fit_data_path = os.path.join(
                base_path, method_to_dir[method], dataset_to_dir[dataset], data_trafo, 'fit_time_df.csv'
            )
            pred_data_path = os.path.join(
                base_path, method_to_dir[method], dataset_to_dir[dataset], data_trafo, 'samples_pred_times_df.csv'
            )

            try:
                fit_df = pd.read_csv(fit_data_path, index_col=0)
                pred_df = pd.read_csv(pred_data_path, index_col=0)

                fit_times.append(fit_df.loc[0, 'wall_time'])
                pred_times_sample_wise_avg.append(pred_df['wall_time'].mean())
                pred_times_sample_wise_std.append(pred_df['wall_time'].std())
                fit_mem_peaks_cpu.append(fit_df.loc[0, 'mem_peak_cpu'])
                pred_mem_peaks_cpu_sample_wise_avg.append(pred_df['mem_peak_cpu'].mean())
                pred_mem_peaks_cpu_sample_wise_std.append(pred_df['mem_peak_cpu'].std())
                fit_mem_peaks_gpu.append(fit_df.loc[0, 'mem_peak_gpu'])
                pred_mem_peaks_gpu_sample_wise_avg.append(pred_df['mem_peak_gpu'].mean())

            except FileNotFoundError:
                fit_times.append(np.nan)
                pred_times_sample_wise_avg.append(np.nan)
                fit_mem_peaks_cpu.append(np.nan)
                pred_mem_peaks_cpu_sample_wise_avg.append(np.nan)
                fit_mem_peaks_gpu.append(np.nan)
                pred_mem_peaks_gpu_sample_wise_avg.append(np.nan)

            datasets_df.append(dataset)
            methods_df.append(method)

    res_df = pd.DataFrame({
        'dataset': datasets_df,
        'method': methods_df,
        'fit_time': fit_times,
        'pred_time_mean': pred_times_sample_wise_avg,
        'pred_time_std': pred_times_sample_wise_std,
        'fit_mem_peak_cpu': fit_mem_peaks_cpu,
        'pred_mem_peak_cpu_mean': pred_mem_peaks_cpu_sample_wise_avg,
        'pred_mem_peak_cpu_std': pred_mem_peaks_cpu_sample_wise_std,
        'fit_mem_peak_gpu': fit_mem_peaks_gpu,
        'pred_mem_peak_gpu': pred_mem_peaks_gpu_sample_wise_avg,
    })

    res_df_hr = res_df.copy()
    res_df_hr['fit_time'] = res_df_hr['fit_time'].apply(get_time_str)
    res_df_hr['pred_time_mean'] = res_df_hr['pred_time_mean'].apply(get_time_str)
    res_df_hr['pred_time_std'] = res_df_hr['pred_time_std'].apply(get_time_str)
    res_df_hr['fit_mem_peak_cpu'] = res_df_hr['fit_mem_peak_cpu'].apply(get_gb_mb_str)
    res_df_hr['pred_mem_peak_cpu_mean'] = res_df_hr['pred_mem_peak_cpu_mean'].apply(get_gb_mb_str)
    res_df_hr['pred_mem_peak_cpu_std'] = res_df_hr['pred_mem_peak_cpu_std'].apply(get_gb_mb_str)
    res_df_hr['fit_mem_peak_gpu'] = res_df_hr['fit_mem_peak_gpu'].apply(get_gb_mb_str)

    # print(res_df)
    print(res_df_hr[['dataset', 'method', 'pred_mem_peak_cpu_mean', 'pred_mem_peak_cpu_std']])

    save_dir = './results/time_table_aggregation'
    os.makedirs(save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(save_dir, f'time_table_all_data_exact.csv'))
    res_df_hr.to_csv(os.path.join(save_dir, f'time_table_all_data_exact_human_readable.csv'))

    print('\n# --- Local training --- #')

    # --- Local training results
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT2', 'LT1b', 'LT2b']
    methods = ['FCNN', 'SOM-classifier']
    base_path = 'results/local_training'
    fit_times = []
    pred_times_sample_wise_avg = []
    pred_times_sample_wise_std = []
    fit_mem_peaks_cpu = []
    pred_mem_peaks_cpu_sample_wise_avg = []
    pred_mem_peaks_cpu_sample_wise_std = []
    fit_mem_peaks_gpu = []
    pred_mem_peaks_gpu_sample_wise_avg = []
    datasets_df = []
    methods_df = []

    for dataset in datasets:
        for method in methods:

            data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'

            res_path = os.path.join(
                base_path,
                method_to_dir[method] + ('_20' if method == 'SOM-classifier' else ''),
                dataset_to_dir[dataset],
                data_trafo,
            )
            fit_data_path = os.path.join(res_path,  'fit_time_df.csv')
            pred_data_path = os.path.join(res_path, 'samples_pred_times_df.csv')

            try:
                fit_df = pd.read_csv(fit_data_path, index_col=0)
                pred_df = pd.read_csv(pred_data_path, index_col=0)

                fit_times.append(fit_df.loc[0, 'wall_time'])
                pred_times_sample_wise_avg.append(pred_df['wall_time'].mean())
                pred_times_sample_wise_std.append(pred_df['wall_time'].std())
                fit_mem_peaks_cpu.append(fit_df.loc[0, 'mem_peak_cpu'])
                pred_mem_peaks_cpu_sample_wise_avg.append(pred_df['mem_peak_cpu'].mean())
                pred_mem_peaks_cpu_sample_wise_std.append(pred_df['mem_peak_cpu'].std())
                fit_mem_peaks_gpu.append(fit_df.loc[0, 'mem_peak_gpu'])
                pred_mem_peaks_gpu_sample_wise_avg.append(pred_df['mem_peak_gpu'].mean())

            except FileNotFoundError:
                fit_times.append(np.nan)
                pred_times_sample_wise_avg.append(np.nan)
                fit_mem_peaks_cpu.append(np.nan)
                pred_mem_peaks_cpu_sample_wise_avg.append(np.nan)
                fit_mem_peaks_gpu.append(np.nan)
                pred_mem_peaks_gpu_sample_wise_avg.append(np.nan)

            datasets_df.append(dataset)
            methods_df.append(method)

    res_df = pd.DataFrame({
        'dataset': datasets_df,
        'method': methods_df,
        'fit_time': fit_times,
        'pred_time_mean': pred_times_sample_wise_avg,
        'pred_time_std': pred_times_sample_wise_std,
        'fit_mem_peak_cpu': fit_mem_peaks_cpu,
        'pred_mem_peak_cpu_mean': pred_mem_peaks_cpu_sample_wise_avg,
        'pred_mem_peak_cpu_std': pred_mem_peaks_cpu_sample_wise_std,
        'fit_mem_peak_gpu': fit_mem_peaks_gpu,
        'pred_mem_peak_gpu': pred_mem_peaks_gpu_sample_wise_avg,
    })

    res_df_hr = res_df.copy()
    res_df_hr['fit_time'] = res_df_hr['fit_time'].apply(get_time_str)
    res_df_hr['pred_time_mean'] = res_df_hr['pred_time_mean'].apply(get_time_str)
    res_df_hr['pred_time_std'] = res_df_hr['pred_time_std'].apply(get_time_str)
    res_df_hr['fit_mem_peak_cpu'] = res_df_hr['fit_mem_peak_cpu'].apply(get_gb_mb_str)
    res_df_hr['pred_mem_peak_cpu_mean'] = res_df_hr['pred_mem_peak_cpu_mean'].apply(get_gb_mb_str)
    res_df_hr['pred_mem_peak_cpu_std'] = res_df_hr['pred_mem_peak_cpu_std'].apply(get_gb_mb_str)
    res_df_hr['fit_mem_peak_gpu'] = res_df_hr['fit_mem_peak_gpu'].apply(get_gb_mb_str)

    # print(res_df)
    # print(res_df_hr[['dataset', 'method', 'pred_time_mean', 'pred_time_std', 'pred_mem_peak_cpu_mean', 'pred_mem_peak_cpu_std']])

    res_df.to_csv(os.path.join(save_dir, f'time_table_local_training_exact.csv'))
    res_df_hr.to_csv(os.path.join(save_dir, f'time_table_local_training_exact_human_readable.csv'))


def main_pipeline_workflow():
    """
    End-to-end workflow for training and applying FLAG-X gating pipelines on raw FCS data.

    This script:
        1. Loads predefined training and test FCS files for the selected dataset
           (Imstat, LT1, or LT2), applies channel alignment, preprocessing, and
           optional relabeling.
        2. Trains two gating pipelines (if `train=True`):
               • SOM-classifier pipeline
               • FCNN-Softmax pipeline
           and saves the trained models.
        3. Runs inference with both pipelines on training data, concatenated
           test data, and test samples individually. Annotated FCS files
           are written to disk.
        4. Generates diagnostic scatter plots for SOM/PCA/UMAP embeddings and
           predicted gating labels to validate the output.

    Saves:
        - Trained pipeline `.pkl` files
        - Annotated FCS files
        - Auxiliary files
    """

    import os
    import random
    import readfcs
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    random.seed(42)

    from seaborn import scatterplot
    from flagx import GatingPipeline

    train = True

    # ###### Initial training ###### #
    # ### Set parameters
    dataset = 'LT1'  # Imstat, LT1, LT2

    if dataset == 'Imstat':
        channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
        label_key = 'population'
        cutoff_dict = {
            'FS INT': 100000, 'SS INT': 20000, '16-FITC': 250, '56-PE': 450, '3-ECD': 700,
            '4-PC7': 1200,
            '19-APC': 1700,
            '14-APC700': 900,
            '8-PB': 450, '45-CO': 500
        }
        relabel_data_kwargs = None
        train_data_fns = [
            '20150312-1 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019511 001.fcs',
            'ER_000000_H1_150305_ED.fcs',
            '20150320-1 IOTest Test 16-56-3-4-19-14-8-45 00019651 001.fcs',
            '20150317-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019558 001.fcs',
            '20150318-3 VersaLyse VersaLyseFix 16-56-3-4-19-14-8-45 00019593 001.fcs',
            'ER_000050_H1_150311_ED.fcs',
            '20150312-2 IOTest Test 16-56-3-4-19-14-8-45 00019495 001.fcs',
            '20150312-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019496 001.fcs',
            '20150320-1 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019654 001.fcs',
            'ER_000018_H1_150306_ED.fcs',
            '20150320-3 IOTest Test 16-56-3-4-19-14-8-45 00019661 001.fcs',
            '20150319-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019617 001.fcs',
            '20150320-2 VersaLyse VersaLyse 16-56-3-4-19-14-8-45 00019658 001.fcs',
            'ER_000025_H1_150309_ED.fcs',
            '20150318-3 IOTest Test 16-56-3-4-19-14-8-45 00019588 001.fcs',
            '20150318-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019572 001.fcs',
            '20150317-2 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019557 001.fcs',
            'ER_000006_H1_150305_ED.fcs',
            '20150312-1 IOTest Test 16-56-3-4-19-14-8-45 00019508 001.fcs',
            '20150319-3 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019637 001.fcs',
        ]
        test_data_fns = [
            'ER_000057_H1_150311_ED.fcs',
            '20150318-2 Quick Quick 16-56-3-4-19-14-8-45 00019581 001.fcs',
            '20150318-1 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019574 001.fcs',
            '20150320-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019657 001.fcs',
            '20150318-1 IOTest Test 16-56-3-4-19-14-8-45 00019571 001.fcs',
            'ER_000001_H1_150305_ED.fcs',
            '20150320-3 Quick Quick 16-56-3-4-19-14-8-45 00019660 001.fcs',
            '20150312-2 VersaLyse VersaLyse 16-56-3-4-19-14-8-45 00019500 001.fcs',
            '20150318-3 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019592 001.fcs',
            '20150319-3 IOTest Test 16-56-3-4-19-14-8-45 00019619 001.fcs',
        ]

    elif dataset == 'LT1':
        channels = [
            'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7',
            'CD38_APC', 'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
        ]
        label_key = 'population'
        cutoff_dict = {
            'FS': 100000, 'SS': 20000, 'kappavCD8_FITC': 500, 'lambdavCD7_PE': 400, 'CD23_ECD': 500,
            'CD79bvCD4_PC5.5': 1200, 'CD5_PC7': 300, 'CD38_APC': 700,
            'CD19_APC_A700': 150,
            'CD20vCD3_APC_A750': 500, 'FMC7vCD2_PB': 500, 'CD45_KrOr': 1000
        }
        relabel_data_kwargs = {'old_to_new_label_mapping': {1: 1, 9: 1, 7: 0, 10: 0}, 'new_label_key': 'binary_labels'}
        train_data_fns = [
            '5680_NB_T1_d13_N100k.fcs',
            '2400_CLL_T1_d13_N100k.fcs',
            '3280_DLBCL_T1_d13_N73k.fcs',
            '1470_MCL_T1_d13_N100k.fcs',
            '0470_FL_T1_d13_N100k.fcs',
            '2740_HCL_T1_d13_N40k.fcs',
            '2660_LPL_T1_d13_N100k.fcs',
            '0020_MZL_T1_d13_N100k.fcs',
            '3260_MBL_T1_d13_N100k.fcs',
            '3390_UC_T1_d13_N100k.fcs',
            '3110_BL_T1_d13_N100k.fcs',
            '5630_NB_T1_d13_N100k.fcs',
            '2170_CLL_T1_d13_N100k.fcs',
            '2630_DLBCL_T1_d13_N100k.fcs',
            '1550_MCL_T1_d13_N100k.fcs',
            '1990_FL_T1_d13_N100k.fcs',
            '3630_HCL_T1_d13_N100k.fcs',
            '1740_LPL_T1_d13_N100k.fcs',
            '0200_MZL_T1_d13_N77k.fcs',
            '2620_MBL_T1_d13_N100k.fcs',
            '2210_UC_T1_d13_N100k.fcs',
        ]
        test_data_fns = [
            '6080_NB_T1_d13_N100k.fcs',
            '0910_CLL_T1_d13_N100k.fcs',
            '1920_DLBCL_T1_d13_N100k.fcs',
            '3910_MCL_T1_d13_N100k.fcs',
            '0290_FL_T1_d13_N100k.fcs',
            '2640_HCL_T1_d13_N69k.fcs',
            '2420_LPL_T1_d13_N100k.fcs',
            '3730_MZL_T1_d13_N100k.fcs',
            '3850_MBL_T1_d13_N100k.fcs',
            '1580_UC_T1_d13_N100k.fcs',
            '1230_BL_T1_d13_N100k.fcs',
        ]

    else:  # LT2
        channels = [
            'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7',
            'CD52_APC', 'CD11c_APC_A700', 'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
        ]
        label_key = 'population'
        cutoff_dict = {
            'FS': 100000, 'SS': 20000, 'CD103_FITC': 300, 'CD43_PE': 1500, 'CD25_ECD': 1000,
            'CD10_PC5.5': 1000, 'CD200_PC7': 1000, 'CD52_APC': 150, 'CD11c_APC_A700': 200,
            'CD20_APC_A750': 300, 'IgM_PB': 400, 'CD19_KrOr': 200,
        }
        relabel_data_kwargs = {'old_to_new_label_mapping': {1: 1, 9: 1, 7: 0, 10: 0}, 'new_label_key': 'binary_labels'}
        train_data_fns = [
            '5680_NB_T2_d13_N100k.fcs',
            '2400_CLL_T2_d13_N100k.fcs',
            '3280_DLBCL_T2_d13_N73k.fcs',
            '1470_MCL_T2_d13_N100k.fcs',
            '0470_FL_T2_d13_N100k.fcs',
            '2740_HCL_T2_d13_N40k.fcs',
            '2660_LPL_T2_d13_N100k.fcs',
            '0020_MZL_T2_d13_N100k.fcs',
            '3260_MBL_T2_d13_N100k.fcs',
            '3390_UC_T2_d13_N100k.fcs',
            '3110_BL_T2_d13_N100k.fcs',
            '5630_NB_T2_d13_N100k.fcs',
            '2170_CLL_T2_d13_N100k.fcs',
            '2630_DLBCL_T2_d13_N100k.fcs',
            '1550_MCL_T2_d13_N100k.fcs',
            '1990_FL_T2_d13_N100k.fcs',
            '3630_HCL_T2_d13_N100k.fcs',
            '1740_LPL_T2_d13_N100k.fcs',
            '0200_MZL_T2_d13_N77k.fcs',
            '2620_MBL_T2_d13_N100k.fcs',
            '2210_UC_T2_d13_N100k.fcs',
        ]
        test_data_fns = [
            '6080_NB_T2_d13_N100k.fcs',
            '0910_CLL_T2_d13_N100k.fcs',
            '1920_DLBCL_T2_d13_N100k.fcs',
            '3910_MCL_T2_d13_N100k.fcs',
            '0290_FL_T2_d13_N100k.fcs',
            '2640_HCL_T2_d13_N69k.fcs',
            '2420_LPL_T2_d13_N100k.fcs',
            '3730_MZL_T2_d13_N100k.fcs',
            '3850_MBL_T2_d13_N100k.fcs',
            '1580_UC_T2_d13_N100k.fcs',
            '1230_BL_T2_d13_N100k.fcs',
        ]

    dataset_to_dir = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels',
        'LT2': 'lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels',
    }

    data_subdir = dataset_to_dir[dataset]

    save_path = os.path.join('results/pipeline_workflow', dataset)
    os.makedirs(save_path, exist_ok=True)

    data_dir = os.path.join('./data/raw', data_subdir)

    with open(os.path.join(save_path, 'train_samples.txt'), 'w') as f:
        for line in train_data_fns:
            f.write(line + '\n')

    with open(os.path.join(save_path, 'test_samples.txt'), 'w') as f:
        for line in test_data_fns:
            f.write(line + '\n')

    preprocessing_kwargs = {'flavour': 'log10_w_custom_cutoffs', 'flavour_kwargs': {'cutoffs': cutoff_dict}}

    save_path_som = os.path.join(save_path, 'som')
    os.makedirs(save_path_som, exist_ok=True)

    save_path_fcnn = os.path.join(save_path, 'fcnn')
    os.makedirs(save_path_fcnn, exist_ok=True)

    if train:
        # ### Train the SOM-classifier gating pipeline
        som_kwargs = {
            'som_topology': 'planar',
            'som_grid_type': 'rectangular',
            'som_dimensions': (25, 25),
            'neighborhood': 'gaussian',
            'gaussian_neighborhood_sigma': 0.1,
            'initialization': 'pca',
            'n_epochs': 1000,
            'radius_0': -0.5,
            'radius_n': 0.1,
            'radius_cooling': 'exponential',
            'learning_rate_0': 0.1,
            'learning_rate_n': 0.001,
            'learning_rate_decay': 'exponential',
            'verbosity': 2
        }
        gp_som = GatingPipeline(
            train_data_file_path=data_dir,
            train_data_file_names=train_data_fns,
            train_data_file_type='fcs',
            save_path=save_path_som,
            channels=channels,
            label_key=label_key,
            channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
            relabel_data_kwargs=relabel_data_kwargs,
            preprocessing_kwargs=preprocessing_kwargs,
            gating_method='som',
            gating_method_kwargs=som_kwargs,
            verbosity=2,
        )
        gp_som.train()
        gp_som.save(filename='trained_pipeline_som.pkl')

        # ### Train the FCNN-softmax-classifier gating pipeline
        fcnn_kwargs = {'layer_sizes': (128, 64, 32), 'n_epochs': 20, 'device': 'cuda', 'verbosity': 2}
        gp_fcnn = GatingPipeline(
            train_data_file_path=data_dir,
            train_data_file_names=train_data_fns,
            train_data_file_type='fcs',
            save_path=save_path_fcnn,
            channels=channels,
            label_key=label_key,
            channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
            relabel_data_kwargs=relabel_data_kwargs,
            preprocessing_kwargs=preprocessing_kwargs,
            gating_method='fcnn',
            gating_method_kwargs=fcnn_kwargs,
            verbosity=2,
        )
        gp_fcnn.train()
        gp_fcnn.save(filename='trained_pipeline_fcnn.pkl')
        del gp_som, gp_fcnn

    # ###### Inference with new data ###### #
    # ### Set parameters
    output_dir = os.path.join(save_path, 'output')
    output_dir_test_samples = os.path.join(output_dir, 'test_samples')
    os.makedirs(output_dir_test_samples, exist_ok=True)

    dim_red_methods = ('som', 'pca', 'umap')
    dim_red_method_kwargs = (None, None, {'n_jobs': 12})

    # ### Inference with the SOM pipeline
    gp_som = GatingPipeline.load(filename='trained_pipeline_som.pkl', filepath=save_path_som)
    gp_som.verbosity = 2
    # Training data
    gp_som.inference(
        data_file_path=data_dir,
        data_file_names=train_data_fns,
        sample_wise=False,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_path=output_dir,
        save_filename='annotated_train_data.fcs',
        scale_channels=[label_key, ],
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
    )
    # Test data samples concatenated
    gp_som.inference(
        data_file_path=data_dir,
        data_file_names=test_data_fns,
        sample_wise=False,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_path=output_dir,
        save_filename='annotated_test_data.fcs',
        scale_channels=[label_key],
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
    )
    # Test data individual samples
    gp_som.inference(
        data_file_path=data_dir,
        data_file_names=test_data_fns,
        sample_wise=True,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_path=output_dir_test_samples,
        save_filename='annotated_test_data.fcs',
        scale_channels=[label_key],
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
    )

    # ### Inference with the FCNN pipeline
    gp_fcnn = GatingPipeline.load(filename='trained_pipeline_fcnn.pkl', filepath=save_path_fcnn)
    # Training data
    gp_fcnn.inference(
        data_file_path=output_dir,
        data_file_names=['annotated_train_data.fcs', ],
        gate=True,
        dim_red_methods=None,
        dim_red_method_kwargs=None,
        save_path=output_dir,
        save_filename='annotated_train_data.fcs',
        scale_channels=None,
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
    )
    # Test data samples concatenated
    gp_fcnn.inference(
        data_file_path=output_dir,
        data_file_names=['annotated_test_data.fcs', ],
        gate=True,
        dim_red_methods=None,
        dim_red_method_kwargs=None,
        save_path=output_dir,
        save_filename='annotated_test_data.fcs',
        scale_channels=None,
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
    )
    # Test data individual samples
    gp_fcnn.inference(
        data_file_path=output_dir_test_samples,
        data_file_names=[f'annotated_test_data_sample_id_{i + 1}.fcs' for i in range(len(test_data_fns))],
        gate=True,
        dim_red_methods=None,
        dim_red_method_kwargs=None,
        save_path=output_dir_test_samples,
        save_filename='annotated_test_data.fcs',
        scale_channels=None,
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
    )

    del gp_som, gp_fcnn

    # ###### Output validation ###### #
    annotated_test_data = readfcs.read(os.path.join(output_dir, 'annotated_test_data.fcs'))
    print("# ### Annotated test data:\n", annotated_test_data)
    df = annotated_test_data.to_df()
    print("# Channels:\n", df.columns)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=df, x=f'{drm}_1', y=f'{drm}_2', s=1, hue='sample_id_unscaled', palette='deep', ax=ax)
        plt.legend(title='Sample ID', markerscale=4)
        plt.savefig(os.path.join(output_dir, f'sample_id_dimred_{drm}.png'), dpi=300)
        plt.close('all')
        for gm in ['som', 'fcnn']:
            fig, ax = plt.subplots(dpi=300)
            scatterplot(data=df, x=f'{drm}_1', y=f'{drm}_2', s=1, hue=f'pred_{gm}_unscaled', palette='deep', ax=ax)
            plt.legend(title='Pred', markerscale=4)
            plt.savefig(os.path.join(output_dir, f'gating_{gm}_dimred_{drm}.png'), dpi=300)
            plt.close('all')


def main_pipeline_output_downsampling():

    import os
    import numpy as np

    from flagx.io import FlowDataManager, export_to_fcs
    from validation.utils.val_utils import get_downsampling_bool

    np.random.seed(42)

    # ### Set parameters
    datasets = ['Imstat', 'LT1', 'LT2']
    data_files = ['annotated_train_data.fcs', 'annotated_test_data.fcs']
    label_key = 'population'
    sample_id_key = 'sample_id'

    target_num_events = 100000

    double_stratified = False
    # Stratify w.r.t. num events per sample and cell types, if False fixed  num events per sample

    for dataset in datasets:

        results_path = os.path.join(os.getcwd(), 'results/pipeline_workflow', dataset, 'output')

        for data_file in data_files:

            # Instantiate a datamanager
            fdm = FlowDataManager(
                data_file_names=[data_file, ],
                data_file_type=None,
                data_file_path=results_path,
                save_path=results_path,
                verbosity=2,
            )

            # Load data file to anndata
            fdm.load_data_files_to_anndata()

            adata = fdm.anndata_list_[0]

            # Extract the labels
            col_index = adata.var_names.get_loc(label_key)
            labels = adata.X[:, col_index]

            # Extract the sample ids
            col_index = adata.var_names.get_loc(sample_id_key)
            sample_ids = adata.X[:, col_index]

            if double_stratified:  # ### Double stratified downsampling

                # Get the downsampling bool where stratification w.r.t. num events per sample is used
                ds_bool_sample_based = get_downsampling_bool(
                    y=sample_ids,
                    target_num_events=target_num_events,
                    stratified=True
                )

                # Get the event count per sample
                # (used as target num events for population size-based, sample-wise stratified downsampling)
                sample_ids_downsampled = sample_ids[ds_bool_sample_based]
                sample_ids_unique, counts = np.unique(sample_ids_downsampled, return_counts=True)

                # Downsample per sample, stratify w.r.t. population sizes
                ds_bool = np.zeros_like(sample_ids).astype(bool)
                for sample_id, count in zip(sample_ids_unique, counts):

                    sample_id_bool = (sample_ids == sample_id)

                    labels_current_sample = labels[sample_id_bool]

                    ds_bool_current_sample = get_downsampling_bool(
                        y=labels_current_sample,
                        target_num_events=count,
                        stratified=True
                    )

                    ds_bool[sample_id_bool] = ds_bool_current_sample
            else:  # ### Same num events per sample, stratify w.r.t. population sizes

                # Define num events per sample such that target_num_events is reached
                unique_sample_ids = np.unique(sample_ids)
                num_samples = unique_sample_ids.shape[0]
                base = target_num_events // num_samples
                remainder = target_num_events % num_samples
                events_per_sample = np.full(num_samples, base, dtype=int)
                events_per_sample[:remainder] += 1

                # Downsample per sample, stratify w.r.t. population sizes
                ds_bool = np.zeros_like(sample_ids).astype(bool)
                for sample_id, num_events in zip(unique_sample_ids, events_per_sample):
                    sample_bool = (sample_ids == sample_id)
                    labels_current_sample = labels[sample_bool]

                    ds_bool_current_sample = get_downsampling_bool(
                        y=labels_current_sample,
                        target_num_events=num_events,
                        stratified=True
                    )
                    ds_bool[sample_bool] = ds_bool_current_sample

            # Apply downsampling
            adata_downsampled = adata[ds_bool, :].copy()

            print(f'# ### Num events before: {adata.n_obs}, after: {adata_downsampled.n_obs}')
            print(adata_downsampled)

            export_to_fcs(
                data_list=[adata_downsampled, ],
                save_path=results_path,
                save_filenames=data_file[:-4] + f'_downsampled_{target_num_events}_events.fcs',
            )


if __name__ == '__main__':

    # main_data_processing()

    # main_som_parameter_influence_study()

    # main_som_parameter_tuning()

    # main_som_n_epochs_calibration()

    # main_som_classifier()

    # main_gatemeclass()

    # main_dgcytof()

    # main_fcnn()

    # main_num_samples_num_events_experiment()

    # main_random_sample_order_trials()

    # main_local_training()

    # main_probabilistic_predictions()

    # main_time_table_aggregation()

    # main_pipeline_workflow()

    print('done')
