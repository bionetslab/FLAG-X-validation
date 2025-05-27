

def main_data_preparation():
    """
    Script for loading the .fcs files of the datasets, perform data split on sample level,
    preprocess/apply transformation, subset to selected channels, save resulting data matrices to .npy files.
    Additionally:
     - Save data from all samples concatenated in one matrix.
     - Downsample (stratified per sample) and save sample-wise and concatenated data matrices.

    Returns:
        None
    """

    import os
    import copy
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import List, Dict, Union, Tuple, Any
    from flagx.utils import set_random_seed, log10_trafo_w_cutoff_channel_wise
    from flagx.io import FlowDataManager

    def helper(
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
            for pf in ['arcsinh', 'custom'] :

                # Set random seeds for random, numpy and torch
                seed = 42
                set_random_seed(seed=seed)

                # Set label key
                current_lk = lk

                # Define path where results should be stored and create dir
                if pf == 'arcsinh':
                    pf_str = 'arcsinh_cofactor150'
                else:
                    if cd is not None:  # If passed, channel wise cutoffs will be applied ...
                        pf_str = 'log10_channelwisecutoff'
                    else:  # ... otherwise log10 with cutoff 100
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
                else:  # Custom flavour
                    if cd is not None:  # If passed, apply channel wise cutoffs ...
                        prepr_kwargs = {
                            'preprocessing_method': log10_trafo_w_cutoff_channel_wise,
                            'channel_to_cutoff_dict': cd,
                        }
                    else:  # ... otherwise log10 with cutoff 100
                        pf = 'log10_w_cutoff'
                        prepr_kwargs = {'cutoff': 100}

                fdm.sample_wise_preprocessing(flavour=pf, save_raw_to_layer='no_trafo', **prepr_kwargs)

                # Split samples into train and test split
                split_kwargs = {'shuffle': True, 'random_state': seed}
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
                            shuffle=True,  # Used pytorch dataloader under the hood, seed is set by set_random_seed()
                            precision='32bit',  # Same as FCS
                        )

                # Downsample events per sample in train data and save to numpy file as well
                for frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    ds_train_data_list = fdm.sample_wise_downsampling_worker(
                        data_list=fdm.train_data_,
                        fraction=frac,
                        stratified=True,
                        label_key=current_lk,
                        label_layer_key='no_trafo',
                        inplace=False,
                    )

                    for sw in [False, True]:
                        dummy_save_path = os.path.join(
                            np_data_p,
                            'downsampled',
                            str(frac).replace('.', '_'),
                            '' if not sw else 'sample_wise_train'
                        )
                        os.makedirs(dummy_save_path, exist_ok=True)
                        fdm.save_to_numpy_files_worker(
                            data_list=ds_train_data_list,
                            sample_wise=sw,
                            save_path=dummy_save_path,
                            filename_suffix='_train',
                            channels=cn,
                            layer_key=None,  # Use adata.X, transformed data
                            label_key=current_lk,
                            label_layer_key='no_trafo',  # Use labels from original untransformed data
                            shuffle=True,  # Used pytorch dataloader under the hood, seed is set by set_random_seed()
                            precision='32bit',  # Same as FCS

                        )


    # ### Set flags and important variables here #######################################################################

    # ### Immunstatus data ###
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
        '4-PC7': 1200,  # In Stefans Liste: 4-CD7
        '19-APC': 1700,
        '14-APC700': 900,  # In Stefans Liste: 14-APC750
        '8-PB': 450, '45-CO': 500
    }
    label_key_imstat = 'population'
    relabel_dict_imstat = None


    # ### Lymphoma tube 1 ###
    ds_name_lt1 = 'lymphoma_tube1'
    raw_data_p_lt1 = os.path.join(
        os.getcwd(),
        'data/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels'
    )
    data_file_type_lt1 = 'fcs'
    data_split_lt1 = (0.75, 0.25)
    channels_lt1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7',
        'CD38_APC', 'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    cutoff_dict_lt1 = {
        'FS': 100000, 'SS': 20000, 'kappavCD8_FITC': 500, 'lambdavCD7_PE': 400, 'CD23_ECD': 500,
        'CD79bvCD4_PC5.5': 1200, 'CD5_PC7': 300, 'CD38_APC': 700,
        'CD19_APC_A700': 150,  # Nicht vorhanden in Stefans Liste -> verwende 150 (siehe Mail)
        'CD20vCD3_APC_A750': 500, 'FMC7vCD2_PB': 500, 'CD45_KrOr': 1000
    }
    label_key_lt1 = 'population'
    relabel_dict_lt1 = None


    # ### Lymphoma tube 1, binary case, ###
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


    # ### Lymphoma tube 2 ###
    ds_name_lt2 = 'lymphoma_tube2'
    raw_data_p_lt2 = os.path.join(
        os.getcwd(),
        'data/raw/lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels'
    )
    data_file_type_lt2 = 'fcs'
    data_split_lt2 = (0.75, 0.25)
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


    # ### Lymphoma tube 2, binary case ###
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


    # ### Flowcyt ###
    ds_name_flowcyt = 'flowcyt'
    raw_data_p_flowcyt = os.path.join(os.getcwd(), 'data/raw/flowcyt/data_original')
    data_file_type_flowcyt = 'csv'
    data_split_flowcyt = (0.75, 0.25)
    channels_flowcyt = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]
    cutoff_dict_flowcyt = None
    label_key_flowcyt = 'label'
    relabel_dict_flowcyt = None

    dataset_names = [
        ds_name_imstat,
        ds_name_lt1, ds_name_lt2,
        ds_name_lt1_binary, ds_name_lt2_binary,
        ds_name_flowcyt
    ]
    raw_data_paths = [
        raw_data_p_imstat,
        raw_data_p_lt1, raw_data_p_lt2,
        raw_data_p_lt1_binary, raw_data_p_lt2_binary,
        raw_data_p_flowcyt
    ]
    data_file_types = [
        data_file_type_imstat,
        data_file_type_lt1, data_file_type_lt2,
        data_file_type_lt1_binary, data_file_type_lt2_binary,
        data_file_type_flowcyt
    ]
    data_splits = [
        data_split_imstat,
        data_split_lt1, data_split_lt2,
        data_split_lt1_binary, data_split_lt2_binary,
        data_split_flowcyt
    ]
    channel_names = [
        channels_imstat,
        channels_lt1, channels_lt2,
        channels_lt1_binary, channels_lt2_binary,
        channels_flowcyt
    ]
    cutoff_dicts = [
        cutoff_dict_imstat,
        cutoff_dict_lt1, cutoff_dict_lt2,
        cutoff_dict_lt1_binary, cutoff_dict_lt2_binary,
        cutoff_dict_flowcyt
    ]
    label_keys = [
        label_key_imstat,
        label_key_lt1, label_key_lt2,
        label_key_lt1_binary, label_key_lt2_binary,
        label_key_flowcyt
    ]
    relabel_dicts = [
        relabel_dict_imstat,
        relabel_dict_lt1, relabel_dict_lt2,
        relabel_dict_lt1_binary, relabel_dict_lt2_binary,
        relabel_dict_flowcyt
    ]

    helper(
        dataset_names=dataset_names,
        raw_data_paths=raw_data_paths,
        data_file_types=data_file_types,
        label_keys=label_keys,
        relabel_dicts=relabel_dicts,
        cutoff_dicts=cutoff_dicts,
        data_splits=data_splits,
        channel_names=channel_names,
    )


def main_param_influence_study():
    """
    Script for running parameter-wise hyperparameter tuning on the
    immunstatus dataset. Its purpose is to determine whether the
    individual parameters have an influence on the SOM classifier's
    performance and, if so, get an intuition of the magnitude.
    The workflow is as follows:
        - Preprocessed data is loaded.
        - Data is downsampled for faster training.
        - k-fold cross validation is performed.
        - Results are printed and plotted.

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

    from flagx.io import FlowDataManager
    from flagx.gating import SomClassifier
    from validation.plt import plot_param_lineplot, plot_param_stripplot
    from validation.utils import set_pandas_print_options

    # ### Set flags and variables ######################################################################################
    inference = True  # Whether to do the hyperparameter tuning or just view the results
    test_n_epochs = False

    random_seed = 42
    downsampling_frac = 0.25
    n_splits = 3

    trafo = 'log10_channelwisecutoff'  # 'arcsinh_cofactor150', 'log10_channelwisecutoff'

    # Based on gridsearch for n_epochs, selected n_epochs such that performance is stable with default parameters
    n_epochs = 6000   # arcsinh_cofactor150: 6000, log10_channelwisecutoff: 6000
    ####################################################################################################################

    # ### Load the train data
    data_p = os.path.join(os.getcwd(), f'data/np_files/imstat/{trafo}')

    x = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
    y = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int32)

    # ### Downsample for faster inference time (4864323 * 0.2 = 972864,6)
    np.random.seed(random_seed)
    downsampling_bool = FlowDataManager._get_downsampling_bool(y=y, fraction=downsampling_frac, stratified=True)
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
    save_p = os.path.join(os.getcwd(), f'results/parameter_influence_study/{trafo}')

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


def main_param_tuning():
    """
    Script for running hyperparameter tuning on the immunstatus dataset. The workflow is as follows:
        - Preprocessed data is loaded.
        - Data is downsampled for faster training.
        - Data is split into train and val.
        - Results are printed.

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
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split, PredefinedSplit

    from flagx.io import FlowDataManager
    from flagx.gating import SomClassifier
    from validation.utils import set_pandas_print_options
    from validation.plt import plot_param_lineplot

    # ### Set flags and variables ######################################################################################
    inference = True  # Whether to do the hyperparameter tuning or just view the results
    trafo = 'arcsinh_cofactor150'  # arcsinh_cofactor150, log10_channelwisecutoff

    random_seed = 42
    downsampling_frac = 0.25
    val_frac = 0.34

    grid = 'full_grid'  # 'n_epochs_dim', 'full_grid', 'dummy'

    # Based on gridsearch for n_epochs with som_dimensions=(25, 25),
    # selected n_epochs such that performance is stable:
    # 5000 for log10_channelwisecutoff, 1000 for arcsinh_cofactor150
    n_epochs = 1000  # 5000, 1000
    ####################################################################################################################

    # ### Load the train data
    data_p = os.path.join(os.getcwd(), f'data/np_files/imstat/{trafo}')

    x = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
    y = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int32)

    # ### Downsample for faster inference time (4864323 * 0.2 = 972864,6)
    np.random.seed(random_seed)
    downsampling_bool = FlowDataManager._get_downsampling_bool(y=y, fraction=downsampling_frac, stratified=True)
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
    if grid == 'n_epochs_dim25':
        n_epochs_list = list(range(10, 101, 10)) + list(range(200, 1001, 100)) + list(range(2000, 15001, 1000))
        param_grid = {
            'som_topology': ['planar', ],
            'som_grid_type': ['rectangular', ],
            'som_dimensions': [(25, 25), ],
            'neighborhood': ['gaussian', ],
            'gaussian_neighborhood_sigma': [0.25, ],
            'initialization': ['pca', ],
            'n_epochs': n_epochs_list,
            'radius_0': [-0.5, ],
            'radius_n': [0.1, ],
            'radius_cooling': ['linear', ],
            'learning_rate_0': [0.1, ],
            'learning_rate_n': [0.01, ],
            'learning_rate_decay': ['exponential', ],
        }

    elif grid == 'full_grid':
        param_grid = {
            'som_topology': ['planar', ],
            'som_grid_type': ['rectangular', ],
            'som_dimensions': [(15, 15), (20, 20), (25, 25)],
            'neighborhood': ['gaussian', ],
            'gaussian_neighborhood_sigma': [0.25, 0.1],
            'initialization': ['pca', ],
            'n_epochs': [n_epochs, ],
            'radius_0': [-0.25, -0.5, -0.75],
            'radius_n': [0.1, 0.01],
            'radius_cooling': ['linear', ],
            'learning_rate_0': [0.1, 0.5, 1.0],
            'learning_rate_n': [0.001, 0.05, 0.1],
            'learning_rate_decay': ['exponential', ],
        }

    else:  # dummy
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

    # Define dir for saving the results
    save_p = os.path.join(os.getcwd(), f'results/parameter_tuning/{trafo}/{grid}')
    os.makedirs(save_p, exist_ok=True)

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

    if grid == 'n_epochs_dim25':
        plot_param_lineplot(
            res_df=res_df,
            x_col='param_n_epochs',
            y_col='mean_test_score',
            xlog10=True,
            xlog10plusone=False,
            custom_x_ticks='log10_scale',
            x_label='n_epochs',
            y_label='Macro F1',
            x_axis_grid=True,
            dpi=300,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_p, 'n_epochs_dim25.png'), dpi=300)
        plt.close('all')


def main_n_epochs_calibration():
    """
    Script for finding the optimal number of epochs tp train for given a set of optimized parameters.
    The workflow is as follows:
        - Preprocessed data is loaded.
        - Data is split into train and val.
        - Results are printed.

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
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split, PredefinedSplit

    from flagx.gating import SomClassifier
    from validation.utils import set_pandas_print_options
    from validation.plt import plot_param_lineplot

    # ### Set flags and variables ######################################################################################
    inference = True  # Whether to do the hyperparameter tuning or just view the results
    trafo = 'arcsinh_cofactor150'  # arcsinh_cofactor150, log10_channelwisecutoff

    random_seed = 42
    val_frac = 0.34

    # Gridsearch for n_epochs
    n_epochs = list(range(10, 101, 10)) + list(range(200, 1001, 100))
    if trafo == 'log10_channelwisecutoff':
        n_epochs += list(range(1100, 2001, 100)) + list(range(3000, 5001, 1000))
    ####################################################################################################################

    # ### Load the train data
    data_p = os.path.join(os.getcwd(), f'data/np_files/imstat/{trafo}')

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

    # ### Load the previously optimized parameters and define a parameter grid with them
    som_clf_param_tuning = SomClassifier.load(
        filepath=os.path.join(os.getcwd(), f'results/parameter_tuning/{trafo}/full_grid')
    )
    best_params = som_clf_param_tuning.grid_search_.best_params_

    print('# ### Best parameters:', best_params)

    # Define dir for saving the results
    save_p = os.path.join(os.getcwd(), f'results/n_epoch_calibration/{trafo}')
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

    import os
    import time
    import numpy as np
    import pandas as pd

    from flagx.gating import SomClassifier
    from validation.utils import get_time_str, eval_wrapper, eval_wrapper_sample_wise
    # ### Set flags and important variables here #######################################################################
    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    others_labels = [8, None, None, None, None, 5]
    pos_labels = [None, None, None, 1, 1, None]

    preprocessing_trafos = ['arcsinh_cofactor150', 'log10_channelwisecutoff', 'log10_cutoff100']

    fit = True
    predict = True
    evaluate = True
    ####################################################################################################################

    for data_set, others_label, pos_label in zip(data_sets, others_labels, pos_labels):
        for trafo in preprocessing_trafos:

            print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

            # Check whether train data exists for this dataset and trafo, if not continue
            data_p = os.path.join(os.getcwd(), f'data/np_files/{data_set}/{trafo}')
            alternate_data_p = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

            if not os.path.exists(data_p):
                data_p = alternate_data_p

            if not os.path.exists(data_p):
                print(f'# ### No data found for dataset "{data_set}" and transformation "{trafo}". Continue.\n')
                continue

            # Define path where results will be saved to
            save_p = os.path.join(os.getcwd(), f'results/pred_eval/som_classifier/{data_set}/{trafo}')
            os.makedirs(save_p, exist_ok=True)

            if fit:

                # Load training data
                x_train = np.load(os.path.join(data_p, 'x_train.npy'))
                y_train = np.load(os.path.join(data_p, 'y_train.npy'))

                # Instantiate the SOM classifier
                if trafo == 'arcsinh_cofactor150':

                    som_clf = SomClassifier(
                        som_topology='planar',
                        som_grid_type='rectangular',
                        som_dimensions=(25, 25),
                        neighborhood='gaussian',
                        gaussian_neighborhood_sigma=0.25,
                        initialization='pca',
                        n_epochs=200,
                        radius_0=-0.25,
                        radius_n=0.01,
                        radius_cooling='linear',
                        learning_rate_0=0.5,
                        learning_rate_n=0.05,
                        learning_rate_decay='exponential',
                        verbosity=2,
                    )
                else:
                    som_clf = SomClassifier(
                        som_topology='planar',
                        som_grid_type='rectangular',
                        som_dimensions=(25, 25),
                        neighborhood='gaussian',
                        gaussian_neighborhood_sigma=0.1,
                        initialization='pca',
                        n_epochs=1000,
                        radius_0=-0.25,
                        radius_n=0.1,
                        radius_cooling='linear',
                        learning_rate_0=0.1,
                        learning_rate_n=0.05,
                        learning_rate_decay='exponential',
                        verbosity=2,
                    )

                # Fit and track time
                print('# ### Starting fit ...')
                st_fit = time.time()
                som_clf.fit(X=x_train, y=y_train)
                et_fit = time.time()
                fit_time_sek = et_fit - st_fit
                fit_time_str, h, m, s = get_time_str(seconds=fit_time_sek)
                print(f'# ### Fit finished, time: {fit_time_sek} s = {fit_time_str}\n')

                fit_time_df = pd.DataFrame(
                    data=[[fit_time_sek, h, m, s]], index=['fit_time'], columns=['total s', 'h', 'm', 's']
                )
                fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))

                som_clf.save(filepath=save_p)

            else:
                # Load the SOM classifier
                som_clf = SomClassifier.load(filepath=save_p)

            if predict:
                # Load the test data
                x_test = np.load(os.path.join(data_p, 'x_test.npy'))

                print('# ### Starting prediction ...')
                st_pred = time.time()
                y_pred = som_clf.predict(X=x_test)
                et_pred = time.time()
                pred_time_sek = et_pred - st_pred
                pred_time_str, h, m, s = get_time_str(seconds=pred_time_sek)
                print(f'# ### Prediction finished, time: {pred_time_sek} s = {pred_time_str}\n')

                pred_time_df = pd.DataFrame(
                    data=[[pred_time_sek, h, m, s]], index=['pred_time'], columns=['total s', 'h', 'm', 's']
                )
                pred_time_df.to_csv(os.path.join(save_p, 'pred_time_df.csv'))

                np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)

                # Load the sample-wise test data
                samples_p = os.path.join(data_p, 'sample_wise_test')
                n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
                sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
                samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
                samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]

                samples_y_pred = []
                samples_pred_times = []

                print('# ### Starting sample-wise prediction ...')
                for x in samples_x_test:

                    st = time.time()
                    samples_y_pred.append(som_clf.predict(X=x))
                    et = time.time()
                    samples_pred_times.append(et - st)

                samples_pred_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'])
                samples_pred_times_df['pred_time'] = samples_pred_times
                m = samples_pred_times_df['pred_time'].mean(axis=0)
                std = samples_pred_times_df['pred_time'].std(axis=0)
                samples_pred_times_df.loc['mean'] = m
                samples_pred_times_df.loc['std'] = std
                samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

                print(f'# ### Sample-wise prediction finished, avg time per sample: {m}\n')

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

    import os
    import time
    import numpy as np
    import pandas as pd
    from validation.gating.gatemeclass import GateMeClassClassifier
    from validation.utils import get_time_str, eval_wrapper, eval_wrapper_sample_wise, get_error_dataframe

    # ### Set flags and important variables here #######################################################################
    fit = True
    predict = True
    evaluate = True

    allow_abstention = True
    abstention_label = -1 if allow_abstention else None

    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    others_labels = [8, None, None, None, None, 5]
    pos_labels = [None, None, None, 1, 1, None]

    preprocessing_trafos = ['arcsinh_cofactor150', 'log10_channelwisecutoff', 'log10_cutoff100']

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


    ####################################################################################################################


    for data_set, others_label, pos_label, marker_names  in zip(
            data_sets, others_labels, pos_labels, marker_names_list
    ):
        for trafo in preprocessing_trafos:

            print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

            # Define lists to track errors
            failure_combinations = []
            failure_points = []
            error_types = []
            error_messages = []

            # Check whether train data exists for this dataset and trafo, if not continue
            data_p = os.path.join(os.getcwd(), f'data/np_files/{data_set}/{trafo}')
            alternate_data_p = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

            if not os.path.exists(data_p):
                data_p = alternate_data_p

            if not os.path.exists(data_p):
                print(f'# ### No data found for dataset "{data_set}" and transformation "{trafo}". Continue.\n')
                continue

            # Define path where results will be saved to
            abstention_str = '_w_abstention' if allow_abstention else '_no_abstention'
            save_p = os.path.join(os.getcwd(), f'results/pred_eval/gatemeclass{abstention_str}/{data_set}/{trafo}')
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
                    print('# ### Starting fit ...')
                    st_fit = time.time()
                    gmc_clf.fit(X=x_train, y=y_train)
                    et_fit = time.time()
                    fit_time_sek = et_fit - st_fit
                    fit_time_str, h, m, s = get_time_str(seconds=fit_time_sek)
                    print(f'# ### Fit finished, time: {fit_time_sek} s = {fit_time_str}\n')

                    fit_time_df = pd.DataFrame(
                        data=[[fit_time_sek, h, m, s]], index=['fit_time'], columns=['total s', 'h', 'm', 's']
                    )
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
                    st_pred = time.time()
                    y_pred = gmc_clf.predict(X=x_test)
                    et_pred = time.time()
                    pred_time_sek = et_pred - st_pred
                    pred_time_str, h, m, s = get_time_str(seconds=pred_time_sek)
                    print(f'# ### Prediction finished, time: {pred_time_sek} s = {pred_time_str}\n')

                    pred_time_df = pd.DataFrame(
                        data=[[pred_time_sek, h, m, s]], index=['pred_time'], columns=['total s', 'h', 'm', 's']
                    )
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
                samples_pred_times = []
                successful_fit_idx = []
                print('# ### Starting sample-wise prediction ...')
                for i, (x, sn) in enumerate(zip(samples_x_test, sample_names)):
                    try:
                        st = time.time()
                        samples_y_pred.append(gmc_clf.predict(X=x))
                        et = time.time()
                        samples_pred_times.append(et - st)
                        successful_fit_idx.append(i)
                    except Exception as e:
                        samples_pred_times.append(np.nan)
                        failure_combinations.append(f'{data_set}_{trafo}')
                        failure_points.append(f'predict_{sn}')
                        error_types.append(type(e).__name__)
                        error_messages.append(str(e))

                        error_df = get_error_dataframe(
                            failure_combinations, failure_points, error_types, error_messages
                        )
                        error_df.to_csv(os.path.join(save_p, 'errors.csv'))

                samples_pred_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'])
                samples_pred_times_df['pred_time'] = samples_pred_times
                m = samples_pred_times_df['pred_time'].mean(axis=0)
                std = samples_pred_times_df['pred_time'].std(axis=0)
                samples_pred_times_df.loc['mean'] = m
                samples_pred_times_df.loc['std'] = std
                samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

                print(f'# ### Sample-wise prediction finished, avg time per sample: {m}\n')

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

    import os
    import time
    import numpy as np
    import pandas as pd
    from validation.gating.dgcytof import DgcytofClassifier
    from validation.utils import get_time_str, eval_wrapper, eval_wrapper_sample_wise, get_error_dataframe

    # ### Set flags and important variables here #######################################################################
    fit = True
    predict = True
    evaluate = True

    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    others_labels = [8, None, None, None, None, 5]
    pos_labels = [None, None, None, 1, 1, None]

    data_sets = [
        'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'  # Todo
    ]
    others_labels = [None, None, None, 5]  # Todo
    pos_labels = [None, 1, 1, None]  # Todo

    preprocessing_trafos = ['arcsinh_cofactor150', 'log10_channelwisecutoff', 'log10_cutoff100']

    ####################################################################################################################

    for data_set, others_label, pos_label in zip(data_sets, others_labels, pos_labels):
        for trafo in preprocessing_trafos:

            print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

            # Define lists to track errors
            failure_combinations = []
            failure_points = []
            error_types = []
            error_messages = []

            # Check whether train data exists for this dataset and trafo, if not continue
            data_p = os.path.join(os.getcwd(), f'data/np_files/{data_set}/{trafo}')
            alternate_data_p = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

            if not os.path.exists(data_p):
                data_p = alternate_data_p

            if not os.path.exists(data_p):
                print(f'# ### No data found for dataset "{data_set}" and transformation "{trafo}". Continue.\n')
                continue

            # Define path where results will be saved to
            save_p = os.path.join(os.getcwd(), f'results/pred_eval/dgcytof/{data_set}/{trafo}')
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
                    st_fit = time.time()
                    dgcytof_clf.fit(X=x_train, y=y_train)
                    et_fit = time.time()
                    fit_time_sek = et_fit - st_fit
                    fit_time_str, h, m, s = get_time_str(seconds=fit_time_sek)
                    print(f'# ### Fit finished, time: {fit_time_sek} s = {fit_time_str}\n')

                    fit_time_df = pd.DataFrame(
                        data=[[fit_time_sek, h, m, s]], index=['fit_time'], columns=['total s', 'h', 'm', 's']
                    )
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
                # Load the test data
                x_test = np.load(os.path.join(data_p, 'x_test.npy'))

                try:
                    print('# ### Starting prediction ...')
                    st_pred = time.time()
                    y_pred = dgcytof_clf.predict(X=x_test)
                    et_pred = time.time()
                    pred_time_sek = et_pred - st_pred
                    pred_time_str, h, m, s = get_time_str(seconds=pred_time_sek)
                    print(f'# ### Prediction finished, time: {pred_time_sek} s = {pred_time_str}\n')

                    pred_time_df = pd.DataFrame(
                        data=[[pred_time_sek, h, m, s]], index=['pred_time'], columns=['total s', 'h', 'm', 's']
                    )
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
                samples_pred_times = []
                successful_fit_idx = []
                print('# ### Starting sample-wise prediction ...')
                for i, (x, sn) in enumerate(zip(samples_x_test, sample_names)):
                    try:
                        st = time.time()
                        samples_y_pred.append(dgcytof_clf.predict(X=x))
                        et = time.time()
                        samples_pred_times.append(et - st)
                        successful_fit_idx.append(i)
                    except Exception as e:
                        samples_pred_times.append(np.nan)
                        failure_combinations.append(f'{data_set}_{trafo}')
                        failure_points.append(f'predict_{sn}')
                        error_types.append(type(e).__name__)
                        error_messages.append(str(e))

                        error_df = get_error_dataframe(
                            failure_combinations, failure_points, error_types, error_messages
                        )
                        error_df.to_csv(os.path.join(save_p, 'errors.csv'))

                samples_pred_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'])
                samples_pred_times_df['pred_time'] = samples_pred_times
                m = samples_pred_times_df['pred_time'].mean(axis=0)
                std = samples_pred_times_df['pred_time'].std(axis=0)
                samples_pred_times_df.loc['mean'] = m
                samples_pred_times_df.loc['std'] = std
                samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

                print(f'# ### Sample-wise prediction finished, avg time per sample: {m}\n')

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
                    abstention_label=-1,  # Dgcytof can predict events to be of 'unknown'/-1 class
                    others_label=others_label,
                    pos_label=pos_label,
                    verbosity=2
                )

                out[0].to_csv(os.path.join(save_p, 'res_df_avg.csv'))
                out[1].to_csv(os.path.join(save_p, 'res_df_cw.csv'))
                out[2].to_csv(os.path.join(save_p, 'cf_mat.csv'))
                out[3].to_csv(os.path.join(save_p, 'abst_counts.csv'))

                # Get the sample-wise test data for which the prediction was successful
                sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples) if i in successful_fit_idx]
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


def main_softmax():
    import os
    import time
    import numpy as np
    import pandas as pd

    from flagx.gating import SoftmaxClassifier
    from validation.utils import get_time_str, eval_wrapper, eval_wrapper_sample_wise

    # ### Set flags and important variables here #######################################################################
    data_sets = [
        'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'
    ]
    pos_labels = [None, None, None, 1, 1, None]

    preprocessing_trafos = ['arcsinh_cofactor150', 'log10_channelwisecutoff', 'log10_cutoff100']

    fit = True
    predict = True
    evaluate = True
    ####################################################################################################################

    for data_set, pos_label in zip(data_sets, pos_labels):
        for trafo in preprocessing_trafos:

            print(f'# ###### Data set: {data_set}, trafo: {trafo} ###### #')

            # Check whether train data exists for this dataset and trafo, if not continue
            data_p = os.path.join(os.getcwd(), f'data/np_files/{data_set}/{trafo}')
            alternate_data_p = f'/home/woody/iwbn/iwbn107h/data/np_files/{data_set}/{trafo}'

            if not os.path.exists(data_p):
                data_p = alternate_data_p

            if not os.path.exists(data_p):
                print(f'# ### No data found for dataset "{data_set}" and transformation "{trafo}". Continue.\n')
                continue

            # Define path where results will be saved to
            save_p = os.path.join(os.getcwd(), f'results/pred_eval/softmax_classifier/{data_set}/{trafo}')
            os.makedirs(save_p, exist_ok=True)

            if fit:

                # Load training data
                x_train = np.load(os.path.join(data_p, 'x_train.npy'))
                y_train = np.load(os.path.join(data_p, 'y_train.npy'))

                # Instantiate the Softmax classifier with default parameters
                softmax_clf = SoftmaxClassifier(
                    layer_sizes=(128, 64, 32),
                    n_epochs=20,
                    data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
                    device=None,  # Tries to use default cuda device, if none available cpu
                    verbosity=2
                )

                # Fit and track time
                print('# ### Starting fit ...')
                st_fit = time.time()
                softmax_clf.fit(X=x_train, y=y_train)
                et_fit = time.time()
                fit_time_sek = et_fit - st_fit
                fit_time_str, h, m, s = get_time_str(seconds=fit_time_sek)
                print(f'# ### Fit finished, time: {fit_time_sek} s = {fit_time_str}\n')

                fit_time_df = pd.DataFrame(
                    data=[[fit_time_sek, h, m, s]], index=['fit_time'], columns=['total s', 'h', 'm', 's']
                )
                fit_time_df.to_csv(os.path.join(save_p, 'fit_time_df.csv'))

                softmax_clf.save(filepath=save_p)

            else:
                # Load the SOM classifier
                softmax_clf = SoftmaxClassifier.load(filepath=save_p)

            if predict:
                # Load the test data
                x_test = np.load(os.path.join(data_p, 'x_test.npy'))

                print('# ### Starting prediction ...')
                st_pred = time.time()
                y_pred = softmax_clf.predict(X=x_test)
                et_pred = time.time()
                pred_time_sek = et_pred - st_pred
                pred_time_str, h, m, s = get_time_str(seconds=pred_time_sek)
                print(f'# ### Prediction finished, time: {pred_time_sek} s = {pred_time_str}\n')

                pred_time_df = pd.DataFrame(
                    data=[[pred_time_sek, h, m, s]], index=['pred_time'], columns=['total s', 'h', 'm', 's']
                )
                pred_time_df.to_csv(os.path.join(save_p, 'pred_time_df.csv'))

                np.save(os.path.join(save_p, 'y_pred.npy'), y_pred)

                # Load the sample-wise test data
                samples_p = os.path.join(data_p, 'sample_wise_test')
                n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
                sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
                samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
                samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]

                samples_y_pred = []
                samples_pred_times = []

                print('# ### Starting sample-wise prediction ...')
                for x in samples_x_test:
                    st = time.time()
                    samples_y_pred.append(softmax_clf.predict(X=x))
                    et = time.time()
                    samples_pred_times.append(et - st)

                samples_pred_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'])
                samples_pred_times_df['pred_time'] = samples_pred_times
                m = samples_pred_times_df['pred_time'].mean(axis=0)
                std = samples_pred_times_df['pred_time'].std(axis=0)
                samples_pred_times_df.loc['mean'] = m
                samples_pred_times_df.loc['std'] = std
                samples_pred_times_df.to_csv(os.path.join(save_p, 'samples_pred_times_df.csv'))

                print(f'# ### Sample-wise prediction finished, avg time per sample: {m}\n')

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


def main_n_samples_experiment():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from validation.utils.val_utils import n_samples_experiment_helper
    from flagx.gating import SomClassifier, SoftmaxClassifier
    from validation.plt import plot_n_samples_n_events

    # ### Set flags and important variables here #######################################################################
    data_set = 'imstat'
    # 'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'

    preprocessing_trafo = 'log10_channelwisecutoff'
    # 'arcsinh_cofactor150', 'log10_channelwisecutoff', 'log10_cutoff100'

    random_sample_order = True

    classifier = 'softmax'  # 'som', 'softmax'

    inference = True

    downsampling_fractions = [0.01, 0.25, 0.5, 0.75, 1.0]

    ####################################################################################################################

    base_p = os.path.join(os.getcwd(), 'results/n_samples_n_events')

    # Load the sample order file
    sample_order_file = None
    if not random_sample_order:
        if data_set != 'lymphoma_tube1_binary':
            print(f'No sample order available for {data_set}. Continuing with random order.')
        else:
            sample_order_file = os.path.join(
                base_p, f'sample_order_{data_set}_{preprocessing_trafo}.txt'
            )

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
    data_p = os.path.join(os.getcwd(), 'data/np_files', data_set, preprocessing_trafo)

    if inference:
        if classifier == 'som':
            # Instantiate the SOM classifier
            if preprocessing_trafo == 'arcsinh_cofactor150':

                clf = SomClassifier(
                    som_topology='planar',
                    som_grid_type='rectangular',
                    som_dimensions=(25, 25),
                    neighborhood='gaussian',
                    gaussian_neighborhood_sigma=0.25,
                    initialization='pca',
                    n_epochs=200,
                    radius_0=-0.25,
                    radius_n=0.01,
                    radius_cooling='linear',
                    learning_rate_0=0.5,
                    learning_rate_n=0.05,
                    learning_rate_decay='exponential',
                    verbosity=2,
                )
            else:
                clf = SomClassifier(
                    som_topology='planar',
                    som_grid_type='rectangular',
                    som_dimensions=(25, 25),
                    neighborhood='gaussian',
                    gaussian_neighborhood_sigma=0.1,
                    initialization='pca',
                    n_epochs=1000,
                    radius_0=-0.25,
                    radius_n=0.1,
                    radius_cooling='linear',
                    learning_rate_0=0.1,
                    learning_rate_n=0.05,
                    learning_rate_decay='exponential',
                    verbosity=2,
                )
        else:
            clf = SoftmaxClassifier(
                layer_sizes=(128, 64, 32),
                n_epochs=20,
                data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
                device=None,  # Tries to use default cuda device, if none available cpu
                verbosity=2
            )

        n_samples_experiment_helper(
            classifier=clf,
            downsampling_fractions=downsampling_fractions,
            data_p=data_p,
            save_p=save_p,
            abstention_label=abstention_label,
            others_label=others_label,
            pos_label=pos_label,
            sample_order_file=sample_order_file,
        )

    res_df = pd.read_csv(os.path.join(save_p, 'res_df_f1_binary.csv'), index_col=0)

    print(res_df)

    plot_n_samples_n_events(res_df=res_df, cmap_name='magma')
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'f1_binary.png'))


def main_prec_vs_recall():

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from flagx.gating import SomClassifier, SoftmaxClassifier
    from validation.plt import plot_prec_rec_vs_thresh

    # ### Set flags and important variables here #######################################################################
    data_set = 'lymphoma_tube1_binary' # 'lymphoma_tube1_binary', 'lymphoma_tube2_binary'
    trafo = 'log10_channelwisecutoff'  # 'arcsinh_cofactor150', 'log10_channelwisecutoff'

    classifier = 'som'  # 'som', 'softmax'

    # thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = np.linspace(0, 1, 20).tolist()
    ####################################################################################################################

    # Load the test data
    data_p = os.path.join(os.getcwd(), 'data/np_files', data_set, trafo)

    x_test = np.load(os.path.join(data_p, 'x_test.npy'))
    y_test = np.load(os.path.join(data_p, 'y_test.npy'))

    # Load the sample-wise test data
    samples_p = os.path.join(data_p, 'sample_wise_test')
    n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
    sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
    samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names]
    samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test_filenames]
    samples_y_test_filenames = [f'y_{sn}.npy' for sn in sample_names]
    samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test_filenames]

    # Load the previously trained classifier
    if classifier == 'som':

        clf = SomClassifier.load(
            filepath=os.path.join(os.getcwd(), 'results/pred_eval/som_classifier', data_set, trafo)
        )

    else:
        clf = SoftmaxClassifier.load(
            filepath=os.path.join(os.getcwd(), 'results/pred_eval/softmax_classifier', data_set, trafo)
        )

    # Predict probabilities and save predictions
    save_p = os.path.join(os.getcwd(), 'results/prec_vs_recall', classifier, data_set, trafo)
    os.makedirs(save_p, exist_ok=True)

    y_proba = clf.predict_proba(X=x_test)

    np.save(os.path.join(save_p, 'y_proba.npy'), y_proba)

    samples_y_proba = []

    for x in samples_x_test:
        samples_y_proba.append(clf.predict_proba(X=x))

    os.makedirs(os.path.join(save_p, 'samples_y_pred'), exist_ok=True)
    for y, sn in zip(samples_y_proba, sample_names):
        np.save(os.path.join(save_p, 'samples_y_pred', f'y_proba_{sn}.npy'), y)

    # Evaluate the prediction performance
    y_proba = y_proba[:, 1]
    samples_y_proba = [y[:, 1] for y in samples_y_proba]

    fig, ax = plt.subplots(dpi=300)
    RocCurveDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_proba,
        name='SOM classifier' if classifier == 'som' else 'Softmax classifier',
        ax=ax,
        plot_chance_level=True,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'roc.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    PrecisionRecallDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_proba,
        name='SOM classifier' if classifier == 'som' else 'Softmax classifier',
        ax=ax,
        plot_chance_level=True,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'prec_rec.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    plot_prec_rec_vs_thresh(
        y_trues=samples_y_test,
        y_probs=samples_y_proba,
        thresholds=thresholds,
        pos_label=1,
        neg_label=0,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'prec_rec_thresh.png'), dpi=300)
    plt.close('all')

    print(y_test)


def main_som_plots():
    pass


def main_performance_score_plots():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    import matplotlib
    matplotlib.use('Agg')

    from validation.plt import plot_performance_score_box_plot


    # ### Set flags and important variables here #######################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'binary'  # macro, micro, weighted, binary

    dataset_names = ['imstat', 'flowcyt', 'lt1', 'lt1_b', 'lt2', 'lt2_b']

    method_names = ['GMC', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    data_trafo = 'arcsinh_cofactor150'  # log10_channelwisecutoff, arcsinh_cofactor150

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'imstat': 'imstat',
        'lt1': 'lymphoma_tube1', 'lt1_b': 'lymphoma_tube1_binary',
        'lt2': 'lymphoma_tube2', 'lt2_b': 'lymphoma_tube2_binary',
        'flowcyt': 'flowcyt'
    }
    dataset_dirs = [conversion_mapping_datasets[ds] for ds in dataset_names]

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GMC': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier',
        'SOM-Classifier': 'som_classifier_ramses',
    }

    method_dirs = [conversion_mapping_methods[m] for m in method_names]

    # Load the results dataframes
    base_path = os.path.join(os.getcwd(), 'results/pred_eval')
    res_dfs = []
    for ds in dataset_dirs:
        res_dfs_sub = []
        for m in method_dirs:
            if ds == 'flowcyt' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'log10_cutoff100'
            else:
                data_trafo_load = data_trafo

            res_df_path = os.path.join(base_path, m, ds, data_trafo_load, f'res_df_sw_avg_{performance_score}.csv')

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')
            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)


    conversion_mapping_y_label = {'f1': 'F1', 'prec': 'Precision', 'rec': 'Recall'}

    plot_performance_score_box_plot(
        sample_wise_res_dfs=res_dfs,
        method_names=method_names,
        dataset_names=dataset_names,
        score_mode=performance_score_mode,
        y_label=conversion_mapping_y_label[performance_score],
        sns_boxplot_kwargs=None,
        plot_points=True,
        point_kwargs=None,
        boxplot_alpha=0.9,
        ax=None,
    )

    plt.savefig(os.path.join(plot_dir, f'{performance_score_mode}_{performance_score}_box_plot.png'), dpi=300)


def main_cell_percentage_plots():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import matplotlib
    matplotlib.use('Agg')

    from validation.plt import plot_cell_pop_size_pred_vs_gt


    # ### Set flags and important variables here #######################################################################
    dataset_name = 'Imstat'  # 'imstat', 'lt1', 'lt1_b', 'lt2', 'lt2_b', 'flowcyt'

    method_name = 'DGCyTOF'  # 'GMC na', 'GMC wa', 'DGCyTOF', 'FCNN'

    data_trafo = 'log10_channelwisecutoff'  # log10_channelwisecutoff, arcsinh_cofactor150, log10_cutoff100

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################


    if dataset_name == 'Flowcyt' and data_trafo == 'log10_channelwisecutoff':
        data_trafo = 'log10_cutoff100'

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'lt1': 'lymphoma_tube1', 'lt1_b': 'lymphoma_tube1_binary',
        'lt2': 'lymphoma_tube2', 'lt2_b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }
    dataset_dir = conversion_mapping_datasets[dataset_name]

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GMC na': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier'
    }
    method_dir = conversion_mapping_methods[method_name]

    # Load the results dataframes
    base_path_y_true = os.path.join(os.getcwd(), 'data/np_files')
    base_path_y_pred = os.path.join(os.getcwd(), 'results/pred_eval')

    # Load the sample-wise data (ground truth and prediction)
    y_true_path = os.path.join(base_path_y_true, dataset_dir, data_trafo, 'sample_wise_test')

    n_samples = len([f for f in os.listdir(y_true_path) if f.startswith('y_')])
    y_true_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
    y_trues = [np.load(os.path.join(y_true_path, f)).astype(int) for f in y_true_filenames]

    y_pred_path = os.path.join(base_path_y_pred, method_dir, dataset_dir, data_trafo, 'samples_y_pred')
    y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
    # y_preds = [np.load(os.path.join(y_pred_path, f)).astype(int) for f in y_pred_filenames]
    y_preds = []
    for f in y_pred_filenames:
        try:
            y_preds.append(np.load(os.path.join(y_pred_path, f)).astype(int))
        except FileNotFoundError:
            print('filename not found: ', f)
            y_preds.append(np.array([]))

    # Keep elements only where the corresponding y_pred array is NOT empty
    filtered = [(a, b) for a, b in zip(y_trues, y_preds) if b.size > 0]
    y_trues, y_preds = map(list, zip(*filtered))

    plot_cell_pop_size_pred_vs_gt(
        y_trues=y_trues,
        y_preds=y_preds,
        percentage=True,
        palette=None,
        title='Predicted vs True Cell Type Proportions',
        point_size=6.0,
        show_r2=True,
        show_pearson=True,
        ax=None,
    )

    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f'{method_name}_{dataset_name}_population_sizes.png'), dpi=300)


def main_time_table():

    import os
    import numpy as np
    import pandas as pd

    # ### Set flags and important variables here #######################################################################

    dataset_names = ['Imstat', 'Flowcyt', 'LT1', 'LT1 b', 'LT2', 'LT2 b']

    method_names = ['GMC', 'DGCyTOF', 'FCNN', 'SOM-clf']

    data_trafo = 'log10_channelwisecutoff'  # log10_channelwisecutoff, arcsinh_cofactor150

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }


    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GMC': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier',
        'SOM-clf': 'som_classifier',
    }

    dataset_dirs = [conversion_mapping_datasets[ds] for ds in dataset_names]
    method_dirs = [conversion_mapping_methods[m] for m in method_names]

    # Load the results dataframes
    base_path = os.path.join(os.getcwd(), 'results/pred_eval')
    inference_time_dfs = []
    train_times = []
    table_data = []
    for ds in dataset_dirs:
        inference_time_dfs_sub = []
        train_times_sub = []
        table_data_sub = []
        for m in method_dirs:

            # Always show results for arcsinh for gatemeclass
            # if m == 'gatemeclass_no_abstention' and data_trafo == 'log10_channelwisecutoff':
            #     data_trafo_load = 'arcsinh_cofactor150'
            if ds == 'flowcyt' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'log10_cutoff100'
            else:
                data_trafo_load = data_trafo

            inference_time_df_path = os.path.join(base_path, m, ds, data_trafo_load, 'samples_pred_times_df.csv')
            train_time_df_path = os.path.join(base_path, m, ds, data_trafo_load, 'fit_time_df.csv')

            try:
                inference_time_df = pd.read_csv(inference_time_df_path, index_col=0)
                train_time_df = pd.read_csv(train_time_df_path, index_col=0)
                train_time = train_time_df.loc['fit_time', 'total s']
            except FileNotFoundError:
                inference_time_df = pd.DataFrame()
                train_time_df = pd.DataFrame()
                train_time = np.nan
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            inference_time_dfs_sub.append(inference_time_df)
            train_times_sub.append(train_time)
            table_data_sub.append(inference_time_df.loc['mean', 'pred_time'] if not inference_time_df.empty else np.nan)

        inference_time_dfs.append(inference_time_dfs_sub)
        train_times.append(train_times_sub)
        table_data.append(train_times_sub)
        table_data.append(table_data_sub)

    column_tuples = [(dataset, phase) for dataset in dataset_names for phase in ['Train', 'Inference']]
    multi_columns = pd.MultiIndex.from_tuples(column_tuples, names=['Dataset', 'Phase'])

    # Create empty DataFrame with method names as rows and multi-level columns
    df = pd.DataFrame(
        data=np.array(table_data).T,
        index=method_names,
        columns=multi_columns
    )

    df.to_csv(os.path.join(plot_dir, f'times_table_{data_trafo}.csv'), index=False)

    print(df)


def main_performance_plots():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from validation.plt import plot_performance_score_box_plot, plot_cell_pop_size_pred_vs_gt, annotate_mosaic


    ####################################################################################################################
    dataset_name_psize = 'Imstat'
    dataset_names_perf = ['Imstat', 'Flowcyt', 'LT1', 'LT1 b', 'LT2', 'LT2 b']

    method_names = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    data_trafo = 'log10_channelwisecutoff'  # log10_channelwisecutoff, arcsinh_cofactor150, log10_cutoff100

    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GateMeClass': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier',
        'SOM-Classifier': 'som_classifier'
    }

    conversion_mapping_y_label = {'f1': 'F1', 'prec': 'Precision', 'rec': 'Recall'}

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EE
        """,
        gridspec_kw={'height_ratios': [1/4, 1/4, 1/2]}
    )

    # ### Plot the population percentages
    # Load the plot data
    base_path_y_true = os.path.join(os.getcwd(), 'data/np_files')
    base_path_y_pred = os.path.join(os.getcwd(), 'results/pred_eval')

    # Load the sample-wise data (ground truth and prediction)
    dataset_dir_psize = conversion_mapping_datasets[dataset_name_psize]
    y_true_path = os.path.join(base_path_y_true, dataset_dir_psize, data_trafo, 'sample_wise_test')

    n_samples = len([f for f in os.listdir(y_true_path) if f.startswith('y_')])
    y_true_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
    y_trues = [np.load(os.path.join(y_true_path, f)).astype(int) for f in y_true_filenames]

    # Define color mapping
    color_mapping = dict(
        (str(k), c)
        for k, c in zip([-1] + list(range(1, 9)), sns.color_palette("Accent", 9))
    )

    for key, method in zip(['A', 'C', 'D', 'B'], method_names):

        method_dir = conversion_mapping_methods[method]

        # Always show results for arcsinh for gatemeclass
        if method_dir == 'gatemeclass_no_abstention' and data_trafo == 'log10_channelwisecutoff':
            data_trafo_load = 'arcsinh_cofactor150'
        elif dataset_dir_psize == 'flowcyt' and data_trafo == 'log10_channelwisecutoff':
            data_trafo_load = 'log10_cutoff100'
        else:
            data_trafo_load = data_trafo

        y_pred_path = os.path.join(base_path_y_pred, method_dir, dataset_dir_psize, data_trafo_load, 'samples_y_pred')
        y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
        # y_preds = [np.load(os.path.join(y_pred_path, f)).astype(int) for f in y_pred_filenames]

        # Load the predictions, skip missing files
        y_preds = []
        for f in y_pred_filenames:
            try:
                y_preds.append(np.load(os.path.join(y_pred_path, f)).astype(int))
            except FileNotFoundError:
                y_preds.append(np.array([]))
                print(f'# ### No y_pred found for: {method}, {dataset_name_psize}, {data_trafo_load}, {f}')
        # Keep elements only where the corresponding y_pred array is NOT empty
        y_trues_plot = [yt for yt, yp in zip(y_trues, y_preds) if yp.size > 0]
        y_preds_plot = [yp for yp in y_preds if yp.size > 0]

        plot_cell_pop_size_pred_vs_gt(
            y_trues=y_trues_plot,
            y_preds=y_preds_plot,
            percentage=True,
            palette=color_mapping,
            title=method,
            point_size=11.0,
            show_r2=True,
            show_pearson=True,
            ax=axd[key],
        )

    # ### Plot the performance scores
    # Load the results dataframes
    base_path = os.path.join(os.getcwd(), 'results/pred_eval')
    dataset_dirs_perf = [conversion_mapping_datasets[ds] for ds in dataset_names_perf]
    method_dirs_perf = [conversion_mapping_methods[method] for method in method_names]
    res_dfs = []
    for ds in dataset_dirs_perf:
        res_dfs_sub = []
        for m in method_dirs_perf:

            # Always show results for arcsinh for gatemeclass
            if m == 'gatemeclass_no_abstention' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'arcsinh_cofactor150'
            elif ds == 'flowcyt' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'log10_cutoff100'
            else:
                data_trafo_load = data_trafo

            res_df_path = os.path.join(base_path, m, ds, data_trafo_load, f'res_df_sw_avg_{performance_score}.csv')

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')
            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    plot_performance_score_box_plot(
        sample_wise_res_dfs=res_dfs,
        method_names=method_names,
        dataset_names=dataset_names_perf,
        score_mode=performance_score_mode,
        y_label=conversion_mapping_y_label[performance_score],
        sns_boxplot_kwargs=None,
        plot_points=True,
        point_kwargs=None,
        boxplot_alpha=0.9,
        ax=axd['E'],
    )

    # ### Manually adjust axis labels
    ax_label_fontsize = 12

    for key in ['A', 'B', 'C', 'D']:
        ax = axd[key]
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(labelsize=ax_label_fontsize - 2)

    ax_e = axd['E']
    ax_e.set_xlabel(None)
    ax_e.set_ylabel(ax_e.get_ylabel(), fontsize=ax_label_fontsize)
    ax_e.tick_params(axis='y', labelsize=ax_label_fontsize - 2)
    ax_e.tick_params(axis='x', labelsize=ax_label_fontsize)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig('./results/plots/performance.png', dpi=fig.dpi)


def main_n_samples_plot():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm

    from matplotlib.lines import Line2D
    from validation.plt import annotate_mosaic

    ####################################################################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    ds_fractions = [0.01, 0.25, 0.5, 0.75, 1.0]
    max_n_samples = 75

    plot_dir = os.path.join(os.getcwd(), 'results/plots')

    ####################################################################################################################

    # Load all results
    dataset_names = ['Imstat', 'LT1 b']
    method_names = ['FCNN', 'SOM-Classifier']


    # Results only generated for log10_channelwisecutoff
    data_trafo = 'log10_channelwisecutoff'

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {'Imstat': 'imstat', 'LT1 b': 'lymphoma_tube1_binary'}

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {'FCNN': 'softmax', 'SOM-Classifier': 'som'}
    conversion_mapping_methods_predres = {'FCNN': 'softmax_classifier', 'SOM-Classifier': 'som_classifier'}

    all_records = []

    for ds in dataset_names:
        for m in method_names:
            for order in ['random', 'ordered']:
                dsdir = conversion_mapping_datasets[ds]
                mdir = conversion_mapping_methods[m]
                base_path = os.path.join('./results/n_samples_n_events', mdir, dsdir, data_trafo, order, 'detailed_res')

                for ds_frac in ds_fractions:
                    for i in range(1, max_n_samples + 1):
                        subdir = f'dsfrac_{str(ds_frac).replace(".", "_")}_nsamples_{i}/res_df_sw_avg_{performance_score}.csv'
                        file_path = os.path.join(base_path, subdir)

                        try:
                            df = pd.read_csv(file_path, index_col=0)
                            df = df.drop(index=['mean', 'std'], errors='ignore')  # Keep raw data only
                            for val in df[performance_score_mode]:
                                all_records.append({
                                    'dataset': ds,
                                    'method': m,
                                    'order': order,
                                    'ds_frac': ds_frac,
                                    'n_samples': i,
                                    'score': val
                                })
                        except FileNotFoundError:
                            # print(f"# Missing: {file_path}")
                            continue

    # Create DataFrame
    df_all = pd.DataFrame(all_records)
    df_all['n_samples'] = df_all['n_samples'].astype(int)

    print(df_all)

    # ### Plot performance comparison for methods
    # One plot per dataset, hue=method, fixed: order=random, ds_frac=1.0  => 2 subplots
    ds_frac_plot = 1.0
    plot_trend = False
    plot_slope = False
    abline_all_samples_perf = True
    plot_all_samples_score = True

    # Define a palette
    mn = ['dummy0', 'dummy1', 'FCNN', 'SOM-Classifier']
    palette = dict(zip(mn, sns.color_palette('Set2', len(mn))))

    fig = plt.figure(figsize=(8, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        """,
        gridspec_kw=None
    )

    for ds, plot_label in zip(['Imstat', 'LT1 b'], ['A', 'B']):

        df_sub = df_all.loc[
            (df_all['dataset'] == ds) & (df_all['ds_frac'] == ds_frac_plot) & (df_all['order'] == 'random')
        ].copy()

        ax = axd[plot_label]

        sns.lineplot(
            data=df_sub,
            x='n_samples',
            y='score',
            hue='method',
            errorbar=('ci', 95),
            n_boot=1000,
            seed=42,
            err_style='band',
            marker='o',
            markersize=4,
            palette=palette,
            ax=ax,
        )

        if abline_all_samples_perf:

            for method_name, method_dir in [('FCNN', 'softmax_classifier'), ('SOM-Classifier', 'som_classifier')]:
                try:
                    res_path = os.path.join(
                        './results/pred_eval',
                        method_dir,
                        conversion_mapping_datasets[ds],
                        data_trafo,
                        f'res_df_sw_avg_{performance_score}.csv'
                    )

                    score = pd.read_csv(res_path, index_col=0).loc['mean', performance_score_mode]
                    color = palette.get(method_name, 'grey')

                    ax.axhline(
                        y=score,
                        linestyle='--',
                        linewidth=1,
                        color=color,
                        alpha=0.8,
                        # label=f'{method_name} (all samples)',
                    )

                    x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
                    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                    y_pos = score + y_offset  # just above the line

                    # Draw text
                    ax.text(
                        x=x_pos,
                        y=y_pos,
                        s=f'{score:.3f}',
                        color=color,
                        va='bottom',
                        ha='right',
                        fontsize=8,
                        alpha=0.95,
                        clip_on=True
                    )

                except FileNotFoundError:
                    print('###')
                    pass

            # Define dummy legend entry for all sample performance
            trend_legend = Line2D(
                [], [], linestyle='--', color='grey', linewidth=1, label='All Samples'
            )
            handles, labels = ax.get_legend_handles_labels()
            if 'All Samples' not in labels:
                handles.append(trend_legend)
                labels.append('All Samples')
            ax.legend(handles=handles, labels=labels, title='Method')

        if plot_trend:
            for method_name, color in palette.items():
                df_m = df_sub[df_sub['method'] == method_name]
                if df_m.empty:
                    continue

                reg_df = (
                    df_m.groupby('n_samples', as_index=False)['score'].mean()
                )

                use_regplot = False
                if use_regplot:
                    sns.regplot(
                        data=reg_df,
                        x='n_samples',
                        y='score',
                        scatter=False,
                        ax=ax,
                        color=color,  # Match method color
                        ci=None,  # No confidence interval
                        line_kws={
                            'linestyle': '--',
                            'alpha': 0.8,
                            'linewidth': 1,
                            'zorder': 5,
                        }
                    )
                else:
                    # Prepare X and y for statsmodels
                    x = reg_df['n_samples'].values
                    y = reg_df['score'].values
                    x_const = sm.add_constant(x)  # adds intercept term

                    # Fit linear model
                    model = sm.OLS(y, x_const).fit()
                    slope = model.params[1]
                    intercept = model.params[0]

                    print(f"{method_name} - Slope: {slope:.6f}, Intercept: {intercept:.4f}")

                    # Create fitted line
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = intercept + slope * x_fit

                    # Plot manually
                    ax.plot(
                        x_fit,
                        y_fit,
                        linestyle='--',
                        color=color,
                        alpha=0.8,
                        linewidth=1,
                        zorder=5,
                        label=f'Linear Trend, Slope: {np.round(slope, 4)}' if plot_slope else None,
                    )

            if not plot_slope:
                # Define dummy legend entry for trends
                trend_legend = Line2D(
                    [], [], linestyle='--', color='grey', linewidth=1, label='Linear Trends'
                )
                handles, labels = ax.get_legend_handles_labels()
                if 'Linear Trends' not in labels:
                    handles.append(trend_legend)
                    labels.append('Linear Trends')
                ax.legend(handles=handles, labels=labels, title='Method')
            else:
                ax.legend(title='Method')

        if not plot_all_samples_score and not plot_trend:
            ax.legend(title='Method')

        ax.set_title(ds)
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')

    # Adjust font sizes
    ax_label_fontsize = 12
    for key in ['A', 'B']:
        ax = axd[key]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'n_samples_per_dataset_{performance_score_mode}_{performance_score}.png'),
        dpi=fig.dpi
    )
    plt.close('all')

    # ### Plot performance comparison for random vs ordered and num events
    # One plot per method, per dataset, and per order, hue=ds_frac  => 6 subplots

    ds_frac_plot = [0.01, 0.25, 0.5, 0.75, 1.0]

    plot_combinations = [
        ('Imstat', 'FCNN', 'random'), ('Imstat', 'SOM-Classifier', 'random'),
        ('LT1 b', 'FCNN', 'random'), ('LT1 b', 'SOM-Classifier', 'random'),
        # ('LT1 b', 'FCNN', 'ordered'), ('LT1 b', 'SOM-Classifier', 'ordered')
    ]

    abline_all_samples_perf = True

    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """,
        gridspec_kw=None
    )

    for c, plot_label in zip(plot_combinations, ['A', 'B', 'C', 'D']):
        ds = c[0]
        m = c[1]
        o = c[2]
        df_sub = df_all.loc[
            (df_all['dataset'] == ds) &
            (df_all['method'] == m) &
            (df_all['order'] == o) &
            (df_all['ds_frac'].isin(ds_frac_plot))
            ].copy()

        ax = axd[plot_label]

        sns.lineplot(
            data=df_sub,
            x='n_samples',
            y='score',
            hue='ds_frac',
            errorbar=('ci', 95),
            n_boot=1000,
            seed=42,
            err_style='band',
            marker='o',
            markersize=3,
            palette='magma',
            ax=ax,
        )

        if abline_all_samples_perf:


            try:
                res_path = os.path.join(
                    './results/pred_eval',
                    conversion_mapping_methods_predres[m],
                    conversion_mapping_datasets[ds],
                    data_trafo,
                    f'res_df_sw_avg_{performance_score}.csv'
                )

                score = pd.read_csv(res_path, index_col=0).loc['mean', performance_score_mode]

                ax.axhline(
                    y=score,
                    linestyle='--',
                    linewidth=1,
                    color='grey',
                    alpha=0.8,
                    label='All Samples',
                )

                x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                y_pos = score + y_offset  # just above the line

                # Draw text
                ax.text(
                    x=x_pos,
                    y=y_pos,
                    s=f'{score:.3f}',
                    color='grey',
                    va='bottom',
                    ha='right',
                    fontsize=8,
                    alpha=0.95,
                    clip_on=True
                )


            except FileNotFoundError:
                print('###')
                pass

        ax.set_title(f'{ds} | {m}')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')
        ax.legend(title='Downsampling Fraction')

    # Adjust font sizes
    ax_label_fontsize = 12
    for key in ['A', 'B', 'C', 'D']:
        ax = axd[key]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'n_samples_ds_frac_supplement.png'),
        dpi=fig.dpi
    )
    plt.close('all')

    # ### Plot performance comparison for random vs ordered
    # One plot per method, hue=order, fixed: dataset=LT1 b, ds_frac=1.0  => 2 subplots
    ds_frac_plot = 1.0
    dataset_plot = 'LT1 b'

    fig = plt.figure(figsize=(10, 4), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        """,
        gridspec_kw=None
    )

    for m, plot_label in zip(['FCNN', 'SOM-Classifier'], ['A', 'B']):
        df_sub = df_all.loc[
            (df_all['dataset'] == dataset_plot) &
            (df_all['ds_frac'] == ds_frac_plot) &
            (df_all['method'] == m)
            ].copy()

        ax = axd[plot_label]

        sns.lineplot(
            data=df_sub,
            x='n_samples',
            y='score',
            hue='order',
            errorbar=('ci', 95),
            n_boot=1000,
            seed=42,
            err_style='band',
            marker='o',
            markersize=4,
            palette='Accent',
            ax=ax,
        )

        ax.set_title(m)
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')
        ax.legend(title='Sample Order')

    annotate_mosaic(fig=fig, axd=axd, fontsize=14)
    plt.savefig(
        os.path.join(plot_dir, f'n_samples_per_dataset_random_vs_ordered.png'),
        dpi=fig.dpi
    )
    plt.close('all')



def main_performance_plots_supplement():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from validation.plt import plot_performance_score_box_plot_cw, annotate_mosaic


    ####################################################################################################################
    dataset_names = ['Imstat', 'Flowcyt', 'LT1', 'LT1 b', 'LT2', 'LT2 b']

    method_names = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    data_trafo = 'arcsinh_cofactor150'  # log10_channelwisecutoff, arcsinh_cofactor150, log10_cutoff100

    performance_score = 'f1'  # f1, prec, rec

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GateMeClass': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier',
        'SOM-Classifier': 'som_classifier'
    }

    conversion_mapping_y_label = {'f1': 'F1', 'prec': 'Precision', 'rec': 'Recall'}


    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )

    # Define a palette
    palette = dict(zip(method_names, sns.color_palette("Set2", len(method_names))))

    # ### Plot the performance scores
    # Load the results dataframes
    base_path = os.path.join(os.getcwd(), 'results/pred_eval')
    dataset_dirs = [conversion_mapping_datasets[ds] for ds in dataset_names]
    method_dirs = [conversion_mapping_methods[method] for method in method_names]

    res_dfs = []
    for ds in dataset_dirs:
        res_dfs_sub = []
        for m in method_dirs:

            # Always show results for arcsinh for gatemeclass
            if m == 'gatemeclass_no_abstention' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'arcsinh_cofactor150'
            elif ds == 'flowcyt' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'log10_cutoff100'
            else:
                data_trafo_load = data_trafo

            res_df_path = os.path.join(base_path, m, ds, data_trafo_load, f'res_df_sw_cw_{performance_score}.csv')

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')
            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)


    for rdfl, dsn, label in zip(res_dfs, dataset_names, ['A', 'B', 'C', 'D', 'E', 'F']):

        plot_performance_score_box_plot_cw(
            sample_wise_res_dfs=rdfl,
            method_names=method_names,
            y_label=conversion_mapping_y_label[performance_score],
            title=dsn,
            palette=palette,
            sns_boxplot_kwargs=None,
            plot_points=True,
            point_kwargs=None,
            boxplot_alpha=0.9,
            ax=axd[label],
        )

    '''# ### Manually adjust axis labels
    ax_label_fontsize = 12

    for key in ['A', 'B']:
        ax = axd[key]
        ax.set_xlabel(None, fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)

        ax.tick_params(axis='x', labelsize=ax_label_fontsize)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)'''

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig(f'./results/plots/performance_{performance_score}_supplement.png', dpi=fig.dpi)


def main_dataset_size_plot_supplement():

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from validation.plt import plot_sample_sizes, annotate_mosaic


    ####################################################################################################################
    dataset_names = ['Imstat', 'Flowcyt', 'LT1', 'LT2']

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    y_trains = []
    y_tests = []
    for ds in dataset_names:

        # Load the sample-wise data
        base_path = os.path.join(os.getcwd(), 'data/np_files', conversion_mapping_datasets[ds], 'arcsinh_cofactor150')

        y_train_dir = os.path.join(base_path, 'sample_wise_train')
        num_y_trains = len([f for f in os.listdir(y_train_dir) if f.startswith('y_')])
        y_train_filenames = [f'y_sample_{str(i).zfill(2)}_train.npy' for i in range(num_y_trains)]
        y_trains.append([np.load(os.path.join(y_train_dir, f)).astype(int) for f in y_train_filenames])

        y_test_dir = os.path.join(base_path, 'sample_wise_test')
        num_y_test = len([f for f in os.listdir(y_test_dir) if f.startswith('y_')])
        y_test_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(num_y_test)]
        y_tests.append([np.load(os.path.join(y_test_dir, f)).astype(int) for f in y_test_filenames])

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 11), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        """
    )

    for ds, ytr, yte, labels in zip(dataset_names, y_trains, y_tests, ['AB', 'CD', 'EF', 'GH']):

        plot_sample_sizes(
            ys=ytr, title=f'{ds} Train', abline_mean=True, abline_std=True, print_total=True, ax=axd[labels[0]]
        )
        plot_sample_sizes(
            ys=yte, title=f'{ds} Test', abline_mean=True, abline_std=True, print_total=True, ax=axd[labels[1]]
        )

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig('./results/plots/dataset_sizes_supplement.png', dpi=fig.dpi)


def main_dataset_balance_plot_supplement():

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from validation.plt import plot_class_balance, annotate_mosaic

    ####################################################################################################################
    dataset_names = ['Imstat', 'Flowcyt', 'LT1', 'LT2', 'LT1 b', 'LT2 b']

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    y_trains = []
    y_tests = []
    for ds in dataset_names:
        # Load the sample-wise data
        base_path = os.path.join(os.getcwd(), 'data/np_files', conversion_mapping_datasets[ds],
                                 'arcsinh_cofactor150')

        y_train_dir = os.path.join(base_path, 'sample_wise_train')
        num_y_trains = len([f for f in os.listdir(y_train_dir) if f.startswith('y_')])
        y_train_filenames = [f'y_sample_{str(i).zfill(2)}_train.npy' for i in range(num_y_trains)]
        y_trains.append([np.load(os.path.join(y_train_dir, f)).astype(int) for f in y_train_filenames])

        y_test_dir = os.path.join(base_path, 'sample_wise_test')
        num_y_test = len([f for f in os.listdir(y_test_dir) if f.startswith('y_')])
        y_test_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(num_y_test)]
        y_tests.append([np.load(os.path.join(y_test_dir, f)).astype(int) for f in y_test_filenames])

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 11), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AABB
        CCDD
        EEFF
        GGHH
        IJKL
        """
    )

    for ds, ytr, yte, labels in zip(dataset_names, y_trains, y_tests, ['AB', 'CD', 'EF', 'GH', 'IJ', 'KL']):
        plot_class_balance(
            ys=ytr, title=f'{ds} Train', palette=None, ax=axd[labels[0]]
        )
        plot_class_balance(
            ys=yte, title=f'{ds} Test', ax=axd[labels[1]]
        )

    # ### Manually adjust axis labels
    # for key in ['F', 'G', 'H', 'J', 'K', 'L']:
    #     ax = axd[key]
    #     ax.set_ylabel(None)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig('./results/plots/dataset_balances_supplement.png', dpi=fig.dpi)






def main_pipeline_workflow_som():

    import os
    import random
    import readfcs
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    random.seed(42)

    from seaborn import scatterplot
    from flagx import GatingPipeline

    # ###### Initial training ###### #
    # ### Set parameters
    save_path = os.path.join(os.getcwd(), 'results/pipeline_workflow_som1')
    os.makedirs(save_path, exist_ok=True)

    data_dir = os.path.join(os.getcwd(), 'data/raw/imstat')
    data_fns = sorted(os.listdir(data_dir))
    random.shuffle(data_fns)

    train_data_fns = data_fns[0:75]
    test_data_fns = data_fns[75:78]

    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    label_key = 'population'

    preprocessing_kwargs = {'flavour': 'arcsinh', 'flavour_kwargs': {'cofactor': 150}}

    gating_method_kwargs = {
        'som_topology': 'planar',
        'som_grid_type': 'rectangular',
        'som_dimensions': (10, 10),
        'neighborhood': 'gaussian',
        'gaussian_neighborhood_sigma': 0.25,
        'initialization': 'pca',
        'n_epochs': 200,
        'radius_0': -0.25,
        'radius_n': 0.01,
        'radius_cooling': 'linear',
        'learning_rate_0': 0.5,
        'learning_rate_n': 0.05,
        'learning_rate_decay': 'exponential',
        'verbosity': 2
    }

    # ### Instantiate the pipeline, train, and save
    gp = GatingPipeline(
        train_data_file_path=data_dir,
        train_data_file_names=train_data_fns,
        train_data_file_type='fcs',
        save_path=save_path,
        channels=channels,
        label_key=label_key,
        channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
        relabel_data_kwargs=None,
        preprocessing_kwargs=preprocessing_kwargs,
        gating_method='som',
        gating_method_kwargs=gating_method_kwargs,
        verbosity=2,
    )

    gp.train()

    print("Gating som unit labels training", gp.gating_module_.som_unit_labels_)

    gp.save(filename='trained_pipeline.pkl', filepath=None)

    del gp

    # ###### Inference with new data ###### #
    # ### Set parameters
    output_dir = os.path.join(save_path, 'output')
    os.makedirs(output_dir, exist_ok=True)

    dim_red_methods = ('som', 'pca', 'umap')
    dim_red_method_kwargs = (None, None, {'n_jobs': 12})

    # ### Load the pipeline
    gp = GatingPipeline.load(filename='trained_pipeline.pkl', filepath=save_path)

    print("Gating som unit labels inference", gp.gating_module_.som_unit_labels_)

    gp.inference(
        data_file_path=data_dir,
        data_file_names=test_data_fns,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_sample_wise=False,
        save_path=output_dir,
        save_filenames='annotated_test_data.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    del gp

    # ###### Output validation ###### #

    annotated_test_data = readfcs.view(os.path.join(output_dir, 'annotated_test_data.fcs'))
    print("# ### Annotated test data:\n", annotated_test_data)
    df = annotated_test_data[1]
    df['pred_unscaled'] = df['pred_unscaled'].astype(int)
    print("# Channels:\n", df.columns)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=df, x=f'{drm}_1', y=f'{drm}_2', s=1, hue='pred_unscaled', palette='deep', ax=ax)
        plt.legend(title='Pred', markerscale=4)
        plt.savefig(os.path.join(output_dir, f'{drm}.png'), dpi=300)
        plt.close('all')


def main_pipeline_workflow_fcnn():

    import os
    import random
    import readfcs
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    random.seed(42)

    from seaborn import scatterplot
    from flagx import GatingPipeline

    # ###### Initial training ###### #
    # ### Set parameters
    save_path = os.path.join(os.getcwd(), 'results/pipeline_workflow_fcnn')
    os.makedirs(save_path, exist_ok=True)

    data_dir = os.path.join(os.getcwd(), 'data/raw/imstat')
    data_fns = sorted(os.listdir(data_dir))
    random.shuffle(data_fns)

    train_data_fns = data_fns[0:25]
    test_data_fns = data_fns[75:78]

    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    label_key = 'population'

    preprocessing_kwargs = {'flavour': 'arcsinh', 'flavour_kwargs': {'cofactor': 150}}

    gating_method_kwargs = {'layer_sizes': (128, 64, 32), 'n_epochs': 20, 'device': 'cuda', 'verbosity': 2}

    # ### Instantiate the pipeline, train, and save
    gp = GatingPipeline(
        train_data_file_path=data_dir,
        train_data_file_names=train_data_fns,
        train_data_file_type='fcs',
        save_path=save_path,
        channels=channels,
        label_key=label_key,
        channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
        relabel_data_kwargs=None,
        preprocessing_kwargs=preprocessing_kwargs,
        gating_method='fcnn_softmax',
        gating_method_kwargs=gating_method_kwargs,
        verbosity=2,
    )

    gp.train()

    gp.save(filename='trained_pipeline.pkl', filepath=None)

    del gp

    # ###### Inference with new data ###### #
    # ### Set parameters
    output_dir = os.path.join(save_path, 'output')
    os.makedirs(output_dir, exist_ok=True)

    dim_red_methods = ('pca', 'umap')
    dim_red_method_kwargs = (None, {'n_jobs': 12})

    # ### Load the pipeline
    gp = GatingPipeline.load(filename='trained_pipeline.pkl', filepath=save_path)

    gp.inference(
        data_file_path=data_dir,
        data_file_names=test_data_fns,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_sample_wise=False,
        save_path=output_dir,
        save_filenames='annotated_test_data.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    del gp

    # ###### Output validation ###### #

    output_dir = os.path.join(save_path, 'output')
    dim_red_methods = ('pca', 'umap')

    annotated_test_data = readfcs.view(os.path.join(output_dir, 'annotated_test_data.fcs'))
    print("# ### Annotated test data:\n", annotated_test_data)
    df = annotated_test_data[1]

    print(type(df))
    df['pred_unscaled'] = df['pred_unscaled'].astype(int)
    print("# Channels:\n", df.columns)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=df, x=f'{drm}_1', y=f'{drm}_2', s=1, hue='pred_unscaled', palette='deep', ax=ax)
        plt.legend(title='Pred', markerscale=4)
        plt.savefig(os.path.join(output_dir, f'{drm}.png'), dpi=300)
        plt.close('all')



if __name__ == '__main__':

    # main_data_preparation()

    # main_param_influence_study()

    # main_param_tuning()

    # main_n_epochs_calibration()

    # main_som_classifier()

    # main_gatemeclass()

    # main_dgcytof()

    # main_softmax()

    # main_n_samples_experiment() # todo: started on ramses and weneg

    # main_performance_score_plots()

    # main_cell_percentage_plots()

    # main_time_table()

    # main_performance_plots()

    main_n_samples_plot()

    # main_performance_plots_supplement()

    # main_dataset_size_plot_supplement()

    # main_dataset_balance_plot_supplement()



    # main_prec_vs_recall()  # todo

    # main_pipeline_workflow_som()

    # main_pipeline_workflow_fcnn()


    # Todo: pipeline output for fcnn is not df?
    # Todo: add class-wise results and dataset plots -> add percentages in legend
    # Todo: continue with writing
    # Todo: generate plots and time table with new results

    # Todo: adjust scaling in export
    # Todo: Generate annotated.fcs an Stefan
    # Todo: Restructure supplement to adhere to manuscript structure, adjust references

    print('done')

