

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

    # Based on gridsearch for n_epochs, selected n_epochs such that performance is stable with default parameters
    n_epochs = list(range(100, 901, 100)) + list(range(1000, 10001, 1000))
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
    som_clf_param_tuning = SomClassifier.load(filepath=os.path.join(os.getcwd(), f'results/parameter_tuning/{trafo}'))
    best_params = som_clf_param_tuning.grid_search_.best_params_

    print(best_params)

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

    data_sets = [
        'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'lymphoma_tube1', 'lymphoma_tube2', 'flowcyt'
    ]
    others_labels = [None, None, None, None, 5]
    pos_labels = [1, 1, None, None, None]

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
            if not os.path.exists(data_p):
                print(f'# ### Found no data. Continue.\n')
                continue

            # Define path where results will be saved to
            save_p = os.path.join(os.getcwd(), f'results/pred_eval/som_classifier/{data_set}/{trafo}')
            os.makedirs(save_p, exist_ok=True)

            if fit:

                # Load training data
                x_train = np.load(os.path.join(data_p, 'x_train.npy'))
                y_train = np.load(os.path.join(data_p, 'y_train.npy'))

                # Instantiate the SOM classifier
                # Todo: parameters
                som_clf = SomClassifier(
                    som_topology='planar',
                    som_grid_type='rectangular',
                    som_dimensions=(20, 20),
                    neighborhood='gaussian',
                    gaussian_neighborhood_sigma=0.1,
                    initialization='pca',
                    n_epochs=800,
                    radius_0=-0.75,
                    radius_n=0.01,
                    radius_cooling='linear',
                    learning_rate_0=1.0,
                    learning_rate_n=0.01,
                    learning_rate_decay='linear',
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


def main_som_plots():
    pass


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

    allow_abstention = False
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
                samples_p = os.path.join(data_p, 'sample_wise_test')
                n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
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
                for y, sn in zip(samples_y_pred, sample_names):
                    np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y)

            else:
                # Load the predictions
                try:
                    y_pred = np.load(os.path.join(save_p, 'y_pred.npy'))
                except FileNotFoundError:
                    print(f'# ### Fit was not successful for dataset: {data_set}, trafo: {trafo}. Continue.\n')
                    continue

                samples_y_pred_p = os.path.join(save_p, 'samples_y_pred')
                samples_y_pred_filenames = [
                    f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(len(os.listdir(samples_y_pred_p)))
                ]
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

                # Load the labels of the sample-wise test data
                samples_p = os.path.join(data_p, 'sample_wise_test')
                n_samples = len([f for f in os.listdir(samples_p) if f.startswith('y_')])
                # Get the test samples for which the fit was successful
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
    import torch
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
                # Load the SOM classifier
                try:
                    torch.serialization.safe_globals([DgcytofClassifier])
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
                samples_p = os.path.join(data_p, 'sample_wise_test')
                n_samples = len([f for f in os.listdir(samples_p) if f.startswith('x_')])
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
                for y, sn in zip(samples_y_pred, sample_names):
                    np.save(os.path.join(save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y)

            else:
                # Load the predictions
                try:
                    y_pred = np.load(os.path.join(save_p, 'y_pred.npy'))
                except FileNotFoundError:
                    print(f'# ### Fit was not successful for dataset: {data_set}, trafo: {trafo}. Continue.\n')
                    continue

                samples_y_pred_p = os.path.join(save_p, 'samples_y_pred')
                samples_y_pred_filenames = [
                    f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(len(os.listdir(samples_y_pred_p)))
                ]
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

                # Load the labels of the sample-wise test data
                samples_p = os.path.join(data_p, 'sample_wise_test')
                n_samples = len([f for f in os.listdir(samples_p) if f.startswith('y_')])
                # Get the test samples for which the fit was successful
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
    data_set = 'lymphoma_tube1_binary'
    # 'imstat', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary', 'flowcyt'

    preprocessing_trafo = 'log10_channelwisecutoff'
    # 'arcsinh_cofactor150', 'log10_channelwisecutoff', 'log10_cutoff100'

    random_sample_order = True
    sample_order_file = os.path.join(
        os.getcwd(), 'results/n_samples_n_events', f'sample_order_{data_set}_{preprocessing_trafo}.txt'
    )

    inference = False  # !!!!!!

    others_label = None
    # 8, None, None, None, None, 5

    pos_label = 1
    # None, None, None, 1, 1, None

    classifier = 'softmax'  # 'som', 'softmax'

    abstention_label = -1 if classifier == 'som' else None

    downsampled_data_subdirs = ['0_01', '0_05', '0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7', '0_8', '0_9', '1_0']

    ####################################################################################################################

    sample_order_str = 'random' if random_sample_order else 'ordered'

    save_p = os.path.join(
        os.getcwd(), 'results/n_samples_n_events', classifier, data_set, preprocessing_trafo, sample_order_str
    )
    os.makedirs(save_p, exist_ok=True)

    if inference:
        # Todo: set parameters
        if classifier == 'som':
            clf = SomClassifier(
                som_topology='planar',
                som_grid_type='rectangular',
                som_dimensions=(3, 3),  # Todo
                neighborhood='gaussian',
                gaussian_neighborhood_sigma=0.1,
                initialization='pca',
                n_epochs=6,  # Todo
                radius_0=-0.75,
                radius_n=0.01,
                radius_cooling='linear',
                learning_rate_0=1.0,
                learning_rate_n=0.01,
                learning_rate_decay='linear',
                verbosity=1,
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
            downsampled_data_subdirs=downsampled_data_subdirs,
            data_p=os.path.join(os.getcwd(), 'data/np_files', data_set, preprocessing_trafo),
            save_p=save_p,
            abstention_label=abstention_label,
            others_label=others_label,
            pos_label=pos_label,
            random_sample_order=random_sample_order,
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




if __name__ == '__main__':

    # main_data_preparation()

    # main_param_influence_study()

    # main_param_tuning()

    # main_n_epochs_calibration()  # todo

    # main_som_classifier()  # todo

    main_gatemeclass()  # todo: started inference on woody (no abstention)

    # main_dgcytof()

    # main_softmax()

    # main_n_samples_experiment()
    # todo: softmax random (weneg, ne0), softmax ordered (weneg, ne1)

    # todo: need n samples no ds as well

    # main_prec_vs_recall()

    print('done')

