

def main_data_processing():
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
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import List, Dict, Union, Tuple, Any
    from flagx.utils import set_random_seed
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
                set_random_seed(seed=seed)

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
                            shuffle=True,  # Used pytorch dataloader under the hood, seed is set by set_random_seed()
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
    Script for running parameter-wise hyperparameter tuning on the
    immune status dataset. Its purpose is to determine whether the
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

    from flagx.gating import SomClassifier
    from validation.plt import plot_param_lineplot, plot_param_stripplot
    from validation.utils import set_pandas_print_options, get_downsampling_bool

    # ### Set flags and variables ######################################################################################
    inference = True  # Whether to do the hyperparameter tuning or just view the results
    test_n_epochs = True

    random_seed = 42
    downsampling_frac = 0.10
    n_splits = 3

    trafo = 'log10_w_custom_cutoffs'  # 'arcsinh_cofactor150', 'log10_w_custom_cutoffs'

    # Based on gridsearch for n_epochs, selected n_epochs such that performance is stable with default parameters
    n_epochs = 6000   # arcsinh_cofactor150: 6000, log10_w_custom_cutoffs: 6000
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





if __name__ == '__main__':

    # main_data_processing()

    # main_som_parameter_influence_study()

    print('done')