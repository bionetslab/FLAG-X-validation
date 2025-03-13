
def main_data_preparation():
    """
    Script for loading the .fcs files of our dataset, perform data split on sample level,
    preprocess/apply transformation, subset to selected channels, save resulting data matrices to .npy files
    (also sample-wise)

    Returns:
        None
    """

    import os
    import random
    import torch
    import pandas as pd
    import numpy as np
    import scanpy as sc
    from typing import List, Dict, Union, Tuple, Any
    from flowsrc.flowdata import FlowDataManager

    # ### Define function for setting random seeds
    def set_random_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ### Define custom data transformation function
    def log10_trafo_w_cutoff_channel_wise(adata: sc.AnnData):
        cutoff_dict = {
            'FS INT': 100000, 'SS INT': 20000, '16-FITC': 250, '56-PE': 450, '3-ECD': 700, '4-PC7': 1200,
            '19-APC': 1700, '14-APC700': 900, '8-PB': 450, '45-CO': 500
        }

        x = adata.X.copy()
        for channel, cutoff in cutoff_dict.items():
            col_idx = np.where(adata.var_names == channel)[0][0]
            x_col = adata.X[:, col_idx].copy()
            mask = (x_col > cutoff)
            x_col[mask] = np.log10(x_col[mask])
            x_col[~mask] = np.log10(cutoff)
            x[:, col_idx] = x_col

        adata.X = x

    # ### Define helper function
    def helper(
            datasets: List[str],
            raw_data_paths: List[str],
            data_file_types: List[str],
            data_splits: List[Union[Tuple[float, float], pd.DataFrame]],
            channel_names: List[List[str]],
            label_keys: List[str],
            preprocessing_flavours: List[str],
            preprocessing_kwargs: List[Dict[str, Any]],
    ):
        # ### Set random seed
        set_random_seed()

        # ### Iterate over the parameters
        for dataset, raw_data_path, data_file_type, data_split, channels, og_label_key in zip(
                datasets, raw_data_paths, data_file_types, data_splits, channel_names, label_keys
        ):
            for prepr_flavour, prepr_kwargs in zip(preprocessing_flavours, preprocessing_kwargs):

                # Skip channel-wise cutoff if dataset is not aml
                if prepr_flavour == 'custom' and dataset != 'aml':
                    continue

                # ### Create a directory where the processed files will be stored
                # Get string of stringified preprocessing kwargs
                prepr_kwargs_str = ''.join(
                    f"{key}{value.__name__ if callable(value) else value}"
                    for key, value in prepr_kwargs.items()
                )
                # Define path where results should be stored
                np_data_p = os.path.join(
                    os.getcwd(),
                    f'input/np_files/{dataset}/{prepr_flavour + prepr_kwargs_str}'
                )
                # Create dir
                os.makedirs(os.path.join(np_data_p, 'data_handling'), exist_ok=True)

                # ### Load and process the data set
                # Create a list of the filenames
                filename_list = os.listdir(os.path.join(os.getcwd(), raw_data_path))

                # Instantiate the FlowDataManager
                fdm = FlowDataManager(
                    data_file_names=filename_list,
                    data_file_type=data_file_type,
                    data_file_path=raw_data_path,
                    save_path=os.path.join(np_data_p, 'data_handling'),
                    memory_saving=False,
                    verbosity=1
                )

                # Load data files to anndata
                fdm.load_data_files_to_anndata()

                # Relabel lymphoma data to binary labels (b-cells vs others)
                label_key = og_label_key  # Reset label key
                if dataset == 'lymphoma_tube1_binary' or dataset == 'lymphoma_tube2_binary':
                    fdm.relabel_data(
                        data_set='all',
                        old_to_new_label_mapping={1: 0, 9: 0, 7: 1, 10: 1},
                        label_key=label_key,
                        label_layer_key=None,  # No preprocessing was done yet
                        new_label_key='new_labels',
                    )

                    label_key = 'new_labels'

                # Check the number of events per sample
                fdm.check_sample_sizes(out_filename='sample_sizes.csv')

                # Align channel names, use 1st sample as reference, save original channel names to file
                fdm.align_channel_names(reference_channel_names=0, out_filename='og_channel_names.csv')

                # Apply preprocessing transformation, save non-transformed to layer 'original'
                fdm.sample_wise_preprocessing(flavour=prepr_flavour, save_raw_to_layer='original', **prepr_kwargs)

                # Split the samples into a train and test set, save split (only if no split is passed as input)
                fdm.perform_data_split(
                    data_split=data_split,
                    save=True,
                    **{'shuffle': True, 'random_state': 42}
                )

                # Create dataloaders
                dl_train = fdm.create_data_loader(
                    data_set='train',
                    channels=channels,
                    layer_key=None,
                    label_key=label_key,
                    label_layer_key='original',
                    return_data_loader='np_array',
                    batch_size=-1,
                    shuffle=True,
                    on_disk=True,
                    filename='data_train.npy'
                )
                dl_test = fdm.create_data_loader(
                    data_set='test',
                    channels=channels,
                    layer_key=None,
                    label_key=label_key,
                    label_layer_key='original',
                    return_data_loader='np_array',
                    batch_size=-1,
                    shuffle=False,
                    on_disk=True,
                    filename='data_test.npy'
                )

                # Check the class balance in the train- and test-set
                fdm.check_class_balance(
                    data_loader=dl_train,
                    save=True,
                    plot=True,
                    save_path=fdm.save_path,
                    filename_df='class_balance_train.csv',
                    filename_plot='class_balance_train.png',
                    ax=None,
                    verbosity=fdm.verbosity,
                )

                fdm.check_class_balance(
                    data_loader=dl_test,
                    save=True,
                    plot=True,
                    save_path=fdm.save_path,
                    filename_df='class_balance_test.csv',
                    filename_plot='class_balance_test.png',
                    ax=None,
                    verbosity=fdm.verbosity,
                )

                # Batch size was set to -1 => dataloaders contain one array with all the data, extract and save it
                x_train, y_train = next(iter(dl_train))
                x_test, y_test = next(iter(dl_test))

                np.save(os.path.join(np_data_p, 'x_train.npy'), x_train.astype(float))
                np.save(os.path.join(np_data_p, 'y_train.npy'), y_train.astype(int))
                np.save(os.path.join(np_data_p, 'x_test.npy'), x_test.astype(float))
                np.save(os.path.join(np_data_p, 'y_test.npy'), y_test.astype(int))

                # ### Save the sample wise numpy arrays for the train data, use inbuilt function
                save_p_np_train = os.path.join(np_data_p, 'sample_wise_train')
                os.makedirs(save_p_np_train, exist_ok=True)

                fdm.datalist_to_numpy(
                    data_list=fdm.train_data_,
                    sample_wise=True,
                    save_path=save_p_np_train,
                    filename_suffix='_train',
                    data_path=fdm.save_path,  # Not needed here
                    channels=channels,
                    layer_key=None,  # Use adata.X (preprocessed)
                    label_key=label_key,
                    label_layer_key='original',
                    shuffle=True,
                )

                save_p_np_test = os.path.join(np_data_p, 'sample_wise_test')
                os.makedirs(save_p_np_test, exist_ok=True)

                fdm.datalist_to_numpy(
                    data_list=fdm.test_data_,
                    sample_wise=True,
                    save_path=save_p_np_test,
                    filename_suffix='_test',
                    data_path=fdm.save_path,  # Not needed here
                    channels=channels,
                    layer_key=None,  # Use adata.X (preprocessed)
                    label_key=label_key,
                    label_layer_key='original',
                    shuffle=False,  # Do not shuffle test data
                )

    # ### Set flags and important variables here #######################################################################
    # ### AML data ###
    ds_aml = 'aml'
    raw_data_p_aml = os.path.join(os.getcwd(), 'input/raw/aml')
    data_file_type_aml = 'fcs'
    data_split_aml = pd.read_csv(os.path.join(os.getcwd(), 'input/raw/aml_data_split_development.csv'), index_col=0)
    channels_aml = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    label_key_aml = 'population'

    # ### Flowcyt ###
    ds_flowcyt = 'flowcyt'

    raw_data_p_flowcyt = os.path.join(os.getcwd(), 'input/raw/flowcyt/data_original')
    data_file_type_flowcyt = 'csv'
    data_split_flowcyt = (0.75, 0.25)
    channels_flowcyt = [
            'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
            'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
            'FL10 INT_CD45-KO'
        ]
    label_key_flowcyt = 'label'

    # ### Lymphoma tube 1 ###
    ds_lt1 = 'lymphoma_tube1'
    raw_data_p_lt1 = os.path.join(
        os.getcwd(),
        'input/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels'
    )
    data_file_type_lt1 = 'fcs'
    data_split_lt1 = (0.75, 0.25)
    channels_lt1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    label_key_lt1 = 'population'

    # ### Lymphoma tube 1, binary case, ###
    ds_lt1_binary = 'lymphoma_tube1_binary'
    raw_data_p_lt1_binary = os.path.join(
        os.getcwd(),
        'input/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels'
    )
    data_file_type_lt1_binary = 'fcs'
    data_split_lt1_binary = (0.75, 0.25)
    channels_lt1_binary = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    label_key_lt1_binary = 'population'

    # ### Lymphoma tube 2 ###
    ds_lt2 = 'lymphoma_tube2'
    raw_data_p_lt2 = os.path.join(
        os.getcwd(),
        'input/raw/lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels'
    )
    data_file_type_lt2 = 'fcs'
    data_split_lt2 = (0.75, 0.25)
    channels_lt2 = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7', 'CD52_APC', 'CD11c_APC_A700',
        'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]
    label_key_lt2 = 'population'

    # ### Lymphoma tube 2, binary case ###
    ds_lt2_binary = 'lymphoma_tube2_binary'
    raw_data_p_lt2_binary = os.path.join(
        os.getcwd(),
        'input/raw/lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels'
    )
    data_file_type_lt2_binary = 'fcs'
    data_split_lt2_binary = (0.75, 0.25)
    channels_lt2_binary = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7', 'CD52_APC', 'CD11c_APC_A700',
        'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]
    label_key_lt2_binary = 'population'


    datasets = [ds_aml, ds_flowcyt, ds_lt1, ds_lt2, ds_lt1_binary ,ds_lt2_binary]
    raw_data_paths = [
        raw_data_p_aml, raw_data_p_flowcyt, raw_data_p_lt1, raw_data_p_lt2, raw_data_p_lt1_binary, raw_data_p_lt2_binary
    ]
    data_file_types = [
        data_file_type_aml, data_file_type_flowcyt, data_file_type_lt1, data_file_type_lt2, data_file_type_lt1_binary,
        data_file_type_lt2_binary
    ]
    data_splits = [
        data_split_aml, data_split_flowcyt, data_split_lt1, data_split_lt2, data_split_lt1_binary, data_split_lt2_binary
    ]
    channel_names = [
        channels_aml, channels_flowcyt, channels_lt1, channels_lt2, channels_lt1_binary, channels_lt2_binary
    ]
    label_keys = [
        label_key_aml, label_key_flowcyt, label_key_lt1, label_key_lt2, label_key_lt1_binary, label_key_lt2_binary
    ]

    # Preprocessing
    prepr_flavours = ['arcsinh', 'log10_w_cutoff', 'custom']
    prepr_kwargs = [{'cofactor': 150}, {'cutoff': 100}, {'preprocessing_method': log10_trafo_w_cutoff_channel_wise, }]

    ####################################################################################################################
    # Run data processing
    helper(
        datasets=datasets,
        raw_data_paths=raw_data_paths,
        data_file_types=data_file_types,
        data_splits=data_splits,
        channel_names=channel_names,
        label_keys=label_keys,
        preprocessing_flavours=prepr_flavours,
        preprocessing_kwargs=prepr_kwargs
    )


def main_parameter_tuning():
    """
    Script for running the parameter tuning for the SOM classifier
    Returns: None
    """

    import os
    import matplotlib
    import numpy as np
    import pandas as pd
    from typing import Tuple
    from sklearn.model_selection import train_test_split, PredefinedSplit
    from flowsrc.flowsom import SomClassifier

    # ### Define function for stratified downsampling of training data
    def stratified_downsampling(
            X: np.ndarray,
            y: np.ndarray,
            ds_fraction: float = 0.25,
            random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ### Set random seed
        np.random.seed(random_seed)
        # ### Downsample train set to half its size in a stratified fashion for this study
        # Calculate the original class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_count = y.shape[0]
        class_ratios = class_counts / total_count
        # print(class_ratios)
        # print(class_counts)
        # print(x_train.shape)
        # Calculate the desired count for each class
        target_size = np.ceil(total_count * ds_fraction)
        target_counts = (class_ratios * target_size).round().astype(int)
        # Downsample
        downsampled_x = []
        downsampled_y = []
        for cls, count in zip(unique_classes, target_counts):
            # Get indices of the current class
            class_indices = np.where(y == cls)[0]
            # Randomly sample from these indices
            sampled_indices = np.random.choice(class_indices, size=count, replace=False)
            # Append downsampled data and labels
            downsampled_x.append(X[sampled_indices])
            downsampled_y.append(y[sampled_indices])
        # Concatenate results
        x = np.vstack(downsampled_x)
        y = np.concatenate(downsampled_y)

        return x, y

    # ### Set flags and important variables here #######################################################################
    # Data
    # data_p = os.path.join(os.getcwd(), 'input/np_files/our_data/arcsinhcofactor150')
    # data_p = os.path.join(os.getcwd(), 'input/np_files/our_data/custompreprocessing_methodlog10_trafo_w_cutoff_100')
    data_p = os.path.join(os.getcwd(), 'input_old/np_files/our_data/custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise')

    # Downsampling
    downsampling = True
    ds_frac = 0.25  # Only relevant when downsampling is True

    # Cross validation, data split
    cross_validation = False
    cv = 5
    val_size = 0.32  # Only relevant when cross_validation is False

    # ### Parameter grid
    n_epochs = 10000

    dummy_param_grid = {
        'som_topology': ['planar'],
        'som_grid_type': ['rectangular', ],
        'som_dimensions': [(20, 20), (30, 30), ],
        'neighborhood': ['gaussian', ],
        'gaussian_neighborhood_sigma': [0.5, ],
        'initialization': ['pca', ],
        'n_epochs': [n_epochs, ],
        'radius_0': [-0.75, ],
        'radius_n': [0.1, ],
        'radius_cooling': ['linear', ],
        'learning_rate_0': [0.1, ],
        'learning_rate_n': [0.01, ],
        'learning_rate_decay': ['linear', ],
    }

    param_grid = {
        'som_topology': ['planar', ],
        'som_grid_type': ['rectangular', ],
        'som_dimensions': [(10, 10), (20, 20), (30, 30)],
        'neighborhood': ['gaussian', ],
        'gaussian_neighborhood_sigma': [0.5, 0.25, 0.1],
        'initialization': ['pca', ],
        'n_epochs': [n_epochs, ],
        'radius_0': [-0.5, -0.75],
        'radius_n': [0.75, 0.01, 0.25],
        'radius_cooling': ['linear', ],
        'learning_rate_0': [1.0, 0.2],
        'learning_rate_n': [0.1, 0.01],
        'learning_rate_decay': ['linear', ],
    }

    ####################################################################################################################

    # ### Set the matplotlib backend
    matplotlib.use('Agg')

    # ### Set pandas print options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Define path where results are stored, create dir if necessary
    input_data_trafo = os.path.basename(data_p)
    res_p = os.path.join(
        os.getcwd(),
        f'results/param_tuning/{input_data_trafo}_{f"ds{ds_frac}" if downsampling else "nods"}_'
        f'{f"cv{cv}" if cross_validation else f"nocv{val_size}"}'
    )
    if not os.path.exists(res_p):
        os.makedirs(res_p)

    # ### Load the previously processed data
    x_train = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
    y_train = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int_)

    # ### Downsample
    if downsampling:
        print(f'# ### Downsampling training data set to {ds_frac * 100}% of its original size ...')
        print(f'# Before: {x_train.shape}')
        x_train, y_train = stratified_downsampling(X=x_train, y=y_train, ds_fraction=ds_frac, random_seed=42)
        print(f'# After: {x_train.shape}')

    # ### Train-val-split if no cross validation is to be done
    if not cross_validation:
        x_train_dummy, x_val_dummy, y_train_dummy, y_val_dummy = train_test_split(
            x_train, y_train, test_size=val_size, stratify=y_train, random_state=42
        )

        # Reconcatenate the data (first train, then val)
        x_train = np.vstack((x_train_dummy, x_val_dummy))
        y_train = np.hstack((y_train_dummy, y_val_dummy))

        # Create a PredefinedSplit according to the previous data split
        # (https://scikit-learn.org/1.5/modules/cross_validation.html#predefined-split)
        val_fold = [-1] * len(x_train_dummy) + [0] * len(x_val_dummy)
        cv = PredefinedSplit(test_fold=val_fold)

    # ### Instantiate the SOM classifier
    som_c = SomClassifier(verbosity=2)

    # ### Perform the hyperparameter tuning
    som_c.hyperparameter_tuning(
        X=x_train.copy(), y=y_train.copy(), param_grid=param_grid, cv=cv, scoring='internal', refit=False,
    )

    # ### Save SOM the classifier
    som_c.save(filepath=res_p)

    print(pd.DataFrame(som_c.grid_search_.cv_results_))


def main_n_epochs_calibration():
    import os
    import matplotlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, PredefinedSplit
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_lineplot

    # ### Set flags and important variables here #######################################################################

    inference = True  # Whether to run the gridsearch or just load the result and plot

    # trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_100_ds0.25_nocv0.32',
    trafo_param_tuning= 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise_ds0.25_nocv0.32'
    trafo_data = 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'

    n_epochs = list(range(100, 901, 100)) + list(range(1000, 10001, 1000))

    val_size = 0.32
    ####################################################################################################################

    # ### Set the matplotlib backend
    matplotlib.use('Agg')

    # ### Set pandas print options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # Load the parameter tuning results and print
    clf = SomClassifier.load(
        filepath=f'./results/param_tuning/{trafo_param_tuning}'
    )

    print(f'# ### Best parameters:\n{clf.grid_search_.best_params_}')

    best_params = {key: [val, ] for key, val in clf.grid_search_.best_params_.items()}
    best_params['n_epochs'] = n_epochs

    # best_params_log10 = {
    #     'gaussian_neighborhood_sigma': [0.1, ],
    #     'initialization': ['pca', ],
    #     'learning_rate_0': [1.0, ],
    #     'learning_rate_decay': ['linear', ],
    #     'learning_rate_n': [0.01, ],
    #     'n_epochs': n_epochs,
    #     'neighborhood': ['gaussian', ],
    #     'radius_0': [-0.75, ],
    #     'radius_cooling': ['linear', ],
    #     'radius_n': [0.01, ],
    #     'som_dimensions': [(30, 30), ],
    #     'som_grid_type': ['rectangular', ],
    #     'som_topology': ['planar', ]
    # }

    data_p = os.path.join(os.getcwd(), f'input_old/np_files/our_data/{trafo_data}')

    # ### Define path where results are stored, create dir if necessary
    res_p = os.path.join(
        os.getcwd(),
        f'results/param_tuning/n_epoch_calibration/{trafo_data}_nods_nocv{val_size}'
    )
    os.makedirs(res_p, exist_ok=True)

    if inference:

        # ### Load the previously processed data
        x_train = np.load(os.path.join(data_p, 'x_train.npy')).astype(np.float32)
        y_train = np.load(os.path.join(data_p, 'y_train.npy')).astype(np.int_)

        # ### Train-val-split
        x_train_dummy, x_val_dummy, y_train_dummy, y_val_dummy = train_test_split(
            x_train, y_train, test_size=val_size, stratify=y_train, random_state=42
        )

        # Reconcatenate the data (first train, then val)
        x_train = np.vstack((x_train_dummy, x_val_dummy))
        y_train = np.hstack((y_train_dummy, y_val_dummy))

        # Create a PredefinedSplit according to the previous data split
        # (https://scikit-learn.org/1.5/modules/cross_validation.html#predefined-split)
        val_fold = [-1] * len(x_train_dummy) + [0] * len(x_val_dummy)
        cv = PredefinedSplit(test_fold=val_fold)

        # ### Instantiate the SOM classifier
        som_c = SomClassifier(verbosity=2)

        # ### Perform the hyperparameter tuning
        som_c.hyperparameter_tuning(
            X=x_train.copy(), y=y_train.copy(), param_grid=best_params, cv=cv, scoring='internal', refit=False,
        )

        # ### Save SOM the classifier
        som_c.save(filepath=res_p)

    som_c = SomClassifier.load(filepath=res_p)

    res_df = pd.DataFrame(som_c.grid_search_.cv_results_)
    res_df.sort_values(by=['param_n_epochs'], ascending=True, inplace=True)

    plot_param_lineplot(
        res_df=res_df,
        x_col='param_n_epochs',
        y_col='mean_test_score',
        custom_x_ticks='auto',
        x_label='N Epochs',
        y_label='Macro F1 Score',
        dpi=300,
    )

    plt.savefig(os.path.join(res_p, 'n_epochs.png'))


def main_som_classifier():

    import os
    import time
    import numpy as np
    import pandas as pd
    from flowsrc.flowsom import SomClassifier
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    for data_set in data_sets:
        for trafo in preprocessing_trafos:

            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # Load the train and (sample-wise) test data
            data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
            samples_p = os.path.join(data_p, 'sample_wise_test')

            x_train = np.load(os.path.join(data_p, 'x_train.npy'))
            y_train = np.load(os.path.join(data_p, 'y_train.npy'))

            x_test = np.load(os.path.join(data_p, 'x_test.npy'))
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            samples_x_test = [f for f in os.listdir(samples_p) if f.startswith('x_test')]
            samples_y_test = [f for f in os.listdir(samples_p) if f.startswith('y_test')]

            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test]

            sample_names = [
                f.removeprefix('x_test_').removesuffix('.npy') for f in os.listdir(samples_p) if f.startswith('x_test')
            ]

            # Define reference labels, i.e. all labels that occur in the train and test data and -1 for unknown
            y_reference = np.concatenate((y_train, y_test, np.array([-1,])))
            y_reference = np.unique(y_reference)

            # Instantiate the SOM classifier with the best parameters
            clf = SomClassifier(
                som_topology='planar',
                som_grid_type='rectangular',
                som_dimensions=(30, 30),
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
                verbosity=1,
            )

            print('# ### Starting fit ...')
            st_fit = time.time()
            clf.fit(X=x_train, y=y_train)
            et_fit = time.time()
            fit_time = et_fit - st_fit
            print(f'# ### Fit finished, time: {fit_time}\n')

            # Predict on all test data concatenated to one data matrix
            print('# ### Starting prediction ...')
            st_pred = time.time()
            y_pred = clf.predict(X=x_test)
            et_pred = time.time()
            pred_time = et_pred - st_pred
            print(f'# ### Prediction finished, time: {pred_time}\n')

            # Df with fit and pred time
            times_df = pd.DataFrame(
                data=np.array([fit_time, pred_time]).reshape((1, 2)),
                index=['sek'],
                columns=['fit', 'pred']
            )

            # Predict for each test sample separately
            print('# ### Starting sample-wise prediction ...')
            samples_y_pred = []
            samples_pred_times = []
            for x in samples_x_test:
                
                st = time.time()
                samples_y_pred.append(clf.predict(X=x))
                et = time.time()
                samples_pred_times.append(et - st)

            sw_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'], data=samples_pred_times)
            std = sw_times_df['pred_time'].std(axis=0)
            mean = sw_times_df['pred_time'].mean(axis=0)
            sw_times_df.loc['std'] = std
            sw_times_df.loc['mean'] = mean
            print(f'# ### Sample-wise prediction finished, avg time per sample: {mean}\n')

            # ### Evaluate the classifier's performance
            print('# ### Evaluating prediction, concatenated test data ...\n')
            res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                y_true=y_test,
                y_pred=y_pred,
                exclude_unknowns=True,
                unknown_label=-1,
                verbosity=1,
            )
            
            res_df_class_wise = prec_rec_f1_class_wise(
                y_true=y_test,
                y_pred=y_pred,
                exclude_unknowns=True,
                unknown_label=-1,
                reference_labels=y_reference,
                verbosity=1,
            )

            cf_df = confusion_matrix_df(
                y_true=y_test,
                y_pred=y_pred,
                verbosity=1,
            )

            print('# ### Evaluating prediction, sample-wise test data ...')
            (
                sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex, sw_val_counts_y_true_of_unknowns
            ) = prec_rec_f1_avg_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                sample_names=sample_names,
                exclude_unknowns=True,
                unknown_label=-1,
                verbosity=0,
            )

            (
                sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
            ) = prec_rec_f1_class_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                sample_names=sample_names,
                exclude_unknowns=True,
                unknown_label=-1,
                reference_labels=y_reference,
                verbosity=0,
            )

            sw_cf_df = confusion_matrix_df_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred,
                verbosity=0
            )

            # ### Save the results
            # Define and create dirs for saving
            save_p = os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')
            if not os.path.exists(save_p):
                os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

            clf.save(filepath=save_p)

            res = [times_df, res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
            filenames = [
                'times.csv', 'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                'res_df_class_wise.csv', 'confusion_matrix.csv'
            ]

            for r, f in zip(res, filenames):
                r.to_csv(os.path.join(save_p, 'samples_concat', f))

            sw_res = [
                sw_times_df,
                sw_prec, sw_rec, sw_f1,
                sw_prec_uex, sw_rec_uex, sw_f1_uex,
                sw_val_counts_y_true_of_unknowns,
                sw_prec_cw, sw_rec_cw, sw_f1_cw,
                sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
            ]

            sw_filenames = [
                'times.csv',
                'precision.csv', 'recall.csv', 'f1.csv',
                'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                'val_counts_y_true_of_unknowns.csv',
                'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv', 'f1_unkn_excl_class_wise.csv',
            ]

            for r, f in zip(sw_res, sw_filenames):
                r.to_csv(os.path.join(save_p, 'sample_wise', f))

            for cm, sn in zip(sw_cf_df, sample_names):
                cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))


def main_som_classifier_with_confidence_threshold():

    import os
    import time
    import numpy as np
    import pandas as pd
    from flowsrc.flowsom import SomClassifier
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # ### Set flags and important variables here #######################################################################

    # Confidence threshold
    confidence_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # confidence_thresholds = [0.0, 0.5, 1.0]  # Todo

    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    for data_set in data_sets:
        for trafo in preprocessing_trafos:

            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...')
                continue

            # Load the train and (asmple-wise) test data
            data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
            samples_p = os.path.join(data_p, 'sample_wise_test')

            x_test = np.load(os.path.join(data_p, 'x_test.npy'))
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            samples_x_test = [f for f in os.listdir(samples_p) if f.startswith('x_test')]
            samples_y_test = [f for f in os.listdir(samples_p) if f.startswith('y_test')]

            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test]

            sample_names = [
                f.removeprefix('x_test_').removesuffix('.npy') for f in os.listdir(samples_p) if f.startswith('x_test')
            ]

            # Load the previously trained SOM classifier
            clf = SomClassifier.load(
                filepath=os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')
            )

            # Define reference labels, i.e. all labels that occur in the train and test data and -1 for unknown
            y_reference = np.concatenate((clf.og_classes_, y_test, np.array([-1, ])))
            y_reference = np.unique(y_reference)

            pred_times = []

            for confidence_threshold in confidence_thresholds:

                # Set the confidence threshold
                clf.confidence_threshold = confidence_threshold

                # Predict on all test data concatenated to one data matrix
                print('# ### Starting prediction ...')
                st_pred = time.time()
                y_pred = clf.predict(X=x_test)
                et_pred = time.time()
                pred_time = et_pred - st_pred
                print(f'# ### Prediction finished, time: {pred_time}')

                pred_times.append(pred_time)

                # Predict on all test data concatenated to one data matrix
                print('# ### Starting prediction ...')
                st_pred = time.time()
                y_pred = clf.predict(X=x_test)
                et_pred = time.time()
                pred_time = et_pred - st_pred
                print(f'# ### Prediction finished, time: {pred_time}')

                # Predict for each test sample separately
                print('# ### Starting sample-wise prediction ...')
                samples_y_pred = []
                samples_pred_times = []
                for x in samples_x_test:
                    st = time.time()
                    samples_y_pred.append(clf.predict(X=x))
                    et = time.time()
                    samples_pred_times.append(et - st)

                sw_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'], data=samples_pred_times)
                std = sw_times_df['pred_time'].std(axis=0)
                mean = sw_times_df['pred_time'].mean(axis=0)
                sw_times_df.loc['std'] = std
                sw_times_df.loc['mean'] = mean
                print(f'# ### sample-wise prediction finished, avg time per sample: {mean}')

                # ### Evaluate the classifier's performance
                print('# ### Evaluating prediction, concatenated test data ...')
                res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                    y_true=y_test,
                    y_pred=y_pred,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    verbosity=1,
                )

                res_df_class_wise = prec_rec_f1_class_wise(
                    y_true=y_test,
                    y_pred=y_pred,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    reference_labels=y_reference,
                    verbosity=1,
                )

                cf_df = confusion_matrix_df(
                    y_true=y_test,
                    y_pred=y_pred,
                    verbosity=1,
                )

                print('# ### Evaluating prediction, sample-wise test data ...')
                (
                    sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex, sw_val_counts_y_true_of_unknowns
                ) = prec_rec_f1_avg_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred,
                    sample_names=sample_names,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    verbosity=0,
                )

                (
                    sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                ) = prec_rec_f1_class_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred,
                    sample_names=sample_names,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    reference_labels=y_reference,
                    verbosity=0,
                )

                sw_cf_df = confusion_matrix_df_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred,
                    verbosity=0
                )

                # ### Save the results
                # Define and create dirs for saving
                save_p = os.path.join(
                    os.getcwd(),
                    f'results_new/pred_eval_som_classifier/{data_set}/{trafo}/'
                    f'conf_thresh/conf_thresh_{confidence_threshold}'
                )

                if not os.path.exists(save_p):
                    os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                    os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                clf.save(filepath=save_p)

                res = [res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
                filenames = [
                    'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                    'res_df_class_wise.csv', 'confusion_matrix.csv'
                ]

                for r, f in zip(res, filenames):
                    r.to_csv(os.path.join(save_p, 'samples_concat', f))

                sw_res = [
                    sw_times_df,
                    sw_prec, sw_rec, sw_f1,
                    sw_prec_uex, sw_rec_uex, sw_f1_uex,
                    sw_val_counts_y_true_of_unknowns,
                    sw_prec_cw, sw_rec_cw, sw_f1_cw,
                    sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                ]

                sw_filenames = [
                    'times.csv',
                    'precision.csv', 'recall.csv', 'f1.csv',
                    'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                    'val_counts_y_true_of_unknowns.csv',
                    'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                    'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv',
                    'f1_unkn_excl_class_wise.csv',
                ]

                for r, f in zip(sw_res, sw_filenames):
                    r.to_csv(os.path.join(save_p, 'sample_wise', f))

                for cm, sn in zip(sw_cf_df, sample_names):
                    cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))


            # Df with pred times
            time_df = pd.DataFrame(
                data=pred_times,
                index=confidence_thresholds,
                columns=['pred time']
            )

            time_df.to_csv(
                os.path.join(
                    os.getcwd(),
                    f'results_new/pred_eval_som_classifier/{data_set}/{trafo}/conf_thresh/pred_times.csv'
                )
            )


def main_dgcytof():

    # Dgcytof workflow adapted from
    # https://github.com/lijcheng12/DGCyTOF/tree/main/Code_Study/DGCyTOF/CyTOF2
    # https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb

    import os
    import random
    import time
    import pickle
    import torch
    import numpy as np
    import pandas as pd
    from validation.dgcytof import DgcytofClassifier, predict_softmax
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # Define function for setting random seeds
    def set_random_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    failure_combinations = []
    failure_points = []
    error_types = []
    error_messages = []

    for data_set in data_sets:
        for trafo in preprocessing_trafos:

            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # Set random seeds
            set_random_seed()

            # Load the train and (asmple-wise) test data
            data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
            samples_p = os.path.join(data_p, 'sample_wise_test')

            x_train = np.load(os.path.join(data_p, 'x_train.npy'))
            y_train = np.load(os.path.join(data_p, 'y_train.npy'))

            x_test = np.load(os.path.join(data_p, 'x_test.npy'))
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            samples_x_test = [f for f in os.listdir(samples_p) if f.startswith('x_test')]
            samples_y_test = [f for f in os.listdir(samples_p) if f.startswith('y_test')]

            samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test]
            samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test]

            sample_names = [
                f[7:-4] for f in os.listdir(samples_p) if f.startswith('x_test')
            ]

            # Define reference labels, i.e. all labels that occur in the train and test data and -1 for unknown
            y_reference = np.concatenate((y_train, y_test, np.array([-1, ])))
            y_reference = np.unique(y_reference)

            # Instantiate the Dgcytof classifier with default parameters (but only one worker)
            clf = DgcytofClassifier(
                val_size=0.2,
                layer_sizes=(128, 64, 32),
                n_epochs=20,
                train_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
            )

            # ### Try to fit and predict with the Dgcytof classifier
            dgcytof_no_error = True
            try:
                print('# ### Starting fit ...')
                st_fit = time.time()
                clf.fit(X=x_train, y=y_train)
                et_fit = time.time()
                fit_time = et_fit - st_fit
                print(f'# ### Fit finished, time: {fit_time}\n')
            except Exception as e:
                print(f'# ### Fit failed. Exception:\n{e}')
                failure_combinations.append(f'{data_set}_{trafo}')
                failure_points.append('fit')
                error_messages.append(str(e))
                error_types.append(type(e).__name__)
                dgcytof_no_error = False

            # If fit was successful, try predict for concatenated data
            if dgcytof_no_error:
                try:
                    print('# ### Starting Dgcytof prediction ...')
                    st_pred = time.time()
                    y_pred_dgcytof = clf.predict(X=x_test)
                    et_pred = time.time()
                    pred_time = et_pred - st_pred
                    print(f'# ### Prediction Dgcytof finished, time: {pred_time}')

                    # Df with fit and pred time
                    times_df = pd.DataFrame(
                        data=np.array([fit_time, pred_time]).reshape((1, 2)),
                        index=['sek'],
                        columns=['fit', 'pred']
                    )

                except Exception as e:
                    failure_combinations.append(f'{data_set}_{trafo}')
                    failure_points.append('predict_concatenated')
                    error_messages.append(str(e))
                    error_types.append(type(e).__name__)
                    dgcytof_no_error = False

            # If fit and predict for concatenated data was successfull, try sample-wise predict and save results
            if dgcytof_no_error:
                print('# ### Starting Dgcytof sample-wise prediction ...')
                samples_y_pred = []
                samples_pred_times = []
                no_failure_bool = []
                for sn, x in zip(sample_names, samples_x_test):
                    try:
                        st = time.time()
                        samples_y_pred.append(clf.predict(X=x))
                        et = time.time()
                        samples_pred_times.append(et - st)
                        no_failure_bool.append(True)
                    except Exception as e:
                        samples_pred_times.append(np.nan)
                        failure_combinations.append(f'{data_set}_{trafo}')
                        failure_points.append(f'predict_{sn}')
                        error_messages.append(str(e))
                        error_types.append(type(e).__name__)
                        no_failure_bool.append(False)

                sw_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'], data=samples_pred_times)
                std = sw_times_df['pred_time'].std(axis=0)
                mean = sw_times_df['pred_time'].mean(axis=0)
                sw_times_df.loc['std'] = std
                sw_times_df.loc['mean'] = mean
                print(f'# ### Dgcytof Sample-wise prediction finished, avg time per sample: {mean}\n')

                # ### Evaluate the classifier's performance
                print('# ### Evaluating Dgcytof prediction, concatenated test data ...\n')
                res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                    y_true=y_test,
                    y_pred=y_pred_dgcytof,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    verbosity=1,
                )

                res_df_class_wise = prec_rec_f1_class_wise(
                    y_true=y_test,
                    y_pred=y_pred_dgcytof,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    reference_labels=y_reference,
                    verbosity=1,
                )

                cf_df = confusion_matrix_df(
                    y_true=y_test,
                    y_pred=y_pred_dgcytof,
                    verbosity=1,
                )

                print('# ### Evaluating Dgcytof prediction, sample-wise test data ...')
                sample_names_no_error = [sn for (sn, nf) in zip(sample_names, no_failure_bool) if nf]
                samples_y_test_no_error = [s for (s, nf) in zip(samples_y_test, no_failure_bool) if nf]
                (
                    sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex, sw_val_counts_y_true_of_unknowns
                ) = prec_rec_f1_avg_sample_wise(
                    y_trues=samples_y_test_no_error,
                    y_preds=samples_y_pred,
                    sample_names=sample_names_no_error,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    verbosity=0,
                )

                (
                    sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                ) = prec_rec_f1_class_sample_wise(
                    y_trues=samples_y_test_no_error,
                    y_preds=samples_y_pred,
                    sample_names=sample_names_no_error,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    reference_labels=y_reference,
                    verbosity=0,
                )

                sw_cf_df = confusion_matrix_df_sample_wise(
                    y_trues=samples_y_test_no_error,
                    y_preds=samples_y_pred,
                    verbosity=0
                )

                # ### Save the results
                # Define and create dirs for saving
                save_p = os.path.join(os.getcwd(), f'results_new/pred_eval_dgcytof/{data_set}/{trafo}')
                os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                clf.save(filepath=save_p)

                res = [times_df, res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
                filenames = [
                    'times.csv', 'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                    'res_df_class_wise.csv', 'confusion_matrix.csv'
                ]

                for r, f in zip(res, filenames):
                    r.to_csv(os.path.join(save_p, 'samples_concat', f))

                sw_res = [
                    sw_times_df,
                    sw_prec, sw_rec, sw_f1,
                    sw_prec_uex, sw_rec_uex, sw_f1_uex,
                    sw_val_counts_y_true_of_unknowns,
                    sw_prec_cw, sw_rec_cw, sw_f1_cw,
                    sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                ]

                sw_filenames = [
                    'times.csv',
                    'precision.csv', 'recall.csv', 'f1.csv',
                    'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                    'val_counts_y_true_of_unknowns.csv',
                    'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                    'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv',
                    'f1_unkn_excl_class_wise.csv',
                ]

                for r, f in zip(sw_res, sw_filenames):
                    r.to_csv(os.path.join(save_p, 'sample_wise', f))

                for cm, sn in zip(sw_cf_df, sample_names_no_error):
                    cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))


            # ### Also produce prediction just based on the softmax classifier (also if Dgcytof failed)
            clf_softmax = clf.softmax_classifier_
            label_mapping = clf.new_to_og_classes_dict_

            print('# ### Starting Softmax prediction ...')
            st_pred_softmax = time.time()
            y_pred_softmax = predict_softmax(X=x_test, softmax_classifier=clf_softmax, label_mapping=label_mapping)
            et_pred_softmax = time.time()
            pred_time_softmax = et_pred_softmax - st_pred_softmax
            print(f'# ### Prediction Softmax finished, time: {pred_time_softmax}')

            # Predict for each test sample separately
            print('# ### Starting Softmax sample-wise prediction ...')
            samples_y_pred_softmax = []
            samples_pred_times_softmax = []
            for x in samples_x_test:
                st_softmax = time.time()
                samples_y_pred_softmax.append(
                    predict_softmax(X=x, softmax_classifier=clf_softmax, label_mapping=label_mapping)
                )
                et_softmax = time.time()
                samples_pred_times_softmax.append(et_softmax - st_softmax)

            sw_times_df_softmax = pd.DataFrame(index=sample_names)
            sw_times_df_softmax['pred_time_softmax'] = samples_pred_times_softmax
            std_softmax = sw_times_df_softmax['pred_time_softmax'].std(axis=0)
            mean_softmax = sw_times_df_softmax['pred_time_softmax'].mean(axis=0)
            sw_times_df_softmax.loc['std'] = std_softmax
            sw_times_df_softmax.loc['mean'] = mean_softmax
            print(f'# ### Softmax Sample-wise prediction finished, avg time per sample: {mean_softmax}\n')

            # ### Evaluate the classifier's performance
            res_df_softmax = prec_rec_f1_avg(
                y_true=y_test,
                y_pred=y_pred_softmax,
                exclude_unknowns=False,
                verbosity=1,
            )

            res_df_class_wise_softmax = prec_rec_f1_class_wise(
                y_true=y_test,
                y_pred=y_pred_softmax,
                exclude_unknowns=False,
                reference_labels=y_reference,
                verbosity=1,
            )

            cf_df_softmax = confusion_matrix_df(
                y_true=y_test,
                y_pred=y_pred_softmax,
                verbosity=1,
            )

            print('# ### Evaluating prediction, sample-wise test data ...')
            sw_prec_softmax, sw_rec_softmax, sw_f1_softmax = prec_rec_f1_avg_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred_softmax,
                sample_names=sample_names,
                exclude_unknowns=False,
                verbosity=0,
            )

            sw_prec_cw_softmax, sw_rec_cw_softmax, sw_f1_cw_softmax = prec_rec_f1_class_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred_softmax,
                sample_names=sample_names,
                exclude_unknowns=False,
                reference_labels=y_reference,
                verbosity=0,
            )

            sw_cf_df_softmax = confusion_matrix_df_sample_wise(
                y_trues=samples_y_test,
                y_preds=samples_y_pred_softmax,
                verbosity=0
            )

            # ### Save the results
            # Define and create dirs for saving
            save_p_softmax = os.path.join(os.getcwd(), f'results_new/pred_eval_softmax/{data_set}/{trafo}')
            os.makedirs(os.path.join(save_p_softmax, 'samples_concat'), exist_ok=True)
            os.makedirs(os.path.join(save_p_softmax, 'sample_wise/confusion_matrices'), exist_ok=True)

            softmax_dict = {'clf': clf_softmax, 'label_mapping': label_mapping}
            with open(os.path.join(save_p_softmax, 'softmax_classifier.pkl'), 'wb') as f:
                pickle.dump(softmax_dict, f)

            res_softmax = [res_df_softmax, res_df_class_wise_softmax, cf_df_softmax]
            filenames_softmax = [
                'res_df.csv', 'res_df_class_wise.csv', 'confusion_matrix.csv'
            ]

            for r, f in zip(res_softmax, filenames_softmax):
                r.to_csv(os.path.join(save_p_softmax, 'samples_concat', f))

            # Softmax
            sw_res_softmax = [
                sw_times_df_softmax,
                sw_prec_softmax, sw_rec_softmax, sw_f1_softmax,
                sw_prec_cw_softmax, sw_rec_cw_softmax, sw_f1_cw_softmax,
            ]

            sw_filenames_softmax = [
                'times.csv',
                'precision.csv', 'recall.csv', 'f1.csv',
                'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
            ]

            for r, f in zip(sw_res_softmax, sw_filenames_softmax):
                r.to_csv(os.path.join(save_p_softmax, 'sample_wise', f))

            for cm, sn in zip(sw_cf_df_softmax, sample_names):
                cm.to_csv(os.path.join(save_p_softmax, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))

            error_df = pd.DataFrame()
            error_df['setting'] = failure_combinations
            error_df['failure_point'] = failure_points
            error_df['error_type'] = error_types
            error_df['error_message'] = error_messages
            if not error_df.empty:
                error_df['error_message'] = error_df['error_message'].str.replace("\n", " ", regex=True)

            error_df.to_csv(os.path.join(os.getcwd(), 'results_new/pred_eval_dgcytof/errors.csv'))


def main_gatemeclass():

    import os
    import time
    import numpy as np
    import pandas as pd

    from validation.gatemeclass import GateMeClassClassifier
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]

    marker_names_aml = [
        'FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'
    ]
    marker_names_flowcyt = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]
    marker_names_lymphoma_t1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    marker_names_lymphoma_t2 = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7', 'CD52_APC', 'CD11c_APC_A700',
        'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]
    marker_names = [
        marker_names_aml,
        marker_names_flowcyt,
        marker_names_lymphoma_t1, marker_names_lymphoma_t2,
        marker_names_lymphoma_t1, marker_names_lymphoma_t2
    ]

    allow_unknowns = [False, True]
    ####################################################################################################################

    failure_combinations = []
    failure_points = []
    error_types = []
    error_messages = []

    for data_set, mn in zip(data_sets, marker_names):
        for trafo in preprocessing_trafos:
            for au in allow_unknowns:

                print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

                # Channel-wise cutoffs were only defined for the aml data
                if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                    print('# ### Continue ...\n')
                    continue

                # Load the train and (asmple-wise) test data
                data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
                samples_p = os.path.join(data_p, 'sample_wise_test')

                x_train = np.load(os.path.join(data_p, 'x_train.npy'))
                y_train = np.load(os.path.join(data_p, 'y_train.npy'))

                x_test = np.load(os.path.join(data_p, 'x_test.npy'))
                y_test = np.load(os.path.join(data_p, 'y_test.npy'))

                samples_x_test = [f for f in os.listdir(samples_p) if f.startswith('x_test')]
                samples_y_test = [f for f in os.listdir(samples_p) if f.startswith('y_test')]

                samples_x_test = [np.load(os.path.join(samples_p, f)) for f in samples_x_test]
                samples_y_test = [np.load(os.path.join(samples_p, f)) for f in samples_y_test]

                sample_names = [
                    f.removeprefix('x_test_').removesuffix('.npy')
                    for f in os.listdir(samples_p) if f.startswith('x_test')
                ]

                # Define reference labels, i.e. all labels that occur in the train and test data and -1 for unknown
                y_reference = np.concatenate((y_train, y_test, np.array([-1,])))
                y_reference = np.unique(y_reference)

                # Instantiate the GMC classifier
                clf = GateMeClassClassifier(
                    marker_names=mn,
                    gmc_gmm_parameterization="V",
                    gmc_k=20,
                    gmc_sampling=0.1,
                    gmc_reject_option=au,
                    gmc_seed=1,
                    time_fit_pred=True,
                    verbosity=1
                )

                try:
                    print('# ### Starting fit ...')
                    st_fit = time.time()
                    clf.fit(X=x_train, y=y_train)
                    et_fit = time.time()
                    fit_time = et_fit - st_fit
                    print(f'# ### Fit finished, time: {fit_time}')
                except Exception as e:
                    failure_combinations.append(f'{data_set}_{trafo}_{"wunkn" if au else "nounkn"}')
                    failure_points.append('fit')
                    error_messages.append(str(e))
                    error_types.append(type(e).__name__)
                    continue

                try:
                    print('# ### Starting prediction ...')
                    st_pred = time.time()
                    y_pred = clf.predict(X=x_test)
                    et_pred = time.time()
                    pred_time = et_pred - st_pred
                    print(f'# ### Prediction finished, time: {pred_time}')
                except Exception as e:
                    failure_combinations.append(f'{data_set}_{trafo}_{"wunkn" if au else "nounkn"}')
                    failure_points.append('predict_concatenated')
                    error_messages.append(str(e))
                    error_types.append(type(e).__name__)
                    continue

                # Df with fit and pred time
                times_df = pd.DataFrame(
                    data=np.array([fit_time, pred_time]).reshape((1, 2)),
                    index=['sek'],
                    columns=['fit', 'pred']
                )

                print('# ### Starting sample-wise prediction ...')
                samples_y_pred = []
                samples_pred_times = []
                no_failure_bool = []
                for sn, x in zip(sample_names, samples_x_test):
                    try:
                        st = time.time()
                        samples_y_pred.append(clf.predict(X=x))
                        et = time.time()
                        samples_pred_times.append(et - st)
                        no_failure_bool.append(True)
                    except Exception as e:
                        samples_pred_times.append(np.nan)
                        failure_combinations.append(f'{data_set}_{trafo}_{"wunkn" if au else "nounkn"}')
                        failure_points.append(f'predict_{sn}')
                        error_messages.append(str(e))
                        error_types.append(type(e).__name__)
                        no_failure_bool.append(False)

                sw_times_df = pd.DataFrame(index=sample_names, columns=['pred_time'], data=samples_pred_times)
                std = sw_times_df['pred_time'].std(axis=0)
                mean = sw_times_df['pred_time'].mean(axis=0)
                sw_times_df.loc['std'] = std
                sw_times_df.loc['mean'] = mean
                print(f'# ### Sample-wise prediction finished, avg time per sample: {mean}\n')

                # ### Evaluate the classifier's performance
                # Subset sample names and test sample labels to cases where no failure occurred
                sample_names = [sn for (sn, nf) in zip(sample_names, no_failure_bool) if nf]
                samples_y_test = [s for (s, nf) in zip(samples_y_test, no_failure_bool) if nf]
                if au:
                    print('# ### Evaluating prediction, concatenated test data ...\n')
                    res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                        y_true=y_test,
                        y_pred=y_pred,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        verbosity=1,
                    )

                    res_df_class_wise = prec_rec_f1_class_wise(
                        y_true=y_test,
                        y_pred=y_pred,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        reference_labels=y_reference,
                        verbosity=1,
                    )

                    cf_df = confusion_matrix_df(
                        y_true=y_test,
                        y_pred=y_pred,
                        verbosity=1,
                    )

                    print('# ### Evaluating prediction, sample-wise test data ...')
                    (
                        sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex, sw_val_counts_y_true_of_unknowns
                    ) = prec_rec_f1_avg_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_pred,
                        sample_names=sample_names,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        verbosity=0,
                    )

                    (
                        sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                    ) = prec_rec_f1_class_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_pred,
                        sample_names=sample_names,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        reference_labels=y_reference,
                        verbosity=0,
                    )

                    sw_cf_df = confusion_matrix_df_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_pred,
                        verbosity=0
                    )

                    save_p = os.path.join(
                        os.getcwd(),
                        f'results_new/pred_eval_gatemeclass/{data_set}/{trafo}/with_unknowns'
                    )
                    os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                    os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                    clf.save(filepath=save_p)

                    res = [times_df, res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
                    filenames = [
                        'times.csv', 'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                        'res_df_class_wise.csv', 'confusion_matrix.csv'
                    ]

                    for r, f in zip(res, filenames):
                        r.to_csv(os.path.join(save_p, 'samples_concat', f))

                    sw_res = [
                        sw_times_df,
                        sw_prec, sw_rec, sw_f1,
                        sw_prec_uex, sw_rec_uex, sw_f1_uex,
                        sw_val_counts_y_true_of_unknowns,
                        sw_prec_cw, sw_rec_cw, sw_f1_cw,
                        sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                    ]

                    sw_filenames = [
                        'times.csv',
                        'precision.csv', 'recall.csv', 'f1.csv',
                        'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                        'val_counts_y_true_of_unknowns.csv',
                        'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                        'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv',
                        'f1_unkn_excl_class_wise.csv',
                    ]

                    for r, f in zip(sw_res, sw_filenames):
                        r.to_csv(os.path.join(save_p, 'sample_wise', f))

                    for cm, sn in zip(sw_cf_df, sample_names):
                        cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))

                else:
                    print('# ### Evaluating prediction, concatenated test data ...\n')
                    res_df = prec_rec_f1_avg(
                        y_true=y_test,
                        y_pred=y_pred,
                        exclude_unknowns=False,
                        verbosity=1,
                    )

                    res_df_class_wise = prec_rec_f1_class_wise(
                        y_true=y_test,
                        y_pred=y_pred,
                        exclude_unknowns=False,
                        reference_labels=y_reference,
                        verbosity=1,
                    )

                    cf_df = confusion_matrix_df(
                        y_true=y_test,
                        y_pred=y_pred,
                        verbosity=1,
                    )

                    print('# ### Evaluating prediction, sample-wise test data ...')
                    sw_prec, sw_rec, sw_f1 = prec_rec_f1_avg_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_pred,
                        sample_names=sample_names,
                        exclude_unknowns=False,
                        verbosity=0,
                    )

                    sw_prec_cw, sw_rec_cw, sw_f1_cw = prec_rec_f1_class_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_pred,
                        sample_names=sample_names,
                        exclude_unknowns=False,
                        reference_labels=y_reference,
                        verbosity=0,
                    )

                    sw_cf_df = confusion_matrix_df_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_pred,
                        verbosity=0
                    )

                    save_p = os.path.join(
                        os.getcwd(),
                        f'results_new/pred_eval_gatemeclass/{data_set}/{trafo}/without_unknowns'
                    )
                    os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                    os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                    clf.save(filepath=save_p)

                    res = [times_df, res_df, res_df_class_wise, cf_df]
                    filenames = [
                        'times.csv', 'res_df.csv',
                        'res_df_class_wise.csv', 'confusion_matrix.csv'
                    ]

                    for r, f in zip(res, filenames):
                        r.to_csv(os.path.join(save_p, 'samples_concat', f))

                    sw_res = [
                        sw_times_df,
                        sw_prec, sw_rec, sw_f1,
                        sw_prec_cw, sw_rec_cw, sw_f1_cw,
                    ]

                    sw_filenames = [
                        'times.csv',
                        'precision.csv', 'recall.csv', 'f1.csv',
                        'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                    ]

                    for r, f in zip(sw_res, sw_filenames):
                        r.to_csv(os.path.join(save_p, 'sample_wise', f))

                    for cm, sn in zip(sw_cf_df, sample_names):
                        cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))

                error_df = pd.DataFrame()
                error_df['setting'] = failure_combinations
                error_df['failure_point'] = failure_points
                error_df['error_type'] = error_types
                error_df['error_message'] = error_messages
                if not error_df.empty:
                    error_df['error_message'] = error_df['error_message'].str.replace("\n", " ", regex=True)

                error_df.to_csv(os.path.join(os.getcwd(), 'results_new/pred_eval_gatemeclass/errors.csv'))


def main_test_n_labeled_samples_som_classifier():
    import os
    import time
    import numpy as np
    import pandas as pd
    from flowsrc.flowsom import SomClassifier
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    for data_set in data_sets:
        for trafo in preprocessing_trafos:

            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # ### Load data
            data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
            # Load train samples
            samples_p_train = os.path.join(data_p, 'sample_wise_train')
            samples_x_train = [f for f in os.listdir(samples_p_train) if f.startswith('x_train')]
            samples_y_train = [f for f in os.listdir(samples_p_train) if f.startswith('y_train')]

            samples_x_train = [np.load(os.path.join(samples_p_train, f)) for f in samples_x_train]
            samples_y_train = [np.load(os.path.join(samples_p_train, f)) for f in samples_y_train]

            sample_names_train = [
                f.removeprefix('x_train_').removesuffix('.npy') for f in os.listdir(samples_p_train) if f.startswith("x_train")
            ]

            # Load test data concatenated
            x_test = np.load(os.path.join(data_p, 'x_test.npy'))
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            # Load test samples
            samples_p_test = os.path.join(data_p, 'sample_wise_test')
            samples_x_test = [f for f in os.listdir(samples_p_test) if f.startswith('x_test')]
            samples_y_test = [f for f in os.listdir(samples_p_test) if f.startswith('y_test')]

            samples_x_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_x_test]
            samples_y_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_y_test]

            sample_names_test = [
                f.removeprefix('x_test_').removesuffix('.npy') for f in os.listdir(samples_p_test) if f.startswith("x_test")
            ]

            for i in range(1, len(samples_x_train) + 1):
                print(f'# ### Case: Unlabeled: {len(samples_x_train) - i}, Labeled: {i}')

                # Concatenate data: unlabeled samples get label Nan = -999, this is recognized by the classifier
                x_train = np.vstack(samples_x_train)
                y_train = [y if j < i else np.full(y.shape, -999) for j, y in enumerate(samples_y_train)]
                y_train = np.hstack(y_train)

                # Define reference labels, i.e. all labels that occur in the train and test data and -1 for unknown
                y_reference = np.concatenate((y_train, y_test, np.array([-1, ])))
                y_reference = np.unique(y_reference)[1:]  # Exclude -999

                # Load the previously trained SOM classifier
                clf = SomClassifier.load(
                    filepath=os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')
                )

                # Annotate on the basis of the labeled data
                st_anno = time.time()
                clf.annotate_som(X=x_train, y=y_train)
                et_anno = time.time()
                anno_time = et_anno - st_anno
                print(f'# ### Annotation finished, time: {anno_time}')

                # Predict on all test data concatenated to one data matrix
                print('# ### Starting prediction ...')
                st_pred = time.time()
                y_pred = clf.predict(X=x_test)
                et_pred = time.time()
                pred_time = et_pred - st_pred
                print(f'# ### Prediction finished, time: {pred_time}')

                # Df with anno and pred time
                times_df = pd.DataFrame(
                    data=np.array([anno_time, pred_time]).reshape((1, 2)),
                    index=['sek'],
                    columns=['anno', 'pred']
                )

                # Predict for each test sample separately
                print('# ### Starting sample-wise prediction ...')
                samples_y_pred = []
                samples_pred_times = []
                for x in samples_x_test:
                    st = time.time()
                    samples_y_pred.append(clf.predict(X=x))
                    et = time.time()
                    samples_pred_times.append(et - st)

                sw_times_df = pd.DataFrame(index=sample_names_test, columns=['pred_time'], data=samples_pred_times)
                std = sw_times_df['pred_time'].std(axis=0)
                mean = sw_times_df['pred_time'].mean(axis=0)
                sw_times_df.loc['std'] = std
                sw_times_df.loc['mean'] = mean
                print(f'# ### Sample-wise prediction finished, avg time per sample: {mean}\n')

                # ### Evaluate the classifier's performance
                print('# ### Evaluating prediction, concatenated test data ...\n')
                res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                    y_true=y_test,
                    y_pred=y_pred,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    verbosity=1,
                )

                res_df_class_wise = prec_rec_f1_class_wise(
                    y_true=y_test,
                    y_pred=y_pred,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    reference_labels=y_reference,
                    verbosity=1,
                )

                cf_df = confusion_matrix_df(
                    y_true=y_test,
                    y_pred=y_pred,
                    verbosity=1,
                )

                print('# ### Evaluating prediction, sample-wise test data ...')
                (
                    sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex, sw_val_counts_y_true_of_unknowns
                ) = prec_rec_f1_avg_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred,
                    sample_names=sample_names_test,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    verbosity=0,
                )

                (
                    sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                ) = prec_rec_f1_class_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred,
                    sample_names=sample_names_test,
                    exclude_unknowns=True,
                    unknown_label=-1,
                    reference_labels=y_reference,
                    verbosity=0,
                )

                sw_cf_df = confusion_matrix_df_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred,
                    verbosity=0
                )

                # ### Save the results
                # Define and create dirs for saving
                save_p = os.path.join(
                    os.getcwd(),
                    f'results_new/n_labeled_samples_eval_som_classifier/{data_set}/{trafo}',
                    f'{str(len(samples_x_train) - i).zfill(2)}unlabeled_{str(i).zfill(2)}labeled'
                )
                os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                clf.save(filepath=save_p)

                res = [times_df, res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
                filenames = [
                    'times.csv', 'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                    'res_df_class_wise.csv', 'confusion_matrix.csv'
                ]

                for r, f in zip(res, filenames):
                    r.to_csv(os.path.join(save_p, 'samples_concat', f))

                sw_res = [
                    sw_times_df,
                    sw_prec, sw_rec, sw_f1,
                    sw_prec_uex, sw_rec_uex, sw_f1_uex,
                    sw_val_counts_y_true_of_unknowns,
                    sw_prec_cw, sw_rec_cw, sw_f1_cw,
                    sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                ]

                sw_filenames = [
                    'times.csv',
                    'precision.csv', 'recall.csv', 'f1.csv',
                    'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                    'val_counts_y_true_of_unknowns.csv',
                    'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                    'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv',
                    'f1_unkn_excl_class_wise.csv',
                ]

                for r, f in zip(sw_res, sw_filenames):
                    r.to_csv(os.path.join(save_p, 'sample_wise', f))

                for cm, sn in zip(sw_cf_df, sample_names_test):
                    cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))


def main_test_n_labeled_samples_dgcytof():
    import os
    import random
    import time
    import pickle
    import torch
    import numpy as np
    import pandas as pd
    from validation.dgcytof import DgcytofClassifier, predict_softmax
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # Define function for setting random seeds
    def set_random_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    # preprocessing_trafos = [
    #     'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    # ]
    preprocessing_trafos = ['arcsinhcofactor150', ]
    ####################################################################################################################

    failure_combinations = []
    failure_n_samples = []
    failure_points = []
    error_types = []
    error_messages = []

    for data_set in data_sets:
        for trafo in preprocessing_trafos:

            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # Set random seeds
            set_random_seed()

            # ### Load data
            data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
            # Load train samples
            samples_p_train = os.path.join(data_p, 'sample_wise_train')
            samples_x_train = [f for f in os.listdir(samples_p_train) if f.startswith('x_train')]
            samples_y_train = [f for f in os.listdir(samples_p_train) if f.startswith('y_train')]

            samples_x_train = [np.load(os.path.join(samples_p_train, f)) for f in samples_x_train]
            samples_y_train = [np.load(os.path.join(samples_p_train, f)) for f in samples_y_train]

            # Load test data concatenated
            x_test = np.load(os.path.join(data_p, 'x_test.npy'))
            y_test = np.load(os.path.join(data_p, 'y_test.npy'))

            # Load test samples
            samples_p_test = os.path.join(data_p, 'sample_wise_test')
            samples_x_test = [f for f in os.listdir(samples_p_test) if f.startswith('x_test')]
            samples_y_test = [f for f in os.listdir(samples_p_test) if f.startswith('y_test')]

            samples_x_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_x_test]
            samples_y_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_y_test]

            sample_names_test = [
                f[7:-4] for f in os.listdir(samples_p_test) if f.startswith("x_test")
            ]

            for i in range(1, len(samples_x_train) + 1):

                print(f'# ### Case: Labeled: {i}, Unlabeled: {len(samples_x_train) - i}')

                x_train = np.vstack(samples_x_train[:i])
                y_train = np.hstack(samples_y_train[:i])

                # Define reference labels, i.e. all labels that occur in the train and test data and -1 for unknown
                y_reference = np.concatenate((y_train, y_test, np.array([-1, ])))
                y_reference = np.unique(y_reference)

                # Instantiate the Dgcytof classifier with default parameters (but only one worker)
                clf = DgcytofClassifier(
                    val_size=0.2,
                    layer_sizes=(128, 64, 32),
                    n_epochs=20,
                    train_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
                )

                # ### Try to fit and predict with the Dgcytof classifier
                dgcytof_no_error = True
                try:
                    print('# ### Starting fit ...')
                    st_fit = time.time()
                    clf.fit(X=x_train, y=y_train)
                    et_fit = time.time()
                    fit_time = et_fit - st_fit
                    print(f'# ### Fit finished, time: {fit_time}\n')
                except Exception as e:
                    print(f'# ### Fit failed. Exception:\n{e}')
                    failure_combinations.append(f'{data_set}_{trafo}')
                    failure_n_samples.append(f'labeled_{i}_unlabeled_{len(samples_x_train) - i}')
                    failure_points.append('fit')
                    error_messages.append(str(e))
                    error_types.append(type(e).__name__)
                    dgcytof_no_error = False

                # If fit was successful, try predict for concatenated data
                if dgcytof_no_error:
                    try:
                        print('# ### Starting Dgcytof prediction ...')
                        st_pred = time.time()
                        y_pred_dgcytof = clf.predict(X=x_test)
                        et_pred = time.time()
                        pred_time = et_pred - st_pred
                        print(f'# ### Prediction Dgcytof finished, time: {pred_time}')

                        # Df with fit and pred time
                        times_df = pd.DataFrame(
                            data=np.array([fit_time, pred_time]).reshape((1, 2)),
                            index=['sek'],
                            columns=['fit', 'pred']
                        )

                    except Exception as e:
                        failure_combinations.append(f'{data_set}_{trafo}')
                        failure_n_samples.append(f'labeled_{i}_unlabeled_{len(samples_x_train) - i}')
                        failure_points.append('predict_concatenated')
                        error_messages.append(str(e))
                        error_types.append(type(e).__name__)
                        dgcytof_no_error = False

                # If fit and predict for concatenated data was successfull, try sample-wise predict and save results
                if dgcytof_no_error:
                    print('# ### Starting Dgcytof sample-wise prediction ...')
                    samples_y_pred = []
                    samples_pred_times = []
                    no_failure_bool = []
                    for sn, x in zip(sample_names_test, samples_x_test):
                        try:
                            st = time.time()
                            samples_y_pred.append(clf.predict(X=x))
                            et = time.time()
                            samples_pred_times.append(et - st)
                            no_failure_bool.append(True)
                        except Exception as e:
                            samples_pred_times.append(np.nan)
                            failure_combinations.append(f'{data_set}_{trafo}')
                            failure_n_samples.append(f'labeled_{i}_unlabeled_{len(samples_x_train) - i}')
                            failure_points.append(f'predict_{sn}')
                            error_messages.append(str(e))
                            error_types.append(type(e).__name__)
                            no_failure_bool.append(False)

                    sw_times_df = pd.DataFrame(index=sample_names_test, columns=['pred_time'], data=samples_pred_times)
                    std = sw_times_df['pred_time'].std(axis=0)
                    mean = sw_times_df['pred_time'].mean(axis=0)
                    sw_times_df.loc['std'] = std
                    sw_times_df.loc['mean'] = mean
                    print(f'# ### Dgcytof Sample-wise prediction finished, avg time per sample: {mean}\n')

                    # ### Evaluate the classifier's performance
                    print('# ### Evaluating Dgcytof prediction, concatenated test data ...\n')
                    res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                        y_true=y_test,
                        y_pred=y_pred_dgcytof,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        verbosity=1,
                    )

                    res_df_class_wise = prec_rec_f1_class_wise(
                        y_true=y_test,
                        y_pred=y_pred_dgcytof,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        reference_labels=y_reference,
                        verbosity=1,
                    )

                    cf_df = confusion_matrix_df(
                        y_true=y_test,
                        y_pred=y_pred_dgcytof,
                        verbosity=1,
                    )

                    print('# ### Evaluating Dgcytof prediction, sample-wise test data ...')
                    sample_names_no_error = [sn for (sn, nf) in zip(sample_names_test, no_failure_bool) if nf]
                    samples_y_test_no_error = [s for (s, nf) in zip(samples_y_test, no_failure_bool) if nf]
                    (
                        sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex, sw_val_counts_y_true_of_unknowns
                    ) = prec_rec_f1_avg_sample_wise(
                        y_trues=samples_y_test_no_error,
                        y_preds=samples_y_pred,
                        sample_names=sample_names_no_error,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        verbosity=0,
                    )

                    (
                        sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                    ) = prec_rec_f1_class_sample_wise(
                        y_trues=samples_y_test_no_error,
                        y_preds=samples_y_pred,
                        sample_names=sample_names_no_error,
                        exclude_unknowns=True,
                        unknown_label=-1,
                        reference_labels=y_reference,
                        verbosity=0,
                    )

                    sw_cf_df = confusion_matrix_df_sample_wise(
                        y_trues=samples_y_test_no_error,
                        y_preds=samples_y_pred,
                        verbosity=0
                    )

                    # ### Save the results
                    # Define and create dirs for saving
                    save_p = os.path.join(
                        os.getcwd(),
                        f'results_new/n_labeled_samples_eval_dgcytof/{data_set}/{trafo}',
                        f'{str(len(samples_x_train) - i).zfill(2)}unlabeled_{str(i).zfill(2)}labeled'
                    )
                    os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                    os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                    clf.save(filepath=save_p)

                    res = [times_df, res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
                    filenames = [
                        'times.csv', 'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                        'res_df_class_wise.csv', 'confusion_matrix.csv'
                    ]

                    for r, f in zip(res, filenames):
                        r.to_csv(os.path.join(save_p, 'samples_concat', f))

                    sw_res = [
                        sw_times_df,
                        sw_prec, sw_rec, sw_f1,
                        sw_prec_uex, sw_rec_uex, sw_f1_uex,
                        sw_val_counts_y_true_of_unknowns,
                        sw_prec_cw, sw_rec_cw, sw_f1_cw,
                        sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                    ]

                    sw_filenames = [
                        'times.csv',
                        'precision.csv', 'recall.csv', 'f1.csv',
                        'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                        'val_counts_y_true_of_unknowns.csv',
                        'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                        'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv',
                        'f1_unkn_excl_class_wise.csv',
                    ]

                    for r, f in zip(sw_res, sw_filenames):
                        r.to_csv(os.path.join(save_p, 'sample_wise', f))

                    for cm, sn in zip(sw_cf_df, sample_names_no_error):
                        cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))

                # ### Also produce prediction just based on the softmax classifier (also if Dgcytof failed)
                clf_softmax = clf.softmax_classifier_
                label_mapping = clf.new_to_og_classes_dict_

                print('# ### Starting Softmax prediction ...')
                st_pred_softmax = time.time()
                y_pred_softmax = predict_softmax(X=x_test, softmax_classifier=clf_softmax, label_mapping=label_mapping)
                et_pred_softmax = time.time()
                pred_time_softmax = et_pred_softmax - st_pred_softmax
                print(f'# ### Prediction Softmax finished, time: {pred_time_softmax}')

                # Predict for each test sample separately
                print('# ### Starting Softmax sample-wise prediction ...')
                samples_y_pred_softmax = []
                samples_pred_times_softmax = []
                for x in samples_x_test:
                    st_softmax = time.time()
                    samples_y_pred_softmax.append(
                        predict_softmax(X=x, softmax_classifier=clf_softmax, label_mapping=label_mapping)
                    )
                    et_softmax = time.time()
                    samples_pred_times_softmax.append(et_softmax - st_softmax)

                sw_times_df_softmax = pd.DataFrame(index=sample_names_test)
                sw_times_df_softmax['pred_time_softmax'] = samples_pred_times_softmax
                std_softmax = sw_times_df_softmax['pred_time_softmax'].std(axis=0)
                mean_softmax = sw_times_df_softmax['pred_time_softmax'].mean(axis=0)
                sw_times_df_softmax.loc['std'] = std_softmax
                sw_times_df_softmax.loc['mean'] = mean_softmax
                print(f'# ### Softmax Sample-wise prediction finished, avg time per sample: {mean_softmax}\n')

                # ### Evaluate the classifier's performance
                res_df_softmax = prec_rec_f1_avg(
                    y_true=y_test,
                    y_pred=y_pred_softmax,
                    exclude_unknowns=False,
                    verbosity=1,
                )

                res_df_class_wise_softmax = prec_rec_f1_class_wise(
                    y_true=y_test,
                    y_pred=y_pred_softmax,
                    exclude_unknowns=False,
                    reference_labels=y_reference,
                    verbosity=1,
                )

                cf_df_softmax = confusion_matrix_df(
                    y_true=y_test,
                    y_pred=y_pred_softmax,
                    verbosity=1,
                )

                print('# ### Evaluating prediction, sample-wise test data ...')
                sw_prec_softmax, sw_rec_softmax, sw_f1_softmax = prec_rec_f1_avg_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred_softmax,
                    sample_names=sample_names_test,
                    exclude_unknowns=False,
                    verbosity=0,
                )

                sw_prec_cw_softmax, sw_rec_cw_softmax, sw_f1_cw_softmax = prec_rec_f1_class_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred_softmax,
                    sample_names=sample_names_test,
                    exclude_unknowns=False,
                    reference_labels=y_reference,
                    verbosity=0,
                )

                sw_cf_df_softmax = confusion_matrix_df_sample_wise(
                    y_trues=samples_y_test,
                    y_preds=samples_y_pred_softmax,
                    verbosity=0
                )

                # ### Save the results
                # Define and create dirs for saving
                save_p_softmax = os.path.join(
                    os.getcwd(),
                    f'results_new/n_labeled_samples_eval_softmax/{data_set}/{trafo}',
                    f'{str(len(samples_x_train) - i).zfill(2)}unlabeled_{str(i).zfill(2)}labeled'
                )
                os.makedirs(os.path.join(save_p_softmax, 'samples_concat'), exist_ok=True)
                os.makedirs(os.path.join(save_p_softmax, 'sample_wise/confusion_matrices'), exist_ok=True)

                softmax_dict = {'clf': clf_softmax, 'label_mapping': label_mapping}
                with open(os.path.join(save_p_softmax, 'softmax_classifier.pkl'), 'wb') as f:
                    pickle.dump(softmax_dict, f)

                res_softmax = [res_df_softmax, res_df_class_wise_softmax, cf_df_softmax]
                filenames_softmax = [
                    'res_df.csv', 'res_df_class_wise.csv', 'confusion_matrix.csv'
                ]

                for r, f in zip(res_softmax, filenames_softmax):
                    r.to_csv(os.path.join(save_p_softmax, 'samples_concat', f))

                # Softmax
                sw_res_softmax = [
                    sw_times_df_softmax,
                    sw_prec_softmax, sw_rec_softmax, sw_f1_softmax,
                    sw_prec_cw_softmax, sw_rec_cw_softmax, sw_f1_cw_softmax,
                ]

                sw_filenames_softmax = [
                    'times.csv',
                    'precision.csv', 'recall.csv', 'f1.csv',
                    'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                ]

                for r, f in zip(sw_res_softmax, sw_filenames_softmax):
                    r.to_csv(os.path.join(save_p_softmax, 'sample_wise', f))

                for cm, sn in zip(sw_cf_df_softmax, sample_names_test):
                    cm.to_csv(
                        os.path.join(save_p_softmax, 'sample_wise/confusion_matrices', f'confusion_matrix_{sn}.csv'))

                error_df = pd.DataFrame()
                error_df['setting'] = failure_combinations
                error_df['n_samples'] = failure_n_samples
                error_df['failure_point'] = failure_points
                error_df['error_type'] = error_types
                error_df['error_message'] = error_messages
                if not error_df.empty:
                    error_df['error_message'] = error_df['error_message'].str.replace("\n", " ", regex=True)

                error_df.to_csv(os.path.join(os.getcwd(), 'results_new/n_labeled_samples_eval_dgcytof/errors.csv'))


def main_test_n_labeled_samples_gatemeclass():

    import os
    import time
    import numpy as np
    import pandas as pd

    from validation.gatemeclass import GateMeClassClassifier
    from validation.val_utils import (
        prec_rec_f1_avg, prec_rec_f1_class_wise, confusion_matrix_df,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise, confusion_matrix_df_sample_wise
    )

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    # preprocessing_trafos = [
    #     'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    # ]

    preprocessing_trafos = ['arcsinhcofactor150', ]

    marker_names_aml = [
        'FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'
    ]
    marker_names_flowcyt = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]
    marker_names_lymphoma_t1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    marker_names_lymphoma_t2 = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7', 'CD52_APC', 'CD11c_APC_A700',
        'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]
    marker_names = [
        marker_names_aml,
        marker_names_flowcyt,
        marker_names_lymphoma_t1, marker_names_lymphoma_t2,
        marker_names_lymphoma_t1, marker_names_lymphoma_t2
    ]

    allow_unknowns = [False, True]
    ####################################################################################################################

    failure_combinations = []
    failure_n_samples = []
    failure_points = []
    error_types = []
    error_messages = []

    for data_set, mn in zip(data_sets, marker_names):
        for trafo in preprocessing_trafos:
            for au in allow_unknowns:
                print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

                # Channel-wise cutoffs were only defined for the aml data
                if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                    print('# ### Continue ...\n')
                    continue

                # ### Load data
                data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/')
                # Load train samples
                samples_p_train = os.path.join(data_p, 'sample_wise_train')
                samples_x_train = [f for f in os.listdir(samples_p_train) if f.startswith('x_train')]
                samples_y_train = [f for f in os.listdir(samples_p_train) if f.startswith('y_train')]

                samples_x_train = [np.load(os.path.join(samples_p_train, f)) for f in samples_x_train]
                samples_y_train = [np.load(os.path.join(samples_p_train, f)) for f in samples_y_train]

                # Load test data concatenated
                x_test = np.load(os.path.join(data_p, 'x_test.npy'))
                y_test = np.load(os.path.join(data_p, 'y_test.npy'))

                # Load test samples
                samples_p_test = os.path.join(data_p, 'sample_wise_test')
                samples_x_test = [f for f in os.listdir(samples_p_test) if f.startswith('x_test')]
                samples_y_test = [f for f in os.listdir(samples_p_test) if f.startswith('y_test')]

                samples_x_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_x_test]
                samples_y_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_y_test]

                sample_names_test = [
                    f.removeprefix('x_test_').removesuffix('.npy') for f in os.listdir(samples_p_test) if
                    f.startswith("x_test")
                ]

                for i in range(1, len(samples_x_train) + 1):
                    print(f'# ### Case: Labeled: {i}, Unlabeled: {len(samples_x_train) - i}')

                    x_train = np.vstack(samples_x_train[:i])
                    y_train = np.hstack(samples_y_train[:i])

                    # Define reference labels, i.e. all labels that occur in the train and test data and -1=unknown
                    y_reference = np.concatenate((y_train, y_test, np.array([-1, ])))
                    y_reference = np.unique(y_reference)

                    # Instantiate the GMC classifier
                    clf = GateMeClassClassifier(
                        marker_names=mn,
                        gmc_gmm_parameterization="V",
                        gmc_k=20,
                        gmc_sampling=0.1,
                        gmc_reject_option=au,
                        gmc_seed=1,
                        time_fit_pred=True,
                        verbosity=1
                    )

                    try:
                        print('# ### Starting fit ...')
                        st_fit = time.time()
                        clf.fit(X=x_train, y=y_train)
                        et_fit = time.time()
                        fit_time = et_fit - st_fit
                        print(f'# ### Fit finished, time: {fit_time}')
                    except Exception as e:
                        failure_combinations.append(f'{data_set}_{trafo}_{"wunkn" if au else "nounkn"}')
                        failure_n_samples.append(f'labeled_{i}_unlabeled_{len(samples_x_train) - i}')
                        failure_points.append('fit')
                        error_messages.append(str(e))
                        error_types.append(type(e).__name__)
                        continue

                    try:
                        print('# ### Starting prediction ...')
                        st_pred = time.time()
                        y_pred = clf.predict(X=x_test)
                        et_pred = time.time()
                        pred_time = et_pred - st_pred
                        print(f'# ### Prediction finished, time: {pred_time}')
                    except Exception as e:
                        failure_combinations.append(f'{data_set}_{trafo}_{"wunkn" if au else "nounkn"}')
                        failure_n_samples.append(f'labeled_{i}_unlabeled_{len(samples_x_train) - i}')
                        failure_points.append('predict_concatenated')
                        error_messages.append(str(e))
                        error_types.append(type(e).__name__)
                        continue

                    # Df with fit and pred time
                    times_df = pd.DataFrame(
                        data=np.array([fit_time, pred_time]).reshape((1, 2)),
                        index=['sek'],
                        columns=['fit', 'pred']
                    )

                    print('# ### Starting sample-wise prediction ...')
                    samples_y_pred = []
                    samples_pred_times = []
                    no_failure_bool = []
                    for sn, x in zip(sample_names_test, samples_x_test):
                        try:
                            st = time.time()
                            samples_y_pred.append(clf.predict(X=x))
                            et = time.time()
                            samples_pred_times.append(et - st)
                            no_failure_bool.append(True)
                        except Exception as e:
                            samples_pred_times.append(np.nan)
                            failure_combinations.append(f'{data_set}_{trafo}_{"wunkn" if au else "nounkn"}')
                            failure_n_samples.append(f'labeled_{i}_unlabeled_{len(samples_x_train) - i}')
                            failure_points.append(f'predict_{sn}')
                            error_messages.append(str(e))
                            error_types.append(type(e).__name__)
                            no_failure_bool.append(False)

                    sw_times_df = pd.DataFrame(index=sample_names_test, columns=['pred_time'], data=samples_pred_times)
                    std = sw_times_df['pred_time'].std(axis=0)
                    mean = sw_times_df['pred_time'].mean(axis=0)
                    sw_times_df.loc['std'] = std
                    sw_times_df.loc['mean'] = mean
                    print(f'# ### Sample-wise prediction finished, avg time per sample: {mean}\n')

                    # ### Evaluate the classifier's performance
                    sample_names_no_error = [sn for (sn, nf) in zip(sample_names_test, no_failure_bool) if nf]
                    samples_y_test_no_error = [s for (s, nf) in zip(samples_y_test, no_failure_bool) if nf]
                    if au:
                        print('# ### Evaluating prediction, concatenated test data ...\n')
                        res_df, val_counts_y_true_of_unknowns = prec_rec_f1_avg(
                            y_true=y_test,
                            y_pred=y_pred,
                            exclude_unknowns=True,
                            unknown_label=-1,
                            verbosity=1,
                        )

                        res_df_class_wise = prec_rec_f1_class_wise(
                            y_true=y_test,
                            y_pred=y_pred,
                            exclude_unknowns=True,
                            unknown_label=-1,
                            reference_labels=y_reference,
                            verbosity=1,
                        )

                        cf_df = confusion_matrix_df(
                            y_true=y_test,
                            y_pred=y_pred,
                            verbosity=1,
                        )

                        print('# ### Evaluating prediction, sample-wise test data ...')
                        (
                            sw_prec, sw_rec, sw_f1, sw_prec_uex, sw_rec_uex, sw_f1_uex,
                            sw_val_counts_y_true_of_unknowns
                        ) = prec_rec_f1_avg_sample_wise(
                            y_trues=samples_y_test_no_error,
                            y_preds=samples_y_pred,
                            sample_names=sample_names_no_error,
                            exclude_unknowns=True,
                            unknown_label=-1,
                            verbosity=0,
                        )

                        (
                            sw_prec_cw, sw_rec_cw, sw_f1_cw, sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw,
                            sw_f1_wout_unkn_cw
                        ) = prec_rec_f1_class_sample_wise(
                            y_trues=samples_y_test_no_error,
                            y_preds=samples_y_pred,
                            sample_names=sample_names_no_error,
                            exclude_unknowns=True,
                            unknown_label=-1,
                            reference_labels=y_reference,
                            verbosity=0,
                        )

                        sw_cf_df = confusion_matrix_df_sample_wise(
                            y_trues=samples_y_test_no_error,
                            y_preds=samples_y_pred,
                            verbosity=0
                        )

                        save_p = os.path.join(
                            os.getcwd(),
                            f'results_new/n_labeled_samples_eval_gatemeclass/{data_set}/{trafo}/with_unknowns',
                            f'{str(len(samples_x_train) - i).zfill(2)}unlabeled_{str(i).zfill(2)}labeled'
                        )
                        os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                        os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                        clf.save(filepath=save_p)

                        res = [times_df, res_df, val_counts_y_true_of_unknowns, res_df_class_wise, cf_df]
                        filenames = [
                            'times.csv', 'res_df.csv', 'val_counts_y_true_of_unknowns.csv',
                            'res_df_class_wise.csv', 'confusion_matrix.csv'
                        ]

                        for r, f in zip(res, filenames):
                            r.to_csv(os.path.join(save_p, 'samples_concat', f))

                        sw_res = [
                            sw_times_df,
                            sw_prec, sw_rec, sw_f1,
                            sw_prec_uex, sw_rec_uex, sw_f1_uex,
                            sw_val_counts_y_true_of_unknowns,
                            sw_prec_cw, sw_rec_cw, sw_f1_cw,
                            sw_prec_wout_unkn_cw, sw_rec_wout_unkn_cw, sw_f1_wout_unkn_cw
                        ]

                        sw_filenames = [
                            'times.csv',
                            'precision.csv', 'recall.csv', 'f1.csv',
                            'precision_unkn_excl.csv', 'recall_unkn_excl.csv', 'f1_unkn_excl.csv',
                            'val_counts_y_true_of_unknowns.csv',
                            'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                            'precision_unkn_excl_class_wise.csv', 'recall_unkn_excl_class_wise.csv',
                            'f1_unkn_excl_class_wise.csv',
                        ]

                        for r, f in zip(sw_res, sw_filenames):
                            r.to_csv(os.path.join(save_p, 'sample_wise', f))

                        for cm, sn in zip(sw_cf_df, sample_names_no_error):
                            cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices',
                                                   f'confusion_matrix_{sn}.csv'))

                    else:
                        print('# ### Evaluating prediction, concatenated test data ...\n')
                        res_df = prec_rec_f1_avg(
                            y_true=y_test,
                            y_pred=y_pred,
                            exclude_unknowns=False,
                            verbosity=1,
                        )

                        res_df_class_wise = prec_rec_f1_class_wise(
                            y_true=y_test,
                            y_pred=y_pred,
                            exclude_unknowns=False,
                            reference_labels=y_reference,
                            verbosity=1,
                        )

                        cf_df = confusion_matrix_df(
                            y_true=y_test,
                            y_pred=y_pred,
                            verbosity=1,
                        )

                        print('# ### Evaluating prediction, sample-wise test data ...')
                        sw_prec, sw_rec, sw_f1 = prec_rec_f1_avg_sample_wise(
                            y_trues=samples_y_test_no_error,
                            y_preds=samples_y_pred,
                            sample_names=sample_names_no_error,
                            exclude_unknowns=False,
                            verbosity=0,
                        )

                        sw_prec_cw, sw_rec_cw, sw_f1_cw = prec_rec_f1_class_sample_wise(
                            y_trues=samples_y_test_no_error,
                            y_preds=samples_y_pred,
                            sample_names=sample_names_no_error,
                            exclude_unknowns=False,
                            reference_labels=y_reference,
                            verbosity=0,
                        )

                        sw_cf_df = confusion_matrix_df_sample_wise(
                            y_trues=samples_y_test_no_error,
                            y_preds=samples_y_pred,
                            verbosity=0
                        )

                        save_p = os.path.join(
                            os.getcwd(),
                            f'results_new/n_labeled_samples_eval_gatemeclass/{data_set}/{trafo}/without_unknowns',
                            f'{str(len(samples_x_train) - i).zfill(2)}unlabeled_{str(i).zfill(2)}labeled'
                        )
                        os.makedirs(os.path.join(save_p, 'samples_concat'), exist_ok=True)
                        os.makedirs(os.path.join(save_p, 'sample_wise/confusion_matrices'), exist_ok=True)

                        clf.save(filepath=save_p)

                        res = [times_df, res_df, res_df_class_wise, cf_df]
                        filenames = [
                            'times.csv', 'res_df.csv',
                            'res_df_class_wise.csv', 'confusion_matrix.csv'
                        ]

                        for r, f in zip(res, filenames):
                            r.to_csv(os.path.join(save_p, 'samples_concat', f))

                        sw_res = [
                            sw_times_df,
                            sw_prec, sw_rec, sw_f1,
                            sw_prec_cw, sw_rec_cw, sw_f1_cw,
                        ]

                        sw_filenames = [
                            'times.csv',
                            'precision.csv', 'recall.csv', 'f1.csv',
                            'precision_class_wise.csv', 'recall_class_wise.csv', 'f1_class_wise.csv',
                        ]

                        for r, f in zip(sw_res, sw_filenames):
                            r.to_csv(os.path.join(save_p, 'sample_wise', f))

                        for cm, sn in zip(sw_cf_df, sample_names_no_error):
                            cm.to_csv(os.path.join(save_p, 'sample_wise/confusion_matrices',
                                                   f'confusion_matrix_{sn}.csv'))

                    error_df = pd.DataFrame()
                    error_df['setting'] = failure_combinations
                    error_df['n_samples'] = failure_n_samples
                    error_df['failure_point'] = failure_points
                    error_df['error_type'] = error_types
                    error_df['error_message'] = error_messages
                    if not error_df.empty:
                        error_df['error_message'] = error_df['error_message'].str.replace("\n", " ", regex=True)

                    error_df.to_csv(
                        os.path.join(os.getcwd(), 'results_new/n_labeled_samples_eval_gatemeclass/errors.csv')
                    )


def main_view_parameter_tuning_results():

    import os
    import pandas as pd
    from flowsrc.flowsom import SomClassifier

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    uninteresting_cols = [
        'std_fit_time', 'std_score_time', 'param_gaussian_neighborhood_sigma', 'param_initialization',
        'param_learning_rate_decay', 'param_n_epochs', 'param_neighborhood', 'param_radius_cooling',
        'param_som_grid_type', 'param_som_topology', 'params', 'split0_test_score', 'std_test_score',
    ]

    res_dir = os.path.join(
            os.getcwd(),
            'results/param_tuning/custompreprocessing_methodlog10_trafo_w_cutoff_100_ds0.25_nocv0.32'
        )
    somc = SomClassifier.load(filepath=res_dir)
    res_df = pd.DataFrame(somc.grid_search_.cv_results_).sort_values(by='mean_test_score', ascending=False)
    print(
        'Results, log10 with cutoff 100:\n',
        res_df.loc[:, [False if col in uninteresting_cols else True for col in res_df.columns]].head(10)
    )
    print('Best parameters, log10 with cutoff 100:\n', somc.grid_search_.best_params_)


    res_dir = os.path.join(
            os.getcwd(),
            'results/param_tuning/custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise_ds0.25_nocv0.32'
        )
    somc = SomClassifier.load(filepath=res_dir)
    print(
        'Results, log10 with channel-wise cutoff:\n',
        res_df.loc[:, [False if col in uninteresting_cols else True for col in res_df.columns]].head(10)
    )
    print('Best parameters, log10 with channel-wise cutoff:\n', somc.grid_search_.best_params_)


def main_view_n_epochs_calibration_results():

    import os
    import pandas as pd
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_lineplot

    somc = SomClassifier.load(
        filepath=os.path.join(
            os.getcwd(),
            'results/param_tuning/n_epoch_calibration/custompreprocessing_methodlog10_trafo_w_cutoff_100_nods_nocv0.32'
        )
    )

    print('Best n_epochs, log10 with cutoff 100: ', somc.grid_search_.best_params_['n_epochs'])

    somc = SomClassifier.load(
        filepath=os.path.join(
            os.getcwd(),
            'results/param_tuning/n_epoch_calibration/custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise_nods_nocv0.32'
        )
    )

    print('Best n_epochs, log10 with channel-wise cutoff: ', somc.grid_search_.best_params_['n_epochs'])

    res_df = pd.DataFrame(somc.grid_search_.cv_results_)  # .sort_values(by='mean_test_score', ascending=False)





def main_som_plots():

    import os
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    from flowsrc.flowsom import SomClassifier
    from validation.plotting import (
        plot_som, plot_support, plot_som_pies, annotate_mosaic, plot_support_entropy_scatter, plot_umatrix
    )


    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    for data_set in data_sets:
        for trafo in preprocessing_trafos:
            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # Load SOM classifier
            som_c = SomClassifier.load(
                filepath=os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')
            )

            # Extract properties of the SOM for plotting
            support = som_c.class_counts_per_unit_.sum(axis=2).astype(int)

            # Define a consistent colormap
            num_classes = len(som_c.new_to_og_classes_dict_)
            palette = sns.color_palette('deep', num_classes)
            cmap = mcolors.ListedColormap(palette)

            def scientific_formatter(x, pos):
                """Format colorbar ticks in scientific notation."""
                return f"{x:.1e}"

            save_p = os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}/plots')
            os.makedirs(save_p, exist_ok=True)

            plot_som(
                labels_unit_wise=som_c.som_unit_labels_,
                label_mapping=som_c.new_to_og_classes_dict_,
                mask=None,
                cmap=cmap,
                prefer_seaborn_cmap=True,
                plot_legend=False,
                title='SOM',
                dpi=300,
                ax=None,
                **{
                    'square': True,
                    'xticklabels': True,
                    'yticklabels': True,
                    'linewidths': 1,
                    'linecolor': 'black',
                    'annot_kws': {'fontsize': 6}
                }
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_p, 'som.png'), dpi=300)
            plt.close('all')

            plot_som_pies(
                class_counts_per_unit=som_c.class_counts_per_unit_,
                som_dimensions=som_c.som_dimensions,
                label_mapping=som_c.new_to_og_classes_dict_,
                radius=1.0,
                cmap=cmap,
                prefer_seaborn_cmap=True,
                add_grid_labels=False,
                plot_legend=True,
                apply_tightlayout=False,
                title='SOM pies',
                ax=None,
                dpi=300,
            )
            plt.savefig(os.path.join(save_p, 'som_pies.png'), bbox_inches='tight', pad_inches=0.05, dpi=300)
            plt.close('all')

            plot_support(
                support=support,
                plot_activation_frequency=False,
                cmap='Blues',
                prefer_seaborn_cmap=True,
                plot_cbar=True,
                annotate=False,
                title='Support',
                dpi=300,
                ax=None,
                **{
                    'square': True,
                    'xticklabels': True,
                    'yticklabels': True,
                    'linewidths': 1,
                    'linecolor': 'black',
                    # 'cbar_kws': {'format': FuncFormatter(scientific_formatter)}
                }
            )
            plt.savefig(os.path.join(save_p, 'support.png'), dpi=300)
            plt.close('all')

            plot_umatrix(
                umatrix=som_c.som_.umatrix, cmap='magma', plot_cbar=True, title='U-matrix', ax=None,
                **{'interpolation': 'none', 'aspect': 'equal'}
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_p, 'u_matrix.png'), dpi=300)
            plt.close('all')

            # Plot support size vs entropy in a scatter plot
            plot_support_entropy_scatter(
                class_counts_per_unit=som_c.class_counts_per_unit_,
                label_mapping=som_c.new_to_og_classes_dict_,
                cmap=cmap,
                prefer_seaborn_cmap=True,
                plot_legend=True,
                title='Support vs. Entropy',
                ax=None,
                **{'s': 2.0}
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_p, 'support_vs_entropy.png'), dpi=300)
            plt.close('all')


def main_threshold_plots():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    from flowsrc.flowsom import SomClassifier
    from validation.plotting import plot_som, annotate_mosaic

    # ### Set flags and important variables here #######################################################################
    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    for data_set in data_sets:
        for trafo in preprocessing_trafos:
            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # Load SOM classifier
            som_c = SomClassifier.load(
                filepath=os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')
            )

            # Extract properties of the SOM for plotting
            som_grid_fractions = np.zeros_like(som_c.class_counts_per_unit_)
            n_events_per_unit = som_c.class_counts_per_unit_.sum(axis=2, keepdims=True)
            nonzero_bool = (n_events_per_unit != 0).squeeze(axis=2)
            som_grid_fractions[nonzero_bool, :] = (
                    som_c.class_counts_per_unit_[nonzero_bool, :] / n_events_per_unit[nonzero_bool, :]
            )
            max_fracs = som_grid_fractions.max(axis=2)

            # Define colormap
            num_classes = len(som_c.new_to_og_classes_dict_)
            palette = sns.color_palette('deep', num_classes)
            cmap = mcolors.ListedColormap(palette)

            fig = plt.figure(figsize=(10, 6), constrained_layout=True, dpi=300)
            axd = fig.subplot_mosaic(
                """
                ABC
                DEF
                """
            )

            # Confidence threshold
            confidence_thresholds = [0.1, 0.25, 0.5, 0.75, 0.95, 0.99]
            keys = ["A", "B", "C", "D", "E", "F"]

            for threshold, key in zip(confidence_thresholds, keys):

                plot_som(
                    labels_unit_wise=som_c.som_unit_labels_,
                    label_mapping=som_c.new_to_og_classes_dict_,
                    mask=(max_fracs < threshold),  # Mask where confidence is lower than threshold
                    cmap=cmap,
                    prefer_seaborn_cmap=True,
                    plot_legend=False,
                    title=f'Confidence threshold: {threshold}',
                    dpi=300,
                    ax=axd[key],
                    **{
                        'square': True,
                        'xticklabels': False,
                        'yticklabels': False,
                        'linewidths': 0.5,
                        'linecolor': 'black',
                        'annot_kws': {'fontsize': 3}
                    }
                )

            annotate_mosaic(fig=fig, axd=axd, fontsize=14)

            save_p = os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}/plots')
            os.makedirs(save_p, exist_ok=True)
            plt.savefig(os.path.join(save_p, 'thresholds.png'), dpi=300)
            plt.close('all')


def main_som_to_fcs():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from validation.plotting import plot_som_disks


    # ### Set flags and important variables here #######################################################################
    channels_aml = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    channels_flowcyt = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]
    channels_lt1 = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]
    channels_lt2 = [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7', 'CD52_APC', 'CD11c_APC_A700',
        'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
    ]

    channel_names_list = [channels_aml, channels_flowcyt, channels_lt1, channels_lt2, channels_lt1, channels_lt2]

    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2', 'lymphoma_tube1_binary', 'lymphoma_tube2_binary']

    preprocessing_trafos = [
        'arcsinhcofactor150', 'log10_w_cutoffcutoff100', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'
    ]
    ####################################################################################################################

    for data_set, channel_names_X in zip(data_sets, channel_names_list):
        for trafo in preprocessing_trafos:
            print(f'# ###### Dataset: {data_set}, Transformation: {trafo} ###### #')

            # Channel-wise cutoffs were only defined for the aml data
            if data_set != 'aml' and trafo == 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise':
                print('# ### Continue ...\n')
                continue

            # Load the sample-wise train data
            data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/sample_wise_train')

            samples_x_train = [f for f in os.listdir(data_p) if f.startswith('x_train')]
            samples_x_train = [np.load(os.path.join(data_p, f)) for f in samples_x_train]

            sample_names = [
                f.removeprefix('x_train_').removesuffix('.npy') for f in os.listdir(data_p) if
                f.startswith('x_train')
            ]

            sample_ids = [i for i in range(len(sample_names))]

            # Load the original data
            samples_x_raw_train = []
            sample_names_mapping_df = pd.read_csv(
                os.path.join(
                    os.getcwd(),
                    'input/np_files',
                    data_set,
                    trafo,
                    f'sample_wise_train/sample_names_mapping_train.csv'
                )
            )

            if data_set == 'aml':
                raw_data_p = os.path.join(os.getcwd(), f'input/raw/aml/')
            elif data_set == 'flowcyt':
                raw_data_p = os.path.join(os.getcwd(), f'input/raw/flowcyt/data_original')
            elif data_set == 'lymphoma_tube1' or data_set == 'lymphoma_tube1_binary':
                raw_data_p = os.path.join(
                    os.getcwd(), f'input/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels'
                )
            elif data_set == 'lymphoma_tube2' or data_set == 'lymphoma_tube2_binary':
                raw_data_p = os.path.join(
                    os.getcwd(), f'input/raw/lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels'
                )
            else:
                raise ValueError(f'{data_set} does not exist.')

            for sn in sample_names:
                idx = sample_names_mapping_df[sample_names_mapping_df['new_sample_name'] == sn + '.npy'].index[0]
                og_sample_name = sample_names_mapping_df.at[idx, 'og_sample_name']

                fdm = FlowDataManager(
                    data_file_names=[og_sample_name, ],
                    data_file_type=None,  # file type is inferred from file ending, .fcs or .csv
                    data_file_path=raw_data_p,
                    save_path=None,
                    memory_saving=False,
                    verbosity=1,
                )

                fdm.load_data_files_to_anndata()

                fcs_df = fdm.anndata_list_[0].to_df()

                samples_x_raw_train.append(fcs_df)

            # Load SOM classifier
            save_p = os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')

            som_c = SomClassifier.load(filepath=save_p)

            # Annotate and save to fcs
            fcs_df = som_c.export_fcs(
                X=samples_x_train,
                channel_names_X=channel_names_X,
                X_raw=samples_x_raw_train,
                channel_names_X_raw=None,  # Use existing column names of df
                keep_X=True,
                val_range=(0.0, 2**20),
                keep_unscaled=True,
                sample_ids=sample_ids,
                compute_umap=True,
                umap_kwargs=None,
                save_mode='fcs',
                fcs_metadata_dict=None,
                save_path=save_p,
                filename='som.fcs',
            )

            fcs_df.to_csv(os.path.join(save_p, f'som.csv'))
            # print(fcs_df)
            # print(fcs_df.columns)

            # Define colormap
            num_classes = len(som_c.new_to_og_classes_dict_)
            palette = sns.color_palette('deep', num_classes)
            cmap = mcolors.ListedColormap(palette)

            labels = fcs_df['label' if data_set == 'flowcyt' else 'population'].to_numpy()
            if data_set in ['lymphoma_tube1_binary', 'lymphoma_tube2_binary']:
                mapping = {1: 0, 9: 0, 7: 1, 10: 1}
                labels = np.vectorize(mapping.get)(labels)

            plot_som_disks(
                x_vals=fcs_df['bmu1_plot'].to_numpy(),
                y_vals=fcs_df['bmu2_plot'].to_numpy(),
                scatter_kwargs={'s': 0.5},
                labels=labels,
                cmap=cmap,
                plot_legend=True,
                circle_x_y_r=fcs_df[['bmu1', 'bmu2', 'radius']].drop_duplicates(inplace=False).to_numpy(),
                title='SOM disks',
                dpi=300,
                aspect_ratio='auto',
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_p, 'plots/som_disks.png'))
            plt.close('all')


def main_view_n_labeled_samples_results():

    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    # ### Set Pandas print options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Set flags ####################################################################################################
    # Dataset
    # data_set = 'aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube1_binary', 'lymphoma_tube2', 'lymphoma_tube2_binary'
    data_set = 'aml'

    # Transformation
    trafo = 'arcsinhcofactor150'
    # trafo = 'log10_w_cutoffcutoff100'
    # trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'

    class_wise = True  # Whether to load class-wise results
    unkn_excl  = False  # Whether to load results where unknowns are excluded from evaluation

    # Evaluation metric
    eval_metric = 'f1'  # 'prec', 'rec', 'f1', 'unknown_count'

    # Methods
    method = 'softmax'  # 'som_classifier', 'dgcytof', 'softmax', 'gatemeclass'

    gmc_mode = 'without_unknowns'  # 'without_unknowns', 'with_unknowns'

    ####################################################################################################################

    cw = '_class_wise' if class_wise else ''
    wu = '_unkn_excl' if unkn_excl else ''

    res_p = os.path.join(
        os.getcwd(), 'results_new', f'n_labeled_samples_eval_{method}', data_set, trafo, gmc_mode if method == 'gatemeclass' else '')

    n_labeled = []
    n_unlabeled = []
    res_dfs = []
    for d in os.listdir(res_p):

        nunl_nl = list(map(int, re.findall(r'\d+', d)))    # [n unlabeled, n labeled]

        if nunl_nl == [] or nunl_nl[1] == 0:
            continue

        n_unlabeled.append(nunl_nl[0])
        n_labeled.append(nunl_nl[1])

        if not eval_metric == 'unknown_count':
            dummy_res_df = pd.read_csv(
                os.path.join(res_p, d, 'sample_wise', f'{eval_metric}{wu}{cw}.csv'),
                index_col=0
            )
            res_dfs.append(dummy_res_df.loc['mean'])
        else:
            res_dfs.append(
                pd.read_csv(os.path.join(res_p, d, 'samples_concat/val_counts_y_true_of_unknowns.csv'), index_col=0)
            )

    res_df = pd.concat(res_dfs, axis=1).T.reset_index(drop=True)
    res_df['n_unlabeled'] = n_unlabeled
    res_df['n_labeled'] = n_labeled

    print(res_df)

    save_p = os.path.join(res_p, 'summary')
    os.makedirs(save_p, exist_ok=True)
    res_df.to_csv(os.path.join(save_p, f'{eval_metric}{wu}{cw}.csv'))

    x_data = res_df['n_labeled'].to_numpy().astype(int)
    y_data = res_df.drop(columns=['n_labeled', 'n_unlabeled'])

    cmap = plt.get_cmap('tab10')
    num_lines = len(y_data.columns)
    colors = [cmap(i / num_lines) for i in range(num_lines)]

    fig, ax = plt.subplots(dpi=300)

    for i, col in enumerate(y_data.columns):
        if col == '-1':
            continue
        ax.plot(x_data, y_data[col].to_numpy(), label=col, color=colors[i], marker='o', ms=2, linestyle='-')

    if class_wise:
        if '-1' in y_data.columns:
            y_data.drop(columns=['-1'], inplace=True)

        ax.plot(x_data, y_data.to_numpy().mean(axis=1), label='macro F1', marker='x', ms=3, color='red')

    # Customizations
    ax.set_xlabel('# labeled train samples')
    ax.set_ylabel(eval_metric)
    ax.set_title(f'{eval_metric}{wu}{cw}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_p, f'{eval_metric}{wu}{cw}.png'))


def main_plot_n_labeled_samples():
    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    # ### Set Pandas print options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Set flags ####################################################################################################
    # Dataset
    # data_set = 'aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube1_binary', 'lymphoma_tube2', 'lymphoma_tube2_binary'
    data_set = 'aml'

    # Transformation
    trafo = 'arcsinhcofactor150'
    # trafo = 'log10_w_cutoffcutoff100'
    # trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'

    # Evaluation metric
    eval_metric = 'f1'  # 'prec', 'rec', 'f1'

    # Methods
    methods = ['som_classifier', 'dgcytof', 'softmax', 'gatemeclass']

    ####################################################################################################################

    res_dfs_meta = []
    for method in methods:

        res_p = os.path.join(
            os.getcwd(),
            'results_new',
            f'n_labeled_samples_eval_{method}',
            data_set,
            trafo,
            'without_unknowns' if method == 'gatemeclass' else ''
        )

        n_labeled = []
        n_unlabeled = []
        res_dfs = []

        for d in os.listdir(res_p):

            nunl_nl = list(map(int, re.findall(r'\d+', d)))  # [n unlabeled, n labeled]

            if nunl_nl == [] or nunl_nl[1] == 0:
                continue

            n_unlabeled.append(nunl_nl[0])
            n_labeled.append(nunl_nl[1])


            dummy_res_df = pd.read_csv(
                os.path.join(res_p, d, 'sample_wise', f'{eval_metric}_class_wise.csv'),
                index_col=0
            )
            res_dfs.append(dummy_res_df.loc['mean'])

        res_df = pd.concat(res_dfs, axis=1).T.reset_index(drop=True)
        res_df['n_unlabeled'] = n_unlabeled
        res_df['n_labeled'] = n_labeled

        res_dfs_meta.append(res_df)


    min_len = min([df.shape[0] for df in res_dfs_meta]) + 6

    fig, ax = plt.subplots(dpi=300)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / len(res_dfs_meta)) for i in range(len(res_dfs_meta))]

    for i, (method, df) in enumerate(zip(methods, res_dfs_meta)):
        x_data = df['n_labeled'].to_numpy().astype(int)[0:min_len]
        drop_cols = ['n_labeled', 'n_unlabeled', '-1'] if '-1' in df.columns else ['n_labeled', 'n_unlabeled']
        y_data = df.drop(columns=drop_cols).to_numpy().mean(axis=1)[0:min_len]

        ax.plot(x_data, y_data, label=method, color=colors[i], marker='o', ms=2, linestyle='-')

    ax.set_xlabel('# labeled train samples')
    ax.set_ylabel(f'macro {eval_metric}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(), f'results_new/plots/macro_{eval_metric}_n_labeled_samples.png'))


def main_view_prediction_evaluation_results():

    import os
    import pandas as pd
    import numpy as np

    # ### Set Pandas print options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Set flags ####################################################################################################
    # Dataset
    # data_set = 'aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube1_binary', 'lymphoma_tube2', 'lymphoma_tube2_binary'
    data_set = 'aml'

    # Transformation
    # trafo = 'arcsinhcofactor150'
    # trafo = 'log10_w_cutoffcutoff100'
    trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'

    class_wise = True  # Whether to load class-wise results
    unkn_excl = False  # Whether to load results where unknowns are excluded from evaluation
    print_cf_matrices = False  # Whether to load and print the confusion matrices

    # Evaluation metric
    eval_metric = 'f1'  # 'prec', 'rec', 'f1', 'unknown_count'

    ####################################################################################################################

    cw = '_class_wise' if class_wise else ''


    # ### Load the evaluation results
    base_p = os.path.join(os.getcwd(), f'results_new/')
    res_dfs = []
    methods = []
    for method in ['som_classifier', 'dgcytof', 'softmax', 'gatemeclass']:

        unknown_modes = ['without_unknowns', 'with_unknowns'] if method == 'gatemeclass' else ['']

        for unknown_mode in unknown_modes:

            if eval_metric != 'unknown_count':

                som_dg = method in ['som_classifier', 'dgcytof']
                gmc_w_unkn = (method == 'gatemeclass' and unknown_mode == 'with_unknowns')
                wu = '_unkn_excl' if unkn_excl and (som_dg or gmc_w_unkn) else ''

                try:
                    res_dfs.append(
                        pd.read_csv(
                            os.path.join(
                                base_p,
                                f'pred_eval_{method}',
                                data_set,
                                trafo,
                                unknown_mode,
                                'sample_wise',
                                f'{eval_metric}{cw}{wu}.csv'
                            ),
                            index_col=0
                        )
                    )
                    methods.append(method + (f'_{unknown_mode}' if unknown_mode else ''))

                except FileNotFoundError:
                    print(
                        f'# ### For data set "{data_set}" and data transformation "{trafo}" '
                        f'the method "{method}" could not be run without error.'
                    )

            else:
                try:
                    res_dfs.append(
                        pd.read_csv(
                            os.path.join(
                                base_p,
                                f'pred_eval_{method}',
                                data_set,
                                trafo,
                                unknown_mode,
                                'samples_concat/val_counts_y_true_of_unknowns.csv'
                            ),
                            index_col=0
                        )
                    )
                    methods.append(method + (f'_{unknown_mode}' if unknown_mode else ''))

                except FileNotFoundError:
                    print(
                        f'# ### For data set "{data_set}" and data transformation "{trafo}" '
                        f'the method "{method + (f'_{unknown_mode}' if unknown_mode else '')}" '
                        f'did not predict any events as "unknown".'
                    )

    # ### Concatenate into one df
    if eval_metric != 'unknown_count':
        res_df = pd.concat(
            [df.loc[['mean']].assign(Method=name) for name, df in zip(methods, res_dfs)]
        ).set_index('Method')

    else:
        methods = [m for (i, m) in enumerate(methods) if not res_dfs[i].empty]
        res_dfs = [df for df in res_dfs if not df.empty]

        res_df = pd.concat(res_dfs, axis=1).T
        res_df.index = methods
        res_df.sort_index(axis=1, inplace=True)

    # print(res_df)

    if class_wise and not unkn_excl:
        res_df_macro = res_df.drop(columns=['-1'])
        res_df_macro['macro_f1'] = res_df_macro.mean(axis=1)
        print(res_df_macro)


    # Load and print the confusion matrices
    if print_cf_matrices:

        for method in ['som_classifier', 'dgcytof', 'softmax', 'gatemeclass']:

            unknown_modes = ['wout_unkn', 'w_unkn'] if method == 'gatemeclass' else ['']

            for unknown_mode in unknown_modes:

                try:

                    cm = pd.read_csv(
                        os.path.join(base_p, f'pred_eval_{method}', data_set, trafo, unknown_mode, 'samples_concat',
                                     'cf_df.csv'),
                        index_col=0
                    )

                    print(f'# ### Confusion matrix {method + f'_{unknown_mode}' if unknown_mode else ''}:\n{cm}\n')

                except FileNotFoundError:
                    print(
                        f'# ### For data set "{data_set}" and data transformation "{trafo}" '
                        f'the method "{method}" could not be run without error.'
                    )




def main_plotting():

    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as mcolors
    import numpy as np
    import scanpy as sc
    import pandas as pd
    import matplotlib.image as mpimg
    import matplotlib.cm as cm

    from matplotlib.ticker import FuncFormatter
    from flowsrc.flowsom import SomClassifier
    from validation.plotting import (
        plot_som, plot_support, plot_som_pies, annotate_mosaic, plot_support_entropy_scatter, plot_umatrix
    )

    save_p = os.path.join(os.getcwd(), 'results/plots')
    if not os.path.exists(save_p):
        os.makedirs(save_p)

    plot_nr = '2'  # '1', '2', '3', '4'

    if plot_nr == '1':
        # Load SOM classifier
        som_c = SomClassifier.load(
            os.path.join(os.getcwd(), 'results/pred_eval_som_classifier/our_data/arcsinhcofactor150/som_classifier.pkl')
        )

        # Extract properties of the SOM for plotting
        som_grid_fractions = np.zeros_like(som_c.class_counts_per_unit_)
        n_events_per_unit = som_c.class_counts_per_unit_.sum(axis=2, keepdims=True)
        nonzero_bool = (n_events_per_unit != 0).squeeze(axis=2)
        som_grid_fractions[nonzero_bool, :] = (
                som_c.class_counts_per_unit_[nonzero_bool, :] / n_events_per_unit[nonzero_bool, :]
        )
        max_fracs = som_grid_fractions.max(axis=2)

        support = som_c.class_counts_per_unit_.sum(axis=2).astype(int)

        # Define a consistent colormap
        num_classes = len(som_c.new_to_og_classes_dict_)
        palette = sns.color_palette('deep', num_classes)
        cmap = mcolors.ListedColormap(palette)

        def scientific_formatter(x, pos):
            """Format colorbar ticks in scientific notation."""
            return f"{x:.1e}"

        fig = plt.figure(figsize=(10, 6), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            """
            ABC
            DEF
            """
        )

        plot_som(
            labels_unit_wise=som_c.som_unit_labels_,
            label_mapping=som_c.new_to_og_classes_dict_,
            confidence_mask=None,
            cmap=cmap,
            prefer_seaborn_cmap=True,
            plot_legend=False,
            title='SOM',
            dpi=300,
            ax=axd['A'],
            **{
                'square': True,
                'xticklabels': True,
                'yticklabels': True,
                'linewidths': 1,
                'linecolor': 'black',
            }
        )

        plot_support(
            support=support,
            plot_activation_frequency=False,
            cmap='Blues',
            prefer_seaborn_cmap=True,
            plot_cbar=True,
            annotate=False,
            title='Support',
            dpi=300,
            ax=axd['B'],
            **{
                'square': True,
                'xticklabels': True,
                'yticklabels': True,
                'linewidths': 1,
                'linecolor': 'black',
                # 'cbar_kws': {'format': FuncFormatter(scientific_formatter)}
            }
        )

        plot_som_pies(
            class_counts_per_unit=som_c.class_counts_per_unit_,
            som_dimensions=som_c.som_dimensions,
            label_mapping=som_c.new_to_og_classes_dict_,
            cmap=cmap,
            prefer_seaborn_cmap=True,
            add_grid_labels=False,
            plot_legend=True,
            apply_tightlayout=True,
            title='SOM pies',
            ax=axd['C'],
            dpi=300,
        )

        plot_som(
            labels_unit_wise=som_c.som_unit_labels_,
            label_mapping=som_c.new_to_og_classes_dict_,
            confidence_mask=(max_fracs <= 0.5),
            cmap=cmap,
            prefer_seaborn_cmap=True,
            plot_legend=False,
            title='Confidence threshold: 0.5',
            dpi=300,
            ax=axd['D'],
            **{
                'square': True,
                'xticklabels': False,
                'yticklabels': False,
                'linewidths': 1,
                'linecolor': 'black',
            }
        )

        plot_som(
            labels_unit_wise=som_c.som_unit_labels_,
            label_mapping=som_c.new_to_og_classes_dict_,
            confidence_mask=(max_fracs <= 0.75),
            cmap=cmap,
            prefer_seaborn_cmap=True,
            plot_legend=False,
            title='Confidence threshold: 0.75',
            dpi=300,
            ax=axd['E'],
            **{
                'square': True,
                'xticklabels': False,
                'yticklabels': False,
                'linewidths': 1,
                'linecolor': 'black',
            }
        )

        plot_som(
            labels_unit_wise=som_c.som_unit_labels_,
            label_mapping=som_c.new_to_og_classes_dict_,
            confidence_mask=(max_fracs <= 0.95),
            cmap=cmap,
            prefer_seaborn_cmap=True,
            plot_legend=False,
            title='Confidence threshold: 0.95',
            dpi=300,
            ax=axd['F'],
            **{
                'square': True,
                'xticklabels': False,
                'yticklabels': False,
                'linewidths': 1,
                'linecolor': 'black',
            }
        )

        annotate_mosaic(fig=fig, axd=axd, fontsize=14)

        plt.savefig(os.path.join(save_p, 'plot_01.png'), dpi=300)
        plt.close('all')

    elif plot_nr == '2':

        # Load SOM classifier
        som_c = SomClassifier.load(
            os.path.join(os.getcwd(), 'results/pred_eval_som_classifier/our_data/arcsinhcofactor150/som_classifier.pkl')
        )

        # Define a consistent colormap
        num_classes = len(som_c.new_to_og_classes_dict_)
        palette = sns.color_palette('deep', num_classes)
        cmap = mcolors.ListedColormap(palette)

        # cache_dir = save_p
        # cmap_u = plt.get_cmap('OrRd')
        # som_c.som_.view_umatrix(colormap=cmap_u, colorbar=False, bestmatches=True)

        # plt.savefig(os.path.join(cache_dir, 'umap.png'), bbox_inches='tight', pad_inches=0.05, dpi=300)
        # plt.close('all')

        fig = plt.figure(figsize=(6, 3), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            """
            AB
            """
        )

        # Plot U-matrix
        # img = mpimg.imread(os.path.join(cache_dir, 'umap.png'))
        # norm = mcolors.Normalize(vmin=np.min(som_c.som_.umatrix), vmax=np.max(som_c.som_.umatrix))
        # cmap_u = cm.ScalarMappable(norm=norm, cmap=cmap_u)
        # cbar = plt.colorbar(cmap_u, ax=axd['A'], orientation='horizontal', shrink=0.5)
        # cbar.set_label('U-matrix values')
        # axd['A'].imshow(img)
        # axd['A'].axis('off')
        # axd['A'].set_title('U-matrix')
        # os.remove(os.path.join(cache_dir, 'umap.png'))

        plot_umatrix(
            umatrix=som_c.som_.umatrix, cmap='magma', plot_cbar=True, title='U-matrix', ax=axd['A'],
            **{'interpolation': 'none'}
        )

        # Plot support size vs entropy in a scatter plot
        plot_support_entropy_scatter(
            class_counts_per_unit=som_c.class_counts_per_unit_,
            label_mapping=som_c.new_to_og_classes_dict_,
            cmap=cmap,
            prefer_seaborn_cmap=True,
            plot_legend=True,
            title='Support vs. Entropy',
            ax=axd['B'],
            **{'s': 2.0}
        )

        annotate_mosaic(fig=fig, axd=axd, fontsize=14)

        plt.savefig(os.path.join(save_p, 'plot_02.png'), dpi=300)
        plt.close('all')

    elif plot_nr == '3':

        # Set flags
        sample = '12'
        plot_som_units = True


        # Load SOM classifier
        som_c = SomClassifier.load(
            os.path.join(os.getcwd(), 'results/pred_eval_som_classifier/our_data/arcsinhcofactor150/som_classifier.pkl')
        )

        # Load data from samples
        x_test = np.load(
            os.path.join(os.getcwd(),
                         f'input/np_files/our_data/arcsinhcofactor150/sample_wise_test/x_test_sample_{sample}.npy'
                         )
        )
        y_test = np.load(
            os.path.join(os.getcwd(),
                         f'input/np_files/our_data/arcsinhcofactor150/sample_wise_test/y_test_sample_{sample}.npy'
                         )
        )

        y_pred = som_c.predict(x_test).astype(int)
        y_pred_proba = som_c.predict_proba(x_test)
        y_conf = y_pred_proba.max(axis=1)

        if plot_som_units:

            x_codebook = som_c.som_.codebook
            x_codebook = x_codebook.reshape(-1, x_codebook.shape[2])

            y_codebook = np.array([som_c.new_to_og_classes_dict_[c] for c in som_c.som_unit_labels_.flatten()])

            y_conf_codebook = np.array([np.nan] * y_codebook.shape[0])

            x_test = np.vstack((x_test, x_codebook))
            y_test = np.hstack((y_test, y_codebook))
            y_pred = np.hstack((y_pred, y_codebook))
            y_conf = np.hstack((y_conf, y_conf_codebook))

            som_unit_bool = np.zeros_like(y_pred)
            som_unit_bool[-y_codebook.shape[0]:] = 1
            som_unit_bool = som_unit_bool.astype(bool)

        adata_p = os.path.join(save_p, f'adata_umap_{sample}.h5ad')
        try:
            adata = sc.read_h5ad(adata_p)
        except FileNotFoundError:
            print('# ### No AnnData with precomputed umap found. Computing knn-graph and umap ...')
            adata = sc.AnnData(x_test)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.write(adata_p)

        # Add labels to adata for plotting
        adata.obs['y_test'] = [str(label) for label in y_test]
        adata.obs['y_pred'] = [str(label) for label in y_pred]
        adata.obs['y_conf'] = y_conf

        if plot_som_units:
            adata.obs['som_unit_bool'] = som_unit_bool

        # Define a consistent colormap
        num_classes = len(som_c.new_to_og_classes_dict_)
        palette = sns.color_palette('deep', num_classes)
        cmap = {str(label): color for label, color in zip(np.unique(np.hstack((y_test, y_pred))), palette)}

        # Define marker size
        marker_size = 120000 / adata.X.shape[0]

        # Define adata for plotting
        if plot_som_units:
            adata_plot = adata[np.logical_not(adata.obs['som_unit_bool']), :].copy()
        else:
            adata_plot = adata.copy()

        fig = plt.figure(figsize=(12, 3), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            """
            ABC
            """
        )

        sc.pl.umap(
            adata_plot,
            color='y_test',
            ax=axd['A'],
            show=False,
            title="UMAP - Ground Truth",
            palette=cmap,
            size=marker_size
        )

        # UMAP colored by predicted labels (y_pred)
        sc.pl.umap(
            adata_plot,
            color='y_pred',
            ax=axd['B'],
            show=False,
            title="UMAP - Predicted Labels",
            palette=cmap,
            size=marker_size
        )

        if plot_som_units:
            axd['B'].scatter(
                adata.obsm['X_umap'][adata.obs['som_unit_bool'], 0].copy(),
                adata.obsm['X_umap'][adata.obs['som_unit_bool'], 1].copy(),
                s=marker_size * 6,
                marker='*',  # Different marker (star)
                color=[cmap[str(label)] for label in adata.obs['y_test'][adata.obs['som_unit_bool']].copy()],
                label='SOM units',
                edgecolors='black',
                linewidths=0.5,
                alpha=0.9,
            )

        sc.pl.umap(
            adata_plot,
            color='y_conf',
            ax=axd['C'],
            show=False,
            title="UMAP - Confidence",
            color_map='Greens',
            size=marker_size,
        )

        annotate_mosaic(fig=fig, axd=axd, fontsize=12)

        plt.savefig(os.path.join(save_p, 'plot_03.png'), dpi=300)
        plt.close('all')

    elif plot_nr == '4':

        data_set = 'our_data'  # 'our_data', 'flowcyt'
        trafo = 'arcsinhcofactor150'
        # trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_100'
        # trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'

        wout_unkn = True  # Whether to load results where unknowns are excluded from evaluation

        eval_metric = 'f1'  # 'prec', 'rec', 'f1'

        wu = '_wout_unk' if wout_unkn else ''

        # ### Load the evaluation results
        base_p = os.path.join(os.getcwd(), f'results/')

        confidence_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        perf_metrics = []  # macro, micro, weighted
        perf_metrics_class_wise = []  # -1, 0, 1, ...
        n_unkn_class_wise  =[]
        n_unkn = []

        for threshold in confidence_thresholds:

            res_df = pd.read_csv(
                os.path.join(
                    base_p, 'pred_eval_som_classifier', data_set, trafo, 'conf_thresh', f'conf_thresh_{threshold}',
                    'sample_wise', f'{eval_metric}{wu}.csv'
                ),
                index_col=0
            )
            perf_metrics.append(res_df.loc['mean'].to_numpy())

            res_df_class_wise = pd.read_csv(
                os.path.join(
                    base_p, 'pred_eval_som_classifier', data_set, trafo, 'conf_thresh', f'conf_thresh_{threshold}',
                    'sample_wise', f'{eval_metric}_class_wise{wu}.csv'
                ),
                index_col=0
            )

            if '-1' in res_df_class_wise.columns:
                perf_metrics_class_wise.append(res_df_class_wise.loc['mean'].to_numpy())
            else:
                perf_metrics_class_wise.append(np.hstack((np.array([np.nan]), res_df_class_wise.loc['mean'].to_numpy())))

            cf_matrices = [
                f for f in os.listdir(
                    os.path.join(
                        base_p, 'pred_eval_som_classifier', data_set, trafo, 'conf_thresh', f'conf_thresh_{threshold}',
                        'sample_wise'
                    )
                )
                if f.startswith('confusion_matrix_')
            ]

            n_unknowns = 0
            n_unknowns_class_wise = np.zeros(perf_metrics_class_wise[0].shape[0] - 1)
            for cf_mat_fn in cf_matrices:
                cf_mat = pd.read_csv(os.path.join(
                        base_p, 'pred_eval_som_classifier', data_set, trafo, 'conf_thresh', f'conf_thresh_{threshold}',
                        'sample_wise', cf_mat_fn
                    ),
                    index_col=0
                )

                # Add missing rows
                expected_indices = [-1, ] + [i for i in range(1, perf_metrics_class_wise[0].shape[0])]
                expected_cols = ['-1', ] + [str(i) for i in range(1, perf_metrics_class_wise[0].shape[0])]
                # cf_before = cf_mat.copy()
                cf_mat = cf_mat.reindex(index=expected_indices, columns=expected_cols, fill_value=0)

                # if not cf_before.equals(cf_mat):
                #     print('######')
                #     print(cf_before)
                #     print(cf_mat)

                cf_mat_col = cf_mat['-1'].to_numpy()
                n_unknowns_class_wise += cf_mat_col[1:]  # exclude entry corresponding to (-1, -1) in the cf matrix
                n_unknowns += cf_mat_col[1:].sum()

            n_unkn_class_wise.append(n_unknowns_class_wise)
            n_unkn.append(n_unknowns)

        # print(perf_metrics)
        # print(perf_metrics_class_wise)
        # print(n_unkn_class_wise)
        # print(n_unkn)

        # Remove nan entries corresponding to class -1 and results corresponding to threshold 1.0
        perf_metrics = perf_metrics[:-1]
        perf_metrics_class_wise = [a[1:] for a in perf_metrics_class_wise[:-1]]


        classes = list(range(1, perf_metrics_class_wise[0].shape[0] + 1))
        num_classes = len(classes)
        palette = sns.color_palette('deep', num_classes)
        cmap = {str(label): color for label, color in zip(classes, palette)}

        fig = plt.figure(figsize=(12, 9), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            """
            ABC
            DEF
            GH.
            """
        )

        # Plot number of cells predicted as unknown
        axd['A'].plot(
            confidence_thresholds[:-1],
            n_unkn[:-1],
            label='n_unknowns',
            color='red',
            marker="o",
            linestyle="-"
        )
        axd['A'].legend()
        axd['A'].set_title('Number of Unknowns')
        axd['A'].set_xlabel('Confidence Threshold')
        axd['A'].set_ylabel('n')

        # Plot class-wise number of cells predicted as unknown
        for i, label in enumerate(classes):
            y_values = np.array([entry[i] for entry in n_unkn_class_wise[:-1]])
            axd['B'].plot(confidence_thresholds[:-1], y_values, label=label, color=cmap[str(label)], marker="o",
                          linestyle="-")
        axd['B'].legend()
        axd['B'].set_title('Number of Unknowns by Class')
        axd['B'].set_xlabel('Confidence Threshold')
        axd['B'].set_ylabel('n')


        # Plot class-wise relative number of unknown predictions
        # n cells of class i pred as unkn / n cells pred as class i if thresh were 0
        for i, label in enumerate(classes):
            y_values = np.array([entry[i] / n_unkn_class_wise[-1][i] for entry in n_unkn_class_wise[:-1]])

            axd['C'].plot(confidence_thresholds[:-1], y_values, label=label, color=cmap[str(label)], marker="o",
                          linestyle="-")
        axd['C'].legend()
        axd['C'].set_title('Fraction of Unknowns by Class')
        axd['C'].set_xlabel('Confidence Threshold')
        axd['C'].set_ylabel('n class i pred as unkn / n pred as class i if thresh = 0')

        ################################################################################################################
        axd['D'].plot(
            confidence_thresholds,
            np.log10(1 + np.array(n_unkn)),
            label='n_unknowns',
            color='red',
            marker="o",
            linestyle="-"
        )
        axd['D'].legend()
        axd['D'].set_title('Number of Unknowns')
        axd['D'].set_xlabel('Confidence Threshold')
        axd['D'].set_ylabel('log10(1 + n)')

        # Plot class-wise number of cells predicted as unknown
        for i, label in enumerate(classes):
            y_values = np.log10(1 + np.array([entry[i] for entry in n_unkn_class_wise]))
            axd['E'].plot(confidence_thresholds, y_values, label=label, color=cmap[str(label)], marker="o",
                          linestyle="-")
        axd['E'].legend()
        axd['E'].set_title('Number of Unknowns by Class')
        axd['E'].set_xlabel('Confidence Threshold')
        axd['E'].set_ylabel('log10(1 + n)')

        # Plot class-wise relative number of unknown predictions
        # n cells of class i pred as unkn / n cells pred as class i if thresh were 0
        for i, label in enumerate(classes):
            y_values = np.array([entry[i] / n_unkn_class_wise[-1][i] for entry in n_unkn_class_wise])
            log_bool = (y_values > 0)
            y_values[log_bool] = - np.log(y_values[y_values > 0])
            y_values[~log_bool] = np.nan
            axd['F'].plot(confidence_thresholds, y_values, label=label, color=cmap[str(label)], marker="o",
                          linestyle="-")
        axd['F'].legend()
        axd['F'].set_title('Fraction of Unknowns by Class')
        axd['F'].set_xlabel('Confidence Threshold')
        axd['F'].set_ylabel('-log10(.)')

        ################################################################################################################
        # Plot macro, micro and weighted performance score
        f1_macro_micro_weighted_cmap = plt.get_cmap('Set2')
        score_types = ['macro', 'micro', 'weighted']
        for i, score_type in enumerate(score_types):
            y_values = [entry[i] for entry in perf_metrics]
            axd['G'].plot(
                confidence_thresholds[:-1],
                y_values,
                label=score_type,
                color=f1_macro_micro_weighted_cmap(i),
                marker="o",
                linestyle="-"
            )

        axd['G'].legend()
        axd['G'].set_title('F1')
        axd['G'].set_xlabel('Confidence Threshold')
        axd['G'].set_xlabel(eval_metric.capitalize())

        # Plot class-wise performance scores
        for i, label in enumerate(classes):
            y_values = [entry[i] for entry in perf_metrics_class_wise]
            axd['H'].plot(confidence_thresholds[:-1], y_values, label=label, color=cmap[str(label)], marker="o", linestyle="-")

        axd['H'].legend()
        axd['H'].set_title('Class-wise F1')
        axd['H'].set_xlabel('Confidence Threshold')
        axd['H'].set_xlabel(eval_metric.capitalize())


        annotate_mosaic(fig, axd, fontsize=15)

        plt.savefig(os.path.join(save_p, 'plot_04.png'), dpi=300)
        plt.close('all')


def main_simple_umap():

    import numpy as np
    import scanpy as sc
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.metrics import f1_score
    from flowsrc.flowsom import SomClassifier
    from validation.plotting import plot_som_pies

    np.random.seed(42)

    num_samples = 1000

    means0 = np.array([0, 0, 0, 1, 1, 1])
    std_devs0 = np.array([1, 1, 1, 2, 2, 2])
    x0 = np.random.normal(loc=means0, scale=std_devs0, size=(num_samples, means0.shape[0]))
    y0 = np.zeros(num_samples)

    means1 = np.array([3, 3, 3, 2, 2, 2])
    std_devs1 = np.array([2, 2, 2, 1, 1, 1])
    x1 = np.random.normal(loc=means1, scale=std_devs1, size=(num_samples, means0.shape[0]))
    y1 = np.ones(num_samples)

    x = np.vstack((x0, x1))
    y = np.hstack((y0, y1)).astype(int)

    shuffle_idx = np.random.permutation(num_samples * 2)
    x = x[shuffle_idx, :]
    y = y[shuffle_idx]

    som_c = SomClassifier(n_epochs=100, verbosity=2)

    som_c.fit(x, y)

    y_pred = som_c.predict(x)

    print(f'# ### F1: {f1_score(y, y_pred, average="macro")}')

    x_codebook = som_c.som_.codebook
    x_codebook = x_codebook.reshape(-1, x_codebook.shape[2])

    y_codebook = np.array([som_c.new_to_og_classes_dict_[c] for c in som_c.som_unit_labels_.flatten()])

    x = np.vstack((x, x_codebook))
    y = np.hstack((y, y_codebook))
    y_pred = np.hstack((y_pred, y_codebook))

    som_unit_bool = np.zeros_like(y_pred)
    som_unit_bool[-y_codebook.shape[0]:] = 1
    som_unit_bool = som_unit_bool.astype(bool)


    adata = sc.AnnData(x)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # Add labels to adata for plotting
    adata.obs['y_test'] = [str(label) for label in y]
    adata.obs['y_pred'] = [str(label) for label in y_pred]
    adata.obs['som_unit_bool'] = som_unit_bool

    # Define a consistent colormap
    num_classes = len(som_c.new_to_og_classes_dict_)
    palette = sns.color_palette('deep', num_classes)
    cmap = {str(label): color for label, color in zip(np.unique(np.hstack((y, y_pred))), palette)}
    listed_cmap = ListedColormap([cmap[str(label)] for label in sorted(cmap.keys())])

    # Define marker size
    marker_size = 120000 / adata.X.shape[0]

    # Define adata for plotting
    adata_plot = adata[np.logical_not(adata.obs['som_unit_bool']), :].copy()


    fig = plt.figure(figsize=(9, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        """
    )

    plot_som_pies(
        class_counts_per_unit=som_c.class_counts_per_unit_,
        som_dimensions=som_c.som_dimensions,
        label_mapping=som_c.new_to_og_classes_dict_,
        title='SOM Pies',
        cmap=listed_cmap,
        apply_tightlayout=False,
        ax=axd['A'],
    )

    sc.pl.umap(
        adata_plot,
        color='y_test',
        ax=axd['B'],
        show=False,
        title="UMAP - Ground Truth",
        palette=cmap,
        size=marker_size
    )

    # UMAP colored by predicted labels (y_pred)
    sc.pl.umap(
        adata_plot,
        color='y_pred',
        ax=axd['C'],
        show=False,
        title="UMAP - Predicted Labels",
        palette=cmap,
        size=marker_size
    )

    axd['C'].scatter(
        adata.obsm['X_umap'][adata.obs['som_unit_bool'], 0].copy(),
        adata.obsm['X_umap'][adata.obs['som_unit_bool'], 1].copy(),
        s=marker_size * 2,
        marker='*',  # Different marker (star)
        color=[cmap[str(label)] for label in adata.obs['y_test'][adata.obs['som_unit_bool']].copy()],
        label='SOM units',
        edgecolors='black',
        linewidths=0.5,
        alpha=0.9,
    )

    plt.savefig('./results/plots/simple_umap.png', dpi=300)
    plt.close('all')



def main_aml_to_fcs():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from validation.plotting import plot_som_disks


    # ### Set flags and important variables here #######################################################################
    data_set = 'aml'
    trafo = 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise'

    channels_aml = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    ####################################################################################################################

    # ### Load the sample-wise train data
    data_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/{trafo}/sample_wise_train')

    samples_x_train = [f for f in os.listdir(data_p) if f.startswith('x_train')]
    samples_x_train = [np.load(os.path.join(data_p, f)) for f in samples_x_train]

    sample_names = [
        f.removeprefix('x_train_').removesuffix('.npy') for f in os.listdir(data_p) if
        f.startswith('x_train')
    ]

    sample_ids = [i for i in range(len(sample_names))]

    # ### Load the original, untransformed data
    # Load the dataframe in which the original (.fcs) and new (.npy) data filenames are saved
    sample_names_mapping_df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            'input/np_files',
            data_set,
            trafo,
            f'sample_wise_train/sample_names_mapping_train.csv'
        )
    )

    raw_data_p = os.path.join(os.getcwd(), f'input/raw/aml/')

    samples_x_raw_train = []

    for sn in sample_names:
        # Get corresponding .fcs filename for each .npy sample name
        idx = sample_names_mapping_df[sample_names_mapping_df['new_sample_name'] == sn + '.npy'].index[0]
        og_sample_name = sample_names_mapping_df.at[idx, 'og_sample_name']

        # Create flow data manager for current sample, load to anndata and transform to dataframe
        fdm = FlowDataManager(
            data_file_names=[og_sample_name, ],
            data_file_type=None,  # file type is inferred from file ending, .fcs or .csv
            data_file_path=raw_data_p,
            save_path=None,
            memory_saving=False,
            verbosity=1,
        )

        fdm.load_data_files_to_anndata()

        x_raw_train = fdm.anndata_list_[0].to_df()

        samples_x_raw_train.append(x_raw_train)

    # Load SOM classifier
    som_p = os.path.join(os.getcwd(), f'results_new/pred_eval_som_classifier/{data_set}/{trafo}')
    som_c = SomClassifier.load(filepath=som_p)

    # Annotate and save to fcs
    samples_x_train = samples_x_train[0:3]
    samples_x_raw_train = samples_x_raw_train[0:3]
    sample_ids = sample_ids[0:3]
    save_p = os.getcwd()
    fcs_df = som_c.export_fcs(
        X=samples_x_train,
        channel_names_X=channels_aml,
        X_raw=samples_x_raw_train,
        channel_names_X_raw=None,  # Use existing column names of df
        keep_X=True,
        val_range=(0.0, 2**20),
        keep_unscaled=True,
        sample_ids=sample_ids,
        compute_umap=True,
        umap_kwargs=None,
        save_mode='fcs',
        fcs_metadata_dict=None,
        save_path=save_p,
        filename='som.fcs',
    )

    fcs_df.to_csv(os.path.join(save_p, f'som.csv'))
    print(fcs_df)
    print(fcs_df.columns)

    # Define colormap
    num_classes = len(som_c.new_to_og_classes_dict_)
    palette = sns.color_palette('deep', num_classes)
    cmap = mcolors.ListedColormap(palette)

    plot_som_disks(
        x_vals=fcs_df['bmu1_plot'].to_numpy(),
        y_vals=fcs_df['bmu2_plot'].to_numpy(),
        scatter_kwargs={'s': 0.5},
        labels=fcs_df['population'].to_numpy().astype(int),
        cmap=cmap,
        plot_legend=True,
        circle_x_y_r=fcs_df[['bmu1', 'bmu2', 'radius']].drop_duplicates(inplace=False).to_numpy(),
        title='SOM disks',
        dpi=300,
        aspect_ratio='auto',
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_p, 'som_disks.png'))
    plt.close('all')


if __name__ == '__main__':

    # ### Load, process and save data to numpy files
    # main_data_preparation()

    # ### Parameter tuning
    # main_parameter_tuning()  # todo: ran with log10 (som0, 21.02.), ran with log10_channel_wise (som0, 28.02.), run for arcsinh
    # main_view_parameter_tuning_results()

    # main_n_epochs_calibration()  # todo: ran in som0, done: now view results
    # main_view_n_epochs_calibration_results()


    # ### Generate results
    # ## Run with flowcy env
    # main_som_classifier()
    # main_som_classifier_with_confidence_threshold()
    # main_test_n_labeled_samples_som_classifier()

    # ## Run with dgcytof env
    # main_dgcytof()  # todo: ran on weneg (dg0)
    #
    # Bug report: our_data, log10_100 (dg0); Corr matrix was scalar => IndexError: invalid index to scalar variable.
    #             -> Try again with converting to matrix added to function
    # main_test_n_labeled_samples_dgcytof()  # todo: ran on weneg (dg1)

    # ## Run with gmc env
    # main_gatemeclass()  # todo: ran in gmc0
    #
    # BUG report: our_data, log10_100, w(out)unknowns (gmc0,1), our_data, log10_cw, w(out)unkn (gmc2, 3);
    # - error: rpy2.rinterface_lib.embedded.RRuntimeError: Error in predict.Mclust(cl, X[-index_sel]) : object not of class 'Mclust'
    # - GateMeClass_annotate() (annotates cells) calls set_marker_expression() (sets the marker signature of each cell)
    #   calls set_marker_expression_GMM(X, ...) (applies GMM to classify marker expressions, X is expression vector of marker)
    # - Now: if RSS is active, if data negatively skewed then test is dominated by low values
    # - cl <- Mclust(test, G = 2, verbose = F, modelNames = GMM_parameterization) Fails!!!
    # - probably this fails because data is negatively skewed due to cutoff and all values are the same ...
    # - Note that if parameterization is 'E' instead of 'V' then test=X, this probably resolves the error
    # - ??? Why only problem at sample level ???
    # main_test_n_labeled_samples_gatemeclass()  # todo: ran in gmc1

    # ### Analyze results
    # main_som_plots()

    # main_threshold_plots()

    # main_som_to_fcs()  # todo: ran with umap (som1)

    # main_view_prediction_evaluation_results()

    # main_view_n_labeled_samples_results()

    # main_plot_n_labeled_samples()

    # ### View and plot the result
    # main_plotting()

    # main_simple_umap()

    main_aml_to_fcs()



    # Todo:
    #  - Debug relabeling (2nd iteration of inner loop fails?)  #
    #  - Go over evaluation metrics calculation  #
    #  - Go over the FlowDataManager  #
    #  - Integrate lymphoma dataset  #
    #  - run dgcytof on weneg #
    #  - Also do n samples for som and other methods #
    #  - Parameter tuning?
    #  - Should exclude -1 channel from macro f1 calculation
    #  - Do all coding, then run again from scratch ???
    #  - Maybe include sklearn classifiers: idea: similar performance to classical, worse than dl but explainability
    #  - Maybe 20x20 is better, generate results
    #  - verbosity = 0 nothing, 1 warnings, 2, info




    # conda create -n flowcy scanpy pytometry imbalanced-learn scikit-learn numba
    # pip install somoclu
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    print('done')

