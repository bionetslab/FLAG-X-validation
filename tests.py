
# Todo: write proper unittests for main classes of package


def test_load_anndata():
    import os
    import random
    import torch
    import numpy as np
    from flagx.io import FlowDataManager

    # ### Define function for setting random seeds
    def set_random_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # ### Set random seed
    set_random_seed()

    # ### Create a list of the filenames
    save_path = os.path.join(os.getcwd(), "results/test/data_handling")
    raw_data_path = os.path.join(os.getcwd(), "input/raw/imstat")
    filename_list = os.listdir(raw_data_path)[0:3]

    # ### Test loading of data to anndata
    # No memory_saving, 2 invalid
    fdm = FlowDataManager(
        data_file_names=filename_list + ["invalid0", "invalid1.csv"],
        data_file_type=None,
        data_file_path=raw_data_path,
        save_path=save_path,
        memory_saving=False,
        verbosity=1
    )
    print("# ### No memory saving, two invalid files")
    print("'data_file_type': ", fdm.data_file_type)
    fdm.load_data_files_to_anndata()
    print("'anndata_list_':\n", fdm.anndata_list_)
    print("'invalid_files_':\n", fdm.invalid_files_)

    # memory_saving all invalid
    fdm = FlowDataManager(
        data_file_names=filename_list,
        data_file_type="csv",
        data_file_path=raw_data_path,
        save_path=save_path,
        memory_saving=True,
        verbosity=1
    )
    print("# ### Memory saving but wrong filetype (all should be invalid)")
    print("'data_file_type': ", fdm.data_file_type)
    fdm.load_data_files_to_anndata()
    print("'anndata_list_':\n", fdm.anndata_list_)
    print("'invalid_files_':\n", fdm.invalid_files_)

    # memory_saving, 2 invalid
    fdm = FlowDataManager(
        data_file_names=filename_list + ["invalid0", "invalid1.csv"],
        data_file_type=None,
        data_file_path=raw_data_path,
        save_path=save_path,
        memory_saving=True,
        verbosity=1
    )
    print("# ### Memory saving but 2 invalid files")
    print("'data_file_type': ", fdm.data_file_type)
    fdm.load_data_files_to_anndata()
    print("'anndata_list_':\n", fdm.anndata_list_)
    print("'invalid_files_':\n", fdm.invalid_files_)


def test_flowio():

    import os
    import flowio
    import readfcs
    import numpy as np

    n_channels = 6
    channel_names = [f"ch_{i}" for i in range(n_channels)]
    x = np.random.randint(low=0, high=100 ,size=(10, n_channels))

    meta_dict = {f"$P{i}R": str(12) for i in range(1, n_channels + 1)}

    print(f"Min value: {x.min()}, max: {x.max()}")

    save_path = os.path.join(os.getcwd(), "results/test/check_flowio_range/")
    os.makedirs(save_path, exist_ok=True)
    fn = "test.fcs"

    with open(os.path.join(save_path, fn), "wb") as f:
        flowio.create_fcs(
            file_handle=f,
            event_data=x.flatten().tolist(),
            channel_names=channel_names,
            opt_channel_names=channel_names,
            metadata_dict=meta_dict,
        )

    adata = readfcs.view(os.path.join(save_path, fn))
    print(adata)


def test_export_fcs():

    import os
    import readfcs
    import numpy as np

    from flagx.gating import SomClassifier

    np.random.seed(42)

    n_channels = 3
    channel_names = [f"ch_{i}" for i in range(n_channels)]
    x = np.random.randint(low=0, high=100, size=(6, n_channels))
    y = np.full(x.shape[0], -999)

    save_path = os.path.join(os.getcwd(), "results/test/check_flowio_range/")
    os.makedirs(save_path, exist_ok=True)
    fn = "test.fcs"

    sc = SomClassifier(som_dimensions=(2, 2), unlabeled_label=-999, verbosity=2)

    sc.fit(x, y)


    sc.export_fcs(
        X=x, channel_names_X=channel_names, val_range=(0, 2**20), save_unscaled_data=False,
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### No meta_dict, save_unscaled_data = False:\n", adata)

    meta_dict = {f"$P{i}G": str(11) for i in range(1, n_channels + 1)}
    sc.export_fcs(
        X=x, channel_names_X=channel_names, val_range=(0, 2**20), save_unscaled_data=False,
        fcs_metadata_dict=meta_dict, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### With meta_dict save_unscaled_data = False:\n", adata)

    sc.export_fcs(
        X=x, channel_names_X=channel_names, val_range=(0, 2 ** 20), save_unscaled_data=True,
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### No meta_dict, save_unscaled_data = True:\n", adata)

    # With metadict
    meta_dict = {f"$P{i}G": str(111111111111111111111111) for i in range(1, n_channels + 1)}
    sc.export_fcs(
        X=x, channel_names_X=channel_names, val_range=(0, 2 ** 20), save_unscaled_data=True,
        fcs_metadata_dict=meta_dict, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### With meta_dict save_unscaled_data = True:\n", adata)


    # With x_raw
    channel_names_raw = [f"raw_ch_{i}" for i in range(n_channels)]
    x_raw = np.random.randint(low=0, high=100, size=(6, n_channels))

    sc.export_fcs(
        X=x, channel_names_X=channel_names, X_raw=x_raw, channel_names_X_raw=channel_names_raw, keep_X=False,
        val_range=(0, 2 ** 10), save_unscaled_data=False,
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### No meta_dict input, x_raw, save_unscaled_data = False, keep_X = False:\n", adata)

    sc.export_fcs(
        X=x, channel_names_X=channel_names, X_raw=x_raw, channel_names_X_raw=channel_names_raw, keep_X=True,
        val_range=(0, 2 ** 10), save_unscaled_data=False,
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### No meta_dict input, x_raw, save_unscaled_data = False, keep_X = True:\n", adata)

    sc.export_fcs(
        X=x, channel_names_X=channel_names, X_raw=x_raw, channel_names_X_raw=channel_names_raw, keep_X=False,
        val_range=(0, 2 ** 10), save_unscaled_data=True,
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### No meta_dict input, x_raw, save_unscaled_data = True, keep_X = False:\n", adata)

    sc.export_fcs(
        X=x, channel_names_X=channel_names, X_raw=x_raw, channel_names_X_raw=channel_names_raw, keep_X=True,
        val_range=(0, 2 ** 10), save_unscaled_data=True,
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print("# ### No meta_dict input, x_raw, save_unscaled_data = True, keep_X = True:\n", adata)

    sc.export_fcs(
        X=x, channel_names_X=channel_names, X_raw=x_raw, channel_names_X_raw=channel_names_raw, keep_X=True,
        val_range=(0, 2 ** 10), save_unscaled_data=True, scale_X_raw_channels=['raw_ch_2'],
        fcs_metadata_dict=None, save_mode="fcs", save_path=save_path, filename=fn
    )
    adata = readfcs.view(os.path.join(save_path, fn))
    print(
        "# ### No meta_dict input, x_raw, save_unscaled_data = True, keep_X = True, scale one of the raw channels:\n",
        adata[-1]
    )


def test_ram():

    import os
    import psutil
    import resource

    from typing import List

    from flagx.io import FlowDataManager

    # Get the current process ID
    pid = os.getpid()
    process = psutil.Process(pid)

    # Function to get the .fcs/.csv file sizes
    def get_total_size(data_file_path: str, data_file_names: List[str]):
        total_size = 0
        for fn in data_file_names:
            file_path = os.path.join(data_file_path, fn)
            total_size += os.path.getsize(file_path)  # Get file size in bytes

        # Convert bytes to MB
        total_size = total_size / ((1024 * 1024))

        return total_size

    # Function to monitor memory usage
    def memory_usage():
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    mem_init = memory_usage()
    print(f"Initial Memory Usage: {mem_init :.2f} MB")

    # dfp = os.path.join(os.getcwd(), 'data/raw/lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels')
    dfp = os.path.join(os.getcwd(), 'data/raw/lymphoma/concatenated_labled_csv_format_22Blood4Bcell_T2_Labels')
    # dfp = os.path.join(os.getcwd(), 'data/raw/flowcyt/data_original/')
    # dfp = os.path.join(os.getcwd(), 'data/raw/imstat')
    dfn = os.listdir(dfp)

    # Use double of our biggest dataset (lymphoma tube 2: 2 x 9548950 events)
    dfn += dfn

    total_filesize = get_total_size(dfp, dfn)
    print(f"Total File Size: {total_filesize :.2f} MB")

    fdm = FlowDataManager(data_file_names=dfn, data_file_path=dfp, memory_saving=False)

    fdm.load_data_files_to_anndata()

    mem_after = memory_usage()
    print(f"Memory Usage after data loading: {mem_after :.2f} MB")

    print(f"FCS data in RAM approx.: {mem_after - mem_init :.2f} MB")

    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak memory usage: {peak_memory / 1024:.2f} MB")

    print(type(fdm.anndata_list_[0].X))
    print(type(fdm.anndata_list_[0].X.dtype))

    # Fcs:
    # Locally:
    # Initial Memory Usage: 668.77 MB
    # Total File Size: 1020.20 MB
    # Memory Usage after data loading: 3141.42 MB
    # FCS data in RAM approx.: 2472.65 MB
    # Peak memory usage: 3141.42 MB

    # Ramses:
    # Initial Memory Usage: 498.89 MB
    # Total File Size: 1020.20 MB
    # Memory Usage after data loading: 2973.34 MB
    # FCS data in RAM approx.: 2474.45 MB
    # Peak memory usage: 2974.10 MB

    # Csv (64 bit):
    # Locally:
    # Initial Memory Usage: 680.45 MB
    # Total File Size: 1798.12 MB
    # Memory Usage after data loading: 4187.11 MB
    # FCS data in RAM approx.: 3506.66 MB
    # Peak memory usage: 4213.61 MB

    # Ramses:
    # Initial Memory Usage: 498.96 MB
    # Total File Size: 1798.12 MB
    # Memory Usage after data loading: 4000.11 MB
    # FCS data in RAM approx.: 3501.15 MB
    # Peak memory usage: 4022.31 MB

    # Csv (32 bit):
    # Locally:
    # Initial Memory Usage: 666.20 MB
    # Total File Size: 1798.12 MB
    # Memory Usage after data loading: 3106.89 MB
    # FCS data in RAM approx.: 2440.69 MB
    # Peak memory usage: 3126.83 MB

    # Ramses:
    # Initial Memory Usage: 498.21 MB
    # Total File Size: 1798.12 MB
    # Memory Usage after data loading: 2935.08 MB
    # FCS data in RAM approx.: 2436.87 MB
    # Peak memory usage: 2949.78 MB


def test_flowdatamanager():

    import os
    import random
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from flagx.io import FlowDataManager

    # ### Define function for setting random seeds
    def set_random_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ### Set random seed
    set_random_seed()

    # ### Create a list of the filenames
    save_path = os.path.join(os.getcwd(), "results/test/test_fdm/data_handling")
    raw_data_path = os.path.join(os.getcwd(), "data/raw/imstat")
    filename_list = os.listdir(raw_data_path)[0:6]

    # ### Test loading of data to anndata
    # 2 invalid filenames
    fdm = FlowDataManager(
        data_file_names=filename_list + ["invalid0", "invalid1.csv"],
        data_file_type=None,
        data_file_path=raw_data_path,
        save_path=save_path,
        verbosity=2,
    )
    print("# ### Load fcs files, two invalid files")
    print("'data_file_type' before loading: ", fdm.data_file_type)
    fdm.load_data_files_to_anndata()
    print("'data_file_type' after loading: ", fdm.data_file_type)
    print("'anndata_list_':\n", len(fdm.anndata_list_), type(fdm.anndata_list_[0]))
    print("'invalid_files_':\n", fdm.invalid_files_)

    print('# ### Check sample sizes')
    fdm.check_sample_sizes(filename_sample_sizes_df='sample_sizes.csv')
    print("'sample_sizes_:'\n", fdm.sample_sizes_)

    print('# ### Align channel names, also calls check_og_channel_names_df()')
    fdm.align_channel_names(reference_channel_names=0, filename_log_df='og_channel_names.csv')  # Use 1st file in list as reference
    print("'og_channel_names_:'\n", fdm.og_channel_names_)

    print('# ### Apply preprocessing transformation')
    print('# Arcsinh')
    print('adata before pepr:\n', fdm.anndata_list_[0], '\n',fdm.anndata_list_[0].X)
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='raw')
    print('adata after pepr:\n', fdm.anndata_list_[0], '\n', fdm.anndata_list_[0].X)
    print('# Log10 with cutoff 100')
    print('adata before pepr:\n', fdm.anndata_list_[0], '\n', fdm.anndata_list_[0].X)
    fdm.sample_wise_preprocessing(flavour='log10_w_cutoff', save_raw_to_layer=None)
    print(
        'adata after pepr:\n', fdm.anndata_list_[0], '\n', fdm.anndata_list_[0].X,
        '\n', fdm.anndata_list_[0].layers['raw']
    )

    print('# ### Perform data split')
    print('# train and test')
    fdm.perform_data_split((0.5, 0.5), filename_data_split='data_split_train_test.csv')
    print('train_data_', len(fdm.train_data_), 'val_data_', fdm.val_data_,'test_data_', len(fdm.test_data_))
    ds_df = pd.read_csv(os.path.join(fdm.save_path, 'data_split_train_test.csv'), index_col=0)
    print(ds_df)
    fdm.perform_data_split(ds_df)
    print('train_data_', len(fdm.train_data_), 'val_data_', fdm.val_data_, 'test_data_', len(fdm.test_data_))

    print('# train, val and test')
    fdm.perform_data_split((0.5, 0.25, 0.25), filename_data_split='data_split_train_val_test.csv')
    print('train_data_', len(fdm.train_data_), 'val_data_', len(fdm.val_data_), 'test_data_', len(fdm.test_data_))
    ds_df = pd.read_csv(os.path.join(fdm.save_path, 'data_split_train_val_test.csv'), index_col=0)
    print(ds_df)
    fdm.perform_data_split(ds_df)
    print('train_data_', len(fdm.train_data_), 'val_data_', len(fdm.val_data_), 'test_data_', len(fdm.test_data_))

    print('# ### Sample wise downsampling')
    print('# non stratified')
    y = fdm._get_numpy_label_vector(data_list=fdm.train_data_[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts / counts.sum())}
    print('Before ds, n events: ', counts.sum(), 'relative counts: ', value_counts)
    train_data_list = fdm.sample_wise_downsampling_worker(
        data_list=fdm.train_data_,
        fraction=0.1,
        stratified=False,
        label_key='population',
        label_layer_key='raw',
        inplace=False
    )
    y = fdm._get_numpy_label_vector(data_list=train_data_list[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts / counts.sum())}
    print('After ds, n events: ', counts.sum(), 'relative counts: ', value_counts)

    print('# stratified')
    y = fdm._get_numpy_label_vector(data_list=fdm.train_data_[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts / counts.sum())}
    print('Before ds, n events: ', counts.sum(), 'relative counts: ', value_counts)
    train_data_list = fdm.sample_wise_downsampling_worker(
        data_list=fdm.train_data_,
        fraction=0.1,
        stratified=True,
        label_key='population',
        label_layer_key='raw',
        inplace=False
    )
    y = fdm._get_numpy_label_vector(data_list=train_data_list[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts / counts.sum())}
    print('After ds, n events: ', counts.sum(), 'relative counts: ', value_counts)

    print('# stratified and inplace')
    y = fdm._get_numpy_label_vector(data_list=fdm.train_data_[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts / counts.sum())}
    print('Before ds, n events: ', counts.sum(), 'relative counts: ', value_counts)
    fdm.sample_wise_downsampling(
        data_set='train',
        fraction=0.1,
        stratified=True,
        label_key='population',
        label_layer_key='raw',
    )
    y = fdm._get_numpy_label_vector(data_list=fdm.train_data_[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts / counts.sum())}
    print('After ds, n events: ', counts.sum(), 'relative counts: ', value_counts)

    print('# ### Relabel data')
    y = fdm._get_numpy_label_vector(data_list=fdm.train_data_[:1], label_key='population', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts)}
    print('Before relabeling, counts: ', value_counts)
    fdm.relabel_data(
        data_set='train',
        old_to_new_label_mapping={1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 300},
        label_key='population',
        label_layer_key='raw',
        new_label_key='new_labels',
    )
    y = fdm._get_numpy_label_vector(data_list=fdm.train_data_[:1], label_key='new_labels', layer_key='raw', verbosity=2)
    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = {int(c): round(float(frac), 4) for c, frac in zip(unique_values, counts)}
    print('After relabeling, counts: ', value_counts)

    print('# ### Get data loader')
    dl = fdm.get_data_loader(
        data_set='train',
        channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '19-APC'],
        layer_key='raw',
        label_key='population',
        label_layer_key='raw',
        batch_size=-1,
        shuffle=True,
        return_data_loader='np_array',
        on_disk=True,
        filename_np='dl_data.npy'
    )

    x, y = next(iter(dl))

    print('x: ', x)
    print('y: ', y)

    print('# ### Save to numpy')
    sp_np = os.path.join(fdm.save_path, 'np_files')
    os.makedirs(sp_np, exist_ok=True)
    fdm.save_to_numpy_files(
        data_set='train',
        sample_wise=False,
        save_path=sp_np,
        filename_suffix='_train',
        channels=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '19-APC'],
        layer_key='raw',
        label_key='population',
        label_layer_key='raw',
        shuffle=True,
    )

    fdm.save_to_numpy_files(
        data_set='train',
        sample_wise=True,
        save_path=sp_np,
        filename_suffix='_train',
        channels=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '19-APC'],
        layer_key='raw',
        label_key='population',
        label_layer_key='raw',
        shuffle=True,
    )
    print('Saved files: ', os.listdir(sp_np))

    fdm.save_to_numpy_files(
        data_set='train',
        sample_wise=True,
        save_path=sp_np,
        filename_suffix='_train',
        channels=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '19-APC'],
        layer_key='raw',
        label_key='population',
        label_layer_key='raw',
        shuffle=True,
    )

    print('# ### Check class balance')
    cb_df = fdm.check_class_balance(
        data_set='train',
        label_key='population',
        label_layer_key='raw',
        filename_class_balance_df='class_balance_train.csv',
    )
    print('class balance train:\n', cb_df)

    fdm.plot_class_balance_df(class_balance_df=cb_df, dpi=300)
    plt.savefig(os.path.join(fdm.save_path, 'class_balance_train.png'))



    # Todo:
    #  - test all fdm functionality here, assume int labels
    #  - Tool box dim red: vis module, map pred cell type to data, export to fcs (maybe sth like pipeline function)
    #  - Implement softmax in gating module
    #  - Go over other classifiers in validation, refactor to package
    #    ->  go over which plotting/cal funct may be needed in flagx
    #  - Refactor main scripts, add proposed experiments (if predicts unknown, change pred to others or ignore pred = pass labels=np.unique(y_true))
    #  - Look at report




# pytometry 0.1.6 is latest, mamba only has 0.1.4

# mamba create -n flowy scanpy python-igraph leidenalg somoclu numba umap-learn scipy scikit-learn seaborn matplotlib=3.6
# pip install flowio pytometry torch torchvision torchaudio
# pip uninstall matplotlib
# mamba uninstall matplotlib
# mamba install matplotlib=3.6





if __name__ == '__main__':

    # test_load_anndata()

    # test_flowio()

    # test_export_fcs()

    # test_ram()

    test_flowdatamanager()

    print('done')
