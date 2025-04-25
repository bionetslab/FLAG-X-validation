import numpy as np


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
    # dfp = os.path.join(os.getcwd(), 'data/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels')
    # dfp = os.path.join(os.getcwd(), 'data/raw/flowcyt/data_original/')
    # dfp = os.path.join(os.getcwd(), 'data/raw/imstat')
    dfn = os.listdir(dfp)

    # Use double of our biggest dataset (lymphoma tube 2: 2 x 9548950 events)
    dfn += dfn

    total_filesize = get_total_size(dfp, dfn)
    print(f"Total File Size: {total_filesize :.2f} MB")

    fdm = FlowDataManager(data_file_names=dfn, data_file_path=dfp)

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
    fdm.plot_sample_size_df(sample_size_df=fdm.sample_sizes_, dpi=300, ax=None)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sample_sizes.png"))

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

    print('x: ', x.shape)
    print('y: ', y)

    print('# ### Save to numpy')
    sp_np = os.path.join(fdm.save_path, 'np_files')
    os.makedirs(sp_np, exist_ok=True)
    fdm.save_to_numpy_files(
        data_set='train',
        sample_wise=False,  # !!!
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
        sample_wise=True,  # !!!
        save_path=sp_np,
        filename_suffix='_train',
        channels=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '19-APC'],
        layer_key='raw',
        label_key='population',
        label_layer_key='raw',
        shuffle=True,
    )
    print('Saved files: ', os.listdir(sp_np))

    print('Dtype saved files: ', np.load(os.path.join(sp_np, 'x_train.npy')).dtype)

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
        precision='16bit',  # !!!
    )
    print('Dtype saved files: ', np.load(os.path.join(sp_np, 'x_train.npy')).dtype)


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


def test_torch_randomness():

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Define a function to test reproducibility
    def test_seed(seed: int):

        # ### 1) Set seed and generate random numbers on cpu and gpu
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Generate random numbers on CPU
        cpu_rand_1 = torch.rand(3, 3)

        # Generate random numbers on CUDA (if available)
        cuda_rand_1 = torch.rand(3, 3, device="cuda") if torch.cuda.is_available() else None

        # ### 2) Reset the seed and generate numbers again
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        cpu_rand_2 = torch.rand(3, 3)
        cuda_rand_2 = torch.rand(3, 3, device="cuda") if torch.cuda.is_available() else None

        # Check if values match after resetting seed
        cpu_match = torch.equal(cpu_rand_1, cpu_rand_2)
        cuda_match = torch.equal(cuda_rand_1, cuda_rand_2) if cuda_rand_1 is not None else True

        return cpu_match, cuda_match, cpu_rand_1, cuda_rand_1

    # Run the test with a fixed seed
    s = 42
    cpu_match, cuda_match, cpu_rand, cuda_rand = test_seed(s)

    # Print results
    print("CPU Match:", cpu_match)
    print("CUDA Match:", cuda_match)
    print("First CPU Random Tensor:\n", cpu_rand)
    if cuda_rand is not None:
        print("First CUDA Random Tensor:\n", cuda_rand)


    # Function to test deterministic shuffling with seeds
    def test_dataloader_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Sample dataset: 10 data points (features) with 2 values each
        data = torch.arange(20).view(10, 2).float()
        labels = torch.arange(10)  # Labels from 0 to 9
        print(data)

        # Create a TensorDataset
        dataset = TensorDataset(data, labels)

        # Create a DataLoader with shuffling enabled
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

        # Collect batches
        batches = []
        for batch in dataloader:
            features, targets = batch
            batches.append((features, targets))

        return batches

    # Run the test with a fixed seed twice
    s = 42
    batches_1 = test_dataloader_seed(s)
    batches_2 = test_dataloader_seed(s)

    # Check if shuffling is deterministic (batches should be identical)
    deterministic = all(torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1]) for b1, b2 in zip(batches_1, batches_2))

    # Display results
    print(deterministic)
    print(batches_1)
    print(batches_2)


def test_softmax_classifier():

    import os
    import time
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_blobs
    from sklearn.metrics import f1_score

    from flagx.utils import set_random_seed
    from flagx.gating import SoftmaxClassifier

    set_random_seed(seed=42)

    X, y = make_blobs(n_samples=100000, n_features=11, centers=6, cluster_std=6.0, shuffle=True, random_state=42)

    clf0 = SoftmaxClassifier(
        layer_sizes=(128, 64, 32),
        n_epochs=6,
        # data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
        device=None,
        verbosity=2,
    )

    clf0.fit(X, y)

    x_test, y_test = make_blobs(
        n_samples=1000, n_features=11, centers=6, cluster_std=6.0, shuffle=False, random_state=42
    )

    y_proba = clf0.predict_proba(x_test)
    y_pred = clf0.predict(x_test)

    print('# Predicted probs:\n', y_proba)
    print('# Predicted labels:\n', y_pred)

    f1 = f1_score(y_test, y_pred, average=None)

    print(f1)

    print(clf0.og_classes_)

    res_df = pd.DataFrame(columns=clf0.og_classes_)
    res_df.loc['f1'] = f1

    print('f1_scores:\n', res_df)

    print('# ### Test saving and loading')
    fp = os.path.join(os.getcwd(), 'results/test/test_softmax')
    os.makedirs(fp, exist_ok=True)
    clf0.save(filepath=fp)
    del clf0
    clf0 = SoftmaxClassifier.load(filepath=fp)
    y_pred_loaded = clf0.predict(x_test)
    print('# Predicted labels:\n', y_pred_loaded)
    print('# Same prediction:', np.all(y_pred == y_pred_loaded))

    print('# ### Test GPU vs CPU training')
    n_epochs = 10

    clf1 = SoftmaxClassifier(n_epochs=n_epochs, device='cpu')
    st_cpu = time.time()
    clf1.fit(X[:1000, :], y[:1000])
    et_cpu = time.time()
    print(f'# Epoch on CPU took: {(et_cpu - st_cpu) /n_epochs:.2f} seconds')

    if torch.cuda.is_available():
        print('# Cuda is available')
        print(f'# Found {torch.cuda.device_count()} GPUs')

        clf2 = SoftmaxClassifier(n_epochs=n_epochs, device='cuda:0')
        st_gpu = time.time()
        clf2.fit(X[:1000, :], y[:1000])
        et_gpu = time.time()
        print(f'# Epoch on GPU took: {(et_gpu - st_gpu) / n_epochs:.2f} seconds')
    else:
        print('# Cuda is not available')


def test_param_tuning():

    from sklearn.datasets import make_blobs

    from flagx.utils import set_random_seed
    from flagx.gating import SomClassifier

    set_random_seed(seed=42)

    x, y = make_blobs(n_samples=1000, n_features=11, centers=6, cluster_std=6.0, shuffle=True, random_state=42)

    # Define parameter grid for neighborhood parameters
    param_grid_0 = {
        'gaussian_neighborhood_sigma': [0.5, 1.0],
        'radius_0': [-0.5, -0.75],
        'radius_n': [1.0, 0.01, ],
        'n_epochs': [6,],
    }

    param_grid_1 = {
        'gaussian_neighborhood_sigma': [0.5, 1.0],
        'radius_0': [-0.5, -0.75],
        'radius_n': [1.0, 0.01, ],
    }

    # Set no params
    som_clf_0 = SomClassifier(verbosity=0)
    print('# ### Case 0')
    som_clf_0.hyperparameter_tuning(X=x, y=y, param_grid=param_grid_0, cv=2, scoring='internal', refit=True)
    for attr, value in vars(som_clf_0).items():
        print(f"{attr} = {value}")


    # Set some params t other than default value
    som_clf_1 = SomClassifier(som_dimensions=(5, 5), n_epochs=6, verbosity=0)
    print('# ### Case 1')
    som_clf_1.hyperparameter_tuning(X=x, y=y, param_grid=param_grid_1, cv=2, scoring='internal', refit=True)
    for attr, value in vars(som_clf_1).items():
        print(f"{attr} = {value}")

    # Also: Put print(f'# ### SOM dimensions: {self.som_dimensions}') in .fit(), seems to work as expected

    best_params = som_clf_1.grid_search_.best_params_

    print(best_params)

    som_clf_2 = SomClassifier(verbosity=2, **best_params)

    print(vars(som_clf_2))


def test_metrics_calc():

    import numpy as np

    from sklearn.metrics import precision_score, recall_score, f1_score

    from validation.utils import (
        eval_score_with_abstention,
        prec_rec_f1_avg, prec_rec_f1_class_wise,
        prec_rec_f1_avg_sample_wise, prec_rec_f1_class_sample_wise,
        abstention_counts, abstention_counts_sample_wise,
        confusion_matrix_df, confusion_matrix_df_sample_wise,
        eval_wrapper, eval_wrapper_sample_wise
    )

    np.random.seed(0)

    print('# ###### Binary input ###### #')
    y_true = np.random.randint(low=0, high=2, size=(10,))
    print('# ### y_true: ', y_true)
    y_pred = np.random.randint(low=0, high=2, size=(10,))
    print('# ### y_pred: ', y_pred, '\n')

    print('# ### No avg specified (binary)')
    score_sk = f1_score(y_true, y_pred)
    score_abst = eval_score_with_abstention(y_true, y_pred, eval_func=f1_score, eval_func_kwargs=None)
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Binary')
    score_sk = f1_score(y_true, y_pred, average='binary', pos_label=1)
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'binary', 'pos_label': 1}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Macro')
    score_sk = f1_score(y_true, y_pred, average='macro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'macro'}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Micro')
    score_sk = f1_score(y_true, y_pred, average='micro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'micro'}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Weighted')
    score_sk = f1_score(y_true, y_pred, average='weighted')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'weighted'}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ###### Binary input, with unknowns ###### #')
    y_true = np.random.randint(low=0, high=2, size=(20,))
    print('# ### y_true: ', y_true)
    y_pred = np.random.randint(low=0, high=2, size=(20,))
    y_pred[np.random.randint(low=0, high=y_pred.shape[0], size=(6,))] = -1
    print('# ### y_pred: ', y_pred, '\n')

    print('# ### No avg specified (binary)')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'pos_label': 1}, abstention_label=-1
    )
    print('# Abstention: ', score_abst, '\n')

    print('# ### Binary')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'binary', 'pos_label': 1},
        abstention_label=-1
    )
    print('# Abstention: ', score_abst, '\n')

    print('# ### Macro')
    score_sk = f1_score(y_true, y_pred, average='macro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'macro'}, abstention_label=-1
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Micro')
    score_sk = f1_score(y_true, y_pred, average='micro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'micro'}, abstention_label=-1
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Weighted')
    score_sk = f1_score(y_true, y_pred, average='weighted')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'weighted'}, abstention_label=-1
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')


    print('# ###### Multi class input ###### #')
    y_true = np.random.randint(low=0, high=6, size=(100,))
    print('# ### y_true: ', y_true)
    y_pred = np.random.randint(low=0, high=6, size=(100,))
    print('# ### y_pred: ', y_pred, '\n')

    print('# ### Macro')
    score_sk = f1_score(y_true, y_pred, average='macro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'macro'}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Micro')
    score_sk = f1_score(y_true, y_pred, average='micro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'micro'}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Weighted')
    score_sk = f1_score(y_true, y_pred, average='weighted')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'weighted'}
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')


    print('# ###### Multi class input, with unknowns ###### #')
    y_true = np.random.randint(low=0, high=6, size=(100,))
    print('# ### y_true: ', y_true)
    y_pred = np.random.randint(low=0, high=6, size=(100,))
    y_pred[np.random.randint(low=0, high=y_pred.shape[0], size=(20,))] = -1
    print('# ### y_pred: ', y_pred, '\n')

    print('# ### Macro')
    score_sk = f1_score(y_true, y_pred, average='macro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'macro'}, abstention_label=-1
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Micro')
    score_sk = f1_score(y_true, y_pred, average='micro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'micro'}, abstention_label=-1
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Weighted')
    score_sk = f1_score(y_true, y_pred, average='weighted')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'weighted'}, abstention_label=-1
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')


    print('# ###### Multi class input, with unknowns, with others ###### #')
    y_true = np.random.randint(low=0, high=6, size=(100,))
    print('# ### y_true: ', y_true)
    y_pred = np.random.randint(low=0, high=6, size=(100,))
    y_pred[np.random.randint(low=0, high=y_pred.shape[0], size=(20,))] = -1
    print('# ### y_pred: ', y_pred, '\n')

    print('# ### Macro')
    score_sk = f1_score(y_true, y_pred, average='macro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'macro'}, abstention_label=-1, others_label=0
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Micro')
    score_sk = f1_score(y_true, y_pred, average='micro')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'micro'}, abstention_label=-1,  others_label=0
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    print('# ### Weighted')
    score_sk = f1_score(y_true, y_pred, average='weighted')
    score_abst = eval_score_with_abstention(
        y_true, y_pred, eval_func=f1_score, eval_func_kwargs={'average': 'weighted'},
        abstention_label=-1, others_label=0
    )
    print('# Sklearn: ', score_sk, ' # Abstention: ', score_abst, '\n')

    def test_no_abstentions():
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        assert eval_score_with_abstention(y_true, y_pred, precision_score) == 1.0

    def test_abstentions_binary():
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, -1, 0])
        score = eval_score_with_abstention(
            y_true, y_pred, recall_score,
            eval_func_kwargs={'average': 'binary', 'pos_label': 1},
            abstention_label=-1
        )
        assert score == 0.5

    def test_abstentions_macro():
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, -1, 0])
        score = eval_score_with_abstention(
            y_true, y_pred, f1_score,
            eval_func_kwargs={'average': 'macro'},
            abstention_label=-1
        )
        assert isinstance(score, float)

    def test_abstentions_with_others():
        y_true = np.array([0, 1, 99, 0])  # others_label exists in ground truth
        y_pred = np.array([0, 1, -1, 0])
        score = eval_score_with_abstention(
            y_true, y_pred, f1_score,
            eval_func_kwargs={'average': 'macro'},
            abstention_label=-1,
            others_label=99
        )
        assert isinstance(score, float)

    test_no_abstentions()
    test_abstentions_binary()
    test_abstentions_macro()
    test_abstentions_with_others()

    print('# ### Results dataframe averaged ### #')
    y_true = np.random.randint(low=0, high=6, size=(100,))
    print('# ### y_true: ', y_true)
    y_pred = np.random.randint(low=0, high=6, size=(100,))
    y_pred[np.random.randint(low=0, high=y_pred.shape[0], size=(20,))] = -1
    print('# ### y_pred: ', y_pred, '\n')

    prec_rec_f1_avg(y_true=y_true, y_pred=y_pred, abstention_label=-1, others_label=None, verbosity=2)
    prec_rec_f1_avg(y_true=y_true, y_pred=y_pred, abstention_label=-1, others_label=0, verbosity=2)

    print('# ### Results dataframe class-wise ### #')
    prec_rec_f1_class_wise(y_true=y_true, y_pred=y_pred, abstention_label=-1, others_label=None, verbosity=2)
    prec_rec_f1_class_wise(y_true=y_true, y_pred=y_pred, abstention_label=-1, others_label=0, verbosity=2)


    print('# ### Results dataframe averaged, sample-wise ### #')
    y_trues = ([np.random.randint(low=0, high=6, size=(100,)) for i in range(6)] +
               [np.random.randint(low=0, high=5, size=(100,))])
    print('# ### y_true: ', y_true)
    y_preds = ([np.random.randint(low=-1, high=6, size=(100,)) for i in range(6)] +
               [np.random.randint(low=-1, high=5, size=(100,))])
    print('# ### y_pred: ', y_pred, '\n')

    prec_rec_f1_avg_sample_wise(y_trues=y_trues, y_preds=y_preds, abstention_label=-1, others_label=None, verbosity=2)
    prec_rec_f1_avg_sample_wise(y_trues=y_trues, y_preds=y_preds, abstention_label=-1, others_label=0, verbosity=2)

    print('# ### Results dataframe class-wise, sample-wise ### #')
    prec_rec_f1_class_sample_wise(y_trues=y_trues, y_preds=y_preds, abstention_label=-1, others_label=None, verbosity=2)
    prec_rec_f1_class_sample_wise(y_trues=y_trues, y_preds=y_preds, abstention_label=-1, others_label=0, verbosity=2)


    print('# ### Abstention counts ### #')
    abstention_counts(y_true=y_trues[0], y_pred=y_preds[0], abstention_label=-1, verbosity=2)

    print('# ### Abstention counts, sample-wise ### #')
    abstention_counts_sample_wise(y_trues=y_trues, y_preds=y_preds, abstention_label=-1, verbosity=2)


    print('# ### Confusion matrix ### #')
    confusion_matrix_df(y_true=y_trues[0], y_pred=y_preds[0], verbosity=2)

    print('# ### Confusion matrix sample-wise ### #')
    #print(confusion_matrix_df_sample_wise(y_trues=y_trues, y_preds=y_preds))

    print('# ### Eval wrapper ### #')
    out_avg = eval_wrapper(y_true=y_true, y_pred=y_pred, abstention_label=-1, others_label=None, verbosity=2)

    print('# ### Eval wrapper, sample-wise ### #')
    out_sw = eval_wrapper_sample_wise(
        y_trues=y_trues, y_preds=y_preds, abstention_label=-1, others_label=None, verbosity=2
    )


def test_binary_and_micro():

    import numpy as np
    from sklearn.metrics import precision_score, recall_score

    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]

    prec_binary = precision_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=1)
    rec_binary = recall_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=1)

    prec_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    rec_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')

    print(f'# ### Binary: prec: {prec_binary}, rec: {rec_binary}')
    print(f'# ### Micro: precision: {prec_micro}, recall: {rec_micro}')

    prec_binaries = []
    rec_binaries = []
    prec_micros = []
    rec_micros = []

    for i in range(100):

        np.random.seed(i)

        y_true = np.random.randint(low=0, high=2, size=(1000,))
        y_pred = np.random.randint(low=0, high=2, size=(1000,))

        prec_binary = precision_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=1)
        rec_binary = recall_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=1)

        prec_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        rec_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')

        prec_binaries.append(prec_binary)
        rec_binaries.append(rec_binary)

        prec_micros.append(prec_micro)
        rec_micros.append(rec_micro)

    # print(f'# ### Binary:\nprec: {prec_binaries},\nrec: {rec_binaries}\n')
    # print(f'# ### Micro:\nprecision: {prec_micros},\nrecall: {rec_micros}\n')

    print('# ### Precision: ', np.all(np.array(prec_binaries) == np.array(prec_micros)))
    print('# ### Recall: ', np.all(np.array(rec_binaries) == np.array(rec_micros)))


def test_data_processing_results():

    import os
    import numpy as np
    import pandas as pd
    from validation.utils import set_pandas_print_options

    set_pandas_print_options()

    ds = 'lymphoma_tube2_binary'
    trafo = 'log10_channelwisecutoff'

    data_p = os.path.join(os.getcwd(), 'data/np_files/', ds, trafo)

    sample_mapping_df_test = pd.read_csv(
        os.path.join(data_p, 'sample_wise_test/sample_names_mapping_test.csv'),
        index_col=0,
    )
    print('# ### Sample mapping test:\n', sample_mapping_df_test.head(6))

    sample_mapping_df_train = pd.read_csv(
        os.path.join(data_p, 'sample_wise_train/sample_names_mapping_train.csv'),
        index_col=0,
    )
    print('# ### Sample mapping train:\n', sample_mapping_df_train.head(6))

    x_test = np.load(os.path.join(data_p, 'x_test.npy'))
    y_test = np.load(os.path.join(data_p, 'y_test.npy'))

    print('# ### x_test:\n', x_test)
    print('# ### y_test:\n', y_test)

    x_train = np.load(os.path.join(data_p, 'x_train.npy'))
    y_train = np.load(os.path.join(data_p, 'y_train.npy'))

    print('# ### x_train:\n', x_train)
    print('# ### y_train:\n', y_train)


def test_expanding_iterator():

    from validation.utils.val_utils import _get_expanding_iterator_list

    out = _get_expanding_iterator_list(n=6, m=3)

    print(out)


def test_copy():

    import copy

    from flagx.gating import SomClassifier, SoftmaxClassifier

    clf0 = SomClassifier()

    clf1 = copy.deepcopy(clf0)

    clf0.verbosity = 2

    print(clf0.verbosity)
    print(clf1.verbosity)

    clf0 = SoftmaxClassifier()

    clf1 = copy.deepcopy(clf0)

    clf0.verbosity = 2

    print(clf0.verbosity)
    print(clf1.verbosity)


def test_gpu_som_training():

    import numpy as np

    from flagx.gating import SomClassifier

    n_events = 10000
    x = np.random.randn(n_events, 11).astype(np.float32)
    y = np.random.randint(low=0, high=3, size=(n_events,)).astype(np.int32)

    clf = SomClassifier(som_dimensions=(30, 30), kernel_type=0, verbosity=2)

    clf.fit(x, y)


def test_my_som():

    import os
    import time
    import numpy as np
    import numba

    import matplotlib.pyplot as plt


    from somoclu import Somoclu
    # from flagx.utils.som import SOM

    # numba.config.DEBUG_ARRAY_OPT_STATS = True

    n_samples = 10000
    # x = np.random.normal(loc=0, scale=6, size=(n_samples, 12)).astype(np.float32)
    # x = np.random.random((n_samples, 12)).astype(np.float32)
    x = np.vstack((np.zeros((int(n_samples/2), 12)), np.ones((int(n_samples/2), 12)))).astype(np.float32)

    save_p = os.path.join(os.getcwd(), 'results/test/som_windows')
    os.makedirs(save_p, exist_ok=True)

    fig, ax = plt.subplots(dpi=300)
    ax.hist(x.flatten(), bins=100, edgecolor='black')
    plt.savefig(os.path.join(save_p, 'x.png'))

    n_epochs = 100
    som_dims = (10, 10)

    test_online_numba = False
    test_batch_numba = False
    batch_size = n_samples
    test_somoclu = True

    if test_online_numba:
        som0 = SOM(
            som_dimensions=som_dims,
            distance_metric='euclidean',
            neighborhood= 'gaussian',
            gaussian_neighborhood_sigma=0.5,
            initialization='pca',
            n_epochs=n_epochs,
            radius_0=6,
            radius_n=1.0,
            radius_cooling='linear',
            learning_rate_0=0.1,
            learning_rate_n=0.01,
            learning_rate_decay='linear',
            batch_size=None,
            verbosity=1
        )

        st = time.time()
        som0.fit(x)
        et = time.time()

        fit_time = et - st

        print('# ### Fit time:', fit_time)
        print('# ### Time per epoch:', fit_time / n_epochs)

    if test_batch_numba:
        som1 = SOM(
            som_dimensions=som_dims,
            distance_metric='euclidean',
            neighborhood='gaussian',
            gaussian_neighborhood_sigma=0.5,
            initialization='pca',
            n_epochs=n_epochs,
            radius_0=6,
            radius_n=1.0,
            radius_cooling='linear',
            learning_rate_0=0.1,
            learning_rate_n=0.01,
            learning_rate_decay='linear',
            batch_size=batch_size,
            verbosity=1
        )

        st = time.time()
        som1.fit(x)
        et = time.time()

        fit_time = et - st

        print('# ### Numba batch ### #')
        print('# ### Fit time:', fit_time)
        print('# ### Time per epoch:', fit_time / n_epochs)

        print(som1.codebook_)

        fig, ax = plt.subplots(dpi=300)
        ax.hist(som1.codebook_.flatten(), bins=100, edgecolor='black')
        plt.savefig(os.path.join(save_p, 'my_som.png'))


    if test_somoclu:
        som2 = Somoclu(
                n_columns=som_dims[0],
                n_rows=som_dims[1],
                gridtype='rectangular',
                maptype='planar',
                neighborhood='gaussian',
                std_coeff=0.5,
                initialization='pca',
                initialcodebook=None,
                kerneltype=0,
                verbose=2,
            )

        st = time.time()
        som2.train(
            data=x,
            epochs=n_epochs,
            radius0=6,
            radiusN=1.0,
            radiuscooling='linear',
            scale0=0.1,
            scaleN=0.01,
            scalecooling='linear',
        )
        et = time.time()

        fit_time = et - st
        print('# ### Somoclu ### #')
        print('# ### Fit time:', fit_time)
        print('# ### Time per epoch:', fit_time / n_epochs)

        fig, ax = plt.subplots(dpi=300)
        ax.hist(som2.codebook.flatten(), bins=100, edgecolor='black')
        plt.savefig(os.path.join(save_p, 'somomclu.png'))

        print(som2.codebook)


def test_windows_env():

    import time
    import numpy as np

    from somoclu import Somoclu

    n_samples = 10
    x = np.random.normal(loc=0, scale=6, size=(n_samples, 12)).astype(np.float32)
    # x = np.random.random((n_samples, 12)).astype(np.float32)
    # x = np.vstack((np.zeros((int(n_samples / 2), 12)), np.ones((int(n_samples / 2), 12)))).astype(np.float32)

    n_epochs = 10
    som_dims = (3, 3)

    som = Somoclu(
        n_columns=som_dims[0],
        n_rows=som_dims[1],
        gridtype='rectangular',
        maptype='planar',
        neighborhood='gaussian',
        std_coeff=0.5,
        initialization='pca',
        initialcodebook=None,
        kerneltype=0,
        verbose=2,
    )

    st = time.time()
    som.train(
        data=x,
        epochs=n_epochs,
        radius0=6,
        radiusN=1.0,
        radiuscooling='linear',
        scale0=0.1,
        scaleN=0.01,
        scalecooling='linear',
    )
    et = time.time()

    fit_time = et - st
    print('# ### Somoclu ### #')
    print('# ### Fit time:', fit_time)
    print('# ### Avg time per epoch:', fit_time / n_epochs)

    print(som.codebook)

    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    from flagx import gating
    from flagx import io
    from flagx import dimred
    from flagx import plt
    from flagx import utils


def test_som_transform():

    import numpy as np
    import matplotlib.pyplot as plt

    from flagx.gating import SomClassifier
    from validation.plt.plotting import plot_som_disks

    n_samples = 10000
    x0 = np.random.normal(loc=0, scale=6, size=(int(n_samples / 2), 12)).astype(np.float32)
    x1 = np.random.normal(loc=6, scale=6, size=(int(n_samples / 2), 12)).astype(np.float32)
    x = np.vstack((x0, x1))

    y = np.full(int(n_samples / 2) * 2, -999)

    n_epochs = 100
    som_dims = (2, 3)

    clf = SomClassifier(som_dimensions=som_dims, n_epochs=n_epochs, unlabeled_label=-999, verbosity=2)

    clf.fit(x, y)

    bmus, bmus_scattered, bmu_ids, radii = clf.transform(x)

    print(bmus)
    print(bmus_scattered)
    print(bmu_ids)
    print(radii)

    circle_x_y_r = np.hstack((bmus, radii.reshape(-1, 1)))

    fig, ax = plt.subplots(dpi=300)
    plot_som_disks(
        x_vals=bmus_scattered[:, 0],
        y_vals=bmus_scattered[:, 1],
        scatter_kwargs={'s': 3.0},
        labels=bmu_ids,
        cmap='husl',
        circle_x_y_r=circle_x_y_r,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig('zzz.png', dpi=300)


def test_pipeline0():
    """
    Script for testing the GatingPipeline.
    Setting:
        - Gating method: SOM
        - Dimensionality reduction method: PCA, SOM, UMAP
        - Try sample-wise and all in one .fcs saving
        - Try different input parameters (e.g. dim_red_method_kwargs)
    Returns: None

    """

    import os
    import readfcs
    import matplotlib.pyplot as plt
    from seaborn import scatterplot
    from flagx import GatingPipeline

    train_data_p = os.path.join(os.getcwd(), 'data/raw/test')

    save_p = os.path.join(os.getcwd(), 'results/test/test_pipeline0')
    save_p_fdm = os.path.join(save_p, 'train_fdm')
    os.makedirs(save_p_fdm, exist_ok=True)

    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    label_key = 'population'

    relabel_data_kwargs = {
        'old_to_new_label_mapping': {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1},
        'new_label_key': 'relabeled_label'
    }

    preprocessing_kwargs = {'flavour': 'arcsinh', 'flavour_kwargs': {'cofactor': 150}, 'save_raw_to_layer': 'no_trafo'}

    gating_method_kwargs = {'som_dimensions': (3, 3), 'n_epochs': 10, 'radius_0': -0.6, 'radius_n': 0.01}

    gp = GatingPipeline(
        train_data_file_path=train_data_p,
        train_data_file_names=None,  # all files in dir
        train_data_file_type='fcs',
        train_data_manager_save_path=save_p_fdm,
        channels=channels,
        label_key=label_key,
        channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
        relabel_data_kwargs=None,  # relabel_data_kwargs,
        preprocessing_kwargs=preprocessing_kwargs,
        gating_method='som',
        gating_method_kwargs=gating_method_kwargs,
        verbosity=2,
    )

    print(gp)

    gp.train()

    print(gp.gating_module_)

    dim_red_methods = ('som', 'umap', 'pca')

    gp.inference(
        data_file_path=train_data_p,
        data_file_names=None,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=save_p,
        save_filenames='train_data_anno.fcs',
        val_range=(0.0, 2**20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    fcs0 = readfcs.view(os.path.join(save_p, 'train_data_anno.fcs'))
    print("# ### All samples in one .fcs:\n", fcs0)
    print("# Channels:\n", fcs0[1].columns)

    data_tab = fcs0[1]
    data_tab['pred_unscaled'] = data_tab['pred_unscaled'].astype(int)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=data_tab, x=f'{drm}_1', y=f'{drm}_2', s=3, hue='pred_unscaled', palette='deep', ax=ax)
        plt.legend(title='Pred', markerscale=4)
        plt.savefig(os.path.join(save_p, f'{drm}.png'))
        plt.close('all')


    gp.inference(
        data_file_path=train_data_p,
        data_file_names=None,
        gate=True,
        dim_red_methods=('som', 'pca'),
        dim_red_method_kwargs=(None, {'whiten': True}),
        save_sample_wise=True,
        save_path=save_p,
        save_filenames=None,
        val_range=(0.0, 2 ** 20),
        keep_unscaled=False,
        fcs_metadata_dicts=None,
    )

    fcs1 = readfcs.view(os.path.join(save_p, 'annotated_' + os.listdir(train_data_p)[0]))
    print("# ### Sample-wise .fcs:\n", fcs1)
    print("# Channels:\n", fcs1[1].columns)


def test_pipeline1():
    """
    Script for testing the GatingPipeline.
    Setting:
        - Gating method: FCNN_Softmax
        - Dimensionality reduction method: PCA, UMAP, SOM (should raise error)
    Returns: None
    """

    import os
    import readfcs
    import matplotlib.pyplot as plt
    from seaborn import scatterplot
    from flagx import GatingPipeline

    train_data_p = os.path.join(os.getcwd(), 'data/raw/test')

    train_data_file_names = [
        '20150210-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019652 001.fcs', 'ER_000000_H1_150305_ED.fcs'
    ]

    save_p = os.path.join(os.getcwd(), 'results/test/test_pipeline1')
    save_p_fdm = os.path.join(save_p, 'train_fdm')
    os.makedirs(save_p_fdm, exist_ok=True)

    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    label_key = 'population'

    relabel_data_kwargs = {
        'old_to_new_label_mapping': {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1},
        'new_label_key': 'relabeled_label'
    }

    preprocessing_kwargs = {'flavour': 'arcsinh', 'flavour_kwargs': {'cofactor': 150}, 'save_raw_to_layer': 'no_trafo'}

    gating_method_kwargs = {
        'layer_sizes': (128, 64, 32), 'n_epochs': 3, 'data_loader_params': None, 'device': None,  'verbosity': 2}

    gp = GatingPipeline(
        train_data_file_path=train_data_p,
        train_data_file_names=train_data_file_names,
        train_data_file_type='fcs',
        train_data_manager_save_path=save_p_fdm,
        channels=channels,
        label_key=label_key,
        channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
        relabel_data_kwargs=relabel_data_kwargs,
        preprocessing_kwargs=preprocessing_kwargs,
        gating_method='fcnn_softmax',
        gating_method_kwargs=gating_method_kwargs,
        verbosity=2,
    )

    print(gp)

    gp.train()

    print(gp.gating_module_)

    dim_red_methods = ('umap', 'pca')

    gp.inference(
        data_file_path=train_data_p,
        data_file_names=['20150312-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019509 001.fcs', ],
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=save_p,
        save_filenames='train_data_anno.fcs',
        val_range=(0.0, 2**20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    fcs0 = readfcs.view(os.path.join(save_p, 'train_data_anno.fcs'))
    print("# ### All samples in one .fcs:\n", fcs0)
    print("# Channels:\n", fcs0[1].columns)

    data_tab = fcs0[1]
    data_tab['pred_unscaled'] = data_tab['pred_unscaled'].astype(int)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=data_tab, x=f'{drm}_1', y=f'{drm}_2', s=3, hue='pred_unscaled', palette='deep', ax=ax)
        plt.legend(title='Pred', markerscale=4)
        plt.savefig(os.path.join(save_p, f'{drm}.png'))
        plt.close('all')


    gp.inference(
        data_file_path=train_data_p,
        data_file_names=None,
        gate=True,
        dim_red_methods=('som', ),
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=save_p,
        save_filenames=None,
        val_range=(0.0, 2 ** 20),
        keep_unscaled=False,
        fcs_metadata_dicts=None,
    )


def test_pipeline2():
    """
    Script for testing the GatingPipeline.
    Setting:
        - Gating method: SOM, unsupervised case
        - Dimensionality reduction method: PCA, SOM, UMAP
        - Try sample-wise and all in one .fcs saving
        - Try different input parameters (e.g. dim_red_method_kwargs)
    Returns: None

    """

    import os
    import readfcs
    import matplotlib.pyplot as plt
    from seaborn import scatterplot
    from flagx import GatingPipeline

    train_data_p = os.path.join(os.getcwd(), 'data/raw/test')

    train_data_file_names = [
        '20150210-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019652 001.fcs', 'ER_000000_H1_150305_ED.fcs'
    ]

    save_p = os.path.join(os.getcwd(), 'results/test/test_pipeline2')
    save_p_fdm = os.path.join(save_p, 'train_fdm')
    os.makedirs(save_p_fdm, exist_ok=True)

    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    label_key = None  # Pass no label key !!!

    preprocessing_kwargs = {'flavour': 'arcsinh', 'flavour_kwargs': {'cofactor': 150}, 'save_raw_to_layer': 'no_trafo'}

    gating_method_kwargs = {'som_dimensions': (3, 3), 'n_epochs': 10, 'radius_0': -0.6, 'radius_n': 0.01}

    gp = GatingPipeline(
        train_data_file_path=train_data_p,
        train_data_file_names=train_data_file_names,
        train_data_file_type='fcs',
        train_data_manager_save_path=save_p_fdm,
        channels=channels,
        label_key=label_key,
        channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
        relabel_data_kwargs=None,
        preprocessing_kwargs=preprocessing_kwargs,
        gating_method='som',
        gating_method_kwargs=gating_method_kwargs,
        verbosity=2,
    )

    print(gp)

    gp.train()

    print(gp.gating_module_)

    dim_red_methods = ('som', 'pca', 'tsne')
    dim_red_method_kwargs = (None, None, {'n_jobs': 16})

    gp.inference(
        data_file_path=train_data_p,
        data_file_names=['20150312-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019509 001.fcs', ],
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_sample_wise=False,
        save_path=save_p,
        save_filenames='train_data_anno.fcs',
        val_range=(0.0, 2**20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    fcs0 = readfcs.view(os.path.join(save_p, 'train_data_anno.fcs'))
    print("# ### All samples in one .fcs:\n", fcs0)
    print("# Channels:\n", fcs0[1].columns)

    data_tab = fcs0[1]
    data_tab['som_unit_id_unscaled'] = data_tab['som_unit_id_unscaled'].astype(int)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=data_tab, x=f'{drm}_1', y=f'{drm}_2', s=3, hue='som_unit_id_unscaled', palette='deep', ax=ax)
        plt.legend(title='som_unit_id', markerscale=4)
        plt.savefig(os.path.join(save_p, f'{drm}.png'))
        plt.close('all')


def test_pipeline_save_load():
    import os
    import readfcs
    from flagx import GatingPipeline

    train_data_p = os.path.join(os.getcwd(), 'data/raw/test')

    save_p = os.path.join(os.getcwd(), 'results/test/test_pipeline_sl')
    save_p_fdm = os.path.join(save_p, 'train_fdm')
    os.makedirs(save_p_fdm, exist_ok=True)

    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
    label_key = 'population'

    relabel_data_kwargs = {
        'old_to_new_label_mapping': {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1},
        'new_label_key': 'relabeled_label'
    }

    preprocessing_kwargs = {'flavour': 'arcsinh', 'flavour_kwargs': {'cofactor': 150}, 'save_raw_to_layer': 'no_trafo'}

    gating_method_kwargs = {'som_dimensions': (3, 3), 'n_epochs': 10, 'radius_0': -0.6, 'radius_n': 0.01}

    gp = GatingPipeline(
        train_data_file_path=train_data_p,
        train_data_file_names=None,  # all files in dir
        train_data_file_type='fcs',
        train_data_manager_save_path=save_p_fdm,
        channels=channels,
        label_key=label_key,
        channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
        relabel_data_kwargs=None,  # relabel_data_kwargs,
        preprocessing_kwargs=preprocessing_kwargs,
        gating_method='som',
        gating_method_kwargs=gating_method_kwargs,
        verbosity=2,
    )

    gp.train()

    dim_red_methods = ('som', 'pca')

    gp.inference(
        data_file_path=train_data_p,
        data_file_names=None,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=save_p,
        save_filenames='pre_save_train_data_anno.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    fcs0 = readfcs.view(os.path.join(save_p, 'pre_save_train_data_anno.fcs'))
    # print("# ### Pre save:\n", fcs0)
    # print("# Channels:\n", fcs0[1].columns)
    print('# Attributes')
    for attr, value in gp.__dict__.items():
        print(f"{attr}: {value}")

    gp.save(filename='gp.pkl', filepath=save_p)

    del gp

    gp = GatingPipeline.load(filename='gp.pkl', filepath=save_p)

    dim_red_methods = ('som', 'pca')

    gp.inference(
        data_file_path=train_data_p,
        data_file_names=None,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=save_p,
        save_filenames='after_save_train_data_anno.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=True,
        fcs_metadata_dicts=None,
    )

    fcs0 = readfcs.view(os.path.join(save_p, 'after_save_train_data_anno.fcs'))
    # print("# ### After save:\n", fcs0)
    # print("# Channels:\n", fcs0[1].columns)
    print('# Attributes')
    for attr, value in gp.__dict__.items():
        print(f"{attr}: {value}")






# Todo:
#  - Go over other flagx modules e.g. plot and add some plotting functionalities
#  - Tool box dim red: vis module, map pred cell type to data, export to fcs (maybe sth like pipeline function)
#  - Go over which plotting/cal funct may be needed in flagx, (adjust __init__ of plt and utils)
#  - Main: Refactor main scripts, add proposed experiments (grid of n samples and  downsampling frac, fit and analyze results, write toolbox simultaneously) !!!!!!
#    => Run analysis again!!!! (generate res on my machine for runtimes also)
#  - Look at report
#  - Collect/brainstorm ideas for MRD detection, use Stefan paper as starting point (diff healthy and MRD at sample level, then outlier detection)
#  - Argument (parm search etc in supplement) that SOM is not very sensitive to hyperparam choice
#  - Should probably focus on Recall as well (i.e. Do not want to accidentally exclude MRD,
#    maybe ask stefan to annotate MRD in a few samples)
#    => Idea: Use SOM nodes not only where majority is Blast but also where blast frac >= some threshold
#      (plot precision vs recall, works only for the binary case,
#       include as use case in study: usually interested in rough gating to some population,
#       in comparison to DL, SOM probs are interpretable)
#  - Write manuscript and run analysis in parallel (tuning etc on workstations, benchmark runs locally)






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

    # test_flowdatamanager()

    # test_torch_randomness()

    # test_softmax_classifier()

    # test_param_tuning()

    # test_metrics_calc()

    # test_binary_and_micro()

    # test_data_processing_results()

    # test_expanding_iterator()

    # test_copy()

    # test_gpu_som_training()

    # test_my_som()

    # test_windows_env()

    # test_som_transform()

    # test_pipeline0()

    # test_pipeline1()

    # test_pipeline2()

    test_pipeline_save_load()

    print('done')
