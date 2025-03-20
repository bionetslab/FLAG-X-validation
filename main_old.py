import copy
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd


def check_load_data():
    from flowio import FlowData
    import numpy as np
    fp = './input/dummy/20150210-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019652 001.fcs'
    fp = './input/dummy/ER_000047_H1_150311_ED.fcs'

    # Load .fcs file using flowio package
    fcdata = FlowData(fp)
    print(f'# ### fcdata.analysis:\n{fcdata.analysis}')
    print(f'# ### fcdata.channel_count:\n{fcdata.channel_count}')
    print(f'# ### fcdata.channels:\n{fcdata.channels}')  # Channel information: {channel_number: {PnN: val, PnS: val}},
    # PnN = formal channel name, PnS = descriptive channel name
    print(f'# ### fcdata.event_count:\n{fcdata.event_count}')  # Number of events
    # print(f'# ### fcdata.events:\n{fcdata.events}')  # 1d-array of event data
    print(f'# ### fcdata.file_size:\n{fcdata.file_size}')
    print(f'# ### fcdata.header:\n{fcdata.header}')  # Header info of the fcs file
    print(f'# ### fcdata.name:\n{fcdata.name}')
    print(f'# ### fcdata.text:\n{fcdata.text}')  # Dictionary of key/value pairs from the TEXT section
    # Compensation matrix would be stored in text segment under 'spillover' or 'comp'

    # Create data matrix from events
    fc_datamatrix = np.reshape(fcdata.events, (-1, fcdata.channel_count))
    print(np.unique(fc_datamatrix[:, -1]))  # Population labels


def check_pytometry():
    import pytometry as pm
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    fp = './input/dummy/ER_000047_H1_150311_ED.fcs'
    fldata = pm.io.read_fcs(fp)

    print(fldata)
    print(fldata.var_names)

    # fldata.X = (fldata.X - np.min(fldata.X, axis=0)) / (np.max(fldata.X, axis=0) - np.min(fldata.X, axis=0)) * 2 ** 20

    # pm.tl.normalize_arcsinh(fldata, cofactor=150, inplace=True)

    pm.pl.scatter_density(fldata, x='FS INT', y='SS INT')
    plt.savefig('./z_fs_vs_ss_no_norm.png')

    pm.pl.scatter_density(fldata, x='TIME', y='FS INT')
    plt.savefig('./z_fs_vs_time_no_norm.png')

    pm.pl.scatter_density(fldata, x='FS PEAK', y='FS INT')
    plt.savefig('./z_fspeak_vs_fsint_no_norm.png')

    # pm.pl.scatter_density(fldata, x='FS PEAK', y='FS TOF')
    # plt.savefig('./z_fspeak_vs_fstof_no_norm.png')

    pm.pl.plotdata(fldata, option='other')
    plt.savefig('./z_hist_no_norm.png')

    print(fldata.X)

    # pm.tl.normalize_arcsinh(fldata, cofactor=150, inplace=True)
    pm.tl.normalize_logicle(fldata)
    # fldata.X = np.log10(fldata.X, out=np.full(fldata.X.shape, np.log10(100), dtype=float), where=(fldata.X > 100))
    # sc.pp.log1p(fldata)

    pm.pl.scatter_density(fldata, x='FS INT', y='SS INT')
    plt.savefig('./z_fs_vs_ss_norm.png')

    pm.pl.scatter_density(fldata, x='TIME', y='FS INT')
    plt.savefig('./z_fs_vs_time_norm.png')

    pm.pl.scatter_density(fldata, x='FS PEAK', y='FS INT')
    plt.savefig('./z_fspeak_vs_fsint_norm.png')

    pm.pl.scatter_density(fldata, x='FS TOF', y='FS PEAK')
    plt.savefig('./z_fstof_vs_fspeak_norm.png')

    pm.pl.scatter_density(fldata, x='population', y='FS TOF')
    plt.savefig('./z_population_vs_fstof_norm.png')

    pm.pl.plotdata(fldata, option='other')
    plt.savefig('./z_hist_norm.png')

    print(fldata.X)


def test_anndata():
    import scanpy as sc
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    import time

    download = False
    if download:
        adata = sc.datasets.visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1')
        sc.write('zzz.h5ad', adata)

    class TestData(Dataset):
        def __init__(self, file_paths: list[str], array: bool = True):
            """
            Args:
                file_paths (list of str): List of file paths to AnnData (.h5ad) datasets.
            """

            self.array = array
            self.file_paths = file_paths

            # self.ad = sc.read_h5ad(self.file_paths[0], backed='r')
            # self.ad = sc.read_h5ad(self.file_paths[0])
            self.ad = np.memmap(file_paths[0], mode='r')
            print(f'### shape: {self.ad.shape}')
            # self.ad = np.load(file_paths[0])

        def __len__(self):
            return self.ad.shape[0]  # self.ad.n_obs

        def __getitem__(self, idx):

            # Retrieve the specific cell (row) as a numpy array
            # cell_data = self.ad.X[idx, :].toarray().squeeze()  # Convert sparse to dense if needed
            cell_data = self.ad[idx, :]

            if not self.array:
                # Convert the cell data to a PyTorch tensor
                cell_data = torch.tensor(cell_data, dtype=torch.float32)
            return cell_data

        def __getitems__(self, idxs):

            # cell_data = self.ad.X[idxs, :].toarray()
            cell_data = self.ad[idxs, :]

            if not self.array:
                cell_data = [torch.tensor(row, dtype=torch.float32) for row in cell_data]

            print('### ###')

            return cell_data

    n = 100  # Number of test iterations
    k = 3798  # Number of cells in the dataset

    '''rand_ints = [np.array([])] * n
    for j in range(n):
        rand_ints[j] = np.random.permutation(k)

    st = time.time()
    # Retrieve cells by batch-wise indexing
    for j in range(n):
        adata = sc.read_h5ad('zzz.h5ad', backed='r')
        # adata = sc.read_h5ad('zzz.h5ad')
        dummy = adata.X[rand_ints[j], :].toarray()
        print(dummy.shape)
    et = time.time()
    print(f'# ### Batch wise indexing: {et - st} sec')
    print(f'# ### Avg time {(et - st) / n} ')'''

    # adata = sc.read_h5ad('./zzz.h5ad')
    # a = adata.X.toarray()
    # np.save('zzz.npy', a)

    ds = TestData(file_paths=['./zzz.npy'], array=True)
    dl = DataLoader(ds, batch_size=k, shuffle=True)
    # Retrieve cells
    st = time.time()
    for j in range(n):
        print(next(iter(dl)).shape)
    et = time.time()
    print(f'# ### Torch: {et - st} sec')
    print(f'# ### Avg time {(et - st) / n} ')


def check_fcs_files():
    import pytometry as pm
    import os
    from time import time
    import matplotlib

    matplotlib.use('Agg')

    filename_list = os.listdir(os.path.join(os.getcwd(), 'input', 'concatenated_labeled_fcs_format'))

    print(filename_list)
    print('# ### Num .fcs files: ', len(filename_list))

    n_events = [0] * len(filename_list)

    st = time()
    for i in range(len(filename_list)):
        ad = pm.io.read_fcs(os.path.join(os.getcwd(), 'input', 'concatenated_labeled_fcs_format', filename_list[i]))
        n_events[i] = ad.shape[0]
    et = time()

    print(f'# ### Time: {et - st}')

    # plt.hist(np.array(n_events), bins=20)
    # plt.savefig('zz_hist.png')

    print(ad)
    print(f'# ### ad.X: \n{ad.X}')
    print(f'# ### ad.var["n"]: \n{ad.var["n"]}')  # Channel number
    print(f'# ### ad.var["channel"]: \n{ad.var["channel"]}')  # Channel name
    print(f'# ### ad.var["marker"]: \n{ad.var["marker"]}')  # Name of marker/fluorochrome corresponding to channel
    print(f'# ### ad.var["$PnB"]: \n{ad.var["$PnB"]}')  # Bit depth of values of channel
    print(f'# ### ad.var["$PnE"]: \n{ad.var["$PnE"]}')  # Scaling of the data, 0~linear, else~logarithmic
    print(f'# ### ad.var["$PnG"]: \n{ad.var["$PnG"]}')  # Amplifier gain, 1.0 ~= no amplification
    print(f'# ### ad.var["$PnR"]: \n{ad.var["$PnR"]}')  # Parameter range (1048676 = 2**20)
    print(f'# ### ad.uns["meta"]: \n{ad.uns["meta"]}')  # Metadata: Text segment, header, ...


def test_data_manager():
    from flowsrc.flowdata import FlowDataManager
    import scanpy as sc
    import os

    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))
    # filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    dummy_ref = {
        'FS PEAK': 'a', 'FS INT': 'b', 'FS TOF': 'c', 'SS INT': 'd', '16-FITC': 'e', '56-PE': 'f', '3-ECD': 'g',
        'FL4 INT': 'h', '4-PC7': 'i', '19-APC': 'j', '14-APC700': 'k', 'FL8 INT': 'l', '8-PB': 'm', '45-CO': 'n',
        'TIME': 'o', 'population': 'p', '-PC5.5': 'q'}

    memory_saving = False
    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=memory_saving, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format'
    )

    flow_manager.perform_data_split(data_split=(0.75, 0.25))

    if memory_saving:
        fcd = sc.read_h5ad(
            os.path.join(os.getcwd(), 'fc_data_processed', filename_list[0][:-4] + '.h5ad'))

    dl = flow_manager.create_data_loader(
        data_set='train', channels=['FS INT', 'SS INT', 'FL8 INT', 'population'], layer_key=None,
        return_torch_tensor=False, batch_size=-1, shuffle=True, on_disk=True, filename='zdummy_data.npy')

    print(dl)

    for x in dl:
        print(x)
        print(x.shape)

    print('###')
    # 145856.00000, 856701.00000


def some_plots():
    import pytometry as pm
    import scanpy as sc
    import os
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    filename_list = os.listdir(os.path.join(os.getcwd(), 'input', 'concatenated_labeled_fcs_format'))
    # print(filename_list)

    k = 66  # 12 (flow almost stops), 50 (in general unstable flow), 66 (startup)
    fcd = pm.io.read_fcs(os.path.join(os.getcwd(), 'input', 'concatenated_labeled_fcs_format', filename_list[k]))
    print(fcd.var_names)

    def plot_events_per_time(
            fcdata: sc.AnnData,
            time_key: str = 'TIME',
            n_time_bins: int = 100,
            verbosity: int = 0,
    ):
        times = fcdata[:, time_key].X.squeeze()
        t_min = times.min()
        t_max = times.max()

        if verbosity >= 1:
            print(f'# ### t min: {t_min}, t max: {t_max}')

        bins = np.linspace(t_min, t_max, n_time_bins + 1)
        bin_counts = np.zeros(n_time_bins)
        for i in range(bins.shape[0] - 1):
            bin_counts[i] = ((bins[i] <= times) * (times <= bins[i + 1])).sum()

        fig, ax = plt.subplots()
        ax.plot((bins[:-1] + bins[1:]) / 2, bin_counts, marker='o', markersize=3, linestyle='-', color='b')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'number of events during {(t_max - t_min) / n_time_bins} time units')
        plt.savefig('z_events_per_timeinterval.png')

    plot_events_per_time(fcdata=fcd, n_time_bins=200, verbosity=1)

    def plot_intensity_per_time(
            fcdata: sc.AnnData,
            channel_key: str = 'FS INT',
            time_key: str = 'TIME',
            n_time_bins: int = 100,
            fn: str = 'z_intensities_per_timeinterval.png',
            verbosity: int = 0,
    ):
        times = fcdata[:, time_key].X.squeeze()
        t_min = times.min()
        t_max = times.max()

        if verbosity >= 1:
            print(f'# ### t min: {t_min}, t max: {t_max}')

        intensities = fcdata[:, channel_key].X.squeeze()

        bins = np.linspace(t_min, t_max, n_time_bins + 1)
        bin_intensities = np.zeros(n_time_bins)
        for i in range(bins.shape[0] - 1):
            bin_intensities[i] = intensities[((bins[i] <= times) * (times <= bins[i + 1]))].mean()

        fig, ax = plt.subplots()
        ax.plot((bins[:-1] + bins[1:]) / 2, bin_intensities, marker='o', markersize=3, linestyle='-', color='b')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Avg intensity during {(t_max - t_min) / n_time_bins} time units')
        plt.savefig(fn)

    plot_intensity_per_time(
        fcdata=fcd, channel_key='FS INT', n_time_bins=200, fn='z_intensities_per_timeinterval_fsint.png', verbosity=1)
    plot_intensity_per_time(
        fcdata=fcd, channel_key='8-PB', n_time_bins=200, fn='z_intensities_per_timeinterval_8-PB.png', verbosity=1)

    # pm.pl.scatter_density(fcd, x='TIME', y='FS PEAK', y_scale='logit')
    # plt.savefig('z_time_vs_intensity_scatter.png')

    plt.close('all')

    fcd.layers['og'] = fcd.X.copy()
    pm.tl.normalize_logicle(fcd)
    # pm.tl.normalize_arcsinh(fcd, cofactor=150)

    channels = ['FS PEAK', 'FS INT', 'FS TOF', 'SS INT', '16-FITC', '56-PE', '3-ECD',
                '4-PC7', '19-APC', '14-APC700', 'FL8 INT', '8-PB', '45-CO']
    for channel in channels:
        x = fcd[:, 'TIME'].layers['og'].copy()
        # y = fcd[:, channel].X.copy()
        y = fcd[:, channel].layers['og'].copy()

        plt.scatter(x=x, y=y, linewidth=0, edgecolor='none', s=1, c=y)
        plt.xlabel('Time')
        plt.ylabel(f'Intensity {channel}')
        plt.savefig(f'z_time_vs_intensity_scatter_{channel}.png')
        plt.close()

    print(fcd.uns['meta'])

    # pm.pl.scatter_density(fcd, x='FS PEAK', y='FS INT', y_scale='logit')
    # plt.savefig('z_fstof_vs_fspeak_scatter.png')


def set_random_seed(seed: int = 42):
    import torch
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_deepcopy():
    import scanpy as sc
    adata = sc.datasets.blobs(n_variables=11, n_centers=5, n_observations=640)

    bdata = adata
    # bdata = copy.deepcopy(adata)

    bdata.obs['dummy'] = np.ones(bdata.n_obs)

    # bdata = bdata[0:100]

    print('# ### adata: ')
    print(adata)

    print('# ### bdata: ')
    print(bdata)


def speed_comparison_som():
    from flowsrc.flowdata import FlowDataManager
    from minisom import MiniSom
    from somoclu import Somoclu
    import os
    import multiprocessing
    import psutil
    import logging
    import time
    from datetime import datetime
    from typing import Union, Callable, Literal

    def minisom_clustering(
            x: np.ndarray,
            som_dim: tuple[int, int] = (10, 10),
            sigma: float = 3.0,  # Spread of the neighborhood function
            learning_rate: float = 0.5,
            lr_decay_function: Union[Literal['inverse_decay_to_zero', 'linear_decay_to_zero', 'asymptotic_decay'],
                                     Callable] = 'inverse_decay_to_zero',  # Function for learning rate decay
            neighborhood_function: Literal['gaussian', 'mexican_hat', 'bubble', 'triangle'] = 'gaussian',
            neighborhood_decay_function: Literal['inverse_decay_to_one', 'linear_decay_to_one',
                                                 'asymptotic_decay'] = 'inverse_decay_to_one',
            topology: Literal['rectangular', 'hexagonal'] = 'rectangular',
            similarity_function: Literal['euclidean', 'cosine', 'manhattan', 'chebyshev'] = 'euclidean',
            weight_init: Literal['random', 'pca'] = 'random',
            n_epochs: int = 10,
            seed: int = 42,
            verbose: bool = False,
            logger: Union[logging.Logger, None] = None,
    ) -> MiniSom:

        som = MiniSom(
            topology=topology,
            x=som_dim[0],
            y=som_dim[1],
            input_len=x.shape[1],
            activation_distance=similarity_function,
            sigma=sigma,
            learning_rate=learning_rate,
            decay_function=lr_decay_function,
            neighborhood_function=neighborhood_function,
            sigma_decay_function=neighborhood_decay_function,
            random_seed=seed,

        )
        if weight_init == "random":
            som.random_weights_init(x)
        elif weight_init == "pca":
            som.pca_weights_init(x)
        else:
            raise ValueError('Unknown weight_init, must be one of "random" or "pca"')
        # som.train_batch(x, batch_size, verbose=verbose)
        if logger is not None:
            logger.info(f'# ### Starting MiniSOM training at {datetime.now()}')
        st = time.time()
        som.train(
            data=x,
            num_iteration=n_epochs,
            random_order=True,
            verbose=verbose,
            use_epochs=True
        )
        et = time.time()
        if logger is not None:
            logger.info(f'# ### Ended MiniSOM training at {datetime.now()}')
            logger.info(f'# ### Elapsed time: {et - st}')
        return som

    # print(adata_flowsom.obs["flowsom_clusters"].value_counts())

    def somoclu_clustering(
            x: np.ndarray,
            som_dim: tuple[int, int] = (10, 10),
            sigma0: float = 3.0,  # Spread of the neighborhood function
            sigmaN: float = 1.0,
            learning_rate0: float = 0.01,
            learning_rateN: float = 0.001,
            lr_decay_function: Literal['linear', 'exponential'] = 'linear',
            neighborhood_function: Literal['gaussian', 'bubble'] = 'gaussian',
            neighborhood_decay_function: Literal['linear', 'exponential'] = 'linear',
            topology: Literal['rectangular', 'hexagonal'] = 'rectangular',
            similarity_function: Literal['euclidean', 'cosine', 'manhattan', 'chebyshev'] = 'euclidean',
            weight_init: Literal['random', 'pca'] = 'random',
            n_epochs: int = 10,
            verbose: int = 0,
            logger: Union[logging.Logger, None] = None,
    ):

        som = Somoclu(
            n_columns=som_dim[1],
            n_rows=som_dim[0],
            gridtype=topology,
            maptype='planar',
            neighborhood=neighborhood_function,
            std_coeff=sigma0,
            initialization=weight_init,
            verbose=verbose
        )

        if logger is not None:
            logger.info(f'# ### Starting Somoclu training at {datetime.now()}')
        st = time.time()
        som.train(
            data=x,
            epochs=n_epochs,
            radius0=sigma0,
            radiusN=sigmaN,
            radiuscooling=neighborhood_decay_function,
            scale0=learning_rate0,
            scaleN=learning_rateN,
            scalecooling=lr_decay_function,
        )
        et = time.time()
        if logger is not None:
            logger.info(f'# ### Ended Somoclu training at {datetime.now()}')
            logger.info(f'# ### Elapsed time: {et - st}')

        return som

    # ### Set up logger
    log = logging.getLogger('SOM_speed_logger')
    handler = logging.FileHandler(os.path.join(os.getcwd(), 'z_speed_comparison.log'), mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    # ### Limit number of threads that are opened during training
    # n_threads = 20
    # os.environ["OMP_NUM_THREADS"] = f'{n_threads}'  # Limit OpenMP to n threads
    # os.environ["MKL_NUM_THREADS"] = f'{n_threads}'  # Limit Intel MKL to n threads
    # os.environ["NUMEXPR_NUM_THREADS"] = f'{n_threads}'  # Limit NumExpr to n threads
    # os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}'  # Limit OpenBLAS to n threads
    # os.environ["GOTO_NUM_THREADS"] = f"{n_threads}"  # Limit GOTO BLAS to n threads
    # os.environ["BLIS_NUM_THREADS"] = f"{n_threads}"  # Limit BLIS to n threads

    # ### Limit the number of cores which are used for training
    print(f'# ### Number of available cores: {multiprocessing.cpu_count()}')
    p = psutil.Process(os.getpid())
    cpus = list(range(85, 95))
    # cpus = [0, 1]
    p.cpu_affinity(cpus)
    log.info(f'# ### Limited available cores for training to: {min(cpus)}-{max(cpus)}, total: {len(cpus)}')

    # ### Load and process training data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))
    # filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    memory_saving = False
    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=memory_saving, channel_name_reference=0, save_path='results/dummy',
        data_file_path='input/concatenated_labeled_fcs_format'
    )

    flow_manager.perform_data_split(data_split=(0.75, 0.25))

    dl = flow_manager.create_data_loader(
        data_set='train', channels=list(range(14)), layer_key=None, label_key='population',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='zdummy_data.npy')

    x_train, y_train = next(iter(dl))
    x_train = x_train[:1000, :]
    y_train = y_train[:1000]
    print(x_train)
    print(y_train)

    epochs = 1000

    log.info(f'# ### Training data dimensions: {x_train.shape}')

    trained_minisom = minisom_clustering(x=x_train, weight_init='pca', n_epochs=epochs, verbose=True, logger=log)
    print(trained_minisom)

    trained_somoclu = somoclu_clustering(x=x_train, weight_init='pca', n_epochs=epochs, verbose=2, logger=log)
    print(trained_somoclu)


def test_flowsom():
    import multiprocessing
    import os
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results, evaluate_som_results
    import matplotlib

    matplotlib.use('Agg')

    # ### Load and process .fcs data
    use_all = False
    if use_all:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
    else:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path='results/data_handling',
        filenames={'fn_og_channel_names': 'zzz_channel_names.csv', }
    )

    print('###')

    flow_manager.perform_data_split(data_split=(0.5, 0.25, 0.25))

    # ### Define channels to be used for training and testins the SOM and create a dataloader for the train-/test-data
    print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
    channels = ['FS PEAK', 'FS INT', 'FS TOF', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700',
                'FL8 INT', '8-PB', '45-CO']

    # Set flag if dummy data should be used (useful for developing purposes)
    dev = False

    train = True
    if train:
        # ### Create a dataloader for the train data and check the class balance in the train data
        dl_train = flow_manager.create_data_loader(
            data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='dummy_data_train.npy')
        FlowDataManager.check_class_balance(
            dl=dl_train, save=True, save_kwargs=None,
            plot=True, plot_kwargs={'filename': 'z_class_balance_train.png', 'filepath': 'results'})

        # Batch size was set to -1 => dl contains one array with all the data, extract it
        x_train, y_train = next(iter(dl_train))
        if dev:
            x_train = x_train[:1000, :]
            y_train = y_train[:1000]

        # ### Initialize SomClassifier
        print(f'# ### Number of available cores: {multiprocessing.cpu_count()}')
        cpus = list(range(85, 95))
        som_c = SomClassifier(cores=cpus, verbosity=2)

        # Start training and save the trained SOM classifier
        som_c.fit(X=x_train, y=y_train, n_epochs=100, radius_cooling='exponential')

        som_c.save(filepath=os.path.join(os.getcwd(), 'results'))
        # del som_c

    else:
        # ### Load a previously trained SOM classifier
        som_c = SomClassifier.load(filename='som_classifier.pkl',
                                   filepath=os.path.join(os.getcwd(), 'results'))

    # som_c.som.view_activation_map(data_index=0, filename='zz_activation_map_0.png')
    # som_c.som.view_activation_map(data_index=1, filename='zz_activation_map_1.png')
    # som_c.som.view_similarity_matrix(filename='zz_similarity_matrix.png')
    # som_c.som.view_component_planes(filename='zz_component_planes.png')
    # som_c.som.view_umatrix(filename='zz_umatrix.png')

    # ### Create a dataloader for the test data and check the class balance in the test data
    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=False, on_disk=True, filename='dummy_data_test.npy')

    FlowDataManager.check_class_balance(
        dl=dl_test, plot=True, plot_kwargs={'filename': 'z_class_balance_test.png', 'filepath': 'results'})

    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_test, y_test = next(iter(dl_test))
    if dev:
        x_test = x_test[:1000, :]
        y_test = y_test[:1000]

    # ### Predict labels for test data
    y_pred = som_c.predict(X=x_test)

    # ### If not in train mode load data used for training
    try:
        x_train, y_train
    except NameError:
        x_train = np.load(os.path.join(os.getcwd(), 'results/data_handling', 'dummy_data_train.npy'))
        y_train = x_train[:, x_train.shape[1] - 1]
        x_train = x_train[:, :x_train.shape[1] - 1]

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'random_state': 42},
        plot=True, additional_plot_kwargs={'show': False, 'save': True, 'filename': 'zzz.png', 'cmap': plt.cm.Blues},
        log_filename='eval_log.log'
    )

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, report_matrices=False, log_filename='eval_log.log')


def test_plot():
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_som_results, evaluate_classification_results, generate_som_plots_general, \
        generate_som_plots_x

    matplotlib.use('Agg')

    x_train = np.load(os.path.join(os.getcwd(), 'results/data_handling', 'dummy_data_train.npy'))
    y_train = x_train[:, x_train.shape[1] - 1]
    x_train = x_train[:, :x_train.shape[1] - 1]

    x_test = np.load(os.path.join(os.getcwd(), 'results/data_handling', 'dummy_data_test.npy'))
    y_test = x_test[:, x_test.shape[1] - 1]
    x_test = x_test[:, :x_test.shape[1] - 1]

    # ### Load a previously trained SOM classifier
    som_c = SomClassifier.load(filename='som_classifier.pkl',
                               filepath=os.path.join(os.getcwd(), 'results'))

    y_pred = som_c.predict(X=x_test)

    fp = os.path.join(os.getcwd(), 'results/dummy')
    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train)},
        plot=True, additional_plot_kwargs={'save': True, 'cmap': plt.cm.Blues},
        log_filename='eval_log.log',
        filepath=fp,
    )

    evaluate_som_results(som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename='eval_log.log', filepath=fp)

    generate_som_plots_general(som_classifier=som_c, filepath=fp)
    generate_som_plots_x(som_classifier=som_c, x=x_train, filename_prefix='train_', filepath=fp)
    generate_som_plots_x(som_classifier=som_c, x=x_test, filename_prefix='test_', filepath=fp)


def test_save_datasplit():
    import os
    from flowsrc.flowdata import FlowDataManager
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path='results/data_handling',
        filenames={'fn_og_channel_names': 'zzz_channel_names.csv', }
    )

    # split = (2/3, 1/3)
    split = (0.5, 0.25, 0.25)

    flow_manager.perform_data_split(data_split=split, save=True)
    del flow_manager

    split_df = pd.read_csv(os.path.join(os.getcwd(), 'results/data_handling/data_split.csv'), index_col=0)
    print(split_df)

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=True, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path='results/data_handling',
        filenames={'fn_og_channel_names': 'zzz_channel_names.csv', }
    )

    flow_manager.perform_data_split(data_split=split_df)

    print(f'# ### Train data:\n{flow_manager.train_data}')
    print(f'# ### Val data:\n{flow_manager.val_data}')
    print(f'# ### Test data:\n{flow_manager.test_data}')

    dl = flow_manager.create_data_loader(data_set='all', return_data_loader='np_array')
    print(dl)
    print(next(iter(dl)).shape)
    dl = flow_manager.create_data_loader(data_set='train', return_data_loader='np_array')
    print(dl)
    print(next(iter(dl)).shape)


def generate_baseline_results():
    import multiprocessing
    import os
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results, evaluate_som_results, generate_som_plots_general, \
        generate_som_plots_x
    import matplotlib

    matplotlib.use('Agg')

    # Create list of .fcs data file names
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    # Instantiate FlowDataManager
    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path='results/baseline/data_handling'
    )

    # Perform data split into train- and test-set
    split = (0.6, 0.4)
    flow_manager.perform_data_split(
        data_split=split, save=True, additional_split_kwargs={'shuffle': True})

    # Define channels to be used for training and testing the SOM
    print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
    channels = ['FS PEAK', 'FS INT', 'FS TOF', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700',
                'FL8 INT', '8-PB', '45-CO']

    # Create dataloader for train- and test-data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_test.npy')

    # Check the class balance in the train- and test-set
    FlowDataManager.check_class_balance(
        dl=dl_train,
        save=True, save_kwargs={'filename': 'class_balance_train.csv', 'filepath': 'results/baseline'},
        plot=True, plot_kwargs={'filename': 'class_balance_train.png', 'filepath': 'results/baseline'})
    FlowDataManager.check_class_balance(
        dl=dl_test,
        save=True, save_kwargs={'filename': 'class_balance_test.csv', 'filepath': 'results/baseline'},
        plot=True, plot_kwargs={'filename': 'class_balance_test.png', 'filepath': 'results/baseline'})

    # Batch size was set to -1 => dataloaders contain one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))
    x_test, y_test = next(iter(dl_test))

    # ### Initialize SomClassifier and train
    # Set parameters to default settings except for the number of cores used during training
    print(f'# ### Number of available cores: {multiprocessing.cpu_count()}')
    cpus = list(range(85, 95))
    som_c = SomClassifier(cores=cpus, verbosity=2)

    # Start training and save the trained SOM classifier
    n_epochs = 1000
    som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
    som_c.save(filename=f'som_{n_epochs}.pkl', filepath=os.path.join(os.getcwd(), 'results/baseline'))

    # ### Evaluate the performance on the test-set
    # Predict labels with the trained SomClassifier
    y_pred = som_c.predict(X=x_test)

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
        plot=True,
        additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
        log_filename=f'eval_{n_epochs}.log', filepath=os.path.join(os.getcwd(), 'results/baseline'))

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log',
        filepath=os.path.join(os.getcwd(), 'results/baseline'))

    generate_som_plots_x(
        som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath='results/baseline')
    generate_som_plots_x(
        som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath='results/baseline')
    generate_som_plots_general(
        som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath='results/baseline')

    # ### Reset SomClassifier and train and save again ### #
    som_c.reset()
    n_epochs = 10000
    som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
    som_c.save(filename=f'som_{n_epochs}.pkl', filepath=os.path.join(os.getcwd(), 'results/baseline'))

    # ### Evaluate the performance on the test-set
    # Predict labels with the trained SomClassifier
    y_pred = som_c.predict(X=x_test)

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
        plot=True,
        additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
        log_filename=f'eval_{n_epochs}.log', filepath=os.path.join(os.getcwd(), 'results/baseline'))

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log',
        filepath=os.path.join(os.getcwd(), 'results/baseline'))

    generate_som_plots_x(
        som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath='results/baseline')
    generate_som_plots_x(
        som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath='results/baseline')
    generate_som_plots_general(
        som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath='results/baseline')


def test_sampling_strategies():
    import os
    from typing import Tuple
    class_balance = pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/class_balance_train.csv'), index_col=0)
    print(class_balance)
    count = class_balance.loc['count'].to_numpy()
    print(count)
    print(count.mean())
    print('###')
    median = np.median(count)
    print(f'Median: {median}')
    mad = np.median(np.abs(count - np.median(count)))
    print(f'MAD: {mad}')
    print(f'Median + MAD: {median + mad}')
    print(f'Median - MAD: {median - mad}')

    labels = np.random.randint(low=0, high=8, size=(100, ))
    print(labels)

    from flowsrc.flowdata import FlowDataManager

    us, ds = FlowDataManager._create_sampling_strategies(y=labels)

    print(us)
    print(ds)


def train_balanced():
    import os
    import pickle
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results, evaluate_som_results, generate_som_plots_general, \
        generate_som_plots_x
    import matplotlib

    matplotlib.use('Agg')

    train = True
    if train:
        # ### Load and process .fcs data
        use_all = True
        if use_all:
            filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
        else:
            filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

        flow_manager = FlowDataManager(
            filename_list=filename_list, preprocessing_flavour='arcsinh',
            additional_preprocessing_kwargs={'cofactor': 150},
            memory_saving=False, channel_name_reference=0,
            data_file_path='input/concatenated_labeled_fcs_format', save_path='results/balanced/data_handling',
        )

        # Load data split used for baseline results
        flow_manager.perform_data_split(
            data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                                   index_col=0))

        # Define channels to be used for training and testing the SOM
        print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
        channels = ['FS PEAK', 'FS INT', 'FS TOF', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC',
                    '14-APC700',
                    'FL8 INT', '8-PB', '45-CO']

        # Define filepath for saving results
        fp = os.path.join(os.getcwd(), 'results/balanced')

        # Create dataloader for train- and test-data
        # Unbalanced train loader just for comparison
        dl_train = flow_manager.create_data_loader(
            data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
        FlowDataManager.check_class_balance(
            dl=dl_train,
            save=True, save_kwargs={'filename': 'class_balance_train.csv', 'filepath': fp},
            plot=True, plot_kwargs={'filename': 'class_balance_train.png', 'filepath': fp})

        dl_train_balanced = flow_manager.create_data_loader(
            data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            balance=True, return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True,
            filename='data_train_balanced.npy')
        FlowDataManager.check_class_balance(
            dl=dl_train_balanced,
            save=True, save_kwargs={'filename': 'class_balance_train_balanced.csv', 'filepath': fp},
            plot=True, plot_kwargs={'filename': 'class_balance_train_balanced.png', 'filepath': fp})

        dl_test = flow_manager.create_data_loader(
            data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_val.npy')

        # Batch size was set to -1 => dataloaders contain one array with all the data, extract it
        x_train, y_train = next(iter(dl_train_balanced))
        x_test, y_test = next(iter(dl_test))

        # ### Initialize SomClassifier and train
        # Set parameters to default settings except for the number of cores used during training
        cpus = list(range(85, 95))
        som_c = SomClassifier(cores=cpus, verbosity=2)

        # Start training and save the trained SOM classifier
        n_epochs = 1000
        som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
        som_c.save(filename=f'som_{n_epochs}.pkl', filepath=fp)

        # ### Evaluate the performance on the test-set
        # Predict labels with the trained SomClassifier
        y_pred = som_c.predict(X=x_test)

        evaluate_classification_results(
            y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
            compare_to_dummy_classifier=True,
            additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
            plot=True,
            additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
            log_filename=f'eval_{n_epochs}.log', filepath=fp)

        plt.close('all')

        evaluate_som_results(
            som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log', filepath=fp)

        generate_som_plots_x(
            som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath=fp)
        generate_som_plots_x(
            som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath=fp)
        generate_som_plots_general(
            som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath=fp)

        # ### Reset SomClassifier and train and save again ### #
        som_c.reset()
        n_epochs = 10000
        som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
        som_c.save(filename=f'som_{n_epochs}.pkl', filepath=fp)

        # ### Evaluate the performance on the test-set
        # Predict labels with the trained SomClassifier
        y_pred = som_c.predict(X=x_test)

        evaluate_classification_results(
            y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
            compare_to_dummy_classifier=True,
            additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
            plot=True,
            additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
            log_filename=f'eval_{n_epochs}.log', filepath=fp)

        plt.close('all')

        evaluate_som_results(
            som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log',
            filepath=fp)

        generate_som_plots_x(
            som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath=fp)
        generate_som_plots_x(
            som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath=fp)
        generate_som_plots_general(
            som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath=fp)


def generate_baseline_results_less_channels():
    import multiprocessing
    import os
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results, evaluate_som_results, generate_som_plots_general, \
        generate_som_plots_x
    import matplotlib

    matplotlib.use('Agg')

    # Create list of .fcs data file names
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    # Instantiate FlowDataManager
    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path='results/baseline_less_channels/data_handling'
    )

    # Perform data split into train- and test-set
    # Load data split used for baseline results
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # Define channels to be used for training and testing the SOM
    print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # Create dataloader for train- and test-data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_test.npy')

    fp = os.path.join(os.getcwd(), 'results/baseline_less_channels')

    # Check the class balance in the train- and test-set
    FlowDataManager.check_class_balance(
        dl=dl_train,
        save=True, save_kwargs={'filename': 'class_balance_train.csv', 'filepath': fp},
        plot=True, plot_kwargs={'filename': 'class_balance_train.png', 'filepath': fp})
    FlowDataManager.check_class_balance(
        dl=dl_test,
        save=True, save_kwargs={'filename': 'class_balance_test.csv', 'filepath': fp},
        plot=True, plot_kwargs={'filename': 'class_balance_test.png', 'filepath': fp})

    # Batch size was set to -1 => dataloaders contain one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))
    x_test, y_test = next(iter(dl_test))

    # ### Initialize SomClassifier and train
    # Set parameters to default settings except for the number of cores used during training
    print(f'# ### Number of available cores: {multiprocessing.cpu_count()}')
    cpus = list(range(85, 95))
    som_c = SomClassifier(cores=cpus, verbosity=2)

    # Start training and save the trained SOM classifier
    n_epochs = 1000
    som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
    som_c.save(filename=f'som_{n_epochs}.pkl', filepath=os.path.join(os.getcwd(), fp))

    # ### Evaluate the performance on the test-set
    # Predict labels with the trained SomClassifier
    y_pred = som_c.predict(X=x_test)

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
        plot=True,
        additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
        log_filename=f'eval_{n_epochs}.log', filepath=os.path.join(os.getcwd(), fp))

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log',
        filepath=os.path.join(os.getcwd(), fp))

    generate_som_plots_x(
        som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath=fp)
    generate_som_plots_x(
        som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath=fp)
    generate_som_plots_general(
        som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath=fp)

    # ### Reset SomClassifier and train and save again ### #
    som_c.reset()
    n_epochs = 10000
    som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
    som_c.save(filename=f'som_{n_epochs}.pkl', filepath=os.path.join(os.getcwd(), fp))

    # ### Evaluate the performance on the test-set
    # Predict labels with the trained SomClassifier
    y_pred = som_c.predict(X=x_test)

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
        plot=True,
        additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
        log_filename=f'eval_{n_epochs}.log', filepath=os.path.join(os.getcwd(), fp))

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log',
        filepath=os.path.join(os.getcwd(), fp))

    generate_som_plots_x(
        som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath=fp)
    generate_som_plots_x(
        som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath=fp)
    generate_som_plots_general(
        som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath=fp)


def train_balanced_less_channels():
    import os
    import pickle
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results, evaluate_som_results, generate_som_plots_general, \
        generate_som_plots_x
    import matplotlib

    matplotlib.use('Agg')

    # ### Load and process .fcs data
    use_all = True
    if use_all:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
    else:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh',
        additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format',
        save_path='results/balanced_less_channels/data_handling',
    )

    # Load data split used for baseline results
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # Define channels to be used for training and testing the SOM
    print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # Create dataloader for train- and test-data
    # Unbalanced train loader just for comparison
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')

    # Define filepath for saving results
    fp = os.path.join(os.getcwd(), 'results/balanced_less_channels')

    FlowDataManager.check_class_balance(
        dl=dl_train,
        save=True, save_kwargs={'filename': 'class_balance_train.csv', 'filepath': fp},
        plot=True, plot_kwargs={'filename': 'class_balance_train.png', 'filepath': fp})

    dl_train_balanced = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        balance=True, return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True,
        filename='data_train_balanced.npy')
    FlowDataManager.check_class_balance(
        dl=dl_train_balanced,
        save=True, save_kwargs={'filename': 'class_balance_train_balanced.csv', 'filepath': fp},
        plot=True, plot_kwargs={'filename': 'class_balance_train_balanced.png', 'filepath': fp})

    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_test.npy')

    # Batch size was set to -1 => dataloaders contain one array with all the data, extract it
    x_train, y_train = next(iter(dl_train_balanced))
    x_test, y_test = next(iter(dl_test))

    # ### Initialize SomClassifier and train
    # Set parameters to default settings except for the number of cores used during training
    cpus = list(range(76, 85))
    som_c = SomClassifier(cores=cpus, verbosity=2)

    # Start training and save the trained SOM classifier
    n_epochs = 1000
    som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
    som_c.save(filename=f'som_{n_epochs}.pkl', filepath=fp)

    # ### Evaluate the performance on the test-set
    # Predict labels with the trained SomClassifier
    y_pred = som_c.predict(X=x_test)

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
        plot=True,
        additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
        log_filename=f'eval_{n_epochs}.log', filepath=fp)

    plt.close('all')

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log', filepath=fp)

    generate_som_plots_x(
        som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath=fp)
    generate_som_plots_x(
        som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath=fp)
    generate_som_plots_general(
        som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath=fp)

    # ### Reset SomClassifier and train and save again ### #
    som_c.reset()
    n_epochs = 10000
    som_c.fit(X=x_train, y=y_train, n_epochs=n_epochs)
    som_c.save(filename=f'som_{n_epochs}.pkl', filepath=fp)

    # ### Evaluate the performance on the test-set
    # Predict labels with the trained SomClassifier
    y_pred = som_c.predict(X=x_test)

    evaluate_classification_results(
        y_true=y_test, y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
        compare_to_dummy_classifier=True,
        additional_dummy_classifier_kwargs={'train_data': (x_train, y_train), 'strategy': 'stratified'},
        plot=True,
        additional_plot_kwargs={'save': True, 'filename': f'confusion_matrix_{n_epochs}.png'},
        log_filename=f'eval_{n_epochs}.log', filepath=fp)

    plt.close('all')

    evaluate_som_results(
        som_classifier=som_c, x_train=x_train, x_test=x_test, log_filename=f'eval_{n_epochs}.log',
        filepath=fp)

    generate_som_plots_x(
        som_classifier=som_c, x=x_train, filename_prefix=f'train_{n_epochs}_', filepath=fp)
    generate_som_plots_x(
        som_classifier=som_c, x=x_test, filename_prefix=f'test_{n_epochs}_', filepath=fp)
    generate_som_plots_general(
        som_classifier=som_c, filename_prefix=f'{n_epochs}_', filepath=fp)


def test_predict_proba():
    from flowsrc.flowsom import SomClassifier
    from flowsrc.flowdata import FlowDataManager
    import os

    som_c = SomClassifier.load('som_10000.pkl', os.path.join(os.getcwd(), 'results/balanced_less_channels'))

    # ### Load and process .fcs data
    use_all = False
    if use_all:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
    else:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path='results/dummy/data_handling',
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    flow_manager.perform_data_split(data_split=(0.6, 0.4))

    # ### Define channels to be used for training and testing the SOM and create a dataloader for the train-/test-data
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # ### Create a dataloader for the train data and check the class balance in the train data
    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')

    # Batch size was set to -1 => dl contains one array with all the data, extract it
    # Set flag if dummy data should be used (useful for developing purposes)
    dev = True
    x_test, y_test = next(iter(dl_test))
    if dev:
        x_test = x_test[1000:1010, :]
        y_test = y_test[1000:1010]

    y_pred = som_c.predict(X=x_test)
    y_pred_proba = som_c.predict_proba(X=x_test)
    print(f'# ### y_test:\n{y_test}')
    print(f'# ### y_pred:\n{y_pred}')
    print(f'# ### y_pred_proba:\n{np.round(y_pred_proba, 3)}')


def train_epoch_wise():
    import os
    import pickle
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_1d_metric_over_time, plot_nd_metric_over_time, plot_som_pies
    import matplotlib

    matplotlib.use('Agg')

    # n_epochs = 14
    # tracking_interval = 4
    # save_intermediate_interval = 8
    n_epochs = 10000
    tracking_interval = 4
    save_intermediate_interval = 120
    # Per track and save takes ~ 30s, 12 * 60 * 60 s / 30 s = 43200 / 30 = 1440
    # 10000 / 1440 ~= 6.94 => can afford tracking frequency of 6 if we want to finnish within 12h
    # n_epochs = 10000
    # tracking_interval = 6
    # save_intermediate_interval = 120
    fp = os.path.join(os.getcwd(), f'results/epoch_wise_{n_epochs}')
    if not os.path.exists(fp):
        os.makedirs(fp)
        os.makedirs(os.path.join(fp, 'data_handling'))

    train = False
    if train:
        # ### Load and process .fcs data
        use_all = True
        if use_all:
            filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
        else:
            filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

        flow_manager = FlowDataManager(
            filename_list=filename_list, preprocessing_flavour='arcsinh',
            additional_preprocessing_kwargs={'cofactor': 150}, memory_saving=False, channel_name_reference=0,
            data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(fp, 'data_handling'),
        )

        # Load data split used for baseline results
        flow_manager.perform_data_split(
            data_split=pd.read_csv(os.path.join(os.getcwd(),
                                                'results/baseline/data_handling/data_split.csv'), index_col=0))
        # flow_manager.perform_data_split(data_split=(0.75, 0.25))

        # Define channels to be used for training and testing the SOM
        print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
        channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

        # Create dataloader for train- and test-data
        dl_train = flow_manager.create_data_loader(
            data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=False, filename='data_train.npy')
        dl_val = flow_manager.create_data_loader(
            data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=False, filename='data_val.npy')

        som_c = SomClassifier(verbosity=2)

        x_train, y_train = next(iter(dl_train))
        x_val, y_val = next(iter(dl_val))

        som_c.fit_epoch_wise(
            X=x_train, y=y_train, n_epochs=n_epochs, tracking_interval=tracking_interval, track_losses=False,
            track_impurities=True, save_intermediate_interval=save_intermediate_interval, val_data=(x_val, y_val),
            save_res_dict=True, filepath=fp)

        som_c.save(filepath=fp)

    # with open(os.path.join(fp, 'res_dict.pkl'), 'rb') as f:
    with open(os.path.join(fp, 'res_dict.pkl'), 'rb') as f:
        rd = pickle.load(f)
    som_c = SomClassifier.load(filepath=fp)

    plot_1d_metric_over_time(
        metric_train=rd['entropy'].mean(axis=(0, 1)), n_epochs=rd['n_epochs'],
        tracking_interval=rd['tracking_interval'], ylabel='Mean entropy', filename='entropy.png', filepath=fp)

    plot_nd_metric_over_time(
        metrics=rd['f1_train'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'], title='F1 train',
        ylabel='F1-score',  # metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
        filename='f1_train.png', filepath=fp)

    plot_nd_metric_over_time(
        metrics=rd['f1_val'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'], title='F1 val',
        ylabel='F1-score',  # metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
        filename='f1_val.png', filepath=fp)

    plot_som_pies(
        class_counts_per_unit=som_c.class_counts_per_unit_, som_dimensions=som_c.som_dimensions,
        label_mapping=som_c.new_to_og_classes_dict_)
    plt.savefig(os.path.join(fp, f'som_pies_{n_epochs}.png'))


def train_epoch_wise_balanced():
    import os
    import pickle
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_1d_metric_over_time, plot_nd_metric_over_time, plot_som_pies
    import matplotlib

    matplotlib.use('Agg')

    # n_epochs = 14
    # tracking_interval = 4
    # save_intermediate_interval = 8
    n_epochs = 10000
    tracking_interval = 4
    save_intermediate_interval = 120
    # Per track and save takes ~ 30s, 12 * 60 * 60 s / 30 s = 43200 / 30 = 1440
    # 10000 / 1440 ~= 6.94 => can afford tracking frequency of 6 if we want to finnish within 12h
    # n_epochs = 10000
    # tracking_interval = 6
    # save_intermediate_interval = 120
    fp = os.path.join(os.getcwd(), f'results/epoch_wise_balanced_{n_epochs}')
    if not os.path.exists(fp):
        os.makedirs(fp)
        os.makedirs(os.path.join(fp, 'data_handling'))

    train = False
    if train:
        # ### Load and process .fcs data
        use_all = True
        if use_all:
            filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
        else:
            filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

        flow_manager = FlowDataManager(
            filename_list=filename_list, preprocessing_flavour='arcsinh',
            additional_preprocessing_kwargs={'cofactor': 150}, memory_saving=False, channel_name_reference=0,
            data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(fp, 'data_handling'),
        )

        # Load data split used for baseline results
        flow_manager.perform_data_split(
            data_split=pd.read_csv(os.path.join(os.getcwd(),
                                                'results/baseline/data_handling/data_split.csv'), index_col=0))
        # flow_manager.perform_data_split(data_split=(0.75, 0.25))

        # Define channels to be used for training and testing the SOM
        print(f'# ### Channels: {flow_manager.anndata_list[0].var_names.to_list()}')
        channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

        # Create dataloader for train- and test-data
        dl_train_balanced = flow_manager.create_data_loader(
            data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            balance=True, return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True,
            filename='data_train_balanced.npy')
        dl_val = flow_manager.create_data_loader(
            data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
            return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=False, filename='data_val.npy')

        som_c = SomClassifier(verbosity=2)

        x_train, y_train = next(iter(dl_train_balanced))
        x_val, y_val = next(iter(dl_val))

        som_c.fit_epoch_wise(
            X=x_train, y=y_train, n_epochs=n_epochs, tracking_interval=tracking_interval, track_losses=False,
            track_impurities=True, save_intermediate_interval=save_intermediate_interval, val_data=(x_val, y_val),
            save_res_dict=True, filepath=fp)

        som_c.save(filepath=fp)

    with open(os.path.join(fp, 'res_dict.pkl'), 'rb') as f:
        rd = pickle.load(f)
    som_c = SomClassifier.load(filepath=fp)

    plot_1d_metric_over_time(
        metric_train=rd['entropy'].mean(axis=(0, 1)), n_epochs=rd['n_epochs'],
        tracking_interval=rd['tracking_interval'], ylabel='Mean entropy', filename='entropy.png', filepath=fp)

    plot_nd_metric_over_time(
        metrics=rd['f1_train'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'], title='F1 train',
        ylabel='F1-score', metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
        filename='f1_train.png', filepath=fp)

    plot_nd_metric_over_time(
        metrics=rd['f1_val'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'], title='F1 val',
        ylabel='F1-score', metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
        filename='f1_val.png', filepath=fp)

    plot_som_pies(
        class_counts_per_unit=som_c.class_counts_per_unit_, som_dimensions=som_c.som_dimensions,
        label_mapping=som_c.new_to_og_classes_dict_)
    plt.savefig(os.path.join(fp, f'som_pies_{n_epochs}.png'))


def eval_sample_wise():
    import matplotlib
    from flowsrc.floweval import evaluate_sample_wise, plot_sample_wise_evaluation
    from flowsrc.flowsom import SomClassifier
    from flowsrc.flowdata import FlowDataManager

    matplotlib.use('Agg')

    base_fp = os.path.join(os.getcwd(), f'results')
    if not os.path.exists(os.path.join(base_fp, 'sample_wise_eval')):
        os.makedirs(os.path.join(base_fp, 'sample_wise_eval'))

    # Create list of .fcs data file names
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    # Instantiate FlowDataManager
    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh',
        additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0, data_file_path='input/concatenated_labeled_fcs_format',
    )

    # Perform data split into train- and test-set
    # Load data split used for baseline results
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # Define channels on which SOM classifier was trained on
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    som_filepaths = ['baseline_less_channels']
    som_filenames = ['som_10000.pkl']

    # som_filepaths = ['hyperparameter_tuning_all_fixed_but_one/dim']
    # som_filenames = ['som_classifier.pkl']

    for som_fp, som_fn in zip(som_filepaths, som_filenames):
        if not os.path.exists(os.path.join(base_fp, 'sample_wise_eval', som_fp)):
            os.makedirs(os.path.join(base_fp, 'sample_wise_eval', som_fp))
        # Load SOM classifier
        som_c = SomClassifier.load(
            filename=som_fn,
            filepath=os.path.join(base_fp, som_fp))

        res_df = evaluate_sample_wise(
            som_c=som_c, flow_manager=flow_manager, mode='test',
            create_data_loader_kwargs={
                'channels': channels,
                'layer_key': None,
                'label_key': 'population',
                'label_layer_key': 'original',
                'return_data_loader': 'np_array',
                'on_disk': False,
            },
            log_filename='eval_sample_wise.log',
            filepath=os.path.join(base_fp, 'sample_wise_eval', som_fp),
            save_res_df=True
        )

        samples_few_events = np.where(res_df['# 1.0'].to_numpy() <= 10)[0]
        samples_low_f1 = np.where(res_df['F1 1.0'].to_numpy() <= 0.2)[0]
        print(f'# ### Samples with few class 1 events:\n{samples_few_events}')
        print(f'# ### Samples with a low F1 score for class 1:\n{samples_low_f1}')

        fs = 10
        # Accuracy = Precision micro = Recall micro = F1 micro
        plot_sample_wise_evaluation(res_df=res_df[[
            'Accuracy',
            'Precision macro', 'Precision weighted']], fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'precision.png'), dpi=300)

        plot_sample_wise_evaluation(res_df=res_df[[
            'Accuracy',
            'Recall macro', 'Recall weighted']], fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'recall.png'), dpi=300)

        plot_sample_wise_evaluation(res_df=res_df[[
            'Accuracy',
            'F1 macro', 'F1 weighted']], fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'f1.png'), dpi=300)

        plot_sample_wise_evaluation(res_df=res_df[[f'F1 {cl}' for cl in som_c.og_classes_]], fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'f1_class_wise.png'), dpi=300)

        plot_sample_wise_evaluation(res_df=res_df[[f'Prec {cl}' for cl in som_c.og_classes_]], fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'precision_class_wise.png'), dpi=300)

        plot_sample_wise_evaluation(res_df=res_df[[f'Rec {cl}' for cl in som_c.og_classes_]], fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'recall_class_wise.png'), dpi=300)

        plot_sample_wise_evaluation(
            res_df=np.log10(res_df[[f'# {cl}' for cl in som_c.og_classes_]] + 1), ylabel='log10(1 + # events)',
            fontsize=fs)
        plt.savefig(os.path.join(base_fp, 'sample_wise_eval', som_fp, 'n_events_class_wise.png'), dpi=300)


def param_influence_study():
    import os
    from sklearn.model_selection import train_test_split
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import (
        evaluate_classification_results, plot_nd_metric_over_time, plot_1d_metric_over_time, plot_som_pies
    )
    import matplotlib

    matplotlib.use('Agg')

    base_p = os.path.join(os.getcwd(), 'results/param_influence_study')
    if not os.path.exists(base_p):
        os.makedirs(base_p)
        os.makedirs(os.path.join(base_p, 'data_handling'))

    # ### Load and process .fcs data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # Perform data split into train- and test-set, load data split used for baseline results (60, 40)
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # ### Define channels to be used for training and testing the SOM and create a dataloader for the train-/test-data
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # ### Create a dataloader for the train data and check the class balance in the train data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))

    downsample = True
    if downsample:
        # ### Downsample train set to half its size in a stratified fashion for this study
        # Calculate the original class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_count = y_train.shape[0]
        class_ratios = class_counts / total_count
        # print(class_ratios)
        # print(class_counts)
        # print(x_train.shape)
        # Calculate the desired count for each class
        ds_fraction = 0.25
        target_size = np.ceil(total_count * ds_fraction)
        target_counts = (class_ratios * target_size).round().astype(int)
        # Downsample
        downsampled_x = []
        downsampled_y = []
        for cls, count in zip(unique_classes, target_counts):
            # Get indices of the current class
            class_indices = np.where(y_train == cls)[0]
            # Randomly sample from these indices
            sampled_indices = np.random.choice(class_indices, size=count, replace=False)
            # Append downsampled data and labels
            downsampled_x.append(x_train[sampled_indices])
            downsampled_y.append(y_train[sampled_indices])
        # Concatenate results
        x_train = np.vstack(downsampled_x)
        y_train = np.concatenate(downsampled_y)

    # unique_classes, class_counts = np.unique(y_train, return_counts=True)
    # total_count = y_train.shape[0]
    # class_ratios = class_counts / total_count
    # print(class_ratios)
    # print(class_counts)
    # print(x_train.shape)

    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_test.npy')
    x_test, y_test = next(iter(dl_test))

    param_grid_nepochs = {
        'n_epochs': [10, 20, 30, 40, 50, 60, 70, 80, 90],  # [22000, 24000, 26000, 28000, 30000],
        # [12000, 14000, 16000, 18000, 20000],  # [3000, 4000, 6000, 7000, 8000, 9000],  # [300, 400, 500, 600, 700, 800, 900],  # [200, 3000, 11000],  # [100, 1000, 2000, 5000, 10000, 15000],
    }

    # Based on previous findings, selected n_epochs such that performance is stable with default parameters
    n_epochs = 2000

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

    param_grid_neigh_r0 = {
        'radius_0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        # 0.0 => min(n_columns, n_rows)/2, rn_default = 1.0
        'n_epochs': [n_epochs, ],
    }

    param_grid_neigh_rn = {
        'radius_n': [4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
        # r0_default = min(n_columns, n_rows)/2 = 5
        'n_epochs': [n_epochs, ],
    }

    param_grid_neigh_rcooling = {
        'radius_cooling': ['linear', 'exponential'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_lr_lr0 = {
        'learning_rate_0': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0],
        # lr_n = 0.01 in default setting
        'n_epochs': [n_epochs, ],
    }

    param_grid_lr_lrn = {
        'learning_rate_n': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001],
        # lr_0 = 0.1 in default setting
        'n_epochs': [n_epochs, ],
    }

    param_grid_lr_lrdecay = {
        'learning_rate_decay': ['linear', 'exponential'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_topology = {
        'som_topology': ['planar', 'toroid'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_gridtype = {
        'som_grid_type': ['rectangular', 'hexagonal'],
        'n_epochs': [n_epochs, ],
    }

    set00 = False
    set0 = False
    set1 = False
    set2 = False
    set3 = False
    set4 = False
    set5 = False
    set6 = False

    if set00:
        # Parameters for fit_epoch_wise()
        n_epochs = 600
        tracking_interval = 8
        save_intermediate_interval = None
        grids = [None, ]
        grid_names = [f'epoch_wise_{n_epochs}', ]
    elif set0:
        grids = [param_grid_nepochs]
        grid_names = ['nepochs_06', ]
    elif set1:
        grids = [param_grid_dimension, ]
        grid_names = ['dim', ]
    elif set2:
        grids = [param_grid_neigh_fct, param_grid_neigh_sigma]
        grid_names = ['neigh_fct', 'sigma']
    elif set3:
        grids = [param_grid_neigh_r0, param_grid_neigh_rn]
        grid_names = ['neigh_r0', 'neigh_rn']
    elif set4:
        grids = [param_grid_neigh_rcooling, param_grid_lr_lr0]
        grid_names = ['rcooling', 'lr0']
    elif set5:
        grids = [param_grid_lr_lrn, param_grid_lr_lrdecay]
        grid_names = ['lrn', 'lrdecay']
    elif set6:
        grids = [param_grid_topology, param_grid_gridtype]
        grid_names = ['topology', 'gridtype']
    else:
        grids = [{'n_epochs': [n_epochs, ], }, ]
        grid_names = ['dummy', ]

    # Set number of splits to 3
    cv = 3

    for grid, grid_name in zip(grids, grid_names):

        if not os.path.exists(os.path.join(base_p, grid_name)):
            os.makedirs(os.path.join(base_p, grid_name))

        som_c = SomClassifier(verbosity=2)

        if set00:
            # ### Train model epoch wise
            # Split into val and train set
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
            # Define filepath for saving
            fp = os.path.join(base_p, grid_name)
            # Train SOM epoch wise
            som_c.fit_epoch_wise(
                X=x_train, y=y_train, n_epochs=n_epochs, tracking_interval=tracking_interval, track_losses=False,
                track_impurities=True, save_intermediate_interval=save_intermediate_interval, val_data=(x_val, y_val),
                save_res_dict=True, filepath=fp)
            som_c.save(filepath=fp)
            # Plot results
            rd = som_c.epoch_wise_som_training_metrics_
            plot_1d_metric_over_time(
                metric_train=rd['entropy'].mean(axis=(0, 1)), n_epochs=rd['n_epochs'],
                tracking_interval=rd['tracking_interval'], ylabel='Mean entropy', filename='entropy.png', filepath=fp)
            plt.close('all')
            plot_nd_metric_over_time(
                metrics=rd['f1_train'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'],
                title='F1 train',
                ylabel='F1-score',
                # metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
                filename='f1_train.png', filepath=fp)
            plt.close('all')
            plot_nd_metric_over_time(
                metrics=rd['f1_val'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'],
                title='F1 val', ylabel='F1-score',
                # metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
                filename='f1_val.png', filepath=fp)
            plt.close('all')
            plot_som_pies(
                class_counts_per_unit=som_c.class_counts_per_unit_, som_dimensions=som_c.som_dimensions,
                label_mapping=som_c.new_to_og_classes_dict_)
            plt.savefig(os.path.join(fp, f'som_pies_{n_epochs}.png'))
            plt.close('all')
        else:
            # ### Perform cross-validated grid-search
            som_c.hyperparameter_tuning(
                X=x_train.copy(), y=y_train.copy(), param_grid=grid, cv=cv, scoring='internal', refit=True,
                # gridsearchcv_kwargs={'n_jobs': -1},
            )
            # Evaluate performance of best found model on test-set
            y_pred = som_c.predict(X=x_test.copy())
            evaluate_classification_results(
                y_true=y_test.copy(), y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
                compare_to_dummy_classifier=False,
                filepath=os.path.join(base_p, grid_name))
            # Save SOM classifier with results
            som_c.save(filepath=os.path.join(base_p, grid_name))


def view_results_param_influence_study_nepochs():
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_lineplot, plot_param_stripplot
    import matplotlib
    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/param_influence_study/')
    if not os.path.exists(os.path.join(base_p, 'nepochs')):
        os.makedirs(os.path.join(base_p, 'nepochs'))

    res_ps = ['nepochs_00', 'nepochs_01', 'nepochs_02', 'nepochs_03', 'nepochs_04', 'nepochs_05', 'nepochs_06']

    res_dfs = [pd.DataFrame()] * len(res_ps)
    for i, p in enumerate(res_ps):
        som_c = SomClassifier.load(
            os.path.join(base_p, p, 'som_classifier.pkl'))
        res_dfs[i] = pd.DataFrame(som_c.grid_search_.cv_results_)

    res_df = pd.concat(res_dfs, ignore_index=True)
    res_df['rank_test_score'] = res_df['mean_test_score'].rank(ascending=False, method='dense').astype(int)
    res_df.sort_values(by='param_n_epochs', axis=0, ascending=True, inplace=True, ignore_index=True)

    res_df.to_csv(os.path.join(base_p, 'nepochs', 'res_df.csv'))
    print(res_df)

    plot_param_lineplot(
        res_df=res_df, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
        markersize=3.0, x_axis_grid=True, dpi=300)
    plt.savefig(os.path.join(base_p, 'nepochs', 'nepochs.png'), dpi=300)


def view_results_param_influence_study():
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import (
        plot_param_lineplot, plot_param_stripplot, plot_som_pies, plot_support_hists, plot_support_hist_w_class_perc
    )
    import matplotlib
    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/param_influence_study/')

    val_var = 'mean_test_score'
    res_ps_categorical = ['topology', 'gridtype', 'dim', 'lrdecay', 'neigh_fct', 'rcooling']
    id_vars_categorical = [
        'param_som_topology', 'param_som_grid_type', 'param_som_dimensions', 'param_learning_rate_decay',
        'param_neighborhood', 'param_radius_cooling'
    ]
    for i, p in enumerate(res_ps_categorical):
        som_c = SomClassifier.load(
            os.path.join(base_p, p, 'som_classifier.pkl'))
        res_df = pd.DataFrame(som_c.grid_search_.cv_results_)
        print(f'# ###### {p} ###### #\n{res_df}')
        res_df = res_df[[id_vars_categorical[i], val_var]]
        plot_param_stripplot(
            res_df=res_df, id_var=id_vars_categorical[i], val_var=val_var, val_name='Mean Test Score', xlabel=p,
            dpi=300)
        plt.savefig(os.path.join(base_p, f'{p}.png'))
        plt.close('all')

        if id_vars_categorical[i] == 'param_som_dimensions':
            plot_som_pies(
                class_counts_per_unit=som_c.class_counts_per_unit_,
                som_dimensions=som_c.som_dimensions,
                label_mapping=som_c.new_to_og_classes_dict_,
                figsize=(9, 9),
                dpi=300
            )
            plt.savefig(os.path.join(base_p, 'som_pies.png'))
            plt.close('all')
            plot_support_hists(
                som_c=som_c, verbosity=1, dpi=300, plot_class_wise=True, plot_act_freq=True, save_p=base_p)
            plot_support_hist_w_class_perc(
                som_c=som_c, class_label=None, n_bins=30, plot_percentages=True, fontsize=8, plot_title=True, dpi=300)
            plt.savefig(os.path.join(base_p, 'support_hist_w_perc.png'))
            plt.close('all')
            for c in som_c.og_classes_:
                plot_support_hist_w_class_perc(
                    som_c=som_c, class_label=c, n_bins=30, plot_percentages=True, fontsize=8, plot_title=True, dpi=300)
                plt.savefig(os.path.join(base_p, f'support_hist_w_perc_c{c}.png'))
                plt.close('all')

    res_ps_numerical = ['lr0', 'lrn', 'neigh_r0', 'neigh_rn', 'sigma']
    id_vars_numerical = [
        'param_learning_rate_0', 'param_learning_rate_n', 'param_radius_0', 'param_radius_n',
        'param_gaussian_neighborhood_sigma'
    ]
    for i, p in enumerate(res_ps_numerical):
        som_c = SomClassifier.load(
            os.path.join(base_p, p, 'som_classifier.pkl'))
        res_df = pd.DataFrame(som_c.grid_search_.cv_results_)
        print(f'# ###### {p} ###### #\n{res_df}')
        plot_param_lineplot(
            res_df=res_df, x_col=id_vars_numerical[i], y_col=val_var, x_label=p, x_axis_grid=False,
            abline_param_values=True, dpi=300)
        plt.savefig(os.path.join(base_p, f'{p}.png'))
        plt.close('all')


def set_wise_hyperparameter_tuning():
    import os
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    import matplotlib

    matplotlib.use('Agg')

    if not os.path.exists(os.path.join(os.getcwd(), 'results/set_wise_param_tuning')):
        os.makedirs(os.path.join(os.getcwd(), 'results/set_wise_param_tuning/data_handling'))
    base_p = os.path.join(os.getcwd(), 'results/set_wise_param_tuning/')

    # ### Load and process .fcs data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # Perform data split into train- and test-set, load data split used for baseline results (60, 40)
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # ### Define channels to be used for training and testing the SOM and create a dataloader for the train-/test-data
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # ### Create a dataloader for the train data and check the class balance in the train data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))

    downsample = True
    if downsample:
        # ### Downsample train set to half its size in a stratified fashion for this study
        # Calculate the original class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_count = y_train.shape[0]
        class_ratios = class_counts / total_count
        # print(class_ratios)
        # print(class_counts)
        # print(x_train.shape)
        # Calculate the desired count for each class
        ds_fraction = 0.25
        target_size = np.ceil(total_count * ds_fraction)
        target_counts = (class_ratios * target_size).round().astype(int)
        # Downsample
        downsampled_x = []
        downsampled_y = []
        for cls, count in zip(unique_classes, target_counts):
            # Get indices of the current class
            class_indices = np.where(y_train == cls)[0]
            # Randomly sample from these indices
            sampled_indices = np.random.choice(class_indices, size=count, replace=False)
            # Append downsampled data and labels
            downsampled_x.append(x_train[sampled_indices])
            downsampled_y.append(y_train[sampled_indices])
        # Concatenate results
        x_train = np.vstack(downsampled_x)
        y_train = np.concatenate(downsampled_y)

    n_epochs = 2000

    param_grid_neighborhood = {
        'neighborhood': ['gaussian', ],
        'gaussian_neighborhood_sigma': [0.5, 0.25, 0.1],
        'radius_0': [5.0, 6.0, 7.0, 8.0, 9.0],
        'radius_n': [0.75, 0.5, 0.25, 0.1, 0.01],
        'radius_cooling': ['linear', 'exponential'],
        'n_epochs': [n_epochs, ],
    }

    param_grid_learning_rate = {
        'learning_rate_0': [0.1, 0.2, 0.6, 1.0],
        'learning_rate_n': [0.1, 0.01, 0.001],
        'learning_rate_decay': ['linear', 'exponential'],
        'n_epochs': [n_epochs, ],
    }

    # grids = [param_grid_neighborhood, param_grid_learning_rate]
    # grid_names = ['neigh', 'lr']

    # grids = [param_grid_neighborhood, ]
    # grid_names = ['neigh', ]  # ['neigh', ]

    grids = [param_grid_learning_rate, ]
    grid_names = ['lr', ]

    cv = 3

    for grid, grid_name in zip(grids, grid_names):

        if not os.path.exists(os.path.join(base_p, grid_name)):
            os.makedirs(os.path.join(base_p, grid_name))

        som_c = SomClassifier(verbosity=2)
        som_c.hyperparameter_tuning(
            X=x_train.copy(), y=y_train.copy(), param_grid=grid, cv=cv, scoring='internal', refit=False,
        )

        som_c.save(filepath=os.path.join(base_p, grid_name))


def view_results_set_wise_lr():
    import os
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results, plot_param_stripplot
    import matplotlib

    matplotlib.use('Agg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/set_wise_param_tuning')

    som_c_lr = SomClassifier.load(os.path.join(base_p, 'lr/som_classifier.pkl'))

    res_df_lr = pd.DataFrame(som_c_lr.grid_search_.cv_results_)
    best_params_lr = som_c_lr.grid_search_.best_params_
    print(res_df_lr)
    print(best_params_lr)
    print(som_c_lr.grid_search_.best_score_)

    x_coords = res_df_lr['mean_test_score'].to_numpy()
    y_coords = res_df_lr['std_test_score'].to_numpy()
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(x=x_coords, y=y_coords)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.text(
            x=x,
            y=y,
            s=str(i),
            fontsize=8,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )
    ax.set_xlabel('mean_test_score')
    ax.set_ylabel('std_test_score')
    ax.set_title(
        f'Min score: {np.round(x_coords.min(), 4)}, max score: {np.round(x_coords.max(), 4)}, '
        f'range: {np.round(x_coords.max() - x_coords.min(), 4)}')
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'lr/scatter.png'))

    plot_param_stripplot(
        res_df=res_df_lr, id_var='param_learning_rate_decay', val_var='mean_test_score', val_name='Mean Score',
        xlabel='', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'lr/decay.png'))

    plot_param_stripplot(
        res_df=res_df_lr, id_var='param_learning_rate_0', val_var='mean_test_score', val_name='Mean Score',
        jitter=0.2, xlabel='LR0', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'lr/lr0.png'))

    plot_param_stripplot(
        res_df=res_df_lr, id_var='param_learning_rate_n', val_var='mean_test_score', val_name='Mean Score',
        jitter=0.2, xlabel='LRN', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'lr/lrn.png'))


def view_results_set_wise_neigh():
    import os
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_stripplot, plot_param_heatmap
    import matplotlib

    matplotlib.use('Agg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/set_wise_param_tuning')

    som_c_neigh = SomClassifier.load(os.path.join(base_p, 'neigh/som_classifier.pkl'))

    res_df_neigh = pd.DataFrame(som_c_neigh.grid_search_.cv_results_)
    best_params_neigh = som_c_neigh.grid_search_.best_params_
    print(res_df_neigh)
    print(best_params_neigh)
    print(som_c_neigh.grid_search_.best_score_)

    x_coords = res_df_neigh['mean_test_score'].to_numpy()
    y_coords = res_df_neigh['std_test_score'].to_numpy()
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(x=x_coords, y=y_coords)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.text(
            x=x,
            y=y,
            s=str(i),
            fontsize=8,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )
    ax.set_xlabel('mean_test_score')
    ax.set_ylabel('std_test_score')
    ax.set_title(
        f'Min score: {np.round(x_coords.min(), 4)}, max score: {np.round(x_coords.max(), 4)}, '
        f'range: {np.round(x_coords.max() - x_coords.min(), 4)}')
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/scatter.png'))

    plot_param_stripplot(
        res_df=res_df_neigh, id_var='param_gaussian_neighborhood_sigma', val_var='mean_test_score', val_name='Mean Score',
        xlabel='Sigma', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/sigma.png'))

    plot_param_stripplot(
        res_df=res_df_neigh, id_var='param_radius_n', val_var='mean_test_score', val_name='Mean Score',
        jitter=0.2, xlabel='R0', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/r0.png'))

    plot_param_stripplot(
        res_df=res_df_neigh, id_var='param_radius_n', val_var='mean_test_score', val_name='Mean Score',
        jitter=0.2, xlabel='RN', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/rn.png'))

    plot_param_stripplot(
        res_df=res_df_neigh, id_var='param_radius_cooling', val_var='mean_test_score', val_name='Mean Score',
        jitter=0.2, xlabel='Rcooling', dpi=300)
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/rcooling.png'))

    plot_param_heatmap(
        res_df=res_df_neigh, param_row='param_radius_n', param_col='param_radius_0',
        other_params={'param_gaussian_neighborhood_sigma': 0.5, 'param_radius_cooling': 'linear'})
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/heatmap_sig0.5_linear.png'))

    plot_param_heatmap(
        res_df=res_df_neigh, param_row='param_radius_n', param_col='param_radius_0',
        other_params={'param_gaussian_neighborhood_sigma': 0.25, 'param_radius_cooling': 'linear'})
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/heatmap_sig0.25_linear.png'))

    plot_param_heatmap(
        res_df=res_df_neigh, param_row='param_radius_n', param_col='param_radius_0',
        other_params={'param_gaussian_neighborhood_sigma': 0.1, 'param_radius_cooling': 'linear'})
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/heatmap_sig0.1_linear.png'))

    plot_param_heatmap(
        res_df=res_df_neigh, param_row='param_radius_n', param_col='param_radius_0',
        other_params={'param_gaussian_neighborhood_sigma': 0.5, 'param_radius_cooling': 'exponential'})
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/heatmap_sig0.5_exp.png'))

    plot_param_heatmap(
        res_df=res_df_neigh, param_row='param_radius_n', param_col='param_radius_0',
        other_params={'param_gaussian_neighborhood_sigma': 0.25, 'param_radius_cooling': 'exponential'})
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/heatmap_sig0.25_exp.png'))

    plot_param_heatmap(
        res_df=res_df_neigh, param_row='param_radius_n', param_col='param_radius_0',
        other_params={'param_gaussian_neighborhood_sigma': 0.1, 'param_radius_cooling': 'exponential'})
    plt.tight_layout()
    plt.savefig(os.path.join(base_p, 'neigh/heatmap_sig0.1_exp.png'))


def input_data_processing_study():
    import os
    import scanpy as sc
    import pytometry as pm
    from typing import List, Union, Callable
    from sklearn.model_selection import StratifiedKFold
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import evaluate_classification_results
    import matplotlib

    matplotlib.use('Agg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    def stratified_downsampling(X: np.ndarray, y: np.ndarray, ds_fraction: float = 0.25, random_seed: int = 42):
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

    def custom_stratified_cv(
            classifier: SomClassifier,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 3,
            random_state: Union[int, None] = None,
            shuffle: bool = False,
            prepr_fct: Union[Callable, None] = None,
    ) -> List[float]:
        # ### Split data into 'n_splits' folds, fit a classifier on each fold, and validate on the remaining data,
        # return the validation scores of each fold

        # Initialize StratifiedKFold splitter
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

        scores = []
        # Loop through each fold
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            print(f'# ###### Fold {fold + 1}/{n_splits} ###### #')

            # Split the data into training and testing sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_test = y[train_index], y[val_index]

            # Perform preprocessing function to data if a function was passed
            if prepr_fct is not None:
                X_train = prepr_fct(X_train)
                X_val = prepr_fct(X_val)

            # Train the classifier
            classifier.fit(X=X_train, y=y_train)

            # Compute test score
            test_score = classifier.score(X=X_val, y=y_test)

            # Store results
            scores.append(test_score)

        print(f'# ### CV finnished, Mean score: {np.mean(scores):.4f}, Std dev: {np.std(scores):.4f}\n')

        return scores

    def plot_helper(x: np.ndarray, channels: List[str], subdir: Union[str, None] = None):
        # ### For a datamatrix and channel names plot the histogram of the intensities of events (channel-wise)
        for i in range(x.shape[1]):
            fig, ax = plt.subplots(dpi=300)
            ax.hist(x[:, i], bins=200, color='lightblue', edgecolor='lightgrey')
            ax.set_xlabel('Intensity')
            ax.set_ylabel('# events')
            ax.set_title(channels[i])

            if subdir is None:
                subdir = ''

            if not os.path.exists(os.path.join(base_p, subdir)):
                os.makedirs(os.path.join(base_p, subdir))

            plt.savefig(os.path.join(base_p, subdir, f'hist_{channels[i]}.png'))
            plt.close('all')

    # ### Set base file path
    base_p = os.path.join(os.getcwd(), 'results/data_processing_study_cv5_3')
    if not os.path.exists(base_p):
        os.makedirs(base_p)

    # ### Load and process .fcs data
    use_all = True
    if use_all:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
    else:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

    # ### Set parameters for the helper function here
    # ### Set downsampling flag (same as for parameter influence study)
    downsample = True
    frac = 0.25
    # ### Set parameters for cv
    n_folds = 5  # 3
    rs = 42
    shuffle_events = True
    # ### Set number of epochs
    n_epochs = 2000

    def helper(settings: dict, train: bool = True, cv: bool = False, plot: bool = False):
        # ### Note: There are 2 options for preprocessing:
        # 1) 'preprocessing_flavour' (Applied by sample wise by FlowDataManager during data loading)
        # 2) 'data_set_wise_preprocessing_fct' (Applied before training on whole data set)

        # ### Extract settings
        name = settings.get('name', None)
        preprocessing_flavour = settings.get('preprocessing_flavour', None)
        additional_preprocessing_kwargs = settings.get('additional_preprocessing_kwargs', None)
        channels = settings.get('channels', None)
        data_set_wise_preprocessing_fct = settings.get('data_set_wise_preprocessing_fct', None)

        # ### Create subdir for respective settings dict
        if not os.path.exists(os.path.join(base_p, name)):
            os.makedirs(os.path.join(base_p, name, 'data_handling'))

        # ### Instantiate FlowManager (sample-wise perprocessing is performed here as well)
        flow_manager = FlowDataManager(
            filename_list=filename_list,
            preprocessing_flavour=preprocessing_flavour,
            additional_preprocessing_kwargs=additional_preprocessing_kwargs,
            memory_saving=False, channel_name_reference=0,
            data_file_path='input/concatenated_labeled_fcs_format',
            save_path=os.path.join(base_p, name, 'data_handling'),
            filenames={'fn_og_channel_names': 'og_channel_names.csv', }
        )

        # ### Split data into train and test set, use same data split as for baseline results
        flow_manager.perform_data_split(
            data_split=pd.read_csv(
                os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'), index_col=0)
        )

        # ### Create a dataloader for the train data and extract the data matrices from it
        dl_train = flow_manager.create_data_loader(
            data_set='train', channels=channels, layer_key=None, label_key='population',
            label_layer_key='original' if preprocessing_flavour is not None else None,
            return_data_loader='np_array', batch_size=-1, shuffle=False, on_disk=True, filename='data_train.npy')
        x_train, y_train = next(iter(dl_train))

        # ### Downsample the data set in a stratified fashion
        if downsample:
            x_train, y_train = stratified_downsampling(X=x_train, y=y_train, ds_fraction=frac)

        # ### Create a dataloader for the test data and extract the data matrices from it
        dl_test = flow_manager.create_data_loader(
            data_set='test', channels=channels, layer_key=None, label_key='population',
            label_layer_key='original' if preprocessing_flavour is not None else None,
            return_data_loader='np_array', batch_size=-1, shuffle=False, on_disk=True, filename='data_test.npy')
        x_test, y_test = next(iter(dl_test))

        # ### Plot the histogram of event intensities (channel-wise)
        if plot:
            x_plot = x_train.copy()
            if data_set_wise_preprocessing_fct is not None:
                x_plot = data_set_wise_preprocessing_fct(x_plot)
            plot_helper(x=x_plot, channels=channels, subdir=name)

        if train:
            # ### Note: 'data_set_wise_preprocessing_fct()' is applied to each training data set (fold or all data)
            if cv:
                # ### Cross validate on training data
                som_c = SomClassifier(n_epochs=n_epochs, verbosity=2)
                scores = custom_stratified_cv(
                    classifier=som_c, X=x_train, y=y_train, n_splits=n_folds, random_state=rs, shuffle=shuffle_events,
                    prepr_fct=data_set_wise_preprocessing_fct
                )
            else:
                # ### Train on training data, test performance on test data
                if data_set_wise_preprocessing_fct is not None:
                    x_train = data_set_wise_preprocessing_fct(x_train)
                    x_test = data_set_wise_preprocessing_fct(x_test)

                som_c = SomClassifier(n_epochs=n_epochs, verbosity=2)
                som_c.fit(X=x_train, y=y_train)

                scores = [som_c.score(X=x_test, y=y_test), ]

            return name, scores, som_c

    # ### Define settings to test
    shifted = False
    if not shifted:
        raw = {
            'name': 'raw',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        # ### Note: Arcsinh was used up till now
        arcsinh = {
            'name': 'arcsinh',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 150},
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        arcsinh_cof300 = {
            'name': 'arcsinh_cof300',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 300},
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        arcsinh_nofsint = {
            'name': 'arcsinh_nofsint',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 150},
            'channels': ['SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        arcsinh_nossint = {
            'name': 'arcsinh_nossint',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 150},
            'channels': ['FS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        arcsinh_nofsssint = {
            'name': 'arcsinh_nofsssint',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 150},
            'channels': ['16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        biexp = {
            'name': 'biexp',
            'preprocessing_flavour': 'biexp',
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def custom_logicle(adata):
            x = adata.X
            x[x <= 0] = 0
            adata.X = x
            pm.tl.normalize_logicle(adata=adata)

        logicle_geq_zero = {
            'name': 'logicle_geq_zero',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': custom_logicle, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log_trafo(adata):
            x = adata.X
            x[x <= 0] = 0
            x[x != 0] = np.log(x[x != 0])
            adata.X = x

        log = {
            'name': 'log',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log_trafo, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log10_trafo(adata):
            x = adata.X
            x[x <= 0] = 0
            x[x != 0] = np.log10(x[x != 0])
            adata.X = x

        log10 = {
            'name': 'log10',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log10_trafo, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log_trafo_w_cutoff_100(adata):
            cutoff = 100
            x = adata.X
            x = np.log(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
            adata.X = x

        log_w_cutoff_100 = {
            'name': 'log_w_cutoff_100',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log_trafo_w_cutoff_100, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log10_trafo_w_cutoff_100(adata):
            cutoff = 100
            x = adata.X
            x = np.log10(x, out=np.full(x.shape, np.log10(cutoff), dtype=float), where=(x > cutoff))
            adata.X = x

        log10_w_cutoff_100 = {
            'name': 'log10_w_cutoff_100',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log10_trafo_w_cutoff_100, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log_trafo_w_cutoff_300(adata):
            cutoff = 300
            x = adata.X
            x = np.log(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
            adata.X = x

        log_w_cutoff_300 = {
            'name': 'log_w_cutoff_300',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log_trafo_w_cutoff_300, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log10_trafo_w_cutoff_300(adata):
            cutoff = 300
            x = adata.X
            x = np.log10(x, out=np.full(x.shape, np.log10(cutoff), dtype=float), where=(x > cutoff))
            adata.X = x

        log10_w_cutoff_300 = {
            'name': 'log10_w_cutoff_300',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log10_trafo_w_cutoff_300, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def log_trafo_w_cutoff_channel_wise(adata):
            cutoff_dict = {
                'FS INT': 100000, 'SS INT': 20000, '16-FITC': 250, '56-PE': 450, '3-ECD': 700, '4-PC7': 1200,
                '19-APC': 1700, '14-APC700': 900, '8-PB': 450, '45-CO': 500
            }

            x = adata.X.copy()
            for channel, cutoff in cutoff_dict.items():
                col_idx = np.where(adata.var_names == channel)[0][0]
                x_col = adata.X[:, col_idx].copy()
                mask = (x_col > cutoff)
                x_col[mask] = np.log(x_col[mask])
                x_col[~mask] = np.log(cutoff)
                x[:, col_idx] = x_col

            adata.X = x

        log_w_cutoff_channel_wise = {
            'name': 'log_w_cutoff_channel_wise',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log_trafo_w_cutoff_channel_wise, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

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

        log10_w_cutoff_channel_wise = {
            'name': 'log10_w_cutoff_channel_wise',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log10_trafo_w_cutoff_channel_wise, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

        def z_score_norm(data: np.ndarray):
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            standardized_data = (data - mean) / std
            return standardized_data

        z_score = {
            'name': 'normalize',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': z_score_norm,
        }

        arcsinh_z_score = {
            'name': 'arcsinh_z_score',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 150},
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': z_score_norm,
        }

        logicle_z_score = {
            'name': 'logicle_z_score',
            'preprocessing_flavour': 'logicle',
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': z_score_norm
        }

        log_w_cutoff_100_z_score = {
            'name': 'log_w_cutoff_100_z_score',
            'preprocessing_flavour': 'custom',
            'additional_preprocessing_kwargs': {'preprocessing_method': log_trafo_w_cutoff_100, },
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': z_score_norm
        }

        def min_max_norm(data: np.ndarray):
            val_range = 10
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            normalized_data = (data - mins) / ranges * val_range
            return normalized_data

        arcsinh_minmax = {
            'name': 'arcsinh_minmax',
            'preprocessing_flavour': 'arcsinh',
            'additional_preprocessing_kwargs': {'cofactor': 150},
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': min_max_norm
        }

        logicle = {
            'name': 'logicle',
            'preprocessing_flavour': 'logicle',
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': None
        }

    else:
        def pos_shift(x):
            # Determine the min value of each channel
            mins = x.min(axis=0)
            values_to_add = np.where(mins < 0, -mins, 0)
            return x + values_to_add

        def prepr_raw(x):
            x = pos_shift(x=x)
            return x

        raw = {
            'name': 'raw',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_raw,
        }

        def prepr_arcsinh(x):
            x = pos_shift(x=x)
            dummy_ad = sc.AnnData(x)
            pm.tl.normalize_arcsinh(adata=dummy_ad, cofactor=150)
            return dummy_ad.X

        arcsinh = {
            'name': 'arcsinh',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_arcsinh,
        }

        arcsinh_nofsint = {
            'name': 'arcsinh_nofsint',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_arcsinh,
        }

        arcsinh_nossint = {
            'name': 'arcsinh_nossint',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_arcsinh,
        }

        arcsinh_nofsssint = {
            'name': 'arcsinh_nofsssint',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_arcsinh,
        }

        def prepr_biexp(x):
            x = pos_shift(x=x)
            dummy_ad = sc.AnnData(x)
            pm.tl.normalize_biExp(dummy_ad)
            return dummy_ad.X

        biexp = {
            'name': 'biexp',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_biexp,
        }

        def prepr_logicle(x):
            x = pos_shift(x=x)
            dummy_ad = sc.AnnData(x)
            pm.tl.normalize_logicle(adata=dummy_ad)
            return dummy_ad.X

        logicle = {
            'name': 'logicle',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_logicle
        }

        def prepr_log_trafo(x):
            x = pos_shift(x=x)
            x[x > 0] = np.log(x[x > 0])
            return x

        log = {
            'name': 'log',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_log_trafo,
        }

        def prepr_log_trafo_w_cutoff(x):
            cutoff = 100
            x = pos_shift(x=x)
            x = np.log(x, out=np.full(x.shape, np.log10(cutoff), dtype=float), where=(x > cutoff))
            return x

        log_w_cutoff = {
            'name': 'log_w_cutoff',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_log_trafo_w_cutoff,
        }

        def z_score_norm(data: np.ndarray):
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            standardized_data = (data - mean) / std
            return standardized_data

        def prepr_zscore(x):
            return z_score_norm(prepr_raw(x=x))

        z_score = {
            'name': 'normalize',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_zscore,
        }

        def prepr_arcsinh_z_score(x):
            return z_score_norm(prepr_arcsinh(x=x))

        arcsinh_z_score = {
            'name': 'arcsinh_z_score',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_arcsinh_z_score,
        }

        def prepr_logicle_z_score(x):
            return z_score_norm(prepr_logicle(x=x))

        logicle_z_score = {
            'name': 'logicle_z_score',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_logicle_z_score,
        }

        def min_max_norm(data: np.ndarray):
            val_range = 10
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            normalized_data = (data - mins) / ranges * val_range
            return normalized_data

        def prepr_arcsinh_minmax(x):
            return min_max_norm(prepr_arcsinh(x=x))

        arcsinh_minmax = {
            'name': 'arcsinh_minmax',
            'preprocessing_flavour': None,
            'additional_preprocessing_kwargs': None,
            'channels': ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB',
                         '45-CO'],
            'data_set_wise_preprocessing_fct': prepr_arcsinh_minmax,
        }

    # ### Run the analysis
    # settings_list = [
    #     log, log_w_cutoff_100, log_w_cutoff_300, log_w_cutoff_channel_wise,
    #     log10, log10_w_cutoff_100, log10_w_cutoff_300, log10_w_cutoff_channel_wise,
    #     raw,
    #     logicle, logicle_geq_zero,
    #     arcsinh, arcsinh_cof300, arcsinh_nofsint, arcsinh_nossint, arcsinh_nofsssint, arcsinh_minmax,
    #     biexp,
    #     z_score, arcsinh_z_score, logicle_z_score
    # ]

    settings_list = [
        log_w_cutoff_channel_wise, log10_w_cutoff_channel_wise, log_w_cutoff_100_z_score
    ]

    just_plot = False

    if not just_plot:
        res_df = pd.DataFrame(
            columns=['setting', ] +
                    [f'split{i}_test_score' for i in range(n_folds)] +
                    ['mean_test_score', 'std_test_score']
        )

        for s in settings_list:
            print(f'# ###### Setting: {s["name"]} ###### #')
            s_name, s_scores, _ = helper(settings=s, train=True, cv=True, plot=True)
            res_df.loc[len(res_df)] = [s_name, ] + s_scores + [np.mean(np.array(s_scores)), np.std(np.array(s_scores))]

        res_df['rank_test_score'] = res_df['mean_test_score'].rank(ascending=False, method='min').astype(int)

        print(res_df)
        best_idx = res_df['mean_test_score'].idxmax()
        print(f'# ### The best setting was {res_df["setting"].iloc[best_idx]}')
        res_df.to_csv(os.path.join(base_p, 'res_df.csv'))

        # Retrain with the best
        # _, _, som_classifier = helper(settings=settings_list[best_idx], train=True, cv=False, plot=False)
        # som_classifier.save(filepath=base_p)
    else:
        for s in settings_list:
            print(f'# ###### Setting: {s["name"]} ###### #')
            helper(settings=s, train=False, cv=False, plot=True)


def view_results_input_data_processing_study():

    import os
    import matplotlib

    matplotlib.use('Agg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # df0: Old not all trafos
    # df0 = pd.read_csv(os.path.join(os.getcwd(), 'results/data_processing_study/res_df.csv'), index_col=0)
    # df1: log-log10 mixup
    # df1 = pd.read_csv(os.path.join(os.getcwd(), 'results/data_processing_study_cv5/res_df.csv'), index_col=0)
    # df2: Trafos should be correct except that the channel wise cutoff is not working
    df2 = pd.read_csv(os.path.join(os.getcwd(), 'results/data_processing_study_cv5_2/res_df.csv'), index_col=0)
    # df3: Channel wise cutoff is
    df3 = pd.read_csv(os.path.join(os.getcwd(), 'results/data_processing_study_cv5_3/res_df.csv'), index_col=0)

    # Set 'setting' columns as index
    df = df2.copy()
    df.set_index('setting', inplace=True)
    df3.set_index('setting', inplace=True)

    # Add missing/replace faulty rows
    df.loc['log_w_cutoff_channel_wise'] = df3.loc['log_w_cutoff_channel_wise']
    df.loc['log10_w_cutoff_channel_wise'] = df3.loc['log10_w_cutoff_channel_wise']
    df.loc['log_w_cutoff_100_z_score'] = df3.loc['log_w_cutoff_100_z_score']

    # Recompute ranking
    df['rank_test_score'] = df['mean_test_score'].rank(ascending=False, method="min").astype(int)

    print(df.sort_values(by='rank_test_score'))


def majority_class_downsampling_study():
    import os
    import time
    from typing import Union, Callable, List, Tuple
    from sklearn.model_selection import StratifiedKFold
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.metrics import f1_score
    from flowsrc.flowsom import SomClassifier
    from flowsrc.flowdata import FlowDataManager

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    def majority_class_downsampling(X: np.ndarray, y: np.ndarray):
        y_series = pd.Series(y)
        count_series = y_series.value_counts()
        count_array = count_series.values

        median = np.floor(np.median(count_array))
        mad = np.median(np.abs(count_array - median))
        upper = int(median + mad)
        print(
            f'# ### Class counts have: median: {median}, mad: {mad} => down-sampling to median + mad = {upper}'
        )

        ds_dict = {}
        for idx, val in count_series.items():
            if val >= upper:
                ds_dict[idx] = upper

        rds = RandomUnderSampler(random_state=42, sampling_strategy=ds_dict)

        X_ds, y_ds = rds.fit_resample(X, y)

        return X_ds, y_ds

    def custom_stratified_cv(
            classifier: SomClassifier,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 3,
            random_state: Union[int, None] = None,
            shuffle: bool = False,
            majority_downsample: bool = False,
    ) -> Tuple[List[float], List[np.ndarray], List[float]]:
        # ### Split data into 'n_splits' folds, fit a classifier on each fold, and validate on the remaining data,
        # return the validation scores of each fold

        # Initialize StratifiedKFold splitter
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

        scores = []
        class_wise_scores = []
        fit_times = []
        # Loop through each fold
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            print(f'# ###### Fold {fold + 1}/{n_splits} ###### #')

            # Split the data into training and testing sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            if majority_downsample:
                X_train, y_train = majority_class_downsampling(X=X_train, y=y_train)

            # Train the classifier
            st = time.time()
            classifier.fit(X=X_train, y=y_train)
            et = time.time()
            fit_time = et - st
            fit_times.append(fit_time)

            # Compute test score
            test_score = classifier.score(X=X_val, y=y_val)
            scores.append(test_score)

            # Compute the class-wise F1 score
            y_pred = classifier.predict(X=X_val)
            f1 = f1_score(y_val, y_pred, average=None)
            class_wise_scores.append(f1)

        print(
            f'# ### CV finnished, Mean score: {np.mean(scores):.4f}, Std dev: {np.std(scores):.4f}, '
            f'Mean fit time: {np.mean(fit_times):.4f}\n'
        )

        return scores, class_wise_scores, fit_times


    # ### Set base file path
    base_p = os.path.join(os.getcwd(), 'results/downsampling_study2')
    if not os.path.exists(base_p):
        os.makedirs(os.path.join(base_p, 'data_handling'))

    use_all = True
    if use_all:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))
    else:
        filename_list = os.listdir(os.path.join(os.getcwd(), 'input/dummy'))

    # ### Instantiate FlowManager (sample-wise perprocessing is performed here as well)
    flow_manager = FlowDataManager(
        filename_list=filename_list,
        preprocessing_flavour='arcsinh',
        additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format',
        save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # ### Split data into train and test set, use same data split as for baseline results
    flow_manager.perform_data_split(
        data_split=pd.read_csv(
            os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'), index_col=0)
    )

    # ### Create a dataloader for the train data and extract the data matrices from it
    dl_train = flow_manager.create_data_loader(
        data_set='train',
        channels=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
        layer_key=None, label_key='population',
        label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
    x_train, y_train = next(iter(dl_train))

    # ### Run CV with and without downsampling
    n_epochs = 7000
    n_splits = 5
    som_c = SomClassifier(n_epochs=n_epochs, verbosity=2)
    scores, class_wise_scores, fit_times = custom_stratified_cv(
        classifier=som_c,
        X=x_train.copy(), y=y_train.copy(),
        n_splits=n_splits, random_state=42, shuffle=True, majority_downsample=False
    )

    som_c_ds = SomClassifier(n_epochs=n_epochs, verbosity=2)
    scores_ds, class_wise_scores_ds, fit_times_ds = custom_stratified_cv(
        classifier=som_c_ds,
        X=x_train.copy(), y=y_train.copy(),
        n_splits=n_splits, random_state=42, shuffle=True, majority_downsample=True
    )

    res_df_general = pd.DataFrame(
        columns=[f'score_split_{i}' for i in range(n_splits)] + ['mean_score', 'std_score'] +
                [f'time_split_{i}' for i in range(n_splits)] + ['mean_time', 'std_time']
    )

    res_df_general.loc['no_ds'] = (
            scores + [np.mean(scores), np.std(scores)] + fit_times + [np.mean(fit_times), np.std(fit_times)]
    )
    res_df_general.loc['ds'] = (
            scores_ds + [np.mean(scores_ds), np.std(scores_ds)] +
            fit_times_ds + [np.mean(fit_times_ds), np.std(fit_times_ds)]
    )

    print(res_df_general)

    res_df_general.to_csv(os.path.join(base_p, 'res_df_general.csv'))

    class_wise_scores = np.array(class_wise_scores)
    class_wise_scores_mean = class_wise_scores.mean(axis=0)
    class_wise_scores_std = class_wise_scores.std(axis=0)
    class_wise_scores = np.vstack((class_wise_scores, class_wise_scores_mean, class_wise_scores_std))

    class_wise_scores_ds = np.array(class_wise_scores_ds)
    class_wise_scores_ds_mean = class_wise_scores_ds.mean(axis=0)
    class_wise_scores_ds_std = class_wise_scores_ds.std(axis=0)
    class_wise_scores_ds = np.vstack((class_wise_scores_ds, class_wise_scores_ds_mean, class_wise_scores_ds_std))

    res_df_channel_wise = pd.DataFrame(
        data=np.vstack((class_wise_scores, class_wise_scores_ds)),
        index=[f'split_{i}_no_ds' for i in range(n_splits)] + ['mean_score_no_ds', 'std_score_no_ds'] +
              [f'split_{i}_ds' for i in range(n_splits)] + ['mean_score_ds', 'std_score_ds'],
        columns=list(range(1, class_wise_scores.shape[1] + 1))
    )

    print(res_df_channel_wise)

    res_df_channel_wise.to_csv(os.path.join(base_p, 'res_df_channel_wise.csv'))

    # Also save SOM classifiers => Training settings are stored as well
    som_c.save(filename='som_classifier.pkl', filepath=base_p)
    som_c_ds.save(filename='som_classifier_ds.pkl', filepath=base_p)


def view_results_majority_class_downsampling_study():

    from flowsrc.flowsom import SomClassifier

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/downsampling_study')
    res_df_general = pd.read_csv(os.path.join(base_p, 'res_df_general.csv'))
    res_df_channel_wise = pd.read_csv(os.path.join(base_p, 'res_df_channel_wise.csv'))
    print(res_df_general)
    print(res_df_channel_wise)

    print('# ##### 2nd run ######')

    base_p = os.path.join(os.getcwd(), 'results/downsampling_study2')
    res_df_general = pd.read_csv(os.path.join(base_p, 'res_df_general.csv'))
    res_df_channel_wise = pd.read_csv(os.path.join(base_p, 'res_df_channel_wise.csv'))
    print(res_df_general)
    print(res_df_channel_wise)


# ### Parameter Tuning
def estimate_n_epochs():
    import os
    from sklearn.model_selection import train_test_split
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import (
        evaluate_classification_results, plot_nd_metric_over_time, plot_1d_metric_over_time, plot_som_pies
    )
    import matplotlib

    matplotlib.use('Agg')

    if not os.path.exists(os.path.join(os.getcwd(), 'results/param_tuning')):
        os.makedirs(os.path.join(os.getcwd(), 'results/param_tuning/data_handling'))
        os.makedirs(os.path.join(os.getcwd(), 'results/param_tuning/nepochs'))
    base_p = os.path.join(os.getcwd(), 'results/param_tuning/nepochs')

    # ### Load and process .fcs data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour='arcsinh', additional_preprocessing_kwargs={'cofactor': 150},
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # Perform data split into train- and test-set, load data split used for baseline results (60, 40)
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # ### Define channels to be used for training and testing the SOM and create a dataloader for the train-/test-data
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # ### Create a dataloader for the train data and check the class balance in the train data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')
    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))

    dl_test = flow_manager.create_data_loader(
        data_set='test', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_test.npy')
    x_test, y_test = next(iter(dl_test))

    som_c = SomClassifier(verbosity=2)

    gs = False
    if gs:
        # ### Cross-validated grid-search
        param_grid_nepochs = {
            'n_epochs': [20, 40, 60, 80, 100, 200, 400, 600, 700, 800, 900, 1000, 2000, 4000, 6000, 8000, 10000],
        }
        # Define filepaths for saving
        fp = os.path.join(base_p, 'gridsearch')
        if not os.path.exists(fp):
            os.makedirs(fp)

        # Set number of splits
        cv = 3

        # ### Perform cross-validated grid-search
        som_c.hyperparameter_tuning(
            X=x_train.copy(), y=y_train.copy(), param_grid=param_grid_nepochs, cv=cv, scoring='internal', refit=True,
        )
        # Evaluate performance of best found model on test-set
        y_pred = som_c.predict(X=x_test.copy())
        evaluate_classification_results(
            y_true=y_test.copy(), y_pred=y_pred, classes=som_c.og_classes_, class_priors=som_c.class_priors_,
            compare_to_dummy_classifier=False,
            filepath=fp)
        # Save SOM classifier with results
        som_c.save(filepath=fp)

    else:
        # ### Train model epoch wise
        n_epochs = [20, 40, 60, 80, 100, 200, 400, 600, 700, 800, 900, 1000, 2000, 4000, 6000, 8000, 10000]
        # Split into val and train set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
        # Set tracking interval
        for e in n_epochs:

            if e <= 100:
                tracking_interval = 4
            elif e <= 1000:
                tracking_interval = 8
            else:
                tracking_interval = 20

            fp = os.path.join(base_p, f'epoch_wise_{e}')
            if not os.path.exists(fp):
                os.makedirs(fp)

            # Train SOM epoch wise
            som_c.fit_epoch_wise(
                X=x_train, y=y_train, n_epochs=e, tracking_interval=tracking_interval, track_losses=False,
                track_impurities=True, save_intermediate_interval=None, val_data=(x_val, y_val),
                save_res_dict=True, filepath=fp)
            som_c.save(filepath=fp)

            # Plot results
            rd = som_c.epoch_wise_som_training_metrics_
            plot_1d_metric_over_time(
                metric_train=rd['entropy'].mean(axis=(0, 1)), n_epochs=rd['n_epochs'],
                tracking_interval=rd['tracking_interval'], ylabel='Mean entropy', filename='entropy.png', filepath=fp)
            plt.close('all')
            plot_nd_metric_over_time(
                metrics=rd['f1_train'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'],
                title='F1 train',
                ylabel='F1-score',
                # metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
                filename='f1_train.png', filepath=fp)
            plt.close('all')
            plot_nd_metric_over_time(
                metrics=rd['f1_val'], n_epochs=rd['n_epochs'], tracking_interval=rd['tracking_interval'],
                title='F1 val', ylabel='F1-score',
                # metric_labels=[som_c.new_to_og_classes_dict_[i] for i in range(som_c.classes_.shape[0])],
                filename='f1_val.png', filepath=fp)
            plt.close('all')
            plot_som_pies(
                class_counts_per_unit=som_c.class_counts_per_unit_, som_dimensions=som_c.som_dimensions,
                label_mapping=som_c.new_to_og_classes_dict_)
            plt.savefig(os.path.join(fp, f'som_pies_{e}.png'))
            plt.close('all')


def view_results_n_epochs_estimation():
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_lineplot
    import matplotlib
    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/param_tuning/nepochs/gridsearch')

    som_c = SomClassifier.load(os.path.join(base_p, 'som_classifier.pkl'))
    res_df = pd.DataFrame(som_c.grid_search_.cv_results_)

    res_df.to_csv(os.path.join(base_p, 'res_df.csv'))
    print(res_df)

    plot_param_lineplot(
        res_df=res_df, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
        markersize=3.0, x_axis_grid=True, dpi=300)
    plt.savefig(os.path.join(base_p, 'nepochs.png'), dpi=300)


def estimate_n_epochs_2():
    import os
    import matplotlib
    import scanpy as sc
    from sklearn.model_selection import train_test_split, PredefinedSplit
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier

    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Define some auxiliary functions
    def log_trafo_w_cutoff_100(adata: sc.AnnData):
        cutoff = 100
        x = adata.X
        x = np.log(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
        adata.X = x

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

    # ##################
    # ### Set parameters and flags here
    # Preprocessing
    # prepr_flavour = 'arcsinh'
    # add_prepr_kwargs = {'cofactor': 150}

    # prepr_flavour = 'custom'
    # add_prepr_kwargs = {'preprocessing_method': log_trafo_w_cutoff_100, }

    prepr_flavour = 'custom'
    add_prepr_kwargs = {'preprocessing_method': log10_trafo_w_cutoff_channel_wise, }

    # Cross validation, data split
    cross_validation = False
    cv = 5
    val_size = 0.32  # Only relevant when cross_validation is False

    # Channels to be used for training and testing the SOM
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # Parameter grid
    param_grid_nepochs = {
        # 'n_epochs': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,],
        # 'n_epochs': [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
        'n_epochs': [13000, 14000, 15000, 16000, 17000, 18000, 19000, 200000],
        'som_dimensions': [(10, 10), (20, 20)],
    }
    # ##################

    # ### Define path where results are stored, create dir if necessary
    prepr_kwargs_str = ''.join(
        f"{key}{value.__name__ if callable(value) else value}" for key, value in add_prepr_kwargs.items()
    )

    base_p = os.path.join(
        os.getcwd(),
        f'results/param_tuning/nepochs_5/{prepr_flavour + prepr_kwargs_str}_'
        f'{f"cv{cv}" if cross_validation else f"nocv{val_size}"}'
    )
    if not os.path.exists(base_p):
        os.makedirs(os.path.join(base_p, 'data_handling'))

    # ### Load and process .fcs data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour=prepr_flavour,
        additional_preprocessing_kwargs=add_prepr_kwargs,
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # Perform data split into train- and test-set, load data split used for baseline results (60, 40)
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # ### Create a dataloader for the train-/test-data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')

    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))

    if not cross_validation:
        # Due to running time constraints generate a train test split of the data, No cross validation!
        # Split the data into train (70%) and test (30%) sets
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
        X=x_train.copy(), y=y_train.copy(), param_grid=param_grid_nepochs, cv=cv, scoring='internal', refit=False,
    )

    print(pd.DataFrame(som_c.grid_search_.cv_results_))

    # ### Save SOM the classifier
    som_c.save(filepath=base_p)


def view_results_n_epochs_2():
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_lineplot
    import matplotlib
    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    n = 4

    base_p = os.path.join(os.getcwd(), f'results/param_tuning/nepochs_{n}/')

    modes = [
        'arcsinhcofactor150_nocv0.32', 'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise_nocv0.32',
        'custompreprocessing_methodlog_trafo_w_cutoff_100_nocv0.32'
    ]


    for mode in modes:

        som_c = SomClassifier.load(os.path.join(base_p, mode, 'som_classifier.pkl'))
        res_df = pd.DataFrame(som_c.grid_search_.cv_results_)

        res_df.drop_duplicates(subset=['param_n_epochs', 'param_som_dimensions'], inplace=True, ignore_index=True)

        res_df_10 = res_df[res_df['param_som_dimensions'] == (10, 10)].copy()
        res_df_10['rank_test_score'] = res_df_10['mean_test_score'].rank(ascending=False, method="min").astype(int)
        res_df_10.sort_values(by='param_n_epochs', inplace=True)
        res_df_10.to_csv(os.path.join(base_p, mode, 'res_df_10.csv'))
        plot_param_lineplot(
            res_df=res_df_10, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
            markersize=3.0, x_axis_grid=True, dpi=300)
        plt.savefig(os.path.join(base_p, f'nepochs_dim10_{mode}.png'), dpi=300)
        plt.close('all')

        if n == 2 or n == 4:
            res_df_20 = res_df[res_df['param_som_dimensions'] == (20, 20)].copy()
            res_df_20['rank_test_score'] = res_df_20['mean_test_score'].rank(ascending=False, method="min").astype(int)
            res_df_20.sort_values(by='param_n_epochs', inplace=True)
            res_df_20.to_csv(os.path.join(base_p, mode, 'res_df_20.csv'))
            plot_param_lineplot(
                res_df=res_df_20, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
                markersize=3.0, x_axis_grid=True, dpi=300)
            plt.savefig(os.path.join(base_p, f'nepochs_dim20_{mode}.png'), dpi=300)
            plt.close('all')

        print(f'# ### Mode: {mode}')
        print(res_df_10.sort_values(by='rank_test_score'))
        if n == 2 or n == 4:
            print(res_df_20.sort_values(by='rank_test_score'))

    concat_all = True
    if concat_all:
        for mode in modes:
            for dim in [10, 20]:
                som_c0 = SomClassifier.load(
                    os.path.join(os.getcwd(), f'results/param_tuning/nepochs_4/', mode, 'som_classifier.pkl')
                )
                res_df0 = pd.DataFrame(som_c0.grid_search_.cv_results_)
                res_df0 = res_df0[res_df0['param_som_dimensions'] == (dim, dim)].copy()

                if dim == 10:
                    som_c1 = SomClassifier.load(
                        os.path.join(os.getcwd(), f'results/param_tuning/nepochs_3/', mode, 'som_classifier.pkl')
                    )
                    res_df1 = pd.DataFrame(som_c1.grid_search_.cv_results_)
                else:
                    som_c1 = SomClassifier.load(
                        os.path.join(os.getcwd(), f'results/param_tuning/nepochs_2/', mode, 'som_classifier.pkl')
                    )
                    res_df1 = pd.DataFrame(som_c1.grid_search_.cv_results_)
                    res_df1 = res_df1[res_df1['param_som_dimensions'] == (20, 20)].copy()

                res_df = pd.concat([res_df0, res_df1], axis=0)

                res_df.drop_duplicates(subset=['param_n_epochs', 'param_som_dimensions'], inplace=True, ignore_index=True)

                res_df['rank_test_score'] = res_df['mean_test_score'].rank(ascending=False, method="min").astype(int)
                res_df.sort_values(by='param_n_epochs', inplace=True)
                res_df.to_csv(os.path.join(os.getcwd(), f'results/param_tuning/nepochs_4/', mode, 'res_df_all.csv'))

                plot_param_lineplot(
                    res_df=res_df, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
                    markersize=3.0, x_axis_grid=True, dpi=300)
                plt.savefig(
                    os.path.join(
                        os.getcwd(), f'results/param_tuning/nepochs_4/nepochs_{mode}_dim{dim}_all.png'),
                    dpi=300
                )
                plt.close('all')


def n_epochs_experiment():

    # ### Run gridsearch over n_epochs with faster lr decay -> does the performance stabilize?

    import os
    import matplotlib
    import scanpy as sc
    from sklearn.model_selection import train_test_split, PredefinedSplit
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier

    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Define some auxiliary functions
    def log_trafo_w_cutoff_100(adata: sc.AnnData):
        cutoff = 100
        x = adata.X
        x = np.log(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
        adata.X = x

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

    # ##################
    # ### Set parameters and flags here
    # Preprocessing
    # prepr_flavour = 'arcsinh'
    # add_prepr_kwargs = {'cofactor': 150}

    # prepr_flavour = 'custom'
    # add_prepr_kwargs = {'preprocessing_method': log_trafo_w_cutoff_100, }

    prepr_flavour = 'custom'
    add_prepr_kwargs = {'preprocessing_method': log10_trafo_w_cutoff_channel_wise, }

    # Cross validation, data split
    cross_validation = False
    cv = 5
    val_size = 0.32  # Only relevant when cross_validation is False

    # Channels to be used for training and testing the SOM
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    # Parameter grid
    param_grid_nepochs = {
        'n_epochs': [
            10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
        ],
        'som_dimensions': [(10, 10), (20, 20)],
        'gaussian_neighborhood_sigma': [0.1, ],
        'radius_0': [-0.5, ],
        'radius_n': [0.75, ],
        'radius_cooling': ['linear', ],
        'learning_rate_0': [0.2],
        'learning_rate_n': [0.01],
        'learning_rate_decay': ['linear', ],
    }

    # ### Define path where results are stored, create dir if necessary
    prepr_kwargs_str = ''.join(
        f"{key}{value.__name__ if callable(value) else value}" for key, value in add_prepr_kwargs.items()
    )

    base_p = os.path.join(
        os.getcwd(),
        f'results/param_tuning/nepochs_experiment/{prepr_flavour + prepr_kwargs_str}_'
        f'{f"cv{cv}" if cross_validation else f"nocv{val_size}"}'
    )
    if not os.path.exists(base_p):
        os.makedirs(os.path.join(base_p, 'data_handling'))

    # ### Load and process .fcs data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour=prepr_flavour,
        additional_preprocessing_kwargs=add_prepr_kwargs,
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # Perform data split into train- and test-set, load data split used for baseline results (60, 40)
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # ### Create a dataloader for the train-/test-data
    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')

    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))

    if not cross_validation:
        # Due to running time constraints generate a train test split of the data, No cross validation!
        # Split the data into train (70%) and test (30%) sets
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
        X=x_train.copy(), y=y_train.copy(), param_grid=param_grid_nepochs, cv=cv, scoring='internal', refit=False,
    )

    print(pd.DataFrame(som_c.grid_search_.cv_results_))

    # ### Save SOM the classifier
    som_c.save(filepath=base_p)


def view_results_nepochs_experiment():
    from flowsrc.flowsom import SomClassifier
    from flowsrc.floweval import plot_param_lineplot
    import matplotlib
    matplotlib.use('Agg')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/param_tuning/nepochs_experiment')

    modes = [
        'custompreprocessing_methodlog10_trafo_w_cutoff_channel_wise_nocv0.32',
        'custompreprocessing_methodlog_trafo_w_cutoff_100_nocv0.32', 'arcsinhcofactor150_nocv0.32'
    ]

    for mode in modes:
        print(f'# ### Preprocessing mode: {mode}')

        som_c = SomClassifier.load(os.path.join(base_p, mode, 'som_classifier.pkl'))

        res_df = pd.DataFrame(som_c.grid_search_.cv_results_)

        res_df_10 = res_df[res_df['param_som_dimensions'] == (10, 10)].copy()
        res_df_10.to_csv(os.path.join(base_p, mode, 'res_df_10.csv'))

        print(res_df_10.sort_values(by='param_n_epochs')[
                  ['mean_fit_time', 'mean_score_time', 'param_n_epochs', 'mean_test_score', 'rank_test_score']]
              )

        plot_param_lineplot(
            res_df=res_df_10, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
            markersize=3.0, x_axis_grid=True, dpi=300)
        plt.savefig(os.path.join(base_p, mode, f'nepochs_dim10.png'), dpi=300)
        plt.close('all')

        res_df_20 = res_df[res_df['param_som_dimensions'] == (20, 20)].copy()
        res_df_20.to_csv(os.path.join(base_p, mode, 'res_df_20.csv'))

        print(res_df_20.sort_values(by='param_n_epochs')[
                  ['mean_fit_time', 'mean_score_time', 'param_n_epochs', 'mean_test_score', 'rank_test_score']]
              )

        plot_param_lineplot(
            res_df=res_df_20, x_col='param_n_epochs', y_col='mean_test_score', xlog10=True, custom_x_ticks='auto',
            markersize=3.0, x_axis_grid=True, dpi=300)
        plt.savefig(os.path.join(base_p, mode, f'nepochs_dim20.png'), dpi=300)
        plt.close('all')


def grid_def_helper():
    import os
    import pandas as pd
    from flowsrc.flowsom import SomClassifier

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # Results for neighborhood params
    somc = SomClassifier.load(filepath=os.path.join(os.getcwd(), 'results/set_wise_param_tuning/neigh'))

    res_df = pd.DataFrame(somc.grid_search_.cv_results_)
    res_df = res_df[[
        'param_gaussian_neighborhood_sigma', 'param_radius_0', 'param_radius_n',
        'param_radius_cooling', 'mean_test_score', 'rank_test_score']
    ]
    res_df = res_df.sort_values('rank_test_score', axis=0)
    print(res_df)

    # ### Results for lr params
    somc = SomClassifier.load(filepath=os.path.join(os.getcwd(), 'results/set_wise_param_tuning/lr'))

    res_df = pd.DataFrame(somc.grid_search_.cv_results_)

    print(res_df)
    res_df = res_df[[
        'param_learning_rate_0', 'param_learning_rate_n', 'param_learning_rate_decay',
        'mean_test_score', 'rank_test_score']
    ]
    res_df = res_df.sort_values('rank_test_score', axis=0)
    print(res_df)


def param_tuning():

    import os
    import matplotlib
    import scanpy as sc
    from typing import Tuple
    from sklearn.model_selection import train_test_split, PredefinedSplit
    from flowsrc.flowdata import FlowDataManager
    from flowsrc.flowsom import SomClassifier

    # ### Set the matplotlib backend
    matplotlib.use('Agg')

    # ### Set pandas print options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Define stratified downsampling function
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

    def log_trafo_w_cutoff_100(adata: sc.AnnData):
        cutoff = 100
        x = adata.X
        x = np.log(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
        adata.X = x

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

    # ##################
    # ### Set parameters and flags here
    # Preprocessing
    # prepr_flavour = 'arcsinh'
    # add_prepr_kwargs = {'cofactor': 150}
    # prepr_flavour = 'custom'
    # add_prepr_kwargs = {'preprocessing_method': log_trafo_w_cutoff_100, }
    prepr_flavour = 'custom'
    add_prepr_kwargs = {'preprocessing_method': log10_trafo_w_cutoff_channel_wise, }

    # Downsampling
    downsampling = False
    ds_frac = 0.25  # Only relevant when downsampling is True

    # Cross validation, data split
    cross_validation = False
    cv = 5
    val_size = 0.32  # Only relevant when cross_validation is False
    # ##################

    # ### Define path where results are stored, create dir if necessary
    prepr_kwargs_str = ''.join(
        f"{key}{value.__name__ if callable(value) else value}" for key, value in add_prepr_kwargs.items()
    )
    base_p = os.path.join(
        os.getcwd(),
        f'results/param_tuning/{prepr_flavour + prepr_kwargs_str}_{f"ds{ds_frac}" if downsampling else "nods"}_'
        f'{f"cv{cv}" if cross_validation else f"nocv{val_size}"}'
    )
    if not os.path.exists(base_p):
        os.makedirs(os.path.join(base_p, 'data_handling'))

    # ### Load and process .fcs data
    filename_list = os.listdir(os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format'))

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour=prepr_flavour, additional_preprocessing_kwargs=add_prepr_kwargs,
        memory_saving=False, channel_name_reference=0,
        data_file_path='input/concatenated_labeled_fcs_format', save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )

    # Perform data split into train- and test-set, load data split used for baseline results (60, 40)
    flow_manager.perform_data_split(
        data_split=pd.read_csv(os.path.join(os.getcwd(), 'results/baseline/data_handling/data_split.csv'),
                               index_col=0))

    # ### Define channels to be used for training and testing the SOM and create a dataloader for the train-/test-data
    channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']

    dl_train = flow_manager.create_data_loader(
        data_set='train', channels=channels, layer_key=None, label_key='population', label_layer_key='original',
        return_data_loader='np_array', batch_size=-1, shuffle=True, on_disk=True, filename='data_train.npy')

    # Batch size was set to -1 => dl contains one array with all the data, extract it
    x_train, y_train = next(iter(dl_train))

    if downsampling:
        x_train, y_train = stratified_downsampling(X=x_train, y=y_train, ds_fraction=ds_frac, random_seed=42)

    if not cross_validation:
        # Due to running time constraints generate a train test split of the data, No cross validation!
        # Split the data into train (70%) and test (30%) sets
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

    # ### Define the parameter grid
    n_epochs = 10
    # n_epochs = 100  # channel wise cutoff, dummy_param_grid, no ds => macro f1: 0.9683004262969135

    # Grid with the top 2/3/4 values from the parameter influence study
    param_grid_topn_paramstudy = {
        'som_topology': ['planar', ],
        'som_grid_type': ['rectangular', ],
        'som_dimensions': [(40, 40), (35, 35)],
        'neighborhood': ['gaussian', ],
        'gaussian_neighborhood_sigma': [0.1, 0.25, 0.5],
        'initialization': ['pca', ],
        'initial_codebook': [None, ],
        'n_epochs': [n_epochs, ],
        'radius_0': [-0.7, -0.5, -0.8],  # 7, 5, 8, 6
        'radius_n': [0.01, 0.75, 0.25],
        'radius_cooling': ['linear', ],
        'learning_rate_0': [0.2, 0.01, 0.8],  # 0.2, 2.0, 0.01, 0.8
        'learning_rate_n': [0.1, 0.005, 0.09, 0.001],
        'learning_rate_decay': ['linear', ],
    }

    # Grid with the top 4 combinations of the set wise results
    param_grid_topn_setwise = {
        'som_topology': ['planar', ],
        'som_grid_type': ['rectangular', ],
        'som_dimensions': [(40, 40), (35, 35)],
        'neighborhood': ['gaussian', ],
        'gaussian_neighborhood_sigma': [0.1, 0.5, 0.25],  # 0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.25
        'initialization': ['pca', ],
        'initial_codebook': [None, ],
        'n_epochs': [n_epochs, ],
        'radius_0': [-0.5, -0.8],  # 5, 5, 5, 8, 5, 5, 5
        'radius_n': [0.75, 0.5, 0.01, 0.25],  # 0.75, 0.5, 0.01, 0.01, 0.25, 0.1, 0.75
        'radius_cooling': ['linear', 'exponential'],  # lin, lin, lin, exp, lin, lin, exp
        'learning_rate_0': [1.0, 0.6, 0.2],  # 1.0, 0.6, 0.1/0.1, 0.2, 1.0, 0.2
        'learning_rate_n': [0.1, 0.01],  # 0.1, 0.1, 0.1/0.1, 0.01, 0.001, 0.1
        'learning_rate_decay': ['linear', 'exponential'],  # lin, exp, lin/exp, lin, lin
    }

    dummy_param_grid = {
        'som_topology': ['planar'],
        'som_grid_type': ['rectangular', ],
        'som_dimensions': [(10, 10), ],
        'neighborhood': ['gaussian', ],
        'gaussian_neighborhood_sigma': [0.5, ],
        'initialization': ['pca', ],
        'initial_codebook': [None, ],
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
        'initial_codebook': [None, ],
        'n_epochs': n_epochs,
        'radius_0': [-0.5, -0.75],
        'radius_n': [0.75, 0.01, 0.25],
        'radius_cooling': ['linear', ],
        'learning_rate_0': [1.0, 0.2],
        'learning_rate_n': [0.1, 0.01],
        'learning_rate_decay': ['linear', ],
    }

    # ### Instantiate the SOM classifier
    som_c = SomClassifier(verbosity=2)

    # ### Perform the hyperparameter tuning
    som_c.hyperparameter_tuning(
        X=x_train.copy(), y=y_train.copy(), param_grid=param_grid, cv=cv, scoring='internal', refit=False,
    )

    # ### Save SOM the classifier
    som_c.save(filepath=base_p)

    sc = SomClassifier.load(filepath=base_p)
    print(pd.DataFrame(sc.grid_search_.cv_results_))


def check_flowcap():
    # Pytometry uses older numpy functionality???
    # conda create -n flowcap numpy=1 pandas scanpy matplotlib pytometry scikit-learn imbalanced-learn pyarrow
    # conda install pytorch torchvision torchaudio cpuonly -c pytorch

    # conda create -n flowcap numpy=1 pandas scanpy python-igraph leidenalg matplotlib pytometry scikit-learn imbalanced-learn pyarrow pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch::cpuonly -y

    # conda create -n flowcap numpy=1 pandas scanpy python-igraph leidenalg matplotlib pytometry scikit-learn imbalanced-learn pyarrow datashader=0.14.4 dask=2023.9.0 pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch::cpuonly -y

    import pytometry as pm
    import readfcs

    base_p = os.path.join(os.getcwd(), 'results/check_flowcap')
    if not os.path.exists(base_p):
        os.makedirs(os.path.join(base_p, 'data_handling'))

    # Here data loading works
    data_p = os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format')
    fn = '20150210-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019652 001.fcs'
    # adata = pm.io.read_fcs(os.path.join(data_p, fn))
    adata = readfcs.view(os.path.join(data_p, fn))
    print(adata)

    print('######')

    # Here data loading does not work
    data_p = os.path.join(os.getcwd(), 'input/flowcap_I_nd')
    fn = '002.fcs'
    # adata = pm.io.read_fcs(os.path.join(data_p, fn))
    adata = readfcs.view(os.path.join(data_p, fn))
    print(adata)

    # Probably because PnS (shortname for the longer channel name stored at PnN) are missing
    # Pytometry uses readfcs (https://readfcs.lamin.ai/)
    # which in term uses flowio (https://flowio.readthedocs.io/en/latest/index.html)

    from flowio import FlowData

    # Load the problematic FCS file
    flow_data = FlowData(os.path.join(data_p, fn))
    npy_data = np.reshape(flow_data.events, (-1, flow_data.channel_count))
    print(flow_data)
    # print(dir(flow_data))
    # attrs = ['analysis', 'channel_count', 'channels', 'event_count', 'events', 'file_size', 'header', 'name', 'text',
    #          'write_fcs']
    # attrs = ['analysis', 'channel_count', 'channels', 'event_count', 'file_size', 'header', 'name', 'text',
    #          'write_fcs']
    # for a in attrs:
    #     print(f'# ### {a}:')
    #     print(getattr(flow_data, a))
    # ### This yields:
    # [{}, 12, {i: {'PnN': 'FS INT'}, 66541, some long list of floats?, 3196024, ..., <bound method FlowData.write_fcs of FlowData(002.fcs)>]
    # ### header:
    # {'version': '3.0', 'text_start': 256, 'text_stop': 1714, 'data_start': 2048, 'data_stop': 3196015,
    #  'analysis_start': 0, 'analysis_stop': 0}
    # ### text:
    # {'mode': 'L', 'datatype': 'F', 'par': '12', 'byteord': '4,3,2,1', 'fil': '002.fcs', 'p1n': 'FSC-A', 'p1b': '32',
    #  'p1r': '262144', 'p1e': '0,0', 'p2n': 'SSC-A', 'p2b': '32', 'p2r': '262144', 'p2e': '0,0', 'p3n': 'FITC-A',
    #  'p3b': '32', 'p3r': '6', 'p3e': '0,0', 'p4n': 'PerCP-Cy5-5-A', 'p4b': '32', 'p4r': '6', 'p4e': '0,0',
    #  'p5n': 'Pacific Blue-A', 'p5b': '32', 'p5r': '6', 'p5e': '0,0', 'p6n': 'Pacifc Orange-A', 'p6b': '32', 'p6r': '6',
    #  'p6e': '0,0', 'p7n': 'QDot 605-A', 'p7b': '32', 'p7r': '6', 'p7e': '0,0', 'p8n': 'APC-A', 'p8b': '32', 'p8r': '6',
    #  'p8e': '0,0', 'p9n': 'Alexa 700-A', 'p9b': '32', 'p9r': '6', 'p9e': '0,0', 'p10n': 'PE-A', 'p10b': '32',
    #  'p10r': '6', 'p10e': '0,0', 'p11n': 'PE-Cy5-A', 'p11b': '32', 'p11r': '6', 'p11e': '0,0', 'p12n': 'PE-Cy7-A',
    #  'p12b': '32', 'p12r': '6', 'p12e': '0,0', 'note': 'empty', 'originality': 'DataModified',
    #  'last_modified': '04-JAN-2010 20:01:27', 'last_last_modifier': 'naghaeep running flowCore version 1.11.16',
    #  'last_modifier': 'naghaeep running LogicleTransform, version alpha 2009-08-04.',
    #  'transformation_details': 'FSC-A:N/A;SSC-A:N/A;FITC-A:logicle(T=139340.45,M=4.5,W=0.5676668,A=0.0);PerCP-Cy5-5-A:logicle(T=73035.82,M=4.5,W=1.102671,A=0.0);Pacific Blue-A:logicle(T=136631.7,M=4.5,W=0.020792143,A=0.0);Pacifc Orange-A:logicle(T=139223.53,M=4.5,W=0.67307407,A=0.0);QDot 605-A:logicle(T=140307.73,M=4.5,W=0.69525903,A=0.0);APC-A:logicle(T=140544.02,M=4.5,W=1.1654433,A=0.0);Alexa 700-A:logicle(T=139400.16,M=4.5,W=1.069203,A=0.0);PE-A:logicle(T=121094.9,M=4.5,W=1.557299,A=0.0);PE-Cy5-A:logicle(T=11383.872,M=4.5,W=2.120766,A=0.0);PE-Cy7-A:logicle(T=73210.04,M=4.5,W=1.8580425,A=0.0)',
    #  'tot': '66541', 'beginstext': '0', 'endstext': '0', 'begindata': '2048', 'enddata': '3196015',
    #  'beginanalysis': '0', 'endanalysis': '0', 'nextdata': '0'}

    # ### Add missing $PnS fields to the channels attribute
    print(flow_data.channels)
    for i in range(1, flow_data.channel_count + 1):
        param_name = flow_data.channels[str(i)].get('PnN', f'Param_{i}')
        print(param_name)
        if "PnS" not in flow_data.channels[str(i)]:
            flow_data.channels[str(i)]["PnS"] = param_name  # Use $PnN as a placeholder for $PnS

    # Save the corrected FCS file
    # with open("corrected_file.fcs", "wb") as corrected_file:
    #     corrected_file.write(flow_data.export())
    flow_data.write_fcs(filename=os.path.join(data_p, 'zzz.fcs'))

    # Check file with readfcs
    adata = readfcs.view(os.path.join(data_p, 'zzz.fcs'))
    print(adata)

    # Load file with pytometry/readfcs
    adata = pm.io.read_fcs(os.path.join(data_p, 'zzz.fcs'))
    print(adata)

    print(adata.uns['meta'])



    '''from flowsrc.flowdata import FlowDataManager
    filename_list = os.listdir(data_p)
    print(filename_list)

    flow_manager = FlowDataManager(
        filename_list=filename_list, preprocessing_flavour=None,
        additional_preprocessing_kwargs=None,
        memory_saving=False, channel_name_reference=0,
        data_file_path=data_p, save_path=os.path.join(base_p, 'data_handling'),
        filenames={'fn_og_channel_names': 'og_channel_names.csv', }
    )'''


def check_flowcyt():

    # Problem : Label files are not available for download

    # Email flowcyt:
    # However, "A_graph.pt" and "sub_graph.pt" contain the 30 graph-patients already encoded as k-NN graphs together
    # with their cell labels. Otherwise,  you can either use the raw FCS data (uploaded in the above link) or follow
    # the instructions at https://github.com/LorenzoBini4/FlowCyt-Classification-Benchmark/tree/main/data
    # under the "Usage" section to exactly recreate the CSV along with their labels.

    # I. Try their instructions for recovering the labels from the graph
    # II. Create from cell-type-wise .fcs files (this is what the do ...)

    # I. Ran: python ./validation/FlowCyt-Classification-Benchmark/data/A_generation.py
    #    => Case_n.csv for each case/sample with all cell types and cell type labels:
    #    0: T Lymphocytes (O)
    #    1: B Lymphocytes (N)
    #    2: Monocytes (G)
    #    3: Mast cells (P)
    #    4: HSPCs (K)
    #    5: Others (B)
    # II. No need to implement this myself ...

    import pytometry as pm
    import readfcs
    from pathlib import Path

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    base_p = os.path.join(os.getcwd(), 'results/check_flowcyt')
    if not os.path.exists(base_p):
        os.makedirs(os.path.join(base_p, 'data_handling'))

    # ### Load and inspect one of the created .csv files
    df = pd.read_csv(
        os.path.join(os.getcwd(), 'validation/FlowCyt-Classification-Benchmark/data/data_original/Case_1.csv'))
    print(df.columns)
    print(df.index)
    print(df['label'].value_counts())
    print('######')

    def _load_data_files_to_anndata(filename_list, data_file_path, data_file_type, save_path, memory_saving=False):
        # ### Load AnnData objects into memory and store in list
        anndata_list = []
        unreadable_list = []
        if not memory_saving:
            for fn in filename_list:
                if data_file_type is None:
                    if fn.endswith('.fcs'):
                        data_file_type = 'fcs'
                    elif fn.endswith('.csv'):
                        data_file_type = 'csv'
                    else:
                        warnings.warn(f"Skipping invalid file: '{fn}'. Not a CSV or FCS file.")
                        unreadable_list.append(fn)
                        continue

                if data_file_type == 'fcs':
                    adata = pm.io.read_fcs(os.path.join(data_file_path, fn))
                elif data_file_type == 'csv':
                    pass

                adata.uns['filename'] = fn
                anndata_list.append(adata)
        # ### Load AnnData objects one by one and save them to disk
        else:
            for fn in filename_list:
                adata = pm.io.read_fcs(os.path.join(data_file_path, fn))
                adata.uns['filename'] = fn
                ad_fn = fn[:-4] + '.h5ad'
                adata.write_h5ad(filename=Path(os.path.join(save_path, ad_fn)))
                anndata_list.append(ad_fn)
                del adata
        return anndata_list


    check_fcs = True
    if check_fcs:
        # ### Define path to files
        data_p = os.path.join(os.getcwd(), 'input/flowcyt/raw')
        fn0 = 'Case3_A.fcs'
        fn1 = 'Case3_B.fcs'  # B, G, K, N, O, P

        # ### From comparing the pytometry loaded fcs (A vs others) we get:
        # Case1:
        # A-B: $PnG not in B, $PnR Ranges differ for some channels, $PnV not in B,
        #      'channel': channel names differ slightly, same for 'marker'
        # A-G: All the same
        # A-K: All the same
        # A-N: All the same
        # A-O: All the same
        # A-P: All the same
        # Case2: Same, Case3: Same => Assuming this structure is the same across all 30 cases

        # ### Load and view the header of some of the .fcs files with readfcs
        # adata = readfcs.view(os.path.join(data_p, fn0))
        # bdata = readfcs.view(os.path.join(data_p, fn1))
        # print(adata[0])
        # print(bdata[0])
        # print('######')

        # ### Load with pytometry and inspect the resulting anndata for some of the .fcs files
        adata = pm.io.read_fcs(os.path.join(data_p, fn0))
        bdata = pm.io.read_fcs(os.path.join(data_p, fn1))
        print(adata)
        print(bdata)

        print(np.union1d(adata.var.columns.to_numpy(), bdata.var.columns.to_numpy()))

        for v in np.union1d(adata.var.columns.to_numpy(), bdata.var.columns.to_numpy()):
            print(f'# ### {v}:')
            print('# adata:')
            try:
                # print(adata.var[v])
                adata.var[v]
                ad = True
            except KeyError:
                ad = False
                print(f'# {v} does not exist in adata.var')

            print('# bdata:')
            try:
                # print(bdata.var[v])
                bdata.var[v]
                bd = True
            except KeyError:
                bd = False
                print(f'# {v} does not exist in bdata.var')

            if ad and bd:
                same = np.all(adata.var[v].to_numpy() == bdata.var[v].to_numpy())
                print(f'# np.all(adata.var[v].to_numpy() == bdata.var[v].to_numpy()): {same}')

                if not same:
                    print(adata.var[v])
                    print(bdata.var[v])

        print('######')


        # ### Load one of our .fcs files
        data_p1 = os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format')
        fn3 = '20150312-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019509 001.fcs'
        cdata = pm.io.read_fcs(os.path.join(data_p1, fn3))
        print(cdata)
        for v in cdata.var.columns.to_numpy():
            print(cdata.var[v])


def test_anndata_to_fcs():

    import warnings
    import flowio
    import readfcs
    import  scanpy as sc
    import pytometry as pm

    from typing import Union

    # Pytometry uses readfcs (https://readfcs.lamin.ai/),
    # which in term uses flowio (https://flowio.readthedocs.io/en/latest/index.html)
    # Details:
    # - readfcs.read() creates instance of readfcs.ReadFCS for the corresponding file and call ReadFCS.to_anndata()
    # - ReadFCS.to_anndata() just creates the Anndata object, all critical  steps are done in the init method:
    # - meta = copy.deepcopy(self._meta) is assigned to adata.uns['meta']
    # - self._meta = self._flow_data.text
    # - self._flow_data is results of flowio.read_multiple_data_sets(...)[0] (flowio.FlowData instance)
    # - Before assignment meta is altered as follows:
    #   -

    def anndata_to_fcs(
            adata: sc.AnnData,
            filename: Union[str, None] = None,
            filepath: Union[str, None] = None,
    ):
        # ### Set filename and filepath
        if filename is None:
            try:
                filename = adata.uns['filename']
            except KeyError:
                filename = 'zzz_flowcy.fcs'
                warnings.warn(
                    'No filename was passed and the key "filename" was not found in adata.uns. Using flowcy.fcs'
                )

        if filepath is None:
            filepath = os.getcwd()

        with open(os.path.join(filepath, filename), 'wb') as fh:
            flowio.create_fcs(
                file_handle=fh,
                event_data=adata.X.flatten().tolist(),
                channel_names=adata.var['marker'].to_list(),
                opt_channel_names=adata.var['channel'].to_list(),
                metadata_dict={'dummy_key': 'dummy_val'}  # adata.uns['meta'],
            )

    data_p = os.path.join(os.getcwd(), 'input/flowcyt/raw')
    fn = 'Case1_A.fcs'
    # data_p = os.path.join(os.getcwd(), 'input/concatenated_labeled_fcs_format')
    # fn = '20150312-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019509 001.fcs'
    cdata = pm.io.read_fcs(os.path.join(data_p, fn))
    print(cdata)
    # for v in cdata.var.columns.to_numpy():
    #    print(cdata.var[v])

    print('#######')

    print(cdata.uns['meta'])

    print('#######')

    readfcs_obj = readfcs.ReadFCS(os.path.join(data_p, fn))
    print(readfcs_obj)
    # print(dir(readfcs_obj))

    print('#######')

    flowdata = flowio.FlowData(os.path.join(data_p, fn))
    print(flowdata)
    # print(dir(flowdata))
    # print(flowdata.header)
    print(flowdata.text)
    # Keywords are in lower case, corrected by readfcs
    # (see comment: https://github.com/laminlabs/readfcs/blob/main/readfcs/_core.py)

    anndata_to_fcs(adata=cdata)

    flowdata2 = flowio.FlowData(os.path.join(os.getcwd(), 'zzz_flowcy.fcs'))
    print('# ### Header of reloaded .fcs, flowio:')
    print(flowdata2.text)

    print('\n')

    bdata = readfcs.view(os.path.join(os.getcwd(), 'zzz_flowcy.fcs'))
    print('# ### Header of reloaded .fcs, readfcs:')
    print(bdata[0])


def check_flowcal():

    # ### Run with flowcyt2

    import FlowCal

    data_p = os.path.join(os.getcwd(), 'input/flowcyt/raw')
    fn = 'Case1_A.fcs'

    flcal = FlowCal.io.FCSData(os.path.join(data_p, fn))

    print(flcal)
    print(type(flcal))
    print(dir(flcal))


def time_get_surface_state():

    import time
    from scipy.spatial.distance import cdist
    from numba import njit, prange

    def _custom_get_surface_state(
            codebook: np.ndarray,
            data: np.ndarray,
    ):
        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> somdim0 * somdim1, n_features)

        codebook_reshaped = codebook.reshape(-1, codebook.shape[2])

        # ### Compute Euclidean distances in chunks for memory efficiency
        # Split data into 200 chunks along axis 0
        num_splits = 200
        chunks = np.array_split(data, num_splits, axis=0)
        # Compute for each chunk the euclidean distance to the codebook, stack results
        activation_map = np.vstack(
            [cdist(chunk, codebook_reshaped, metric='euclidean') for chunk in chunks]
        )

        return activation_map

    def _custom_get_surface_state2(
            codebook: np.ndarray,
            data: np.ndarray,
    ):
        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> somdim0 * somdim1, n_features)

        codebook_reshaped = codebook.reshape(-1, codebook.shape[2])

        # ### Compute Euclidean distances in chunks for memory efficiency
        # Split data into 200 chunks along axis 0
        num_splits = 1000
        chunks = np.array_split(data, num_splits, axis=0)
        # Compute for each chunk the euclidean distance to the codebook, stack results
        activation_map = np.vstack(
            [cdist(chunk, codebook_reshaped, metric='euclidean') for chunk in chunks]
        )

        return activation_map

    def _custom_get_surface_state3(
            codebook: np.ndarray,
            data: np.ndarray,
    ):
        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> somdim0 * somdim1, n_features)

        codebook_reshaped = codebook.reshape(-1, codebook.shape[2])

        # ### Compute Euclidean distances in chunks for memory efficiency
        # Split data into 200 chunks along axis 0
        num_splits = 10000
        chunks = np.array_split(data, num_splits, axis=0)
        # Compute for each chunk the euclidean distance to the codebook, stack results
        activation_map = np.vstack(
            [cdist(chunk, codebook_reshaped, metric='euclidean') for chunk in chunks]
        )

        return activation_map

    def _custom_get_surface_state5(
            codebook: np.ndarray,
            data: np.ndarray,
    ):
        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> somdim0 * somdim1, n_features)

        codebook_reshaped = codebook.reshape(-1, codebook.shape[2])

        num_splits = 1000
        chunks = np.array_split(data, num_splits, axis=0)
        # Compute for each chunk the euclidean distance to the codebook, stack results
        activation_map = np.vstack(
            [np.sqrt(np.sum((chunk[:, np.newaxis, :] - codebook_reshaped) ** 2, axis=2)) for chunk in chunks]
        )

        return activation_map

    def _custom_get_surface_state4(
            codebook: np.ndarray,
            data: np.ndarray,
    ):

        # ### Reshape codebook for efficient computation
        # (somdim0, somdim1, n_features) -> somdim0 * somdim1, n_features)

        codebook_reshaped = codebook.reshape(-1, codebook.shape[2])

        activation_map = _compute_distances(data, codebook_reshaped)

        return activation_map

    @njit(parallel=True, fastmath=True, nogil=True)
    def _compute_distances(data, codebook):
        num_samples, num_codebook = data.shape[0], codebook.shape[0]
        distances = np.empty((num_samples, num_codebook), dtype=np.float64)

        for i in prange(num_samples):  # Parallel loop
            for j in range(num_codebook):
                diff = data[i] - codebook[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))

        return distances

    dim = 10
    n_iter = 10

    cb = np.random.rand(dim, dim, 10)
    # inp = np.random.randn(1980290, 10)
    inp = np.random.randn(100000, 10)

    print(inp.shape)
    print(cb.shape)

    fcts = [_custom_get_surface_state4, _custom_get_surface_state, _custom_get_surface_state2, _custom_get_surface_state3, ]

    for i, f in enumerate(fcts):
        times = []
        for j in range(n_iter):
            st = time.time()
            act_map = f(cb, inp)
            et = time.time()
            times.append(et - st)
        print(f'{i+1}) avg time: {sum(times) / n_iter}')

    # ### Notes:
    # - 5) is out, is slowest regardless of som size
    # - som30, ds all:
    # -som10, ds all: (1980290, 10) (10, 10, 10)
    # 1) avg time: 27.544566226005553
    # 2) avg time: 16.467934679985046
    # 3) avg time: 17.797820925712585
    # 4) avg time: 1.6527134656906128


def check_restarted_training():
    import numpy as np
    from flowsrc.flowsom import SomClassifier

    np.random.seed(42)

    num_samples = 1000

    # Generate dataset 0
    means0 = np.array([0, 0, 0, 1, 1, 1])
    std_devs0 = np.array([1, 1, 1, 2, 2, 2])
    x0 = np.random.normal(loc=means0, scale=std_devs0, size=(num_samples, means0.shape[0]))
    y0 = np.zeros(num_samples)

    means1 = np.array([3, 3, 3, 2, 2, 2])
    std_devs1 = np.array([2, 2, 2, 1, 1, 1])
    x1 = np.random.normal(loc=means1, scale=std_devs1, size=(num_samples, means0.shape[0]))
    y1 = np.ones(num_samples)

    x_ds0= np.vstack((x0, x1))
    y_ds0 = np.hstack((y0, y1)).astype(int)

    shuffle_idx = np.random.permutation(num_samples * 2)
    x_ds0 = x_ds0[shuffle_idx, :]
    y_ds0 = y_ds0[shuffle_idx]

    # Generate dataset 1
    means0 = np.array([0, 0, 0, 1, 1, 1])
    std_devs0 = np.array([1, 1, 1, 2, 2, 2])
    x0 = np.random.normal(loc=means0, scale=std_devs0, size=(num_samples, means0.shape[0]))
    y0 = np.zeros(num_samples)

    means1 = np.array([3, 3, 3, 2, 2, 2])
    std_devs1 = np.array([2, 2, 2, 1, 1, 1])
    x1 = np.random.normal(loc=means1, scale=std_devs1, size=(num_samples, means0.shape[0]))
    y1 = np.ones(num_samples)

    x_ds1 = np.vstack((x0, x1))
    y_ds1 = np.hstack((y0, y1)).astype(int)

    shuffle_idx = np.random.permutation(num_samples * 2)
    x_ds1 = x_ds1[shuffle_idx, :]
    y_ds1 = y_ds1[shuffle_idx]

    # Instantiate SOM classifier
    som_c = SomClassifier(n_epochs=10, verbosity=2)

    # Train on dataset 0
    som_c.fit(x_ds0, y_ds0)

    x_codebook_ds0 = som_c.som_.codebook

    # Save SOM classifier to a file and reload
    som_c.save()
    del som_c
    som_c_0 = SomClassifier.load()
    som_c_1 = SomClassifier.load()
    som_c_2 = SomClassifier.load()

    # Automated restart
    som_c_0.fit(x_ds1, y_ds1)
    cb0 = som_c_0.som_.codebook
    cc0 = som_c_0.class_counts_per_unit_

    # Manual restart
    som_c_1.initial_codebook = x_codebook_ds0
    som_c_1.initialization = None
    som_c_1.fit(x_ds1, y_ds1)
    cb1 = som_c_1.som_.codebook
    cc1 = som_c_1.class_counts_per_unit_

    # Restart from scratch
    som_c_2.reset()
    som_c_2.fit(x_ds1, y_ds1)
    cb2 = som_c_2.som_.codebook
    cc2 = som_c_2.class_counts_per_unit_

    print('# ### Codebooks the same: ')
    print(np.array_equal(cb0, cb1))
    print(np.array_equal(cb1, cb2))
    print(np.array_equal(cb2, cb0))

    print('# ### Class counts the same: ')
    print(np.array_equal(cc0, cc1))
    print(np.array_equal(cc1, cc2))
    print(np.array_equal(cc2, cc0))

    print('# ### Test Somoclu')
    from somoclu import Somoclu
    som = Somoclu(n_columns=10, n_rows=10, initialization='pca', verbose=2)
    som.train(data=x_ds0, epochs=10)

    som2 = Somoclu(n_columns=10, initialcodebook=som.codebook.copy(), n_rows=10, initialization=None, verbose=2)

    som.train(data=x_ds1, epochs=10)
    som2.train(data=x_ds1, epochs=10)

    print(np.array_equal(som.codebook, som2.codebook))




def check_unlabeled_data_handling():
    import numpy as np
    from typing import Tuple
    from sklearn.metrics import f1_score
    from flowsrc.flowsom import SomClassifier

    np.random.seed(42)

    def create_dummy_dataset(
            n_unlabeled: int = 1000,
            n_labeled: int = 1000,
            unlabeld_label: int = -999
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Generate unlabeled data
        means0 = np.array([0, 0, 0, 1, 1, 1])
        std_devs0 = np.array([1, 1, 1, 2, 2, 2])
        x0 = np.random.normal(loc=means0, scale=std_devs0, size=(n_unlabeled, means0.shape[0]))
        y0 = np.full(n_unlabeled, unlabeld_label)

        means1 = np.array([3, 3, 3, 2, 2, 2])
        std_devs1 = np.array([2, 2, 2, 1, 1, 1])
        x1 = np.random.normal(loc=means1, scale=std_devs1, size=(n_unlabeled, means0.shape[0]))
        y1 = np.full(n_unlabeled, unlabeld_label)

        # Generate labeled data
        means0 = np.array([0, 0, 0, 1, 1, 1])
        std_devs0 = np.array([1, 1, 1, 2, 2, 2])
        x2 = np.random.normal(loc=means0, scale=std_devs0, size=(n_labeled, means0.shape[0]))
        y2 = np.zeros(n_labeled)

        means1 = np.array([3, 3, 3, 2, 2, 2])
        std_devs1 = np.array([2, 2, 2, 1, 1, 1])
        x3 = np.random.normal(loc=means1, scale=std_devs1, size=(n_labeled, means0.shape[0]))
        y3 = np.ones(n_labeled)

        x = np.vstack((x0, x1, x2, x3))
        y = np.hstack((y0, y1, y2, y3)).astype(int)

        shuffle_idx = np.random.permutation((n_unlabeled + n_labeled) * 2)
        x = x[shuffle_idx, :]
        y = y[shuffle_idx]

        return x, y

    nan_val = -999

    x_test, y_test = create_dummy_dataset(n_labeled=1000, n_unlabeled=0, unlabeld_label=-999)

    # All labeled
    x, y = create_dummy_dataset(n_unlabeled=0, n_labeled=1000, unlabeld_label=nan_val)
    somc = SomClassifier(n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    somc.fit(x, y)
    y_pred = somc.predict(x_test)
    y_proba = somc.predict_proba(x_test)
    print(
        f'# ### All labeled\n# F1: {f1_score(y_test, y_pred, average="macro")}\n'
        f'# y_pred: {y_pred}\n# y_proba: {y_proba}'
    )
    del somc, y_pred, y_proba

    # All unlabeled
    x, y = create_dummy_dataset(n_unlabeled=1000, n_labeled=0, unlabeld_label=nan_val)
    somc = SomClassifier(n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    somc.fit(x, y)
    y_pred = somc.predict(x_test)
    y_proba = somc.predict_proba(x_test)
    print(
        f'# ### All unlabeled\n# F1: {f1_score(y_test, y_pred, average="macro")}\n'
        f'# y_pred: {y_pred}\n# y_proba: {y_proba}'
    )
    del somc, y_pred, y_proba

    # Mixed
    x, y = create_dummy_dataset(n_unlabeled=1000, n_labeled=1000, unlabeld_label=nan_val)
    somc = SomClassifier(n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    somc.fit(x, y)
    y_pred = somc.predict(x_test)
    y_proba = somc.predict_proba(x_test)
    print(
        f'# ### All mixed\n# F1: {f1_score(y_test, y_pred, average="macro")}\n'
        f'# y_pred: {y_pred}\n# y_proba: {y_proba}'
    )
    del somc, y_pred, y_proba

    # Unlabeled, then retrain on labeled
    somc = SomClassifier(n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    x, y = create_dummy_dataset(n_unlabeled=1000, n_labeled=0, unlabeld_label=nan_val)
    somc.fit(x, y)
    x, y = create_dummy_dataset(n_unlabeled=0, n_labeled=100, unlabeld_label=nan_val)
    somc.fit(x, y)

    y_pred = somc.predict(x_test)
    y_proba = somc.predict_proba(x_test)
    print(
        f'# ### Unlabeled then labeled\n# F1: {f1_score(y_test, y_pred, average="macro")}\n'
        f'# y_pred: {y_pred}\n# y_proba: {y_proba}'
    )
    del somc, y_pred, y_proba

    # Labeled, then retrain on unlabeled, then annotate again
    somc = SomClassifier(n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    x0, y0 = create_dummy_dataset(n_unlabeled=0, n_labeled=1000, unlabeld_label=nan_val)
    somc.fit(x0, y0)
    y_pred = somc.predict(x_test)
    y_proba = somc.predict_proba(x_test)
    print(f'# ### Labeled then unlabeled: labeled:\n{vars(somc)}')

    x1, y1 = create_dummy_dataset(n_unlabeled=1000, n_labeled=0, unlabeld_label=nan_val)
    somc.fit(x1, y1)
    y_pred = somc.predict(x_test)
    y_proba = somc.predict_proba(x_test)
    print(f'# ### Labeled then unlabeled: unlabeled:\n{vars(somc)}')

    somc.annotate_som(X=x0, y=y0)
    print(f'# ### Labeled then unlabeled: annotate:\n{vars(somc)}')

    del somc, y_pred, y_proba


def check_annotate_and_export():

    import readfcs
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Tuple
    from flowsrc.flowsom import SomClassifier
    from flowsrc.flowdata import FlowDataManager

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    np.random.seed(42)

    def create_dummy_dataset(
            n_unlabeled: int = 1000,
            n_labeled: int = 1000,
            unlabeld_label: int = -999
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        # Generate unlabeled data
        means0 = np.array([0, 0, 0, 1, 1, 1])
        std_devs0 = np.array([1, 1, 1, 2, 2, 2])
        x0 = np.random.normal(loc=means0, scale=std_devs0, size=(n_unlabeled, means0.shape[0]))
        y0 = np.full(n_unlabeled, unlabeld_label)

        means1 = np.array([3, 3, 3, 2, 2, 2])
        std_devs1 = np.array([2, 2, 2, 1, 1, 1])
        x1 = np.random.normal(loc=means1, scale=std_devs1, size=(n_unlabeled, means0.shape[0]))
        y1 = np.full(n_unlabeled, unlabeld_label)

        # Generate labeled data
        means0 = np.array([0, 0, 0, 1, 1, 1])
        std_devs0 = np.array([1, 1, 1, 2, 2, 2])
        x2 = np.random.normal(loc=means0, scale=std_devs0, size=(n_labeled, means0.shape[0]))
        y2 = np.zeros(n_labeled)

        means1 = np.array([3, 3, 3, 2, 2, 2])
        std_devs1 = np.array([2, 2, 2, 1, 1, 1])
        x3 = np.random.normal(loc=means1, scale=std_devs1, size=(n_labeled, means0.shape[0]))
        y3 = np.ones(n_labeled)

        x = np.vstack((x0, x1, x2, x3))
        y = np.hstack((y0, y1, y2, y3)).astype(int)

        shuffle_idx = np.random.permutation((n_unlabeled + n_labeled) * 2)
        x = x[shuffle_idx, :]
        y = y[shuffle_idx]

        x_raw = pd.DataFrame(
            data=x,
            columns=[f'channel_{i}' for i in range(x.shape[1])]
        )
        x_raw['time'] = np.linspace(start=0, stop=16, num=x_raw.shape[0])

        return x, y, x_raw

    nan_val = -999

    # All labeled
    x0, y0, x0_raw = create_dummy_dataset(n_unlabeled=2, n_labeled=0, unlabeld_label=nan_val)
    x1, y1, x1_raw = create_dummy_dataset(n_unlabeled=0, n_labeled=2, unlabeld_label=nan_val)
    somc = SomClassifier(som_dimensions=(3, 3), n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    somc.fit(x0, y0)

    print('# ### x0, no labeled data')
    df = somc.annotate_and_export(X=x0)
    print(df)

    print('# ### [x0, x0]')
    df = somc.annotate_and_export(X=[x0, x0])
    print(df)

    somc.fit(x1, y1)

    print('# ### x1, with labeled data')
    df = somc.annotate_and_export(X=x1)
    print(df)

    print('# ### [x0, x1]')
    df = somc.annotate_and_export(X=[x0, x1])
    print(df)

    print('# ### [x0, x1], with raw')
    df = somc.annotate_and_export(X=[x0, x1], X_raw=[x0_raw, x1_raw])
    print(df)

    print('# ### All in')
    df = somc.annotate_and_export(
        X=[x0, x1], X_raw=[x0_raw, x1_raw], keep_unscaled=True, compute_umap=False, save_mode='fcs'
    )
    print(df)

    # Inspect saved .fcs file as a loaded AnnData object
    dm = FlowDataManager(filename_list=['som.fcs'])
    adata = dm.anndata_list[0]
    print(adata)
    print(adata.uns['meta'])

    # Inspect saved .fcs file with readfcs
    bdata = readfcs.view(os.path.join(os.getcwd(), 'som.fcs'))
    print(bdata)


    # ### Check Kaluza visualization annotations
    x0, y0, x0_raw = create_dummy_dataset(n_unlabeled=10000, n_labeled=1000, unlabeld_label=nan_val)
    somc = SomClassifier(som_dimensions=(10, 10), n_epochs=10, unlabeled_label=nan_val, verbosity=2)
    somc.fit(x0, y0)

    df = somc.annotate_and_export(X=x0, X_raw=x0_raw, keep_unscaled=True, compute_umap=False)

    fig, ax = plt.subplots(dpi=300)
    ax.scatter(df['bmu1_plot'].to_numpy(), df['bmu2_plot'].to_numpy(), s=1.0, edgecolors='none')

    df_circles = df.drop_duplicates(subset=['bmu1', 'bmu2', 'radius'], inplace=False)
    for bmu1, bmu2, radius in zip(df_circles['bmu1'], df_circles['bmu2'], df_circles['radius']):
        circle = plt.Circle((bmu1, bmu2), radius, alpha=0.6, edgecolor='red', facecolor='none', linewidth=0.5)
        ax.add_patch(circle)

    plt.savefig('zzz.png', dpi=300)



def check_fit_with_checkpoints():
    import pickle
    import numpy as np
    from typing import Tuple
    from sklearn.metrics import f1_score
    from flowsrc.flowsom import SomClassifier

    np.random.seed(42)

    def create_dummy_dataset(
            n_unlabeled: int = 1000,
            n_labeled: int = 1000,
            unlabeld_label: int = -999
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Generate unlabeled data
        means0 = np.array([0, 0, 0, 1, 1, 1])
        std_devs0 = np.array([1, 1, 1, 2, 2, 2])
        x0 = np.random.normal(loc=means0, scale=std_devs0, size=(n_unlabeled, means0.shape[0]))
        y0 = np.full(n_unlabeled, unlabeld_label)

        means1 = np.array([3, 3, 3, 2, 2, 2])
        std_devs1 = np.array([2, 2, 2, 1, 1, 1])
        x1 = np.random.normal(loc=means1, scale=std_devs1, size=(n_unlabeled, means0.shape[0]))
        y1 = np.full(n_unlabeled, unlabeld_label)

        # Generate labeled data
        means0 = np.array([0, 0, 0, 1, 1, 1])
        std_devs0 = np.array([1, 1, 1, 2, 2, 2])
        x2 = np.random.normal(loc=means0, scale=std_devs0, size=(n_labeled, means0.shape[0]))
        y2 = np.zeros(n_labeled)

        means1 = np.array([3, 3, 3, 2, 2, 2])
        std_devs1 = np.array([2, 2, 2, 1, 1, 1])
        x3 = np.random.normal(loc=means1, scale=std_devs1, size=(n_labeled, means0.shape[0]))
        y3 = np.ones(n_labeled)

        x = np.vstack((x0, x1, x2, x3))
        y = np.hstack((y0, y1, y2, y3)).astype(int)

        shuffle_idx = np.random.permutation((n_unlabeled + n_labeled) * 2)
        x = x[shuffle_idx, :]
        y = y[shuffle_idx]

        return x, y

    nan_val = -999

    x_val, y_val = create_dummy_dataset(n_labeled=1000, n_unlabeled=0, unlabeld_label=-999)

    p = os.path.join(os.getcwd(), 'zzz')
    os.makedirs(p, exist_ok=True)

    # All labeled
    x, y = create_dummy_dataset(n_unlabeled=0, n_labeled=1000, unlabeld_label=nan_val)
    somc = SomClassifier(n_epochs=100, som_dimensions=(5, 5), unlabeled_label=nan_val, verbosity=2)
    somc.fit_with_checkpoints(
        X=x,
        y=y,
        checkpoint_interval=50,
        tracking_interval=10,
        track_losses=True,
        track_impurities=True,
        X_y_val=(x_val, y_val),
        checkpoint_dir=p,
        save_tracked=True,
    )

    with open(os.path.join(p, 'res_dict.pkl'), 'rb') as f:
        d = pickle.load(f)

    print(d['quantization_loss'])
    print(d['topographical_loss'])
    print(d['f1_macro_train'])
    print(d['f1_macro_val'])


def check_lymphoma_dataset():
    import os
    from flowsrc.flowdata import FlowDataManager

    raw_data_p = os.path.join(os.getcwd(), 'input/raw/lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels')
    filename_list = os.listdir(raw_data_p)

    dm = FlowDataManager(
        data_file_names=filename_list[:3],
        data_file_type=None,
        data_file_path=raw_data_p,
        save_path=None,
        memory_saving=False,
        verbosity=1,
    )

    # ### Load .fcs to anndata
    dm.load_data_files_to_anndata()
    # print(dm.anndata_list_)

    adata = dm.anndata_list_[0]
    print(adata)
    print(adata.var_names)
    print(adata.X)

    # ### Align channel names, save to file
    dm.align_channel_names(reference_channel_names=0, out_filename='og_channel_names.csv')

    # ### Check the sample sizes
    dm.check_sample_sizes()

    # ### Relabel data
    dm.relabel_data(
        data_set='all',
        old_to_new_label_mapping={1: 0, 9: 0, 7: 1, 10: 1},
        label_key='population',
        label_layer_key=None,
        new_label_key='new_labels',
    )

    print(dm.anndata_list_[0].obs['new_labels'])

    # ### Apply preprocessing transformation
    prepr_kwargs = {'cutoff': 100}
    dm.sample_wise_preprocessing(flavour='log10_w_cutoff', save_raw_to_layer='no_trafo', **prepr_kwargs)

    adata = dm.anndata_list_[0]
    print(adata)
    print(adata.X)

    # ### Perform data split
    dm.perform_data_split(data_split=(0.75, 0.25), save=True, **{'shuffle': True, 'random_state': 42})

    # print(dm.train_data_)
    # print(dm.val_data_)
    # print(dm.test_data_)

    # ### Create dataloader
    label_key = 'population'
    channels = [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
    ]

    dl = dm.create_data_loader(
        data_set='all', channels=channels, label_key=label_key, batch_size=-1, label_layer_key='no_trafo')

    x, y = next(iter(dl))

    print('######')
    print(x)
    print(y)
    y_unique, counts = np.unique(y, return_counts=True)
    print(y_unique)
    print(counts)

    # ### Create dataloader, binary
    dl_binary = dm.create_data_loader(
        data_set='all', channels=channels, label_key='new_labels', batch_size=-1, label_layer_key=None)

    x_binary, y_binary = next(iter(dl_binary))

    print('# ### binary')
    y_unique_binary, counts_binary = np.unique(y_binary, return_counts=True)
    print(y_unique_binary)
    print(counts_binary)

    # ### Check the class balance in the dataset
    dm.check_class_balance(
        data_loader=dl,
        save=True,
        plot=True,
        save_path=dm.save_path,
        verbosity=dm.verbosity,
    )

    print('# ### binary')
    dm.check_class_balance(
        data_loader=dl_binary,
        save=True,
        plot=True,
        filename_df='class_balances_binary.csv',
        filename_plot='class_balances_binary.png',
        save_path=dm.save_path,
        verbosity=dm.verbosity,
    )

    # ### Save to numpy
    dm.datalist_to_numpy(
        data_list=dm.anndata_list_,
        sample_wise=False,
        filename_suffix='_all',
        save_path=dm.save_path,
        data_path=None,
        channels=channels,
        layer_key=None,
        label_key='population',
        label_layer_key='no_trafo',
        shuffle=False,
    )

    x = np.load(os.path.join(dm.save_path, 'x_all.npy'))
    y = np.load(os.path.join(dm.save_path, 'y_all.npy'))

    print(x)

    dm.datalist_to_numpy(
        data_list=dm.anndata_list_,
        sample_wise=True,
        filename_suffix='_all_binary',
        save_path=dm.save_path,
        data_path=None,
        channels=channels,
        layer_key=None,
        label_key='new_labels',
        label_layer_key=None,
        shuffle=False,
    )

    x = np.load(os.path.join(dm.save_path, 'x_all_binary_sample_00.npy'))
    y = np.load(os.path.join(dm.save_path, 'y_all_binary_sample_00.npy'))

    print(x)


def plot_sample_sizes():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    save_p = os.path.join(os.getcwd(), 'results_new/plots/sample_sizes')
    os.makedirs(save_p, exist_ok=True)

    data_sets = ['aml', 'flowcyt', 'lymphoma_tube1', 'lymphoma_tube2']

    for data_set in data_sets:

        df_p = os.path.join(os.getcwd(), f'input/np_files/{data_set}/arcsinhcofactor150/data_handling/sample_sizes.csv')

        df = pd.read_csv(df_p, index_col=0)

        y_vals = df.loc[~df['sample'].isin(['mean', 'std', 'total']), 'n_events'].to_numpy()
        x_vals = list(range(y_vals.shape[0]))

        m = df.loc[df['sample'] == 'mean', 'n_events'].squeeze()
        std = df.loc[df['sample'] == 'std', 'n_events'].squeeze()
        total = df.loc[df['sample'] == 'total', 'n_events'].squeeze()

        fig, ax = plt.subplots(dpi=300)

        ax.bar(x_vals, np.sort(y_vals), color='skyblue', edgecolor='black')

        ax.set_xlabel('Sample id')
        ax.set_ylabel('n events per sample')

        xtick_vals = np.linspace(min(x_vals), max(x_vals), 6, dtype=int)
        ax.set_xticks(xtick_vals)

        ax.axhline(m, color='red', linestyle='-', linewidth=2, label=f'Mean: {m:.2f}', alpha=0.6)

        ax.axhline(m - std, color='orange', linestyle='dashed', label=f'Std: {std:.2f}', linewidth=2, alpha=0.5)
        ax.axhline(m + std, color='orange', linestyle='dashed', linewidth=2, alpha=0.5)

        ax.legend()

        ax.set_title(f'n samples: {y_vals.shape[0]}, Total events: {int(total)}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_p, f'{data_set}.png'))



def check_gpus():
    import torch

    print("GPU available: ", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Device 0 name: ", torch.cuda.get_device_name(0))  # NVIDIA RTX 6000 Ada Generation
    print("Device 1 name: ", torch.cuda.get_device_name(1))  # NVIDIA RTX 6000 Ada Generation

    print("Torch version: ", torch.__version__)
    print("Torch cuda version: ", torch.version.cuda)


def check_som_gpu_training():

    # mamba create -n flowcy python=3.12 -y
    # mamba install cuda cuda-nvcc cuda-toolkit -y
    # mamba install somoclu numpy pandas matplotlib seaborn scanpy python-igraph leidenalg scikit-learn scipy imbalanced-learn umap-learn numba -y
    # pip install pytometry flowio torch torchvision torchaudio

    # export CUDAHOME = / usr / local / cuda

    import time
    import numpy as np
    from sklearn.metrics import f1_score
    from flowsrc.flowsom import SomClassifier

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

    som_c = SomClassifier(som_dimensions=(30, 30), n_epochs=100, kernel_type=1, verbosity=2)

    som_c.fit(x, y)

    y_pred = som_c.predict(x)

    print(f'# ### F1: {f1_score(y, y_pred, average="macro")}')


if __name__ == '__main__':

    set_random_seed()

    # check_load_data()

    # check_pytometry()

    # test_anndata()

    # check_fcs_files()

    # dummy_examples()

    # test_data_manager()

    # some_plots()

    # check_deepcopy()

    # speed_comparison_som()

    # train_classifier()

    # test_flowsom()

    # test_plot()

    # test_save_datasplit()

    # generate_baseline_results()

    # test_sampling_strategies()

    # train_balanced()

    # generate_baseline_results_less_channels()

    # train_balanced_less_channels()

    # test_predict_proba()

    # train_epoch_wise()

    # train_epoch_wise_balanced()

    # eval_sample_wise()

    # param_influence_study()

    # view_results_param_influence_study_nepochs()

    # view_results_param_influence_study()

    # set_wise_hyperparameter_tuning()

    # view_results_set_wise_lr()

    # view_results_set_wise_neigh()

    # input_data_processing_study()

    # view_results_input_data_processing_study()

    # majority_class_downsampling_study()

    # view_results_majority_class_downsampling_study()

    # estimate_n_epochs()

    # view_results_n_epochs_estimation()

    # estimate_n_epochs_2()

    # view_results_n_epochs_2()

    # n_epochs_experiment()

    # view_results_nepochs_experiment()

    # grid_def_helper()

    # time_get_surface_state()

    # param_tuning()



    # check_flowcap()

    # check_flowcyt()

    # test_anndata_to_fcs()

    # check_flowcal()

    # check_restarted_training()

    # check_unlabeled_data_handling()

    # check_annotate_and_export()

    # check_fit_with_checkpoints()

    # check_lymphoma_dataset()

    plot_sample_sizes()

    # check_gpus()

    # check_som_gpu_training()

    print('done')

    # Todo Flowcy:
    #  - Analyze how much data is needed for training -> plot: n train samples vs test performance
    #  - Implement back compatibility to .fcs
    #  - Look at GateNet (DL, 2024)
    #  - check gpu training
    #  - Do eval run (channel wise f1) sample-wise on train samples,
    #    flag samples where pred for one class is worse than the mean (in pipeline, need manager for this)
    #  - Add a confidence threshold: if y_proba <= thresh, predict as unknown
    #    (default = 0, also produce results for 0.5 = absolute mehrheit)
    #  - Add preprocessing flavours
    #  - Remove logger

    # Todo others:
    #  - Opt in to new phd reg (TechFak)
    #  - Dienstreise Zugang

    # Todo, tomorrow:
    #  - Write .fcs back compatibility -> Stefan: Wie pred labels in .fcs, see: main.py, pipeline.py
    #  - Implement results with confidence thresh
    #  - restart parameter training, takes too long (train on subset and then afterwards optimize the nepochs)
    #  - Continue with framework for generating publishable results !!! -> main_experiments
    #  - Friday: run param tuning, generate results for all methods/ds, plots, maybe gatenet


    # Todo: Remember
    #  - GateNet install:
    #    conda create -n gatenet python=3.9
    #    conda activate gatenet
    #    mamba install fastai anaconda
    #    mamba install -c bioconda fcsparser
    #    mamba install -c conda-forge pyarrow
    #    mamba install -c anaconda seaborn




# Notes:
# - Confusion matrix: row = true label, columns = predicted label
# - GMC uses simple arcsinh trafo for preprocessing
# - Dgcytof makes no comment on preprocessing/data transformation, also use arcsinh
# - ours: 75 train_samples, 2918593.8 aprox_n_train_events;  flowcyt: 22.5, 15941149.5