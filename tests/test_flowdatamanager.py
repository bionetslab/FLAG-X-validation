
import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from flagx.io import FlowDataManager


# ------------------------------------------------------------
# Global expected channels (MUST match your test files)
# ------------------------------------------------------------
TEST_DATA_DIR = Path(__file__).parent / 'test_data'
EXPECTED_CHANNELS = [
    'FS INT', 'SS PEAK', 'SS INT', 'SS TOF',
    'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD',
    'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC',
    'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
    'FL10 INT_CD45-KO', 'TIME', 'label'
]
N_CHANNELS = len(EXPECTED_CHANNELS)
N_FILES = 5


# ------------------------------------------------------------
# Parametrization: test CSV and FCS separately
# ------------------------------------------------------------
@pytest.fixture(scope='session', params=['csv', 'fcs'])
def data_format(request):
    return request.param


@pytest.fixture(scope='function')
def test_files(data_format):
    """Collect only files of the correct format."""
    if data_format == 'csv':
        files = sorted(TEST_DATA_DIR.glob('*.csv'))
    else:
        files = sorted(TEST_DATA_DIR.glob('*.fcs'))

    assert len(files) == N_FILES, f'Expected {N_FILES} {data_format} files.'

    return [f.name for f in files]


@pytest.fixture(scope='function')
def fdm(test_files, tmp_path_factory):
    """Separate FlowDataManager instance for each format."""
    save_dir = tmp_path_factory.mktemp('fdm_output')

    manager = FlowDataManager(
        data_file_names=test_files,
        data_file_path=str(TEST_DATA_DIR),
        save_path=str(save_dir),
        verbosity=0,
    )
    return manager


# ------------------------------------------------------------
# 1. Initialization & Loading
# ------------------------------------------------------------
def test_initialization_and_load(fdm):
    fdm.load_data_files_to_anndata()

    assert len(fdm.anndata_list_) == N_FILES
    assert fdm.invalid_files_ == []

    for adata in fdm.anndata_list_:
        assert adata.n_vars == N_CHANNELS
        assert list(adata.var_names) == EXPECTED_CHANNELS
        assert 'filename' in adata.uns


def test_invalid_filetype_raises(fdm):
    fdm._data_file_names[0] = 'Case_1.txt'
    with pytest.raises(ValueError):
        fdm.load_data_files_to_anndata()


def test_invalid_files_sorted_out(fdm):
    # Remove last file, append one correct one and two invalid
    fdm._data_file_names.pop()
    fdm._data_file_names.append('Case_5.fcs')
    fdm._data_file_names.append('Case_5.csv')
    fdm._data_file_names.append('Case_6.txt')

    fdm.load_data_files_to_anndata()

    assert len(fdm.invalid_files_) == 2


# ------------------------------------------------------------
# 2. Sample size checking
# ------------------------------------------------------------
def test_check_sample_sizes(fdm):
    fdm.load_data_files_to_anndata()

    out_file = 'sample_sizes.csv'
    fdm.check_sample_sizes(filename_sample_sizes_df=out_file)

    df = pd.read_csv(Path(fdm.save_path) / out_file)
    assert df.shape[0] == N_FILES + 3
    assert 'sample' in df.columns
    assert 'n_events' in df.columns


def test_plot_sample_size_df():
    # Build a dummy sample-size DataFrame
    df = pd.DataFrame({
        'sample': ['s1', 's2', 's3', 'mean', 'std', 'total'],
        'n_events': [100, 200, 150, 150, 50, 450]
    })

    # Call the plotting function
    ax = FlowDataManager.plot_sample_size_df(df)


    assert isinstance(ax, matplotlib.axes.Axes)

    bar_patches = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert len(bar_patches) == 3  # s1, s2, s3

    hlines = [l for l in ax.lines if l.get_linestyle() in ["-", "--"]]
    assert len(hlines) == 3

    assert ax.get_xlabel() == 'Sample id'
    assert ax.get_ylabel() == 'n events per sample'

    title = ax.get_title()
    assert 'n samples: 3' in title
    assert 'Total events: 450' in title

    plt.close(ax.figure)


# ------------------------------------------------------------
# 3. Channel name alignment
# ------------------------------------------------------------
def test_align_channel_names(fdm):
    fdm.load_data_files_to_anndata()

    # Alter channel name in 2nd anndata
    adata = fdm.anndata_list_[1]
    var_names = list(adata.var_names)
    var_names[0] = 'channel 0'
    var_names[3] = 'channel_xyz'
    adata.var_names = var_names
    fdm.anndata_list_[1] = adata

    fdm.align_channel_names(reference_channel_names=0, filename_log_df='channel_log.csv')

    with pytest.warns(UserWarning):
        fdm.verbosity = 1
        fdm.check_og_channel_names_df()
        fdm.verbosity = 0

    log_df = pd.read_csv(Path(fdm.save_path) / 'channel_log.csv', index_col=0)
    assert log_df.shape == (N_FILES, N_CHANNELS + 1)

    for adata in fdm.anndata_list_:
        assert list(adata.var_names) == EXPECTED_CHANNELS

    # Custom rename
    custom_map = {old: f'ch{i}' for i, old in enumerate(EXPECTED_CHANNELS)}
    fdm.align_channel_names(reference_channel_names=custom_map)
    assert list(fdm.anndata_list_[0].var_names) == [f'ch{i}' for i in range(N_CHANNELS)]


# ------------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------------
@pytest.mark.parametrize('flavour', [
    'logicle', 'arcsinh', 'biexp', 'log10_w_cutoff', 'log10_w_custom_cutoffs', 'custom'
])
def test_preprocessing(fdm, flavour):
    fdm.load_data_files_to_anndata()

    if flavour == 'log10_w_cutoff':
        fdm.sample_wise_preprocessing(flavour=flavour, save_raw_to_layer='raw', cutoff=100)
    elif flavour == 'log10_w_custom_cutoffs':
        cutoffs = {channel: i * 10 for i, channel in enumerate(EXPECTED_CHANNELS[:-2])}
        fdm.sample_wise_preprocessing(flavour=flavour, save_raw_to_layer='raw', cutoffs=cutoffs)
    elif flavour == 'custom':
        def custom_prepr_fn(ad, param):
            ad.X = ad.X * param
            return ad
        fdm.sample_wise_preprocessing(
            flavour=flavour, save_raw_to_layer='raw',
            preprocessing_method=custom_prepr_fn, param=2.0
        )
    else:
        fdm.sample_wise_preprocessing(flavour=flavour, save_raw_to_layer='raw')

    for adata in fdm.anndata_list_:
        assert 'raw' in adata.layers
        assert not np.allclose(adata.X, adata.layers['raw'])


# ------------------------------------------------------------
# 5. Data splitting
# ------------------------------------------------------------
def test_data_split_train_test_tuple(fdm):
    fdm.load_data_files_to_anndata()

    fdm.perform_data_split(data_split=(0.6, 0.4))

    assert len(fdm.train_data_) == 3
    assert fdm.val_data_ is None
    assert len(fdm.test_data_) == 2
    assert len(fdm.train_data_) + len(fdm.test_data_) == N_FILES

def test_data_split_train_val_test_tuple(fdm):
    fdm.load_data_files_to_anndata()

    fdm.perform_data_split(data_split=(0.4, 0.2, 0.4))

    assert len(fdm.train_data_) == 2
    assert len(fdm.val_data_) == 1
    assert len(fdm.test_data_) == 2
    assert len(fdm.train_data_) + len(fdm.val_data_) + len(fdm.test_data_) == N_FILES

def test_data_split_train_test_tuple_stratified(fdm):
    fdm.load_data_files_to_anndata()

    fdm.perform_data_split(data_split=(0.6, 0.4), stratify=[1, 1, 1, 0, 0])

    assert len(fdm.train_data_) == 3
    assert fdm.val_data_ is None
    assert len(fdm.test_data_) == 2
    assert len(fdm.train_data_) + len(fdm.test_data_) == N_FILES

def test_data_split_train_val_test_tuple_stratified(fdm):
    fdm.load_data_files_to_anndata()

    # Extend list by additional anndata to allow for stratification to work
    fdm.anndata_list_.append(fdm.anndata_list_[0].copy())

    fdm.perform_data_split(data_split=(1/3, 1/3, 1/3), stratify=[1, 1, 1, 0, 0, 0])

    assert len(fdm.train_data_) == 2
    assert len(fdm.val_data_) == 2
    assert len(fdm.test_data_) == 2
    assert len(fdm.train_data_) + len(fdm.val_data_) + len(fdm.test_data_) == N_FILES + 1

def test_data_split_saving(fdm):
    fdm.load_data_files_to_anndata()

    fdm.perform_data_split(data_split=(2/5, 1/5, 2/5), filename_data_split='data_split.csv')

    split_df = pd.read_csv(Path(fdm.save_path) / 'data_split.csv', index_col=0)

    assert (split_df['mode'] == 'train').sum() == 2
    assert (split_df['mode'] == 'val').sum() == 1
    assert (split_df['mode'] == 'test').sum() == 2

def test_data_split_df(fdm):
    fdm.load_data_files_to_anndata()

    df = pd.DataFrame({
        'filename': [a.uns['filename'] for a in fdm.anndata_list_],
        'mode': ['train', 'val', 'test', 'train', 'test']
    })

    fdm.perform_data_split(data_split=df)

    assert len(fdm.train_data_) == 2
    assert len(fdm.val_data_) == 1
    assert len(fdm.test_data_) == 2


# ------------------------------------------------------------
# 6. Downsampling
# ------------------------------------------------------------
def test_downsampling_frac(fdm):
    fdm.load_data_files_to_anndata()

    orig_sizes = [a.n_obs for a in fdm.anndata_list_]

    fdm.sample_wise_downsampling(
        data_set='all',
        target_num_events=0.5,
        label_key='label',
    )

    new_sizes = [a.n_obs for a in fdm.anndata_list_]

    for old, new in zip(orig_sizes, new_sizes):
        assert new <= old
        assert new > 0
        assert new == round(old * 0.5)


def test_downsampling_target_num(fdm):
    fdm.load_data_files_to_anndata()

    orig_sizes = [a.n_obs for a in fdm.anndata_list_]

    fdm.sample_wise_downsampling(
        data_set='all',
        target_num_events=100,
        label_key='label',
    )

    new_sizes = [a.n_obs for a in fdm.anndata_list_]

    for old, new in zip(orig_sizes, new_sizes):
        assert new <= old
        assert new > 0
    assert all(new == 100 for new in new_sizes)


def test_downsampling_stratified(fdm):
    fdm.load_data_files_to_anndata()

    old_labels = [a[:, 'label'].X.flatten() for a in fdm.anndata_list_]

    fdm.sample_wise_downsampling(
        data_set='all',
        target_num_events=1000,
        stratified=True,
        label_key='label'
    )

    new_labels = [a[:, 'label'].X.flatten() for a in fdm.anndata_list_]

    # Compare class fractions for each sample
    for idx, (old_labels, new_labels) in enumerate(zip(old_labels, new_labels)):
        # Count occurrences
        old_counts = np.bincount(old_labels.astype(int))
        new_counts = np.bincount(new_labels.astype(int), minlength=len(old_counts))

        # Convert to fractions
        old_frac = old_counts / old_counts.sum()
        new_frac = new_counts / new_counts.sum()

        # Assert close
        assert np.allclose(old_frac, new_frac, rtol=1e-03, atol=1e-06)


# ------------------------------------------------------------
# 7. Class balance
# ------------------------------------------------------------
def test_class_balance(fdm):
    fdm.load_data_files_to_anndata()

    df = fdm.check_class_balance('all', 'label')

    assert df.loc['count'].sum() == sum(a.n_obs for a in fdm.anndata_list_)

def test_plot_class_balance_df():
    # Build a dummy class-balance DataFrame
    df = pd.DataFrame(
        data=[[100, 200, 111, 6], [0.1, 0.3, 0.2, 0.4]],
        columns=[1, 2, 3, 5],
        index=['count', 'fraction']
    )

    # Call the plotting function
    ax = FlowDataManager.plot_class_balance_df(class_balance_df=df)

    assert isinstance(ax, matplotlib.axes.Axes)

    # Number of bars = number of classes
    num_classes = df.shape[1]

    # --- Check that bars were plotted ---
    bars = [c for c in ax.get_children() if isinstance(c, plt.Rectangle)]
    # The first rectangle is the background; skip it
    # Bar count should equal number of classes
    assert len(bars) >= num_classes + 1  # +1 for the background rectangle

    # --- Check axis labels and title ---
    assert ax.get_xlabel() == 'Class'
    assert ax.get_ylabel() == 'Frequency'
    assert ax.get_title() == 'Class Balance'

    # --- Check x tick labels ---
    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == ['1', '2', '3', '5']

    # --- Check y-limit logic ---
    expected_ymax = df.loc['fraction'].max() * 1.1
    actual_ymax = ax.get_ylim()[1]
    assert np.isclose(actual_ymax, expected_ymax)

    # --- Check annotation text ---
    texts = [t.get_text() for t in ax.texts]
    expected_texts = [
        f'total: {c}, frac: {round(f, 4)}'
        for c, f in zip(df.loc['count'], df.loc['fraction'])
    ]
    for exp in expected_texts:
        assert exp in texts

    plt.close(ax.figure)

# ------------------------------------------------------------
# 8. Dataloader
# ------------------------------------------------------------
@pytest.mark.parametrize('rtype', ['np_array', 'torch_tensor'])
def test_get_data_loader(fdm, rtype):
    fdm.load_data_files_to_anndata()
    fdm.perform_data_split(data_split=(0.8, 0.2))

    dl = fdm.get_data_loader("train",
                             channels=EXPECTED_CHANNELS[:-1],
                             label_key="label",
                             return_data_loader=rtype)

    x, y = next(iter(dl))
    assert x.shape[1] == N_CHANNELS - 1


def test_get_data_loader_on_disk(fdm):
    fdm.load_data_files_to_anndata()
    fdm.align_channel_names(reference_channel_names=0)

    name = "full_data.npy"

    dl = fdm.get_data_loader(
        "all",
        channels=EXPECTED_CHANNELS[:-1],
        label_key="label",
        on_disk=True,
        filename_np=name,
        return_data_loader="np_array"
    )

    assert (Path(fdm.save_path) / name).exists()


@pytest.mark.parametrize('subset', ['all', 'train', 'test', 'val'])
def test_get_data_loader_subsets(fdm, subset):
    fdm.load_data_files_to_anndata()
    fdm.perform_data_split((0.6, 0.2, 0.2))

    dl = fdm.get_data_loader(subset, channels=[0, 1], return_data_loader='np_array')
    if subset == 'val' and fdm.val_data_ is None:
        assert dl is None
    else:
        assert dl is not None


def test_get_data_loader_subset_val_warns(fdm):
    fdm.load_data_files_to_anndata()
    fdm.perform_data_split((0.6, 0.4))

    fdm.verbosity = 1
    with pytest.warns(UserWarning):
        dl = fdm.get_data_loader('val', channels=[0, 1])

    assert dl is None


def test_invalid_dataset(fdm):
    with pytest.raises(ValueError):
        fdm.get_data_loader('invalid')


@pytest.mark.parametrize('label_key', ['label', 0])
def test_label_key_variants(fdm, label_key):
    fdm.load_data_files_to_anndata()
    fdm.perform_data_split((0.8, 0.2))

    dl = fdm.get_data_loader('train', label_key=label_key)

    x, y = next(iter(dl))
    assert len(y.shape) == 1


def test_obs_label_key(fdm):
    fdm.load_data_files_to_anndata()
    for ad in fdm.anndata_list_:
        ad.obs['obs_label'] = np.zeros(ad.n_obs)

    dl = fdm.get_data_loader('all', label_key='obs_label')
    _, y = next(iter(dl))
    assert (y == 0).all()


def test_layer_key_usage(fdm):
    fdm.load_data_files_to_anndata()

    for ad in fdm.anndata_list_:
        ad.layers['raw'] = ad.X.copy() * 2

    dl = fdm.get_data_loader('all', shuffle=False, layer_key='raw')

    x = next(iter(dl))

    x_concat = np.concatenate([a.X for a in fdm.anndata_list_])

    assert np.allclose(x, 2 * x_concat)


def test_label_layer_key_usage(fdm):
    fdm.load_data_files_to_anndata()
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='raw', cofactor=150)

    dl = fdm.get_data_loader('all', layer_key=None, label_key='label', label_layer_key='raw', shuffle=False)

    x, y = next(iter(dl))

    assert np.allclose(y, np.round(y))  # All labels should be int


def test_batch_size_minus_one(fdm):
    fdm.load_data_files_to_anndata()

    dl = fdm.get_data_loader('all', channels=[0, 1], batch_size=-1)

    x = next(iter(dl))
    assert x.shape[0] == sum(ad.n_obs for ad in fdm.anndata_list_)


def test_torch_tensor_return(fdm):
    fdm.load_data_files_to_anndata()
    fdm.perform_data_split((0.8, 0.2))

    dl = fdm.get_data_loader('train', channels=[0, 1], label_key='label', return_data_loader='torch_tensor')

    x, y = next(iter(dl))

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_on_disk_mmap_mode(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    fdm._save_path = str(tmp_path)

    dl = fdm.get_data_loader('all', channels=list(range(6)), label_key='label', on_disk=True, return_data_loader='torch_tensor')

    arr = dl.dataset.data

    assert isinstance(arr, np.memmap)
    assert arr.flags['WRITEABLE'] is False
    assert arr.flags['OWNDATA'] is False


# ------------------------------------------------------------
# 9. Saving numpy files
# ------------------------------------------------------------
def test_save_combined(fdm):
    fdm.load_data_files_to_anndata()
    fdm.perform_data_split(data_split=(0.8, 0.2))

    fdm.save_to_numpy_files(
        'train', sample_wise=False,
        save_path=fdm.save_path, filename_suffix='_train', channels=[1, 0, 3, 6, 10], label_key='label'
    )

    assert (Path(fdm.save_path) / 'x_train.npy').exists()
    assert (Path(fdm.save_path) / 'y_train.npy').exists()


def test_save_sample_wise(fdm):
    fdm.load_data_files_to_anndata()

    fdm.save_to_numpy_files(
        'all',
        sample_wise=True,
        save_path=fdm.save_path,
        filename_suffix='_all',
        label_key='label'
    )

    mapping = Path(fdm.save_path) / 'sample_names_mapping_all.csv'
    assert mapping.exists()

    for i, a in enumerate(fdm.anndata_list_):
        fn_x = f'x_sample_{str(i).zfill(2)}_all.npy'
        fn_y = f'y_sample_{str(i).zfill(2)}_all.npy'
        sample_x = Path(fdm.save_path) / fn_x
        sample_y = Path(fdm.save_path) / fn_y
        assert sample_x.exists()
        assert sample_y.exists()

# ------------------------------------------------------------
# 10. Relabeling
# ------------------------------------------------------------
def test_relabel_data(fdm):
    fdm.load_data_files_to_anndata()
    fdm.align_channel_names(reference_channel_names=0)

    mapping = {0: 9, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, }

    fdm.relabel_data('all',
                     old_to_new_label_mapping=mapping,
                     label_key='label',
                     new_label_key='relabeled')

    for adata in fdm.anndata_list_:
        assert 'relabeled' in adata.obs
        assert set(adata.obs['relabeled']).issubset(mapping.values())
