
import pytest
import flowio
import numpy as np
import pandas as pd
from flagx.io import FlowDataManager
from flagx.io.export import export_to_fcs, _init_fcs_dfs, _scale_columns


# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------
def test_init_fcs_dfs(fdm):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_
    dfs = _init_fcs_dfs(adatas, layer_key=None)

    assert len(dfs) == len(adatas)
    for df, ad in zip(dfs, adatas):
        assert df.shape == ad.shape
        assert list(df.columns) == list(ad.var_names)


def test_scale_columns_global(fdm):
    """Test global and sample-wise scaling using real values."""
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_
    dfs = _init_fcs_dfs(adatas, layer_key=None)

    # pick a real column from first dataframe
    col_arrays = [df.iloc[:, 0].to_numpy() for df in dfs]

    # global scale
    scaled = _scale_columns(col_arrays, val_range=(0, 100), sample_wise=False)
    assert len(scaled) == len(col_arrays)
    assert all(s.min() >= 5 for s in scaled)      # 5% margin
    assert all(s.max() <= 95 for s in scaled)


def test_scale_columns_sample_wise(fdm):
    """Test global and sample-wise scaling using real values."""
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_
    dfs = _init_fcs_dfs(adatas, layer_key=None)

    # pick a real column from first dataframe
    col_arrays = [df.iloc[:, 0].to_numpy() for df in dfs]

    # sample-wise
    scaled = _scale_columns(col_arrays, val_range=(0, 100), sample_wise=True)
    assert len(scaled) == len(col_arrays)
    assert all(s.min() >= 5 for s in scaled)  # 5% margin
    assert all(s.max() <= 95 for s in scaled)

    assert all(np.isclose(s.min(), 5) for s in scaled)
    assert all(np.isclose(s.max(), 95) for s in scaled)


# ------------------------------------------------------------
# 2. Main export_to_fcs tests (end-to-end)
# ------------------------------------------------------------

def test_export_to_fcs_single_concat_file(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    outfile = 'all_samples.fcs'

    export_to_fcs(
        data_list=adatas,
        save_path=str(tmp_path),
        save_filenames=outfile,
        sample_wise=False,
    )

    f = tmp_path / outfile
    assert f.exists()
    assert f.stat().st_size > 0


def test_export_to_fcs_sample_wise(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    save_filenames = [f'sample_{i}.fcs' for i in range(len(adatas))]

    export_to_fcs(
        data_list=adatas,
        save_path=str(tmp_path),
        save_filenames=save_filenames,
        sample_wise=True,
    )

    for fn in save_filenames:
        fpath = tmp_path / fn
        assert fpath.exists()
        assert fpath.stat().st_size > 0


def test_export_to_fcs_creates_sample_id_csv(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    outfile = 'all_samples.fcs'
    export_to_fcs(
        data_list=adatas,
        save_path=str(tmp_path),
        save_filenames=outfile,
        sample_wise=False,
    )

    mapping_file = tmp_path / 'filenames_and_sample_id.csv'
    assert mapping_file.exists()

    df = pd.read_csv(mapping_file)
    assert 'filenames' in df.columns
    assert 'sample_id' in df.columns
    assert df.shape[0] == len(adatas)


def test_export_to_fcs_add_columns_and_scale(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    # add 1 new column per sample
    extra_cols = [np.arange(ad.shape[0]) for ad in adatas]

    export_to_fcs(
        data_list=adatas,
        add_columns=[extra_cols, ],
        add_columns_names=['extra', ],
        scale_columns=['extra', ],
        val_range=(0, 100),
        keep_unscaled=True,
        save_path=str(tmp_path),
        save_filenames='out.fcs',
        sample_wise=False,
    )

    f = tmp_path / 'out.fcs'
    assert f.exists()
    assert f.stat().st_size > 0

    fdm2 = FlowDataManager(data_file_names=['out.fcs'], data_file_path=str(tmp_path))
    fdm2.load_data_files_to_anndata()
    adata = fdm2.anndata_list_[0]

    assert 'extra' in adata.var_names
    assert adata[:, 'extra'].X.min() >= 0
    assert adata[:, 'extra'].X.max() <= 100

    assert 'extra_unscaled' in adata.var_names
    expected = np.concatenate([np.arange(ad.shape[0]) for ad in adatas])
    assert np.allclose(adata[:, 'extra_unscaled'].X.flatten(), expected)


def test_export_to_fcs_metadata_ranges(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    outfile = 'meta_check.fcs'

    export_to_fcs(
        data_list=adatas,
        save_path=str(tmp_path),
        save_filenames=outfile,
        val_range=(0, 2**20),
    )

    f = tmp_path / outfile
    assert f.exists()
    assert f.stat().st_size > 0

    # Load with flowio
    fd = flowio.FlowData(str(f))

    # metadata dict is in fd.text
    metadata = fd.text

    # number of channels (PnN keys give their names)
    n_channels = len([k for k in metadata.keys() if k.endswith('N') and k.startswith('P')])

    expected_range = str(2**20)

    # For each channel i, check that PiR exists and is correct
    for i in range(1, n_channels + 1):
        key = f'P{i}R'
        assert key in metadata
        assert metadata[key] == expected_range


def test_export_to_fcs_add_columns_length_error(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    with pytest.raises(ValueError):
        export_to_fcs(
            data_list=adatas,
            add_columns=[[np.zeros(ad.shape[0]) for ad in adatas],   # one list
                         [np.ones(ad.shape[0]) for ad in adatas]],   # second list → mismatch
            add_columns_names=['only_one_name'],
            save_path=str(tmp_path),
            save_filenames='bad.fcs',
        )


def test_export_to_fcs_add_columns_names_missing(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    adatas = fdm.anndata_list_

    with pytest.raises(ValueError):
        export_to_fcs(
            data_list=adatas,
            add_columns=[[np.zeros(ad.shape[0]) for ad in adatas]],
            add_columns_names=None,
            save_path=str(tmp_path),
            save_filenames='ignored.fcs',
        )

def test_export_to_fcs_specify_layer(fdm, tmp_path):
    fdm.load_data_files_to_anndata()
    def custom_prepr(ad, factor=2):
        ad.X = ad.X * factor
    fdm.sample_wise_preprocessing(flavour='custom', save_raw_to_layer='raw', preprocessing_method=custom_prepr, factor=2)
    adatas = fdm.anndata_list_

    export_to_fcs(
        data_list=adatas,
        layer_key='raw',
        save_path=str(tmp_path),
        save_filenames=None,
        sample_wise=True,
    )

    fdm2 = FlowDataManager(data_file_names=[f'sample_{i}.fcs' for i in range(len(adatas))], data_file_path=str(tmp_path))
    fdm2.load_data_files_to_anndata()
    adatas2 = fdm2.anndata_list_

    for adata, adata2 in zip(adatas, adatas2):
        x = adata.layers['raw']
        x2 = adata2.X
        assert np.allclose(x, x2)
