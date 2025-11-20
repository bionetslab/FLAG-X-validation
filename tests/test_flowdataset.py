
import numpy as np
import pytest
from flagx.io.flowdataset import FlowDataset


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def tmp_npy(tmp_path):
    """Create a temp .npy file with data and labels."""
    arr = np.hstack([
        np.random.rand(10, 3).astype(np.float32),
        np.random.randint(0, 3, size=(10, 1)).astype(np.int32)
    ])
    file = tmp_path / 'data.npy'
    np.save(file, arr)
    return file, arr


@pytest.fixture
def small_array():
    """Array without labels."""
    return np.random.rand(5, 4).astype(np.float32)


@pytest.fixture
def labeled_array():
    """Array with labels in last column."""
    X = np.random.rand(6, 3).astype(np.float32)
    y = np.random.randint(0, 2, size=(6, 1)).astype(np.int32)
    return np.hstack([X, y])


# ----------------------------------------------------------------------
# Initialization Tests
# ----------------------------------------------------------------------

def test_init_with_ndarray(small_array):
    ds = FlowDataset(data=small_array, on_disk=False, includes_labels=False)
    assert isinstance(ds.data, np.ndarray)
    assert len(ds) == small_array.shape[0]


def test_init_with_file_on_disk(tmp_npy):
    file, arr = tmp_npy
    ds = FlowDataset(data=str(file), on_disk=True, includes_labels=True)
    assert isinstance(ds.data, np.memmap)
    assert ds.data.flags['WRITEABLE'] is False
    assert len(ds) == arr.shape[0]


def test_init_invalid_data_type():
    with pytest.raises(ValueError):
        FlowDataset(data=123)


# ----------------------------------------------------------------------
# Label Handling Tests
# ----------------------------------------------------------------------

def test_init_labels_check_shape():
    bad = np.random.rand(10).astype(np.float32)  # 1D → invalid
    with pytest.raises(ValueError):
        FlowDataset(data=bad, includes_labels=True)


def test_init_labels_check_column_count():
    bad = np.random.rand(5, 1).astype(np.float32)  # 1 column → no room for labels
    with pytest.raises(ValueError):
        FlowDataset(data=bad, includes_labels=True)


# ----------------------------------------------------------------------
# __getitem__ Tests
# ----------------------------------------------------------------------

def test_getitem_no_labels(small_array):
    ds = FlowDataset(data=small_array, includes_labels=False)
    x = ds[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (small_array.shape[1],)


def test_getitem_with_labels(labeled_array):
    ds = FlowDataset(data=labeled_array, includes_labels=True)
    x, y = ds[0]
    assert isinstance(x, np.ndarray)
    assert isinstance(y, int)
    assert x.shape == (labeled_array.shape[1] - 1,)
    assert y == labeled_array[0, -1]


def test_getitem_label_memmap(tmp_npy):
    file, arr = tmp_npy
    ds = FlowDataset(str(file), on_disk=True, includes_labels=True)
    x, y = ds[3]
    assert isinstance(x, np.ndarray)
    assert isinstance(y, int)
    assert y == arr[3, -1]


# ----------------------------------------------------------------------
# Length Tests
# ----------------------------------------------------------------------

def test_len(small_array):
    ds = FlowDataset(data=small_array)
    assert len(ds) == small_array.shape[0]


# ----------------------------------------------------------------------
# File loading behavior tests
# ----------------------------------------------------------------------

def test_memmap_vs_loaded(tmp_npy):
    file, arr = tmp_npy

    ds_memmap = FlowDataset(str(file), on_disk=True)
    ds_loaded = FlowDataset(str(file), on_disk=False)

    assert isinstance(ds_memmap.data, np.memmap)
    assert isinstance(ds_loaded.data, np.ndarray)

    assert np.allclose(ds_memmap[0], ds_loaded[0])


# ----------------------------------------------------------------------
# Boundary and edge cases
# ----------------------------------------------------------------------

def test_index_out_of_bounds(small_array):
    ds = FlowDataset(data=small_array)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_negative_index(small_array):
    ds = FlowDataset(data=small_array)
    x = ds[-1]  # Should wrap around like numpy
    assert np.allclose(x, small_array[-1])


