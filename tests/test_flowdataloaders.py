
import pytest
import numpy as np
from torch.utils.data import DataLoader
from flagx.io import FlowDataset, FlowDataLoaders


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def unlabeled_array():
    return np.random.rand(12, 4).astype(np.float32)


@pytest.fixture
def labeled_array():
    X = np.random.rand(10, 3).astype(np.float32)
    y = np.random.randint(0, 5, size=(10, 1)).astype(np.int32)
    return np.hstack([X, y])


@pytest.fixture
def unlabeled_dataset(unlabeled_array):
    return FlowDataset(data=unlabeled_array, includes_labels=False)


@pytest.fixture
def labeled_dataset(labeled_array):
    return FlowDataset(data=labeled_array, includes_labels=True)


# ======================================================================
# 1. np_collate basic behavior
# ======================================================================

def test_np_collate_unlabeled(unlabeled_dataset):
    loader = FlowDataLoaders(unlabeled_dataset, batch_size=4)

    batch = next(iter(loader.pytorch_np_dataloader))
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (4, unlabeled_dataset.data.shape[1])


def test_np_collate_labeled(labeled_dataset):
    loader = FlowDataLoaders(labeled_dataset, batch_size=3)

    batch_data, batch_labels = next(iter(loader.pytorch_np_dataloader))

    assert batch_data.shape == (3, labeled_dataset.data.shape[1] - 1)
    assert batch_labels.shape == (3,)
    assert batch_labels.dtype == int


# ======================================================================
# 2. Consistency checks
# ======================================================================

def test_np_collate_mixed_batch():
    batch = [
        (np.array([1, 2]), 0),
        np.array([3, 4])  # mixes labeled + unlabeled → should error
    ]

    with pytest.raises(ValueError):
        FlowDataLoaders._np_collate(batch)


def test_np_collate_empty_batch():
    with pytest.raises(ValueError):
        FlowDataLoaders._np_collate([])


# ======================================================================
# 3. Ensure labels are converted to int
# ======================================================================

def test_np_collate_label_conversion():
    batch = [
        (np.array([0.1, 0.2]), np.float32(5.0)),
        (np.array([0.3, 0.4]), np.float32(6.0)),
    ]

    data, labels = FlowDataLoaders._np_collate(batch)
    assert labels.tolist() == [5, 6]
    assert labels.dtype == int


# ======================================================================
# 4. DataLoader integration (PyTorch vs NumPy collate)
# ======================================================================

def test_pytorch_dataloader_unlabeled(unlabeled_dataset):
    loader = FlowDataLoaders(unlabeled_dataset, batch_size=5)

    # default PyTorch collation
    batch = next(iter(loader.pytorch_dataloader))
    # batch is a tensor; ensure shape matches
    assert batch.shape[0] == 5


def test_pytorch_dataloader_labeled(labeled_dataset):
    loader = FlowDataLoaders(labeled_dataset, batch_size=6)

    data, labels = next(iter(loader.pytorch_dataloader))
    assert data.shape[0] == 6
    assert labels.shape[0] == 6


# ======================================================================
# 5. Batch size edge cases
# ======================================================================

def test_batch_size_one(labeled_dataset):
    loader = FlowDataLoaders(labeled_dataset, batch_size=1)
    data, labels = next(iter(loader.pytorch_np_dataloader))

    assert data.shape == (1, labeled_dataset.data.shape[1] - 1)
    assert labels.shape == (1,)


def test_batch_size_larger_than_dataset(unlabeled_dataset):
    loader = FlowDataLoaders(unlabeled_dataset, batch_size=100)
    batch = next(iter(loader.pytorch_np_dataloader))

    assert batch.shape == (len(unlabeled_dataset), unlabeled_dataset.data.shape[1])


# ======================================================================
# 6. Verify FlowDataLoaders stores dataset and loaders correctly
# ======================================================================

def test_loader_attributes(unlabeled_dataset):
    loader = FlowDataLoaders(unlabeled_dataset, batch_size=3)

    assert loader.dataset is unlabeled_dataset
    assert isinstance(loader.pytorch_dataloader, DataLoader)
    assert isinstance(loader.pytorch_np_dataloader, DataLoader)
