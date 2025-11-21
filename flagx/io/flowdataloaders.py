
import numpy as np

from torch.utils.data import Dataset, DataLoader
from typing import Union, Sequence, Tuple


class FlowDataLoaders:
    """
    Wrapper providing two PyTorch dataloaders for a given `Dataset`:

    - `pytorch_dataloader`: standard PyTorch dataloader using default collation.
    - `pytorch_np_dataloader`: PyTorch dataloader that returns NumPy arrays instead of tensors via a custom `np_collate` function.

    This class exists to conveniently switch between tensor-based and NumPy-based dataloading pipelines without modifying the underlying dataset or training code.

    Attributes:
        dataset (Dataset): The dataset from which batches are generated.
        pytorch_dataloader (DataLoader): Standard PyTorch dataloader using default tensor collation.
        pytorch_np_dataloader (DataLoader): Dataloader returning NumPy arrays through a custom `np_collate` function.
    """
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
    ):
        """
        Initialize dataloaders for the provided dataset.

        Args:
            dataset (Dataset): Any PyTorch-compatible dataset instance producing samples or (sample, label) pairs.
            **kwargs: Additional arguments forwarded directly to `torch.utils.data.DataLoader`, such as `batch_size`, `shuffle`, or `num_workers`.
        """
        self.dataset = dataset
        self.pytorch_dataloader = DataLoader(dataset, **kwargs)
        self.pytorch_np_dataloader = DataLoader(dataset, collate_fn=FlowDataLoaders._np_collate, **kwargs)
        # Might be adding other custom Dataloaders later on ...

    @staticmethod
    def _np_collate(
            batch: Union[Sequence[Tuple[np.ndarray, int]],
            Sequence[np.ndarray]]
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        # Handle empty batch
        if not batch:
            raise ValueError('Batch is empty. Cannot collate an empty batch.')

        # Check if the batch contains labels, use 1st element as reference
        is_labeled = isinstance(batch[0], tuple) and len(batch[0]) == 2

        # Ensure consistency (all elements should be either labeled or unlabeled)
        if not all(isinstance(item, tuple) == is_labeled for item in batch):
            raise ValueError('Inconsistent batch format: mixed labeled and unlabeled samples.')

        # Check if the batch contains labels by looking at the first item
        if is_labeled:
            # Separate data and labels
            batch_data = np.stack([item[0] for item in batch], axis=0)
            batch_labels = np.stack([item[1] for item in batch], axis=0).astype(int)
            return batch_data, batch_labels
        else:
            # Only data without labels
            batch_data = np.stack(batch, axis=0)
            return batch_data