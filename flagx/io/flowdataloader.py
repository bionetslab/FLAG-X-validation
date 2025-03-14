
import numpy as np

from torch.utils.data import Dataset, DataLoader
from typing import Union, Sequence, Tuple


class FlowDataLoader:
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
    ):
        self.dataset = dataset
        self.pytorch_dataloader = DataLoader(dataset, **kwargs)
        # Might be adding other custom Dataloaders later on
        self.pytorch_np_dataloader = DataLoader(dataset, collate_fn=FlowDataLoader.np_collate, **kwargs)

    @staticmethod
    def np_collate(batch: Union[Sequence[Tuple[np.ndarray, int]], Sequence[np.ndarray]]):
        # Check if the batch contains labels by looking at the first item
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # Separate data and labels
            batch_data = np.stack([item[0] for item in batch], axis=0)
            batch_labels = np.stack([item[1] for item in batch], axis=0)
            return batch_data, batch_labels
        else:
            # Only data without labels
            batch_data = np.stack(batch, axis=0)
            return batch_data