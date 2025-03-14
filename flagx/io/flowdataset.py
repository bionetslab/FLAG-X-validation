
import numpy as np

from typing import Union, Tuple
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    def __init__(
            self,
            data: Union[str, np.ndarray],
            on_disk: bool = True,
            includes_labels: bool = False,
    ):

        if not (isinstance(data, str) or isinstance(data, np.ndarray)):
            raise TypeError("'data' must be path to data file (.npy) or Numpy array")

        self.on_disk = on_disk

        if isinstance(data, str):
            self.file_path = data
            # ### Load data in previously defined mode

            if self.on_disk:
                self.data = np.load(self.file_path)
            else:
                self.data = np.load(self.file_path, mmap_mode="r")

        else:
            self.data = data

        # Slicing on memory-mapped arrays only works for contiguous slices, expect labels in last column
        self.includes_labels = includes_labels
        if self.includes_labels:
            if not (self.data.ndim == 2 and self.data.shape[1] > 1):
                raise ValueError("Data must have at least two dimensions with labels in the last column")

            self.label_idx = self.data.shape[1] - 1

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, int], np.ndarray]:

        if self.includes_labels:
            event = self.data[idx, :self.label_idx]
            label = int(self.data[idx, self.label_idx].item())

            return event, label
        else:
            event = self.data[idx, :]
            return event