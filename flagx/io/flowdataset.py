
import numpy as np

from typing import Union, Tuple
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    """
    A lightweight dataset wrapper for flow cytometry matrices stored in memory or on disk.

    This dataset supports two usage modes:
    - In-memory: a NumPy array is passed directly.
    - On-disk: a `.npy` file path is passed and optionally memory-mapped for efficient loading of large datasets.

    If `includes_labels=True`, the dataset assumes that the last column of the
    matrix contains integer class labels.

    Attributes:
        data (np.ndarray or np.memmap): The underlying data matrix.
        on_disk (bool): Whether the dataset is backed by a memory-mapped `.npy` file or in memory.
        includes_labels (bool): Whether the last column contains labels.
        label_idx (int): Column index of the last columns where the labels are stored (only when `includes_labels=True`).
        file_path (str): Path to `.npy` file when loading from disk.
    """

    def __init__(
            self,
            data: Union[str, np.ndarray],
            on_disk: bool = True,
            includes_labels: bool = False,
    ):
        """
            Initializes the FlowDataset.

            Args:
                data (str or np.ndarray): Either a path to a `.npy` file or a NumPy array containing the data. If a path is given, the file is loaded in the mode determined by `on_disk`.
                on_disk (bool): If True, the `.npy` file is memory-mapped instead of fully loaded into RAM. Has no effect when `data` is already an array.
                includes_labels (bool): If True, the last column of the data matrix is interpreted as label values.

            Raises:
                ValueError: If `data` is neither a file path nor a NumPy array.
                ValueError: If `includes_labels=True` but the provided matrix does not have at least two columns.
            """

        if not (isinstance(data, str) or isinstance(data, np.ndarray)):
            raise ValueError("'data' must be path to data file (.npy) or Numpy array")

        self.on_disk = on_disk

        if isinstance(data, str):
            self.file_path = data
            # ### Load data in previously defined mode

            if not self.on_disk:
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
        """
        Return the number of events (rows) in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, int], np.ndarray]:
        """
        Retrieve a single event or (event, label) pair.

        Args:
            idx (int): Row index of the event to fetch.

        Returns:
            np.ndarray:
            Event feature vector when `includes_labels=False`.
            tuple (np.ndarray, int):
                When `includes_labels=True`, returns a tuple containing:
                - event (np.ndarray): the feature vector
                - label (int): the integer class label
        """

        if self.includes_labels:
            event = self.data[idx, :self.label_idx]
            label = int(self.data[idx, self.label_idx].item())

            return event, label
        else:
            event = self.data[idx, :]
            return event