
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FCNNModel(nn.Module):
    # Dimensions chosen as described in paper and
    # https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb
    def __init__(self, in_size, out_size, layer_sizes: Tuple[int, int, int] = (128, 64, 32)):
        super(FCNNModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], out_size, bias=True)
        # self.softmax = nn.Softmax(dim=1)  # Softmax for the output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.softmax(x)  # Softmax activation for output
        # Do not apply softmax, CrossEntropyLoss does so internally
        return x
