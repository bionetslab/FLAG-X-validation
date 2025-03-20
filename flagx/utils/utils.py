

import random
import torch
import numpy as np

def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)  # Sets the seed for generating random numbers on all devices.
    torch.cuda.manual_seed_all(seed)  # Set the seed for generating random numbers on all GPUs.
    # torch.cuda.manual_seed(seed)  # Set the seed for generating random numbers for the current GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)









