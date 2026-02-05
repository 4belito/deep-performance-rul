import os
import random

import numpy as np
import torch


def set_global_seed(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
