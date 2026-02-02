import random

import numpy as np
import torch


def set_global_deterministic_seed(seed: int) -> None:
    """
    Set a global random seed and enforce deterministic behavior across
    Python, NumPy, and PyTorch
    """
    # Python and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
