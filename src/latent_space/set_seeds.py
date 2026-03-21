import os
import random

import numpy as np
import torch


def set_seeds(seed: int, *, cu_benchmark: bool = True) -> np.random.Generator:
    """Set all random seeds for reproducibility. Returns a numpy random generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    if cu_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return rng
