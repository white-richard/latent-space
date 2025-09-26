import os
import random
import numpy as np
import torch


def set_global_deterministic_seed(seed: int) -> None:
    """
    Set a global random seed and enforce deterministic behavior across
    Python, NumPy, and PyTorch (CPU and GPU).
    """
    # Python and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure cuBLAS deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
