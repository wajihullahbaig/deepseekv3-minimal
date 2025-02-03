import torch
import random
from typing import  Optional
import numpy as np
import os

def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)