"""Deterministic seeding helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic CPU ops where possible
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
