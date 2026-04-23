"""Deterministic seed helpers."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Set the process seed for deterministic placeholders."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
