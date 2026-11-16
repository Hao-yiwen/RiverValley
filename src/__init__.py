# -*- coding: utf-8 -*-
"""
RiverValley - A GPT-based Language Model for Pretraining and SFT

Supports:
- Pretraining with text data
- Supervised Fine-Tuning (SFT) with conversation data
"""

__version__ = "0.1.0"
__author__ = "Author"

from config import RiverValleyConfig
from model import RiverValleyModel

__all__ = [
    "RiverValleyConfig",
    "RiverValleyModel",
    "__version__",
]
