"""
RibonanzaNet feature extractor: single and pair representations from sequence.
Requires RibonanzaNet (e.g. from Kaggle / Shujun-He/RibonanzaNet).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def get_ribonanzanet_features(
    sequence: str,
    model_path: Optional[str | Path] = None,
    device: str = "cuda",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (single_repr, pair_repr) or (None, None) if RibonanzaNet not available.
    single_repr: (1, L, 256), pair_repr: (1, L, L, 64) or similar.
    """
    try:
        # Placeholder: actual implementation would load RibonanzaNet and run forward
        # from RibonanzaNet.Network import RibonanzaNet
        raise ImportError("RibonanzaNet not installed")
    except ImportError:
        return None, None
