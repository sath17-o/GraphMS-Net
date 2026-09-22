"""Frozen cross-fitted Stage11-v1 post-processing."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

RECIPES = {
    0: {"threshold": 0.40, "minvox": 10},
    1: {"threshold": 0.40, "minvox": 10},
    2: {"threshold": 0.45, "minvox": 10},
    3: {"threshold": 0.40, "minvox": 10},
    4: {"threshold": 0.40, "minvox": 10},
}

def apply_stage11(probability: np.ndarray, outer_fold: int) -> np.ndarray:
    """Threshold + 26-connected CCA + removal of components smaller than 10 voxels."""
    if outer_fold not in RECIPES:
        raise ValueError(f"Unknown outer fold: {outer_fold}")
    r = RECIPES[outer_fold]
    mask = np.asarray(probability >= r["threshold"], dtype=bool)
    labels, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n:
        counts = np.bincount(labels.ravel())
        keep = counts >= int(r["minvox"])
        keep[0] = False
        mask = keep[labels]
    return mask.astype(np.uint8)
