"""(3) Label-free class-skew correction.

Using presence ("is X anywhere in the image?") as a scene classifier makes predictions pile
up on concepts that appear in many scenes (e.g. parking lots). On AID test, 'Parking' was
predicted 408 times for 80 true images and the top-5 classes took 51% of all predictions.

This module adds one offset b_c per class so that the *soft* prediction distribution over
the (unlabelled) test batch becomes uniform. The only assumption is that classes have roughly
equal frequency in the batch. No labels are used; the softness T is chosen on val.
"""
from __future__ import annotations

import numpy as np


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def balance(P: np.ndarray, T: float = 2.0, iters: int = 500, lr: float = 0.5, prior=None,
            is_logit: bool = False):
    """P: [N, C] scores in (0, 1) (SAM 3 presence), or logits if is_logit=True (e.g. CLIP
    similarities x logit scale). Returns (corrected logits [N, C], offsets b [C])."""
    L = np.asarray(P, float) if is_logit else logit(P)
    N, C = L.shape
    target = np.full(C, 1.0 / C) if prior is None else np.asarray(prior, float) / np.sum(prior)
    b = np.zeros(C)
    for _ in range(iters):
        Z = (L + b) / T
        Z -= Z.max(1, keepdims=True)
        S = np.exp(Z)
        S /= S.sum(1, keepdims=True)
        b -= lr * (np.log(S.mean(0) + 1e-9) - np.log(target))
    return L + b, b


def top_k_share(pred: np.ndarray, n_classes: int, k: int = 5) -> float:
    """Share (%) of predictions taken by the k most-predicted classes (uniform: 100*k/C)."""
    c = np.bincount(pred, minlength=n_classes)
    return 100.0 * np.sort(c)[-k:].sum() / len(pred)
