"""(2) Phrase ensemble: ask several short questions for the same class and combine the answers.

SAM 3 takes one short noun phrase per query (its paper lists multi-attribute / long phrases as
out of scope), and nothing inside SAM 3 combines answers across prompts. So instead of one long
phrase we ask r separate phrases ("harbor", "dock") and average their presence.

Phrases are ranked per class by val AUROC (correct-class images vs the rest). On AID the
ensemble alone did *not* help (scales differ across classes); combined with balance.py it was
the best training-free result.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def rank_phrases(H: dict, bank: dict, classes: list[str], y: np.ndarray) -> dict[str, list[str]]:
    out = {}
    for k, c in enumerate(classes):
        pos = y == k
        out[c] = [p for _, p in sorted(((roc_auc_score(pos, H[p]), p) for p in bank[c]), reverse=True)]
    return out


def ensemble_scores(H: dict, ranking: dict, classes: list[str], r: int = 2, how: str = "mean") -> np.ndarray:
    cols = []
    for c in classes:
        M = np.stack([H[p] for p in ranking[c][:r]], 1)
        cols.append(M.mean(1) if how == "mean" else M.max(1))
    return np.stack(cols, 1)
