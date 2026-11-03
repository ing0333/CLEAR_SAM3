"""Evaluation helpers and the diagnosis used in the write-up."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    return 100.0 * float((scores.argmax(1) == y).mean())


def fixed_broken(pred_before: np.ndarray, pred_after: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    fixed = int(((pred_before != y) & (pred_after == y)).sum())
    broken = int(((pred_before == y) & (pred_after != y)).sum())
    return fixed, broken


def error_types(P: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Split errors of argmax-presence by the presence of the *true* class.
    A: true class also clearly present (>=0.5) -> another concept is really there too
    C: 0.1 <= true presence < 0.5          -> ambiguous
    B: true presence < 0.1                 -> SAM 3 does not see the concept at all"""
    err = P.argmax(1) != y
    pt = P[np.arange(len(y)), y]
    return {"A_other_concept_present": err & (pt >= .5),
            "C_ambiguous": err & (pt >= .1) & (pt < .5),
            "B_not_seen": err & (pt < .1)}


def trigger_signals(P: np.ndarray, L: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """'Uncertainty' signals (higher = more uncertain) for a look-again trigger.
    P: presence [N, C]; L: best localization score per class [N, C]."""
    s = np.sort(P, 1)
    return {
        "a_low_confidence": 1 - s[:, -1],
        "b_small_margin": -(s[:, -1] - s[:, -2]),
        "c_disagreement": (L.argmax(1) != P.argmax(1)).astype(float) + 1e-3 * (1 - s[:, -1]),
        "r_random": np.random.default_rng(seed).random(len(P)),
    }


def trigger_report(P: np.ndarray, L: np.ndarray, y: np.ndarray, rates=(.1, .2, .3)) -> dict:
    err = P.argmax(1) != y
    out = {}
    for k, u in trigger_signals(P, L).items():
        order = np.argsort(-u)
        rec = [100 * err[order[:int(len(u) * r)]].sum() / max(1, err.sum()) for r in rates]
        out[k] = {"auroc": float(roc_auc_score(err, u)), "error_recall_at_rates": rec}
    return out
