"""(1) Concept translation: class name -> noun phrases for what is *visible from above*.

SAM 3 grounds short noun phrases of visible things ("office building", "harbor") far better
than names of functions or uses ("commercial", "port"). Each class gets a small bank of
candidate phrases (original name first, see prompts/phrases.json); this module picks one
phrase per class.

  S1 (supervised selection): coordinate ascent on val accuracy (20 labelled images/class).
  S2 (label-free selection): prefer phrases that fire strongly on a few images and weakly on
      the rest ("contrast"), since a scene class covers roughly 1/C of a balanced batch.
"""
from __future__ import annotations

import json

import numpy as np


def load_bank(path: str) -> tuple[dict[str, list[str]], str]:
    d = json.load(open(path))
    return d["phrases"], d.get("instruction", "")


def unique_phrases(bank: dict[str, list[str]]) -> list[str]:
    return sorted({p for v in bank.values() for p in v})


def score_matrix(H: dict[str, np.ndarray], bank: dict, classes: list[str], sel: dict[str, int]) -> np.ndarray:
    """H: phrase -> presence vector [N]. Returns [N, C] using phrase index sel[c] for class c."""
    return np.stack([H[bank[c][sel[c]]] for c in classes], 1)


def select_supervised(H: dict, bank: dict, classes: list[str], y: np.ndarray, passes: int = 2) -> dict[str, int]:
    """S1: for each class in turn, keep the phrase that maximises val accuracy given the others."""
    sel = {c: 0 for c in classes}
    for _ in range(passes):
        for c in classes:
            accs = []
            for j in range(len(bank[c])):
                s = dict(sel); s[c] = j
                accs.append((score_matrix(H, bank, classes, s).argmax(1) == y).mean())
            sel[c] = int(np.argmax(accs))
    return sel


def contrast(p: np.ndarray, frac: float) -> float:
    """Mean presence of the top `frac` images minus mean presence of the rest."""
    q = np.sort(p)[::-1]
    n = max(1, int(len(p) * frac))
    return float(q[:n].mean() - q[n:].mean())


def select_label_free(H: dict, bank: dict, classes: list[str]) -> dict[str, int]:
    """S2: pick, per class, the phrase with the highest contrast (no labels)."""
    frac = 1.0 / len(classes)
    return {c: int(np.argmax([contrast(H[p], frac) for p in bank[c]])) for c in classes}
