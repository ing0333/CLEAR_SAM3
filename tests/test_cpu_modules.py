"""CPU-only sanity tests on synthetic scores (no SAM 3 / GPU needed). Run: pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam3_rs import balance as BAL
from sam3_rs import concept_translation as CT
from sam3_rs import ensemble as ENS
from sam3_rs import metrics as M
from sam3_rs.composition import PartComposer

rng = np.random.default_rng(0)
C, N = 4, 400
classes = [f"c{i}" for i in range(C)]
y = np.repeat(np.arange(C), N // C)


def synth(signal, n=N):
    """presence-like score: high on class `signal`, low elsewhere."""
    base = rng.uniform(0.0, 0.2, n)
    if signal is not None:
        base[y == signal] += 0.6
    return np.clip(base, 0, 1)


def test_balance_removes_skew():
    P = np.stack([synth(k) for k in range(C)], 1)
    P[:, 0] += 0.75                    # class 0 swallows most images
    P = np.clip(P, 0, 0.999)
    before = BAL.top_k_share(P.argmax(1), C, k=1)
    L, b = BAL.balance(P, T=1.0)
    after = BAL.top_k_share(L.argmax(1), C, k=1)
    assert before > 60 and after < before
    assert b[0] < 0                    # the over-predicted class is pushed down
    assert M.accuracy(L, y) > M.accuracy(P, y)


def test_selection_prefers_informative_phrase():
    bank = {c: [f"{c}_name", f"{c}_good"] for c in classes}
    H = {}
    for k, c in enumerate(classes):
        H[f"{c}_name"] = synth(None)   # the original name carries no signal
        H[f"{c}_good"] = synth(k)
    s1 = CT.select_supervised(H, bank, classes, y)
    s2 = CT.select_label_free(H, bank, classes)
    assert all(v == 1 for v in s1.values())
    assert all(v == 1 for v in s2.values())
    assert M.accuracy(CT.score_matrix(H, bank, classes, s1), y) > 90


def test_ensemble_and_composition_run():
    bank = {c: [f"{c}_a", f"{c}_b", f"{c}_c"] for c in classes}
    H = {p: synth(int(p[1])) for c in classes for p in bank[c]}
    rank = ENS.rank_phrases(H, bank, classes, y)
    P = ENS.ensemble_scores(H, rank, classes, r=2)
    assert P.shape == (N, C) and M.accuracy(P, y) > 90
    psc = PartComposer(CT.unique_phrases(bank), cv=3).fit(H, y)
    assert M.accuracy(psc.predict_proba(H), y) > 90
    pos, neg = psc.top_parts(0)
    assert pos[0][0].startswith("c0_")


def test_error_types_partition_errors():
    P = rng.uniform(0, 1, (N, C))
    et = M.error_types(P, y)
    total = sum(v.sum() for v in et.values())
    assert total == int((P.argmax(1) != y).sum())
