"""CPU tests for the Part 2 code: transductive baselines, logit-mode modules, lazy loading."""
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
from sam3_rs import balance as BAL                 # noqa: E402
from sam3_rs import transductive as TR             # noqa: E402
from sam3_rs.benchmarks import SPECS, load_phrase_bank  # noqa: E402

rng = np.random.default_rng(1)
C, N, D = 5, 300, 32
y = rng.integers(0, C, N)
F = rng.normal(size=(C, D))[y] * 3 + rng.normal(size=(N, D))
L = np.eye(C)[y] + rng.normal(size=(N, C)) * 1.2
L[:, 0] += 1.5                                     # text prior biased toward class 0
acc = lambda Z: float((Z.argmax(1) == y).mean())


def test_transclip_lite_undoes_text_bias():
    assert acc(TR.transclip_lite(L, F, T=1.0, lam=0.3)) > acc(L) + 0.3


def test_label_propagation_is_smoothing_only():
    Z = TR.label_propagation(L, F, alpha=0.7, k=10, T=1.0)
    assert Z.shape == L.shape and np.isfinite(Z).all()
    assert abs(acc(Z) - acc(L)) < 0.15             # smoothing alone cannot remove the bias


def test_balance_logit_mode_matches_prob_mode():
    P = 1 / (1 + np.exp(-L))
    a, _ = BAL.balance(P, T=1.0)
    b, _ = BAL.balance(BAL.logit(P), T=1.0, is_logit=True)
    assert (a.argmax(1) == b.argmax(1)).mean() > 0.99


def test_phrase_banks_cover_their_datasets():
    for name in ("ucm", "resisc45"):
        bank = load_phrase_bank(name)
        assert all(len(v) >= 2 for v in bank.values())
    assert SPECS["sun_aid"]["view"] == "ground"


def test_lazy_loader_defers_model():
    import multi_benchmark as mb
    made = []
    z = mb._Lazy(lambda: made.append(1) or type("M", (), {"f": lambda self: 7})())
    assert made == []
    assert z.f() == 7 and made == [1]
