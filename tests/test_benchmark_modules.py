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


def test_segmentation_selection_picks_informative_phrase():
    from sam3_rs import segmentation as SG
    r = np.random.default_rng(0)
    N, H, C = 6, 32, 3
    gt = r.integers(0, C, (N, H, H)).astype(np.uint8)
    gt[:, :2] = 255                                    # ignored rows
    maps = np.zeros((N, 2 * C, H, H), np.uint8)        # phrase 2c = useless name, 2c+1 = good phrase
    for c in range(C):
        maps[:, 2 * c] = r.integers(0, 120, (N, H, H))
        maps[:, 2 * c + 1] = np.where(gt == c, 200, 30)
    cand = [[2 * c, 2 * c + 1] for c in range(C)]
    names = SG.miou(SG.confusion(maps, gt, [[2 * c] for c in range(C)], C))
    chosen, _ = SG.select_phrases(maps, gt, cand, [2 * c for c in range(C)], C)
    assert chosen == [2 * c + 1 for c in range(C)]
    assert SG.miou(SG.confusion(maps, gt, [[j] for j in chosen], C)) > names + 30
    bg = SG.confusion(maps, gt, [[j] for j in chosen], C, bg_thd=0.99, bg_idx=0)
    assert bg[:, 0].sum() == (gt != 255).sum()         # everything below threshold -> background


def test_rstc_wrapper_matches_official_zero_shot_path():
    """Needs torch and a clone of RS-TransCLIP (RSTC_DIR, default ../ext/RS-TransCLIP)."""
    import contextlib
    import io
    import pytest
    torch = pytest.importorskip("torch")
    rdir = os.environ.get("RSTC_DIR", os.path.join(HERE, "..", "..", "ext", "RS-TransCLIP"))
    if not os.path.isdir(rdir):
        pytest.skip("RS-TransCLIP not cloned")
    if not torch.cuda.is_available():   # the solver calls .cuda(); make it a no-op on CPU
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self
    import seeds_transclip as ST
    run = ST.make_rstc(rdir)
    import TransCLIP_solver.TransCLIP as TC
    r = np.random.default_rng(0)
    K, n, d = 8, 400, 48
    yy = r.integers(0, K, n)
    cent = r.normal(size=(K, d))
    Fx = cent[yy] * 0.6 + r.normal(size=(n, d))
    Fx /= np.linalg.norm(Fx, axis=1, keepdims=True)
    Tx = cent / np.linalg.norm(cent, axis=1, keepdims=True) + r.normal(size=(K, d)) * 0.3
    Tx /= np.linalg.norm(Tx, axis=1, keepdims=True)
    Ft, Tt = torch.tensor(Fx).float(), torch.tensor(Tx.T).float()
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, official = TC.TransCLIP_solver(None, None, None, None, Ft, torch.tensor(yy), Tt, verbose=True)
    ours = run(Fx, yy, torch.softmax(100 * Ft @ Tt, 1).numpy())
    assert abs(ours - official) < 1e-6
