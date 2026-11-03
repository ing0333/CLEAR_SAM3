"""CLEAR-FS4: all four modules together (OCT + MPV + PPR + PSC), on all 12 settings x 5 splits.

CLEAR-FS (PSC + PPR) learns a classifier over the phrase presences but ignores the zero-shot
CLEAR prediction. CLEAR-FS4 multiplies the two:

    log p(c | x) = log PSC(c | x) + w * log CLEAR(c | x)        then PPR on the result

CLEAR(c | x) = softmax((OCT+MPV scores + PPR offsets) / T), with r / T chosen on val as in
multi_benchmark.evaluate. The weight w is chosen on val with 5-fold out-of-fold PSC
probabilities (C fixed to the one PSC chose on the full val split); test is read once.

python experiments/fs4.py --work_dir $WORK --datasets aid --tag _aid
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import multi_benchmark as MB                      # noqa: E402
import seeds_transclip as ST                      # noqa: E402
from sam3_rs import balance as BAL                # noqa: E402
from sam3_rs import ensemble as ENS               # noqa: E402
from sam3_rs import metrics as M                  # noqa: E402
from sam3_rs.composition import PartComposer      # noqa: E402

WS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def log_softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def evaluate_fs4(bank, classes, S, R, kind):
    col = {p: j for j, p in enumerate(R)}
    H = {k: {p: np.nan_to_num(S[k]["X"][:, j]) for p, j in col.items()} for k in S}
    yv, yt = S["val"]["y"], S["test"]["y"]
    isl = kind == "logit"
    names = [bank[c][0] for c in classes]
    base = {k: np.stack([H[k][n] for n in names], 1) for k in S}

    # CLEAR (same selection rules as multi_benchmark.evaluate)
    T = max((0.25, 0.5, 1.0, 2.0, 4.0), key=lambda t: M.accuracy(BAL.balance(base["val"], t, is_logit=isl)[0], yv))
    bal = lambda P: BAL.balance(P, T, is_logit=isl)[0]
    rank = ENS.rank_phrases(H["val"], bank, classes, yv)
    g2 = [(r, h) for h in ("mean", "max") for r in (1, 2, 3, 5)]
    r, how = max(g2, key=lambda g: M.accuracy(bal(ENS.ensemble_scores(H["val"], rank, classes, *g)), yv))
    Zc = {k: log_softmax(bal(ENS.ensemble_scores(H[k], rank, classes, r, how)) / T) for k in S}

    # PSC (+ PPR) = CLEAR-FS
    psc = PartComposer(R, is_logit=isl).fit(H["val"], yv)
    Pp = psc.predict_proba(H["test"])
    res = {"baseline": M.accuracy(base["test"], yt), "CLEAR": M.accuracy(Zc["test"], yt),
           "PSC": M.accuracy(Pp, yt), "CLEAR-FS": M.accuracy(bal(Pp), yt)}

    # out-of-fold PSC on val -> choose w
    Xv = (psc._X(H["val"]) - psc.mu) / psc.sd
    C = float(np.atleast_1d(psc.clf.C_)[0])
    oof = np.zeros((len(yv), len(classes)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xv, yv):
        m = LogisticRegression(C=C, max_iter=5000).fit(Xv[tr], yv[tr])
        oof[np.ix_(te, m.classes_)] = m.predict_proba(Xv[te])
    lo = np.log(oof + 1e-9)
    w = max(WS, key=lambda w: M.accuracy(BAL.balance(lo + w * Zc["val"], 1.0, is_logit=True)[0], yv))
    fused = np.log(Pp + 1e-9) + w * Zc["test"]
    res["PSC+CLEAR (no PPR)"] = M.accuracy(fused, yt)
    res["CLEAR-FS4"] = M.accuracy(BAL.balance(fused, 1.0, is_logit=True)[0], yt)
    res["FS4_w"] = w
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--data_root", default="/content/bench_data")
    ap.add_argument("--datasets", default="aid,ucm,resisc45,sun_aid")
    ap.add_argument("--models", default="sam3,openai,remoteclip")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.join(a.work_dir, "bench", f"fs4{a.tag}.json")
    for ds in a.datasets.split(","):
        for model in a.models.split(","):
            bank, classes, R, X, F, y = ST.load_pool(ds, model, a.work_dir, a.data_root)
            for seed in map(int, a.seeds.split(",")):
                key = f"{ds}/{model}/seed{seed}"
                done = json.load(open(out)) if os.path.exists(out) else {}
                if key in done:
                    continue
                X2, _, y2 = ST.resplit(X, F, y, seed)
                S = {k: {"X": X2[k], "y": y2[k]} for k in ("val", "test")}
                res = evaluate_fs4(bank, classes, S, R, MB.KIND[model])
                done = json.load(open(out)) if os.path.exists(out) else {}
                done[key] = res
                json.dump(done, open(out, "w"), indent=1)
                print(f"  [{key}] CLEAR {res['CLEAR']:.1f} | PSC {res['PSC']:.1f} | CLEAR-FS {res['CLEAR-FS']:.1f}"
                      f" | CLEAR-FS4 {res['CLEAR-FS4']:.1f} (w={res['FS4_w']})", flush=True)


if __name__ == "__main__":
    main()
