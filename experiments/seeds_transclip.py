"""Repeat the Part 2 evaluation over 5 val/test splits and add the official RS-TransCLIP solver.

Scores are read from the caches written by multi_benchmark.py, so this is CPU work except for
the RS-TransCLIP solver (it calls .cuda()). Seed 0 is the original split; seeds 1-4 re-split the
same scored images per class with the same number of val images per class.

RS-TransCLIP: https://github.com/elkhouryk/RS-TransCLIP (MIT). Clone it and pass --rstc_dir.
Its zero-shot path softmaxes 100*cos(image, text) logits; our CLIP cache stores exactly those
logits, so passing softmax(logits) as `initial_predictions` reproduces the official path.
For SAM 3 there is no image embedding; the solver then runs on the presence vector of all
phrases (standardised logits), which is the same feature our other transductive baselines use.

python experiments/seeds_transclip.py --work_dir $WORK --rstc_dir /content/RS-TransCLIP \
    --datasets aid,ucm,resisc45,sun_aid --models sam3,openai,remoteclip --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import multi_benchmark as MB                      # noqa: E402
from sam3_rs import concept_translation as CT     # noqa: E402
from sam3_rs.benchmarks import SPECS              # noqa: E402


def make_rstc(rstc_dir):
    sys.path.insert(0, rstc_dir)
    import torch
    import TransCLIP_solver.TransCLIP as TC
    import TransCLIP_solver.TransCLIP_utils as TU
    from TransCLIP_solver.TransCLIP import TransCLIP_solver

    def cls_acc(output, target, topk=1):   # upstream float(np.array([x])) fails on numpy>=2
        pred = output.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return 100.0 * correct[:topk].reshape(-1).float().sum().item() / target.shape[0]
    TU.cls_acc = TC.cls_acc = cls_acc

    # Upstream init_z(softmax=False) returns the *same* tensor as prior y_hat and assignment z,
    # so update_z's in-place write to z also overwrites the prior. On the zero-shot path
    # (softmax=True) they are separate tensors. Cloning restores the zero-shot semantics.
    def init_z(affinity, softmax=True):
        if softmax:
            return torch.softmax(affinity, 1), torch.softmax(affinity, 1)
        return affinity, affinity.clone()
    TU.init_z = TC.init_z = init_z

    def run(F, y, init):
        Fq = torch.tensor(np.asarray(F), dtype=torch.float32)
        Fq = Fq / Fq.norm(dim=-1, keepdim=True)
        K = init.shape[1]
        g = torch.Generator().manual_seed(0)         # only used for the solver's printout
        proto = torch.nn.functional.normalize(torch.randn(Fq.shape[1], K, generator=g), dim=0)
        _, _, _, acc = TransCLIP_solver(None, None, None, None, Fq, torch.tensor(y), proto,
                                        initial_predictions=torch.tensor(init, dtype=torch.float32),
                                        verbose=False)
        return float(acc)
    return run


def cached_split(ds, work_dir, data_root):
    """Read the split written by multi_benchmark.py without touching the images (the score
    caches are all that is needed here), so a fresh runtime does not re-download datasets."""
    if ds == "aid":
        sp = json.load(open(os.path.join(work_dir, "split_seed0.json")))
        sp["classes"] = sorted({c for _, c in sp["val"] + sp["test"]})
        return sp, os.path.join(data_root, "aid"), "blanchon/AID"
    return json.load(open(os.path.join(work_dir, "bench", ds, "split.json"))), os.path.join(data_root, ds), None


def load_pool(ds, model, work_dir, data_root):
    sp, data_dir, repo = cached_split(ds, work_dir, data_root)
    classes = sp["classes"]
    bank = MB.bank_for(ds, classes)
    R = CT.unique_phrases(bank)
    y = {k: np.array([classes.index(c) for _, c in sp[k]]) for k in ("val", "test")}
    if ds == "aid" and model == "sam3":
        X = MB.aid_sam3_from_legacy(work_dir, sp, R)
        F = None
    else:
        X, F = {}, {}
        for k in ("val", "test"):
            path = os.path.join(work_dir, "bench", ds, f"{model}_{k}.npz")
            assert os.path.exists(path), f"run multi_benchmark.py first: {path}"
            X[k], F[k] = MB.score_split(model, None, sp[k], R, data_dir, repo, SPECS[ds]["view"], path)
        if model == "sam3":
            F = None
    return bank, classes, R, X, F, y


def resplit(X, F, y, seed):
    if seed == 0:
        return X, F, y
    Xa = np.concatenate([X["val"], X["test"]])
    ya = np.concatenate([y["val"], y["test"]])
    Fa = None if F is None else np.concatenate([F["val"], F["test"]])
    rng = np.random.default_rng(seed)
    nv = np.bincount(y["val"], minlength=ya.max() + 1)
    vi = []
    for c in range(ya.max() + 1):
        idx = np.flatnonzero(ya == c)
        vi += list(rng.permutation(idx)[:nv[c]])
    vm = np.zeros(len(ya), bool); vm[vi] = True
    X2 = {"val": Xa[vm], "test": Xa[~vm]}
    y2 = {"val": ya[vm], "test": ya[~vm]}
    F2 = None if Fa is None else {"val": Fa[vm], "test": Fa[~vm]}
    return X2, F2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--data_root", default="/content/bench_data")
    ap.add_argument("--datasets", default="aid,ucm,resisc45,sun_aid")
    ap.add_argument("--models", default="sam3,openai,remoteclip")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--rstc_dir", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rstc = make_rstc(a.rstc_dir) if a.rstc_dir else None
    out = os.path.join(a.work_dir, "bench", f"seeds{a.tag}.json")
    for ds in a.datasets.split(","):
        for model in a.models.split(","):
            bank, classes, R, X, F, y = load_pool(ds, model, a.work_dir, a.data_root)
            kind = MB.KIND[model]
            for seed in map(int, a.seeds.split(",")):
                key = f"{ds}/{model}/seed{seed}"
                done = json.load(open(out)) if os.path.exists(out) else {}
                if key in done:
                    continue
                X2, F2, y2 = resplit(X, F, y, seed)
                S = {k: {"X": X2[k], "y": y2[k]} for k in ("val", "test")}
                res = MB.evaluate(bank, classes, S, R, kind, F2, rstc=rstc)
                done = json.load(open(out)) if os.path.exists(out) else {}
                done[key] = res
                json.dump(done, open(out, "w"), indent=1)
                print(f"  [{key}] base {res['baseline']:.1f} | PPR {res['+PPR']:.1f} | "
                      f"RS-TransCLIP {res.get('+RS-TransCLIP', float('nan')):.1f} | CLEAR {res['CLEAR']:.1f} | "
                      f"CLEAR+RSTC {res.get('CLEAR+RS-TransCLIP', float('nan')):.1f} | CLEAR-FS {res['CLEAR-FS']:.1f}",
                      flush=True)


if __name__ == "__main__":
    main()
