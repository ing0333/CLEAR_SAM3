"""Do the CLEAR modules carry over to other datasets, other viewpoints and other models?

Datasets : AID, UC Merced, NWPU-RESISC45 (overhead) + SUN-AID (ground-level, same concepts as AID)
Models   : SAM 3 (presence), CLIP ViT-L/14 (OpenAI), RemoteCLIP ViT-L/14
Methods  : the same modules for every model -- OCT, MPV, PPR, PSC -- plus extra baselines:
           two simplified transductive corrections in the spirit of RS-TransCLIP
           (TransLP: k-NN label propagation; TransCLIP-lite: EM prototype refinement with a
           text prior) and, for CLIP models, a few-shot linear probe on image embeddings.

Usage (Colab):
  python experiments/multi_benchmark.py --work_dir /content/drive/MyDrive/SAM3_PGLC \
      --datasets ucm,resisc45,sun_aid,aid --models openai,remoteclip,sam3
Every choice (phrase, T, r, alpha, regularisation) is made on val; test is read once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam3_rs import balance as BAL                      # noqa: E402
from sam3_rs import concept_translation as CT          # noqa: E402
from sam3_rs import ensemble as ENS                    # noqa: E402
from sam3_rs import metrics as M                       # noqa: E402
from sam3_rs import transductive as TR                 # noqa: E402
from sam3_rs.benchmarks import SPECS, TEMPLATES, load_phrase_bank, prepare  # noqa: E402
from sam3_rs.composition import PartComposer           # noqa: E402
from sam3_rs.data import download, load_img            # noqa: E402

KIND = {"sam3": "prob", "openai": "logit", "remoteclip": "logit"}


# ------------------------------------------------------------------------- datasets
def get_split(ds, work_dir, data_root):
    if ds == "aid":
        sp = json.load(open(os.path.join(work_dir, "split_seed0.json")))
        sp["classes"] = sorted({c for _, c in sp["val"] + sp["test"]})
        data_dir = os.path.join(data_root, "aid")
        download([f for f, _ in sp["val"] + sp["test"]], data_dir)
        return sp, data_dir, "blanchon/AID"
    data_dir = os.path.join(data_root, ds)
    sp = prepare(ds, data_dir, os.path.join(work_dir, "bench", ds, "split.json"))
    return sp, data_dir, None


def bank_for(ds, classes):
    bank = load_phrase_bank(ds if ds in ("ucm", "resisc45") else "aid")
    return {c: bank[c] for c in classes}


# -------------------------------------------------------------------------- scoring
def score_split(model, scorer, items, phrases, data_dir, repo, view, out_path, save_every=200):
    files = [f for f, _ in items]
    N, P = len(files), len(phrases)
    if os.path.exists(out_path):
        z = np.load(out_path)
        if [str(p) for p in z["phrases"]] == list(phrases) and int(z["done"]) >= N:
            return z["X"], (z["F"] if "F" in z.files else None)
    X = np.full((N, P), np.nan, np.float32)
    F = None
    t0 = time.time()
    if model == "sam3":
        T = scorer.encode_texts(phrases)
        for i, f in enumerate(files):
            img = load_img(data_dir, f, repo)
            if img is not None:
                X[i] = scorer.presence(img, T, chunk=30)
            if (i + 1) % save_every == 0 or i == N - 1:
                np.savez(out_path, phrases=np.array(phrases), X=X, done=i + 1)
                print(f"    sam3 {i+1}/{N} {(time.time()-t0)/60:.1f} min", flush=True)
    else:
        Tf = scorer.text_features(phrases, TEMPLATES[view])
        feats = []
        for s in range(0, N, 64):
            imgs = [load_img(data_dir, f, repo) for f in files[s:s + 64]]
            feats.append(scorer.image_features([im if im is not None else load_img(data_dir, files[0], repo) for im in imgs]))
        F = np.concatenate(feats).astype(np.float32)
        X = (100.0 * F @ Tf.T).astype(np.float32)
        np.savez(out_path, phrases=np.array(phrases), X=X, F=F, done=N)
    return X, F


def aid_sam3_from_legacy(work_dir, split, R):
    """Rebuild AID SAM 3 presence for all phrases from the development caches (no GPU)."""
    c = os.path.join(work_dir, "cache")
    bank = load_phrase_bank("aid")
    classes = sorted(bank)
    out = {}
    for k in ("val", "test"):
        z = np.load(os.path.join(c, f"pass1_{k}.npz"))
        assert [str(f) for f in z["files"]] == [f for f, _ in split[k]]
        H = {bank[cl][0]: z["pres"][:, j] for j, cl in enumerate(classes)}
        if k == "val":
            hard = ["Church", "School", "Viaduct", "Commercial", "Port", "Playground", "Desert", "RailwayStation"]
            PV = np.load(os.path.join(c, "prompt_variants_val.npy"))
            for i, (cl, j) in enumerate((cl, j) for cl in hard for j in range(5)):
                H.setdefault(bank[cl][j], PV[:, i])
            PN = np.load(os.path.join(c, "phrases_new_val.npy"))
            new = [cl for cl in bank if cl not in hard]   # phrases.json order == development order
            for i, (cl, j) in enumerate((cl, j) for cl in new for j in range(1, 5)):
                H.setdefault(bank[cl][j], PN[:, i])
        else:
            for fn in ("phrases_selected_test.npz", "phrases_rest_test.npz"):
                zz = np.load(os.path.join(c, fn))
                for j, p in enumerate(zz["need"]):
                    H.setdefault(str(p), zz["P"][:, j])
        missing = [p for p in R if p not in H]
        assert not missing, missing[:5]
        out[k] = np.stack([np.nan_to_num(H[p]) for p in R], 1).astype(np.float32)
    return out


class _Lazy:
    """Load a model only when a cache miss actually needs it (re-evaluation is CPU-only)."""
    def __init__(self, make):
        self._make, self._obj = make, None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._make()
        return getattr(self._obj, name)


# ------------------------------------------------------------------------ evaluation
def evaluate(bank, classes, S, R, kind, feats=None, rstc=None):
    """S[k] = dict(X=[N,P], y=[N]) for k in val/test; feats[k] = image features or None.
    rstc(features, labels, initial_probs) -> accuracy: optional official RS-TransCLIP solver."""
    col = {p: j for j, p in enumerate(R)}
    H = {k: {p: np.nan_to_num(S[k]["X"][:, j]) for p, j in col.items()} for k in S}
    yv, yt = S["val"]["y"], S["test"]["y"]
    isl = kind == "logit"
    names = [bank[c][0] for c in classes]
    base = {k: np.stack([H[k][n] for n in names], 1) for k in S}
    to_L = lambda P: P if isl else BAL.logit(P)
    G = {}  # graph features for TransLP
    for k in S:
        if feats is not None and feats.get(k) is not None:
            G[k] = feats[k]
        else:
            Z = BAL.logit(np.nan_to_num(S[k]["X"])) if not isl else S[k]["X"]
            G[k] = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    res = {"n_classes": len(classes), "n_val": int(len(yv)), "n_test": int(len(yt))}

    res["baseline"] = M.accuracy(base["test"], yt)
    Ts = (0.25, 0.5, 1.0, 2.0, 4.0)
    T = max(Ts, key=lambda t: M.accuracy(BAL.balance(base["val"], t, is_logit=isl)[0], yv))
    bal = lambda P: BAL.balance(P, T, is_logit=isl)[0]
    res["T"] = T
    res["+PPR"] = M.accuracy(bal(base["test"]), yt)
    # fair grid: CLIP logits are on a x100 scale, so T must reach well above 1
    grid = [(a, t, kk) for a in (0.3, 0.5, 0.7, 0.9) for t in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
            for kk in (5, 10, 20)]
    a_, t_, k_ = max(grid, key=lambda g: M.accuracy(
        TR.label_propagation(to_L(base["val"]), G["val"], g[0], k=g[2], T=g[1]), yv))
    tlp = lambda P: TR.label_propagation(to_L(P), G["test"], a_, k=k_, T=t_)
    res["TransLP_cfg"] = [a_, t_, k_]
    res["+TransLP"] = M.accuracy(tlp(base["test"]), yt)
    g3 = [(t, l) for t in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) for l in (0.03, 0.1, 0.3, 1.0)]
    t3, l3 = max(g3, key=lambda g: M.accuracy(TR.transclip_lite(to_L(base["val"]), G["val"], g[0], g[1]), yv))
    tcl = lambda P: TR.transclip_lite(to_L(P), G["test"], t3, l3)
    res["+TransCLIP-lite"] = M.accuracy(tcl(base["test"]), yt)
    res["TransCLIP-lite_cfg"] = [t3, l3]

    s1 = CT.select_supervised(H["val"], bank, classes, yv)
    P1 = CT.score_matrix(H["test"], bank, classes, s1)
    res["OCT-S1"] = M.accuracy(P1, yt)
    res["OCT-S1+PPR"] = M.accuracy(bal(P1), yt)
    s2 = CT.select_label_free(H["val"], bank, classes)
    res["OCT-S2+PPR"] = M.accuracy(bal(CT.score_matrix(H["test"], bank, classes, s2)), yt)

    rank = ENS.rank_phrases(H["val"], bank, classes, yv)
    g2 = [(r, h) for h in ("mean", "max") for r in (1, 2, 3, 5)]
    r, how = max(g2, key=lambda g: M.accuracy(bal(ENS.ensemble_scores(H["val"], rank, classes, *g)), yv))
    Pe = ENS.ensemble_scores(H["test"], rank, classes, r, how)
    res["MPV"] = M.accuracy(Pe, yt)
    res["CLEAR"] = M.accuracy(bal(Pe), yt)
    res["OCT+MPV+TransLP"] = M.accuracy(tlp(Pe), yt)
    res["OCT+MPV+TransCLIP-lite"] = M.accuracy(tcl(Pe), yt)
    res["MPV_setting"] = [r, how]
    if rstc is not None:   # official RS-TransCLIP solver (fixed published hyper-parameters)
        sm = lambda Z: np.exp(Z - Z.max(1, keepdims=True)) / np.exp(Z - Z.max(1, keepdims=True)).sum(1, keepdims=True)
        res["+RS-TransCLIP"] = rstc(G["test"], yt, sm(to_L(base["test"])))
        res["CLEAR+RS-TransCLIP"] = rstc(G["test"], yt, sm(bal(Pe) / T))

    ctrl = PartComposer(names, is_logit=isl).fit(H["val"], yv)
    res["LR(class names)"] = M.accuracy(ctrl.predict_proba(H["test"]), yt)
    psc = PartComposer(R, is_logit=isl).fit(H["val"], yv)
    Pp = psc.predict_proba(H["test"])
    res["PSC"] = M.accuracy(Pp, yt)
    res["CLEAR-FS"] = M.accuracy(bal(Pp), yt)
    if feats is not None and feats.get("val") is not None:
        from sklearn.linear_model import LogisticRegressionCV
        Fv, Ft = feats["val"], feats["test"]
        mu, sd = Fv.mean(0), Fv.std(0) + 1e-6
        lp = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1.0], cv=5, max_iter=5000).fit((Fv - mu) / sd, yv)
        res["linear probe (image emb.)"] = M.accuracy(lp.predict_proba((Ft - mu) / sd), yt)
    return res


ROWS = ["baseline", "+PPR", "+TransLP", "+TransCLIP-lite", "OCT-S1", "OCT-S1+PPR", "OCT-S2+PPR", "MPV", "CLEAR",
        "OCT+MPV+TransLP", "OCT+MPV+TransCLIP-lite", "LR(class names)", "PSC", "CLEAR-FS", "linear probe (image emb.)"]


def summary_table(allres):
    keys = [k for k in allres]
    lines = ["| method | " + " | ".join(keys) + " |", "| --- |" + " --- |" * len(keys)]
    for r in ROWS:
        cells = [f"{allres[k][r]:.1f}" if r in allres[k] else "–" for k in keys]
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--data_root", default="/content/bench_data")
    ap.add_argument("--datasets", default="ucm,resisc45,sun_aid,aid")
    ap.add_argument("--models", default="openai,remoteclip,sam3")
    ap.add_argument("--out", default="results/multi_benchmark.json")
    a = ap.parse_args()
    out_path = os.path.join(a.work_dir, "bench", "multi_benchmark.json")
    allres = json.load(open(out_path)) if os.path.exists(out_path) else {}
    scorers = {}
    for ds in a.datasets.split(","):
        sp, data_dir, repo = get_split(ds, a.work_dir, a.data_root)
        classes = sp["classes"]
        bank = bank_for(ds, classes)
        R = CT.unique_phrases(bank)
        view = SPECS[ds]["view"]
        y = {k: np.array([classes.index(c) for _, c in sp[k]]) for k in ("val", "test")}
        print(f"\n=== {ds}: {len(classes)} classes, val {len(y['val'])} / test {len(y['test'])}, {len(R)} phrases, view={view}", flush=True)
        for model in a.models.split(","):
            if ds == "aid" and model == "sam3":
                X = aid_sam3_from_legacy(a.work_dir, sp, R)
                S = {k: {"X": X[k], "y": y[k]} for k in X}; feats = None
            else:
                if model not in scorers:
                    if model == "sam3":
                        from sam3_rs.scoring import SAM3Scorer
                        scorers[model] = _Lazy(lambda: SAM3Scorer(ckpt_cache_dir="/content/hf_local"))
                    else:
                        from sam3_rs.clip_scoring import CLIPScorer
                        scorers[model] = _Lazy(lambda m=model: CLIPScorer(m))
                S, feats = {}, {}
                os.makedirs(os.path.join(a.work_dir, "bench", ds), exist_ok=True)
                for k in ("val", "test"):
                    Xk, Fk = score_split(model, scorers[model], sp[k], R, data_dir, repo, view,
                                         os.path.join(a.work_dir, "bench", ds, f"{model}_{k}.npz"))
                    S[k] = {"X": Xk, "y": y[k]}; feats[k] = Fk
                if model == "sam3":
                    feats = None
            res = evaluate(bank, classes, S, R, KIND[model], feats)
            # re-read before writing so concurrent runs (e.g. CLIP + SAM 3) don't clobber each other
            allres = json.load(open(out_path)) if os.path.exists(out_path) else {}
            allres[f"{ds}/{model}"] = res
            json.dump(allres, open(out_path, "w"), indent=1)
            print(f"  [{ds}/{model}] baseline {res['baseline']:.1f} | +PPR {res['+PPR']:.1f} | +TransLP {res['+TransLP']:.1f} | +TransCLIP-lite {res['+TransCLIP-lite']:.1f} | "
                  f"CLEAR {res['CLEAR']:.1f} | LR(names) {res['LR(class names)']:.1f} | CLEAR-FS {res['CLEAR-FS']:.1f}"
                  + (f" | linear probe {res['linear probe (image emb.)']:.1f}" if "linear probe (image emb.)" in res else ""), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(allres, open(a.out, "w"), indent=1)
    print("\n" + summary_table(allres))


if __name__ == "__main__":
    main()
