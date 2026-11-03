"""End-to-end reproduction on AID: scores -> diagnosis -> CLEAR modules -> few-shot PSC.

Usage (Colab, GPU):
    python experiments/run_pipeline.py --work_dir /content/drive/MyDrive/SAM3_PGLC \
        --data_dir /content/aid --phrases prompts/phrases.json

Stage 1 (GPU) caches, per split, the presence and best localization score of every phrase in
prompts/phrases.json (149 unique phrases; ~3.4 s/image on an L4, ~3 h for 3,000 images,
resumable). Stage 2 (CPU, seconds) reproduces every table in the README.
All choices (phrases, T, r, regularisation) are made on val; test is only read at the end.
"""
from __future__ import annotations

import argparse
import collections
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
from sam3_rs.composition import PartComposer           # noqa: E402
from sam3_rs.data import class_to_prompt, download, load_img, load_or_make_split  # noqa: E402


# ----------------------------------------------------------------------------- stage 1
def compute_scores(split_items, phrases, data_dir, out_path, save_every=300):
    from sam3_rs.scoring import SAM3Scorer
    files = [f for f, _ in split_items]
    N, P = len(files), len(phrases)
    pres = np.full((N, P), np.nan, np.float32)
    loc = np.full((N, P), np.nan, np.float32)
    start = 0
    if os.path.exists(out_path):
        z = np.load(out_path)
        if [str(p) for p in z["phrases"]] == list(phrases):
            pres, loc, start = z["pres"], z["loc"], int(z["done"])
    if start >= N:
        return pres, loc
    scorer = SAM3Scorer(enable_segmentation=False, ckpt_cache_dir="/content/hf_local")
    T = scorer.encode_texts(phrases)
    t0 = time.time()
    for i in range(start, N):
        img = load_img(data_dir, files[i])
        if img is not None:
            p, l, _ = scorer.score(img, T, chunk=30, topk=1)
            pres[i], loc[i] = p, l[:, 0]
        if (i + 1) % save_every == 0 or i == N - 1:
            np.savez(out_path, phrases=np.array(phrases), files=np.array(files), pres=pres, loc=loc, done=i + 1)
            print(f"  {os.path.basename(out_path)} {i+1}/{N}  {(time.time()-t0)/60:.1f} min", flush=True)
    return pres, loc


# ----------------------------------------------------------------------------- stage 2
def evaluate(split, bank, classes, S, R):
    """S[name] = dict(pres=[N,P], loc=[N,P], y=[N]); phrase order = R (list of phrases)."""
    col = {p: j for j, p in enumerate(R)}
    H = {k: {p: np.nan_to_num(S[k]["pres"][:, j]) for p, j in col.items()} for k in S}
    yv, yt = S["val"]["y"], S["test"]["y"]
    names = [class_to_prompt(c) for c in classes]
    base = {k: np.stack([H[k][n] for n in names], 1) for k in S}
    locb = {k: np.stack([np.nan_to_num(S[k]["loc"][:, col[n]]) for n in names], 1) for k in S}
    res = collections.OrderedDict()

    # diagnosis ------------------------------------------------------------------
    res["baseline_test_acc"] = M.accuracy(base["test"], yt)
    res["baseline_loc_only_test_acc"] = M.accuracy(locb["test"], yt)
    res["baseline_product_test_acc"] = M.accuracy(base["test"] * locb["test"], yt)
    res["trigger_signals_test"] = M.trigger_report(base["test"], locb["test"], yt)
    et = M.error_types(base["test"], yt)
    res["error_types_test"] = {k: int(v.sum()) for k, v in et.items()}
    pred0 = base["test"].argmax(1)
    res["skew_top5_share_test"] = BAL.top_k_share(pred0, len(classes))
    res["most_predicted_test"] = {classes[k]: int(n) for k, n in
                                  collections.Counter(pred0).most_common(5)}

    # (3) balance: choose T on val -------------------------------------------------
    Ts = (0.1, 0.25, 0.5, 1.0, 2.0)
    T = max(Ts, key=lambda t: M.accuracy(BAL.balance(base["val"], t)[0], yv))
    res["balance_T"] = T
    bal = lambda P: BAL.balance(P, T)[0]
    res["baseline+PPR"] = M.accuracy(bal(base["test"]), yt)

    # (1) concept translation -----------------------------------------------------
    s1 = CT.select_supervised(H["val"], bank, classes, yv)
    s2 = CT.select_label_free(H["val"], bank, classes)
    for tag, sel in (("OCT_S1", s1), ("OCT_S2", s2)):
        Pt = CT.score_matrix(H["test"], bank, classes, sel)
        res[tag] = M.accuracy(Pt, yt)
        res[tag + "+PPR"] = M.accuracy(bal(Pt), yt)
        res[tag + "_phrases"] = {c: bank[c][sel[c]] for c in classes}

    # (2) ensemble: choose (r, how) on val with PPR --------------------------------
    rank = ENS.rank_phrases(H["val"], bank, classes, yv)
    grid = [(r, h) for h in ("mean", "max") for r in (1, 2, 3, 5)]
    r, how = max(grid, key=lambda g: M.accuracy(bal(ENS.ensemble_scores(H["val"], rank, classes, *g)), yv))
    Pt = ENS.ensemble_scores(H["test"], rank, classes, r, how)
    res["MPV"] = M.accuracy(Pt, yt)
    res["CLEAR (OCT+MPV+PPR)"] = M.accuracy(bal(Pt), yt)
    res["MPV_setting"] = {"r": r, "how": how}

    # (4) part composition (few-shot) ----------------------------------------------
    ctrl = PartComposer(names).fit(H["val"], yv)
    res["control_LR_on_class_names"] = M.accuracy(ctrl.predict_proba(H["test"]), yt)
    psc = PartComposer(R).fit(H["val"], yv)
    Pp = psc.predict_proba(H["test"])
    res["PSC"] = M.accuracy(Pp, yt)
    res["CLEAR-FS (PSC+PPR)"] = M.accuracy(bal(Pp), yt)
    res["PSC_top_parts"] = {c: psc.top_parts(classes.index(c)) for c in
                            ("School", "Church", "Park", "Stadium", "Viaduct")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--data_dir", default="/content/aid")
    ap.add_argument("--phrases", default="prompts/phrases.json")
    ap.add_argument("--out", default="results/results.json")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.work_dir, "cache"), exist_ok=True)

    split = load_or_make_split(os.path.join(a.work_dir, "split_seed0.json"))
    classes = sorted({c for _, c in split["val"] + split["test"]})
    bank, _ = CT.load_bank(a.phrases)
    R = CT.unique_phrases(bank)
    download([f for f, _ in split["val"] + split["test"]], a.data_dir)

    S = {}
    for k in ("val", "test"):
        pres, loc = compute_scores(split[k], R, a.data_dir, os.path.join(a.work_dir, "cache", f"all_phrases_{k}.npz"))
        S[k] = {"pres": pres, "loc": loc, "y": np.array([classes.index(c) for _, c in split[k]])}

    res = evaluate(split, bank, classes, S, R)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k:<34s} {v:6.1f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
