"""Part 3: do OCT / MPV help SAM 3 segment land cover?

Datasets (val splits; official test labels are not public):
  OpenEarthMap  EVER-Z/open_earth_map, val 500 images, 8 classes (label 0 = unknown, ignored)
  LoveDA        chloechia/loveda (official val split), 500 of 1,669 images drawn with seed 0,
                7 classes (label 0 = no data, ignored; 'background' is a real class)
Each seed splits the images into 100 for choosing prompts and 400 for testing.

Prompt sets compared on the same SAM 3 maps:
  names        the dataset's official class names
  SegEarth-OV3 hand-written synonym lists from SegEarth-OV3 (max over synonyms)
  OCT-all      all 5 of our phrases per class, max (no labels used)
  OCT-S1       one phrase per class chosen on the 100 selection images
  OCT+MPV      top-r phrases per class (ranked on the selection images), max or mean
Settings follow SegEarth-OV3 configs: instance threshold 0.5 / 0.1 (LoveDA / OEM),
background threshold 0.5 for LoveDA. Maps and labels are compared at 256 x 256.

python experiments/seg_benchmark.py --work_dir $WORK --datasets openearthmap,loveda --stage score
python experiments/seg_benchmark.py --work_dir $WORK --datasets openearthmap,loveda --stage eval
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam3_rs import segmentation as SG   # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..", "prompts", "seg_phrases.json")
CFG = {"openearthmap": dict(conf_thr=0.1, bg_thd=None, n=500),
       "loveda": dict(conf_thr=0.5, bg_thd=0.5, n=500)}
SIZE = 256


def spec(ds):
    s = json.load(open(HERE))[ds]
    classes = [s["labels"][k] for k in sorted(s["labels"], key=int)]
    return classes, s["phrases"], s["segearth_ov3"]


def phrase_list(ds):
    classes, ours, se = spec(ds)
    R = []
    for c in classes:
        for p in ours[c] + se[c]:
            if p not in R:
                R.append(p)
    return R


def to_gt(mask: np.ndarray) -> np.ndarray:
    """labels 1..C -> 0..C-1, 0 -> 255 (ignore); nearest-neighbour resize to SIZE."""
    g = np.where(mask == 0, 255, mask.astype(np.int16) - 1).astype(np.uint8)
    return np.array(Image.fromarray(g).resize((SIZE, SIZE), Image.NEAREST))


# --------------------------------------------------------------------------- data
def prepare(ds, data_root):
    """Write images to data_root/ds/img/*.png and labels to data_root/ds/gt.npy (idempotent)."""
    d = os.path.join(data_root, ds)
    if os.path.exists(os.path.join(d, "gt.npy")):
        return sorted(glob.glob(os.path.join(d, "img", "*.png"))), np.load(os.path.join(d, "gt.npy"))
    os.makedirs(os.path.join(d, "img"), exist_ok=True)
    gts = []
    if ds == "openearthmap":
        from datasets import load_dataset
        dset = load_dataset("EVER-Z/open_earth_map", data_files={"val": "data/val-*"}, split="val",
                            verification_mode="no_checks")
        for i, ex in enumerate(dset):
            ex["image"].convert("RGB").save(os.path.join(d, "img", f"{i:04d}.png"))
            gts.append(to_gt(np.array(ex["mask"])))
    elif ds == "loveda":
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import HfApi, hf_hub_download
        repo = "chloechia/loveda"          # mirror of the official val split (urban + rural)
        names = sorted(f.split("/")[-1] for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if f.startswith("urban:rural val images/"))
        random.Random(0).shuffle(names)
        names = sorted(names[:CFG[ds]["n"]], key=lambda s: int(s.split(".")[0]))
        get = lambda f: hf_hub_download(repo, f, repo_type="dataset")
        with ThreadPoolExecutor(16) as ex:
            ip = list(ex.map(get, [f"urban:rural val images/{n}" for n in names]))
            mp = list(ex.map(get, [f"urban:rural val masks/{n}" for n in names]))
        for i, (a, b) in enumerate(zip(ip, mp)):
            Image.open(a).convert("RGB").save(os.path.join(d, "img", f"{i:04d}.png"))
            m = np.array(Image.open(b))
            assert m.ndim == 2 and m.max() <= 7, (b, m.shape, m.max())
            gts.append(to_gt(m))
        json.dump(names, open(os.path.join(d, "source_files.json"), "w"))
    gt = np.stack(gts)
    np.save(os.path.join(d, "gt.npy"), gt)
    return sorted(glob.glob(os.path.join(d, "img", "*.png"))), gt


# ------------------------------------------------------------------------- scoring
def score(ds, work_dir, data_root):
    files, gt = prepare(ds, data_root)
    R = phrase_list(ds)
    out = os.path.join(work_dir, "seg", ds)
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, "phrases.json"), "w"))
    todo = [i for i in range(len(files)) if not os.path.exists(os.path.join(out, f"{i:04d}.npz"))]
    print(f"=== {ds}: {len(files)} images, {len(R)} phrases, {len(todo)} to score", flush=True)
    if not todo:
        return
    from sam3_rs.scoring import SAM3Scorer
    sc = SAM3Scorer(enable_segmentation=True, ckpt_cache_dir="/content/hf_local")
    T = sc.encode_texts(R)
    t0 = time.time()
    for n, i in enumerate(todo):
        m = SG.seg_maps(sc, Image.open(files[i]).convert("RGB"), T, SIZE, CFG[ds]["conf_thr"])
        np.savez_compressed(os.path.join(out, f"{i:04d}.npz"), m=m)
        if (n + 1) % 50 == 0 or n == len(todo) - 1:
            print(f"    {n+1}/{len(todo)} {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------- evaluation
def evaluate(ds, work_dir, data_root, seeds, n_sel=100):
    classes, ours, se = spec(ds)
    C = len(classes)
    R = json.load(open(os.path.join(work_dir, "seg", ds, "phrases.json")))
    col = {p: j for j, p in enumerate(R)}
    _, gt = prepare(ds, data_root)
    maps = np.stack([np.load(os.path.join(work_dir, "seg", ds, f"{i:04d}.npz"))["m"]
                     for i in range(len(gt))])
    kw = dict(bg_thd=CFG[ds]["bg_thd"], bg_idx=0)
    cand = [[col[p] for p in ours[c]] for c in classes]
    sets = {"names": [[col[ours[c][0]]] for c in classes],
            "SegEarth-OV3": [[col[p] for p in se[c]] for c in classes],
            "OCT-all": cand}
    out = {}
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(len(gt))
        si, ti = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
        Ms, Gs, Mt, Gt = maps[si], gt[si], maps[ti], gt[ti]
        res, per = {}, {}
        for k, sel in sets.items():
            cf = SG.confusion(Mt, Gt, sel, C, "max", **kw)
            res[k], per[k] = SG.miou(cf), (SG.iou(cf) * 100).round(1).tolist()
        start = [cand[c][0] for c in range(C)]
        s1, scores = SG.select_phrases(Ms, Gs, cand, start, C, **kw)
        cf = SG.confusion(Mt, Gt, [[j] for j in s1], C, "max", **kw)
        res["OCT-S1"], per["OCT-S1"] = SG.miou(cf), (SG.iou(cf) * 100).round(1).tolist()
        rank = [sorted(cand[c], key=lambda j: -scores[c].get(j, -1)) for c in range(C)]
        grid = [(r, h) for r in (2, 3, 5) for h in ("max", "mean")]
        val = {g: SG.miou(SG.confusion(Ms, Gs, [rk[:g[0]] for rk in rank], C, g[1], **kw)) for g in grid}
        r, h = max(grid, key=lambda g: val[g])
        cf = SG.confusion(Mt, Gt, [rk[:r] for rk in rank], C, h, **kw)
        res["OCT+MPV"], per["OCT+MPV"] = SG.miou(cf), (SG.iou(cf) * 100).round(1).tolist()
        res["chosen"] = {"S1": [R[j] for j in s1], "MPV": [r, h]}
        out[f"seed{seed}"] = {"mIoU": res, "IoU": per}
        print(f"  [{ds} seed{seed}] " + " | ".join(f"{k} {v:.1f}" for k, v in res.items() if k != "chosen")
              + f" | S1={res['chosen']['S1']} MPV={res['chosen']['MPV']}", flush=True)
    out["classes"] = classes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--data_root", default="/content/seg_data")
    ap.add_argument("--datasets", default="openearthmap,loveda")
    ap.add_argument("--stage", default="all", choices=["score", "eval", "all"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    allres = {}
    for ds in a.datasets.split(","):
        if a.stage in ("score", "all"):
            score(ds, a.work_dir, a.data_root)
        if a.stage in ("eval", "all"):
            allres[ds] = evaluate(ds, a.work_dir, a.data_root, [int(s) for s in a.seeds.split(",")])
    if allres:
        p = os.path.join(a.work_dir, "seg", "seg_results.json")
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(allres)
        json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
