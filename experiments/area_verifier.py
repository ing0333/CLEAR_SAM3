"""Negative result: a 'main subject' verifier based on how much of the image a concept covers.

Hypothesis: presence answers "is X anywhere?", so re-rank the top-M presence candidates by
"how much of the image does X cover?" (SAM 3 semantic head / instance masks).
Result on AID test: every coverage rule lost accuracy (e.g. presence x semantic coverage,
M=3: 36.8% -> 34.5%, +176 fixed / -232 broken). Two reasons:
  1. the defining object of many aerial scenes is small (church, storage tanks, baseball field);
  2. the semantic head, like the localization queries, is trained only on positive concepts,
     so it can paint large areas for absent concepts.

Usage: python experiments/area_verifier.py --work_dir ... --data_dir /content/aid
(needs the class-name presence cache from run_pipeline.py and a GPU)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam3_rs import metrics as M                           # noqa: E402
from sam3_rs.data import class_to_prompt, load_img, load_or_make_split  # noqa: E402

RULES = {
    "presence": lambda F: F[..., 0],
    "presence x sem_cov": lambda F: F[..., 0] * F[..., 1],
    "sem_cov": lambda F: F[..., 1],
    "presence x sem_area": lambda F: F[..., 0] * F[..., 2],
    "presence x inst_area": lambda F: F[..., 0] * F[..., 3],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--data_dir", default="/content/aid")
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    from sam3_rs.scoring import SAM3Scorer
    split = load_or_make_split(os.path.join(a.work_dir, "split_seed0.json"))
    classes = sorted({c for _, c in split["val"] + split["test"]})
    names = [class_to_prompt(c) for c in classes]
    scorer = SAM3Scorer(enable_segmentation=True, ckpt_cache_dir="/content/hf_local")
    T = scorer.encode_texts(names)
    for k in ("val", "test"):
        z = np.load(os.path.join(a.work_dir, "cache", f"all_phrases_{k}.npz"))
        col = {str(p): j for j, p in enumerate(z["phrases"])}
        P = np.stack([np.nan_to_num(z["pres"][:, col[n]]) for n in names], 1)
        y = np.array([classes.index(c) for _, c in split[k]])
        cand = np.argsort(-P, 1)[:, :a.K]
        F = np.zeros((len(y), a.K, 5), np.float32)
        for i, (f, _) in enumerate(split[k]):
            img = load_img(a.data_dir, f)
            if img is not None:
                F[i] = scorer.area_signals(img, scorer.sub_text(T, cand[i]))
        b0 = P.argmax(1)
        print(f"[{k}] baseline {M.accuracy(P, y):.1f}%")
        for M_ in (2, 3, 5):
            for name, rule in RULES.items():
                pred = cand[np.arange(len(y)), rule(F[:, :M_]).argmax(1)]
                fx, br = M.fixed_broken(b0, pred, y)
                print(f"  M={M_} {name:<22s} {100*(pred == y).mean():5.1f}%  (+{fx} / -{br})")


if __name__ == "__main__":
    main()
