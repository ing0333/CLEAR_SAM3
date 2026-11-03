"""Mean ± std over the 5 val/test splits written by seeds_transclip.py, plus the segmentation table.

python experiments/make_seed_report.py results/seeds.json results/seg_results.json > results/seeds.md
"""
from __future__ import annotations

import json
import sys

import numpy as np

DS = [("aid", "AID", "위"), ("ucm", "UC Merced", "위"), ("resisc45", "NWPU-RESISC45", "위"),
      ("sun_aid", "SUN-AID", "앞")]
MODELS = [("sam3", "SAM 3"), ("openai", "CLIP ViT-L/14"), ("remoteclip", "RemoteCLIP ViT-L/14")]
MAIN = ["baseline", "+PPR", "+TransLP", "+TransCLIP-lite", "+RS-TransCLIP", "CLEAR",
        "CLEAR+RS-TransCLIP", "PSC", "CLEAR-FS", "linear probe (image emb.)"]
FULL = ["baseline", "+PPR", "+TransLP", "+TransCLIP-lite", "+RS-TransCLIP", "OCT-S1", "OCT-S1+PPR",
        "OCT-S2+PPR", "MPV", "CLEAR", "OCT+MPV+TransLP", "OCT+MPV+TransCLIP-lite", "CLEAR+RS-TransCLIP",
        "LR(class names)", "PSC", "CLEAR-FS", "linear probe (image emb.)"]
SEG = ["names", "SegEarth-OV3", "OCT-all", "OCT-S1", "OCT+MPV"]


def stats(R, ds, m, k):
    v = [R[key][k] for key in R if key.startswith(f"{ds}/{m}/seed") and k in R[key]]
    return (np.mean(v), np.std(v), len(v)) if v else None


def fmt(s, bold=False):
    if s is None:
        return "–"
    t = f"{s[0]:.1f} ± {s[1]:.1f}"
    return f"**{t}**" if bold else t


def main(seed_path, seg_path=None):
    R = json.load(open(seed_path))
    out = ["## 장면 분류: 5개 split 평균 ± 표준편차 (test 정확도 %)\n"]
    out.append("| 데이터셋 | 모델 | " + " | ".join(MAIN) + " |")
    out.append("| --- | --- |" + " --- |" * len(MAIN))
    for d, dn, view in DS:
        for m, mn in MODELS:
            if stats(R, d, m, "baseline") is None:
                continue
            row = [fmt(stats(R, d, m, k), k == "CLEAR") for k in MAIN]
            out.append(f"| {dn} ({view}) | {mn} | " + " | ".join(row) + " |")
    out.append("\n## 모델별 평균 (데이터셋 4개, 5 split)\n")
    out.append("| 모델 | " + " | ".join(MAIN) + " |")
    out.append("| --- |" + " --- |" * len(MAIN))
    for m, mn in MODELS:
        cells = []
        for k in MAIN:
            v = [stats(R, d, m, k) for d, _, _ in DS]
            v = [x[0] for x in v if x is not None]
            cells.append(f"{np.mean(v):.1f}" if len(v) == len(DS) else "–")
        out.append(f"| {mn} | " + " | ".join(cells) + " |")
    out.append("\n## 비교: CLEAR vs 공식 RS-TransCLIP (split별 승수)\n")
    out.append("| 데이터셋 | 모델 | CLEAR > RS-TransCLIP | CLEAR+RS-TransCLIP > CLEAR |")
    out.append("| --- | --- | --- | --- |")
    for d, dn, _ in DS:
        for m, mn in MODELS:
            keys = [k for k in R if k.startswith(f"{d}/{m}/seed") and "+RS-TransCLIP" in R[k]]
            if not keys:
                continue
            w1 = sum(R[k]["CLEAR"] > R[k]["+RS-TransCLIP"] for k in keys)
            w2 = sum(R[k]["CLEAR+RS-TransCLIP"] > R[k]["CLEAR"] for k in keys)
            out.append(f"| {dn} | {mn} | {w1}/{len(keys)} | {w2}/{len(keys)} |")
    out.append("\n## 전체 ablation (평균 ± 표준편차)\n")
    cols = [(d, m) for d, _, _ in DS for m, _ in MODELS if stats(R, d, m, "baseline")]
    out.append("| method | " + " | ".join(f"{d}/{m}" for d, m in cols) + " |")
    out.append("| --- |" + " --- |" * len(cols))
    for k in FULL:
        out.append(f"| {k} | " + " | ".join(fmt(stats(R, d, m, k)) for d, m in cols) + " |")

    if seg_path:
        S = json.load(open(seg_path))
        out.append("\n## 분할: mIoU % (5개 split, 선택 100장 / test 400장)\n")
        out.append("| 데이터셋 | " + " | ".join(SEG) + " |")
        out.append("| --- |" + " --- |" * len(SEG))
        for ds, res in S.items():
            seeds = [v for k, v in res.items() if k.startswith("seed")]
            cells = []
            for k in SEG:
                v = [s["mIoU"][k] for s in seeds]
                cells.append(f"{np.mean(v):.1f} ± {np.std(v):.1f}")
            out.append(f"| {ds} | " + " | ".join(cells) + " |")
        for ds, res in S.items():
            seeds = [v for k, v in res.items() if k.startswith("seed")]
            out.append(f"\n**{ds} 클래스별 IoU (5 split 평균)**\n")
            out.append("| 클래스 | " + " | ".join(SEG) + " |")
            out.append("| --- |" + " --- |" * len(SEG))
            for c, cname in enumerate(res["classes"]):
                cells = [f"{np.nanmean([s['IoU'][k][c] for s in seeds]):.1f}" for k in SEG]
                out.append(f"| {cname} | " + " | ".join(cells) + " |")
            out.append("\nOCT-S1이 고른 표현 (split별): " + "; ".join(
                ", ".join(s["mIoU"]["chosen"]["S1"]) for s in seeds))
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
