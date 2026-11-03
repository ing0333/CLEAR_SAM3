"""Turn bench/multi_benchmark.json into the markdown tables used in the README.

python experiments/make_report.py results/multi_benchmark.json > results/multi_benchmark.md
"""
from __future__ import annotations

import json
import sys

DS = [("aid", "AID", "위"), ("ucm", "UC Merced", "위"), ("resisc45", "NWPU-RESISC45", "위"),
      ("sun_aid", "SUN-AID", "앞")]
MODELS = [("sam3", "SAM 3"), ("openai", "CLIP ViT-L/14"), ("remoteclip", "RemoteCLIP ViT-L/14")]
LABEL_FREE = ["baseline", "+PPR", "+TransLP", "+TransCLIP-lite", "CLEAR"]
FEW_SHOT = ["LR(class names)", "PSC", "CLEAR-FS", "linear probe (image emb.)"]
ABLATION = ["baseline", "+PPR", "+TransLP", "+TransCLIP-lite", "OCT-S1", "OCT-S1+PPR", "OCT-S2+PPR",
            "MPV", "CLEAR", "OCT+MPV+TransLP", "OCT+MPV+TransCLIP-lite",
            "LR(class names)", "PSC", "CLEAR-FS", "linear probe (image emb.)"]


def f(r, k):
    return f"{r[k]:.1f}" if k in r else "–"


def main(path):
    R = json.load(open(path))
    keys = [(d, m) for d, _, _ in DS for m, _ in MODELS if f"{d}/{m}" in R]
    out = []
    out.append("## 학습 없음 (label-free; val 라벨은 표현·T 고르기에만)\n")
    out.append("| 데이터셋 | 시점 | 모델 | " + " | ".join(LABEL_FREE) + " | CLEAR − baseline |")
    out.append("| --- | --- | --- |" + " --- |" * (len(LABEL_FREE) + 1))
    for d, name, view in DS:
        for m, mname in MODELS:
            r = R.get(f"{d}/{m}")
            if r is None:
                continue
            cells = [f(r, k) for k in LABEL_FREE]
            cells[-1] = f"**{cells[-1]}**"
            out.append(f"| {name} | {view} | {mname} | " + " | ".join(cells)
                       + f" | +{r['CLEAR'] - r['baseline']:.1f} |")
    out.append("\n## few-shot (val 클래스당 20장으로 가중치 학습)\n")
    out.append("| 데이터셋 | 모델 | " + " | ".join(FEW_SHOT) + " |")
    out.append("| --- | --- |" + " --- |" * len(FEW_SHOT))
    for d, name, _ in DS:
        for m, mname in MODELS:
            r = R.get(f"{d}/{m}")
            if r is not None:
                out.append(f"| {name} | {mname} | " + " | ".join(f(r, k) for k in FEW_SHOT) + " |")

    out.append("\n## 평균 (데이터셋 4개)\n")
    out.append("| 모델 | baseline | +PPR | +TransLP | +TransCLIP-lite | CLEAR | CLEAR-FS |")
    out.append("| --- |" + " --- |" * 6)
    for m, mname in MODELS:
        rs = [R[f"{d}/{m}"] for d, _, _ in DS if f"{d}/{m}" in R]
        if not rs:
            continue
        avg = lambda k: sum(r[k] for r in rs) / len(rs)
        tag = "" if len(rs) == len(DS) else f" ({len(rs)}개)"
        out.append(f"| {mname}{tag} | " + " | ".join(f"{avg(k):.1f}" for k in
                   ["baseline", "+PPR", "+TransLP", "+TransCLIP-lite", "CLEAR", "CLEAR-FS"]) + " |")

    out.append("\n## 전체 ablation\n")
    out.append("| method | " + " | ".join(f"{d}/{m}" for d, m in keys) + " |")
    out.append("| --- |" + " --- |" * len(keys))
    for k in ABLATION:
        out.append(f"| {k} | " + " | ".join(f(R[f'{d}/{m}'], k) for d, m in keys) + " |")

    out.append("\n## 고른 설정 (val)\n")
    out.append("| run | PPR T | MPV (r, 방식) | TransLP (α, T, k) | TransCLIP-lite (T, λ) |")
    out.append("| --- | --- | --- | --- | --- |")
    for d, m in keys:
        r = R[f"{d}/{m}"]
        out.append(f"| {d}/{m} | {r.get('T')} | {r.get('MPV_setting')} | {r.get('TransLP_cfg')} | "
                   f"{r.get('TransCLIP-lite_cfg')} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/multi_benchmark.json")
