"""Extra benchmarks: UC Merced, NWPU-RESISC45 (overhead) and SUN-AID (ground-level SUN397
scenes matched to AID concepts). Each is sampled once with a fixed seed, saved as PNG files
plus a split json, so every model sees exactly the same images.
"""
from __future__ import annotations

import io
import json
import os
import random
import re

from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPECS = {
    "ucm": dict(repo="blanchon/UC_Merced", split="train", per_class=50, n_val=20, view="overhead",
                phrases="prompts/phrases_ucm.json"),
    "resisc45": dict(repo="timm/resisc45", split="test", per_class=40, n_val=20, view="overhead",
                     phrases="prompts/phrases_resisc45.json"),
    "sun_aid": dict(repo="tanganke/sun397", split="test", per_class=40, n_val=20, view="ground",
                    phrases="prompts/phrases.json", classmap="prompts/sun_aid_classes.json"),
    "aid": dict(view="overhead", phrases="prompts/phrases.json"),   # original AID split (seed 0)
}

# CLIP prompt template per viewpoint (SAM 3 gets the bare noun phrase)
TEMPLATES = {"overhead": "a satellite photo of {}.", "ground": "a photo of {}."}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_/()\-]+", " ", s)).strip().lower()


def load_phrase_bank(name: str) -> dict[str, list[str]]:
    return json.load(open(os.path.join(HERE, SPECS[name]["phrases"])))["phrases"]


def prepare(name: str, out_dir: str, split_path: str, seed: int = 0) -> dict:
    """Sample per class, save images under out_dir, write split json. Idempotent."""
    if os.path.exists(split_path):
        sp = json.load(open(split_path))
        if all(os.path.exists(os.path.join(out_dir, f)) for f, _ in sp["val"] + sp["test"]):
            return sp
    from datasets import Image as HFImage
    from datasets import load_dataset
    spec = SPECS[name]
    streaming = name == "sun_aid"
    ds = load_dataset(spec["repo"], split=spec["split"], streaming=streaming)
    names = ds.features["label"].names
    ds = ds.cast_column("image", HFImage(decode=False))

    if name == "sun_aid":
        cmap = json.load(open(os.path.join(HERE, spec["classmap"])))["map"]
        lookup = {_norm(n): i for i, n in enumerate(names)}
        label_of, source = {}, {}
        for concept, cands in cmap.items():
            for c in cands:
                if _norm(c) in lookup:
                    label_of[lookup[_norm(c)]] = concept; source[concept] = names[lookup[_norm(c)]]; break
        classes = sorted(source)
    else:
        bank = load_phrase_bank(name)
        label_of = {i: n for i, n in enumerate(names)}
        assert set(names) == set(bank), f"phrase bank keys differ from {name} labels"
        classes, source = sorted(names), {n: n for n in names}

    rng = random.Random(seed)
    picked = {c: [] for c in classes}
    if streaming:   # reservoir-sample per class while streaming the whole split
        seen = {c: 0 for c in classes}
        for ex in ds:
            c = label_of.get(ex["label"])
            if c is None:
                continue
            seen[c] += 1
            if len(picked[c]) < spec["per_class"]:
                picked[c].append(ex["image"]["bytes"])
            else:
                j = rng.randrange(seen[c])
                if j < spec["per_class"]:
                    picked[c][j] = ex["image"]["bytes"]
    else:
        labels = ds["label"]
        for c in classes:
            idx = [i for i, l in enumerate(labels) if label_of[l] == c]
            for i in rng.sample(idx, min(spec["per_class"], len(idx))):
                picked[c].append(ds[i]["image"]["bytes"])

    split = {"val": [], "test": [], "classes": classes, "source_label": source}
    for c in classes:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
        for j, b in enumerate(picked[c]):
            rel = f"{c}/{j:03d}.png"
            Image.open(io.BytesIO(b)).convert("RGB").save(os.path.join(out_dir, rel))
            split["val" if j < spec["n_val"] else "test"].append([rel, c])
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    json.dump(split, open(split_path, "w"))
    return split
