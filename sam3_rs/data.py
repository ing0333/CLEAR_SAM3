"""AID (Aerial Image Dataset) utilities: listing, stratified split, parallel download, loading."""
from __future__ import annotations

import collections
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

AID_REPO = "blanchon/AID"   # 30 classes, 10,000 images, 600x600, one folder per class


def class_to_prompt(c: str) -> str:
    """'BareLand' -> 'bare land' (the plain class-name prompt used as the baseline)."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", c).lower()


def list_aid(repo: str = AID_REPO) -> dict[str, list[str]]:
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.lower().endswith((".jpg", ".png", ".tif")) and f.count("/") >= 2]
    by_cls = collections.defaultdict(list)
    for f in files:
        by_cls[f.split("/")[-2]].append(f)
    return dict(sorted(by_cls.items()))


def make_split(by_cls: dict, per_class: int = 100, n_val: int = 20, seed: int = 0) -> dict:
    """Per class: sample `per_class` images, first `n_val` -> val, rest -> test."""
    rng = random.Random(seed)
    split = {"val": [], "test": []}
    for c in sorted(by_cls):
        pick = rng.sample(sorted(by_cls[c]), per_class)
        split["val"] += [[f, c] for f in pick[:n_val]]
        split["test"] += [[f, c] for f in pick[n_val:]]
    return split


def load_or_make_split(path: str, **kw) -> dict:
    if os.path.exists(path):
        return json.load(open(path))
    split = make_split(list_aid(), **kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(split, open(path, "w"))
    return split


def download(files: list[str], local_dir: str, repo: str = AID_REPO, workers: int = 32) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    missing = [f for f in files if not os.path.exists(os.path.join(local_dir, f))]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda f: hf_hub_download(repo, f, repo_type="dataset", local_dir=local_dir), missing))


def load_img(local_dir: str, f: str, repo: str | None = AID_REPO):
    """Open an image; re-download once from `repo` if broken (repo=None: no retry);
    return None if still unreadable."""
    for attempt in range(2):
        try:
            return Image.open(os.path.join(local_dir, f)).convert("RGB")
        except Exception:
            if attempt == 0 and repo is not None:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo, f, repo_type="dataset", local_dir=local_dir, force_download=True)
    return None
