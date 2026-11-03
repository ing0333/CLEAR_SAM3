"""Does translating class names help SAM 3 *segment* remote-sensing land cover?

Per-phrase score maps follow SegEarth-OV3 (Li et al., 2025) so that the only thing that changes
between methods is the wording of the prompts:

    map(phrase) = max( max_i [ mask_i * score_i ]  (instances with score_i > conf_thr),
                       semantic-head probability ) * presence

score_i = sigmoid(pred_logit_i) * presence, as in the SAM 3 processor. A class map is the max
(or mean) over the phrases chosen for that class; the label of a pixel is the argmax class.
Maps are stored as uint8 at a fixed size so every prompt-selection experiment runs on CPU.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------- GPU side
def seg_maps(scorer, img, text_out: dict, size: int = 256, conf_thr: float = 0.5,
             chunk: int = 8) -> np.ndarray:
    """Return uint8 [P, size, size] fused maps for every encoded prompt (SAM3Scorer built with
    enable_segmentation=True)."""
    import torch
    import torch.nn.functional as Fnn
    assert scorer.model.segmentation_head is not None, "build SAM3Scorer(enable_segmentation=True)"
    with torch.inference_mode():
        bo = scorer._encode_image(img, text_out)
        P = text_out["language_mask"].shape[0]
        outs = []
        for s in range(0, P, chunk):
            e = min(P, s + chunk)
            o = scorer._decode(bo, s, e)
            n = e - s
            pres = o["presence_logit_dec"].float().sigmoid().view(n)
            sem = Fnn.interpolate(o["semantic_seg"].float(), size=(size, size), mode="bilinear",
                                  align_corners=False).sigmoid().view(n, size, size)
            sc = o["pred_logits"].float().sigmoid().view(n, -1) * pres[:, None]
            w = sc * (sc > conf_thr)
            m = Fnn.interpolate(o["pred_masks"].float(), size=(size, size), mode="bilinear",
                                align_corners=False).sigmoid()
            inst = (m * w[:, :, None, None]).amax(1)
            fused = torch.maximum(inst, sem) * pres[:, None, None]
            outs.append((fused.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
            del o, m, inst, sem
        return torch.cat(outs).numpy()


# ----------------------------------------------------------------------- CPU side
def class_maps(maps: np.ndarray, sel: list[list[int]], how: str = "max") -> np.ndarray:
    """maps: uint8 [N, P, H, W]; sel[c] = phrase indices for class c -> float32 [N, C, H, W]."""
    out = np.empty((maps.shape[0], len(sel)) + maps.shape[2:], np.float32)
    for c, idx in enumerate(sel):
        sub = maps[:, idx].astype(np.float32) / 255.0
        out[:, c] = sub.max(1) if how == "max" else sub.mean(1)
    return out


def confusion(maps, gt, sel, n_cls, how="max", bg_thd=None, bg_idx=0, batch=50):
    """gt: uint8 [N, H, W] with 255 = ignore. Returns int64 [C, C] (rows = truth)."""
    conf = np.zeros((n_cls, n_cls), np.int64)
    for s in range(0, len(gt), batch):
        cm = class_maps(maps[s:s + batch], sel, how)
        pred = cm.argmax(1)
        if bg_thd is not None:
            pred[cm.max(1) < bg_thd] = bg_idx
        g = gt[s:s + batch]
        v = g != 255
        conf += np.bincount(g[v].astype(np.int64) * n_cls + pred[v], minlength=n_cls * n_cls
                            ).reshape(n_cls, n_cls)
    return conf


def iou(conf):
    tp = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - tp
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def miou(conf) -> float:
    return float(np.nanmean(iou(conf)) * 100)


def select_phrases(maps, gt, cand: list[list[int]], start: list[int], n_cls, rounds=3, **kw):
    """OCT-S1 for segmentation: coordinate ascent over classes, one phrase per class, maximising
    mIoU on the selection images. Returns (chosen index per class, per-class phrase scores of the
    last pass) -- the scores rank phrases for MPV."""
    cur = list(start)
    best = miou(confusion(maps, gt, [[i] for i in cur], n_cls, **kw))
    scores = [dict() for _ in range(n_cls)]
    for _ in range(rounds):
        changed = False
        for c in range(n_cls):
            for j in cand[c]:
                trial = list(cur); trial[c] = j
                m = best if j == cur[c] else miou(confusion(maps, gt, [[i] for i in trial], n_cls, **kw))
                scores[c][j] = m
                if m > best + 1e-9:
                    best, cur, changed = m, trial, True
        if not changed:
            break
    return cur, scores
