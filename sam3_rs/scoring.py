"""Presence / localization scores from SAM 3, taken before they are multiplied.

SAM 3's decoder runs 200 object queries plus one global presence token per text prompt:

    p(query_i matches NP) = p(query_i matches NP | NP in image) * p(NP in image)
                            `---- localization (pred_logits) -'   `- presence -'

The public processor only returns the product thresholded at 0.5. This module returns
both factors so they can be analysed separately, and decodes many prompts for one image
in a single batched pass (the image backbone runs once per image).
"""
from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sam3.model.data_misc import FindStage
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


class SAM3Scorer:
    def __init__(self, device: str = "cuda", enable_segmentation: bool = False,
                 ckpt_cache_dir: str | None = None, amp_dtype=torch.bfloat16):
        # Download the checkpoint to local disk. Reading the 3.45 GB file back from a
        # network-mounted drive while it is still syncing can corrupt the load.
        ckpt = hf_hub_download("facebook/sam3", "sam3.pt", cache_dir=ckpt_cache_dir)
        self.model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False,
                                            enable_segmentation=enable_segmentation, device=device)
        self.proc = Sam3Processor(self.model, device=device)
        self.device = device
        self.amp = dict(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ text
    @torch.inference_mode()
    def encode_texts(self, prompts: list[str]) -> dict:
        """Encode prompts once and reuse them for every image."""
        with torch.autocast(**self.amp):
            return self.model.backbone.forward_text(list(prompts), device=self.device)

    def sub_text(self, text_out: dict, idx) -> dict:
        """Select a subset of already-encoded prompts."""
        i = torch.as_tensor(np.asarray(idx), device=self.device)
        return {k: (v[i] if k == "language_mask" else v[:, i]) for k, v in text_out.items()}

    # ----------------------------------------------------------------- image
    @torch.inference_mode()
    def _decode(self, backbone_out: dict, s: int, e: int):
        n = e - s
        fs = FindStage(img_ids=torch.zeros(n, device=self.device, dtype=torch.long),
                       text_ids=torch.arange(s, e, device=self.device, dtype=torch.long),
                       input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                       input_points=None, input_points_mask=None)
        with torch.autocast(**self.amp):
            # With the segmentation head disabled SAM 3 pops the image features from the
            # dict it receives, so pass a shallow copy for every chunk.
            return self.model.forward_grounding(
                backbone_out=dict(backbone_out), find_input=fs, find_target=None,
                geometric_prompt=self.model._get_dummy_prompt(num_prompts=n))

    @torch.inference_mode()
    def _encode_image(self, img, text_out: dict) -> dict:
        with torch.autocast(**self.amp):
            state = self.proc.set_image(img)
        bo = state["backbone_out"]
        bo.update(text_out)
        return bo

    @torch.inference_mode()
    def score(self, img, text_out: dict, chunk: int = 30, topk: int = 10):
        """Return presence [P], top-k localization scores [P,k] and their boxes [P,k,4]
        (cx, cy, w, h normalised to 0..1) for every encoded prompt."""
        bo = self._encode_image(img, text_out)
        P = text_out["language_mask"].shape[0]
        pres, locs, boxes = [], [], []
        for s in range(0, P, chunk):
            e = min(P, s + chunk)
            out = self._decode(bo, s, e)
            n = e - s
            p = out["presence_logit_dec"].float().sigmoid().view(n)
            l = out["pred_logits"].float().sigmoid().view(n, -1)
            b = out["pred_boxes"].float().view(n, -1, 4)
            v, i = l.topk(topk, dim=1)
            pres.append(p); locs.append(v)
            boxes.append(torch.gather(b, 1, i[..., None].expand(-1, -1, 4)))
        return (torch.cat(pres).cpu().numpy(), torch.cat(locs).cpu().numpy(),
                torch.cat(boxes).cpu().numpy())

    def presence(self, img, text_out: dict, chunk: int = 30) -> np.ndarray:
        return self.score(img, text_out, chunk=chunk, topk=1)[0]

    @torch.inference_mode()
    def area_signals(self, img, text_out: dict) -> np.ndarray:
        """Signals for the (unsuccessful) 'main subject' verifier. Needs enable_segmentation=True.
        Returns [P, 5]: presence, mean semantic-mask prob, semantic area>0.5,
        union area of instances with total score>0.5, number of such instances."""
        assert self.model.segmentation_head is not None, "build with enable_segmentation=True"
        bo = self._encode_image(img, text_out)
        n = text_out["language_mask"].shape[0]
        out = self._decode(bo, 0, n)
        pres = out["presence_logit_dec"].float().sigmoid().view(n)
        sem = out["semantic_seg"].float().sigmoid().view(n, -1)
        keep = out["pred_logits"].float().sigmoid().view(n, -1) * pres[:, None] > 0.5
        inst = ((out["pred_masks"].float() > 0) & keep[:, :, None, None]).any(1).float().view(n, -1)
        return torch.stack([pres, sem.mean(1), (sem > .5).float().mean(1), inst.mean(1),
                            keep.sum(1).float()], 1).cpu().numpy()
