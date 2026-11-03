"""CLIP ViT-L/14 (OpenAI) and RemoteCLIP ViT-L/14 as comparison models.

Scores are logits: 100 * cos(image, text). Image embeddings are kept for the linear-probe
baseline and for the transductive (graph) correction.
"""
from __future__ import annotations

import numpy as np
import torch


class CLIPScorer:
    def __init__(self, which: str = "openai", device: str = "cuda"):
        import open_clip
        if which == "openai":
            model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        elif which == "remoteclip":
            from huggingface_hub import hf_hub_download
            model, _, pre = open_clip.create_model_and_transforms("ViT-L-14")
            ckpt = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt")
            msg = model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print("RemoteCLIP load:", msg)
        else:
            raise ValueError(which)
        self.model = model.to(device).eval()
        self.pre = pre
        self.tok = open_clip.get_tokenizer("ViT-L-14")
        self.device = device

    @torch.inference_mode()
    def text_features(self, phrases: list[str], template: str = "{}") -> np.ndarray:
        out = []
        for s in range(0, len(phrases), 256):
            t = self.tok([template.format(p) for p in phrases[s:s + 256]]).to(self.device)
            f = self.model.encode_text(t).float()
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.inference_mode()
    def image_features(self, imgs: list) -> np.ndarray:
        x = torch.stack([self.pre(im) for im in imgs]).to(self.device)
        f = self.model.encode_image(x).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
