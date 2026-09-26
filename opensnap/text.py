import os

import clip
import torch
import torch.nn.functional as F

from .paths import DOWNLOADS


class TextEncoder:
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device, jit=False, download_root=os.path.join(DOWNLOADS, "clip"))
        self.model.eval()
        self.device = device
        self.cache = {}

    @torch.no_grad()
    def __call__(self, labels, templates=("a {} in a scene",)):
        key = (tuple(labels), tuple(templates))
        if key not in self.cache:
            embs = []
            for t in templates:
                tok = clip.tokenize([t.format(c) for c in labels]).to(self.device)
                embs.append(F.normalize(self.model.encode_text(tok).float(), dim=-1))
            self.cache[key] = F.normalize(torch.stack(embs).mean(0), dim=-1)
        return self.cache[key]
