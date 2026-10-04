import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import paths

paths.add(paths.SNAP_DIR)
from src.snap import SNAP  # noqa: E402


class SNAPPlus(nn.Module):
    """SNAP + semantic point adapter + K mask tokens (per-mask IoU) + native text prompts."""

    def __init__(self, snap: SNAP, sem_dim=768, num_masks=3, use_sem=True, use_text=True):
        super().__init__()
        self.snap = snap
        D = snap.backbone.backbone_out_channels
        self.use_sem, self.use_text, self.K = use_sem, use_text, num_masks
        if use_sem:
            self.sem_adapter = nn.Sequential(nn.Linear(sem_dim, D), nn.GELU(), nn.Linear(D, D))
            nn.init.zeros_(self.sem_adapter[-1].weight)
            nn.init.zeros_(self.sem_adapter[-1].bias)
        self.mask_tokens = nn.Parameter(snap.mask_token.weight.detach().repeat(num_masks, 1)
                                        + 0.02 * torch.randn(num_masks, D, device=snap.mask_token.weight.device))
        self.iou_mlp = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))
        self.text_type = nn.Parameter(torch.zeros(1, D))

    def backbone(self, data_dict, sem=None):
        data_dict = self.snap.run_input_encoder(data_dict)
        point, _ = self.snap.run_backbone(data_dict)
        if self.use_sem and sem is not None:
            point.feat = point.feat + self.sem_adapter(sem)
        return point

    def decode(self, point, coord, clicks=None, text=None):
        """clicks: (M, P, 3) or None; text: (M, 512) CLIP text or None -> logits (M, K, N), iou (M, K), label token (M, 512)."""
        s = self.snap
        feat = point.feat
        M = (clicks if clicks is not None else text).shape[0]
        prompts = []
        if clicks is not None:
            P = clicks.shape[1]
            enc = s.prompt_encoder(clicks)
            idx = torch.cdist(point.coord, clicks.reshape(-1, 3)).argmin(0)
            prompts.append(enc.view(-1, feat.shape[1]).add(feat[idx]).view(M, P, -1))
        if text is not None and self.use_text:
            prompts.append((s.text_prompt_encoder(text) + self.text_type)[:, None])
        out_tokens = torch.cat([s.iou_token.weight, self.mask_tokens, s.text_token.weight], 0)
        tokens = torch.cat([out_tokens[None].expand(M, -1, -1), *prompts], 1)
        pe = s.point_pos_embeddings_module(coord)
        pts = feat[None].expand(M, -1, -1)
        pts_out, tok_out = s.merge([tokens], [pts], [pe[None].expand(M, -1, -1)])
        pts_out, tok_out = pts_out[0], tok_out[0]
        iou_t, mask_t, txt_t = tok_out[:, 0], tok_out[:, 1:1 + self.K], tok_out[:, 1 + self.K]
        logits = torch.einsum("mnd,mkd->mkn", pts_out, s.mask_head(mask_t))
        iou = torch.sigmoid(self.iou_mlp(torch.cat([iou_t[:, None].expand(-1, self.K, -1), mask_t], -1))).squeeze(-1)
        return logits, iou, s.text_head(txt_t)


def load_snap(ckpt, device="cuda"):
    snap = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, return_mid_points=True)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    missing, _ = snap.load_state_dict(sd, strict=False)
    assert not missing, missing
    return snap.to(device)


class SnapPlusModel:
    """Inference wrapper matching SnapModel's encode/decode_clicks API."""

    def __init__(self, ckpt, device="cuda", domain="Indoor", openscene_ckpt=None):
        from .openscene_model import OpenSceneModel
        from .snap_model import GRID, build_transform
        state = torch.load(ckpt, map_location="cpu")
        cfg = state["cfg"]
        self.device = torch.device(device)
        snap = load_snap(cfg["snap_ckpt"], device)
        self.model = SNAPPlus(snap, sem_dim=cfg["sem_dim"], num_masks=cfg["num_masks"], use_sem=cfg["use_sem"],
                              use_text=cfg["use_text"]).to(self.device)
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        assert not unexpected and all(k.startswith("snap.backbone") for k in missing), (missing[:5], unexpected[:5])
        self.model.eval()
        self.transform = build_transform(domain, GRID[domain])
        self.sem = OpenSceneModel(openscene_ckpt or cfg["openscene_ckpt"], cfg["feature"]) if cfg["use_sem"] else None

    @torch.no_grad()
    def encode(self, coord, color=None, normal=None, strength=None):
        import numpy as np
        from torch_scatter import scatter_mean
        n = len(coord)
        d = dict(coord=coord.astype(np.float32),
                 color=np.zeros((n, 3), np.float32),  # SNAP is trained XYZ-only: colour/normals zeroed before normalisation
                 normal=np.zeros((n, 3), np.float32),
                 strength=np.zeros((n, 1), np.float32) if strength is None else strength.reshape(n, 1).astype(np.float32))
        d = self.transform(d)
        d = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in d.items()}
        sem = None
        if self.sem is not None:
            sem = F.normalize(scatter_mean(self.sem(coord), d["inverse"], dim=0, dim_size=len(d["coord"])), dim=-1)
        self.point = self.model.backbone(d, sem)
        self.coord = d["coord"]
        return d["coord"], d["inverse"]

    @torch.no_grad()
    def decode_clicks(self, clicks):
        with torch.autocast("cuda", dtype=torch.float16):
            logits, iou, _ = self.model.decode(self.point, self.coord, clicks=clicks[None].float())
        return logits[0, int(iou[0].argmax())] > 0

    @torch.no_grad()
    def decode_text(self, text_emb):
        with torch.autocast("cuda", dtype=torch.float16):
            logits, iou, _ = self.model.decode(self.point, self.coord, text=text_emb[None].float())
        return logits[0, int(iou[0].argmax())] > 0
