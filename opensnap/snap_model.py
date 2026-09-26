import numpy as np
import torch
import torch.nn.functional as F

from . import paths

paths.add(paths.SNAP_DIR)
from src.snap import SNAP  # noqa: E402
from datasets.transforms import Compose  # noqa: E402

GRID = {"Indoor": 0.02, "Outdoor": 0.05, "Aerial": 0.33}
CONDITION = {"Indoor": "ScanNet", "Outdoor": "SemanticKITTI", "Aerial": "STPLS3D"}


def build_transform(domain, grid_size):
    keys = ("coord", "color", "normal", "strength")
    t = []
    if domain != "Outdoor":
        t.append(dict(type="CenterShift", apply_z=True))
    t.append(dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
                  keys=keys, return_grid_coord=True, return_inverse=True))
    if domain != "Outdoor":
        t += [dict(type="CenterShift", apply_z=False), dict(type="NormalizeColor")]
    t += [dict(type="Add", keys_dict={"condition": CONDITION[domain], "domain": domain}),
          dict(type="ToTensor"),
          dict(type="Collect", keys=("coord", "grid_coord", "color", "normal", "strength", "condition", "domain", "inverse"))]
    return Compose(t)


class SnapModel:
    def __init__(self, ckpt, device="cuda", domain="Indoor", grid_size=None, chunk=16):
        self.device = torch.device(device)
        self.domain = domain
        self.chunk = chunk
        self.transform = build_transform(domain, grid_size or GRID[domain])
        self.model = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, return_mid_points=True).to(self.device)
        sd = torch.load(ckpt, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
        missing, _ = self.model.load_state_dict(sd, strict=False)  # ckpt carries unused heads (merge_text, confidence_head)
        assert not missing, missing
        self.model.eval()

    @torch.no_grad()
    def encode(self, coord, color=None, normal=None, strength=None):
        n = len(coord)
        d = dict(coord=coord.astype(np.float32),
                 color=np.ones((n, 3), np.float32) * 255 if color is None else color.astype(np.float32),
                 normal=np.zeros((n, 3), np.float32) if normal is None else normal.astype(np.float32),
                 strength=np.zeros((n, 1), np.float32) if strength is None else strength.reshape(n, 1).astype(np.float32))
        d = self.transform(d)
        d = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in d.items()}
        d = self.model.run_input_encoder(d)
        point, _ = self.model.run_backbone(d)
        self.data, self.point = d, point
        return d["coord"], d["inverse"]

    @torch.no_grad()
    def decode(self, prompts):
        """prompts: (M, P, 3) in the encoded (voxel) frame -> logits (M, N), text (M, 512), iou (M,)"""
        logits, texts, ious = [], [], []
        for s in range(0, len(prompts), self.chunk):
            p = prompts[s:s + self.chunk].to(self.device).float()
            d = dict(self.data)
            d["point"], d["point_offset"] = p, [len(p)]
            with torch.autocast("cuda", dtype=torch.float16):
                seg, txt, iou, *_ = self.model.run_mask_decoder(self.point, d)
            logits.append(seg[0].float())
            texts.append(F.normalize(txt[0].float(), dim=-1))
            ious.append(iou[0].float().squeeze(-1))
        return torch.cat(logits), torch.cat(texts), torch.cat(ious)
