import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ObjectTable:
    """Materialised object relation of one scene: one row per object (voxel-level point sets)."""
    points: list                      # per object: sorted int32 voxel indices
    bbox_min: torch.Tensor            # (M, 3)
    bbox_max: torch.Tensor            # (M, 3)
    centroid: torch.Tensor            # (M, 3)
    conf: torch.Tensor                # (M,) SNAP predicted IoU
    token: torch.Tensor               # (M, 512) SNAP CLIP-space token
    pooled: torch.Tensor              # (M, D) mask-pooled dense CLIP feature (or empty)
    provenance: list = field(default_factory=list)
    build_sec: float = 0.0
    cracked: dict = field(default_factory=dict)  # text -> oid created by query-time refinement (or None)

    def __len__(self):
        return len(self.points)

    def append(self, pts, vcoord, conf, token, pooled, source):
        p = vcoord[pts]
        self.points.append(pts.cpu().numpy().astype(np.int32))
        cat = lambda a, b: torch.cat([a, b[None]], 0)
        self.bbox_min, self.bbox_max = cat(self.bbox_min, p.min(0).values), cat(self.bbox_max, p.max(0).values)
        self.centroid = cat(self.centroid, p.mean(0))
        self.conf = cat(self.conf, conf.reshape(()))
        self.token = cat(self.token, token)
        if self.pooled.numel():
            self.pooled = cat(self.pooled, pooled)
        self.provenance.append(source)
        return len(self) - 1

    def nbytes(self):
        pts = sum(p.nbytes for p in self.points)
        emb = sum(t.numel() * 2 for t in (self.token, self.pooled))  # stored as fp16
        geo = sum(t.numel() * 4 for t in (self.bbox_min, self.bbox_max, self.centroid, self.conf))
        return dict(points=pts, embeddings=emb, geometry=geo, total=pts + emb + geo)


def materialize(opensnap, props=None):
    """Build the object relation from automatic SNAP proposals."""
    t0 = time.time()
    props = props if props is not None else opensnap.propose()
    vc, dense = opensnap.scene["vcoord"], opensnap.scene["dense"]
    masks = props.logits > 0
    pts = [torch.nonzero(m)[:, 0] for m in masks]
    if dense is not None:
        w = masks.float() * props.logits.sigmoid()
        pooled = torch.nn.functional.normalize(w @ dense, dim=-1)
    else:
        pooled = torch.zeros(len(pts), 0, device=vc.device)
    stack = lambda f: torch.stack([f(vc[p]) for p in pts]) if pts else torch.zeros(0, 3, device=vc.device)
    table = ObjectTable(points=[p.cpu().numpy().astype(np.int32) for p in pts],
                        bbox_min=stack(lambda x: x.min(0).values), bbox_max=stack(lambda x: x.max(0).values),
                        centroid=stack(lambda x: x.mean(0)), conf=props.scores.clone(), token=props.tokens.clone(),
                        pooled=pooled, provenance=["auto"] * len(pts))
    torch.cuda.synchronize()
    table.build_sec = time.time() - t0
    return table
