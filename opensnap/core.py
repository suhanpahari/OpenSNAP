from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_cluster import fps
from torch_scatter import scatter_mean

SNAP_TEMPLATES = ("segment {}",)
SCENE_TEMPLATES = ("a {} in a scene",)


@dataclass
class Config:
    num_prompts: int = 400
    iou_thresh: float = 0.3
    stability_thresh: float = 0.5
    nms_thresh: float = 0.5
    min_points: int = 30
    tau: float = 0.01
    alpha: float = 0.7
    beta: float = 0.3
    relay_tau: float = 0.05
    relay_weight: float = 0.8


@dataclass
class Proposals:
    logits: torch.Tensor
    tokens: torch.Tensor
    scores: torch.Tensor
    prompts: torch.Tensor


class OpenSNAP:
    """Open-vocabulary SNAP: promptable masks + mask-anchored CLIP evidence."""

    def __init__(self, snap, clip_snap, openscene=None, clip_dense=None, cfg=Config(), anchors=None):
        self.snap, self.clip_snap = snap, clip_snap
        self.openscene, self.clip_dense = openscene, clip_dense
        self.cfg = cfg
        self.anchor_emb = clip_snap(anchors, SNAP_TEMPLATES) if anchors else None

    def encode(self, coord, color=None, normal=None, strength=None):
        vcoord, inverse = self.snap.encode(coord, color, normal, strength)
        dense = None
        if self.openscene is not None:
            dense = F.normalize(scatter_mean(self.openscene(coord), inverse, dim=0, dim_size=len(vcoord)), dim=-1)
        self.scene = dict(vcoord=vcoord, inverse=inverse, dense=dense)
        return self.scene

    @torch.no_grad()
    def propose(self, prompts=None):
        c, vcoord = self.cfg, self.scene["vcoord"]
        if prompts is None:
            ratio = min(1.0, c.num_prompts / len(vcoord))
            prompts = vcoord[fps(vcoord, ratio=ratio, random_start=False)][:, None]
        logits, tokens, scores = self.snap.decode(prompts)
        stability = ((logits > 1).sum(1).float() / (logits > -1).sum(1).clamp(min=1).float())
        size = (logits > 0).sum(1)
        keep = (scores > c.iou_thresh) & (stability > c.stability_thresh) & (size >= c.min_points)
        idx = keep.nonzero().squeeze(1)
        idx = idx[mask_nms((logits[idx] > 0), (scores * stability)[idx], c.nms_thresh)]
        return Proposals(logits[idx], tokens[idx], scores[idx], prompts[idx])

    @torch.no_grad()
    def click(self, point_indices):
        """One object from one or more clicked (original) point indices."""
        idx = self.scene["inverse"][torch.as_tensor(point_indices, device=self.scene["inverse"].device)]
        prompts = self.scene["vcoord"][idx][None]
        logits, tokens, scores = self.snap.decode(prompts)
        return Proposals(logits, tokens, scores, prompts)

    def snap_probs(self, tokens, labels):
        c = self.cfg
        text = self.clip_snap(labels, SNAP_TEMPLATES)
        direct = (tokens @ text.T / c.tau).softmax(-1)
        if self.anchor_emb is None or c.relay_weight == 0:
            return direct
        post = (tokens @ self.anchor_emb.T / c.relay_tau).softmax(-1)
        relay = (post @ (self.anchor_emb @ text.T) / c.tau).softmax(-1)
        return (direct ** (1 - c.relay_weight)) * (relay ** c.relay_weight)

    def dense_probs(self, labels):
        text = self.clip_dense(labels, SCENE_TEMPLATES)
        return (self.scene["dense"] @ text.T / self.cfg.tau).softmax(-1)

    def mask_probs(self, props, labels, use_dense=True, use_snap=True):
        c = self.cfg
        terms, weights = [], []
        if use_dense and self.scene["dense"] is not None:
            w = (props.logits > 0).float() * props.logits.sigmoid()
            pooled = F.normalize(w @ self.scene["dense"], dim=-1)
            terms.append((pooled @ self.clip_dense(labels, SCENE_TEMPLATES).T / c.tau).softmax(-1))
            weights.append(c.alpha)
        if use_snap:
            terms.append(self.snap_probs(props.tokens, labels))
            weights.append(1 - c.alpha if terms[:-1] else 1.0)
        logp = sum(wt * t.clamp(min=1e-8).log() for wt, t in zip(weights, terms)) / sum(weights)
        return logp.softmax(-1)

    def semantic(self, props, labels, use_dense=True, use_snap=True, beta=None):
        """Per original point class probabilities (N, C)."""
        beta = self.cfg.beta if beta is None else beta
        dense = self.dense_probs(labels) if self.scene["dense"] is not None else None
        if len(props.scores) == 0 or not (use_dense or use_snap):
            out = dense
        else:
            pm = self.mask_probs(props, labels, use_dense, use_snap)
            w = (props.logits > 0).float() * props.logits.sigmoid() * props.scores[:, None]
            cover = w.sum(0)
            vote = (w.T @ pm) / cover.clamp(min=1e-6)[:, None]
            if dense is None:
                out = vote
            else:
                covered = (cover > 0)[:, None]
                out = torch.where(covered, beta * dense + (1 - beta) * vote, dense)
        return out[self.scene["inverse"]]

    def query(self, props, text, labels_bg=("wall", "floor", "ceiling", "object")):
        """Rank proposals for a free-form text query against a background vocabulary."""
        pm = self.mask_probs(props, [text, *labels_bg], use_dense=self.scene["dense"] is not None)
        rel = pm[:, 0] * props.scores
        order = rel.argsort(descending=True)
        return order, rel[order]


def mask_nms(masks, scores, thresh):
    order = scores.argsort(descending=True)
    m = masks[order].float()
    inter = m @ m.T
    area = m.sum(1)
    iou = inter / (area[:, None] + area[None] - inter).clamp(min=1)
    keep = torch.ones(len(order), dtype=torch.bool, device=masks.device)
    for i in range(len(order)):
        if keep[i]:
            keep[i + 1:] &= iou[i, i + 1:] <= thresh
    return order[keep]
