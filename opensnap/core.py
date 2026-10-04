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
            dense = self.openscene.voxel_features(coord, inverse, len(vcoord))
        self.scene = dict(vcoord=vcoord, inverse=inverse, dense=dense)
        return self.scene

    @torch.no_grad()
    def propose_c2f(self, v0=0.8, levels=3, cover=0.5):
        """Coarse-to-fine auto prompting: grid-centre prompts, then finer grids only in still-uncovered regions."""
        c, vc = self.cfg, self.scene["vcoord"]
        covered = torch.zeros(len(vc), dtype=torch.bool, device=vc.device)
        L, T, S, P = [], [], [], []
        for k in range(levels):
            v = v0 / 2 ** k
            cand = torch.nonzero(~covered)[:, 0]
            if len(cand) == 0:
                break
            key = torch.floor(vc[cand] / v).long()
            key = key - key.min(0).values
            h = (key[:, 0] * 1000003 + key[:, 1]) * 1000003 + key[:, 2]
            _, inv = torch.unique(h, return_inverse=True)
            centre = torch.zeros(int(inv.max()) + 1, 3, device=vc.device).index_add_(0, inv, vc[cand])
            centre /= torch.bincount(inv).float()[:, None]
            near = torch.cdist(centre, vc[cand]).argmin(1)
            prompts = vc[cand[near]][:, None]
            logits, tokens, scores = self.snap.decode(prompts)
            ok = scores > c.iou_thresh
            L.append(logits[ok]); T.append(tokens[ok]); S.append(scores[ok]); P.append(prompts[ok])
            covered |= ((logits[ok] > 0).float().sum(0) > 0) if ok.any() else covered
        logits, tokens, scores, prompts = torch.cat(L), torch.cat(T), torch.cat(S), torch.cat(P)
        stability = (logits > 1).sum(1).float() / (logits > -1).sum(1).clamp(min=1).float()
        size = (logits > 0).sum(1)
        keep = (stability > c.stability_thresh) & (size >= c.min_points)
        idx = keep.nonzero().squeeze(1)
        idx = idx[mask_nms(logits[idx] > 0, (scores * stability)[idx], c.nms_thresh)]
        return Proposals(logits[idx], tokens[idx], scores[idx], prompts[idx])

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

    @torch.no_grad()
    def relevance(self, text, bg=("wall", "floor", "ceiling", "object")):
        return self.dense_probs([text, *[b for b in bg if b != text]])[:, 0]

    @torch.no_grad()
    def text_click(self, text, region=None, max_clicks=3, tau=0.5, radius=1.0, knn_r=0.1):
        """Language as the click oracle: clicks go where text relevance is high but the SNAP mask is missing."""
        vc = self.scene["vcoord"]
        rel = self.relevance(text)
        if region is not None:
            rel = rel * region
        cand = rel.topk(min(2000, len(rel))).indices
        d = torch.cdist(vc[cand], vc[cand])
        density = ((d < knn_r).float() * rel[cand][None]).sum(1)
        seed = cand[density.argmax()]
        target = (rel > tau * rel[seed]) & ((vc - vc[seed]).norm(dim=1) < radius)
        clicks, best = [seed], None
        for _ in range(max_clicks):
            logits, tokens, scores = self.snap.decode(vc[torch.stack(clicks)][None])
            m = logits[0] > 0
            agree = (m & target).sum() / (m | target).sum().clamp(min=1)
            if best is None or agree > best[0]:
                best = (agree, Proposals(logits, tokens, scores, vc[torch.stack(clicks)][None]))
            missing = target & ~m
            if missing.sum() < 0.1 * target.sum():
                break
            clicks.append(torch.where(missing, rel, torch.zeros_like(rel)).argmax())
        return best[1], float(best[0])

    @torch.no_grad()
    def text_segment(self, text, min_agree=0.3, tau=0.5, radius=1.0):
        """Text -> object mask (voxels). SNAP mask via language clicks; relevance region when geometry cannot separate it."""
        props, agree = self.text_click(text, tau=tau, radius=radius)
        if agree >= min_agree:
            return props.logits[0] > 0, "snap", agree
        vc, rel = self.scene["vcoord"], self.relevance(text)
        seed = props.prompts[0, 0]
        m = (rel > tau * rel.max()) & ((vc - seed).norm(dim=1) < radius)
        return m, "dense", agree

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
