"""Probabilistic execution of OVQL over a materialised ObjectTable, with query-driven refinement."""
import time

import numpy as np

import torch
import torch.nn.functional as F

from ..core import SCENE_TEMPLATES
from .lang import parse

BACKGROUND = ("wall", "floor", "ceiling", "object")


class Engine:
    def __init__(self, opensnap, table, refine=True, refine_below=0.3, background=BACKGROUND, gamma=0.0, dedup=0.5):
        self.m, self.t, self.dedup = opensnap, table, dedup
        self.refine, self.refine_below, self.background, self.gamma = refine, refine_below, background, gamma
        self.stats = dict(refined=0, is_calls=0)

    def is_prob(self, text, idx=None):
        """P(text | object) under an open world calibrated against a background vocabulary."""
        self.stats["is_calls"] += 1
        labels = [text, *[b for b in self.background if b != text]]
        m, t, c = self.m, self.t, self.m.cfg
        sl = slice(None) if idx is None else idx
        terms, weights = [], []
        if t.pooled.numel():
            emb = m.clip_dense(labels, SCENE_TEMPLATES)
            terms.append((t.pooled[sl] @ emb.T / c.tau).softmax(-1))
            weights.append(c.alpha)
        tok = t.token[sl]
        known = tok.norm(dim=-1) > 0
        ps = m.snap_probs(F.normalize(tok, dim=-1), labels)
        if terms:
            ps = torch.where(known[:, None], ps, terms[0])
        terms.append(ps)
        weights.append(1 - c.alpha if len(terms) > 1 else 1.0)
        logp = sum(w * p.clamp(min=1e-8).log() for w, p in zip(weights, terms)) / sum(weights)
        return logp.softmax(-1)[:, 0]

    def refine_is(self, text):
        """No stored object explains `text`: materialise one at query time (language as the click oracle)."""
        m = self.m
        vc, dense = m.scene["vcoord"], m.scene["dense"]
        if m.text_decoder is not None:
            mask, iou = m.text_decoder(m.snap, dense, text)
            return self.add_refined(mask, torch.tensor(iou, device=vc.device), torch.zeros(512, device=vc.device), f"query-decoder:{text}")
        props, agree = m.text_click(text)
        if agree >= 0.3:
            mask, token, source, conf = props.logits[0] > 0, props.tokens[0], "query-snap", props.scores[0] * agree
        else:
            mask, _, _ = m.text_segment(text)
            token, source, conf = torch.zeros_like(props.tokens[0]), "query-dense", torch.tensor(agree, device=props.scores.device)
        return self.add_refined(mask, conf, token, f"{source}:{text}")

    def add_refined(self, mask, conf, token, source):
        m = self.m
        vc, dense = m.scene["vcoord"], m.scene["dense"]
        if mask.sum() == 0:
            return None
        if self.dedup and len(self.t):
            ids = torch.nonzero(mask)[:, 0].cpu().numpy()
            inter = torch.tensor([len(np.intersect1d(p, ids, assume_unique=True)) for p in self.t.points], dtype=torch.float)
            sizes = torch.tensor([len(p) for p in self.t.points], dtype=torch.float)
            iou = inter / (sizes + len(ids) - inter).clamp(min=1)
            if iou.max() > self.dedup:
                self.stats["dedup"] = self.stats.get("dedup", 0) + 1
                return int(iou.argmax())
        pooled = F.normalize(dense[mask].mean(0), dim=-1) if dense is not None else None
        oid = self.t.append(torch.nonzero(mask)[:, 0], vc, conf, token, pooled, source)
        self.stats["refined"] += 1
        return oid

    def plan(self, node):
        """Phase 1: refine the relation for IS predicates no stored object explains, before any scoring."""
        if node.op == "IS":
            text = node.args[0]
            if text in self.t.cracked:
                return
            p = self.is_prob(text)
            if len(p) == 0 or p.max() < self.refine_below:
                self.t.cracked[text] = self.refine_is(text)
            return
        for a in node.args:
            if hasattr(a, "op"):
                self.plan(a)

    def eval(self, node):
        op, a = node.op, node.args
        if op == "IS":
            p = self.is_prob(a[0])
            return p * self.t.conf.clamp(min=1e-3) ** self.gamma if self.gamma else p
        if op == "AND":
            return self.eval(a[0]) * self.eval(a[1])
        if op == "OR":
            x, y = self.eval(a[0]), self.eval(a[1])
            return 1 - (1 - x) * (1 - y)
        if op == "NOT":
            return 1 - self.eval(a[0])
        if op == "CONTAINS":
            q = torch.tensor(a, device=self.t.centroid.device)
            vi = int((self.m.scene["vcoord"] - q).norm(dim=1).argmin())
            hit = torch.tensor([float(vi in set(p.tolist())) for p in self.t.points], device=q.device)
            return hit * self.t.conf
        if op in ("NEAR", "ABOVE", "BELOW", "ON"):
            pa = self.eval(a[0])
            if self.t.room is None:
                rel = self.relation(op, a[1])
                rel.fill_diagonal_(0)
                return (rel * pa[None]).max(1).values
            pad = self.padded()                                   # (R, P) row ids, -1 = padding
            valid = pad >= 0
            ids = pad.clamp(min=0)
            rel = self.relation(op, a[1], ids) * (valid[:, :, None] & valid[:, None, :])
            rel = rel * (1 - torch.eye(pad.shape[1], device=rel.device))[None]
            sub = (rel * pa[ids][:, None, :]).max(2).values       # (R, P)
            out = torch.zeros_like(pa)
            out[pad[valid]] = sub[valid]
            return out
        raise ValueError(op)

    def padded(self):
        """(R, P) object ids per scene, padded with -1 (cached): spatial joins never cross scenes."""
        if getattr(self, "_pad", None) is None or int((self._pad >= 0).sum()) != len(self.t):
            r = self.t.room
            order = torch.argsort(r)
            counts = torch.bincount(r[order])
            P, R = int(counts.max()), len(counts)
            pos = torch.arange(len(r), device=r.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            pad = torch.full((R, P), -1, dtype=torch.long, device=r.device)
            pad[r[order], pos] = order
            self._pad = pad
        return self._pad

    def relation(self, op, dist, idx=None):
        """Relation between subject i and reference j: (M, M), or batched (R, P, P) for padded scene blocks."""
        lo, hi, c = (self.t.bbox_min, self.t.bbox_max, self.t.centroid) if idx is None else \
            (self.t.bbox_min[idx], self.t.bbox_max[idx], self.t.centroid[idx])
        li, hi_, lj, hj = lo[..., :, None, :], hi[..., :, None, :], lo[..., None, :, :], hi[..., None, :, :]
        gap = torch.maximum(li - hj, lj - hi_).clamp(min=0).norm(dim=-1)
        xy = ((torch.minimum(hi_[..., :2], hj[..., :2]) - torch.maximum(li[..., :2], lj[..., :2])) > 0).all(-1)
        if op == "NEAR":
            return (gap <= dist).float()
        if op == "ABOVE":
            return (xy & (c[..., :, None, 2] > hj[..., 2])).float()
        if op == "BELOW":
            return (xy & (c[..., :, None, 2] < lj[..., 2])).float()
        return (xy & ((li[..., 2] - hj[..., 2]).abs() < max(dist, 0.1))).float()

    def run(self, text):
        t0 = time.time()
        q = parse(text) if isinstance(text, str) else text
        if self.refine and q.where is not None:
            self.plan(q.where)
        score = self.eval(q.where) if q.where is not None else self.t.conf.clone()
        order = score.argsort(descending=True)
        order = order[score[order] > q.threshold]
        if q.limit:
            order = order[: q.limit]
        idx, val = order.cpu().numpy(), score[order].cpu().numpy()
        return list(zip(idx.tolist(), val.tolist())), dict(ms=1000 * (time.time() - t0), **self.stats)
