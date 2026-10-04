"""Probabilistic execution of OVQL over a materialised ObjectTable, with query-driven refinement."""
import time

import torch
import torch.nn.functional as F

from ..core import SCENE_TEMPLATES
from .lang import parse

BACKGROUND = ("wall", "floor", "ceiling", "object")


class Engine:
    def __init__(self, opensnap, table, refine=True, refine_below=0.3, background=BACKGROUND):
        self.m, self.t = opensnap, table
        self.refine, self.refine_below, self.background = refine, refine_below, background
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
        props, agree = m.text_click(text)
        vc, dense = m.scene["vcoord"], m.scene["dense"]
        if agree >= 0.3:
            mask, token, source = props.logits[0] > 0, props.tokens[0], "query-snap"
        else:
            mask, _, _ = m.text_segment(text)
            token, source = torch.zeros_like(props.tokens[0]), "query-dense"
        if mask.sum() == 0:
            return None
        pooled = F.normalize(dense[mask].mean(0), dim=-1) if dense is not None else None
        oid = self.t.append(torch.nonzero(mask)[:, 0], vc, props.scores[0], token, pooled, f"{source}:{text}")
        self.stats["refined"] += 1
        return oid

    def plan(self, node):
        """Phase 1: refine the relation for IS predicates no stored object explains, before any scoring."""
        if node.op == "IS":
            p = self.is_prob(node.args[0])
            if len(p) == 0 or p.max() < self.refine_below:
                self.refine_is(node.args[0])
            return
        for a in node.args:
            if hasattr(a, "op"):
                self.plan(a)

    def eval(self, node):
        op, a = node.op, node.args
        if op == "IS":
            return self.is_prob(a[0])
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
            rel = self.relation(op, a[1])
            rel.fill_diagonal_(0)
            return (rel * pa[None]).max(1).values
        raise ValueError(op)

    def relation(self, op, dist):
        """(M, M) boolean relation between objects i (subject) and j (reference)."""
        lo, hi, c = self.t.bbox_min, self.t.bbox_max, self.t.centroid
        gap = (torch.maximum(lo[:, None] - hi[None], lo[None] - hi[:, None]).clamp(min=0)).norm(dim=-1)
        xy = ((torch.minimum(hi[:, None, :2], hi[None, :, :2]) - torch.maximum(lo[:, None, :2], lo[None, :, :2])) > 0).all(-1)
        if op == "NEAR":
            return (gap <= dist).float()
        if op == "ABOVE":
            return (xy & (c[:, None, 2] > hi[None, :, 2])).float()
        if op == "BELOW":
            return (xy & (c[:, None, 2] < lo[None, :, 2])).float()
        return (xy & ((lo[:, None, 2] - hi[None, :, 2]).abs() < max(dist, 0.1))).float()

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
        torch.cuda.synchronize()
        return [(int(i), float(score[i])) for i in order], dict(ms=1000 * (time.time() - t0), **self.stats)
