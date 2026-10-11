"""Scaling the object relation beyond one building: the 16 HouseLayout3D building catalogs replicated into a campus.

Plans for  SELECT WHERE IS(a) AND NEAR(IS(b), d)  with answer threshold tau:
  padded    current executor: per-room padded block join over all rows, (R, P, P)
  pushdown  IS filters first (keep p_a >= tau, p_b >= tau; exact for threshold answers since p_a * max_j rel * p_b >= tau
            needs both factors >= tau), then a uniform-grid spatial join on the surviving candidates only.
Also reports IS latency, catalog bytes (fp16 embeddings + boxes) and GPU memory.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

CAT = os.path.join(paths.DOWNLOADS, "catalogs", "hl3d")


def campus(copies):
    cats = [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(CAT, "*.pt")))]
    parts, room0, x0 = [], 0, 0.0
    for k in range(copies):
        for c in cats:
            span = float((c["bmax"][:, 0].max() - c["bmin"][:, 0].min())) + 10.0
            shift = torch.tensor([x0 - float(c["bmin"][:, 0].min()), 1000.0 * (k % 64), 0.0])
            parts.append(dict(bmin=c["bmin"] + shift, bmax=c["bmax"] + shift, cen=c["cen"] + shift, token=c["token"], pooled=c["pooled"],
                              conf=c["conf"], room=c["room"] + room0))
            room0 += int(c["room"].max()) + 1
            x0 += span
        x0 = 0.0 if (k + 1) % 64 == 0 else x0
    cat = lambda key: torch.cat([p[key] for p in parts])
    return {k: cat(k) for k in ("bmin", "bmax", "cen", "token", "pooled", "conf", "room")}


def cells(lo, hi, cell):
    """(owner, hashed cell key) for every grid cell a box overlaps."""
    a, b = torch.floor(lo / cell).long(), torch.floor(hi / cell).long()
    span = b - a + 1
    n = span.prod(1)
    own = torch.repeat_interleave(torch.arange(len(lo), device=lo.device), n)
    off = torch.arange(int(n.sum()), device=lo.device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
    sx, sy = span[own, 0], span[own, 1]
    cx, cy, cz = a[own, 0] + off % sx, a[own, 1] + (off // sx) % sy, a[own, 2] + off // (sx * sy)
    return own, (cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)


def grid_join(lo_s, hi_s, lo_q, hi_q, pq, d, cell):
    """max_j [gap(i, j) <= d] * pq[j]: references hashed into every cell of their box dilated by d, subjects probe every
    cell of their own box; if gap <= d some subject point lies in a dilated reference cell, so the join is exact."""
    out = torch.zeros(len(lo_s), device=lo_s.device)
    if len(lo_s) == 0 or len(lo_q) == 0:
        return out
    rid, key = cells(lo_q - d, hi_q + d, cell)
    ks, order = torch.sort(key)
    rs = rid[order]
    sown, qk = cells(lo_s, hi_s, cell)
    lo_i = torch.searchsorted(ks, qk)
    hi_i = torch.searchsorted(ks, qk, right=True)
    cnt = hi_i - lo_i
    if int(cnt.sum()) == 0:
        return out
    probe = torch.repeat_interleave(torch.arange(len(qk), device=lo_s.device), cnt)
    pos = torch.arange(int(cnt.sum()), device=lo_s.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt) + lo_i[probe]
    sid = sown[probe]
    j = rs[pos]
    gap = torch.maximum(lo_s[sid] - hi_q[j], lo_q[j] - hi_s[sid]).clamp(min=0).norm(dim=-1)
    val = (gap <= d).float() * pq[j]
    return out.scatter_reduce(0, sid, val, reduce="amax", include_self=True)


def candidate_join(lo_s, hi_s, lo_q, hi_q, pq, d, cell, budget=2e8):
    """Cost-based: dense |S| x |Q| join (chunked) when small, exact grid hash join otherwise."""
    if len(lo_s) * len(lo_q) > budget:
        return grid_join(lo_s, hi_s, lo_q, hi_q, pq, d, cell)
    out = torch.zeros(len(lo_s), device=lo_s.device)
    step = max(1, int(budget // max(len(lo_q), 1) // 8))
    for i in range(0, len(lo_s), step):
        a, b = lo_s[i:i + step, None], hi_s[i:i + step, None]
        gap = torch.maximum(a - hi_q[None], lo_q[None] - b).clamp(min=0).norm(dim=-1)
        out[i:i + step] = ((gap <= d).float() * pq[None]).max(1).values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", default="1,4,16,64")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--d", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/compare/scale.json")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.query.catalog import ObjectTable
    from opensnap.query.engine import Engine
    from opensnap.text import TextEncoder
    clip = TextEncoder()
    m = OpenSNAP(None, clip, None, TextEncoder("ViT-L/14@336px"), Config(), anchors=snap_train_vocab())
    words = ["chair", "table", "sofa", "bed", "cabinet", "picture", "lamp", "sink", "toilet", "shelf"]
    pairs = [(x, y) for x in words for y in words if x != y][:30]
    out = []
    for k in [int(c) for c in a.copies.split(",")]:
        c = campus(k)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t = ObjectTable(points=[None] * len(c["conf"]), bbox_min=c["bmin"].cuda(), bbox_max=c["bmax"].cuda(), centroid=c["cen"].cuda(),
                        conf=c["conf"].cuda(), token=c["token"].cuda().float(), pooled=c["pooled"].cuda().float(), room=c["room"].cuda())
        M, R = len(t.conf), int(t.room.max()) + 1
        eng = Engine(m, t, refine=False)
        nbytes = M * (512 + 768) * 2 + M * 13 * 4
        eng.is_prob(words[0])
        torch.cuda.synchronize(); t0 = time.time()
        P = {w: eng.is_prob(w) for w in words}
        torch.cuda.synchronize(); is_ms = 1000 * (time.time() - t0) / len(words)
        rec = dict(copies=k, rooms=R, objects=M, catalog_MB=nbytes / 1e6, IS_ms=is_ms)
        try:
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(a.repeats):
                for x, y in pairs:
                    pad = eng.padded()
                    valid = pad >= 0
                    ids = pad.clamp(min=0)
                    rel = eng.relation("NEAR", a.d, ids) * (valid[:, :, None] & valid[:, None, :])
                    sub = (rel * P[y][ids][:, None, :]).max(2).values
            torch.cuda.synchronize()
            rec["padded_ms"] = 1000 * (time.time() - t0) / (a.repeats * len(pairs))
        except torch.OutOfMemoryError:
            rec["padded_ms"] = None
            torch.cuda.empty_cache()
        agree, cands = [], []
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(a.repeats):
            for x, y in pairs:
                s = torch.nonzero(P[x] >= a.tau)[:, 0]
                q = torch.nonzero(P[y] >= a.tau)[:, 0]
                sc = candidate_join(t.bbox_min[s], t.bbox_max[s], t.bbox_min[q], t.bbox_max[q], P[y][q], a.d, cell=4.0)
                cands.append(len(s) + len(q))
        torch.cuda.synchronize()
        rec["pushdown_ms"] = 1000 * (time.time() - t0) / (a.repeats * len(pairs))
        rec["candidates"] = float(np.mean(cands))
        if rec.get("padded_ms") is not None:
            for x, y in pairs[:10]:
                res, _ = eng.run(f"SELECT WHERE IS('{x}') AND NEAR(IS('{y}'), {a.d})")
                ans = torch.tensor([i for i, v in res if v >= a.tau], dtype=torch.long, device="cuda")
                s = torch.nonzero(P[x] >= a.tau)[:, 0]
                q = torch.nonzero(P[y] >= a.tau)[:, 0]
                sc = torch.zeros(M, device="cuda")
                sc[s] = P[x][s] * candidate_join(t.bbox_min[s], t.bbox_max[s], t.bbox_min[q], t.bbox_max[q], P[y][q], a.d, cell=4.0)
                rec.setdefault("pushdown_answers_vs_room_join", []).append((int(len(ans)), int((sc[ans] >= a.tau).sum()),
                                                                            int((sc >= a.tau).sum())))
        rec["peak_GPU_MB"] = torch.cuda.max_memory_allocated() / 1e6
        out.append(rec)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != "pushdown_answers_vs_room_join"}), flush=True)
        del t, eng, P, c
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
