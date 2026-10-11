"""Spatial joins over whole multi-floor buildings (the 16 HouseLayout3D / Matterport3D buildings).

Each Matterport region (room) is ingested on its own; all object rows are kept in the shared house frame. A spatial
predicate is then evaluated under different partitionings of the join and compared with the unpartitioned (global)
join over the same relation:
  room        join only inside a region (the default of the building catalog)
  room+halo   region plus every object whose box is within h of the region box (exact for NEAR(d) when h >= d)
  room+door   region plus regions connected to it by an annotated HouseLayout3D door or opening
  level       join inside a floor level (levels found from region floor heights, checked against HouseLayout3D planes)
  grid        layout-free square tiles of side T with halo h
Vertical predicates (ABOVE/BELOW/ON) evaluated globally can pick witnesses on another floor; those answers are counted.
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

HL3D = ["17DRP5sb8fy", "1LXtFkjw3qL", "2t7WUuJeko7", "5LpN3gDmAk7", "HxpKQynjfin", "JeFG25nYj2p", "JmbYfDe2QKZ", "S9hNv5qa7GM",
        "TbHJrupSAjP", "WYY7iVyf5p8", "YFuZgdQ5vWj", "e9zR4mvMWw7", "i5noydFURQK", "jtcxE69GiFV", "p5wJjkQkbXX", "r47D5H71a5s"]
VOCAB = ["chair", "table", "sofa", "bed", "cabinet", "picture", "lamp", "cushion", "plant", "television", "sink", "toilet",
         "door", "window", "shelf", "counter", "mirror", "towel", "curtain", "stool"]
MP = os.path.join(paths.DOWNLOADS, "data", "matterport_3d_160")
HL = os.path.join(paths.DOWNLOADS, "hl3d", "data")
CAT = os.path.join(paths.DOWNLOADS, "catalogs", "hl3d")


def ingest(scan, m, materialize):
    files = sorted(glob.glob(os.path.join(MP, "*", f"{scan}_region*.pth")), key=lambda f: int(f.split("region")[-1][:-4]))
    rows, rooms = [], []
    for r, f in enumerate(files):
        coord = torch.load(f, weights_only=False)[0].astype(np.float32)
        np.random.seed(0); torch.manual_seed(0)
        t0 = time.time()
        m.encode(coord)
        t = materialize(m)
        vc, inv = m.scene["vcoord"], m.scene["inverse"]
        off = (torch.from_numpy(coord).to(vc.device) - vc[inv]).mean(0)
        pooled = t.pooled if t.pooled.numel() else torch.zeros(len(t), 768, device=t.conf.device)
        rows.append(dict(bmin=(t.bbox_min + off).cpu(), bmax=(t.bbox_max + off).cpu(), cen=(t.centroid + off).cpu(),
                         conf=t.conf.cpu(), token=t.token.half().cpu(), pooled=pooled.half().cpu(), npts=[len(p) for p in t.points]))
        rooms.append(dict(file=os.path.basename(f), lo=coord.min(0), hi=coord.max(0), floor=float(np.percentile(coord[:, 2], 2)),
                          sec=time.time() - t0, n=len(t), dense_collapsed=m.scene["dense_collapsed"]))
        print(f"  {scan} region{r} objs={len(t)} {rooms[-1]['sec']:.1f}s", flush=True)
    cat = lambda k: torch.cat([x[k] for x in rows])
    room = torch.cat([torch.full((len(x["conf"]),), i) for i, x in enumerate(rows)])
    return dict(bmin=cat("bmin"), bmax=cat("bmax"), cen=cat("cen"), conf=cat("conf"), token=cat("token"), pooled=cat("pooled"),
                room=room, rooms=rooms)


def levels(rooms, gap=1.2):
    """Training-free floor levels: cluster region floor heights (gaps above `gap` metres start a new level)."""
    z = np.array([r["floor"] for r in rooms])
    order = np.argsort(z)
    lev = np.zeros(len(z), int)
    k = 0
    for a, b in zip(order[:-1], order[1:]):
        if z[b] - z[a] > gap:
            k += 1
        lev[b] = k
    return lev


def hl3d_planes(scan):
    from plyfile import PlyData
    hs = []
    for f in glob.glob(os.path.join(HL, "structures", "layouts_split_by_entity", scan, "*.ply")):
        p = PlyData.read(f)
        z = np.asarray(p["vertex"]["z"])
        if np.ptp(z) < 0.05:
            xy = np.stack([p["vertex"]["x"], p["vertex"]["y"]], 1)
            hs.append((float(z.mean()), float(np.prod(np.ptp(xy, 0)))))
    return hs


def door_edges(scan, rooms, tol=0.35):
    """Regions joined by a HouseLayout3D door/opening: the door's centre lies within `tol` of both region boxes."""
    d = json.load(open(os.path.join(HL, "doors", f"{scan}.json")))["doors"]
    cents = [np.asarray(x["vertices"]).mean(0) for x in d]
    edges = set()
    for c in cents:
        near = [i for i, r in enumerate(rooms) if np.all(c >= r["lo"] - tol) and np.all(c <= r["hi"] + tol)]
        edges |= {(i, j) for i in near for j in near if i != j}
    return edges, len(cents)


def blocks_for(strategy, cat, lev_of_room, edges, h=0.0, tile=4.0):
    """List of (subject ids, reference ids) blocks; every object is a subject in exactly one block."""
    room = cat["room"].numpy()
    lo, hi, cen = cat["bmin"].numpy(), cat["bmax"].numpy(), cat["cen"].numpy()
    R = int(room.max()) + 1
    if strategy == "global":
        a = np.arange(len(room))
        return [(a, a)]
    if strategy == "level":
        lv = lev_of_room[room]
        return [(np.nonzero(lv == k)[0],) * 2 for k in np.unique(lv)]
    if strategy == "room":
        return [(np.nonzero(room == r)[0],) * 2 for r in range(R)]
    if strategy == "room+door":
        out = []
        for r in range(R):
            nb = {r} | {j for i, j in edges if i == r}
            out.append((np.nonzero(room == r)[0], np.nonzero(np.isin(room, list(nb)))[0]))
        return out
    if strategy in ("room+halo", "level+halo"):
        lv = lev_of_room[room]
        out = []
        for r in range(R):
            s = np.nonzero(room == r)[0]
            if len(s) == 0:
                continue
            blo, bhi = lo[s].min(0) - h, hi[s].max(0) + h
            ok = np.all(hi >= blo, 1) & np.all(lo <= bhi, 1)
            if strategy == "level+halo":
                ok &= lv == lev_of_room[r]
            out.append((s, np.nonzero(ok)[0]))
        return out
    if strategy == "grid":
        key = np.floor(cen[:, :2] / tile).astype(int)
        out = []
        for k in np.unique(key, axis=0):
            s = np.nonzero((key == k).all(1))[0]
            blo, bhi = lo[s, :2].min(0) - h, hi[s, :2].max(0) + h
            q = np.nonzero((hi[:, 0] >= blo[0]) & (lo[:, 0] <= bhi[0]) & (hi[:, 1] >= blo[1]) & (lo[:, 1] <= bhi[1]))[0]
            out.append((s, q))
        return out
    raise ValueError(strategy)


class Join:
    """Batched block join: score_i = max_{j in Q(i), j != i} rel(op, i, j) * p_ref(j)."""

    def __init__(self, cat, blocks, device="cuda"):
        self.lo, self.hi, self.c = (cat[k].to(device).float() for k in ("bmin", "bmax", "cen"))
        R = len(blocks)
        Ps, Pq = max(len(s) for s, _ in blocks), max(len(q) for _, q in blocks)
        S = torch.full((R, Ps), -1, dtype=torch.long)
        Q = torch.full((R, Pq), -1, dtype=torch.long)
        for k, (s, q) in enumerate(blocks):
            S[k, :len(s)] = torch.from_numpy(s)
            Q[k, :len(q)] = torch.from_numpy(q)
        self.S, self.Q = S.to(device), Q.to(device)
        self.cells = int(R * Ps * Pq)

    def __call__(self, op, dist, p_ref):
        S, Q = self.S.clamp(min=0), self.Q.clamp(min=0)
        li, hi_, ci = self.lo[S][:, :, None], self.hi[S][:, :, None], self.c[S][:, :, None]
        lj, hj = self.lo[Q][:, None], self.hi[Q][:, None]
        if op == "NEAR":
            rel = torch.maximum(li - hj, lj - hi_).clamp(min=0).norm(dim=-1) <= dist
        else:
            xy = ((torch.minimum(hi_[..., :2], hj[..., :2]) - torch.maximum(li[..., :2], lj[..., :2])) > 0).all(-1)
            if op == "ABOVE":
                rel = xy & (ci[..., 2] > hj[..., 2])
            elif op == "BELOW":
                rel = xy & (ci[..., 2] < lj[..., 2])
            else:
                rel = xy & ((li[..., 2] - hj[..., 2]).abs() < max(dist, 0.1))
        rel &= (self.S >= 0)[:, :, None] & (self.Q >= 0)[:, None, :] & (self.S[:, :, None] != self.Q[:, None, :])
        sub = (rel.float() * p_ref[Q][:, None, :]).max(2).values
        out = torch.zeros(len(self.lo), device=p_ref.device)
        ok = self.S >= 0
        out[self.S[ok]] = sub[ok]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", default=HL3D)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/compare/hl3d_partition.json")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import ObjectTable, materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    os.makedirs(CAT, exist_ok=True)
    report = dict(per_building={}, args=vars(a))
    for scan in a.scans:
        f = os.path.join(CAT, f"{scan}.pt")
        if not os.path.exists(f):
            torch.save(ingest(scan, m, materialize), f)
        cat = torch.load(f, weights_only=False)
        rooms = cat["rooms"]
        lev = levels(rooms)
        planes = hl3d_planes(scan)
        lev_h = [np.median([r["floor"] for r, l in zip(rooms, lev) if l == k]) for k in np.unique(lev)]
        lev_ok = [min([abs(h - z) for z, _ in planes] or [9]) < 0.15 for h in lev_h]
        edges, ndoors = door_edges(scan, rooms)
        t = ObjectTable(points=[None] * len(cat["conf"]), bbox_min=cat["bmin"].cuda(),
                        bbox_max=cat["bmax"].cuda(), centroid=cat["cen"].cuda(), conf=cat["conf"].cuda(),
                        token=cat["token"].float().cuda(), pooled=cat["pooled"].float().cuda())
        eng = Engine(m, t, refine=False)
        p = {w: eng.is_prob(w) for w in VOCAB}
        strategies = {"global": {}, "room": {}, "room+door": {}, "level": {}, "room+halo h=0.5": dict(h=0.5),
                      "room+halo h=1.0": dict(h=1.0), "room+halo h=2.0": dict(h=2.0), "level+halo h=1.0": dict(h=1.0), "level+halo h=2.0": dict(h=2.0),
                      "grid T=4 h=1.0": dict(h=1.0, tile=4.0),
                      "grid T=8 h=2.0": dict(h=2.0, tile=8.0)}
        joins = {s: Join(cat, blocks_for(s.split(" ")[0], cat, lev, edges, **kw)) for s, kw in strategies.items()}
        lev_obj = torch.from_numpy(lev[cat["room"].numpy()]).cuda()
        queries = [(op, d) for op, d in (("NEAR", 0.25), ("NEAR", 0.5), ("NEAR", 1.0), ("NEAR", 2.0), ("ABOVE", 0), ("BELOW", 0), ("ON", 0.1))]
        res = {s: {} for s in strategies}
        for s, J in joins.items():
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(a.repeats):
                for op, d in queries:
                    for y in VOCAB:
                        J(op, d, p[y])
            torch.cuda.synchronize()
            res[s]["ms_per_query"] = 1000 * (time.time() - t0) / (a.repeats * len(queries) * len(VOCAB))
            res[s]["cells"] = J.cells
        PX = torch.stack([p[x] for x in VOCAB])                       # (X, M)
        notself = 1 - torch.eye(len(VOCAB), device=PX.device)            # skip x == y pairs
        for op, d in queries:
            key = f"{op}({d})"
            ref = torch.stack([joins["global"](op, d, p[y]) for y in VOCAB])               # (Y, M)
            G = PX[None] * ref[:, None]                                                      # (Y, X, M)
            A = (G >= a.tau) & notself[:, :, None].bool()
            lvl = None
            if op != "NEAR":
                lvl = PX[None] * torch.stack([joins["level"](op, d, p[y]) for y in VOCAB])[:, None]
            for s, J in joins.items():
                Q = PX[None] * torch.stack([J(op, d, p[y]) for y in VOCAB])[:, None]
                tot, hit = int(A.sum()), int((A & (Q >= a.tau)).sum())
                res[s][key] = dict(answers=tot, recall=hit / tot if tot else float("nan"),
                                   max_abs_diff=float(((G - Q).abs() * notself[:, :, None]).max()))
                if s == "global" and lvl is not None:
                    res[s][key]["cross_level_answers"] = int((A & (lvl < a.tau)).sum())
        report["per_building"][scan] = dict(rooms=len(rooms), objects=len(cat["conf"]), levels=int(lev.max()) + 1,
                                            guard_fired=sum(r.get("dense_collapsed") is not None for r in rooms),
                                            levels_match_hl3d=float(np.mean(lev_ok)), doors=ndoors, door_edges=len(edges) // 2,
                                            ingest_s=float(sum(r["sec"] for r in rooms)), results=res)
        print(scan, len(rooms), "rooms", len(cat["conf"]), "objs", int(lev.max()) + 1, "levels",
              {s: round(res[s]["NEAR(1.0)"]["recall"], 3) for s in res}, flush=True)
    agg = {}
    for s in report["per_building"][a.scans[0]]["results"]:
        agg[s] = {}
        for k in report["per_building"][a.scans[0]]["results"][s]:
            vals = [b["results"][s][k] for b in report["per_building"].values()]
            if isinstance(vals[0], dict):
                tot = sum(v["answers"] for v in vals)
                agg[s][k] = dict(answers=tot, recall=sum(v["recall"] * v["answers"] for v in vals if v["answers"]) / max(tot, 1),
                                 max_abs_diff=max(v["max_abs_diff"] for v in vals))
                if "cross_level_answers" in vals[0]:
                    agg[s][k]["cross_level_answers"] = sum(v["cross_level_answers"] for v in vals)
            else:
                agg[s][k] = float(np.mean(vals))
    report["aggregate"] = agg
    report["buildings"] = dict(n=len(a.scans), rooms=sum(b["rooms"] for b in report["per_building"].values()),
                               objects=sum(b["objects"] for b in report["per_building"].values()),
                               levels=sum(b["levels"] for b in report["per_building"].values()),
                               levels_match_hl3d=float(np.mean([b["levels_match_hl3d"] for b in report["per_building"].values()])))
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(dict(buildings=report["buildings"], aggregate=agg), indent=1))


if __name__ == "__main__":
    main()
