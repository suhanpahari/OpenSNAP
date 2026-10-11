"""Long-running cracking workload: storage growth, dedup and accuracy drift under many unique concepts.

Every room receives a long random stream of IS queries (LVIS nouns, most absent from the room; the segmenter's training
vocabulary; dataset class names and synonyms). After every checkpoint a fixed probe set (IS(c) for every annotated class
present in the room, no refinement) is scored against the annotations, so any damage the accumulated cracked tuples do to
ordinary queries shows up as AP drift. Policies differ only in what cracking may admit and keep.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
from eval_queries import S3DIS, SKIP, average_precision, gt_objects  # noqa: E402

POLICIES = {
    "no cracking": dict(refine=False),
    "cracking (default)": dict(refine=True),
    "+ verify": dict(refine=True, verify=True, admit=0.3),
    "+ verify + budget 20%": dict(refine=True, verify=True, admit=0.3, budget=0.2),
    "+ verify + budget + conf-weighted rank": dict(refine=True, verify=True, admit=0.3, budget=0.2, gamma=1.0),
}
SYN = {"table": "desk", "chair": "seat", "sofa": "couch", "bookcase": "bookshelf", "board": "whiteboard", "door": "doorway",
       "window": "windowpane", "column": "pillar", "beam": "girder"}


def workload(rng, n, lvis, snap, present):
    pool = [(lvis, 0.5), (snap, 0.25), (sorted(set(S3DIS) - {"clutter"}) + list(SYN.values()), 0.25)]
    out = []
    for _ in range(n):
        src = pool[rng.choice(3, p=[w for _, w in pool])][0]
        out.append(str(src[rng.integers(len(src))]))
    return out


def probe(m, table, objs, gamma):
    from opensnap.query.engine import Engine
    eng = Engine(m, table, refine=False, gamma=gamma)
    nv = len(m.scene["vcoord"])
    items = {}
    for c in sorted({c for c, _ in objs}):
        res, _ = eng.run(f"SELECT WHERE IS('{c}')")
        masks = []
        for i, _ in res:
            mk = np.zeros(nv, bool)
            mk[table.points[i]] = True
            masks.append(mk)
        gt = [g for cc, g in objs if cc == c]
        ious = [[(mk & g).sum() / max((mk | g).sum(), 1) for g in gt] for mk in masks]
        items[c] = ([s for _, s in res], ious, len(gt))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pointcept/s3dis/Area_5")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--checkpoints", default="0,50,100,200,400")
    ap.add_argument("--out", default="results/compare/workload_s3dis.json")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import lvis_vocab, snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    lvis, snap = lvis_vocab(), snap_train_vocab()
    cps = [int(x) for x in a.checkpoints.split(",")]
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", a.split, "*")))[::a.stride]
    log = {p: {c: dict(items={}, rows=[], cracked=[], MB=[], rejected=[], evicted=[], dedup=[], ms=[]) for c in cps} for p in POLICIES}
    for f in files:
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        ins = np.load(f"{f}/instance.npy").reshape(-1)
        seg = np.load(f"{f}/segment.npy").reshape(-1)
        np.random.seed(0); torch.manual_seed(0)
        m.encode(coord)
        base = materialize(m)
        inv = m.scene["inverse"].cpu().numpy()
        objs = gt_objects(ins, seg, inv, len(m.scene["vcoord"]), S3DIS, SKIP["s3dis"])
        stream = workload(np.random.default_rng(abs(hash(os.path.basename(f))) % 2 ** 31), a.queries, lvis, snap, {c for c, _ in objs})
        for pname, pol in POLICIES.items():
            table = copy.deepcopy(base)
            kw = {k: v for k, v in pol.items() if k != "gamma"}
            eng = Engine(m, table, gamma=pol.get("gamma", 0.0), **kw)
            lat, done = [], 0
            for c in cps:
                while done < c:
                    t0 = time.time()
                    eng.run(f"SELECT WHERE IS('{stream[done]}')")
                    torch.cuda.synchronize()
                    lat.append(time.time() - t0)
                    done += 1
                L = log[pname][c]
                for k, it in probe(m, table, objs, pol.get("gamma", 0.0)).items():
                    L["items"].setdefault(k, []).append(it)
                L["rows"].append(len(table)); L["cracked"].append(sum(p != "auto" for p in table.provenance))
                L["MB"].append(table.nbytes()["total"] / 1e6)
                L["rejected"].append(eng.stats_extra["rejected"]); L["evicted"].append(eng.stats_extra["evicted"])
                L["dedup"].append(eng.stats.get("dedup", 0)); L["ms"].append(1000 * float(np.mean(lat)) if lat else 0.0)
        print(os.path.basename(f), {p: log[p][cps[-1]]["cracked"][-1] for p in POLICIES}, flush=True)
    out = {}
    for p in POLICIES:
        out[p] = {}
        for c in cps:
            L = log[p][c]
            ap50 = float(np.nanmean([average_precision(v, 0.5) for v in L["items"].values()]))
            apm = float(np.nanmean([np.nanmean([average_precision(v, t) for t in np.arange(0.5, 0.96, 0.05)]) for v in L["items"].values()]))
            out[p][c] = dict(AP=apm, AP50=ap50, rows=float(np.mean(L["rows"])), cracked=float(np.mean(L["cracked"])),
                             MB=float(np.mean(L["MB"])), rejected=float(np.mean(L["rejected"])), evicted=float(np.mean(L["evicted"])),
                             dedup=float(np.mean(L["dedup"])), ms_per_query=float(np.mean(L["ms"])))
    json.dump(dict(rooms=len(files), queries_per_room=a.queries, results=out), open(a.out, "w"), indent=1)
    for p in out:
        print(p, {c: (round(100 * v["AP50"], 1), round(v["cracked"], 1), round(v["MB"], 2)) for c, v in out[p].items()})


if __name__ == "__main__":
    main()
