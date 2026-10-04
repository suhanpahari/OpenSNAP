"""Building-level OVQL: one persisted catalog over all rooms; global ranking, per-room spatial predicates, AP vs annotations."""
import argparse
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
from eval_queries import S3DIS, SKIP, average_precision, gt_objects, relational_gt  # noqa: E402


def sync():
    torch.cuda.synchronize()
    return time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pointcept/s3dis/Area_5")
    ap.add_argument("--catalog", default="../downloads/catalogs/s3dis_area5")
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/building_s3dis_area5.json")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import concat, load, materialize, save
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    os.makedirs(a.catalog, exist_ok=True)
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", a.split, "*")))
    gts, ingest, raw, nvox = [], [], 0, []
    for f in files:
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        ins = np.load(f"{f}/instance.npy").reshape(-1)
        seg = np.load(f"{f}/segment.npy").reshape(-1)
        raw += coord.nbytes
        np.random.seed(0); torch.manual_seed(0)
        t0 = sync(); m.encode(coord); t = materialize(m); ingest.append(sync() - t0)
        save(t, os.path.join(a.catalog, os.path.basename(f) + ".pt"))
        vc = m.scene["vcoord"].cpu().numpy(); inv = m.scene["inverse"].cpu().numpy()
        nvox.append(len(vc))
        gts.append([(c, mk, (vc[mk].min(0), vc[mk].max(0))) for c, mk in gt_objects(ins, seg, inv, len(vc), S3DIS, SKIP["s3dis"])])
        print(f"ingested {os.path.basename(f)} {ingest[-1]:.1f}s objs={len(t)}", flush=True)
    disk = sum(os.path.getsize(os.path.join(a.catalog, x)) for x in os.listdir(a.catalog))
    t0 = sync()
    tables = [load(os.path.join(a.catalog, os.path.basename(f) + ".pt")) for f in files]
    building = concat(tables)
    load_s = sync() - t0
    offsets = np.cumsum([0] + [len(t) for t in tables])
    eng = Engine(m, building, refine=False)
    classes = sorted({c for g in gts for c, _, _ in g})
    queries = [("IS", c, None) for c in classes]
    pairs = [(x, y) for x in classes for y in classes if x != y and any(relational_gt(g, x, y, a.near) for g in gts)]
    queries += [("NEAR", x, y) for x, y in pairs]
    res = {}
    for kind, x, y in queries:
        q = f"SELECT WHERE IS('{x}')" if kind == "IS" else f"SELECT WHERE IS('{x}') AND NEAR(IS('{y}'), {a.near})"
        lat = []
        for _ in range(a.repeats):
            s = sync(); out, _ = eng.run(q); lat.append(sync() - s)
        items = []
        for r, g in enumerate(gts):
            sel = [(i - offsets[r], sc) for i, sc in out if offsets[r] <= i < offsets[r + 1]]
            gt_idx = [k for k, o in enumerate(g) if o[0] == x] if kind == "IS" else relational_gt(g, x, y, a.near)
            masks = [np.isin(np.arange(nvox[r]), tables[r].points[i]) for i, _ in sel]
            ious = [[(mk & g[k][1]).sum() / max((mk | g[k][1]).sum(), 1) for k in gt_idx] for mk in masks]
            items.append(([sc for _, sc in sel], ious, len(gt_idx)))
        res[f"{kind}:{x}" + (f":{y}" if y else "")] = dict(ms=1000 * float(np.median(lat)),
                                                         AP=float(np.mean([average_precision(items, t) for t in np.arange(0.5, 0.96, 0.05)])),
                                                         AP50=average_precision(items, 0.5))
    summ = dict(rooms=len(files), objects=len(building), ingest_s_total=float(np.sum(ingest)), raw_xyz_MB=raw / 1e6,
                catalog_MB=disk / 1e6, catalog_load_s=load_s)
    for kind in ("IS", "NEAR"):
        v = [r for k, r in res.items() if k.startswith(kind)]
        summ[kind] = dict(queries=len(v), AP=float(np.nanmean([r["AP"] for r in v])), AP50=float(np.nanmean([r["AP50"] for r in v])),
                          ms=float(np.mean([r["ms"] for r in v])))
    json.dump(dict(summary=summ, per_query=res), open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
