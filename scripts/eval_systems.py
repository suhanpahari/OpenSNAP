"""Systems experiments: ingest scaling, materialise-once vs per-query re-processing, semantic-cracking amortisation."""
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

SYNONYMS = {"board": ["whiteboard", "blackboard"], "bookcase": ["bookshelf", "shelves"], "chair": ["office chair", "seat"],
            "table": ["desk", "dining table"], "door": ["doorway"], "window": ["glass window"], "sofa": ["couch"],
            "column": ["pillar"], "beam": ["ceiling beam"]}
ABSENT = ["piano", "bathtub", "refrigerator", "bicycle", "toilet", "aquarium"]


def sync():
    torch.cuda.synchronize()
    return time.time()


def ap50(eng, table, res, objs, cls, vc):
    gt = [k for k, (c, _) in enumerate(objs) if c == cls]
    if not gt:
        return None
    masks = [np.isin(np.arange(len(vc)), table.points[i]) for i, _ in res]
    ious = np.array([[(m & objs[g][1]).sum() / max((m | objs[g][1]).sum(), 1) for g in gt] for m in masks]) if masks else np.zeros((0, len(gt)))
    return average_precision([([s for _, s in res], ious.tolist(), len(gt))], 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pointcept/s3dis/Area_5")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--reprocess_queries", type=int, default=3)
    ap.add_argument("--out", default="results/systems_s3dis.json")
    ap.add_argument("--text_ckpt", default="")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    if a.text_ckpt:
        from opensnap.text_decoder import TextDecoder
        m.text_decoder = TextDecoder(a.text_ckpt, m.clip_snap)
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", a.split, "*")))[::a.stride]
    rows = []
    for f in files:
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        ins = np.load(f"{f}/instance.npy").reshape(-1)
        seg = np.load(f"{f}/segment.npy").reshape(-1)
        np.random.seed(0); torch.manual_seed(0)
        t0 = sync(); m.encode(coord); t1 = sync()
        table = materialize(m); t2 = sync()
        vc = m.scene["vcoord"].cpu().numpy(); inv = m.scene["inverse"].cpu().numpy()
        objs = gt_objects(ins, seg, inv, len(vc), S3DIS, SKIP["s3dis"])
        present = sorted({c for c, _ in objs})
        row = dict(scene=os.path.basename(f), points=len(coord), voxels=len(vc), objects=len(table),
                   encode_s=t1 - t0, materialize_s=t2 - t1, relation_bytes=table.nbytes()["total"], raw_bytes=int(coord.nbytes))
        q = f"SELECT WHERE IS('{present[0] if present else 'chair'}')"
        lat = []
        for _ in range(5):
            s = sync(); Engine(m, table, refine=False).run(q); lat.append(sync() - s)
        row["query_cached_ms"] = 1000 * float(np.median(lat))
        rep = []
        for _ in range(a.reprocess_queries):
            s = sync(); m.encode(coord); tb = materialize(m); Engine(m, tb, refine=False).run(q); rep.append(sync() - s)
        row["query_reprocess_ms"] = 1000 * float(np.median(rep))
        workload = [(t, c) for c in present for t in [c, *SYNONYMS.get(c, [])]] + [(w, None) for w in ABSENT[:3]]
        rng = np.random.default_rng(abs(hash(row["scene"])) % 2 ** 31)
        workload = [workload[i] for i in rng.permutation(len(workload))] * 2
        modes = {}
        for mode in ("no_refine", "refine_discard", "cracking"):
            tb = copy.deepcopy(table)
            out = []
            for term, cls in workload:
                eng = Engine(m, copy.deepcopy(tb) if mode == "refine_discard" else tb, refine=mode != "no_refine")
                s = sync(); res, st = eng.run(f"SELECT WHERE IS('{term}')"); ms = 1000 * (sync() - s)
                out.append(dict(term=term, cls=cls, ms=ms, ap50=ap50(eng, eng.t, res, objs, cls, vc) if cls else None,
                                refined=st["refined"]))
            modes[mode] = dict(queries=out, final_objects=len(tb))
        row["workload"] = modes
        rows.append(row)
        w = row["workload"]
        print(row["scene"], f"vox={row['voxels']} enc={row['encode_s']:.1f}s mat={row['materialize_s']:.1f}s "
              f"q={row['query_cached_ms']:.1f}ms vs {row['query_reprocess_ms']:.0f}ms | workload ms " +
              " ".join(f"{k}={np.mean([x['ms'] for x in v['queries']]):.0f}" for k, v in w.items()), flush=True)
    summ = dict(scenes=len(rows), encode_s=float(np.mean([r["encode_s"] for r in rows])),
                materialize_s=float(np.mean([r["materialize_s"] for r in rows])),
                query_cached_ms=float(np.mean([r["query_cached_ms"] for r in rows])),
                query_reprocess_ms=float(np.mean([r["query_reprocess_ms"] for r in rows])),
                relation_MB=float(np.mean([r["relation_bytes"] for r in rows]) / 1e6), raw_MB=float(np.mean([r["raw_bytes"] for r in rows]) / 1e6))
    for mode in ("no_refine", "refine_discard", "cracking"):
        qs = [x for r in rows for x in r["workload"][mode]["queries"]]
        half = [x for r in rows for x in r["workload"][mode]["queries"][len(r["workload"][mode]["queries"]) // 2:]]
        summ[mode] = dict(mean_ms=float(np.mean([x["ms"] for x in qs])), second_pass_ms=float(np.mean([x["ms"] for x in half])),
                          ap50=float(np.mean([x["ap50"] for x in qs if x["ap50"] is not None])),
                          synonym_ap50=float(np.mean([x["ap50"] for x in qs if x["ap50"] is not None and x["term"] != x["cls"]])))
    json.dump(dict(summary=summ, rows=rows), open(a.out, "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
