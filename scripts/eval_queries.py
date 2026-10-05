"""OVQL workload benchmark: concept (IS) and relational (IS AND NEAR) queries scored by instance AP against annotations."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

DATA = {"s3dis": ("pointcept/s3dis", ["Area_5"], "segment"), "scannet": ("pointcept/scannet", ["val"], "segment200"),
        "replica": ("pointcept/replica", ["all"], "segment")}
S3DIS = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]
SKIP = {"s3dis": {"ceiling", "floor", "wall", "clutter"}, "scannet": {"wall", "floor", "ceiling"},
        "replica": {"wall", "floor", "ceiling", "undefined", "wall-plug", "anonymize_text", "anonymize_picture"}}
VARIANTS = {
    "snap_native": dict(dense=False, relay=0.0, refine=False),
    "opensnap": dict(dense=True, relay=0.8, refine=False),
    "opensnap_refine": dict(dense=True, relay=0.8, refine=True),
}
EXTRA = {}
DECODER_VARIANTS = {"opensnap_refine_v4": dict(dense=True, relay=0.8, refine=True, decoder=True, dedup=0.5),
                    "opensnap_refine_v4_nodedup": dict(dense=True, relay=0.8, refine=True, decoder=True, dedup=0.0)}


def sweep_variants():
    return {f"dedup{d}_t{t}": dict(dense=True, relay=0.8, refine=True, theta=t, dedup=d)
            for d in (0.3, 0.5, 0.7) for t in (0.1, 0.2, 0.3)}


def classes_for(dataset):
    if dataset == "s3dis":
        return S3DIS
    if dataset == "replica":
        return json.load(open(os.path.join(paths.DOWNLOADS, "data", "pointcept", "replica", "classes.json")))
    paths.add(paths.SNAP_DIR)
    from datasets.demo import DemoDatset
    d = DemoDatset.__new__(DemoDatset)
    DemoDatset.__init__(d, domain="Indoor")
    return list(d.labels_scannet)


def gt_objects(ins, seg, inv, nvox, classes, skip):
    """Voxelised GT instances: list of (class_name, voxel bool mask)."""
    out = []
    for i in np.unique(ins):
        if i < 0:
            continue
        pts = ins == i
        lab = np.bincount(seg[pts][seg[pts] >= 0], minlength=len(classes)).argmax() if (seg[pts] >= 0).any() else -1
        if lab < 0 or classes[lab] in skip:
            continue
        m = np.zeros(nvox, bool)
        m[inv[pts]] = True
        if m.sum() >= 20:
            out.append((classes[lab], m))
    return out


def relational_gt(objs, a, b, dist):
    """Indices of GT objects of class a whose bbox gap to some object of class b is <= dist."""
    return [k for k, (ca, ma, ba) in enumerate(objs) if ca == a and any(
        cb == b and kk != k and gap(ba, bb) <= dist for kk, (cb, mb, bb) in enumerate(objs))]


def gap(a, b):
    lo = np.maximum(a[0] - b[1], b[0] - a[1]).clip(min=0)
    return float(np.linalg.norm(lo))


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if args.sweep:
        EXTRA.update(sweep_variants())
    decoder = None
    if args.text_ckpt:
        EXTRA.update(DECODER_VARIANTS)
    import dataclasses
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    clip_snap = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, args.feature)
    clip_dense = TextEncoder(dense.clip_name) if dense.clip_name != "ViT-B/32" else clip_snap
    m = OpenSNAP(SnapModel(args.snap_ckpt), clip_snap, dense, clip_dense, Config(), anchors=snap_train_vocab())
    if args.text_ckpt:
        from opensnap.text_decoder import TextDecoder
        decoder = TextDecoder(args.text_ckpt, clip_snap)
    classes = classes_for(args.dataset)
    root, _, segf = DATA[args.dataset]
    for f in files:
        coord = np.load(os.path.join(f, "coord.npy")).astype(np.float32)
        ins = np.load(os.path.join(f, "instance.npy")).reshape(-1).astype(np.int64)
        seg = np.load(os.path.join(f, f"{segf}.npy")).reshape(-1).astype(np.int64)
        np.random.seed(abs(hash(os.path.basename(f))) % 2 ** 31)
        torch.manual_seed(0)
        t0 = time.time()
        m.encode(coord)
        enc = time.time() - t0
        base = materialize(m, m.propose_c2f() if args.proposals == "c2f" else None)
        vc = m.scene["vcoord"].cpu().numpy()
        inv = m.scene["inverse"].cpu().numpy()
        skip = set() if args.keep_all else SKIP[args.dataset]
        objs = [(c, mk, (vc[mk].min(0), vc[mk].max(0))) for c, mk in gt_objects(ins, seg, inv, len(vc), classes, skip)]
        present = sorted({c for c, _, _ in objs})
        queries = [("IS", c, None) for c in present]
        rng = np.random.default_rng(abs(hash(os.path.basename(f))) % 2 ** 31)
        pairs = [(a, b) for a in present for b in present if a != b and relational_gt(objs, a, b, args.near)]
        for k in rng.permutation(len(pairs))[:args.rel_per_scene]:
            queries.append(("NEAR", *pairs[k]))
        rec = dict(scene=os.path.basename(f), n_gt=len(objs), encode_s=enc, build_s=base.build_sec,
                   nbytes=base.nbytes()["total"], raw_bytes=int(coord.nbytes), n_objects=len(base), results={})
        for vname, v in {**VARIANTS, **EXTRA}.items():
            import copy
            table = copy.deepcopy(base)
            if not v["dense"]:
                table.pooled = torch.zeros(len(table), 0, device=table.token.device)
            m.cfg = dataclasses.replace(Config(), relay_weight=v["relay"])
            m.text_decoder = decoder if v.get("decoder") else None
            eng = Engine(m, table, refine=v["refine"], gamma=v.get("gamma", 0.0), refine_below=v.get("theta", 0.3),
                         dedup=v.get("dedup", 0.0))
            out = []
            for kind, a, b in queries:
                q = f"SELECT WHERE IS('{a}')" if kind == "IS" else f"SELECT WHERE IS('{a}') AND NEAR(IS('{b}'), {args.near})"
                res, st = eng.run(q)
                masks = [np.isin(np.arange(len(vc)), table.points[i]) for i, _ in res]
                gt_idx = [k for k, o in enumerate(objs) if o[0] == a] if kind == "IS" else relational_gt(objs, a, b, args.near)
                ious = np.array([[(pm & objs[g][1]).sum() / max((pm | objs[g][1]).sum(), 1) for g in gt_idx] for pm in masks]) \
                    if masks and gt_idx else np.zeros((len(masks), len(gt_idx)))
                out.append(dict(kind=kind, a=a, b=b, scores=[s for _, s in res], ious=ious.tolist(), n_gt=len(gt_idx), ms=st["ms"]))
            rec["results"][vname] = dict(queries=out, refined=eng.stats["refined"], n_objects_after=len(table))
        queue.put(rec)
    queue.put(None)


def average_precision(items, thr):
    """items: list over queries of (scores, ious (P, G), n_gt); AP over the union, matched greedily per query."""
    s_all, tp_all, n_gt = [], [], 0
    for scores, ious, g in items:
        n_gt += g
        ious = np.asarray(ious).reshape(len(scores), g) if g else np.zeros((len(scores), 0))
        used = np.zeros(g, bool)
        for k in np.argsort(-np.asarray(scores)):
            hit = False
            if g:
                cand = np.where(~used & (ious[k] >= thr))[0]
                if len(cand):
                    used[cand[ious[k, cand].argmax()]] = True
                    hit = True
            s_all.append(scores[k])
            tp_all.append(hit)
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-np.asarray(s_all))
    tp = np.asarray(tp_all)[o]
    prec = np.cumsum(tp) / (np.arange(len(tp)) + 1)
    rec = np.cumsum(tp) / n_gt
    prec = np.concatenate([[0.0], prec, [0.0]])
    rec = np.concatenate([[0.0], rec, [rec[-1] if len(rec) else 0.0]])
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def summarize(recs, splits=None):
    out = {}
    for v in recs[0]["results"]:
        for kind in ("IS", "NEAR"):
            per_class = {}
            for r in recs:
                for q in r["results"][v]["queries"]:
                    if q["kind"] == kind:
                        per_class.setdefault(q["a"] if kind == "IS" else (q["a"], q["b"]), []).append((q["scores"], q["ious"], q["n_gt"]))
            if not per_class:
                continue
            res = {}
            for name, thrs in (("AP", np.arange(0.5, 0.96, 0.05)), ("AP50", [0.5]), ("AP25", [0.25])):
                cls_ap = {c: np.nanmean([average_precision(it, t) for t in thrs]) for c, it in per_class.items()}
                res[name] = float(np.nanmean(list(cls_ap.values())))
                if splits and kind == "IS":
                    for sp, members in splits.items():
                        vals = [a for c, a in cls_ap.items() if c in members]
                        if vals:
                            res[f"{name}_{sp}"] = float(np.nanmean(vals))
            res["ms"] = float(np.mean([q["ms"] for r in recs for q in r["results"][v]["queries"] if q["kind"] == kind]))
            out[f"{v}/{kind}"] = res
    out["systems"] = dict(encode_s=float(np.mean([r["encode_s"] for r in recs])), build_s=float(np.mean([r["build_s"] for r in recs])),
                          objects=float(np.mean([r["n_objects"] for r in recs])),
                          relation_MB=float(np.mean([r["nbytes"] for r in recs]) / 1e6), raw_xyz_MB=float(np.mean([r["raw_bytes"] for r in recs]) / 1e6),
                          refined_per_scene=float(np.mean([r["results"]["opensnap_refine"]["refined"] for r in recs])))
    for v in recs[0]["results"]:
        out.setdefault(f"{v}/IS", {})["refined"] = float(np.mean([r["results"][v]["refined"] for r in recs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s3dis", choices=list(DATA))
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--rel_per_scene", type=int, default=5)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="results/queries.json")
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--keep_all", type=int, default=0)
    ap.add_argument("--proposals", default="fps", choices=["fps", "c2f"])
    ap.add_argument("--text_ckpt", default="")
    ap.add_argument("--split_override", default="")
    args = ap.parse_args()

    root, splits, _ = DATA[args.dataset]
    if args.split_override:
        splits = [args.split_override]
    files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))[::args.stride]
    files = files[:args.limit] if args.limit else files
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, files[r::len(gpus)], args, queue)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    recs, done = [], 0
    while done < len(procs):
        r = queue.get()
        if r is None:
            done += 1
            continue
        recs.append(r)
        print(f"[{len(recs)}/{len(files)}] {r['scene']} gt={r['n_gt']} objs={r['n_objects']}", flush=True)
    for p in procs:
        p.join()
    sp = None
    if args.dataset == "scannet":
        sp = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab", "scannet200_splits.json")))
    res = summarize(recs, sp)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=res, records=recs), open(args.out, "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
