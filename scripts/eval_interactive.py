"""Interactive 3D instance segmentation: IoU@k clicks and NoC@{80,85,90} with simulated error-region clicks."""
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

DATA = {"s3dis": ("pointcept/s3dis", ["Area_5"]), "scannet": ("pointcept/scannet", ["val"])}
STRUCT = {"s3dis": {0, 1, 2}, "scannet": set()}


def load(path, use_color):
    d = {k: np.load(os.path.join(path, f"{k}.npy")) for k in ("coord", "color", "normal", "instance")}
    seg_file = "segment.npy" if os.path.exists(os.path.join(path, "segment.npy")) else "segment200.npy"
    d["segment"] = np.load(os.path.join(path, seg_file))
    ins = d["instance"].reshape(-1).astype(np.int64)
    seg = d["segment"].reshape(-1).astype(np.int64)
    color = d["color"].astype(np.float32) if use_color else None
    return d["coord"].astype(np.float32), color, d["normal"].astype(np.float32), ins, seg


def next_click(pred, gt, vcoord):
    """Positive click at the centre of the missed object region (SNAP supports positive clicks only)."""
    err = gt & ~pred
    if err.sum() == 0:
        return None
    return centre(err, vcoord)


def centre(region, vcoord):
    """Point of `region` farthest from everything outside it."""
    r = vcoord[region]
    out = vcoord[~region]
    sub = out[torch.randperm(len(out), device=out.device)[:20000]]
    d = torch.cdist(r, sub).min(1).values
    return torch.nonzero(region)[d.argmax(), 0]


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from opensnap.snap_model import SnapModel
    snap = build_model(args)
    struct = STRUCT[args.dataset]
    for f in files:
        coord, color, normal, ins, seg = load(f, args.color)
        vcoord, inverse = snap.encode(coord, color, normal if args.normals else None)
        inv = inverse
        ids = [i for i in np.unique(ins) if i >= 0 and (ins == i).sum() >= args.min_points
               and (not args.skip_structure or np.bincount(seg[ins == i]).argmax() not in struct)]
        ins_t = torch.from_numpy(ins).to(vcoord.device)
        for i in ids:
            gt_full = ins_t == i
            gt = torch.zeros(len(vcoord), dtype=torch.bool, device=vcoord.device)
            gt[inv[gt_full]] = True
            clicks, ious = [], []
            pred = torch.zeros_like(gt)
            for k in range(args.max_clicks):
                c = next_click(pred, gt, vcoord)
                if c is None:
                    ious += [ious[-1]] * (args.max_clicks - k)
                    break
                clicks.append(c)
                if args.model == "plus":
                    pred = snap.decode_clicks(vcoord[torch.stack(clicks)], select=args.select, gt=gt)
                else:
                    pred = snap.decode_clicks(vcoord[torch.stack(clicks)])
                ious.append(float((pred & gt).sum() / (pred | gt).sum()))
            queue.put(dict(scene=os.path.basename(f), obj=int(i), size=int(gt.sum()), seg=int(np.bincount(seg[ins == i]).argmax()), ious=ious))
    queue.put(None)


def build_model(args):
    from opensnap.snap_model import SnapModel
    if args.model == "snap":
        return SnapModel(args.ckpt, chunk=1)
    from opensnap.snap_plus import SnapPlusModel
    return SnapPlusModel(args.ckpt, use_sem=bool(args.use_sem))


def summarize(recs, max_clicks):
    I = np.array([r["ious"] for r in recs])
    out = {f"IoU@{k}": float(I[:, k - 1].mean()) for k in (1, 3, 5, 10) if k <= max_clicks}
    for t in (0.8, 0.85, 0.9):
        hit = I >= t
        noc = np.where(hit.any(1), hit.argmax(1) + 1, max_clicks)
        out[f"NoC@{int(t * 100)}"] = float(noc.mean())
    out["objects"] = len(recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s3dis", choices=list(DATA))
    ap.add_argument("--model", default="snap", choices=["snap", "plus"])
    ap.add_argument("--ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--max_clicks", type=int, default=10)
    ap.add_argument("--select", default="iou", choices=["iou", "first", "oracle"])
    ap.add_argument("--use_sem", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_points", type=int, default=100)
    ap.add_argument("--skip_structure", type=int, default=0)
    ap.add_argument("--color", type=int, default=1)
    ap.add_argument("--normals", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/interactive.json")
    args = ap.parse_args()
    root, splits = DATA[args.dataset]
    files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))
    files = files[::args.stride]
    files = files[:args.limit] if args.limit else files
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, files[r::len(gpus)], args, queue)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    recs, done, t0 = [], 0, time.time()
    while done < len(procs):
        r = queue.get()
        if r is None:
            done += 1
            continue
        recs.append(r)
        if len(recs) % 100 == 0:
            print(f"{len(recs)} objects {time.time() - t0:.0f}s", flush=True)
    for p in procs:
        p.join()
    res = summarize(recs, args.max_clicks)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=res, records=recs), open(args.out, "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
