"""Text-prompted segmentation: 'segment the <class>' -> mask, IoU vs the class region; ScanNet200 head/common/tail, S3DIS zero-shot."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_queries import DATA, SKIP, classes_for  # noqa: E402


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from opensnap.text import TextEncoder
    classes = classes_for(args.dataset)
    clip = TextEncoder("ViT-B/32")
    if args.method == "trained":
        from opensnap.snap_plus import SnapPlusModel
        model = SnapPlusModel(args.ckpt)
    else:
        from opensnap.core import Config, OpenSNAP
        from opensnap.data import snap_train_vocab
        from opensnap.openscene_model import OpenSceneModel
        from opensnap.snap_model import SnapModel
        dense = OpenSceneModel(args.openscene_ckpt, args.feature)
        model = OpenSNAP(SnapModel(args.snap_ckpt), clip, dense, TextEncoder(dense.clip_name), Config(), anchors=snap_train_vocab())
    _, _, segf = DATA[args.dataset]
    for f in files:
        coord = np.load(os.path.join(f, "coord.npy")).astype(np.float32)
        seg = np.load(os.path.join(f, f"{segf}.npy")).reshape(-1).astype(np.int64)
        if args.method == "trained":
            vcoord, inv = model.encode(coord)
        else:
            sc = model.encode(coord)
            vcoord, inv = sc["vcoord"], sc["inverse"]
        inv = inv.cpu().numpy()
        out = []
        for c in np.unique(seg):
            if c < 0 or classes[c] in SKIP[args.dataset] or (seg == c).sum() < 100:
                continue
            gt = np.zeros(len(vcoord), bool)
            gt[inv[seg == c]] = True
            name = classes[c]
            if args.method == "trained":
                pred = model.decode_text(clip([name], ("segment {}",))[0]).cpu().numpy()
            else:
                pred = model.text_segment(name)[0].cpu().numpy()
            out.append(dict(cls=name, iou=float((pred & gt).sum() / max((pred | gt).sum(), 1)),
                            prec=float((pred & gt).sum() / max(pred.sum(), 1))))
        queue.put(dict(scene=os.path.basename(f), res=out))
    queue.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet", choices=list(DATA))
    ap.add_argument("--method", default="trained", choices=["trained", "oracle_clicks"])
    ap.add_argument("--ckpt", default="../downloads/runs/scannet_plus/last.pth")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="results/text.json")
    args = ap.parse_args()
    root, splits, _ = DATA[args.dataset]
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
    for p in procs:
        p.join()
    per = {}
    for r in recs:
        for x in r["res"]:
            per.setdefault(x["cls"], []).append(x["iou"])
    cls_iou = {c: float(np.mean(v)) for c, v in per.items()}
    res = dict(mIoU=float(np.mean(list(cls_iou.values()))), classes=len(cls_iou))
    if args.dataset == "scannet":
        sp = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab", "scannet200_splits.json")))
        for k, members in sp.items():
            vals = [v for c, v in cls_iou.items() if c in members]
            res[f"mIoU_{k}"] = float(np.mean(vals)) if vals else float("nan")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=res, per_class=cls_iou, records=recs), open(args.out, "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
