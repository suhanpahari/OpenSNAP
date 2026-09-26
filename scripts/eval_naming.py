"""Click-to-name benchmark: simulated clicks -> SNAP mask -> free-form name, scored against GT at the click."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap.core import SNAP_TEMPLATES, Config, Proposals  # noqa: E402
from opensnap.data import LABELSETS, load_scene, prior_vocab, scene_files, snap_train_vocab  # noqa: E402

STRUCTURE = {"wall", "floor", "ceiling", "other", "otherfurniture", "objects", "misc", "void", "unlabeled"}
OPEN_METHODS = ["snap", "opensnap_3d", "vlm", "rav", "vlm_choose", "choose"]
CLOSED_METHODS = ["openscene_point", "opensnap_closed"]


def sample_clicks(gt, labels, k, rng, only=None):
    classes = [c for c in np.unique(gt) if c != 255 and labels[c] not in STRUCTURE and (only is None or labels[c] in only)]
    rng.shuffle(classes)
    return [(int(rng.choice(np.nonzero(gt == c)[0])), int(c)) for c in classes[:k]]


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from opensnap.core import OpenSNAP
    from opensnap.namer import VLMNamer
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    clip_snap = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, args.feature)
    clip_dense = clip_snap if dense.clip_name == "ViT-B/32" else TextEncoder(dense.clip_name)
    anchors = snap_train_vocab()
    prior = prior_vocab(args.prior)
    only = {l for l in LABELSETS[args.dataset] if l.lower() not in set(anchors)} if args.unseen_only else None
    model = OpenSNAP(SnapModel(args.snap_ckpt), clip_snap, dense, clip_dense, Config(), anchors=anchors)
    namer = VLMNamer(args.vlm, style=args.vlm_style) if args.vlm else None
    judge = TextEncoder(args.judge)
    labels = LABELSETS[args.dataset]
    label_emb = judge(labels, ("a {}",))
    snap_vocab = [c for c in anchors]
    snap_text = clip_snap(snap_vocab, SNAP_TEMPLATES)

    def to_label(name):
        if name in labels:
            return labels.index(name)
        return int((judge([name], ("a {}",)) @ label_emb.T).argmax())

    for f in files:
        coord, color, gt = load_scene(f)
        rng = np.random.default_rng(abs(hash(os.path.basename(f))) % 2 ** 31)
        np.random.seed(abs(hash(os.path.basename(f))) % 2 ** 31)
        model.encode(coord, color)
        dense_prob = model.dense_probs(labels)
        for idx, c in sample_clicks(gt, labels, args.clicks, rng, only):
            props = model.click([idx])
            mask = (props.logits[0] > 0)[model.scene["inverse"]].cpu().numpy()
            inter = (mask & (gt == c)).sum()
            rec = dict(scene=os.path.basename(f), gt=labels[c], mask_iou=float(inter / max((mask | (gt == c)).sum(), 1)))
            rec["snap"] = snap_vocab[int((props.tokens @ snap_text.T).argmax())]
            p = model.mask_probs(props, prior)[0]
            rec["opensnap_3d"] = prior[int(p.argmax())]
            rec["openscene_point"] = labels[int(dense_prob[model.scene["inverse"][idx]].argmax())]
            rec["opensnap_closed"] = labels[int(model.mask_probs(props, labels)[0].argmax())]
            if namer is not None:
                t = time.time()
                best, ranked, reply = namer.name(model, props, 0, coord, color, prior_vocab=prior)
                from opensnap.namer import parse
                vlm = parse(reply)
                rec.update(vlm=vlm[0] if vlm else "object", rav=best, vlm_reply=reply, vlm_sec=time.time() - t)
                best, _, vlm_pick = namer.name_choose(model, props, 0, coord, color, prior)
                rec.update(vlm_choose=vlm_pick, choose=best)
            for m in OPEN_METHODS:
                if m in rec:
                    rec[m + "_mapped"] = labels[to_label(rec[m])]
            queue.put(rec)
    queue.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet", choices=list(LABELSETS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/scannet_lseg.pth.tar")
    ap.add_argument("--feature", default="lseg", choices=["lseg", "openseg"])
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vlm_style", default="highlight", choices=["highlight", "isolate"])
    ap.add_argument("--judge", default="ViT-L/14@336px")
    ap.add_argument("--clicks", type=int, default=5)
    ap.add_argument("--prior", default="snap", choices=["snap", "snap+lvis"])
    ap.add_argument("--unseen_only", action="store_true")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="results/naming.json")
    args = ap.parse_args()

    files = scene_files(args.dataset, args.split)[::args.stride]
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
        if len(recs) % 50 == 0:
            print(f"{len(recs)} clicks", flush=True)
    for p in procs:
        p.join()

    seen = set(snap_train_vocab())
    methods = [m for m in OPEN_METHODS if m in recs[0]] + CLOSED_METHODS
    rows = {}
    for m in methods:
        key = m + "_mapped" if m in OPEN_METHODS else m
        hit = np.array([r[key] == r["gt"] for r in recs])
        cls = sorted({r["gt"] for r in recs})
        per_cls = [hit[[r["gt"] == c for r in recs]].mean() for c in cls]
        uns = np.array([r["gt"].lower() not in seen for r in recs])
        rows[m] = dict(acc=float(hit.mean()), macc=float(np.mean(per_cls)),
                       acc_unseen=float(hit[uns].mean()) if uns.any() else float("nan"),
                       acc_seen=float(hit[~uns].mean()) if (~uns).any() else float("nan"))
    report = dict(args=vars(args), clicks=len(recs), mean_mask_iou=float(np.mean([r["mask_iou"] for r in recs])),
                  unseen_clicks=int(sum(r["gt"].lower() not in seen for r in recs)), results=rows, records=recs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\n{args.dataset}/{args.split} clicks={len(recs)} (unseen-class clicks={report['unseen_clicks']}) "
          f"mask IoU={100 * report['mean_mask_iou']:.1f}")
    print("| method | vocab | acc | mAcc | acc seen | acc unseen |\n|---|---|---|---|---|---|")
    for m, r in rows.items():
        v = "open" if m in OPEN_METHODS else "closed (eval labels)"
        print(f"| {m} | {v} | {100 * r['acc']:.1f} | {100 * r['macc']:.1f} | {100 * r['acc_seen']:.1f} | {100 * r['acc_unseen']:.1f} |")


if __name__ == "__main__":
    main()
