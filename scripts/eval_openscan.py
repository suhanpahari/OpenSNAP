"""OpenScan / GOV-3D (AAAI 2026): open-vocabulary instance segmentation where the vocabulary is object attributes (affordance,
property, type, manner, synonym, requirement, element, material) instead of class names. Same stored objects as the
ScanNet200 run; each object is labelled with its most likely attribute of the aspect; ScanNet instance AP per aspect and the
mean over all 341 attributes (the paper's 'Mean')."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import bench, paths  # noqa: E402
from opensnap.core import Config  # noqa: E402

OS = os.path.join(paths.DOWNLOADS, "openscan")
ASPECTS = ["affordance", "property", "type", "manner", "synonyms", "requirement", "element", "material"]
QUERY = {"affordance": "something used to {}", "requirement": "something that needs {}", "element": "something with {}",
         "material": "something made of {}", "manner": "something {}", "property": "something {}", "type": "{}", "synonyms": "{}"}


def vocab(aspect):
    rows = [l.rstrip("\n").split(None, 1) for l in open(os.path.join(OS, "openscan_label_mapping", f"openscan-labels.{aspect}.tsv"))][1:]
    return [int(i) for i, _ in rows], [c.strip() for _, c in rows]


def worker(rank, gpu, scenes, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from opensnap.core import OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    clip = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, "openseg", voxel_size=0.02)
    m = OpenSNAP(SnapModel(args.snap_ckpt, domain="Indoor"), clip, dense, TextEncoder(dense.clip_name), Config(), anchors=snap_train_vocab())
    V = {a: vocab(a) for a in ASPECTS}
    for scene in scenes:
        coord = np.load(os.path.join(paths.DOWNLOADS, "data", "pointcept", "scannet", "val", scene, "coord.npy")).astype(np.float32)
        np.random.seed(abs(hash(scene)) % 2 ** 31)
        torch.manual_seed(0)
        t0 = time.time()
        m.encode(coord)
        props = m.propose()
        inv = m.scene["inverse"].cpu().numpy()
        vm = (props.logits > 0).cpu().numpy()
        rec = dict(scene=scene, masks=[np.nonzero(vm[k][inv])[0].astype(np.int32) for k in range(len(vm))], preds={})
        for a in ASPECTS:
            ids, names = V[a]
            labels = [QUERY[a].format(n) if args.template else n for n in names]
            for var, kw in (("opensnap", {}), ("dense_only", dict(use_snap=False))):
                p, c = m.mask_probs(props, labels, **kw).max(1)
                rec["preds"][(a, var)] = [(ids[int(c[k])], float(p[k] * props.scores[k])) for k in range(len(c))]
        rec["sec"] = time.time() - t0
        queue.put(rec)
    queue.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/scannet_openseg.pth.tar")
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--out", default="results/compare/openscan.json")
    args = ap.parse_args()
    scenes = sorted(f[:-4] for f in os.listdir(os.path.join(OS, "openscan_val_gt", "gt_material")))
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, scenes[r::len(gpus)], args, q)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    recs, done = [], 0
    while done < len(procs):
        r = q.get()
        if r is None:
            done += 1
            continue
        recs.append(r)
        print(f"[{len(recs)}/{len(scenes)}] {r['scene']} {r['sec']:.1f}s", flush=True)
    for p in procs:
        p.join()
    res = {}
    for var in ("opensnap", "dense_only"):
        res[var], allap = {}, []
        for a in ASPECTS:
            ids, names = vocab(a)
            matched = []
            for r in recs:
                gt = np.loadtxt(os.path.join(OS, "openscan_val_gt", f"gt_{a}", f"{r['scene']}.txt"), dtype=np.int64)
                preds = []
                for (lab, cf), pts in zip(r["preds"][(a, var)], r["masks"]):
                    mk = np.zeros(len(gt), bool)
                    mk[pts] = True
                    preds.append((mk, lab, cf))
                matched.append(bench.match_scene(gt, preds, ids))
            apm = bench.evaluate(matched, ids)
            allap.append(apm)
            res[var][a] = {k: v for k, v in bench.summarize_ap(apm, names).items() if k != "per_class_AP"}
        res[var]["mean"] = {k: v for k, v in bench.summarize_ap(np.concatenate(allap), [str(i) for i in range(sum(len(x) for x in allap))]).items()
                            if k != "per_class_AP"}
        print(var, {a: round(100 * res[var][a]["AP"], 1) for a in res[var]})
    json.dump(dict(args=vars(args), n_scenes=len(recs), results=res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
