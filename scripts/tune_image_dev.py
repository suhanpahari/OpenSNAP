"""Choose the image column (weight, encoder mix, image temperature, gamma) on ScanNet *train* development scenes with frames,
then score the test dumps once with that setting. The 3D labelling setting is the dev-chosen one (tuned.json)."""
import argparse
import multiprocessing as mp
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import bench  # noqa: E402
import eval_standard as E  # noqa: E402
import loso_image as L  # noqa: E402
import tune_dev as T  # noqa: E402


def load(dump, recs):
    R = {r["scene"]: r for r in torch.load(recs, weights_only=False)}
    scenes = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(dump, "*.pt")))
    D = {s: torch.load(os.path.join(dump, f"{s}.pt"), weights_only=False) for s in scenes}
    side = dump.rstrip("/") + "_img"
    I = {s: {k: x.float() for k, x in torch.load(os.path.join(side, f"{s}.pt"), weights_only=False).items()} for s in scenes}
    for s in scenes:
        I[s].setdefault("siglip_l", D[s]["terms"]["image"].float())
    return scenes, D, I, {s: R[s]["gt"] for s in scenes}, {s: T.masks_of(D[s]) for s in scenes}


SH = {}


def _match(job):
    s, preds = job
    return bench.match_scene(SH["G"][s], [(mk, l, c) for mk, (l, c) in zip(SH["M"][s], preds)], SH["valid"])


def score(scenes, D, I, G, M, base, g, valid, names=None, splits=None, pool=None):
    w, mix, t_img, gamma = g
    cms, jobs = [], []
    for s in scenes:
        pm = L.probs(D[s], I[s], base, w, mix, t_img)
        cms.append(T.sem_cm(D[s], pm, base[1], base[2], gamma))
        jobs.append((s, T.preds(D[s], pm, base[3])))
    mts = pool.map(_match, jobs)
    ap = bench.summarize_ap(bench.evaluate(mts, valid), names or [str(v) for v in valid], splits)
    ap.pop("per_class_AP")
    return {**T.sem_metrics(sum(cms)), **ap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="../downloads/runs/dump_scannet_sel_img:results/compare/dev/scannet_sel_img_recs.pt")
    ap.add_argument("--test", nargs="+", default=["replica_union:r51:../downloads/runs/dump_replica_ul4:results/compare/replica_ul4_recs.pt",
                                                  "replica_lift:r51:../downloads/runs/dump_replica_lift:results/compare/replica_lift_recs.pt",
                                                  "scannet18:200:../downloads/runs/dump_scannet18_img:results/compare/scannet18_img_recs.pt",
                                                  "scannet200:198:../downloads/runs/dump_scannet200_img:results/compare/scannet200_img_recs.pt"])
    ap.add_argument("--base", default="results/compare/tuned.json")
    ap.add_argument("--out", default="results/compare/scannet_image_dev.json")
    a = ap.parse_args()
    base = tuple(json.load(open(a.base))["chosen"][:4])
    labels, ids, _, inst = E.benchmark("scannet200")
    name_of = dict(zip(ids, labels))
    splits = json.load(open(os.path.join(E.VOCAB, "scannet200_splits.json")))
    scenes, D, I, G, M = load(*a.dev.split(":"))
    SH.update(G=G, M=M, valid=inst["198"])
    pool = mp.get_context("fork").Pool(12)
    mixes = [k for k in L.MIXES if all(e in I[scenes[0]] for e in L.MIXES[k])]
    grid = list(itertools.product((0.5, 0.6, 0.75, 0.9, 1.0), mixes, (1.0, 2.0), (0.0, 1.0)))
    dev = {}
    for n, g in enumerate(grid):
        dev[g] = score(scenes, D, I, G, M, base, g, inst["198"], pool=pool)
        if n % 10 == 0:
            print(f"dev {n + 1}/{len(grid)}", flush=True)
    best = max(grid, key=lambda g: T.objective(dev[g]))
    print("chosen on dev:", best, {k: round(100 * v, 1) for k, v in dev[best].items()}, flush=True)
    res = dict(base=list(base), chosen=list(best), dev_top={str(g): dev[g] for g in sorted(grid, key=lambda g: -T.objective(dev[g]))[:10]}, test={})
    for spec in a.test:
        name, key, dump, recs = spec.split(":", 3)
        sc, D, I, G, M = load(dump, recs)
        pool.close()
        if key == "r51":
            v = json.load(open(os.path.join(L.VOCAB, "replica51.json")))
            valid = [v["labels"].index(l) + 1 for l in v["instance_labels"]]
            names, sp = [v["labels"][i - 1] for i in valid], None
        else:
            valid, sp = inst[key], splits
            names = [name_of[i] for i in valid]
        SH.update(G=G, M=M, valid=valid)
        pool = mp.get_context("fork").Pool(12)
        res["test"][name] = {tag: score(sc, D, I, G, M, base, g, valid, names, sp, pool=pool)
                             for tag, g in (("dev-chosen image setting", best), ("default image weight 0.5, SigLIP-L", (0.5, "L", 1.0, 0.0)))}
        for tag, r in res["test"][name].items():
            print(name, tag, {k: round(100 * v, 1) for k, v in r.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
