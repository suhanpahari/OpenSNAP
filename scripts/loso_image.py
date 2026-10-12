"""Leave-one-scene-out choice of the image column on Replica: encoder mix (SigLIP-L, SigLIP so400m, CLIP-L), image weight,
image temperature, vote specificity gamma and (with lifted proposals) their confidence multiplier. The 3D labelling setting
is the dev-chosen one (tuned.json). Per-scene matches are cached once per setting, so folds only re-aggregate."""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import bench  # noqa: E402
import tune_dev as T  # noqa: E402
import eval_standard as E  # noqa: E402

VOCAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab")
MIXES = {"L": {"siglip_l": 1.0}, "S": {"siglip_so400m": 1.0}, "L+S": {"siglip_l": 0.5, "siglip_so400m": 0.5},
         "L+C": {"siglip_l": 0.5, "clip_l": 0.5}, "L+S+C": {"siglip_l": 1 / 3, "siglip_so400m": 1 / 3, "clip_l": 1 / 3},
         "L+S+.5C": {"siglip_l": 0.4, "siglip_so400m": 0.4, "clip_l": 0.2}}
G, M, VALID = {}, {}, []


def _match(job):
    s, preds = job
    return bench.match_scene(G[s], [(mk, l, c) for mk, (l, c) in zip(M[s], preds)], VALID)


def probs(d, img, base, w, mix, t_img):
    alpha, t = base[0], base[1]
    lp = alpha * T.sharpen(d["terms"]["dense"], t).log() + (1 - alpha) * T.sharpen(d["terms"]["token"], t).log()
    li = sum(pi * T.sharpen(img[k], t_img).log() for k, pi in MIXES[mix].items())
    return ((1 - w) * lp + w * li).softmax(-1).cuda()


def metrics(cms, matches):
    return {**T.sem_metrics(sum(cms)), **{k: v for k, v in bench.summarize_ap(bench.evaluate(matches, VALID), [str(v) for v in VALID]).items()
                                         if k in ("AP", "AP50", "AP25")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../downloads/runs/dump_replica_lift")
    ap.add_argument("--recs", default="results/compare/replica_lift_recs.pt")
    ap.add_argument("--base", default="results/compare/tuned.json")
    ap.add_argument("--out", default="results/compare/replica_lift_loso_image.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bench", default="replica")
    ap.add_argument("--inst", default="")
    ap.add_argument("--weights", default="0.6,0.75,0.9")
    a = ap.parse_args()
    R = {r["scene"]: r for r in torch.load(a.recs, weights_only=False)}
    if a.bench == "replica":
        v = json.load(open(os.path.join(VOCAB, "replica51.json")))
        VALID.extend(v["labels"].index(l) + 1 for l in v["instance_labels"])
    else:
        VALID.extend(E.benchmark(a.bench)[3][a.inst])
    base = tuple(json.load(open(a.base))["chosen"][:4])
    scenes = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.dump, "*.pt")))
    D = {s: torch.load(os.path.join(a.dump, f"{s}.pt"), weights_only=False) for s in scenes}
    side = a.dump.rstrip("/") + "_img"
    I = {s: {k: x.float() for k, x in torch.load(os.path.join(side, f"{s}.pt"), weights_only=False).items()} if os.path.isdir(side)
         else {"siglip_l": D[s]["terms"]["image"].float()} for s in scenes}
    mixes = [k for k in MIXES if all(e in I[scenes[0]] for e in MIXES[k])]
    for s in scenes:
        G[s], M[s] = R[s]["gt"], T.masks_of(D[s])
    lifted = any(R[s].get("n_lifted", 0) for s in scenes)
    grid = list(itertools.product(tuple(float(w) for w in a.weights.split(",")), tuple(mixes), (1.0, 2.0), (0.0, 1.0), (0.25, 0.5) if lifted else (1.0,)))
    pool = mp.get_context("fork").Pool(a.workers)
    cm, mt = {}, {}
    for n, g in enumerate(grid):
        w, mix, t_img, gamma, mult = g
        jobs = []
        for s in scenes:
            d = D[s]
            if R[s].get("n_lifted", 0) and mult != 1.0:
                d = dict(d)
                sc = d["scores"].clone().float()
                sc[len(sc) - R[s]["n_lifted"]:] *= mult
                d["scores"] = sc
            pm = probs(d, I[s], base, w, mix, t_img)
            cm[(s, g)] = T.sem_cm(d, pm, base[1], base[2], gamma)
            jobs.append((s, T.preds(d, pm, base[3])))
        for s, r in zip(scenes, pool.map(_match, jobs)):
            mt[(s, g)] = r
        if n % 12 == 0:
            print(f"{n + 1}/{len(grid)}", flush=True)
    full = {g: metrics([cm[(s, g)] for s in scenes], [mt[(s, g)] for s in scenes]) for g in grid}
    chosen, hcm, hmt = {}, [], []
    for s in scenes:
        rest = [x for x in scenes if x != s]
        sc = {g: T.objective(metrics([cm[(x, g)] for x in rest], [mt[(x, g)] for x in rest])) for g in grid}
        chosen[s] = max(sc, key=sc.get)
        hcm.append(cm[(s, chosen[s])])
        hmt.append(mt[(s, chosen[s])])
    cv = metrics(hcm, hmt)
    json.dump(dict(base=list(base), loso_cv=cv, chosen={s: list(g) for s, g in chosen.items()}, grid={str(g): r for g, r in full.items()}),
              open(a.out, "w"), indent=1)
    print("LOSO-CV:", {k: round(100 * x, 1) for k, x in cv.items()}, chosen, flush=True)
    for g in sorted(full, key=lambda g: -T.objective(full[g]))[:8]:
        print(g, {k: round(100 * x, 1) for k, x in full[g].items()}, flush=True)


if __name__ == "__main__":
    main()
