"""Choose labelling settings on development dumps only, then score test dumps once with the chosen setting.

Settings: alpha (dense vs token), t (temperature multiplier: softmax(x / (t tau)) = renormalised p^(1/t)), beta (per-point
mix of dense evidence), confidence form. Development sets: S3DIS Area 1 and 30 ScanNet train scenes (never reported).
The image weight exists only where frames exist (Replica) and is chosen by leave-one-scene-out CV there."""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import bench  # noqa: E402

CONF = {"p*s": lambda p, s: p * s, "p": lambda p, s: p, "sqrt(p)*s": lambda p, s: p ** 0.5 * s}


def sharpen(p, t):
    p = p.float().clamp(min=1e-8)
    if t == 1.0:
        return p
    q = p ** (1.0 / t)
    return q / q.sum(-1, keepdim=True)


def outputs(d, g):
    alpha, t, beta, conf, w_img, gamma = g if len(g) == 6 else (*g, 0.0)
    lp = alpha * sharpen(d["terms"]["dense"], t).log() + (1 - alpha) * sharpen(d["terms"]["token"], t).log()
    if w_img > 0 and "image" in d["terms"]:
        lp = (1 - w_img) * lp + w_img * sharpen(d["terms"]["image"], t).log()
    pm = lp.softmax(-1).cuda()
    w = d["w"].float().cuda()
    cover = w.sum(0)
    wv = w / (w > 0).float().sum(1, keepdim=True).clamp(min=1) ** gamma if gamma else w
    vote = (wv.T @ pm) / wv.sum(0).clamp(min=1e-12)[:, None]
    dense = sharpen(d["dense_pts"], t).cuda() if d["dense_pts"] is not None else vote
    out = torch.where((cover > 0)[:, None], beta * dense + (1 - beta) * vote, dense)
    ids = np.asarray(d["ids"])
    pv = ids[out.argmax(1)[d["inverse"].cuda()].cpu().numpy()]
    if d["nn"] is not None:
        pv = np.where(d["nn"] >= 0, pv[np.clip(d["nn"], 0, None)], 0)
    sem, C = d["sem"], len(ids) + 1
    lut = np.zeros(int(max(ids.max(), sem.max())) + 1, int)
    lut[ids] = np.arange(1, C)
    ok = sem > 0
    cm = np.bincount(lut[sem[ok]] * C + lut[pv[ok]], minlength=C * C).reshape(C, C)
    p, c = pm.max(1)
    sc = d["scores"].float()
    return cm, [(int(ids[int(c[k])]), float(CONF[conf](float(p[k]), float(sc[k])))) for k in range(len(c))]


def masks_of(d):
    m = (d["w"].float() > 0).numpy()[:, d["inverse"].numpy()]
    if d["nn"] is not None:
        m = np.where(d["nn"][None] >= 0, m[:, np.clip(d["nn"], 0, None)], False)
    return m


def metrics(outs, gts, masks, valid):
    cm = sum(o[0] for o in outs)
    tp, g, pr = np.diag(cm)[1:].astype(float), cm[1:].sum(1).astype(float), cm[:, 1:].sum(0).astype(float)
    pres = g > 0
    sc = [bench.match_scene(gt, [(mk, l, c) for mk, (l, c) in zip(ms, o[1])], valid) for o, gt, ms in zip(outs, gts, masks)]
    ap = bench.summarize_ap(bench.evaluate(sc, valid), [str(v) for v in valid])
    return dict(mIoU=float((tp / np.maximum(g + pr - tp, 1))[pres].mean()), mAcc=float((tp / np.maximum(g, 1))[pres].mean()),
                AP=ap["AP"], AP50=ap["AP50"], AP25=ap["AP25"])


def load_set(dump, recs, valid_key):
    R = {r["scene"]: r for r in torch.load(recs, weights_only=False)}
    scenes = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(dump, "*.pt")))
    D = {s: torch.load(os.path.join(dump, f"{s}.pt"), weights_only=False) for s in scenes}
    return scenes, D, {s: R[s]["gt"] for s in scenes}, {s: masks_of(D[s]) for s in scenes}


def objective(m):
    return (m["mIoU"] + m["AP25"] + m["AP50"] + m["AP"]) / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True, help="dump_dir:recs:valid_json")
    ap.add_argument("--test", nargs="+", required=True, help="name:dump_dir:recs:valid_json")
    ap.add_argument("--out", default="results/compare/tuned.json")
    a = ap.parse_args()
    grid = list(itertools.product((0.3, 0.5, 0.7, 0.9), (0.5, 1.0, 2.0), (0.0, 0.3, 0.6), tuple(CONF), (0.0,), (0.0, 0.5, 1.0)))
    dev_scores = {g: 0.0 for g in grid}
    for spec in a.dev:
        dump, recs, valid = spec.split(":")
        valid = json.loads(valid)
        scenes, D, G, M = load_set(dump, recs, valid)
        for g in grid:
            m = metrics([outputs(D[s], g) for s in scenes], [G[s] for s in scenes], [M[s] for s in scenes], valid)
            dev_scores[g] += objective(m)
        print("dev done", dump, flush=True)
    best = max(grid, key=dev_scores.get)
    default = (0.7, 1.0, 0.3, "p*s", 0.0, 0.0)
    print("chosen on dev:", best, "objective", round(dev_scores[best], 4), "default", round(dev_scores[default], 4), flush=True)
    res = dict(chosen=list(best), default=list(default), dev_objective={str(k): v for k, v in sorted(dev_scores.items(), key=lambda x: -x[1])[:20]}, test={})
    for spec in a.test:
        name, dump, recs, valid = spec.split(":", 3)
        valid = json.loads(valid)
        scenes, D, G, M = load_set(dump, recs, valid)
        row = {}
        for tag, g in (("default", default), ("dev-chosen", best)):
            row[tag] = metrics([outputs(D[s], g) for s in scenes], [G[s] for s in scenes], [M[s] for s in scenes], valid)
        if "image" in D[scenes[0]]["terms"]:
            wgrid = [(best[0], best[1], best[2], best[3], w, best[5]) for w in (0.25, 0.5, 0.75, 0.9, 1.0)]
            held = []
            chosen = {}
            for s in scenes:
                rest = [x for x in scenes if x != s]
                sc = {g: objective(metrics([outputs(D[x], g) for x in rest], [G[x] for x in rest], [M[x] for x in rest], valid)) for g in wgrid}
                chosen[s] = max(sc, key=sc.get)
                held.append(outputs(D[s], chosen[s]))
            row["dev-chosen + image (LOSO w_img)"] = metrics(held, [G[s] for s in scenes], [M[s] for s in scenes], valid)
            row["loso_w_img"] = {s: g[4] for s, g in chosen.items()}
        res["test"][name] = row
        print(name, {k: ({kk: round(100 * x, 1) for kk, x in v.items()} if isinstance(v, dict) and "mIoU" in v else v) for k, v in row.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
