"""Expert weights for the image column by leave-one-scene-out cross-validation (no weight is chosen on the scene it is
scored on). Uses the per-expert probability dumps of eval_standard.py --dump."""
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

VOCAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab")


def fuse(terms, w_img, alpha):
    logp = alpha * (1 - w_img) * terms["dense"].float().clamp(min=1e-8).log() + \
        (1 - alpha) * (1 - w_img) * terms["token"].float().clamp(min=1e-8).log()
    if "image" in terms and w_img > 0:
        logp = logp + w_img * terms["image"].float().clamp(min=1e-8).log()
    return logp.softmax(-1)


def scene_outputs(d, gt, w_img, alpha, beta=0.3):
    ids = np.asarray(d["ids"])
    pm = fuse(d["terms"], w_img, alpha).cuda()
    w = d["w"].float().cuda()
    cover = w.sum(0)
    vote = (w.T @ pm) / cover.clamp(min=1e-6)[:, None]
    dense = d["dense_pts"].float().cuda() if d["dense_pts"] is not None else vote
    out = torch.where((cover > 0)[:, None], beta * dense + (1 - beta) * vote, dense)
    pts = out.argmax(1)[d["inverse"].cuda()].cpu().numpy()
    pv = ids[pts]
    if d["nn"] is not None:
        pv = np.where(d["nn"] >= 0, pv[np.clip(d["nn"], 0, None)], 0)
    sem = d["sem"]
    C = len(ids) + 1
    lut = np.zeros(int(max(ids.max(), sem.max())) + 1, int)
    lut[ids] = np.arange(1, C)
    ok = sem > 0
    conf = np.bincount(lut[sem[ok]] * C + lut[pv[ok]], minlength=C * C).reshape(C, C)
    p, c = pm.max(1)
    return conf, [(int(ids[int(c[k])]), float(p[k] * d["scores"][k])) for k in range(len(c))]


def scene_masks(d):
    masks = (d["w"].float() > 0).numpy()[:, d["inverse"].numpy()]
    if d["nn"] is not None:
        masks = np.where(d["nn"][None] >= 0, masks[:, np.clip(d["nn"], 0, None)], False)
    return masks


def metrics(outs, gts, valid, masks):
    conf = sum(o[0] for o in outs)
    tp = np.diag(conf)[1:].astype(float)
    g, pr = conf[1:].sum(1).astype(float), conf[:, 1:].sum(0).astype(float)
    pres = g > 0
    miou = float((tp / np.maximum(g + pr - tp, 1))[pres].mean())
    macc = float((tp / np.maximum(g, 1))[pres].mean())
    scenes = [bench.match_scene(gt, [(mk, l, cf) for mk, (l, cf) in zip(ms, o[1])], valid) for o, gt, ms in zip(outs, gts, masks)]
    ap = bench.summarize_ap(bench.evaluate(scenes, valid), [str(v) for v in valid])
    return dict(mIoU=miou, mAcc=macc, AP=ap["AP"], AP50=ap["AP50"], AP25=ap["AP25"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../downloads/runs/dump_replica")
    ap.add_argument("--recs", default="results/compare/replica_ovimap_c2f_recs.pt")
    ap.add_argument("--out", default="results/compare/replica_fusion_cv.json")
    a = ap.parse_args()
    recs = {r["scene"]: r for r in torch.load(a.recs, weights_only=False)}
    v = json.load(open(os.path.join(VOCAB, "replica51.json")))
    valid = [v["labels"].index(l) + 1 for l in v["instance_labels"]]
    scenes = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.dump, "*.pt")))
    dumps = {s: torch.load(os.path.join(a.dump, f"{s}.pt"), weights_only=False) for s in scenes}
    grid = list(itertools.product((0.0, 0.25, 0.5, 0.75, 0.9, 1.0), (0.3, 0.5, 0.7, 0.9)))
    cache = {(s, g): scene_outputs(dumps[s], recs[s]["gt"], *g) for s in scenes for g in grid}
    gts = {s: recs[s]["gt"] for s in scenes}
    mks = {s: scene_masks(dumps[s]) for s in scenes}
    full = {f"w_img={g[0]} alpha={g[1]}": metrics([cache[(s, g)] for s in scenes], [gts[s] for s in scenes], valid, [mks[s] for s in scenes])
            for g in grid}
    chosen, held = {}, []
    for s in scenes:
        rest = [x for x in scenes if x != s]
        score = {g: (lambda m: m["mIoU"] + m["AP50"])(metrics([cache[(x, g)] for x in rest], [gts[x] for x in rest], valid, [mks[x] for x in rest]))
                 for g in grid}
        g = max(score, key=score.get)
        chosen[s] = g
        held.append(cache[(s, g)])
    cv = metrics(held, [gts[s] for s in scenes], valid, [mks[s] for s in scenes])
    json.dump(dict(grid=full, loso_cv=cv, chosen={s: list(g) for s, g in chosen.items()}), open(a.out, "w"), indent=1)
    print("LOSO-CV:", {k: round(100 * x, 1) for k, x in cv.items()}, "chosen:", chosen)
    for k in sorted(full, key=lambda k: -full[k]["mIoU"])[:6]:
        print(k, {kk: round(100 * x, 1) for kk, x in full[k].items()})


if __name__ == "__main__":
    main()
