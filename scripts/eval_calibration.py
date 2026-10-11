"""Calibration and sensitivity of the open-vocabulary predicate IS(c).

cache:   per scene, the stored relation's evidence columns (segmenter token, mask-pooled dense feature) and the IoU of
         every stored object with every GT instance (with its class).
analyze: P(IS(c) | o) for every object and every class present in the scene, under background vocabularies B,
         temperatures tau / tau_r, expert weight alpha and relay weight rho; outcome y = object matches a GT instance of
         class c at IoU >= 0.5. Reports ECE (15 bins), Brier, reliability bins and IS-query AP; fits temperature /
         Platt scaling on ScanNet train and applies it unchanged to the other datasets; synonym robustness.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402

SETS = {
    "scannet_train": dict(root="pointcept/scannet/train", seg="segment200", stride=20, ckpt="scannet_openseg", domain="Indoor"),
    "scannet_val": dict(root="pointcept/scannet/val", seg="segment200", stride=2, ckpt="scannet_openseg", domain="Indoor"),
    "replica": dict(root="pointcept/replica/all", seg="segment", stride=1, ckpt="matterport_openseg", domain="Indoor"),
    "s3dis": dict(root="pointcept/s3dis/Area_5", seg="segment", stride=1, ckpt="matterport_openseg", domain="Indoor"),
    "nuscenes": dict(root="pointcept/nuscenes/val", seg="segment", stride=30, ckpt="nuscenes_openseg", domain="Outdoor"),
}
DSKEY = {"scannet_train": "scannet", "scannet_val": "scannet", "replica": "replica", "s3dis": "s3dis", "nuscenes": "nuscenes"}
CACHE = os.path.join(paths.DOWNLOADS, "runs", "calib")
DEFAULT_B = "B=234 training vocab"
SYNONYMS = {"chair": "seat", "table": "desk", "sofa": "couch", "bookcase": "bookshelf", "board": "whiteboard", "door": "doorway",
            "window": "windowpane", "column": "pillar", "beam": "girder", "cabinet": "cupboard", "bed": "mattress",
            "toilet": "lavatory", "sink": "washbasin", "picture": "painting", "pillow": "cushion", "lamp": "light",
            "trash can": "garbage bin", "monitor": "computer screen", "rug": "carpet", "car": "automobile", "pedestrian": "person",
            "truck": "lorry", "motorcycle": "motorbike", "bicycle": "bike", "bus": "coach", "trailer": "semitrailer",
            "traffic cone": "pylon", "barrier": "roadblock", "tv screen": "television", "vase": "flower pot",
            "shelf": "rack", "blinds": "window shade", "cushion": "pillow", "plant": "houseplant", "indoor plant": "houseplant"}


def cache(name, a):
    from eval_queries import SKIP, classes_for, gt_objects
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    s = SETS[name]
    clip = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(f"../downloads/ckpt/{s['ckpt']}.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth", domain=s["domain"]), clip, dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    classes = classes_for(DSKEY[name])
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", s["root"], "*")))[::s["stride"]]
    files = files[a.shard::a.nshards]
    os.makedirs(os.path.join(CACHE, name), exist_ok=True)
    for f in files:
        out = os.path.join(CACHE, name, os.path.basename(f) + ".pt")
        if os.path.exists(out):
            continue
        coord = np.load(os.path.join(f, "coord.npy")).astype(np.float32)
        ins = np.load(os.path.join(f, "instance.npy")).reshape(-1).astype(np.int64)
        seg = np.load(os.path.join(f, f"{s['seg']}.npy")).reshape(-1).astype(np.int64)
        np.random.seed(0); torch.manual_seed(0)
        m.encode(coord)
        t = materialize(m)
        nv = len(m.scene["vcoord"])
        inv = m.scene["inverse"].cpu().numpy()
        gts = gt_objects(ins, seg, inv, nv, classes, SKIP[DSKEY[name]])
        P = np.zeros((len(t), nv), bool)
        for k, p in enumerate(t.points):
            P[k, p] = True
        G = np.stack([g for _, g in gts]) if gts else np.zeros((0, nv), bool)
        inter = P.astype(np.float32) @ G.T.astype(np.float32)
        iou = inter / np.maximum(P.sum(1)[:, None] + G.sum(1)[None] - inter, 1)
        torch.save(dict(token=t.token.half().cpu(), pooled=t.pooled.half().cpu(), conf=t.conf.cpu(), iou=iou.astype(np.float16),
                        gt_cls=[c for c, _ in gts], dense_dim=t.pooled.shape[1], clip_dense=dense.clip_name), out)
        print(name, os.path.basename(f), len(t), len(gts), flush=True)


class Scorer:
    def __init__(self, clip_dense_name):
        from opensnap.data import snap_train_vocab
        from opensnap.text import TextEncoder
        self.clip = TextEncoder("ViT-B/32")
        self.clip_dense = self.clip if clip_dense_name == "ViT-B/32" else TextEncoder(clip_dense_name)
        self.vocab = snap_train_vocab()
        self.anchor = self.clip(self.vocab, ("segment {}",))

    def probs(self, token, pooled, labels, tau=0.01, tau_r=0.05, alpha=0.7, rho=0.8, use_dense=True, use_snap=True):
        """(M, len(labels)) softmax-calibrated probabilities with the same experts as the engine."""
        terms, w = [], []
        if use_dense and pooled.numel():
            terms.append((pooled @ self.clip_dense(labels, ("a {} in a scene",)).T / tau).softmax(-1)); w.append(alpha)
        if use_snap:
            text = self.clip(labels, ("segment {}",))
            tok = F.normalize(token, dim=-1)
            direct = (tok @ text.T / tau).softmax(-1)
            if rho > 0:
                post = (tok @ self.anchor.T / tau_r).softmax(-1)
                relay = (post @ (self.anchor @ text.T) / tau).softmax(-1)
                direct = direct ** (1 - rho) * relay ** rho
            terms.append(direct); w.append(1 - alpha if len(terms) > 1 else 1.0)
        if not terms:
            return torch.full((len(token), len(labels)), 1.0 / len(labels), device=token.device)
        logp = sum(wi * t.clamp(min=1e-8).log() for wi, t in zip(w, terms)) / sum(w)
        return logp.softmax(-1)


def is_scores(sc, d, classes, bg, **kw):
    """P(IS(c)|o) for every class c present in the scene: softmax over [c] + background B (B minus c)."""
    tok, pooled = d["token"].float().cuda(), d["pooled"].float().cuda()
    out = {}
    for c in sorted(set(d["gt_cls"])):
        labels = [c, *[b for b in bg if b != c]] if bg is not None else None
        out[c] = (sc.probs(tok, pooled, labels, **kw)[:, 0] if labels else None)
    return out


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if k.any():
            e += k.mean() * abs(p[k].mean() - y[k].mean())
            rel.append((float(lo), float(hi), int(k.sum()), float(p[k].mean()), float(y[k].mean())))
    return float(e), rel


def collect(sc, data, bg_fn, syn=False, **kw):
    """Pairs (p, y) and per-class ranked lists for AP."""
    from eval_queries import average_precision
    ps, ys, items = [], [], {}
    for d in data:
        if len(d["gt_cls"]) == 0 or len(d["conf"]) == 0:
            continue
        tok, pooled = d["token"].float().cuda(), d["pooled"].float().cuda()
        iou = d["iou"].astype(np.float32)
        for c in sorted(set(d["gt_cls"])):
            q = SYNONYMS.get(c, None) if syn else c
            if q is None:
                continue
            bg = bg_fn(c)
            labels = [q, *[b for b in bg if b not in (c, q)]]
            p = sc.probs(tok, pooled, labels, **kw)[:, 0].cpu().numpy()
            cols = [k for k, g in enumerate(d["gt_cls"]) if g == c]
            y = (iou[:, cols] >= 0.5).any(1)
            ps.append(p); ys.append(y)
            items.setdefault(c, []).append((p.tolist(), iou[:, cols].tolist(), len(cols)))
    p, y = np.concatenate(ps), np.concatenate(ys).astype(float)
    e, rel = ece(p, y)
    ap = float(np.nanmean([np.nanmean([average_precision(it, t) for t in np.arange(0.5, 0.96, 0.05)]) for it in items.values()]))
    return dict(ECE=e, Brier=float(np.mean((p - y) ** 2)), AP=ap, pos_rate=float(y.mean()), mean_p=float(p.mean()), n=int(len(p)),
                reliability=rel), p, y


def analyze(a):
    from eval_queries import classes_for
    from opensnap.data import snap_train_vocab
    data = {n: [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(CACHE, n, "*.pt")))] for n in SETS}
    data = {n: v for n, v in data.items() if v}
    scorers = {}
    vocab = snap_train_vocab()
    small = ("wall", "floor", "ceiling", "object")
    res = dict(n_scenes={n: len(v) for n, v in data.items()})

    def scorer(n):
        cd = data[n][0]["clip_dense"]
        if cd not in scorers:
            scorers[cd] = Scorer(cd)
        return scorers[cd]

    bgs = {"B=4 (previous default)": lambda n: (lambda c: small),
           "B=dataset classes": lambda n: (lambda c, L=classes_for(DSKEY[n]): L),
           "B=234 training vocab": lambda n: (lambda c: vocab),
           "B=1 ('object')": lambda n: (lambda c: ("object",))}
    res["background"] = {}
    for bname, bf in bgs.items():
        res["background"][bname] = {n: collect(scorer(n), data[n], bf(n))[0] for n in data}
        print(bname, {n: (round(v["ECE"], 3), round(v["AP"], 3)) for n, v in res["background"][bname].items()}, flush=True)
    res["tau"], res["alpha"], res["rho"], res["tau_r"] = {}, {}, {}, {}
    for tau in (0.005, 0.01, 0.02, 0.05):
        res["tau"][tau] = {n: collect(scorer(n), data[n], bgs[DEFAULT_B](n), tau=tau)[0] for n in data}
    for al in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        res["alpha"][al] = {n: collect(scorer(n), data[n], bgs[DEFAULT_B](n), alpha=al,
                                       use_snap=al < 1.0, use_dense=al > 0.0)[0] for n in data}
    for rho in (0.0, 0.4, 0.8, 1.0):
        res["rho"][rho] = {n: collect(scorer(n), data[n], bgs[DEFAULT_B](n), rho=rho)[0] for n in data}
    for tr in (0.02, 0.05, 0.1):
        res["tau_r"][tr] = {n: collect(scorer(n), data[n], bgs[DEFAULT_B](n), tau_r=tr)[0] for n in data}
    res["synonyms"] = {n: collect(scorer(n), data[n], bgs[DEFAULT_B](n), syn=True)[0] for n in data}
    for n in data:
        res["synonyms"][n]["AP_class_names_same_subset"] = None
    fit = "scannet_train"
    if fit in data:
        _, p, y = collect(scorer(fit), data[fit], bgs[DEFAULT_B](fit))
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        best = (1e9, 1.0, 0.0)
        for A in np.linspace(0.2, 3.0, 57):
            for B in np.linspace(-6, 4, 101):
                q = 1 / (1 + np.exp(-(A * lg + B)))
                nll = -np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9))
                if nll < best[0]:
                    best = (nll, A, B)
        _, A, B = best
        res["platt"] = dict(A=float(A), B=float(B), fitted_on=fit)
        res["platt_applied"] = {}
        for n in data:
            r, p, y = collect(scorer(n), data[n], bgs[DEFAULT_B](n))
            lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            q = 1 / (1 + np.exp(-(A * lg + B)))
            e, rel = ece(q, y)
            res["platt_applied"][n] = dict(ECE_before=r["ECE"], ECE_after=e, Brier_before=r["Brier"],
                                           Brier_after=float(np.mean((q - y) ** 2)), reliability_after=rel, AP=r["AP"])
        print("platt", res["platt"], {n: (round(v["ECE_before"], 3), round(v["ECE_after"], 3)) for n, v in res["platt_applied"].items()})
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "analyze"])
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="results/compare/calibration.json")
    a = ap.parse_args()
    if a.mode == "cache":
        for n in a.sets:
            cache(n, a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
