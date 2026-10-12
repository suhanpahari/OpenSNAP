"""Leave-one-scene-out choice of the image-only settings on Replica (no image-bearing dev set exists): image weight,
confidence multiplier for 2D-lifted rows and vote specificity gamma. Other settings stay at the system defaults."""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_dev as T  # noqa: E402

VOCAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab")


def with_lift_mult(d, n_lifted, mult):
    if n_lifted == 0 or mult == 1.0:
        return d
    d = dict(d)
    s = d["scores"].clone().float()
    s[len(s) - n_lifted:] *= mult
    d["scores"] = s
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../downloads/runs/dump_replica_lift")
    ap.add_argument("--recs", default="results/compare/replica_lift_recs.pt")
    ap.add_argument("--out", default="results/compare/replica_lift_loso.json")
    ap.add_argument("--base", default="", help="tuned.json: take alpha, t, beta, conf from its dev-chosen setting")
    a = ap.parse_args()
    R = {r["scene"]: r for r in torch.load(a.recs, weights_only=False)}
    v = json.load(open(os.path.join(VOCAB, "replica51.json")))
    valid = [v["labels"].index(l) + 1 for l in v["instance_labels"]]
    scenes = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.dump, "*.pt")))
    D = {s: torch.load(os.path.join(a.dump, f"{s}.pt"), weights_only=False) for s in scenes}
    M = {s: T.masks_of(D[s]) for s in scenes}
    G = {s: R[s]["gt"] for s in scenes}
    grid = list(itertools.product((0.5, 0.75, 1.0), (0.25, 0.5, 1.0), (0.0, 0.5, 1.0)))
    base = tuple(json.load(open(a.base))["chosen"][:4]) if a.base else (0.7, 1.0, 0.3, "p*s")
    out = {(s, g): T.outputs(with_lift_mult(D[s], R[s].get("n_lifted", 0), g[1]), (*base, g[0], g[2]))
           for s in scenes for g in grid}
    full = {str(g): T.metrics([out[(s, g)] for s in scenes], [G[s] for s in scenes], [M[s] for s in scenes], valid) for g in grid}
    held, chosen = [], {}
    for s in scenes:
        rest = [x for x in scenes if x != s]
        sc = {g: T.objective(T.metrics([out[(x, g)] for x in rest], [G[x] for x in rest], [M[x] for x in rest], valid)) for g in grid}
        chosen[s] = max(sc, key=sc.get)
        held.append(out[(s, chosen[s])])
    cv = T.metrics(held, [G[s] for s in scenes], [M[s] for s in scenes], valid)
    json.dump(dict(base=list(base), loso_cv=cv, chosen={s: list(g) for s, g in chosen.items()}, grid=full), open(a.out, "w"), indent=1)
    print("LOSO-CV:", {k: round(100 * x, 1) for k, x in cv.items()}, chosen)
    for k in sorted(full, key=lambda k: -T.objective(full[k]))[:5]:
        print(k, {kk: round(100 * x, 1) for kk, x in full[k].items()})


if __name__ == "__main__":
    main()
