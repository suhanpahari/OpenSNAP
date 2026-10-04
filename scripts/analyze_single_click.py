"""Single-click candidate analysis: true vs predicted IoU of every mask token; selection rules evaluated offline."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
from eval_interactive import DATA, centre, load  # noqa: E402


def collect(args):
    from opensnap.snap_plus import SnapPlusModel
    model = SnapPlusModel(args.ckpt)
    root, splits = DATA[args.dataset]
    files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))[::args.stride]
    rows = []
    for f in files:
        coord, color, normal, ins, seg = load(f, False)
        vcoord, inv = model.encode(coord)
        ins_t = torch.from_numpy(ins).to(vcoord.device)
        for i in np.unique(ins):
            if i < 0 or (ins == i).sum() < 100:
                continue
            gt = torch.zeros(len(vcoord), dtype=torch.bool, device=vcoord.device)
            gt[inv[ins_t == i]] = True
            c = centre(gt, vcoord)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                logits, iou, _ = model.model.decode(model.point, model.coord, clicks=vcoord[c][None, None].float())
            m = logits[0] > 0
            true = ((m & gt).sum(1).float() / (m | gt).sum(1).clamp(min=1).float()).tolist()
            rows.append(dict(scene=os.path.basename(f), true=true, pred=iou[0].float().tolist()))
    return rows


def evaluate(rows, delta):
    t, p = np.array([r["true"] for r in rows]), np.array([r["pred"] for r in rows])
    best_extra = 1 + p[:, 1:].argmax(1)
    use = p[np.arange(len(p)), best_extra] > p[:, 0] + delta
    pick = np.where(use, best_extra, 0)
    return float(t[np.arange(len(t)), pick].mean()), float(use.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = collect(args)
    t = np.array([r["true"] for r in rows])
    res = dict(objects=len(rows), snap_token=float(t[:, 0].mean()), oracle=float(t.max(1).mean()),
               rules={f"{d:.2f}": evaluate(rows, d) for d in np.arange(-0.1, 0.31, 0.05)})
    json.dump(dict(args=vars(args), results=res, rows=rows), open(args.out, "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
