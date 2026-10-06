"""Per-click latency of corrective clicking (SNAP / token tuning / corrective decoder) on full S3DIS rooms."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_interactive import centre  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=6)
    ap.add_argument("--objects", type=int, default=10)
    ap.add_argument("--clicks", type=int, default=6)
    ap.add_argument("--corr_ckpt", default="../downloads/checkpoints/opensnap_corrective_v5_multi.pth")
    ap.add_argument("--out", default="results/aux/click_latency.json")
    a = ap.parse_args()
    from opensnap.clicks import ClickSegmenter, CorrectiveSegmenter
    from opensnap.snap_model import SnapModel
    snap = SnapModel("../downloads/ckpt/SNAP_C.pth", chunk=1)
    segs = {"snap": ClickSegmenter(snap, "snap"), "ttt": ClickSegmenter(snap, "ttt"), "v5": CorrectiveSegmenter(snap, a.corr_ckpt)}
    files = sorted(glob.glob("../downloads/data/pointcept/s3dis/Area_5/*"))[::11][:a.rooms]
    times = {k: [] for k in segs}
    voxels = []
    for f in files:
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        ins = np.load(f"{f}/instance.npy").reshape(-1)
        vc, inv = snap.encode(coord)
        voxels.append(len(vc))
        ids = [i for i in np.unique(ins) if i >= 0 and (ins == i).sum() >= 100][:a.objects]
        for i in ids:
            gt = torch.zeros(len(vc), dtype=torch.bool, device=vc.device)
            gt[inv[torch.from_numpy(ins == i).to(vc.device)]] = True
            for k, seg in segs.items():
                pos, neg, pred = [], [], torch.zeros_like(gt)
                for _ in range(a.clicks):
                    fn, fp = gt & ~pred, pred & ~gt
                    if not fn.any() and not fp.any():
                        break
                    (pos if fn.sum() >= fp.sum() else neg).append(int(centre(fn if fn.sum() >= fp.sum() else fp, vc)))
                    torch.cuda.synchronize(); t = time.time()
                    pred = seg(vc, torch.as_tensor(pos, dtype=torch.long, device=vc.device), torch.as_tensor(neg, dtype=torch.long, device=vc.device))
                    torch.cuda.synchronize()
                    if neg:
                        times[k].append(1000 * (time.time() - t))
    out = {k: dict(median_ms=float(np.median(v)), p90_ms=float(np.percentile(v, 90)), n=len(v)) for k, v in times.items()}
    out["mean_voxels"] = float(np.mean(voxels))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
