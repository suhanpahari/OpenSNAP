"""Phone-video pipeline on the 8 Replica sequences, RGB only (no depth, no poses): VGGT reconstruction, then a similarity
transform from predicted to GT camera centres (Umeyama) used only for scoring against the GT mesh."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

NICE = os.path.join(paths.DOWNLOADS, "replica_nice", "Replica")
OUT = os.path.join(paths.DOWNLOADS, "runs", "video_replica")


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / Xc.var(0).sum()
    return s, R, my - s * R @ mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--max_depth", type=float, default=6.0)
    ap.add_argument("--voxel", type=float, default=0.01)
    a = ap.parse_args()
    from opensnap.video import load_models, reconstruct
    vggt, metric = load_models(os.path.join(paths.DOWNLOADS, "hf"))
    report = {}
    for s in a.scenes:
        n = len([f for f in os.listdir(os.path.join(NICE, s, "results")) if f.startswith("frame")])
        idx = np.linspace(0, n - 1, a.frames).round().astype(int)
        frames = [os.path.join(NICE, s, "results", f"frame{i:06d}.jpg") for i in idx]
        out = os.path.join(OUT, s)
        t0 = time.time()
        P, _ = reconstruct(None, out, vggt, metric, frame_paths=frames, max_depth=a.max_depth, voxel=a.voxel, ply=False,
                           log=lambda m: print(s, m, flush=True))
        sec = time.time() - t0
        v = np.load(os.path.join(out, "views.npz"))
        ex, R, p0, scale = v["extri"], v["R"], v["p0"], float(v["scale"])
        cen = np.stack([-e[:, :3].T @ e[:, 3] for e in ex]) * scale
        cen = (cen - p0) @ R.T
        gt = np.loadtxt(os.path.join(NICE, s, "traj.txt")).reshape(-1, 4, 4)[idx][:, :3, 3]
        sc, Ra, t = umeyama(cen, gt)
        res = np.linalg.norm((sc * cen @ Ra.T + t) - gt, axis=1)
        np.savez(os.path.join(out, "align.npz"), s=sc, R=Ra, t=t)
        report[s] = dict(points=int(len(P)), sec=sec, metric_scale_error=float(abs(1 - sc)), cam_rmse_m=float(np.sqrt((res ** 2).mean())))
        print(s, report[s], flush=True)
    json.dump(report, open(os.path.join(OUT, "reconstruction.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
