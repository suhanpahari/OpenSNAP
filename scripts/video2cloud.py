"""Phone video -> metric, gravity-aligned coloured point cloud for OpenSNAP (VGGT point maps + metric depth scale)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
from opensnap.video import load_models, reconstruct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--conf_pct", type=float, default=65, help="drop points below this confidence percentile")
    ap.add_argument("--max_depth", type=float, default=2.5, help="metres")
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--scale_frames", type=int, default=8)
    a = ap.parse_args()
    out = a.out or os.path.join(paths.DOWNLOADS, "demo", os.path.splitext(os.path.basename(a.video))[0])
    vggt, metric = load_models(os.path.join(paths.DOWNLOADS, "hf"))
    reconstruct(a.video, out, vggt, metric, a.frames, a.conf_pct, a.max_depth, a.voxel, a.scale_frames)


if __name__ == "__main__":
    main()
