"""Bird's-eye open-vocabulary labelling of nuScenes LiDAR sweeps with words outside the nuScenes label set."""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

VOCAB = ["road", "sidewalk", "grass", "tree", "building", "car", "truck", "bus", "person", "bicycle", "motorcycle",
         "traffic cone", "street light", "fence", "parked trailer", "construction barrier"]
NUSC = {"sidewalk", "car", "truck", "bus", "person", "bicycle", "motorcycle", "traffic cone"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--range", type=float, default=40.0)
    ap.add_argument("--out", default="assets/outdoor_gallery.jpg")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import load_scene, snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    dense = OpenSceneModel("../downloads/ckpt/nuscenes_openseg.pth.tar", "openseg", voxel_size=0.05)
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth", domain="Outdoor"), TextEncoder(), dense, TextEncoder(dense.clip_name),
                 Config(), anchors=snap_train_vocab())
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", "nuscenes_3d", "val", "*.pth")))
    files = files[:: len(files) // a.n][: a.n]
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, (a.n + 1) // 2, figsize=(7 * ((a.n + 1) // 2), 14), facecolor="white")
    for ax, f in zip(axes.flat, files):
        coord, _, _ = load_scene(f)
        np.random.seed(0); torch.manual_seed(0)
        m.encode(coord)
        pred = m.semantic(m.propose(), VOCAB).argmax(1).cpu().numpy()
        k = (np.abs(coord[:, :2]) < a.range).all(1)
        ax.scatter(coord[k, 0], coord[k, 1], c=cmap(pred[k] % 20), s=0.3, linewidths=0)
        ax.set_xlim(-a.range, a.range); ax.set_ylim(-a.range, a.range); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(os.path.basename(f)[:8], fontsize=14)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i % 20), label=w + ("" if w in NUSC else " ★")) for i, w in enumerate(VOCAB)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=14, markerscale=2, frameon=False)
    fig.suptitle("OpenSNAP on nuScenes LiDAR (zero-shot, XYZ-only)  ·  ★ = not a nuScenes class", fontsize=18)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=70, pil_kwargs=dict(quality=85, optimize=True))
    print("saved", a.out)


if __name__ == "__main__":
    main()
