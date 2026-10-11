"""Cache SAM automatic masks for posed frames (every `step`-th frame), bit-packed, for lifting 2D masks into 3D proposals."""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

NICE = os.path.join(paths.DOWNLOADS, "replica_nice", "Replica")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"])
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(paths.DOWNLOADS, "runs", "sam_replica"))
    a = ap.parse_args()
    from transformers import pipeline
    g = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0, model_kwargs=dict(cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub")))
    for s in a.scenes:
        n = len([f for f in os.listdir(os.path.join(NICE, s, "results")) if f.startswith("frame")])
        os.makedirs(os.path.join(a.out, s), exist_ok=True)
        for i in range(0, n, a.step):
            f = os.path.join(a.out, s, f"{i:06d}.npz")
            if os.path.exists(f):
                continue
            im = Image.open(os.path.join(NICE, s, "results", f"frame{i:06d}.jpg")).convert("RGB")
            with torch.no_grad():
                out = g(im, points_per_batch=256, pred_iou_thresh=0.88, stability_score_thresh=0.92)
            M = np.stack([np.asarray(m, bool) for m in out["masks"]]) if len(out["masks"]) else np.zeros((0, im.height, im.width), bool)
            np.savez_compressed(f, bits=np.packbits(M, axis=-1), shape=np.array(M.shape), scores=np.asarray(out["scores"], np.float32))
        print(s, "done", flush=True)


if __name__ == "__main__":
    main()
