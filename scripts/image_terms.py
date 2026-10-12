"""Per-object class probabilities from additional image encoders, on the stored objects of a dump (Replica posed frames).
Saved beside the dump as <dump>_img/<scene>.pt: {encoder name: (M, C) half}."""
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
import eval_standard as E  # noqa: E402

ENCODERS = {"siglip_l": "google/siglip-large-patch16-384", "siglip_so400m": "google/siglip-so400m-patch14-384", "clip_l": "clip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", default=["../downloads/runs/dump_replica_ul4", "../downloads/runs/dump_replica_lift"])
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--frame_step", type=int, default=10)
    a = ap.parse_args()
    from opensnap.views import CLIPExpert, ImageExpert, PosedFrames, object_embeddings
    labels, ids, _, _ = E.benchmark("replica")
    intr = json.load(open(os.path.join(E.NICE, "cam_params.json")))["camera"]
    hub = os.path.join(paths.DOWNLOADS, "hf", "hub")
    for name in a.encoders:
        k = ENCODERS[name]
        expert = CLIPExpert(download_root=os.path.join(paths.DOWNLOADS, "clip")) if k == "clip" else ImageExpert(k, cache_dir=hub)
        for dump in a.dump:
            out_dir = dump.rstrip("/") + "_img"
            os.makedirs(out_dir, exist_ok=True)
            for f in sorted(glob.glob(os.path.join(dump, "*.pt"))):
                scene = os.path.basename(f)[:-3]
                o = os.path.join(out_dir, f"{scene}.pt")
                cur = torch.load(o, weights_only=False) if os.path.exists(o) else {}
                if name in cur:
                    continue
                d = torch.load(f, weights_only=False)
                assert list(d["ids"]) == list(ids)
                coord = E.load_replica(scene)[0]
                masks = (d["w"].float() > 0).numpy()[:, d["inverse"].numpy()]
                frames = PosedFrames(os.path.join(E.NICE, scene), intr, step=a.frame_step)
                emb, _ = object_embeddings(frames, expert, coord, masks)
                p = expert.probs(emb, labels)
                p = torch.where((emb.norm(dim=-1) > 0)[:, None], p, torch.full_like(p, 1.0 / len(labels)))
                cur[name] = p.half().cpu()
                torch.save(cur, o)
                print(name, dump, scene, flush=True)
        del expert
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
