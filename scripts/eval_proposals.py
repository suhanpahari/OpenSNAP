"""Class-agnostic proposal recall (IoU>=0.5/0.25) and cost on annotated scenes."""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pointcept/s3dis/Area_5")
    ap.add_argument("--methods", nargs="+", default=["fps400", "fps800", "c2f"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/proposals_s3dis.json")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), cfg=Config())
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", a.split, "*")))
    files = files[:a.limit] if a.limit else files
    stats = {k: dict(r50=[], r25=[], n=[], s=[]) for k in a.methods}
    for f in files:
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        ins = np.load(f"{f}/instance.npy").reshape(-1)
        m.encode(coord)
        inv = m.scene["inverse"].cpu().numpy()
        nv = len(m.scene["vcoord"])
        gts = []
        for i in np.unique(ins):
            if i < 0:
                continue
            g = np.zeros(nv, bool); g[inv[ins == i]] = True
            if g.sum() >= 20:
                gts.append(torch.from_numpy(g).cuda())
        G = torch.stack(gts).float()
        for k in a.methods:
            np.random.seed(0); torch.manual_seed(0)
            t0 = time.time()
            if k.startswith("fps"):
                m.cfg.num_prompts = int(k[3:])
                p = m.propose()
            else:
                p = m.propose_c2f()
            torch.cuda.synchronize()
            P = (p.logits > 0).float()
            inter = P @ G.T
            iou = inter / (P.sum(1)[:, None] + G.sum(1)[None] - inter).clamp(min=1)
            best = iou.max(0).values if len(P) else torch.zeros(len(G), device="cuda")
            st = stats[k]
            st["r50"].append(float((best >= 0.5).float().mean())); st["r25"].append(float((best >= 0.25).float().mean()))
            st["n"].append(len(P)); st["s"].append(time.time() - t0)
        print(os.path.basename(f), {k: (round(v["r50"][-1], 2), v["n"][-1]) for k, v in stats.items()}, flush=True)
    res = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in stats.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
