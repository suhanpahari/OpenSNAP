"""Dense-collapse guard.
v1 (original): top non-structural dense label covers > 0.75 of object voxels.
v2: v1 and exact dense/token label agreement < 0.5.
v3: soft agreement (mean cosine between the text embeddings of the dense label and the segmenter-token label over object
    voxels) below a threshold calibrated on randomly sampled clean scans only (1st percentile; no collapsed scene used).
Test scenes: every clean scan on which v1 fired in the large runs, the lab-desk phone video, the 8 RGB-only Replica
reconstructions; calibration scenes: every 5th ScanNet val scan and every 6th HouseLayout3D room."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

D = paths.DOWNLOADS


def calibration_scenes():
    out = []
    for s in sorted(os.listdir(os.path.join(D, "data", "pointcept", "scannet", "val")))[::5]:
        out.append(("calib: ScanNet val", s, os.path.join(D, "data", "pointcept", "scannet", "val", s, "coord.npy"), "scannet_openseg"))
    rooms = []
    for f in sorted(os.listdir(os.path.join(D, "catalogs", "hl3d"))):
        rooms += [r["file"] for r in torch.load(os.path.join(D, "catalogs", "hl3d", f), weights_only=False)["rooms"]]
    for fn in rooms[::6]:
        split = [s for s in ("train", "val", "test") if os.path.exists(os.path.join(D, "data", "matterport_3d_160", s, fn))][0]
        out.append(("calib: Matterport (HL3D)", fn[:-4], os.path.join(D, "data", "matterport_3d_160", split, fn), "matterport_openseg"))
    return out


def scenes():
    out = []
    r = torch.load("results/compare/scannet200_val_recs.pt", weights_only=False)
    out += [("benchmark: ScanNet val", x["scene"], os.path.join(D, "data", "pointcept", "scannet", "val", x["scene"], "coord.npy"),
             "scannet_openseg") for x in r if x.get("dense_collapsed")]
    for f in sorted(os.listdir(os.path.join(D, "catalogs", "hl3d"))):
        c = torch.load(os.path.join(D, "catalogs", "hl3d", f), weights_only=False)
        for room in c["rooms"]:
            if room.get("dense_collapsed"):
                split = [s for s in ("train", "val", "test") if os.path.exists(os.path.join(D, "data", "matterport_3d_160", s, room["file"]))][0]
                out.append(("benchmark: Matterport (HL3D)", room["file"][:-4], os.path.join(D, "data", "matterport_3d_160", split, room["file"]),
                            "matterport_openseg"))
    out.append(("phone video", "lab_desk", os.path.join(D, "demo", "lab_desk", "coord.npy"), "matterport_openseg"))
    for s in ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"):
        out.append(("RGB-only video (Replica)", s, os.path.join(D, "runs", "video_replica", s, "coord.npy"), "matterport_openseg"))
    return out


def main():
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    anchors = snap_train_vocab()
    clip = TextEncoder()
    models, rows = {}, []
    snap = SnapModel("../downloads/ckpt/SNAP_C.pth")
    struct = torch.tensor([any(w in a for w in ("wall", "floor", "ceiling")) for a in anchors]).cuda()
    T = clip(anchors, ("segment {}",))
    for src, name, path, ck in calibration_scenes() + scenes():
        if ck not in models:
            dense = OpenSceneModel(f"../downloads/ckpt/{ck}.pth.tar", "openseg")
            models[ck] = OpenSNAP(snap, clip, dense, TextEncoder(dense.clip_name), Config(dense_collapse=0.0), anchors=anchors)
        m = models[ck]
        coord = torch.load(path, weights_only=False)[0] if path.endswith(".pth") else np.load(path)
        np.random.seed(0); torch.manual_seed(0)
        m.encode(np.asarray(coord, np.float32))
        lab_d = m.dense_probs(anchors).argmax(1)
        props = m.propose()
        lab_t_prop = m.snap_probs(props.tokens, anchors).argmax(1)
        w = (props.logits > 0).float() * props.scores[:, None]
        covered = w.sum(0) > 0
        lab_t = lab_t_prop[w.argmax(0)]
        obj = ~struct[lab_d] & covered
        counts = torch.bincount(lab_d[obj], minlength=len(anchors))
        top = int(counts.argmax())
        share = float(counts.max() / obj.sum().clamp(min=1))
        agree = float((lab_t[obj] == lab_d[obj]).float().mean())
        soft = float((T[lab_t[obj]] * T[lab_d[obj]]).sum(-1).mean())
        top_t = torch.bincount(lab_t[obj], minlength=len(anchors))
        rows.append(dict(source=src, scene=name, dense_label=anchors[top], share=share, token_share=float(top_t.max() / obj.sum().clamp(min=1)),
                         token_label=anchors[int(top_t.argmax())], agreement=agree, soft_agreement=soft, v1=share > 0.75,
                         v2=share > 0.75 and agree < 0.5))
        print(json.dumps(rows[-1]), flush=True)
    thr = float(np.percentile([r["soft_agreement"] for r in rows if r["source"].startswith("calib")], 1))
    for r in rows:
        r["v3"] = r["soft_agreement"] < thr
    summ = {}
    for src in sorted({r["source"] for r in rows}):
        rr = [r for r in rows if r["source"] == src]
        summ[src] = dict(n=len(rr), v1_fired=sum(r["v1"] for r in rr), v2_fired=sum(r["v2"] for r in rr), v3_fired=sum(r["v3"] for r in rr))
    json.dump(dict(rule_v1="share > 0.75", rule_v2="share > 0.75 and agreement < 0.5",
                   rule_v3=f"soft agreement < {thr:.3f} (1st percentile of clean calibration scans)", summary=summ, rows=rows),
              open("results/compare/guard.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
