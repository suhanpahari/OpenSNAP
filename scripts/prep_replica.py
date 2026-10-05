"""Transfer Replica habitat instance/class labels onto OpenScene Replica points -> pointcept-style folders."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

SCENES = {"room0": "room_0", "room1": "room_1", "room2": "room_2", "office0": "office_0", "office1": "office_1",
          "office2": "office_2", "office3": "office_3", "office4": "office_4"}


def mesh_labels(raw, scene):
    ply = PlyData.read(os.path.join(raw, scene, "habitat", "mesh_semantic.ply"))
    v = np.stack([ply["vertex"][k] for k in "xyz"], 1).astype(np.float32)
    faces = ply["face"]
    idx = np.stack(faces["vertex_indices"])
    oid = np.asarray(faces["object_id"]).astype(np.int64)
    cent = v[idx].mean(1)
    pts = np.concatenate([v[idx].reshape(-1, 3), cent])
    lab = np.concatenate([np.repeat(oid, idx.shape[1]), oid])
    info = json.load(open(os.path.join(raw, scene, "habitat", "info_semantic.json")))
    cls = {o["id"]: o["class_name"] for o in info["objects"]}
    return pts, lab, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(paths.DOWNLOADS, "replica_raw"))
    ap.add_argument("--src", default=os.path.join(paths.DOWNLOADS, "data", "replica_3d"))
    ap.add_argument("--dst", default=os.path.join(paths.DOWNLOADS, "data", "pointcept", "replica", "all"))
    ap.add_argument("--max_dist", type=float, default=0.05)
    a = ap.parse_args()
    per_scene, names = {}, set()
    for s, r in SCENES.items():
        coord, color, _ = torch.load(os.path.join(a.src, f"{s}.pth"), weights_only=False)
        pts, lab, cls = mesh_labels(a.raw, r)
        d, nn = cKDTree(pts).query(coord, k=1, workers=16)
        ins = lab[nn]
        ins[d > a.max_dist] = -1
        per_scene[s] = (coord, color, ins, cls)
        names |= {cls.get(int(i), "") for i in np.unique(ins) if i >= 0}
        print(s, len(coord), f"median nn {np.median(d) * 100:.2f} cm, >{a.max_dist}m: {(d > a.max_dist).mean() * 100:.1f}%", flush=True)
    classes = sorted(n for n in names if n)
    for s, (coord, color, ins, cls) in per_scene.items():
        seg = np.array([classes.index(cls[int(i)]) if i >= 0 and cls.get(int(i)) in classes else -1 for i in ins], np.int64)
        out = os.path.join(a.dst, s)
        os.makedirs(out, exist_ok=True)
        np.save(f"{out}/coord.npy", coord.astype(np.float32))
        np.save(f"{out}/color.npy", ((color + 1) * 127.5).clip(0, 255).astype(np.uint8))
        np.save(f"{out}/instance.npy", np.where(seg >= 0, ins, -1))
        np.save(f"{out}/segment.npy", seg)
        print(s, "instances", len(np.unique(ins[seg >= 0])))
    json.dump(classes, open(os.path.join(os.path.dirname(a.dst), "classes.json"), "w"))
    print(len(classes), "classes:", classes)


if __name__ == "__main__":
    main()
