"""Official nuScenes val keyframes -> semseg .pth (OpenScene format) and pointcept folders with box-derived instances."""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from nuscenes.utils import splits as SPLITS
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

LEARNING_MAP = {2: 6, 3: 6, 4: 6, 6: 6, 9: 0, 12: 7, 14: 1, 15: 2, 16: 2, 17: 3, 18: 4, 21: 5, 22: 8, 23: 9,
                24: 10, 25: 11, 26: 12, 27: 13, 28: 14, 30: 15}
LUT = np.full(32, 255, np.uint8)
for k, v in LEARNING_MAP.items():
    LUT[k] = v
G = {}


def table(root, name):
    return json.load(open(os.path.join(root, "v1.0-trainval", f"{name}.json")))


def to_sensor(ann, pose, cs):
    c = np.array(ann["translation"]) - pose["translation"]
    q = Quaternion(pose["rotation"]).inverse
    c, r = q.rotate(c), q * Quaternion(ann["rotation"])
    q = Quaternion(cs["rotation"]).inverse
    c, r = q.rotate(c - np.array(cs["translation"])), q * r
    return np.asarray(c), r.rotation_matrix, np.array(ann["size"])


def work(item):
    sd_tok, lidar_file, seg_file, pose, cs, anns = item
    root, a = G["root"], G["args"]
    pts = np.fromfile(os.path.join(root, lidar_file), np.float32).reshape(-1, 5)[:, :3]
    raw = np.fromfile(os.path.join(root, seg_file), np.uint8)
    seg = LUT[raw]
    ins = np.full(len(pts), -1, np.int32)
    for k, (cat_idx, ann) in enumerate(anns):
        c, R, (w, l, h) = to_sensor(ann, pose, cs)
        loc = (pts - c) @ R
        inside = (np.abs(loc[:, 0]) <= l / 2) & (np.abs(loc[:, 1]) <= w / 2) & (np.abs(loc[:, 2]) <= h / 2) & (raw == cat_idx)
        ins[inside & (seg != 255)] = k
    if a.sem:
        torch.save((pts, 0, seg), os.path.join(a.sem_dst, f"{sd_tok}.pth"))
    d = os.path.join(a.ins_dst, sd_tok)
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/coord.npy", pts)
    np.save(f"{d}/segment.npy", np.where(seg == 255, -1, seg).astype(np.int16))
    np.save(f"{d}/instance.npy", ins)
    return len(np.unique(ins[ins >= 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f71/soham/lift/nuscense_trainval")
    ap.add_argument("--sem_dst", default=os.path.join(paths.DOWNLOADS, "data", "nuscenes_official", "val"))
    ap.add_argument("--ins_dst", default=os.path.join(paths.DOWNLOADS, "data", "pointcept", "nuscenes", "val"))
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--split", default="val")
    ap.add_argument("--sem", type=int, default=1)
    a = ap.parse_args()
    if a.split != "val":
        a.sem_dst, a.ins_dst = a.sem_dst.replace("/val", f"/{a.split}"), a.ins_dst.replace("/val", f"/{a.split}")
    os.makedirs(a.sem_dst, exist_ok=True)
    os.makedirs(a.ins_dst, exist_ok=True)
    scenes = {s["token"] for s in table(a.root, "scene") if s["name"] in set(getattr(SPLITS, a.split))}
    samples = {s["token"] for s in table(a.root, "sample") if s["scene_token"] in scenes}
    cs = {c["token"]: c for c in table(a.root, "calibrated_sensor")}
    pose = {p["token"]: p for p in table(a.root, "ego_pose")}
    seg = {l["sample_data_token"]: l["filename"] for l in table(a.root, "lidarseg")}
    cat_index = {c["token"]: c["index"] for c in table(a.root, "category")}
    inst_cat = {i["token"]: cat_index[i["category_token"]] for i in table(a.root, "instance")}
    anns = {}
    for x in table(a.root, "sample_annotation"):
        if x["sample_token"] in samples:
            anns.setdefault(x["sample_token"], []).append((inst_cat[x["instance_token"]], x))
    items = [(sd["token"], sd["filename"], seg[sd["token"]], pose[sd["ego_pose_token"]], cs[sd["calibrated_sensor_token"]],
              anns.get(sd["sample_token"], []))
             for sd in table(a.root, "sample_data")
             if sd["sample_token"] in samples and sd["is_key_frame"] and sd["filename"].startswith("samples/LIDAR_TOP")]
    print(len(scenes), "scenes", len(items), "keyframes", flush=True)
    G.update(root=a.root, args=a)
    with Pool(a.procs) as p:
        n = p.map(work, items, chunksize=16)
    print("instances/frame", np.mean(n), "done", flush=True)


if __name__ == "__main__":
    main()
