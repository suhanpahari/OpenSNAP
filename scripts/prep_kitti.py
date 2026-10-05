"""SemanticKITTI sequence 08 -> semseg .pth (OpenScene format) and pointcept folders with panoptic instances."""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

LEARNING_MAP = {10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5, 31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12,
                51: 13, 60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 252: 0, 253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4}
LUT = np.full(260, 255, np.uint8)
for k, v in LEARNING_MAP.items():
    LUT[k] = v
THINGS = 8
A = {}


def work(scan):
    a = A["args"]
    name = os.path.basename(scan)[:-4]
    pts = np.fromfile(scan, np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(scan.replace("velodyne", "labels")[:-4] + ".label", np.uint32)
    seg = LUT[lab & 0xFFFF]
    inst = (lab >> 16).astype(np.int64)
    ins = np.where((seg < THINGS) & (inst > 0), seg.astype(np.int64) * 65536 + inst, -1)
    _, ins_c = np.unique(ins, return_inverse=True)
    ins = np.where(ins >= 0, ins_c - (ins < 0).any(), -1).astype(np.int32)
    torch.save((pts, 0, seg), os.path.join(a.sem_dst, f"{name}.pth"))
    d = os.path.join(a.ins_dst, name)
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/coord.npy", pts)
    np.save(f"{d}/segment.npy", np.where(seg == 255, -1, seg).astype(np.int16))
    np.save(f"{d}/instance.npy", ins)
    return int(ins.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f71/soham/lift/semantic_kitti/dataset/sequences")
    ap.add_argument("--seq", default="08")
    ap.add_argument("--sem_dst", default=os.path.join(paths.DOWNLOADS, "data", "semantickitti", "val"))
    ap.add_argument("--ins_dst", default=os.path.join(paths.DOWNLOADS, "data", "pointcept", "semantickitti", "val"))
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.sem_dst, exist_ok=True)
    os.makedirs(a.ins_dst, exist_ok=True)
    scans = sorted(glob.glob(os.path.join(a.root, a.seq, "velodyne", "*.bin")))
    print(len(scans), "scans", flush=True)
    A["args"] = a
    with Pool(a.procs) as p:
        n = p.map(work, scans, chunksize=16)
    print("instances/scan", np.mean(n), "done", flush=True)


if __name__ == "__main__":
    main()
