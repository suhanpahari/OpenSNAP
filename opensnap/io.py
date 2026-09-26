import os

import numpy as np
import torch


def load_any(path):
    """Load a point cloud: OpenScene .pth, SNAP example dir (coord/color/normal.npy), .ply/.pcd, KITTI .bin, nuScenes .pcd.bin, .npy."""
    p = path.lower()
    if os.path.isdir(path):
        d = {"coord": np.load(os.path.join(path, "coord.npy")).astype(np.float32)}
        for k in ("color", "normal"):
            f = os.path.join(path, f"{k}.npy")
            if os.path.exists(f):
                d[k] = np.load(f).astype(np.float32)
        return d
    if p.endswith(".pth"):
        data = torch.load(path, weights_only=False)
        if isinstance(data, (tuple, list)):
            d = {"coord": np.asarray(data[0], np.float32)}
            if len(data) > 1 and not np.isscalar(data[1]):
                col = np.asarray(data[1], np.float32)
                d["color"] = (col + 1) * 127.5 if col.min() < 0 else col
            return d
        return {k: np.asarray(v, np.float32) for k, v in data.items() if k in ("coord", "color", "normal")}
    if p.endswith(".pcd.bin"):
        x = np.fromfile(path, np.float32).reshape(-1, 5)
        return {"coord": x[:, :3], "strength": x[:, 3:4] / 255.0}
    if p.endswith(".bin"):
        x = np.fromfile(path, np.float32).reshape(-1, 4)
        return {"coord": x[:, :3], "strength": x[:, 3:4]}
    if p.endswith(".npy"):
        x = np.load(path).astype(np.float32)
        d = {"coord": x[:, :3]}
        if x.shape[1] >= 6:
            d["color"] = x[:, 3:6] * (255.0 if x[:, 3:6].max() <= 1 else 1.0)
        return d
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    d = {"coord": np.asarray(pcd.points, np.float32)}
    if pcd.has_colors():
        d["color"] = np.asarray(pcd.colors, np.float32) * 255.0
    if pcd.has_normals():
        d["normal"] = np.asarray(pcd.normals, np.float32)
    return d
