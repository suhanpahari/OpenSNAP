"""Video -> metric, gravity-aligned coloured point cloud (VGGT point maps + metric-depth scale + up-plane alignment)."""
import glob
import os
import subprocess

import numpy as np
import torch
from PIL import Image


def sharpness(path):
    g = np.asarray(Image.open(path).convert("L").resize((320, 240)), np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def extract_frames(video, out, n):
    """Extract 3n evenly spaced frames, keep the sharpest of each consecutive triple."""
    import re
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(out, exist_ok=True)
    info = subprocess.run([ff, "-hide_banner", "-i", video], capture_output=True, text=True).stderr
    h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    dur = int(h) * 3600 + int(m) * 60 + float(sec)
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", video, "-vf", f"fps={3 * n / dur}", "-q:v", "2",
                    os.path.join(out, "raw_%04d.jpg")], check=True)
    raw = sorted(glob.glob(os.path.join(out, "raw_*.jpg")))
    return [max(raw[i:i + 3], key=sharpness) for i in range(0, len(raw), 3)][:n]


def fit_plane(pts, up, iters=3000, tol=0.01, seed=0):
    """Largest plane whose normal is within ~35 deg of the camera's up direction (desk top or floor, not a wall)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up) < 0.8:
            continue
        inl = np.abs((pts - a) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = (n, a), inl.sum()
    return best


def rotation_to_z(n):
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def load_models(cache_dir=None, device="cuda"):
    from transformers import pipeline
    from vggt.models.vggt import VGGT
    vggt = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir=cache_dir).to(device).eval()
    metric = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", device=device,
                      model_kwargs=dict(cache_dir=cache_dir))
    return vggt, metric


def reconstruct(video, out, vggt, metric, frames=48, conf_pct=65, max_depth=2.5, voxel=0.005, scale_frames=8, ply=True, log=print,
                frame_paths=None):
    """Writes coord.npy, color.npy, views.npz (cameras, metric depth, frame paths) and optionally cloud.ply into `out`.
    `frame_paths` skips frame extraction (an image sequence instead of a video file)."""
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    paths = frame_paths if frame_paths is not None else extract_frames(video, os.path.join(out, "frames"), frames)
    log(f"{len(paths)} frames")
    dev = next(vggt.parameters()).device
    images = load_and_preprocess_images(paths).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = vggt(images)
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extri, intri = extri[0].float().cpu().numpy(), intri[0].float().cpu().numpy()
    depth = pred["depth"][0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth, extri, intri)
    rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    ratios = []
    H, W = depth.shape[1:3]
    for i in np.linspace(0, len(paths) - 1, scale_frames).astype(int):
        m = np.asarray(metric(Image.fromarray(rgb[i]))["predicted_depth"].squeeze().cpu().numpy(), np.float32)
        m = np.asarray(Image.fromarray(m).resize((W, H), Image.BILINEAR))
        d = depth[i, ..., 0]
        ok = (conf[i] > np.percentile(conf[i], 50)) & (d > 1e-4) & (m > 0.1)
        ratios.append(np.median(m[ok] / d[ok]))
    scale = float(np.median(ratios))
    log(f"metric scale {scale:.3f} (per-frame spread {np.std(ratios) / scale * 100:.1f}%)")
    keep = (conf > np.percentile(conf, conf_pct)) & (depth[..., 0] * scale < max_depth) & (depth[..., 0] > 0)
    P, C = pts[keep] * scale, rgb[keep]
    q = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    P, C = P[idx].astype(np.float32), C[idx]
    up = -extri[0, 1, :3]
    n, p0 = fit_plane(P[np.random.default_rng(0).choice(len(P), min(len(P), 200000), replace=False)], up)
    if n @ up < 0:
        n = -n
    R = rotation_to_z(n)
    P = (P - p0) @ R.T
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "views.npz"), extri=extri, intri=intri, depth=(depth[..., 0] * scale).astype(np.float16),
                        R=R, p0=p0, scale=scale, net=images.shape[-1], frames=np.array([os.path.relpath(f, out) for f in paths]))
    np.save(os.path.join(out, "coord.npy"), P.astype(np.float32))
    np.save(os.path.join(out, "color.npy"), C.astype(np.uint8))
    if ply:
        with open(os.path.join(out, "cloud.ply"), "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            np.savetxt(f, np.hstack([P, C]), fmt="%.4f %.4f %.4f %d %d %d")
    ext = P.max(0) - P.min(0)
    log(f"saved {len(P)} points; extent {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")
    return P, C
