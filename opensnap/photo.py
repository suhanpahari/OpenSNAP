import os

import numpy as np
from PIL import Image, ImageDraw


class PhotoViews:
    """Source frames of a video reconstruction: project a 3D mask into the frames that see it best and return real
    photo crops with the object outlined, so the VLM names a photo instead of a point-cloud render."""

    def __init__(self, path):
        v = np.load(path, allow_pickle=True)
        self.extri, self.intri, self.depth = v["extri"], v["intri"], v["depth"]
        self.R, self.p0, self.scale = v["R"], v["p0"], float(v["scale"])
        self.frames, self.net = [str(f) for f in v["frames"]], int(v["net"])
        root = os.path.dirname(path)
        self.frames = [f if os.path.isabs(f) else os.path.join(root, f) for f in self.frames]

    def project(self, P):
        """Aligned scene points -> per-frame pixel coords in the network image and a visibility mask."""
        Xs = P @ self.R + self.p0
        Xc = np.einsum("sij,nj->sni", self.extri[:, :, :3], Xs) + self.extri[:, None, :, 3] * self.scale
        z = Xc[..., 2]
        uv = np.einsum("sij,snj->sni", self.intri, Xc)[..., :2] / np.maximum(z[..., None], 1e-6)
        S, H, W = self.depth.shape
        u, v = np.round(uv[..., 0]).astype(int), np.round(uv[..., 1]).astype(int)
        inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = self.depth[np.arange(S)[:, None], v.clip(0, H - 1), u.clip(0, W - 1)]
        vis = inb & (np.abs(z - d) < 0.05 + 0.05 * z)
        return uv, vis

    def crops(self, P, n=2, max_points=4000, margin=0.6, min_side=200):
        if len(P) == 0:
            return []
        if len(P) > max_points:
            P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
        uv, vis = self.project(P)
        S, H, W = self.depth.shape
        frac = vis.mean(1)
        cen = np.array([np.abs(uv[s, vis[s]] / [W, H] - 0.5).max() if vis[s].any() else 1.0 for s in range(S)])
        score = frac * (1.2 - cen)
        out = []
        for s in np.argsort(-score)[:n]:
            if frac[s] < 0.2:
                break
            im = Image.open(self.frames[s]).convert("RGB")
            fx = im.width / self.net
            top = max(0, (round(im.height * self.net / im.width / 14) * 14 - self.net) // 2)
            pts = (uv[s, vis[s]] + [0, top]) * fx
            lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
            c, half = (lo + hi) / 2, max((hi - lo).max() * (0.5 + margin), min_side / 2)
            box = [int(max(0, c[0] - half)), int(max(0, c[1] - half)), int(min(im.width, c[0] + half)), int(min(im.height, c[1] + half))]
            dr = ImageDraw.Draw(im)
            dr.rectangle([int(lo[0]) - 6, int(lo[1]) - 6, int(hi[0]) + 6, int(hi[1]) + 6], outline=(255, 0, 0), width=max(3, im.width // 240))
            out.append(im.crop(box).resize((448, 448)))
        return out
