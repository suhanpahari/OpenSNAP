import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


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

    def lift(self, s, x, y, win=6):
        """Pixel (x, y) of original frame s -> 3D point in the aligned scene frame (via VGGT depth and camera)."""
        im = Image.open(self.frames[s])
        fx = im.width / self.net
        top = max(0, (round(im.height * self.net / im.width / 14) * 14 - self.net) // 2)
        S, H, W = self.depth.shape
        u, v = int(np.clip(x / fx, 0, W - 1)), int(np.clip(y / fx - top, 0, H - 1))
        d = self.depth[s, max(0, v - win):v + win + 1, max(0, u - win):u + win + 1].astype(np.float32)
        d = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        Xc = d * np.linalg.inv(self.intri[s]) @ np.array([u, v, 1.0])
        Re, te = self.extri[s, :, :3], self.extri[s, :, 3] * self.scale
        Xs = Re.T @ (Xc - te)
        return (Xs - self.p0) @ self.R.T

    def overlay(self, s, P, color=(255, 40, 40), max_points=6000):
        """Original frame s with the visible projection of 3D points P painted on it."""
        im = Image.open(self.frames[s]).convert("RGB")
        if len(P):
            if len(P) > max_points:
                P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
            uv, vis = self.project(P)
            fx = im.width / self.net
            top = max(0, (round(im.height * self.net / im.width / 14) * 14 - self.net) // 2)
            pts = (uv[s, vis[s]] + [0, top]) * fx
            dr = ImageDraw.Draw(im)
            r = max(2, im.width // 300)
            for x, y in pts:
                dr.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return im

    @staticmethod
    def highlight(im, pts, dim=0.6):
        """Keep the object's projected pixels bright with a red outline; dim everything else."""
        r = max(4, im.width // 110)
        m = Image.new("L", im.size, 0)
        dr = ImageDraw.Draw(m)
        for x, y in pts:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=255)
        k = 2 * (r // 2) + 1
        m = m.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
        edge = ImageChops.subtract(m.filter(ImageFilter.MaxFilter(max(3, 2 * (im.width // 400) + 1))), m)
        out = Image.composite(im, Image.eval(im, lambda v: int(v * dim)), m)
        out.paste((255, 30, 30), mask=edge)
        return out

    def crops(self, P, n=2, max_points=4000, margin=0.6, min_side=200, first=None, box=True, pair=False):
        """Photo crops of the frames that see P best; `first` forces a frame (e.g. the one the user clicked) to lead."""
        if len(P) == 0:
            return []
        if len(P) > max_points:
            P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
        uv, vis = self.project(P)
        S, H, W = self.depth.shape
        frac = vis.mean(1)
        cen = np.array([np.abs(uv[s, vis[s]] / [W, H] - 0.5).max() if vis[s].any() else 1.0 for s in range(S)])
        order = [s for s in np.argsort(-(frac * (1.2 - cen))) if frac[s] >= 0.2]
        if first is not None and vis[first].sum() >= 5:
            order = [first] + [s for s in order if s != first]
        out = []
        for s in order[:n]:
            im = Image.open(self.frames[s]).convert("RGB")
            fx = im.width / self.net
            top = max(0, (round(im.height * self.net / im.width / 14) * 14 - self.net) // 2)
            pts = (uv[s, vis[s]] + [0, top]) * fx
            lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
            c, half = (lo + hi) / 2, max((hi - lo).max() * (0.5 + margin), min_side / 2)
            crop = [int(max(0, c[0] - half)), int(max(0, c[1] - half)), int(min(im.width, c[0] + half)), int(min(im.height, c[1] + half))]
            if box and pair:
                boxed = im.copy()
                ImageDraw.Draw(boxed).rectangle([int(lo[0]) - 6, int(lo[1]) - 6, int(hi[0]) + 6, int(hi[1]) + 6], outline=(255, 0, 0),
                                                width=max(4, im.width // 160))
                boxed.thumbnail((448, 448))
                out.append(boxed)
            if box:
                im = self.highlight(im, pts)
            out.append(im.crop(crop).resize((448, 448)))
        return out
