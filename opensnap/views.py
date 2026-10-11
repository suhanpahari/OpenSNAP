"""Image evidence column: per-object multi-view crop embeddings from posed RGB-D frames (when a scan has them)."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class PosedFrames:
    """NICE-SLAM style sequence: results/frame%06d.jpg, results/depth%06d.png, traj.txt (camera-to-world, OpenCV axes)."""

    def __init__(self, root, intr, step=10, depth_scale=6553.5):
        self.root, self.step, self.depth_scale = root, step, depth_scale
        poses = np.loadtxt(os.path.join(root, "traj.txt")).reshape(-1, 4, 4)
        self.ids = list(range(0, len(poses), step))
        self.c2w = poses[self.ids].astype(np.float32)
        self.K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float32)
        self.W, self.H = int(intr["w"]), int(intr["h"])

    def color(self, k):
        return Image.open(os.path.join(self.root, "results", f"frame{self.ids[k]:06d}.jpg")).convert("RGB")

    def depth(self, k):
        return np.asarray(Image.open(os.path.join(self.root, "results", f"depth{self.ids[k]:06d}.png")), np.float32) / self.depth_scale

    @torch.no_grad()
    def visibility(self, pts, tol=0.05, device="cuda"):
        """(F, N) bool: point visible (in view, unoccluded) in each frame; plus (F, N, 2) pixel coordinates."""
        P = torch.from_numpy(pts).to(device)
        K = torch.from_numpy(self.K).to(device)
        vis = torch.zeros(len(self.ids), len(pts), dtype=torch.bool, device=device)
        uv = torch.zeros(len(self.ids), len(pts), 2, dtype=torch.float16, device=device)
        for k in range(len(self.ids)):
            w2c = torch.from_numpy(np.linalg.inv(self.c2w[k])).to(device)
            Xc = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = Xc[:, 2]
            p = Xc @ K.T
            u, v = p[:, 0] / z.clamp(min=1e-6), p[:, 1] / z.clamp(min=1e-6)
            ok = (z > 0.1) & (u >= 0) & (u < self.W - 1) & (v >= 0) & (v < self.H - 1)
            d = torch.from_numpy(self.depth(k)).to(device)
            dz = d[v.clamp(0, self.H - 1).long(), u.clamp(0, self.W - 1).long()]
            vis[k] = ok & (dz > 0) & ((z - dz).abs() < tol + 0.02 * z)
            uv[k, :, 0], uv[k, :, 1] = u.half(), v.half()
        return vis, uv


def _tensor(x):
    return x if torch.is_tensor(x) else x.pooler_output


class ImageExpert:
    """SigLIP crop embeddings; same backbone as OVI-MAP (siglip-large-patch16-384), class names as raw text."""

    def __init__(self, name="google/siglip-large-patch16-384", cache_dir=None, device="cuda"):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir).eval().half().to(device)
        self.proc = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
        self.device = device
        self.cache = {}

    @torch.no_grad()
    def text(self, labels):
        key = tuple(labels)
        if key not in self.cache:
            t = self.proc.tokenizer(list(labels), padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            self.cache[key] = F.normalize(_tensor(self.model.get_text_features(**t)).float(), dim=-1)
        return self.cache[key]

    @torch.no_grad()
    def images(self, ims, bs=64):
        out = []
        for i in range(0, len(ims), bs):
            x = self.proc(images=ims[i:i + bs], return_tensors="pt")["pixel_values"].to(self.device).half()
            out.append(F.normalize(_tensor(self.model.get_image_features(pixel_values=x)).float(), dim=-1))
        return torch.cat(out) if out else torch.zeros(0, self.model.config.text_config.hidden_size, device=self.device)

    def probs(self, emb, labels, tau=0.01):
        return (emb @ self.text(labels).T / tau).softmax(-1)


@torch.no_grad()
def object_embeddings(frames, expert, pts, masks, top_k=5, scales=(0.0, 0.1, 0.2), min_vis=20, sub=4):
    """Per object: top-k frames by visible points, multi-scale crops around the visible projection (OpenMask3D-style),
    SigLIP embeddings averaged over crops and views. masks: (M, N) bool over `pts`. Returns (M, D) and #views used."""
    idx = np.arange(0, len(pts), sub)
    vis, uv = frames.visibility(pts[idx])
    mk = torch.from_numpy(masks[:, idx]).to(vis.device)
    counts = vis.half() @ mk.half().T                                   # (F, M)
    crops, owner, nviews = [], [], np.zeros(len(masks), int)
    cache = {}
    for o in range(len(masks)):
        c = counts[:, o].float()
        best = torch.argsort(c, descending=True)[:top_k]
        for k in best.tolist():
            if c[k] < min_vis:
                break
            sel = vis[k] & mk[o]
            p = uv[k, sel].float().cpu().numpy()
            lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
            if k not in cache:
                cache[k] = frames.color(k)
            im = cache[k]
            for s in scales:
                pad = (hi - lo) * s + 4
                box = (max(0, lo[0] - pad[0]), max(0, lo[1] - pad[1]), min(frames.W, hi[0] + pad[0]), min(frames.H, hi[1] + pad[1]))
                if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    continue
                crops.append(im.crop(tuple(int(b) for b in box)))
                owner.append(o)
            nviews[o] += 1
    emb = expert.images(crops)
    D = emb.shape[1] if len(emb) else expert.text(["object"]).shape[1]
    out = torch.zeros(len(masks), D, device=expert.device)
    if len(owner):
        out.index_add_(0, torch.tensor(owner, device=expert.device), emb)
    return F.normalize(out, dim=-1), nviews


class VideoFrames:
    """Same interface as PosedFrames for a video reconstruction (VGGT cameras and metric depth from views.npz)."""

    def __init__(self, views, scale=1.0):
        from .photo import PhotoViews
        self.scale = scale
        self.pv = PhotoViews(views)
        im = Image.open(self.pv.frames[0])
        self.W, self.H = im.size
        self.fx = self.W / self.pv.net
        self.top = max(0, (round(self.H * self.pv.net / self.W / 14) * 14 - self.pv.net) // 2)
        self.ids = list(range(len(self.pv.frames)))

    def color(self, k):
        return Image.open(self.pv.frames[k]).convert("RGB")

    @torch.no_grad()
    def visibility(self, pts, device="cuda"):
        uv, vis = self.pv.project(pts / self.scale)
        uv = (uv + np.array([0, self.top])) * self.fx
        return torch.from_numpy(vis).to(device), torch.from_numpy(uv).half().to(device)
