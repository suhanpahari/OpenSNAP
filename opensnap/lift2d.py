"""2D-lifted proposals: SAM masks of posed frames projected onto the scan and associated across frames into 3D segments.
They complement the promptable segmenter where geometry alone cannot separate an object (windows, pictures, vents)."""
import os

import numpy as np
import torch

from .core import Proposals, mask_nms


def load_masks(path):
    z = np.load(path)
    shape = tuple(z["shape"])
    if shape[0] == 0:
        return np.zeros(shape, bool)
    return np.unpackbits(z["bits"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)


@torch.no_grad()
def lift(frames, sam_dir, coord, inverse, nvox, sub=2, min_vox=30, contain=0.5, min_views=2):
    """Returns a list of (voxel ids, number of views). frames.ids must match the cached SAM frame ids."""
    idx = np.arange(0, len(coord), sub)
    vis, uv = frames.visibility(coord[idx])
    vis, uv = vis.cpu().numpy(), uv.float().cpu().numpy()
    inv = inverse[idx]
    label = np.full(nvox, -1, np.int64)
    segs, views = [], []
    for k, fid in enumerate(frames.ids):
        f = os.path.join(sam_dir, f"{fid:06d}.npz")
        if not os.path.exists(f):
            continue
        M = load_masks(f)
        p = np.nonzero(vis[k])[0]
        if len(p) == 0 or len(M) == 0:
            continue
        u = np.clip(uv[k, p, 0].round().astype(int), 0, M.shape[2] - 1)
        v = np.clip(uv[k, p, 1].round().astype(int), 0, M.shape[1] - 1)
        order = np.argsort([-m.sum() for m in M])
        for j in order:
            S = np.unique(inv[p[M[j][v, u]]])
            if len(S) < min_vox:
                continue
            lab = label[S]
            known = lab[lab >= 0]
            if len(known):
                ids, cnt = np.unique(known, return_counts=True)
                best = int(ids[cnt.argmax()])
                if cnt.max() / len(S) >= contain:
                    segs[best] = np.union1d(segs[best], S)
                    views[best] += 1
                    label[S[lab < 0]] = best
                    continue
            new = S[lab < 0]
            if len(new) < min_vox:
                continue
            segs.append(new)
            views.append(1)
            label[new] = len(segs) - 1
    return [(s, n) for s, n in zip(segs, views) if n >= min_views and len(s) >= min_vox]


def merge_proposals(props, lifted, nvox, dedup=0.5, score=0.4):
    """Append lifted segments that no existing proposal already covers (IoU <= dedup)."""
    if not lifted:
        return props
    dev = props.logits.device
    L = torch.full((len(lifted), nvox), -4.0, device=dev)
    for k, (s, _) in enumerate(lifted):
        L[k, torch.as_tensor(s, device=dev)] = 4.0
    A, B = (props.logits > 0).float(), (L > 0).float()
    inter = B @ A.T
    iou = inter / (B.sum(1, keepdim=True) + A.sum(1)[None] - inter).clamp(min=1)
    keep = (iou.max(1).values <= dedup) if len(A) else torch.ones(len(B), dtype=torch.bool, device=dev)
    L = L[keep]
    if len(L) == 0:
        return props
    sc = torch.tensor([score * min(n, 5) / 5 + score for (s, n), k in zip(lifted, keep.tolist()) if k], device=dev).clamp(max=0.9)
    idx = mask_nms(L > 0, sc, 0.5)
    L, sc = L[idx], sc[idx]
    tok = torch.zeros(len(L), props.tokens.shape[1], device=dev)
    pr = torch.zeros(len(L), 1, 3, device=dev)
    return Proposals(torch.cat([props.logits, L]), torch.cat([props.tokens, tok]), torch.cat([props.scores, sc]),
                     torch.cat([props.prompts[:, :1], pr]))
