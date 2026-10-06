import torch
import torch.nn.functional as F


class ClickSegmenter:
    """Positive + negative clicks on a frozen SNAP: decode positives once, then enforce all clicks on the mask token."""

    def __init__(self, snap, method="ttt", steps=30, lr=0.05, reg=10.0, margin=2.0, radius=0.0):
        self.snap, self.method = snap, method
        self.steps, self.lr, self.reg, self.margin, self.radius = steps, lr, reg, margin, radius

    @torch.no_grad()
    def embed(self, pos):
        """pos: (P, 3) -> refined voxel embeddings E (N, D), mask token m (D,), SNAP logits (N,)."""
        mdl, point, data = self.snap.model, self.snap.point, self.snap.data
        p = pos[None].float()
        with torch.autocast("cuda", dtype=torch.float16):
            enc = mdl.prompt_encoder(p).view(-1, point.feat.size(1))
            idx = torch.cdist(point.coord.float(), p.view(-1, 3)).argmin(0)
            enc = (enc + point.feat[idx]).view(1, len(pos), -1)
            tok = torch.cat([mdl.iou_token.weight, mdl.mask_token.weight, mdl.text_token.weight], 0)[None]
            enc = torch.cat([tok, enc], 1)
            pe = mdl.point_pos_embeddings_module(data["coord"])
            E, T = mdl.merge([enc], [point.feat[None]], [pe[None]])
            m = mdl.mask_head(T[0][:, 1])
        E, m = E[0][0].float(), m[0].float()
        return E, m, E @ m

    def __call__(self, vcoord, pos_idx, neg_idx):
        E, m, logits = self.embed(vcoord[pos_idx])
        if not len(neg_idx) or self.method == "snap":
            return logits > 0
        if self.method == "subtract":
            drop = torch.zeros_like(logits, dtype=torch.bool)
            for j in neg_idx:
                _, _, ln = self.embed(vcoord[torch.as_tensor([j], device=vcoord.device)])
                drop |= (ln > 0) & (ln > logits)
            return (logits > 0) & ~drop
        return self.tune(E, m, vcoord, pos_idx, neg_idx) > 0

    def tune(self, E, m0, vcoord, pos_idx, neg_idx):
        idx = torch.cat([torch.as_tensor(pos_idx, device=E.device), torch.as_tensor(neg_idx, device=E.device)])
        y = torch.cat([torch.ones(len(pos_idx)), torch.zeros(len(neg_idx))]).to(E.device)
        if self.radius > 0:
            d = torch.cdist(vcoord[idx].float(), vcoord.float())
            near, lab = (d < self.radius), y[:, None].expand_as(d)
            w = near.float()
            vox = near.any(0)
            idx = torch.nonzero(vox)[:, 0]
            y = (lab * w).sum(0)[vox] / w.sum(0)[vox]
        Es = E[idx]
        pm = m0.clone().requires_grad_(True)
        b = torch.zeros(1, device=E.device, requires_grad=True)
        opt = torch.optim.Adam([pm, b], lr=self.lr * m0.norm().item() / m0.numel() ** 0.5)
        with torch.enable_grad():
            for _ in range(self.steps):
                z = Es @ pm + b
                loss = F.softplus(-(2 * y - 1) * (z - (2 * y - 1) * self.margin)).mean() + self.reg * ((pm - m0) ** 2).sum() / (m0 ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return (E @ pm + b).detach()


class CorrectiveSegmenter:
    """Trained v5 corrective decoder: positive-only clicks use frozen SNAP exactly; negatives switch to the corrective decoder."""

    def __init__(self, snap, ckpt, tuner=None):
        from .snap_plus import SNAPPlus, load_snap
        state = torch.load(ckpt, map_location="cpu")
        assert state["cfg"]["arch"] == "v5"
        self.model = SNAPPlus(load_snap(state["cfg"]["snap_ckpt"]), num_masks=0, use_sem=False, use_text=False, arch="v5").cuda()
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        assert not unexpected and all(k.startswith("snap.") for k in missing), (missing[:5], unexpected[:5])
        self.model.snap = snap.model
        self.model.eval()
        self.snap, self.prev, self.tuner = snap, None, tuner

    @torch.no_grad()
    def __call__(self, vcoord, pos_idx, neg_idx):
        if len(pos_idx) == 1 and not len(neg_idx):
            self.prev = None
        idx = torch.cat([pos_idx, neg_idx]).long()
        labels = torch.cat([torch.ones(len(pos_idx)), torch.zeros(len(neg_idx))]).long().to(vcoord.device)[None]
        point = self.snap.point
        point.feat = point.feat.float()
        prev = self.prev[None] if self.prev is not None else None
        with torch.autocast("cuda", dtype=torch.float16):
            if self.tuner is not None and len(neg_idx):
                E, m = self.model.decode_v5(point, self.snap.data["coord"], vcoord[idx][None].float(), labels, prev, embed=True)
                E, m = E[0].float(), m[0].float()
            else:
                logits = self.model.decode_v5(point, self.snap.data["coord"], vcoord[idx][None].float(), labels, prev)[0].float()
        if self.tuner is not None and len(neg_idx):
            logits = self.tuner.tune(E, m, vcoord, pos_idx, neg_idx)
        self.prev = logits
        return logits > 0
