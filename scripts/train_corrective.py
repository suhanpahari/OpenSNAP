"""Train the v5 corrective-click decoder on a frozen SNAP with simulated positive/negative corrective clicking."""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import glob
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap.snap_plus import SNAPPlus, load_snap  # noqa: E402
from opensnap import paths  # noqa: E402
from train import Compose, Scenes, mask_loss  # noqa: E402


def outdoor_transform(max_points):
    return Compose([
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02),
        dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
             keys=("coord", "color", "normal", "strength", "instance", "segment"), return_grid_coord=True),
        dict(type="SphereCrop", point_max=max_points, mode="random"),
        dict(type="Add", keys_dict={"domain": "Outdoor", "condition": "SemanticKITTI"}),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "normal", "strength", "instance", "segment", "domain", "condition")),
    ])


class Mixed(Dataset):
    """Equal-probability mix of indoor ScanNet and outdoor LiDAR scans (nuScenes / SemanticKITTI train)."""

    def __init__(self, names, max_points, length=10 ** 7):
        self.indoor = Scenes("scannet", max_points) if "scannet" in names else None
        self.outdoor = {n: sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", "pointcept", n, "train", "*")))
                        for n in names if n != "scannet"}
        self.names, self.t, self.length = names, outdoor_transform(max_points), length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        n = random.choice(self.names)
        if n == "scannet":
            return self.indoor[random.randrange(len(self.indoor.files))]
        f = random.choice(self.outdoor[n])
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        z = np.zeros_like(coord)
        d = dict(coord=coord, color=z, normal=z.copy(), strength=np.zeros((len(coord), 1), np.float32),
                 instance=np.load(f"{f}/instance.npy").astype(np.int64), segment=np.load(f"{f}/segment.npy").astype(np.int64))
        return self.t(d)


def decode_rows(model, point, coord, clicks, labels, prev):
    """Per-row routing: positive-only rows -> frozen SNAP, rows with a negative click -> corrective decoder."""
    out = torch.empty(len(clicks), len(coord), device=coord.device)
    pos = labels.all(1)
    if pos.any():
        out[pos] = model.decode_v5(point, coord, clicks[pos], labels[pos]).float()
    if (~pos).any():
        out[~pos] = model.decode_v5(point, coord, clicks[~pos], labels[~pos], prev[~pos] if prev is not None else None,
                                    force=True).float()
    return out


def next_clicks(coord, logits, target):
    pred = logits > 0
    fn, fp = target & ~pred, pred & ~target
    use_fn = fn.sum(1) >= fp.sum(1)
    region = torch.where(use_fn[:, None], fn, fp)
    region = torch.where(region.any(1, keepdim=True), region, target)
    pick = torch.multinomial(region.float(), 1)[:, 0]
    return coord[pick], use_fn.long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--objects", type=int, default=12)
    ap.add_argument("--max_clicks", type=int, default=10)
    ap.add_argument("--max_points", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="../downloads/runs/corr_v5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    model = SNAPPlus(load_snap(a.snap_ckpt), num_masks=0, use_sem=False, use_text=False, arch="v5").to(dev)
    for p in model.parameters():
        p.requires_grad = False
    params = [model.click_type, model.prev_embed, *model.corr_merge.parameters(), *model.corr_mask_head.parameters()]
    for p in params:
        p.requires_grad = True
    print("trainable params", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    ds = Mixed(a.dataset.split(","), a.max_points) if "," in a.dataset else Scenes(a.dataset, a.max_points, loop=1000)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers, collate_fn=lambda b: b[0], persistent_workers=True,
                        worker_init_fn=lambda w: (random.seed(a.seed * 100 + w), np.random.seed(a.seed * 100 + w)))
    it, t0, log, gain = iter(loader), time.time(), [], []
    model.eval()
    for step in range(a.steps):
        d = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in next(it).items()}
        coord, ins = d["coord"], d["instance"]
        with torch.no_grad():
            point, _ = model.snap.run_backbone(model.snap.run_input_encoder(d))
        point.feat = point.feat.float()
        ids, counts = torch.unique(ins[ins >= 0], return_counts=True)
        ids = ids[counts >= (50 if d["domain"] == "Indoor" else 10)]
        if not len(ids):
            continue
        ids = ids[torch.randperm(len(ids), device=dev)[:a.objects]]
        target = ins[None] == ids[:, None]
        clicks = torch.stack([coord[torch.multinomial(t.float(), 1)[0]] for t in target])[:, None]
        labels = torch.ones(len(ids), 1, dtype=torch.long, device=dev)
        prev = None
        n = random.randint(2, a.max_clicks)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(n - 1):
                prev = decode_rows(model, point, coord, clicks, labels, prev)
                c, l = next_clicks(coord, prev, target)
                clicks, labels = torch.cat([clicks, c[:, None]], 1), torch.cat([labels, l[:, None]], 1)
        neg = ~labels.all(1)
        if not neg.any():
            continue
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.decode_v5(point, coord, clicks[neg], labels[neg], prev[neg], force=True).float()
            loss = mask_loss(logits[:, None], target[neg])[:, 0].mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            t = target[neg]
            iou = lambda z: ((z > 0) & t).sum(1).float() / ((z > 0) | t).sum(1).clamp(min=1).float()
            gain.append(float((iou(logits) - iou(prev[neg])).mean()))
        log.append(float(loss))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{a.steps} loss {np.mean(log[-50:]):.4f} iou_gain_vs_prev {np.mean(gain[-50:]):+.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if (step + 1) % 2000 == 0 or step + 1 == a.steps:
            cfg = dict(snap_ckpt=os.path.abspath(a.snap_ckpt), openscene_ckpt="", feature="openseg", sem_dim=768, num_masks=0,
                       use_sem=False, use_text=False, arch="v5", dataset=a.dataset)
            torch.save(dict(model={k: v for k, v in model.state_dict().items() if not k.startswith("snap.")}, cfg=cfg,
                            step=step + 1), os.path.join(a.out, "last.pth"))
    json.dump(dict(args=vars(a), loss=log, gain=gain), open(os.path.join(a.out, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
