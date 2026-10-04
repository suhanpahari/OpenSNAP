"""Decoder fine-tuning of SNAP (frozen backbone): baseline or SNAP+ (semantic adapter, multi-mask, text prompts)."""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
from opensnap.snap_plus import SNAPPlus, load_snap  # noqa: E402

paths.add(paths.SNAP_DIR)
from datasets.transforms import Compose  # noqa: E402

CLASSES = {
    "s3dis": ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"],
}
SPLITS = {"s3dis": ("pointcept/s3dis", ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]),
          "scannet": ("pointcept/scannet", ["train"])}
SEGMENT_FILE = {"s3dis": "segment", "scannet": "segment200"}


def scannet200_classes():
    paths.add(paths.SNAP_DIR)
    from datasets.demo import DemoDatset
    d = DemoDatset.__new__(DemoDatset)
    DemoDatset.__init__(d, domain="Indoor")
    return list(d.labels_scannet)


def default_holdout(dataset):
    if dataset == "scannet":
        return json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab", "scannet200_splits.json")))["tail"]
    return ["sofa", "bookcase", "board"]
TEMPLATES = ["{}", "a {}", "segment {}", "the {} in the room", "a 3d scan of a {}"]


def train_transform(max_points):
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
        dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02),
        dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
        dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
        dict(type="ChromaticJitter", p=0.95, std=0.05),
        dict(type="DropDataKey", key="color", prob=1.0),
        dict(type="DropDataKey", key="normal", prob=1.0),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
             keys=("coord", "color", "normal", "instance", "segment"), return_grid_coord=True),
        dict(type="SphereCrop", point_max=max_points, mode="random"),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="Add", keys_dict={"domain": "Indoor"}),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "normal", "instance", "segment", "domain")),
    ])


class Scenes(Dataset):
    def __init__(self, dataset, max_points=100000, loop=1):
        root, splits = SPLITS[dataset]
        self.seg = SEGMENT_FILE[dataset]
        self.files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))
        self.t = train_transform(max_points)
        self.loop = loop

    def __len__(self):
        return len(self.files) * self.loop

    def __getitem__(self, i):
        f = self.files[i % len(self.files)]
        d = {k: np.load(os.path.join(f, f"{k}.npy")) for k in ("coord", "color", "normal", "instance")}
        d["segment"] = np.load(os.path.join(f, f"{self.seg}.npy"))
        d["coord"] = d["coord"].astype(np.float32)
        d["color"] = d["color"].astype(np.float32)
        d["normal"] = d["normal"].astype(np.float32)
        d["instance"] = d["instance"].reshape(-1).astype(np.int64)
        d["segment"] = d["segment"].reshape(-1).astype(np.int64)
        return self.t(d)


def mask_loss(logits, target):
    """logits (M, K, N), target (M, N) -> per (M, K) focal*20 + dice."""
    t = target[:, None].float().expand_as(logits)
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    focal = (ce * (1 - pt) ** 2 * (0.25 * t + 0.75 * (1 - t))).mean(-1)
    dice = 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)
    return 20 * focal + dice


def mask_iou(logits, target):
    m = logits > 0
    t = target[:, None].expand_as(m)
    return (m & t).sum(-1).float() / (m | t).sum(-1).clamp(min=1).float()


def objective(model, point, coord, target, clicks=None, text=None, label_emb=None, label_idx=None):
    logits, iou, tok = model.decode(point, coord, clicks=clicks, text=text)
    logits = logits.float()
    lm = mask_loss(logits, target)
    if model.arch == "v1":
        loss = lm.min(1).values.mean()
    elif model.arch == "v3":
        single = clicks is None or clicks.shape[1] == 1
        if not single:
            return None, logits
        loss = lm.min(1).values.mean()
    else:
        single = clicks is None or clicks.shape[1] == 1
        loss = lm[:, 0].mean() + (lm[:, 1:].min(1).values.mean() if single and lm.shape[1] > 1 else 0)
    loss = loss + F.mse_loss(iou.float(), mask_iou(logits, target).detach())
    if label_emb is not None:
        loss = loss + F.cross_entropy(F.normalize(tok.float(), dim=-1) @ label_emb.T, label_idx)
    return loss, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s3dis")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--use_sem", type=int, default=1)
    ap.add_argument("--use_text", type=int, default=1)
    ap.add_argument("--num_masks", type=int, default=3)
    ap.add_argument("--arch", default="v3", choices=["v1", "v2", "v3"])
    ap.add_argument("--holdout", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--objects", type=int, default=12)
    ap.add_argument("--text_classes", type=int, default=6)
    ap.add_argument("--max_clicks", type=int, default=5)
    ap.add_argument("--max_points", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="../downloads/runs/plus")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.arch == "v3":
        args.max_clicks = 1
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")

    model = SNAPPlus(load_snap(args.snap_ckpt), sem_dim=768 if args.feature == "openseg" else 512,
                     num_masks=args.num_masks, use_sem=bool(args.use_sem), use_text=bool(args.use_text), arch=args.arch).to(dev)
    frozen = model.snap.parameters() if args.arch == "v3" else \
        list(model.snap.backbone.parameters()) + list(model.snap.input_encoder.parameters())
    for p in frozen:
        p.requires_grad = False
    if args.arch == "v3":
        for p in model.snap.text_prompt_encoder.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()

    sem_model = None
    if args.use_sem:
        from opensnap.openscene_model import OpenSceneModel
        sem_model = OpenSceneModel(args.openscene_ckpt, args.feature)
    from opensnap.text import TextEncoder
    clip_text = TextEncoder("ViT-B/32")
    classes = CLASSES[args.dataset] if args.dataset in CLASSES else scannet200_classes()
    args.holdout = default_holdout(args.dataset) if args.holdout is None else args.holdout
    assert all(h in classes for h in args.holdout), [h for h in args.holdout if h not in classes]
    label_emb = clip_text(classes, ("segment {}",))
    text_ok = [i for i, c in enumerate(classes) if c not in set(args.holdout)]

    loader = DataLoader(Scenes(args.dataset, args.max_points, loop=1000), batch_size=1, shuffle=True,
                        num_workers=args.workers, collate_fn=lambda b: b[0], persistent_workers=True)
    it, t0, log = iter(loader), time.time(), []
    model.train()
    model.snap.backbone.eval()
    for step in range(args.steps):
        d = next(it)
        d = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in d.items()}
        coord, ins, seg = d["coord"], d["instance"], d["segment"]
        with torch.no_grad():
            sem = sem_model(coord.cpu().numpy()) if sem_model is not None else None
            point, _ = model.snap.run_backbone(model.snap.run_input_encoder(d))
        point.feat = point.feat.float()
        point = model.add_semantics(point, sem)
        ids, counts = torch.unique(ins[ins >= 0], return_counts=True)
        ids = ids[counts >= 50]
        loss = torch.zeros((), device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            if len(ids):
                ids = ids[torch.randperm(len(ids), device=dev)[:args.objects]]
                target = ins[None] == ids[:, None]
                n = random.randint(1, args.max_clicks)
                clicks = torch.stack([coord[torch.multinomial(t.float(), 1)[0]] for t in target])[:, None]
                for _ in range(n - 1):
                    with torch.no_grad():
                        lg, iou, _ = model.decode(point, coord, clicks=clicks)
                        pred = model.select(lg, iou, clicks.shape[1]) > 0
                    err = target & ~pred
                    src = torch.where(err.any(1, keepdim=True), err, target)
                    new = coord[torch.multinomial(src.float(), 1)[:, 0]]
                    clicks = torch.cat([clicks, new[:, None]], 1)
                lab = torch.stack([torch.bincount(seg[t].clamp(min=0), minlength=len(classes)).argmax() for t in target])
                keep = torch.stack([(seg[t] >= 0).any() for t in target])
                target, clicks, lab = target[keep], clicks[keep], lab[keep]
                l_click, _ = objective(model, point, coord, target, clicks=clicks.float(),
                                       label_emb=None if args.arch == "v3" else label_emb, label_idx=lab)
                if l_click is not None:
                    loss = loss + l_click
            if args.use_text:
                present = [c for c in torch.unique(seg).tolist() if c in text_ok and (seg == c).sum() >= 50]
                random.shuffle(present)
                present = present[:args.text_classes]
                if present:
                    emb = torch.cat([clip_text([classes[c]], (random.choice(TEMPLATES),)) for c in present])
                    target = seg[None] == torch.tensor(present, device=dev)[:, None]
                    l_text, _ = objective(model, point, coord, target, text=emb.float())
                    loss = loss + l_text
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        log.append(float(loss))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{args.steps} loss {np.mean(log[-50:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % 2000 == 0 or step + 1 == args.steps:
            cfg = dict(snap_ckpt=os.path.abspath(args.snap_ckpt), openscene_ckpt=os.path.abspath(args.openscene_ckpt),
                       feature=args.feature, sem_dim=768 if args.feature == "openseg" else 512, num_masks=args.num_masks,
                       use_sem=bool(args.use_sem), use_text=bool(args.use_text), classes=classes, holdout=args.holdout,
                       arch=args.arch)
            torch.save(dict(model={k: v for k, v in model.state_dict().items() if not k.startswith("snap.backbone")},
                            cfg=cfg, step=step + 1), os.path.join(args.out, "last.pth"))
    json.dump(dict(args=vars(args), loss=log), open(os.path.join(args.out, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
