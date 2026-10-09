"""v6: LoRA adapters in SNAP's frozen PTv3 backbone + corrective decoder, trained jointly on indoor + outdoor scans.

Losses: positive-click masks through SNAP's frozen decoder (1-3 clicks), corrective masks through the v5 decoder, and
feature distillation to the un-adapted backbone to keep SNAP's cross-dataset behaviour.
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap.lora import add_lora, lora_state, set_lora  # noqa: E402
from opensnap.snap_plus import SNAPPlus, load_snap  # noqa: E402
from train import Scenes, mask_loss  # noqa: E402
from train_corrective import Mixed, decode_rows, next_clicks  # noqa: E402


def backbone(model, d):
    point, _ = model.snap.run_backbone(model.snap.run_input_encoder(copy.copy(d)))
    point.feat = point.feat.float()
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet,nuscenes,semantickitti")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--init", default="../downloads/checkpoints/opensnap_corrective_v5_multi.pth")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--objects", type=int, default=12)
    ap.add_argument("--max_clicks", type=int, default=10)
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="../downloads/runs/adapt_v6")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    model = SNAPPlus(load_snap(a.snap_ckpt), num_masks=0, use_sem=False, use_text=False, arch="v5").to(dev)
    if a.init:
        missing, unexpected = model.load_state_dict(torch.load(a.init, map_location="cpu")["model"], strict=False)
        assert not unexpected and all(k.startswith("snap.") for k in missing)
    for p in model.parameters():
        p.requires_grad = False
    loras = add_lora(model.snap.backbone, a.rank)
    params = [p for m in loras for p in (m.A, m.B)]
    corr = [model.click_type, model.prev_embed, *model.corr_merge.parameters(), *model.corr_mask_head.parameters()]
    for p in params + corr:
        p.requires_grad = True
    print("lora params", sum(p.numel() for p in params), "corrective params", sum(p.numel() for p in corr), flush=True)
    opt = torch.optim.AdamW([dict(params=params, lr=a.lr), dict(params=corr, lr=a.lr / 3)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr / 3], total_steps=a.steps, pct_start=0.05)
    ds = Mixed(a.dataset.split(","), a.max_points) if "," in a.dataset else Scenes(a.dataset, a.max_points, loop=1000)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers, collate_fn=lambda b: b[0], persistent_workers=True,
                        worker_init_fn=lambda w: (random.seed(a.seed * 100 + w), np.random.seed(a.seed * 100 + w)))
    it, t0, log = iter(loader), time.time(), {"pos": [], "corr": [], "distill": []}
    scaler = torch.cuda.amp.GradScaler()
    model.eval()
    for step in range(a.steps):
        d = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in next(it).items()}
        coord, ins = d["coord"], d["instance"]
        ids, counts = torch.unique(ins[ins >= 0], return_counts=True)
        ids = ids[counts >= (50 if d["domain"] == "Indoor" else 10)]
        if not len(ids):
            continue
        ids = ids[torch.randperm(len(ids), device=dev)[:a.objects]]
        target = ins[None] == ids[:, None]
        with torch.no_grad():
            set_lora(loras, False)
            feat0 = backbone(model, d).feat
            set_lora(loras, True)
        point = backbone(model, d)
        distill = ((point.feat - feat0) ** 2).mean() / (feat0 ** 2).mean()
        clicks = torch.stack([coord[torch.multinomial(t.float(), 1)[0]] for t in target])[:, None]
        labels = torch.ones(len(ids), 1, dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            npos = random.randint(1, 3)
            pc = clicks
            with torch.no_grad():
                for _ in range(npos - 1):
                    lg = model.decode_v5(point, coord, pc, torch.ones(pc.shape[:2], dtype=torch.long, device=dev)).float()
                    err = target & ~(lg > 0)
                    src = torch.where(err.any(1, keepdim=True), err, target)
                    pc = torch.cat([pc, coord[torch.multinomial(src.float(), 1)[:, 0]][:, None]], 1)
            lp = model.decode_v5(point, coord, pc, torch.ones(pc.shape[:2], dtype=torch.long, device=dev)).float()
            l_pos = mask_loss(lp[:, None], target)[:, 0].mean()
            prev = None
            n = random.randint(2, a.max_clicks)
            with torch.no_grad():
                for _ in range(n - 1):
                    prev = decode_rows(model, point, coord, clicks, labels, prev)
                    c, l = next_clicks(coord, prev, target)
                    clicks, labels = torch.cat([clicks, c[:, None]], 1), torch.cat([labels, l[:, None]], 1)
            neg = ~labels.all(1)
            l_corr = mask_loss(model.decode_v5(point, coord, clicks[neg], labels[neg], prev[neg], force=True).float()[:, None],
                               target[neg])[:, 0].mean() if neg.any() else torch.zeros((), device=dev)
        loss = l_pos + l_corr + a.distill * distill
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params + corr, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        log["pos"].append(float(l_pos)); log["corr"].append(float(l_corr)); log["distill"].append(float(distill))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{a.steps} pos {np.mean(log['pos'][-50:]):.4f} corr {np.mean(log['corr'][-50:]):.4f} "
                  f"distill {np.mean(log['distill'][-50:]):.5f} {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % 1000 == 0 or step + 1 == a.steps:
            cfg = dict(snap_ckpt=os.path.abspath(a.snap_ckpt), openscene_ckpt="", feature="openseg", sem_dim=768, num_masks=0,
                       use_sem=False, use_text=False, arch="v5", lora_rank=a.rank, dataset=a.dataset)
            state = {k: v for k, v in model.state_dict().items() if not k.startswith("snap.")}
            torch.save(dict(model=state, lora=lora_state(model.snap.backbone), cfg=cfg, step=step + 1),
                       os.path.join(a.out, "last.pth"))
    json.dump(dict(args=vars(a), **log), open(os.path.join(a.out, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
