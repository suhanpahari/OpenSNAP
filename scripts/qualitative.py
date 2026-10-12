"""Qualitative figure: relational queries answered by OVQL vs the phrase-embedding query style of OpenMask3D / Search3D /
OVI-MAP (emulated on the same stored objects and evidence columns). Examples are top queries by measured AP50 margin
in results/compare/agg_{replica,s3dis}.json that also win in >= 4 of 5 seeds (assets/compare/robust_examples*.json);
the overall win/loss counts go in the caption."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
from opensnap.render import look_at, splat  # noqa: E402
from eval_queries import S3DIS, SKIP, classes_for, gt_objects, relational_gt  # noqa: E402

EXAMPLES = [("replica", "room1", "nightstand", "pillow", "c2f"), ("replica", "office4", "window", "chair", "c2f"),
            ("s3dis", "conferenceRoom_2", "sofa", "chair", "fps"), ("s3dis", "office_1", "window", "chair", "fps")]
ROOT = {"replica": "pointcept/replica/all", "s3dis": "pointcept/s3dis/Area_5"}
NEAR = 0.5


def render(coord, color, layers, size=520):
    """Oblique view of the room (ceiling cut), grey context, coloured layers [(mask, rgb)] drawn on top."""
    keep = coord[:, 2] < np.percentile(coord[:, 2], 99) - 0.35
    c, col = coord[keep], color[keep].astype(np.float32) / 255
    rgb = 0.35 + 0.45 * col.mean(1, keepdims=True).repeat(3, 1)
    for layer in layers:
        m, tint = layer[0][keep], np.asarray(layer[1], np.float32)
        a = layer[2] if len(layer) > 2 else 1.0
        rgb[m] = (1 - a) * rgb[m] + a * tint
    ctr = c.mean(0)
    ext = np.ptp(c[:, :2], 0).max()
    eye = ctr + np.array([0.12 * ext, -0.38 * ext, 1.25 * ext])
    img, hit = splat(c, rgb, look_at(eye, ctr), eye, size, focal=size * 0.95, radius=2)
    img[~hit] = 1.0
    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))


def label(im, text, sub=""):
    W, H = im.size
    out = Image.new("RGB", (W, H + 58), "white")
    out.paste(im, (0, 58))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        g = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        f = g = ImageFont.load_default()
    d.text((10, 6), text, fill=(20, 20, 20), font=f)
    d.text((10, 32), sub, fill=(90, 90, 90), font=g)
    return out


def main():
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    rows, record = [], []
    for ds, scene, a, b, prop in EXAMPLES:
        f = os.path.join(paths.DOWNLOADS, "data", ROOT[ds], scene)
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        color = np.load(f"{f}/color.npy")
        ins = np.load(f"{f}/instance.npy").reshape(-1).astype(np.int64)
        seg = np.load(f"{f}/segment.npy").reshape(-1).astype(np.int64)
        np.random.seed(0); torch.manual_seed(0)
        m.encode(coord)
        table = materialize(m, m.propose_c2f() if prop == "c2f" else None)
        vc, inv = m.scene["vcoord"].cpu().numpy(), m.scene["inverse"].cpu().numpy()
        classes = classes_for(ds)
        objs = [(c, mk, (vc[mk].min(0), vc[mk].max(0))) for c, mk in gt_objects(ins, seg, inv, len(vc), classes, SKIP[ds])]
        gt_subj = relational_gt(objs, a, b, NEAR)
        gt_ref = [k for k, o in enumerate(objs) if o[0] == b]
        eng = Engine(m, table, refine=False)
        ovql, _ = eng.run(f"SELECT WHERE IS('{a}') AND NEAR(IS('{b}'), {NEAR})")
        phrase, _ = eng.run(f"SELECT WHERE IS('{a} near a {b}')")
        to_pts = lambda vox: np.isin(inv, vox)
        top_o, top_p = ovql[0][0], phrase[0][0]
        pb = eng.is_prob(b)
        rel = eng.relation("NEAR", NEAR)
        rel.fill_diagonal_(0)
        wit = int((rel[top_o] * pb).argmax())
        best_iou = lambda i: max([((np.isin(np.arange(len(vc)), table.points[i]) & objs[g][1]).sum() /
                                   max((np.isin(np.arange(len(vc)), table.points[i]) | objs[g][1]).sum(), 1)) for g in gt_subj] or [0])
        io, ip = best_iou(top_o), best_iou(top_p)
        gtm = np.zeros(len(coord), bool)
        for g in gt_subj:
            gtm |= objs[g][1][inv]
        refm = np.zeros(len(coord), bool)
        for g in gt_ref:
            refm |= objs[g][1][inv]
        green, blue, red = (0.15, 0.75, 0.25), (0.2, 0.45, 0.95), (0.92, 0.15, 0.15)
        p1 = label(render(coord, color, [(refm, blue), (gtm, green)]), f"{scene}: '{a}' near '{b}'", "ground truth: target (green), reference (blue)")
        p2 = label(render(coord, color, [(to_pts(table.points[top_p]), red)]), "phrase embedding (competitors' query style)",
                   f"IS('{a} near a {b}')  top-1 IoU {ip:.2f}  {'correct' if ip >= 0.5 else 'wrong'}")
        p3 = label(render(coord, color, [(to_pts(table.points[wit]), blue), (to_pts(table.points[top_o]), red)]), "OVQL (ours)",
                   f"IS('{a}') AND NEAR(IS('{b}'), 0.5)  top-1 IoU {io:.2f}  {'correct' if io >= 0.5 else 'wrong'}")
        row = Image.new("RGB", (p1.width * 3 + 20, p1.height), "white")
        for k, p in enumerate((p1, p2, p3)):
            row.paste(p, (k * (p1.width + 10), 0))
        rows.append(row)
        record.append(dict(dataset=ds, scene=scene, a=a, b=b, ovql_top1_iou=float(io), phrase_top1_iou=float(ip),
                           ovql_score=float(ovql[0][1]), phrase_score=float(phrase[0][1])))
        print(record[-1], flush=True)
    fig = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 10 * len(rows)), "white")
    y = 0
    for r in rows:
        fig.paste(r, (0, y))
        y += r.height + 10
    os.makedirs("assets/compare", exist_ok=True)
    fig.save("assets/compare/relational_queries.jpg", quality=92)
    json.dump(record, open("assets/compare/relational_queries.json", "w"), indent=1)


if __name__ == "__main__":
    main()
