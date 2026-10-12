"""Publication figure: GT | Ours | OpenMask3D* | Search3D* | OVI-MAP* on relational queries (Replica, posed frames).
* Each prior's query interface is emulated on the same stored objects with its published image encoder: the phrase
"a near a b" is embedded and the most similar object returned (OpenMask3D: CLIP ViT-L/14@336 crops; Search3D: SigLIP
so400m crops; OVI-MAP: SigLIP-L/16-384 crops from selected views). Ours: IS(a) AND NEAR(IS(b), 0.5).
Rows: candidate queries where OVQL won by AP50 margin, kept if ours is correct and every prior is wrong."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
from qualitative import render  # noqa: E402
from eval_queries import SKIP, classes_for, gt_objects, relational_gt  # noqa: E402

def candidates():
    d = json.load(open("results/compare/agg_replica.json"))
    return sorted({(r["scene"], q["a"], q["b"]) for r in d["records"] for q in r["results"]["opensnap"]["queries"] if q["kind"] == "NEAR"})
PRIORS = [("OpenMask3D*", "clip"), ("Search3D*", "google/siglip-so400m-patch14-384"), ("OVI-MAP*", "google/siglip-large-patch16-384")]
GREEN, BLUE, RED = (0.15, 0.75, 0.25), (0.2, 0.45, 0.95), (0.92, 0.15, 0.15)


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def centred(d, x0, w, y, text, f, fill=(20, 20, 20)):
    tw = d.textlength(text, font=f)
    d.text((x0 + (w - tw) / 2, y), text, font=f, fill=fill)


def title(s):
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def main():
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    from opensnap.views import CLIPExpert, ImageExpert, PosedFrames, object_embeddings
    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    hub = os.path.join(paths.DOWNLOADS, "hf", "hub")
    experts = {k: (CLIPExpert(download_root=os.path.join(paths.DOWNLOADS, "clip")) if k == "clip" else ImageExpert(k, cache_dir=hub))
               for _, k in PRIORS}
    nice = os.path.join(paths.DOWNLOADS, "replica_nice", "Replica")
    intr = json.load(open(os.path.join(nice, "cam_params.json")))["camera"]
    results, panels = [], []
    cache = {}
    for scene, a, b in candidates():
        f = os.path.join(paths.DOWNLOADS, "data", "pointcept", "replica", "all", scene)
        coord = np.load(f"{f}/coord.npy").astype(np.float32)
        color = np.load(f"{f}/color.npy")
        ins = np.load(f"{f}/instance.npy").reshape(-1).astype(np.int64)
        seg = np.load(f"{f}/segment.npy").reshape(-1).astype(np.int64)
        if scene not in cache:
            np.random.seed(0); torch.manual_seed(0)
            m.encode(coord)
            cache.clear()
            cache[scene] = dict(table=materialize(m, m.propose_c2f()), scene_state=m.scene)
        m.scene = cache[scene]["scene_state"]
        table = cache[scene]["table"]
        vc, inv = m.scene["vcoord"].cpu().numpy(), m.scene["inverse"].cpu().numpy()
        objs = [(c, mk, (vc[mk].min(0), vc[mk].max(0))) for c, mk in gt_objects(ins, seg, inv, len(vc), classes_for("replica"), SKIP["replica"])]
        gs, gr = relational_gt(objs, a, b, 0.5), [k for k, o in enumerate(objs) if o[0] == b]
        vox = [np.isin(np.arange(len(vc)), p) for p in table.points]
        iou = lambda i: max([(vox[i] & objs[g][1]).sum() / max((vox[i] | objs[g][1]).sum(), 1) for g in gs] or [0])
        eng = Engine(m, table, refine=False)
        top_o = eng.run(f"SELECT WHERE IS('{a}') AND NEAR(IS('{b}'), 0.5)")[0][0][0]
        rel = eng.relation("NEAR", 0.5)
        rel.fill_diagonal_(0)
        wit = int((rel[top_o] * eng.is_prob(b)).argmax())
        if "emb" not in cache[scene]:
            frames = PosedFrames(os.path.join(nice, scene), intr, step=10)
            pts_masks = np.stack([v[inv] for v in vox])
            cache[scene]["emb"] = {k: object_embeddings(frames, experts[k], coord, pts_masks)[0] for _, k in PRIORS}
        picks = {name: int((cache[scene]["emb"][k] @ experts[k].text([f"{a} near a {b}"]).T)[:, 0].argmax()) for name, k in PRIORS}
        rec = dict(scene=scene, a=a, b=b, ours=float(iou(top_o)), **{n: float(iou(i)) for n, i in picks.items()})
        results.append(rec)
        print(rec, flush=True)
        if rec["ours"] >= 0.5 and all(rec[n] < 0.5 for n, _ in PRIORS):
            to_pts = lambda i: vox[i][inv]
            gtm = np.zeros(len(coord), bool)
            for g in gs:
                gtm |= objs[g][1][inv]
            refm = np.zeros(len(coord), bool)
            for g in gr:
                refm |= objs[g][1][inv]
            ims = [render(coord, color, [(refm, BLUE), (gtm, GREEN)]), render(coord, color, [(to_pts(wit), BLUE), (to_pts(top_o), RED)])]
            ims += [render(coord, color, [(to_pts(picks[n]), RED)]) for n, _ in PRIORS]
            notes = ["Target And Reference", f"Correct (IoU {rec['ours']:.2f})"] + [f"Wrong (IoU {rec[n]:.2f})" for n, _ in PRIORS]
            panels.append((f"Query: {title(a)} Near {title(b)}  ({title(scene)})", ims, notes))
    json.dump(results, open("assets/compare/priors_relational.json", "w"), indent=1)
    win = sum(r["ours"] >= 0.5 and all(r[n] < 0.5 for n, _ in PRIORS) for r in results)
    lose = sum(r["ours"] < 0.5 and any(r[n] >= 0.5 for n, _ in PRIORS) for r in results)
    print(f"queries {len(results)}: ours correct and every prior wrong {win}; ours wrong and some prior correct {lose}", flush=True)
    panels = panels[:3]
    if not panels:
        return
    W, H, pad, top, lab = panels[0][1][0].width, panels[0][1][0].height, 8, 46, 34
    heads = ["GT", "Ours"] + [n for n, _ in PRIORS]
    rowh = lab + H + 30
    fig = Image.new("RGB", (5 * W + 6 * pad, top + len(panels) * rowh + 70), "white")
    d = ImageDraw.Draw(fig)
    for c, h in enumerate(heads):
        centred(d, pad + c * (W + pad), W, 10, h, font(24, True))
    for r, (rt, ims, notes) in enumerate(panels):
        y = top + r * rowh
        centred(d, 0, fig.width, y + 6, rt, font(19, True), fill=(60, 60, 60))
        for c, (im, nt) in enumerate(zip(ims, notes)):
            x = pad + c * (W + pad)
            fig.paste(im, (x, y + lab))
            centred(d, x, W, y + lab + H + 4, nt, font(16), fill=(30, 120, 40) if nt.startswith("Correct") else (60, 60, 60) if c == 0 else (170, 30, 30))
    foot = ("Green: Target Objects; Blue: Reference Objects (Ours: The Witness); Red: Top-1 Answer.  * Query Interface Emulated On The Same "
            "Stored Objects With Each Method's Published Image Encoder.")
    centred(d, 0, fig.width, fig.height - 50, foot, font(15), fill=(70, 70, 70))
    fig.save("assets/compare/relational_priors.jpg", quality=92)


if __name__ == "__main__":
    main()
