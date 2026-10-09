"""Figures for a video-reconstructed scene: frame strip, point-cloud views, click -> mask + photo crop + name, viewer captures."""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)


def caption(im, text, sub=""):
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width, 30), fill=(20, 20, 20))
    d.text((8, 5), text, fill="white", font=FONT)
    if sub:
        d.rectangle((0, im.height - 24, im.width, im.height), fill=(20, 20, 20))
        d.text((8, im.height - 21), sub, fill=(210, 210, 210), font=SMALL)
    return im


def grid(tiles, cols):
    w, h = tiles[0].size
    g = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)), "white")
    for i, t in enumerate(tiles):
        g.paste(t, ((i % cols) * w, (i // cols) * h))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="../downloads/demo/lab_desk")
    ap.add_argument("--capture", default="../downloads/demo/lab_desk_capture")
    ap.add_argument("--out", default="assets/demo_lab_desk")
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import prior_vocab, snap_train_vocab
    from opensnap.io import load_any
    from opensnap.namer import VLMNamer
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.photo import PhotoViews
    from opensnap.render import render_object
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder

    photos = PhotoViews(os.path.join(a.scene, "views.npz"))
    frames = [Image.open(f).convert("RGB").resize((300, 300)) for f in photos.frames[:: max(1, len(photos.frames) // 8)][:8]]
    grid(frames, 8).save(os.path.join(a.out, "frames.jpg"), quality=85)

    d = load_any(a.scene)
    coord, color = d["coord"], d["color"]
    i = np.random.default_rng(0).choice(len(coord), min(len(coord), 500000), replace=False)
    fig = plt.figure(figsize=(21, 7))
    for k, (el, az) in enumerate([(85, -90), (30, -60), (30, 35)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(*coord[i].T, c=color[i] / 255, s=0.5, linewidths=0)
        ax.view_init(el, az); ax.set_box_aspect(np.ptp(coord[i], 0)); ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "cloud.jpg"), dpi=70, pil_kwargs=dict(quality=85)); plt.close()

    dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
    m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), TextEncoder(), dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    np.random.seed(0); torch.manual_seed(0)
    m.encode(coord)
    props = m.propose()
    namer = VLMNamer(a.vlm)
    V = prior_vocab("snap+lvis")
    tiles, log = [], []
    for idx in props.scores.argsort(descending=True).tolist():
        mask = (props.logits[idx] > 0)[m.scene["inverse"]].cpu().numpy()
        if not 3000 <= mask.sum() <= 150000:
            continue
        crops = photos.crops(coord[mask], n=1)
        if not crops:
            continue
        best, ranked, reply = namer.name(m, props, idx, coord, color, prior_vocab=V, photos=photos)
        r = render_object(coord, color, mask, n_views=1, size=360)[0]
        tiles += [caption(r, f"3D mask → {best}"), caption(crops[0].resize((360, 360)), "what the VLM sees", "alts: " + ", ".join(n for n, _ in ranked[1:3]))]
        log.append(f"{best} | VLM: {reply.strip()}")
        if len(tiles) >= 2 * a.objects:
            break
    grid(tiles, 4).save(os.path.join(a.out, "click_naming.jpg"), quality=85)
    open(os.path.join(a.out, "click_naming.txt"), "w").write("\n".join(log) + "\n")
    print("\n".join(log))

    crop = lambda p: Image.open(p).convert("RGB").crop((480, 150, 1120, 790)).resize((480, 480))
    caps = [("3_segments", "segment everything (no names)"), ("2_labels", "open-vocab labels: your class list"),
            ("q5_office_chair", "query: office chair"), ("q2_keyboard", "query: keyboard (miss: lands on screens)"), ("q3_printer", "query: printer (miss: lands on monitor)"),
            ("q0_water_bottle", "query: water bottle (miss)")]
    shots = [caption(crop(os.path.join(a.capture, f"{f}.png")), c) for f, c in caps if os.path.exists(os.path.join(a.capture, f"{f}.png"))]
    if shots:
        grid(shots, 3).save(os.path.join(a.out, "viewer.jpg"), quality=85)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
