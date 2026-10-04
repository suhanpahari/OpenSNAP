"""Visualise building-level OVQL: top hits of each query across all rooms of the S3DIS Area 5 catalog."""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="../downloads/catalogs/s3dis_area5")
    ap.add_argument("--split", default="pointcept/s3dis/Area_5")
    ap.add_argument("--queries", nargs="+", default=["sofa", "whiteboard", "bookcase", "couch", "blackboard", "pillar"])
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="assets/building_hits.png")
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.query.catalog import concat, load
    from opensnap.query.engine import Engine
    from opensnap.render import render_object
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    snap = SnapModel("../downloads/ckpt/SNAP_C.pth")
    m = OpenSNAP(snap, TextEncoder(), None, TextEncoder("ViT-L/14@336px"), Config(), anchors=snap_train_vocab())
    files = sorted(glob.glob(os.path.join(a.catalog, "*.pt")))
    rooms = [os.path.basename(f)[:-3] for f in files]
    tables = [load(f) for f in files]
    eng = Engine(m, concat(tables), refine=False)
    offsets = np.cumsum([0] + [len(t) for t in tables])
    cache = {}
    rows = []
    for q in a.queries:
        res, st = eng.run(f"SELECT WHERE IS('{q}') LIMIT {a.top}")
        tiles = []
        for oid, p in res:
            r = int(np.searchsorted(offsets, oid, side="right") - 1)
            if r not in cache:
                f = os.path.join(paths.DOWNLOADS, "data", a.split, rooms[r])
                coord = np.load(f"{f}/coord.npy").astype(np.float32)
                color = np.load(f"{f}/color.npy").astype(np.float32)
                np.random.seed(0); torch.manual_seed(0)
                _, inv = snap.encode(coord)
                cache[r] = (coord, color, inv.cpu().numpy())
            coord, color, inv = cache[r]
            mask = np.isin(inv, tables[r].points[oid - offsets[r]])
            im = render_object(coord, color, mask, n_views=1, size=360)[0]
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 360, 30), fill=(25, 25, 25))
            d.text((8, 4), f"{rooms[r]}  P={p:.2f}", fill="white", font=FONT)
            tiles.append(im)
        label = Image.new("RGB", (260, 360), (245, 245, 245))
        ImageDraw.Draw(label).text((12, 160), f"IS('{q}')\n{st['ms']:.1f} ms", fill="black", font=FONT)
        row = Image.new("RGB", (260 + 360 * a.top, 360), "white")
        row.paste(label, (0, 0))
        for i, t in enumerate(tiles):
            row.paste(t, (260 + 360 * i, 0))
        rows.append(row)
        print(q, [(rooms[int(np.searchsorted(offsets, o, side='right') - 1)], round(p, 2)) for o, p in res])
    g = Image.new("RGB", (rows[0].width, 360 * len(rows)), "white")
    for i, r in enumerate(rows):
        g.paste(r, (0, 360 * i))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    g.save(a.out, optimize=True)
    print("saved", a.out, g.size)


if __name__ == "__main__":
    main()
