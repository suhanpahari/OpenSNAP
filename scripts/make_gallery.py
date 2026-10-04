"""Compose per-scene open-vocabulary galleries from viewer captures (every query shown; ★ = word absent from SNAP's vocabulary)."""
import glob
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap.data import snap_train_vocab  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, DST = os.path.join(ROOT, "assets", "gallery_src2"), os.path.join(ROOT, "assets", "openvocab_gallery")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
CROP = (0.20, 0.10, 0.78, 0.97)


def tile(path, caption, sub="", w=448, h=376):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    im = im.crop((int(CROP[0] * W), int(CROP[1] * H), int(CROP[2] * W), int(CROP[3] * H))).resize((w, h))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 32), fill=(25, 25, 25))
    d.text((8, 6), caption, fill="white", font=FONT)
    if sub:
        d.rectangle((0, h - 26, w, h), fill=(25, 25, 25))
        d.text((8, h - 23), sub, fill=(200, 200, 200), font=SMALL)
    return im


def grid(tiles, cols=5):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (w * cols, h * rows), "white")
    for i, t in enumerate(tiles):
        g.paste(t, ((i % cols) * w, (i // cols) * h))
    return g


def statuses(log):
    out = {}
    for line in open(log) if os.path.exists(log) else []:
        m = re.match(r"(\S+) \| (.*)", line.strip())
        if m:
            out[m.group(1)] = re.sub(r"\*\*", "", m.group(2))
    return out


def main():
    vocab = set(snap_train_vocab())
    queries = json.load(open(os.path.join(ROOT, "..", "downloads", "demo", "queries.json")))
    st = {**statuses(os.path.join(ROOT, "..", "downloads", "demo", "lane2_gallery_a.log")),
          **statuses(os.path.join(ROOT, "..", "downloads", "demo", "lane2_gallery_b.log"))}
    os.makedirs(DST, exist_ok=True)
    made = []
    for scene, qs in queries.items():
        tiles = []
        for i, q in enumerate(qs):
            p = os.path.join(SRC, f"{scene}_q{i}_{q.replace(' ', '_')}.png")
            if os.path.exists(p):
                tiles.append(tile(p, f'"{q}"' + ("  ★" if q not in vocab else "")))
        clicks = sorted(glob.glob(os.path.join(SRC, f"{scene}_1_select_*.png")))
        for p in clicks:
            key = os.path.basename(p)[:-4]
            s = st.get(key, "")
            name = re.search(r"selected: ([^·]+)", s)
            vlm = re.search(r"VLM: “([^”]+)”", s)
            tiles.append(tile(p, f"click → {name.group(1).strip() if name else '?'}", f"VLM: {vlm.group(1)}" if vlm else ""))
        for suffix, cap in (("2_labels", "open-vocab labels (scene's query words)"), ("3_segments", "segment everything")):
            p = os.path.join(SRC, f"{scene}_{suffix}.png")
            if os.path.exists(p):
                tiles.append(tile(p, cap))
        if tiles:
            out = os.path.join(DST, f"{scene}.jpg")
            grid(tiles).save(out, quality=82, optimize=True, progressive=True)
            made.append((scene, len(tiles), out))
    for m in made:
        print(*m)


if __name__ == "__main__":
    main()
