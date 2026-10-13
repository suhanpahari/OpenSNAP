"""Caption-free PNG elements for the overview figure, one per pipeline stage, all on ScanNet scene0011_00 from one camera
(transparent background, cropped to content) -> assets/overview/."""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
from opensnap.render import look_at  # noqa: E402

SIZE, FOCAL = 1600, 1600 * 0.95
GREY = 0.72


class Cam:
    def __init__(self, coord):
        ctr = coord.mean(0)
        ext = np.ptp(coord[:, :2], 0).max()
        self.eye = ctr + np.array([-0.55 * ext, -0.75 * ext, 1.15 * ext])
        self.R = look_at(self.eye, ctr)

    def project(self, xyz):
        cam = (np.atleast_2d(xyz) - self.eye) @ self.R.T
        z = cam[:, 2]
        return np.stack([FOCAL * cam[:, 0] / z + SIZE / 2, FOCAL * cam[:, 1] / z + SIZE / 2], 1), z


def splat(cam, xyz, rgb, radius=2):
    uv, z = cam.project(xyz)
    ok = z > 0.05
    u, v, z, rgb = uv[ok, 0].astype(np.int32), uv[ok, 1].astype(np.int32), z[ok], rgb[ok]
    img = np.zeros((SIZE, SIZE, 4), np.float32)
    order = np.argsort(-z)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u[order] + dx, v[order] + dy
            m = (uu >= 0) & (uu < SIZE) & (vv >= 0) & (vv < SIZE)
            img[vv[m], uu[m], :3] = rgb[order][m]
            img[vv[m], uu[m], 3] = 1
    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8), "RGBA")


NORMAL = None


def base(color, shade=True):
    if shade:
        g = color.astype(np.float32).mean(1, keepdims=True) / 255
        return np.repeat(0.55 + 0.35 * g, 3, 1)
    light = np.array([-0.35, -0.55, 0.76])
    g = np.abs(NORMAL @ (light / np.linalg.norm(light)))[:, None]
    return np.repeat(0.45 + 0.45 * g, 3, 1).astype(np.float32)


def turbo(q):
    from matplotlib import colormaps
    return np.asarray(colormaps["turbo"](0.08 + 0.9 * np.asarray(q)))[..., :3]


def heat_rgb(color, masks, p):
    rgb = base(color)
    q = np.asarray(p, np.float64) / max(float(np.max(p)), 1e-12)
    for i in np.argsort(q):
        if q[i] >= 0.02:
            a = min(1.0, 0.35 + 0.65 * q[i])
            rgb[masks[i]] = (1 - a) * rgb[masks[i]] + a * turbo(q[i])
    return rgb


def palette(n, seed=3):
    import colorsys
    rng = np.random.default_rng(seed)
    h = rng.permutation(n) / max(n, 1)
    return np.array([colorsys.hsv_to_rgb(x, 0.65 + 0.3 * rng.random(), 0.8 + 0.2 * rng.random()) for x in h], np.float32)


def box_edges(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return c, [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def draw_box(im, cam, pts, rgb, width=3):
    d = ImageDraw.Draw(im)
    c, E = box_edges(pts.min(0), pts.max(0))
    uv, _ = cam.project(c)
    for a, b in E:
        d.line([tuple(uv[a]), tuple(uv[b])], fill=tuple(int(255 * x) for x in rgb) + (255,), width=width)


def trim(im, pad=12):
    x0, y0, x1, y1 = im.getchannel("A").getbbox()
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


class Out:
    def __init__(self, root):
        self.root, self.box = root, None
        os.makedirs(root, exist_ok=True)

    def scene(self, name, im):
        if self.box is None:
            x0, y0, x1, y1 = im.getchannel("A").getbbox()
            self.box = (max(0, x0 - 12), max(0, y0 - 12), min(SIZE, x1 + 12), min(SIZE, y1 + 12))
        im.crop(self.box).save(os.path.join(self.root, name))
        print(name, flush=True)

    def plain(self, name, im):
        im.save(os.path.join(self.root, name))
        print(name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="../SNAP/data_examples/ScanNet/scene0011_00")
    ap.add_argument("--frames", default=os.path.join(paths.DOWNLOADS, "data", "scannet_2d", "scene0011_00"))
    ap.add_argument("--out", default="assets/overview")
    ap.add_argument("--subject", default="chair")
    ap.add_argument("--reference", default="table")
    ap.add_argument("--crack", nargs="+", default=["picture", "lamp", "trash can", "radiator", "backpack", "bag", "shelf"])
    a = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    from opensnap.views import ScanNetFrames

    global NORMAL
    coord = np.load(f"{a.scene}/coord.npy").astype(np.float32)
    color = np.load(f"{a.scene}/color.npy")
    NORMAL = np.load(f"{a.scene}/normal.npy").astype(np.float32)
    cam, out = Cam(coord), Out(a.out)
    rgb_true = color.astype(np.float32) / 255

    out.scene("01_scan_xyz.png", splat(cam, coord, base(color, shade=False)))
    out.scene("02_scan_rgb.png", splat(cam, coord, rgb_true))

    frames = ScanNetFrames(a.frames)
    im = splat(cam, coord, base(color, shade=False))
    d = ImageDraw.Draw(im)
    W, H, K = frames.W, frames.H, frames.K
    for n, k in enumerate(range(0, len(frames.ids), 3)):
        c2w, s = frames.c2w[k], 0.35
        corners = [(0, 0), (W, 0), (W, H), (0, H)]
        rays = np.array([[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0] for u, v in corners]) * s
        pts = np.concatenate([c2w[:3, 3][None], rays @ c2w[:3, :3].T + c2w[:3, 3]])
        uv, _ = cam.project(pts)
        col = tuple(int(255 * x) for x in turbo(k / len(frames.ids))) + (255,)
        for i in range(4):
            d.line([tuple(uv[0]), tuple(uv[1 + i])], fill=col, width=3)
            d.line([tuple(uv[1 + i]), tuple(uv[1 + (i + 1) % 4])], fill=col, width=3)
    out.scene("04_frames_poses.png", im)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    dense = OpenSceneModel(os.path.join(paths.DOWNLOADS, "ckpt", "scannet_openseg.pth.tar"), "openseg")
    m = OpenSNAP(SnapModel(os.path.join(paths.DOWNLOADS, "ckpt", "SNAP_C.pth")), TextEncoder(), dense, TextEncoder(dense.clip_name),
                 Config(), anchors=snap_train_vocab())
    np.random.seed(0); torch.manual_seed(0)
    m.encode(coord)
    inv = m.scene["inverse"].cpu().numpy()
    vc = m.scene["vcoord"].cpu().numpy()
    X = np.c_[vc[inv], np.ones(len(inv))]
    fit = np.linalg.lstsq(X, coord, rcond=None)[0]
    world = lambda x: np.c_[np.atleast_2d(x), np.ones(len(np.atleast_2d(x)))] @ fit

    f = m.scene["dense"].float()
    f = f - f.mean(0)
    _, _, V = torch.pca_lowrank(f, q=3, center=False)
    x = (f @ V).cpu().numpy()
    lo, hi = np.percentile(x, 2, 0), np.percentile(x, 98, 0)
    out.scene("05_encode_dense.png", splat(cam, coord, ((x - lo) / (hi - lo)).clip(0, 1)[inv]))

    props = m.propose_union(levels=4)
    im = splat(cam, coord, base(color, shade=False))
    d = ImageDraw.Draw(im)
    uv, _ = cam.project(world(props.prompts[:, 0].cpu().numpy()))
    for u, v in uv:
        d.ellipse((u - 7, v - 7, u + 7, v + 7), fill=(230, 40, 40, 255), outline=(255, 255, 255, 255), width=2)
    out.scene("06_encode_prompts.png", im)

    table = materialize(m, props)
    order = np.argsort([-len(p) for p in table.points])
    masks = [np.isin(inv, table.points[i]) for i in range(len(table))]
    pal = palette(len(table))
    rgb = base(color, shade=False)
    for r, i in enumerate(order):
        rgb[masks[i]] = pal[r]
    out.scene("07_proposals.png", splat(cam, coord, rgb))

    eng = Engine(m, table, refine=False)
    p_sub = eng.is_prob(a.subject).cpu().numpy()
    o = int(p_sub.argmax())
    p_ref = eng.is_prob(a.reference).cpu().numpy()
    focus = np.zeros(len(coord), bool)
    for i in np.nonzero((p_sub > 0.5) | (p_ref > 0.5))[0]:
        focus |= masks[i]
    fv, _ = frames.visibility(coord[np.nonzero(focus)[0][::4]])
    score = fv.sum(1).cpu().numpy()
    pick = []
    for k in np.argsort(-score):
        if all(abs(int(k) - j) >= 6 for j in pick):
            pick.append(int(k))
        if len(pick) == 4:
            break
    for n, k in enumerate(sorted(pick)):
        out.plain(f"03_frame_{n + 1}.png", frames.color(k).resize((960, 720), Image.LANCZOS))
    pts_mask = masks[o][None]
    idx = np.arange(0, len(coord), 4)
    vis, uvf = frames.visibility(coord[idx])
    mk = torch.from_numpy(pts_mask[:, idx]).to(vis.device)
    cnt = (vis & mk[0]).sum(1).float()
    for n, k in enumerate(torch.argsort(cnt, descending=True)[:3].tolist()):
        p = uvf[k, vis[k] & mk[0]].float().cpu().numpy()
        lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
        pad = (hi - lo) * 0.15 + 4
        box = (max(0, lo[0] - pad[0]), max(0, lo[1] - pad[1]), min(W, hi[0] + pad[0]), min(H, hi[1] + pad[1]))
        crop = frames.color(k).crop(tuple(int(b) for b in box))
        s = 480 / max(crop.size)
        out.plain(f"08_image_crop_{n + 1}.png", crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS))

    labels = ["wall", "floor", "chair", "table", "cabinet", "door", "window", "picture", "other"]
    lab = m.mask_probs(props, labels).argmax(1).cpu().numpy()
    conf = table.conf.cpu().numpy()
    big = sorted((i for i in order if 2 <= lab[i] <= 7 and len(table.points[i]) < 0.03 * len(vc)), key=lambda i: -conf[i])[:20]
    rgb = base(color, shade=False)
    for r, i in enumerate(order):
        rgb[masks[i]] = 0.55 * pal[r] + 0.45 * rgb[masks[i]]
    im = splat(cam, coord, rgb)
    for i in big:
        r = int(np.where(order == i)[0][0])
        draw_box(im, cam, coord[masks[i]], pal[r] * 0.8, width=3)
    out.scene("09_materialized.png", im)

    out.scene("10_query_heat_is.png", splat(cam, coord, heat_rgb(color, masks, p_sub)))
    res, _ = eng.run(f"SELECT WHERE IS('{a.subject}') AND NEAR(IS('{a.reference}'), 0.5)")
    p_join = np.zeros(len(table))
    for i, v in res:
        p_join[i] = v
    ref = int(p_ref.argmax())
    rgb = heat_rgb(color, masks, p_join)
    rgb[masks[ref]] = 0.5 * rgb[masks[ref]] + 0.5 * np.array([0.2, 0.45, 0.95])
    im = splat(cam, coord, rgb)
    draw_box(im, cam, coord[masks[ref]], (0.15, 0.35, 0.9), width=4)
    out.scene("11_query_join.png", im)
    d = ImageDraw.Draw(im)
    for r, (i, v) in enumerate(res[:5]):
        u, w = cam.project(coord[masks[i]].mean(0))[0][0]
        d.ellipse((u - 22, w - 22, u + 22, w + 22), fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=3)
        d.text((u, w), str(r + 1), fill=(20, 20, 20, 255), anchor="mm", font_size=28)
    out.scene("11_query_join_ranked.png", im)

    for c in a.crack:
        before = eng.is_prob(c).cpu().numpy()
        if before.max() >= eng.refine_below:
            print("explained already:", c, float(before.max()), flush=True)
            continue
        eng2 = Engine(m, table, refine=True)
        n0 = len(table)
        eng2.run(f"SELECT WHERE IS('{c}')")
        oid = table.cracked.get(c)
        if oid is None or oid < n0:
            print("not cracked:", c, oid, flush=True)
            continue
        print("cracked:", c, "max p before", float(before.max()), "provenance", table.provenance[oid], flush=True)
        out.scene("12_refine_before.png", splat(cam, coord, heat_rgb(color, masks, before)))
        new = np.isin(inv, table.points[oid])
        rgb = heat_rgb(color, masks, before)
        rgb[new] = np.array([0.95, 0.15, 0.6])
        im = splat(cam, coord, rgb)
        draw_box(im, cam, coord[new], (0.95, 0.15, 0.6), width=4)
        out.scene("13_refine_after.png", im)
        with open(os.path.join(a.out, "concepts.txt"), "w") as fh:
            fh.write(f"subject={a.subject} reference={a.reference} cracked={c} provenance={table.provenance[oid]}\n")
        break

    sem = m.semantic(props, labels).argmax(1).cpu().numpy()
    pal9 = np.array([[0.85, 0.2, 0.35], [0.25, 0.7, 0.3], [0.98, 0.85, 0.1], [0.1, 0.5, 0.8], [0.95, 0.55, 0.15],
                     [0.55, 0.25, 0.7], [0.3, 0.9, 0.9], [0.95, 0.4, 0.75], [0.6, 0.6, 0.6]], np.float32)
    out.scene("14_labels.png", splat(cam, coord, pal9[sem]))


if __name__ == "__main__":
    main()
