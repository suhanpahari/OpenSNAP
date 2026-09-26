"""Drive the viser app and save screenshots of each feature (needs `playwright install chromium`)."""
import argparse
import os
import sys
import time
import types

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as A  # noqa: E402


def random_clicks(app, n, seed, min_pts=800, max_pts=60000):
    """Clicks on randomly chosen SNAP segments (no cherry-picking)."""
    props = app.ensure_props()
    inv = app.m.scene["inverse"].cpu().numpy()
    vc = app.m.scene["vcoord"].cpu().numpy()
    sizes = (props.logits > 0)[:, app.m.scene["inverse"]].sum(1).cpu().numpy()
    ok = [i for i in range(len(sizes)) if min_pts <= sizes[i] <= max_pts]
    pick = np.random.default_rng(seed).choice(ok, size=min(n, len(ok)), replace=False)
    out = []
    for i in pick:
        j = int(np.argmin(np.linalg.norm(vc - props.prompts[i, 0].cpu().numpy(), axis=1)))
        out.append(int(np.nonzero(inv == j)[0][0]))
    print("random clicks:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="../SNAP/data_examples/ScanNet/scene0011_00")
    ap.add_argument("--clicks", nargs="*", type=int, default=[])
    ap.add_argument("--query", default="something to sit on")
    ap.add_argument("--classes", default="wall, floor, chair, table, cabinet, door, window, picture, other")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only_select", action="store_true")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--prior", default="snap+lvis")
    ap.add_argument("--random_objects", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/scannet_lseg.pth.tar")
    ap.add_argument("--feature", default="lseg")
    a = ap.parse_args()
    args = types.SimpleNamespace(scene=a.scene, domain="Indoor", snap_ckpt="../downloads/ckpt/SNAP_C.pth",
                                 openscene_ckpt=a.openscene_ckpt, feature=a.feature, vlm=a.vlm, prior=a.prior,
                                 alpha=0.7, normals=0, point_size=0.015, host="127.0.0.1", port=a.port, demo=False)
    app = A.App(args)
    if a.random_objects:
        a.clicks = random_clicks(app, a.random_objects, a.seed)
    os.makedirs(a.out, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1600, "height": 950})
        pg.goto(f"http://127.0.0.1:{a.port}")
        time.sleep(15)

        def shot(name):
            time.sleep(8)
            pg.screenshot(path=os.path.join(a.out, f"{name}.png"))
            print(name, "|", app.status.content)

        def select(idx):
            app.new_object()
            app.clicks.append(idx)
            app.update_current()

        shot(f"{a.tag}0_scene")
        for n, idx in enumerate(a.clicks):
            select(idx)
            shot(f"{a.tag}1_select_{n}")
        if a.only_select:
            b.close()
            return
        app.clear()
        app.segment_everything()
        shot("2_segments")
        if a.clicks:
            select(a.clicks[0])
            shot("2_segments_select")
        app.clear()
        app.query_box.value = a.query
        app.find()
        shot("3_text_query")
        app.vocab_box.value = a.classes
        app.label_scene()
        shot("4_open_vocab_labels")
        b.close()


if __name__ == "__main__":
    main()
