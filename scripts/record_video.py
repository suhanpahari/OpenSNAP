"""Record a demo video of the viewer: orbit, click-to-name, text queries (needs `playwright install chromium` and ffmpeg)."""
import argparse
import glob
import os
import subprocess
import sys
import time
import types

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as A  # noqa: E402

HIDE = ".mantine-Notifications-root, .mantine-Notification-root { display: none !important; }"


def set_camera(app, pos, look):
    for c in app.server.get_clients().values():
        c.camera.position = tuple(pos)
        c.camera.look_at = tuple(look)


def orbit(app, seconds, start=0.0, sweep=90.0, fps=15):
    lo, hi = app.coord.min(0), app.coord.max(0)
    mid, ext = (lo + hi) / 2, hi - lo
    r, h = 0.6 * ext[:2].max(), mid[2] + 0.4 * ext[:2].max() + ext[2]
    for k in range(int(seconds * fps)):
        a = np.radians(start + sweep * k / (seconds * fps) - 90)
        set_camera(app, (mid[0] + r * np.cos(a), mid[1] + r * np.sin(a), h), mid)
        time.sleep(1 / fps)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clicks", nargs="*", type=int, default=[])
    ap.add_argument("--queries", nargs="*", default=[])
    ap.add_argument("--out", default="assets/demo")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--text_ckpt", default="../downloads/checkpoints/opensnap_text_v4.pth")
    a = ap.parse_args()
    args = types.SimpleNamespace(scene=a.scene, domain="Indoor", snap_ckpt="../downloads/ckpt/SNAP_C.pth",
                                 openscene_ckpt=a.openscene_ckpt, feature=a.feature, vlm=a.vlm, prior="snap+lvis",
                                 alpha=0.7, text_ckpt=a.text_ckpt, normals=0, point_size=0.012, host="127.0.0.1", port=a.port, demo=False)
    app = A.App(args)
    app.ensure_props()
    tmp = a.out + "_raw"
    os.makedirs(tmp, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = b.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=tmp,
                            record_video_size={"width": 1280, "height": 720})
        pg = ctx.new_page()
        pg.goto(f"http://127.0.0.1:{a.port}")
        pg.add_style_tag(content=HIDE)
        time.sleep(12)
        orbit(app, 6, sweep=60)
        for idx in a.clicks:
            app.new_object()
            app.clicks.append(idx)
            app.update_current()
            time.sleep(3)
        app.remove_objects()
        for q in a.queries:
            app.query_box.value = q
            app.find()
            time.sleep(3.5)
        app.remove_objects()
        app.draw_base()
        orbit(app, 4, start=60, sweep=40)
        pg.close()
        ctx.close()
        b.close()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    webm = max(glob.glob(os.path.join(tmp, "*.webm")), key=os.path.getmtime)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", webm, "-ss", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-movflags", "+faststart", a.out + ".mp4"], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", a.out + ".mp4", "-vf",
                    "fps=8,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                    a.out + ".gif"], check=True)
    print("saved", a.out + ".mp4", a.out + ".gif")


if __name__ == "__main__":
    main()
