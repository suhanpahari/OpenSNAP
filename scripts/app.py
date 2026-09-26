"""OpenSNAP interactive viewer: click an object -> SNAP mask + open-vocabulary name."""
import argparse
import os
import sys
import threading
import time

import numpy as np
import torch
import viser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap.core import Config, OpenSNAP  # noqa: E402
from opensnap.data import estimate_normals, snap_train_vocab  # noqa: E402
from opensnap.io import load_any  # noqa: E402
from opensnap.snap_model import SnapModel  # noqa: E402
from opensnap.text import TextEncoder  # noqa: E402

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
                    [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0], [0, 0, 128]])


class App:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        clip_snap = TextEncoder("ViT-B/32")
        dense = clip_dense = None
        if args.openscene_ckpt:
            from opensnap.openscene_model import OpenSceneModel
            dense = OpenSceneModel(args.openscene_ckpt, args.feature)
            clip_dense = clip_snap if dense.clip_name == "ViT-B/32" else TextEncoder(dense.clip_name)
        self.vocab = snap_train_vocab()
        self.m = OpenSNAP(SnapModel(args.snap_ckpt, domain=args.domain), clip_snap, dense, clip_dense,
                          Config(alpha=args.alpha), anchors=self.vocab)
        self.namer = None
        if args.vlm:
            from opensnap.namer import VLMNamer
            self.namer = VLMNamer(args.vlm)
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction("+z")
        self.build_gui()
        self.load(args.scene)
        self.server.scene.on_click()(self.on_click)
        if args.demo:
            self.segment_everything()

    def build_gui(self):
        g = self.server.gui
        g.add_markdown("**OpenSNAP** — click an object to select it: it is segmented and its name appears. "
                       "Shift-click refines the selection.")
        self.status = g.add_markdown("_loading…_")
        self.keep = g.add_checkbox("keep previous objects", initial_value=False)
        self.use_vlm = g.add_checkbox("VLM naming", initial_value=self.namer is not None, disabled=self.namer is None)
        self.vlm_mode = g.add_dropdown("VLM mode", ("choose", "free"), initial_value="choose")
        self.point_size = g.add_slider("point size", 0.002, 0.05, 0.001, self.args.point_size)
        self.point_size.on_update(lambda _: self.draw_base())
        g.add_button("New object").on_click(lambda _: self.new_object())
        g.add_button("Clear all").on_click(lambda _: self.clear())
        with g.add_folder("Segment everything"):
            g.add_button("Show segments").on_click(lambda _: self.run_bg(self.segment_everything))
        with g.add_folder("Open-vocabulary labels"):
            self.vocab_box = g.add_text("classes", "wall, floor, chair, table, sofa, bed, cabinet, door, window, other")
            g.add_button("Label scene").on_click(lambda _: self.run_bg(self.label_scene))
            self.legend = g.add_markdown("")
        with g.add_folder("Text query"):
            self.query_box = g.add_text("query", "something to sit on")
            g.add_button("Find").on_click(lambda _: self.run_bg(self.find))

    def run_bg(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def say(self, text):
        self.status.content = text

    def load(self, path):
        d = load_any(path)
        self.coord, self.color = d["coord"], d.get("color")
        normal = d.get("normal")
        if normal is None and self.args.normals:
            normal = estimate_normals(self.coord)
        t = time.time()
        with self.lock:
            self.m.encode(self.coord, self.color, normal, d.get("strength"))
        self.props = None
        self.base_rgb = (self.color if self.color is not None else np.full_like(self.coord, 180)).astype(np.uint8)
        self.objects, self.clicks, self.nodes, self.seg_id = [], [], {}, None
        self.draw_base()
        lo, hi = self.coord.min(0), self.coord.max(0)
        mid = (lo + hi) / 2
        self.server.initial_camera.look_at = tuple(mid)
        self.server.initial_camera.position = tuple(mid + np.array([0.35, -0.75, 0.75]) * (hi - lo).max())
        self.say(f"{os.path.basename(path.rstrip('/'))}: {len(self.coord):,} points, encoded in {time.time() - t:.1f}s")

    def draw_base(self, rgb=None):
        self.server.scene.add_point_cloud("/scene", self.coord, self.base_rgb if rgb is None else rgb,
                                          point_size=self.point_size.value, point_shape="circle")

    def pick(self, origin, direction):
        o, d = np.asarray(origin), np.asarray(direction)
        d = d / np.linalg.norm(d)
        v = self.coord - o
        t = v @ d
        dist = np.linalg.norm(v - t[:, None] * d, axis=1)
        ok = (t > 0) & (dist < max(self.point_size.value * 2, 0.02))
        if not ok.any():
            return None
        cand = np.nonzero(ok)[0]
        return int(cand[np.argmin(t[cand])])

    def on_click(self, event):
        idx = self.pick(event.ray_origin, event.ray_direction)
        if idx is None:
            return
        refine = event.modifier is not None and self.clicks and self.seg_id is None
        if not refine:
            self.new_object()
        self.clicks.append(idx)
        self.run_bg(self.update_current)

    def update_current(self):
        k = len(self.objects) if self.keep.value else 0
        if not self.keep.value:
            self.remove_objects()
        seg = -1 if self.seg_id is None else int(self.seg_id[self.clicks[-1]])
        with self.lock:
            if seg >= 0:
                props, i = self.props, seg
            else:
                props, i = self.m.click(self.clicks), 0
            mask = (props.logits[i] > 0)[self.m.scene["inverse"]].cpu().numpy()
            name, conf = self.name_fast(props, i)
        self.draw_object(k, mask, f"{name} ({conf:.2f})", clicks=self.clicks)
        self.say(f"selected: **{name}** · IoU≈{props.scores[i]:.2f} · {mask.sum():,} pts")
        if self.use_vlm.value and self.namer is not None:
            self.say(f"selected: **{name}** · asking VLM…")
            with self.lock:
                best, ranked, reply = self.vlm_name(props, i)
            self.draw_object(k, mask, f"{best} ({ranked[0][1]:.2f})", clicks=self.clicks)
            alts = ", ".join(f"{n} {p:.2f}" for n, p in ranked[1:4])
            self.say(f"selected: **{best}** · VLM: “{reply.strip()}” · alt: {alts}")

    def vlm_name(self, props, i):
        if self.vlm_mode.value == "choose":
            return self.namer.name_choose(self.m, props, i, self.coord, self.color, self.vocab)
        return self.namer.name(self.m, props, i, self.coord, self.color, prior_vocab=self.vocab)

    def name_fast(self, props, i):
        sub = type(props)(props.logits[i:i + 1], props.tokens[i:i + 1], props.scores[i:i + 1], props.prompts[i:i + 1])
        p = self.m.mask_probs(sub, self.vocab, use_dense=self.m.scene["dense"] is not None)[0]
        j = int(p.argmax())
        return self.vocab[j], float(p[j])

    def draw_object(self, k, mask, text, clicks=()):
        rgb = PALETTE[k % len(PALETTE)]
        pts = self.coord[mask]
        if len(pts) == 0:
            return
        for h in self.nodes.pop(k, []):
            h.remove()
        sc = self.server.scene
        top = np.array([*np.median(pts[:, :2], 0), pts[:, 2].max() + 0.1])
        hs = [sc.add_point_cloud(f"/objects/{k}/points", pts, np.tile(rgb, (len(pts), 1)).astype(np.uint8),
                                 point_size=self.point_size.value * 1.3, point_shape="circle"),
              sc.add_label(f"/objects/{k}/label", text, position=top)]
        hs += [sc.add_icosphere(f"/objects/{k}/click{j}", radius=0.03, color=(255, 255, 255), position=self.coord[c])
               for j, c in enumerate(clicks)]
        self.nodes[k] = hs

    def new_object(self):
        if self.clicks:
            self.objects.append(self.clicks)
        self.clicks = []

    def remove_objects(self):
        for handles in self.nodes.values():
            for h in handles:
                h.remove()
        self.nodes = {}

    def clear(self):
        self.remove_objects()
        self.objects, self.clicks, self.seg_id = [], [], None
        self.draw_base()

    def ensure_props(self):
        if self.props is None:
            self.say("segmenting everything…")
            with self.lock:
                self.props = self.m.propose()
        return self.props

    def segment_everything(self):
        """Colour all segments (no names); clicking a segment then names it."""
        self.clear()
        props = self.ensure_props()
        inv = self.m.scene["inverse"]
        seg = np.full(len(self.coord), -1)
        best = np.zeros(len(self.coord), np.float32)
        for i in props.scores.argsort().tolist():
            prob = props.logits[i].sigmoid()[inv].cpu().numpy() * float(props.scores[i])
            m = (prob > 0.5 * float(props.scores[i])) & (prob > best)
            seg[m], best[m] = i, prob[m]
        self.seg_id = seg
        rgb = self.base_rgb * 0.3 + 255 * 0.35
        rgb[seg >= 0] = PALETTE[seg[seg >= 0] % len(PALETTE)] * 0.75 + self.base_rgb[seg >= 0] * 0.25
        self.draw_base(rgb.astype(np.uint8))
        self.say(f"{len(props.scores)} segments — click one to name it")

    def label_scene(self):
        labels = [c.strip() for c in self.vocab_box.value.split(",") if c.strip()]
        props = self.ensure_props()
        with self.lock:
            pred = self.m.semantic(props, labels, use_dense=self.m.scene["dense"] is not None).argmax(1).cpu().numpy()
        self.draw_base(PALETTE[pred % len(PALETTE)].astype(np.uint8))
        self.legend.content = "  \n".join(
            f"<span style='color:rgb{tuple(PALETTE[i % len(PALETTE)].tolist())}'>■</span> {l}" for i, l in enumerate(labels))
        self.say(f"labelled {len(labels)} classes")

    def find(self):
        props = self.ensure_props()
        with self.lock:
            order, rel = self.m.query(props, self.query_box.value)
            inv = self.m.scene["inverse"]
            heat = np.zeros(len(self.coord), np.float32)
            for i, r in zip(order[:5].tolist(), rel[:5].tolist()):
                mk = (props.logits[i] > 0)[inv].cpu().numpy()
                heat[mk] = np.maximum(heat[mk], r / max(rel[0].item(), 1e-6))
        rgb = self.base_rgb * 0.35 + 255 * 0.3
        rgb[heat > 0] = (np.array([255, 60, 30]) * heat[heat > 0, None] + rgb[heat > 0] * (1 - heat[heat > 0, None]))
        self.draw_base(rgb.astype(np.uint8))
        self.say(f"query “{self.query_box.value}”: best mask score {rel[0]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--domain", default="Indoor", choices=["Indoor", "Outdoor", "Aerial"])
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/scannet_lseg.pth.tar")
    ap.add_argument("--feature", default="lseg")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct", help="'' to disable")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--normals", type=int, default=0)
    ap.add_argument("--point_size", type=float, default=0.015)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--demo", action="store_true", help="segment & name everything on start")
    args = ap.parse_args()
    App(args)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
