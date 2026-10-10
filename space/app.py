"""OpenSNAP Space: upload a phone video -> 3D scene -> click objects in a frame to segment + name them, or query by text."""
import os
import shutil
import sys
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENSNAP_DOWNLOADS", os.path.join(HERE, "cache"))
os.environ.setdefault("SNAP_DIR", os.path.join(HERE, "third_party", "SNAP"))

try:
    import spaces
    GPU = spaces.GPU
except ImportError:
    def GPU(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

from opensnap import paths  # noqa: E402

WEIGHTS = os.environ.get("OPENSNAP_WEIGHTS", "suhanxvii/opensnap-weights")
LOCAL = os.environ.get("OPENSNAP_LOCAL_CKPT", "")


def weight(name):
    if LOCAL:
        return os.path.join(LOCAL, name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(WEIGHTS, name)


os.makedirs(os.path.join(paths.DOWNLOADS, "ckpt"), exist_ok=True)
SNAP_CKPT = os.path.join(paths.DOWNLOADS, "ckpt", "SNAP_C.pth")
if not os.path.exists(SNAP_CKPT):
    shutil.copy(weight("SNAP_C.pth"), SNAP_CKPT)
CORR_CKPT = weight("opensnap_corrective_v5_multi.pth")

from opensnap.clicks import ClickSegmenter, CorrectiveSegmenter  # noqa: E402
from opensnap.core import Config, OpenSNAP, Proposals  # noqa: E402
from opensnap.data import prior_vocab, snap_train_vocab  # noqa: E402
from opensnap.namer import VLMNamer  # noqa: E402
from opensnap.photo import PhotoViews  # noqa: E402
from opensnap.query.catalog import materialize  # noqa: E402
from opensnap.query.engine import Engine  # noqa: E402
from opensnap.snap_model import SnapModel  # noqa: E402
from opensnap.text import TextEncoder  # noqa: E402
from opensnap.video import load_models, reconstruct  # noqa: E402

VGGT, METRIC = load_models(os.path.join(paths.DOWNLOADS, "hf"))
SNAP = SnapModel(SNAP_CKPT, flash=False if os.environ.get("OPENSNAP_NO_FLASH") else None)
M = OpenSNAP(SNAP, TextEncoder("ViT-B/32"), None, None, Config(), anchors=snap_train_vocab())
TUNER = ClickSegmenter(SNAP, "ttt", tta=8, tta_radius=0.1)
CORR = CorrectiveSegmenter(SNAP, CORR_CKPT, tuner=ClickSegmenter(SNAP, "ttt"))
NAMER = VLMNamer(os.environ.get("OPENSNAP_VLM", "Qwen/Qwen2.5-VL-7B-Instruct"))
VOCAB = prior_vocab("snap+lvis")
BACKGROUND = tuple(dict.fromkeys(("wall", "floor", "ceiling", "object", *snap_train_vocab())))  # open world without dense features
PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40]])


def encode(state):
    coord = np.load(os.path.join(state["dir"], "coord.npy"))
    np.random.seed(0)
    torch.manual_seed(0)
    M.encode(coord)
    return coord


def props_from_state(state):
    p = np.load(os.path.join(state["dir"], "props.npz"))
    dev = M.scene["vcoord"].device
    return Proposals(torch.from_numpy(p["logits"]).to(dev).float(), torch.from_numpy(p["tokens"]).to(dev).float(),
                     torch.from_numpy(p["scores"]).to(dev).float(), torch.from_numpy(p["prompts"]).to(dev).float())


@torch.no_grad()
def photo_embeddings(pv, coord, props):
    """CLIP image embedding of each object's best photo crop: object-anchored evidence when dense 3D features are absent."""
    import clip
    model = M.clip_snap.model
    prep = clip.clip._transform(model.visual.input_resolution)
    inv = M.scene["inverse"].cpu().numpy()
    out = np.zeros((len(props.scores), model.visual.output_dim), np.float32)
    for i, lg in enumerate((props.logits > 0).cpu().numpy()):
        crops = pv.crops(coord[lg[inv]], n=1, margin=0.15, box=False)
        if crops:
            e = model.encode_image(prep(crops[0])[None].to(M.clip_snap.device)).float()
            out[i] = torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()
    return out


def figure(coord, color, highlight=None, n=120000):
    idx = np.random.default_rng(0).choice(len(coord), min(n, len(coord)), replace=False)
    rgb = color[idx].astype(np.float32)
    if highlight is not None:
        for mask, c in highlight:
            sel = mask[idx]
            rgb[sel] = 0.25 * rgb[sel] + 0.75 * np.asarray(c, np.float32)
        if highlight:
            rgb[~np.any([m[idx] for m, _ in highlight], axis=0)] *= 0.75
    cols = [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in rgb]
    p = coord[idx]
    fig = go.Figure(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", marker=dict(size=2.2, color=cols), hoverinfo="skip"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=0.55, y=-0.75, z=0.65))),
                      margin=dict(l=0, r=0, t=0, b=0), height=520, paper_bgcolor="white")
    return fig


@GPU(duration=150)
def build_scene(video, n_frames, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video first.")
    work = tempfile.mkdtemp(prefix="opensnap_")
    progress(0.1, desc="Reconstructing with VGGT…")
    log = []
    coord, color = reconstruct(video, work, VGGT, METRIC, frames=int(n_frames), ply=False, log=log.append)
    progress(0.6, desc="Segmenting objects with SNAP…")
    state = dict(dir=work, pos=[], neg=[], frame=0)
    coord = encode(state)
    props = M.propose()
    np.savez_compressed(os.path.join(work, "props.npz"), logits=props.logits.half().cpu().numpy(), tokens=props.tokens.cpu().numpy(),
                        scores=props.scores.cpu().numpy(), prompts=props.prompts.cpu().numpy(),
                        inverse=M.scene["inverse"].cpu().numpy())
    pv = PhotoViews(os.path.join(work, "views.npz"))
    progress(0.85, desc="Embedding object photos…")
    np.save(os.path.join(work, "photo_emb.npy"), photo_embeddings(pv, coord, props))
    msg = " · ".join(log[-2:]) + f" · {len(props.scores)} objects found"
    return (state, figure(coord, color), gr.update(maximum=len(pv.frames) - 1, value=0), pv.overlay(0, np.zeros((0, 3))), msg,
            None, "")


def show_frame(state, k):
    if not state:
        return None
    pv = PhotoViews(os.path.join(state["dir"], "views.npz"))
    state["frame"] = int(k)
    pts = state.get("mask_pts")
    return pv.overlay(int(k), np.load(pts) if pts else np.zeros((0, 3)))


@GPU(duration=90)
def click(state, k, mode, method, evt: gr.SelectData):
    if not state:
        raise gr.Error("Build a scene first.")
    pv = PhotoViews(os.path.join(state["dir"], "views.npz"))
    color = np.load(os.path.join(state["dir"], "color.npy"))
    P = pv.lift(int(k), *evt.index)
    if mode == "new object" or not state["pos"]:
        state["pos"], state["neg"] = [], []
    (state["neg"] if mode == "negative (remove)" else state["pos"]).append(P.tolist())
    if not state["pos"]:
        raise gr.Error("Start an object with a positive click.")
    coord = encode(state)
    vc, inv = M.scene["vcoord"], M.scene["inverse"]
    ct = torch.from_numpy(coord).to(vc.device)
    near = lambda pts: inv[torch.cdist(torch.tensor(pts, device=vc.device, dtype=ct.dtype), ct).argmin(1)]  # SNAP voxels are centre-shifted
    seg = CORR if method.startswith("corrective") else TUNER
    if seg is CORR and len(state["pos"]) == 1 and not state["neg"]:
        CORR.prev = None
    mask_v = seg(vc, near(state["pos"]), near(state["neg"]) if state["neg"] else torch.zeros(0, dtype=torch.long, device=vc.device))
    mask = mask_v[M.scene["inverse"]].cpu().numpy()
    pts_file = os.path.join(state["dir"], "mask_pts.npy")
    np.save(pts_file, coord[mask])
    state["mask_pts"] = pts_file
    crops = pv.crops(coord[mask], n=1, first=int(k), pair=True, margin=1.0) or pv.crops(coord[mask], n=1, pair=True, margin=1.0)
    if crops:
        cands, reply = NAMER.ask(crops, "photo_pair")
        name = cands[0] if cands else "?"
        alts = ", ".join(cands[1:3])
    else:
        name, alts = "(not visible in any frame)", ""
    info = f"### {name}\nalternatives: {alts}  ·  {mask.sum():,} points  ·  clicks: {len(state['pos'])}+ / {len(state['neg'])}−"
    im = pv.overlay(int(k), coord[mask])
    x, y = evt.index
    r = max(8, im.width // 90)
    ImageDraw.Draw(im).ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255) if mode != "negative (remove)" else (0, 160, 255),
                               width=max(3, r // 3))
    return (state, figure(coord, color, [(mask, (255, 40, 40))]), im, crops[-1] if crops else None, info)


@GPU(duration=90)
def query(state, text, k):
    if not state:
        raise gr.Error("Build a scene first.")
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a query, e.g. chair or SELECT WHERE IS('monitor') AND NEAR(IS('keyboard'), 0.5)")
    q = text if text.upper().startswith("SELECT") else f"SELECT WHERE IS('{text}') LIMIT 3"
    coord = encode(state)
    color = np.load(os.path.join(state["dir"], "color.npy"))
    props = props_from_state(state)
    M.scene["props"] = props
    table = materialize(M, props)
    table.pooled = torch.from_numpy(np.load(os.path.join(state["dir"], "photo_emb.npy"))).to(M.scene["vcoord"].device)
    M.clip_dense = M.clip_snap
    res, st = Engine(M, table, refine=False, background=BACKGROUND).run(q)
    inv = M.scene["inverse"].cpu().numpy()
    hl, lines = [], []
    for r, (oid, p) in enumerate(res[:5]):
        vm = np.zeros(len(M.scene["vcoord"]), bool)
        vm[table.points[oid]] = True
        hl.append((vm[inv], PALETTE[r % len(PALETTE)]))
        lines.append(f"{r + 1}. object {oid} · P = {p:.2f}" + ("  (created by this query)" if table.provenance[oid] != "auto" else ""))
    pv = PhotoViews(os.path.join(state["dir"], "views.npz"))
    top = coord[hl[0][0]] if hl else np.zeros((0, 3))
    if hl:
        np.save(os.path.join(state["dir"], "mask_pts.npy"), top)
        state["mask_pts"] = os.path.join(state["dir"], "mask_pts.npy")
    msg = f"`{q}` · {st['ms']:.1f} ms\n\n" + ("\n".join(lines) if lines else "no match")
    return state, figure(coord, color, hl), pv.overlay(int(k), top), msg


def segments(state):
    if not state:
        raise gr.Error("Build a scene first.")
    coord = np.load(os.path.join(state["dir"], "coord.npy"))
    color = np.load(os.path.join(state["dir"], "color.npy"))
    p = np.load(os.path.join(state["dir"], "props.npz"))
    masks = p["logits"] > 0
    hl = [(masks[i][p["inverse"]], PALETTE[i % len(PALETTE)]) for i in np.argsort(-p["scores"])]
    return figure(coord, color, hl), f"{len(hl)} segments"


with gr.Blocks(title="OpenSNAP", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OpenSNAP — open-vocabulary 3D objects from a phone video\n"
                "1) Upload a short video walking around a desk, shelf or room and press **Build 3D scene**. "
                "2) **Click an object in the frame** to segment it in 3D and name it (add more clicks to refine, switch to "
                "*negative* to remove spill). 3) Or **query by text** (a word, or OVQL). "
                "[Code & paper](https://github.com/suhanpahari/OpenSNAP)")
    state = gr.State({})
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="video (static scene, ~20-90 s, walk around it)")
            n_frames = gr.Slider(16, 64, value=40, step=8, label="frames used for reconstruction (more = better geometry, slower)")
            build = gr.Button("Build 3D scene", variant="primary")
            status = gr.Markdown()
            gr.Examples([[os.path.join(HERE, "examples", "lab_desk.mp4")]], inputs=[video], label="example")
        with gr.Column(scale=2):
            plot = gr.Plot(label="3D scene")
    with gr.Row():
        with gr.Column(scale=2):
            frame_k = gr.Slider(0, 31, value=0, step=1, label="frame")
            frame = gr.Image(label="click an object here", interactive=False, type="pil")
        with gr.Column(scale=1):
            mode = gr.Radio(["new object", "positive (add)", "negative (remove)"], value="new object", label="click type")
            method = gr.Radio(["token tuning (best indoors)", "corrective decoder + tuning"], value="token tuning (best indoors)",
                              label="segmenter")
            info = gr.Markdown()
            crop = gr.Image(label="what the VLM sees", interactive=False, type="pil")
            q = gr.Textbox(label="text query", placeholder="monitor  ·  SELECT WHERE IS('chair') AND NEAR(IS('desk'), 0.5)")
            with gr.Row():
                go_q = gr.Button("Query")
                seg_all = gr.Button("Segment everything")
            qinfo = gr.Markdown()

    build.click(build_scene, [video, n_frames], [state, plot, frame_k, frame, status, crop, info])
    frame_k.change(show_frame, [state, frame_k], [frame])
    frame.select(click, [state, frame_k, mode, method], [state, plot, frame, crop, info])
    go_q.click(query, [state, q, frame_k], [state, plot, frame, qinfo])
    q.submit(query, [state, q, frame_k], [state, plot, frame, qinfo])
    seg_all.click(segments, [state], [plot, qinfo])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
