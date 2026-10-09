import re

import numpy as np
import torch

from . import paths  # noqa: F401  (sets HF_HOME)
from .render import render_object

PROMPTS = {
    "highlight": ("These {n} images are renders of an indoor 3D scan from different viewpoints. "
                  "One object is outlined in red. What is the outlined object? "),
    "photo": ("These {n} images are photos of a real scene. One object is marked by a red box. What is the object "
              "inside the red box? "),
    "isolate": ("These {n} images show the same object from a 3D point cloud scan, seen from different viewpoints. "
                "The object is drawn in full color and its surroundings are dimmed. What is the object? "),
}
PROMPT_TAIL = "Reply with exactly {k} candidate names, most likely first, as short common nouns separated by commas."
STOP = {"object", "thing", "item", "point cloud", "3d model", "scan", "unknown"}


class VLMNamer:
    """Render-Ask-Verify: a small VLM proposes names, 3D CLIP evidence re-ranks them."""

    def __init__(self, model="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda", k=3, style="highlight"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        self.device, self.k, self.style = device, k, style

    @torch.no_grad()
    def ask(self, images, style=None):
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": (PROMPTS[style or self.style] + PROMPT_TAIL).format(n=len(images), k=self.k)})
        msgs = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return parse(reply)[: self.k], reply

    @torch.no_grad()
    def choose(self, images, options, style=None):
        letters = "ABCDEFGHIJ"[: len(options)]
        opts = " ".join(f"{l}) {o}" for l, o in zip(letters, options))
        st = style or self.style
        q = (PROMPTS[st].format(n=len(images)) if st == "photo" else PROMPTS[st]) + f"Options: {opts}. Answer with the letter only."
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": q}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        ids = [self.processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
        return logits[ids].log_softmax(-1).cpu().numpy()

    def name_choose(self, opensnap, props, idx, coord, color, prior_vocab, k=5, weight=0.5, cycle=1.0, use_vlm=True, photos=None):
        """3D proposes top-k names, the VLM chooses, and each name must re-segment the clicked object (cycle)."""
        inv = opensnap.scene["inverse"]
        m0 = props.logits[idx] > 0
        sub = type(props)(props.logits[idx:idx + 1], props.tokens[idx:idx + 1], props.scores[idx:idx + 1], props.prompts[idx:idx + 1])
        p3d = opensnap.mask_probs(sub, prior_vocab, use_dense=opensnap.scene["dense"] is not None)[0]
        top = p3d.topk(k)
        options = [prior_vocab[i] for i in top.indices.tolist()]
        parts = {"lp3d": torch.log_softmax(top.values.clamp(min=1e-8).log(), -1).cpu().numpy()}
        if use_vlm:
            ims, st = self.views(coord, color, m0[inv].cpu().numpy(), photos)
            parts["lvlm"] = self.choose(ims, options, st)
        if cycle:
            click = int((opensnap.scene["vcoord"] - props.prompts[idx, 0]).norm(dim=1).argmin())
            parts["iou"], parts["contain"] = cycle_ious(opensnap, m0, options, click)
        score = fuse(parts, weight, cycle)
        order = np.argsort(-score)
        ranked = [(options[i], float(np.exp(score[i] - np.logaddexp.reduce(score)))) for i in order]
        pick = options[int(parts["lvlm"].argmax())] if use_vlm else options[0]
        return ranked[0][0], ranked, pick, dict(options=options, **{k: v.tolist() for k, v in parts.items()})

    def views(self, coord, color, mask, photos=None):
        """Real photo crops when the scene comes from a video (PhotoViews), else point-cloud renders."""
        if photos is not None:
            ims = photos.crops(coord[mask])
            if ims:
                return ims, "photo"
        return render_object(coord, color, mask, style=self.style), self.style

    def name(self, opensnap, props, idx, coord, color, prior_vocab=None, n_3d=3, gamma=1.0, eps=0.05, photos=None):
        """Name proposal `idx` from VLM candidates + top 3D-evidence names of `prior_vocab`."""
        mask = (props.logits[idx] > 0)[opensnap.scene["inverse"]].cpu().numpy()
        ims, st = self.views(coord, color, mask, photos)
        cands, reply = self.ask(ims, st)
        if st == "photo" and cands:
            w = np.exp(-0.7 * np.arange(len(cands)))
            return cands[0], [(c, float(x / w.sum())) for c, x in zip(cands, w)], reply
        sub = type(props)(props.logits[idx:idx + 1], props.tokens[idx:idx + 1], props.scores[idx:idx + 1], props.prompts[idx:idx + 1])
        use_dense = opensnap.scene["dense"] is not None
        extra = []
        if prior_vocab:
            pv = opensnap.mask_probs(sub, prior_vocab, use_dense=use_dense)[0]
            extra = [prior_vocab[i] for i in pv.topk(n_3d).indices.tolist()]
        vocab = list(dict.fromkeys(cands + extra)) or ["object"]
        p3d = opensnap.mask_probs(sub, vocab, use_dense=use_dense)[0].cpu().numpy()
        prior = np.full(len(vocab), eps)
        prior[: len(cands)] += np.exp(-0.7 * np.arange(len(cands)))
        score = np.log(p3d + 1e-8) + gamma * np.log(prior / prior.sum())
        order = np.argsort(-score)
        ranked = [(vocab[i], float(np.exp(score[i] - np.logaddexp.reduce(score)))) for i in order]
        return ranked[0][0], ranked, reply


def parse(reply):
    reply = reply.strip().lower().split("\n")[0]
    names = []
    for part in re.split(r",|;|/|\bor\b", reply):
        n = re.sub(r"^(\d+[.)]\s*|a |an |the )", "", part.strip()).strip(" .\"'*-")
        if n and n not in STOP and n not in names and len(n) < 40:
            names.append(n)
    return names


def cycle_ious(opensnap, m0, options, click=None, margin=0.5):
    """Per candidate name: IoU of its text-driven mask with the clicked mask, and whether it contains the click."""
    vc = opensnap.scene["vcoord"]
    pts = vc[m0] if m0.any() else vc[click][None]
    lo, hi = pts.min(0).values - margin, pts.max(0).values + margin
    region = ((vc >= lo) & (vc <= hi)).all(1).float()
    iou, contain = [], []
    for name in options:
        p, _ = opensnap.text_click(name, region=region)
        m = p.logits[0] > 0
        iou.append(float((m & m0).sum() / (m | m0).sum().clamp(min=1)))
        contain.append(float(p.logits[0, click].sigmoid()) if click is not None else 0.0)
    return np.array(iou), np.array(contain)


def fuse(parts, weight=0.5, cycle=1.0, eps=0.05, mode="contain"):
    score = parts["lp3d"] * (1 - weight if "lvlm" in parts else 1.0)
    if "lvlm" in parts:
        score = score + weight * parts["lvlm"]
    if cycle and mode in parts:
        score = score + cycle * np.log(np.asarray(parts[mode]) + eps)
    return score
