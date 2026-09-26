import re

import numpy as np
import torch

from . import paths  # noqa: F401  (sets HF_HOME)
from .render import render_object

PROMPTS = {
    "highlight": ("These {n} images are renders of an indoor 3D scan from different viewpoints. "
                  "One object is outlined in red. What is the outlined object? "),
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
    def ask(self, images):
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": (PROMPTS[self.style] + PROMPT_TAIL).format(n=len(images), k=self.k)})
        msgs = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return parse(reply)[: self.k], reply

    @torch.no_grad()
    def choose(self, images, options):
        letters = "ABCDEFGHIJ"[: len(options)]
        opts = " ".join(f"{l}) {o}" for l, o in zip(letters, options))
        q = PROMPTS[self.style] + f"Options: {opts}. Answer with the letter only."
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": q}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        ids = [self.processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
        return logits[ids].log_softmax(-1).cpu().numpy()

    def name_choose(self, opensnap, props, idx, coord, color, prior_vocab, k=5, weight=0.5):
        """3D proposes top-k names from an open prior vocabulary, the VLM chooses; log-linear fusion."""
        mask = (props.logits[idx] > 0)[opensnap.scene["inverse"]].cpu().numpy()
        sub = type(props)(props.logits[idx:idx + 1], props.tokens[idx:idx + 1], props.scores[idx:idx + 1], props.prompts[idx:idx + 1])
        p3d = opensnap.mask_probs(sub, prior_vocab, use_dense=opensnap.scene["dense"] is not None)[0]
        top = p3d.topk(k)
        options = [prior_vocab[i] for i in top.indices.tolist()]
        lp3d = torch.log_softmax(top.values.clamp(min=1e-8).log(), -1).cpu().numpy()
        lvlm = self.choose(render_object(coord, color, mask, style=self.style), options)
        score = (1 - weight) * lp3d + weight * lvlm
        order = np.argsort(-score)
        ranked = [(options[i], float(np.exp(score[i] - np.logaddexp.reduce(score)))) for i in order]
        return ranked[0][0], ranked, options[int(lvlm.argmax())]

    def name(self, opensnap, props, idx, coord, color, prior_vocab=None, n_3d=3, gamma=1.0, eps=0.05):
        """Name proposal `idx` from VLM candidates + top 3D-evidence names of `prior_vocab`."""
        mask = (props.logits[idx] > 0)[opensnap.scene["inverse"]].cpu().numpy()
        cands, reply = self.ask(render_object(coord, color, mask, style=self.style))
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
