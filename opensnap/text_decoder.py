import torch

from .snap_plus import SNAPPlus, load_snap


class TextDecoder:
    """Trained text-prompt decoder (SNAP+ v4) that reuses the segmenter's already-encoded point features."""

    def __init__(self, ckpt, clip_b32, device="cuda"):
        state = torch.load(ckpt, map_location="cpu")
        cfg = state["cfg"]
        assert cfg.get("arch") == "v4", "text decoder checkpoint must be arch v4"
        self.sem_dim, self.feature = cfg["sem_dim"], cfg["feature"]
        self.model = SNAPPlus(load_snap(cfg["snap_ckpt"], device), sem_dim=cfg["sem_dim"], num_masks=0, use_sem=cfg["use_sem"],
                              use_text=True, arch="v4").to(device)
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        assert not unexpected and all(k.startswith("snap.backbone") for k in missing), (missing[:5], unexpected[:5])
        self.model.eval()
        self.clip = clip_b32

    @torch.no_grad()
    def __call__(self, snap_model, dense, text):
        """-> (voxel mask, predicted IoU) for 'segment the <text>' on the scene snap_model encoded."""
        point = snap_model.point
        if dense is not None:
            assert dense.shape[1] == self.sem_dim, f"text decoder expects {self.sem_dim}-d dense features ({self.feature})"
            point.sem_key = self.model.sem_adapter(dense.float())
        emb = self.clip([text], ("segment {}",)).float()
        with torch.autocast("cuda", dtype=torch.float16):
            logits, iou, _ = self.model.decode(point, snap_model.data["coord"], text=emb)
        return logits[0, 0] > 0, float(iou[0, 0])
