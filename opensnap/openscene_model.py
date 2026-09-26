from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import paths

paths.add(paths.OPENSCENE_DIR)


class OpenSceneModel:
    """OpenScene 3D distilled model (MinkUNet18A) giving per-point CLIP-space features."""

    def __init__(self, ckpt, feature="lseg", voxel_size=0.02, device="cuda"):
        import MinkowskiEngine as ME
        from models.disnet import DisNet
        self.ME = ME
        self.device = torch.device(device)
        self.voxel_size = voxel_size
        self.feature = feature
        self.clip_name = "ViT-B/32" if feature == "lseg" else "ViT-L/14@336px"
        self.model = DisNet(SimpleNamespace(feature_2d_extractor=feature, arch_3d="MinkUNet18A")).to(self.device)
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, coord):
        vox = np.floor(coord / self.voxel_size)
        vox -= vox.min(0)
        _, idx, inv = np.unique(vox, axis=0, return_index=True, return_inverse=True)
        c = torch.from_numpy(vox[idx]).int()
        c = torch.cat([torch.zeros(len(c), 1, dtype=torch.int), c], 1).to(self.device)
        x = self.ME.SparseTensor(torch.ones(len(c), 3, device=self.device), c)
        feat = self.model(x)
        return F.normalize(feat[torch.from_numpy(inv.reshape(-1)).to(self.device)].float(), dim=-1)
