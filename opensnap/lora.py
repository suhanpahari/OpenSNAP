import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + s·B A x with B = 0 at init, so the adapted backbone starts exactly at SNAP."""

    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base, self.s, self.on = base, alpha / r, True
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if not self.on:
            return y
        return y + (x.to(self.A.dtype) @ self.A.T @ self.B.T).to(y.dtype) * self.s


def add_lora(backbone, r=16, alpha=16):
    """Wrap qkv / proj of every serialized-attention layer of the PTv3 backbone; returns the LoRA modules."""
    mods = []
    for m in list(backbone.modules()):
        if type(m).__name__ == "SerializedAttention":
            for name in ("qkv", "proj"):
                lora = LoRALinear(getattr(m, name), r, alpha)
                setattr(m, name, lora)
                mods.append(lora)
    return mods


def set_lora(mods, on):
    for m in mods:
        m.on = on


def lora_state(backbone):
    return {k: v for k, v in backbone.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
