from __future__ import annotations

import torch
from torch import nn

from wgram_lm.mixers import OfficialGatedDeltaNet2Mixer

from .config import WGRAMV2Config


def causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((int(seq_len), int(seq_len)), dtype=torch.bool, device=device),
        diagonal=1,
    )


class GatedDeltaLikeLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.delta = nn.Linear(int(d_model), int(d_model))
        self.erase = nn.Linear(int(d_model), int(d_model))
        self.write = nn.Linear(int(d_model), int(d_model))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        erase = torch.sigmoid(self.erase(h))
        write = torch.sigmoid(self.write(h))
        delta = torch.tanh(self.delta(h))
        return x * (1.0 - 0.1 * erase) + self.dropout(write * delta)


class GatedDeltaAttention3To1Core(nn.Module):
    """Small torch smoke implementation of the V2 recurrent-core interface.

    Promotion runs must replace this with the official GatedDeltaNet-2 backend.
    The 3:1 layer order is kept here so tiny tests exercise the same interface.
    """

    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        self.is_torch_smoke_core = True
        self.core_implementation = "torch_smoke"
        self.core_attention_causal = bool(config.core_attention_causal)
        d_model = int(config.d_model)
        self.blocks = nn.ModuleList()
        for _ in range(int(config.core_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "delta0": GatedDeltaLikeLayer(d_model, float(config.dropout)),
                        "delta1": GatedDeltaLikeLayer(d_model, float(config.dropout)),
                        "delta2": GatedDeltaLikeLayer(d_model, float(config.dropout)),
                        "attn_norm": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model,
                            int(config.local_heads),
                            dropout=float(config.dropout),
                            batch_first=True,
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, chunk_states: torch.Tensor, chunk_valid: torch.Tensor, *, think_steps: int) -> torch.Tensor:
        h = chunk_states
        key_padding_mask = ~chunk_valid.bool()
        attn_mask = causal_attention_mask(int(chunk_states.shape[1]), chunk_states.device) if self.core_attention_causal else None
        for _ in range(max(1, int(think_steps))):
            for block in self.blocks:
                h = block["delta0"](h)
                h = block["delta1"](h)
                h = block["delta2"](h)
                attn_in = block["attn_norm"](h)
                attn_out, _ = block["attn"](
                    attn_in,
                    attn_in,
                    attn_in,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                h = h + attn_out
                h = h * chunk_valid.to(h.dtype).unsqueeze(-1)
        return self.final_norm(h)


class OfficialGatedDeltaNet2Core(nn.Module):
    """V2 3:1 core backed by the NVlabs GatedDeltaNet-2 adapter."""

    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        self.is_torch_smoke_core = False
        self.core_implementation = "official_gated_delta2"
        self.core_attention_causal = bool(config.core_attention_causal)
        self.official_gdn2_force_chunk_eval = bool(config.official_gdn2_force_chunk_eval)
        d_model = int(config.d_model)
        n_heads = int(config.local_heads)
        common_kwargs = {
            "head_dim": int(config.official_gdn2_head_dim) if int(config.official_gdn2_head_dim) > 0 else None,
            "num_v_heads": int(config.official_gdn2_num_v_heads)
            if int(config.official_gdn2_num_v_heads) > 0
            else None,
            "expand_v": float(config.official_gdn2_expand_v),
            "mode": str(config.official_gdn2_mode),
            "use_short_conv": bool(config.official_gdn2_use_short_conv),
            "force_chunk_eval": bool(config.official_gdn2_force_chunk_eval),
            "conv_size": int(config.official_gdn2_conv_size),
            "norm_eps": float(config.official_gdn2_norm_eps),
        }
        self.blocks = nn.ModuleList()
        for _ in range(int(config.core_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "delta0": OfficialGatedDeltaNet2Mixer(
                            d_model=d_model,
                            n_heads=n_heads,
                            strict=True,
                            fallback_dropout=float(config.dropout),
                            **common_kwargs,
                        ),
                        "delta1": OfficialGatedDeltaNet2Mixer(
                            d_model=d_model,
                            n_heads=n_heads,
                            strict=True,
                            fallback_dropout=float(config.dropout),
                            **common_kwargs,
                        ),
                        "delta2": OfficialGatedDeltaNet2Mixer(
                            d_model=d_model,
                            n_heads=n_heads,
                            strict=True,
                            fallback_dropout=float(config.dropout),
                            **common_kwargs,
                        ),
                        "attn_norm": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model,
                            n_heads,
                            dropout=float(config.dropout),
                            batch_first=True,
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, chunk_states: torch.Tensor, chunk_valid: torch.Tensor, *, think_steps: int) -> torch.Tensor:
        if chunk_states.device.type != "cuda":
            raise RuntimeError("Official GatedDeltaNet-2 V2 core requires CUDA/Triton tensors")
        h = chunk_states
        key_padding_mask = ~chunk_valid.bool()
        attention_mask = chunk_valid.to(torch.long)
        attn_mask = causal_attention_mask(int(chunk_states.shape[1]), chunk_states.device) if self.core_attention_causal else None
        for _ in range(max(1, int(think_steps))):
            for block in self.blocks:
                h = block["delta0"](h, attention_mask=attention_mask)
                h = block["delta1"](h, attention_mask=attention_mask)
                h = block["delta2"](h, attention_mask=attention_mask)
                attn_in = block["attn_norm"](h)
                attn_out, _ = block["attn"](
                    attn_in,
                    attn_in,
                    attn_in,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                h = (h + attn_out) * chunk_valid.to(h.dtype).unsqueeze(-1)
        return self.final_norm(h)


def build_v2_recurrent_core(config: WGRAMV2Config) -> nn.Module:
    if str(config.core_implementation) == "torch_smoke":
        if not bool(config.allow_torch_smoke_core):
            raise ValueError("torch_smoke V2 core requires allow_torch_smoke_core=True")
        return GatedDeltaAttention3To1Core(config)
    if str(config.core_implementation) == "official_gated_delta2":
        if bool(config.allow_torch_smoke_core):
            raise ValueError("official_gated_delta2 V2 core must not allow torch smoke fallback")
        return OfficialGatedDeltaNet2Core(config)
    raise ValueError(f"unsupported V2 core_implementation: {config.core_implementation}")
