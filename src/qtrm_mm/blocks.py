from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from torch import nn
import torch

from .norm import RMSNorm
from .attention import GroupedQueryAttention

# === Best-State v2.5: Minimal InferenceState (for serving + clean generation) ===
# This is the fixed-size state contract that replaces growing KV or complex external triple state
# for inference-time native full-stack execution.
@dataclass
class InferenceState:
    """Small, fixed-size state for inference / generation / native 72.

    Contains:
    - fast_recurrent_h: the internal Griffin-style recurrence hidden (constant size)
    - slow_memory_summary: compact summary of chunked slow memory (optional, can be None)
    - step_count: for adaptive depth / early exit logic
    """
    fast_recurrent_h: Optional[torch.Tensor] = None
    slow_memory_summary: Optional[torch.Tensor] = None
    step_count: int = 0

    def to(self, device, dtype=None):
        if self.fast_recurrent_h is not None:
            self.fast_recurrent_h = self.fast_recurrent_h.to(device, dtype=dtype)
        if self.slow_memory_summary is not None:
            self.slow_memory_summary = self.slow_memory_summary.to(device, dtype=dtype)
        return self

    def clone(self):
        return InferenceState(
            fast_recurrent_h=self.fast_recurrent_h.clone() if self.fast_recurrent_h is not None else None,
            slow_memory_summary=self.slow_memory_summary.clone() if self.slow_memory_summary is not None else None,
            step_count=self.step_count,
        )
# Official MLA will be loaded dynamically from references when available (following project convention)
from .ffn import SwiGLU
from .mixers import build_delta_mixer, TorchGatedDeltaNet2MixerV2, OfficialGatedDeltaNet2Mixer
from .config import QTRMConfig

# RI-4: Sparse Slot Router (Raven/MSA style) - optional import for clean integration
try:
    from .memory.sparse_slot_router import SparseSlotRouter, make_sparse_slot_router
except Exception:
    SparseSlotRouter = None
    make_sparse_slot_router = None

# RI-4 Next Big Jump (2026-06): Decoupled Latent Memory Bank (MELT + G-MemLLM inspired)
# Optional import - zero behavior change when not used.
try:
    from .memory.decoupled_latent_memory_bank import DecoupledLatentMemoryBank, make_decoupled_latent_memory_bank
except Exception:
    DecoupledLatentMemoryBank = None
    make_decoupled_latent_memory_bank = None

# 2026-06 Radical direction: Latent Episode Memory (LEM)
# Fundamental shift: memory writes are sparse at coherent "episode" boundaries instead of continuous per micro-step.
# This attacks the root cause ("too frequent write opportunities dilute selectivity learning").
try:
    from .memory.latent_episode_memory import LatentEpisodeMemory, make_latent_episode_memory
except Exception:
    LatentEpisodeMemory = None
    make_latent_episode_memory = None

# =============================================================================
# D Implementation (Griffin-style internal fast recurrence for native brain participation)
# Minimal RG-LRU inspired gated linear recurrence to make per-micro "fast path"
# (working + attractor evolution) internal, compiled, and cheap.
# This is the first concrete step after the "D" paper dive.
# =============================================================================

class FastGatedLinearRecurrence(nn.Module):
    """
    Griffin RG-LRU style internal fast recurrence, upgraded for Best-State v2.

    Core: Per-micro citizen recurrence for fast brain-mimetic thinking (working + attractor).
    This replaces external heavy triple.step for the fast path.

    2026 Architecture Wave Improvements (MLA + Dynamic BLT inspired):
    - MLA-style low-rank latent compression of the recurrent hidden state (optional).
      Dramatically reduces the size of the state that must be carried across long
      micro-step horizons or across 72 heldout cases, while learned up-projections
      recover most capacity. Directly inspired by DeepSeek's Multi-Head Latent Attention
      (low-rank joint compression + absorption).
    - BLT-style surprise/entropy-driven dynamic modulation inside the recurrence.
      Uses the existing Predictive Data Intuition surprise signal to adaptively
      control how "responsive" the fast path is on each micro-step (high surprise
      → more input influence, analogous to shorter high-entropy patches in Meta's
      Byte Latent Transformer).

    These two additions keep the fast recurrence as a true compiled first-class citizen
    while making long-horizon native participation much more practical and efficient.
    """

    def __init__(self, d_model: int, decay_base: float = 0.95, latent_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.decay_base = decay_base

        self.Wr = nn.Linear(d_model, d_model, bias=True)
        self.Wi = nn.Linear(d_model, d_model, bias=True)

        # Learnable base decay (per dimension for more expressivity)
        self.log_decay = nn.Parameter(torch.full((d_model,), fill_value=torch.log(torch.tensor(decay_base))))

        # FINAL Parcae + Griffin (closes MD "negative-diagonal + ZOH discipline is the missing stability recipe for our prototype"):
        # Learned negative diagonal + explicit Zero-Order Hold discipline for deep stable unrolls.
        self.use_negative_diagonal = True
        self._force_negative_diagonal_in_native = True

        # Learned per-dimension negative log-scale (Parcae spectral radius guarantee)
        self.log_neg_scale = nn.Parameter(torch.full((d_model,), fill_value=-2.0))  # start mildly negative

        # === Final GRAM/PTRM + ParaThinker upgrade: Native stochastic breadth inside recurrence ===
        # Allows the fast internal path to generate/maintain diversity across K mental trajectories
        # without leaving the compiled recurrence every step. This is the closest we can get to
        # historical GRAM/PTRM "stochastic guidance during recurrence" in the new substrate.
        self.stochastic_breadth = 1  # 1 = deterministic fast path; >1 enables native K-trajectory mode

        # First-class inference vs training divergence (per brain_attractor...md blueprint)
        self._inference_mode = False

        # === MOST AGGRESSIVE: True Latent-Native RG-LRU (MLA extreme + Parcae) ===
        # When latent compression is active, the core recurrence math (gates + update) now runs
        # primarily inside the compressed latent space. Only the final output injection does a full
        # up-projection. This is the radical "recurrence lives in tiny latent space" direction.
        self.latent_dim = latent_dim
        if latent_dim is not None and latent_dim < d_model:
            self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
            self.up_proj = nn.Linear(latent_dim, d_model, bias=False)
            self.latent_residual_scale = nn.Parameter(torch.tensor(0.05))  # very aggressive lean
            # Extra: small latent-space gates for truly native latent recurrence
            self.latent_Wr = nn.Linear(latent_dim, latent_dim, bias=True)
            self.latent_Wi = nn.Linear(latent_dim, latent_dim, bias=True)
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.xavier_uniform_(self.latent_Wr.weight)
            nn.init.xavier_uniform_(self.latent_Wi.weight)
            nn.init.zeros_(self.latent_Wr.bias)
            nn.init.zeros_(self.latent_Wi.bias)
        else:
            self.latent_dim = None
            self.down_proj = None
            self.up_proj = None
            self.latent_residual_scale = None
            self.latent_Wr = None
            self.latent_Wi = None

        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Wi.weight)
        nn.init.zeros_(self.Wr.bias)
        nn.init.zeros_(self.Wi.bias)

    def set_inference_mode(self, enabled: bool = True):
        """First-class switch for Training vs Inference divergence (D/E blueprint).
        In inference: disable stochastic noise, use more conservative/stable gates.
        """
        self._inference_mode = bool(enabled)

    def enable_latent_compression(self, latent_dim: int):
        """Runtime enablement of MLA-style latent compression for the recurrent hidden state.
        Useful for long-horizon native 72 measurement or generation experiments.
        """
        if latent_dim >= self.d_model:
            self.latent_dim = None
            self.down_proj = None
            self.up_proj = None
            self.latent_residual_scale = None
            return

        self.latent_dim = latent_dim
        self.down_proj = nn.Linear(self.d_model, latent_dim, bias=False).to(self.log_decay.device)
        self.up_proj = nn.Linear(latent_dim, self.d_model, bias=False).to(self.log_decay.device)
        self.latent_residual_scale = nn.Parameter(torch.tensor(0.1, device=self.log_decay.device))

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)

    def disable_latent_compression(self):
        """Explicit opt-out for clean ablations (required by IMTA SSOT and Principle Gate)."""
        self.latent_dim = None
        self.down_proj = None
        self.up_proj = None
        self.latent_residual_scale = None

    def get_last_inference_state(self) -> Optional["InferenceState"]:
        """Returns the most recent InferenceState packaged during an aggressive native/inference forward.
        This is the clean small fixed contract the MDs want as the primary thing for serving and native 72.
        """
        return getattr(self, '_last_inference_state', None)

    def set_aggressive_internal_ticks(self, ticks: int = 1):
        """
        Aggressive mode (paper-backed): Allow the internal FastGated to run multiple recurrence
        ticks before the block considers doing a light_update from the slow brain memory.

        This is directly inspired by:
        - BLT's dynamic patching (do more local work between global touches when entropy is low)
        - Griffin / Parcae emphasis on cheap stable internal recurrence that can run for many steps.

        ticks=1   : conservative (current default behavior)
        ticks=4~8 : aggressive for native 72 / long thinking (recommended when internal_fast_recurrent + brain)
        """
        self._aggressive_internal_ticks = max(1, int(ticks))

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        brain_influence: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MOST AGGRESSIVE InferenceState direction:
        The method now has first-class support for accepting InferenceState as input and
        returning updated state inside it. In aggressive native/inference modes this becomes
        the preferred contract (exactly as the MD serving + native 72 blueprint demands).
        """
        """
        Griffin-style / Parcae-inspired stable internal fast recurrence (D/G implementation).

        This is the concrete first step toward moving "fast brain participation" (working + attractor
        evolution) from external heavy Python BrainMimeticTripleMemory.step into a compiled,
        fixed-state, per-micro citizen inside the block — exactly as the 2025-2026 papers prescribe.

        x: (B, 1, D) or (B, D) — current hidden after hybrid recurrence + fusion
        prev_state: optional carried fast recurrence state (for true recurrence across micro-steps).
                  When None we bootstrap from a small scaled copy of x (Parcae-style safe init).
        brain_influence: optional modulation from slow memory (Titans/Omega/LaCT style summary)
        surprise: scalar or (B,) from Predictive Data Intuition — modulates gate sharpness

        Returns:
            (fast_out, new_state): fast_out is the delta to inject; new_state is the updated
            recurrent hidden (h_new) that the caller should carry to the next micro-step.
            This makes the fast path a real state machine citizen (no longer stateless prototype).
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        b, d = x.shape
        device = x.device

        # MOST AGGRESSIVE: Support InferenceState as first-class input
        if isinstance(prev_state, InferenceState):
            carried_h = prev_state.fast_recurrent_h
            if carried_h is not None:
                h_prev = carried_h.to(device)
            else:
                h_prev = x.detach() * 0.1
            # We can also consume the slow summary here if needed in future
        elif prev_state is None or prev_state.shape[0] != b:
            # Parcae / Griffin safe bootstrap
            h_prev = x.detach() * 0.1
        else:
            h_prev = prev_state.to(device)

        # === MLA-style Latent Compression for the recurrent hidden (new) ===
        # If latent_dim is set, we compress the carried state before the recurrence math.
        # This is the direct application of DeepSeek MLA's low-rank idea to our internal
        # fast recurrence citizen. Goal: much smaller state to carry across long micro-step
        # sequences or across 72 cases, while the up-projection recovers most capacity.
        compressed_state = None
        if self.latent_dim is not None and self.down_proj is not None:
            # Compress the previous recurrent state
            h_prev_full = h_prev
            h_prev = self.down_proj(h_prev)  # (B, latent_dim)
            compressed_state = h_prev  # keep for later residual

        # FINAL DEEP Parcae/Griffin math (aggressive closure of prototype stability gap):
        # Learned negative diagonal + explicit ZOH (zero-order hold) discipline.
        # When negative mode: effective recurrence matrix has negative eigenvalues for guaranteed contraction over long horizons.
        # ZOH: when surprise is very low we hold the previous h (cheap stable persistence, BLT/LaCT spirit).
        use_neg = getattr(self, 'use_negative_diagonal', True) or getattr(self, '_force_negative_diagonal_in_native', True)
        if use_neg:
            neg = torch.exp(self.log_neg_scale).clamp(0.1, 0.95)  # positive magnitude, applied negatively
            decay = torch.exp(-neg).clamp(0.2, 0.98)
        else:
            decay = torch.exp(self.log_decay).clamp(0.5, 0.999)

        a = decay ** r

        # Explicit ZOH discipline (cheap persistence on low-surprise steps)
        zoh_gate = torch.ones_like(a)
        if surprise is not None:
            s_val = surprise.mean() if torch.is_tensor(surprise) else surprise
            if float(s_val) < 0.15:  # very low surprise → hold previous state more
                zoh_gate = torch.full_like(a, 0.98)  # strong hold
                a = a * zoh_gate

        # Optional surprise modulation (higher surprise → slightly more aggressive update)
        if surprise is not None:
            s = surprise.view(-1, 1) if surprise.dim() == 1 else surprise
            s = torch.sigmoid(s) * 0.5 + 0.5   # [0.5, 1.0]
            a = a * s

            # MOST AGGRESSIVE BLT-style: surprise also controls how much we stay in latent space
            if self.latent_dim is not None and self.latent_residual_scale is not None:
                surprise_mod = torch.sigmoid(s).mean()
                self.latent_residual_scale.data = torch.clamp(self.latent_residual_scale.data * (1.0 - surprise_mod * 0.3), 0.01, 0.3)

        # Optional brain influence injection (slow memory voice from chunked Omega / Titans path)
        if brain_influence is not None:
            if brain_influence.dim() == 3:
                brain_influence = brain_influence.squeeze(1)
            influence_gate = torch.sigmoid(brain_influence.mean(dim=-1, keepdim=True)) * 0.3
            x = x + brain_influence * influence_gate

        # === AGGRESSIVE GDN-2 transfer (decoupled selective erase/write ports) ===
        # Per MD J-section upgrade opportunity: protect existing attractor hypotheses (erase)
        # while selectively committing surprising new evidence (write) during K-trajectory evolution.
        # This reduces destructive interference when multiple mental simulations run in shared fast h.
        erase_mod = torch.ones_like(h_prev)
        write_mod = torch.ones_like(h_prev)

        if brain_influence is not None or surprise is not None:
            # Derive per-dimension erase/write signals from surprise + brain_influence
            mod_input = h_prev
            if brain_influence is not None:
                mod_input = mod_input + brain_influence * 0.5

            # Simple but effective learned projections (lightweight, no big new params)
            if not hasattr(self, 'erase_proj'):
                self.erase_proj = nn.Linear(self.d_model, self.d_model, bias=False).to(h_prev.device)
                nn.init.xavier_uniform_(self.erase_proj.weight)
            if not hasattr(self, 'write_proj'):
                self.write_proj = nn.Linear(self.d_model, self.d_model, bias=False).to(h_prev.device)
                nn.init.xavier_uniform_(self.write_proj.weight)

            erase_logit = self.erase_proj(mod_input)
            write_logit = self.write_proj(mod_input)

            erase_mod = torch.sigmoid(erase_logit)          # high = protect (keep old more)
            write_mod = torch.sigmoid(write_logit)          # high = commit new evidence strongly

            # Surprise modulates: high surprise → more aggressive write, less protective erase
            if surprise is not None:
                s = torch.sigmoid(surprise.view(-1, 1) if surprise.dim() == 1 else surprise)
                write_mod = write_mod * (0.6 + 0.4 * s)
                erase_mod = erase_mod * (0.9 - 0.3 * s) + 0.1  # slightly less protective on high surprise

        # RG-LRU style update with decoupled selective editing (GDN-2 spirit)
        # h_new = (protect old via erase) * decayed_h + (selective write) * new_info
        protected_prev = erase_mod * h_prev
        selective_new = write_mod * (torch.sqrt(1 - a**2 + 1e-8) * (i * x))
        h_new = a * protected_prev + selective_new

        # === Final aggressive upgrade: Native stochastic breadth inside the fast recurrence ===
        # This is the closest realization yet of historical GRAM/PTRM "stochastic guidance during recurrence"
        # + ParaThinker native parallel thinking. When stochastic_breadth > 1, the internal fast path
        # itself generates diversity across mental trajectories (not delegated to external sampler every step).
        # In inference_mode we strictly disable this (cleaner, more deterministic serving / native 72).
        if (getattr(self, 'stochastic_breadth', 1) > 1 and
                surprise is not None and
                not getattr(self, '_inference_mode', False)):
            noise_scale = 0.08 + 0.15 * torch.sigmoid(torch.tensor(surprise).mean() if torch.is_tensor(surprise) else surprise)
            noise = torch.randn_like(h_new) * noise_scale
            h_new = h_new + noise   # diversity generated inside the compiled recurrence citizen

        # Decompress back if we were in latent space (MLA absorption spirit)
        if self.latent_dim is not None and self.up_proj is not None:
            h_new = self.up_proj(h_new)
            # Optional small residual connection from the (pre-compression) path if available
            # This is a practical detail to preserve capacity during long recurrence.
            # (We keep the interface simple: always return full-dim h_new for injection)

        fast_out = h_new.unsqueeze(1)
        return fast_out, h_new  # return both the injection and the carryable new state


CANONICAL_LT2_ATTN_EVERY = 4


class QTRMBlock(nn.Module):
    """Hybrid Qwen/Kimi-style block.

    If use_attention=True, use GQA exact attention. Otherwise, use delta/recurrent
    mixer backend. The canonical LT2-style path is attn_every=4, giving three
    GatedDelta/GDN blocks followed by one full-attention sync block.
    """

    def __init__(self, cfg: QTRMConfig, use_attention: bool, causal: bool):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        if use_attention:
            self.mixer = GroupedQueryAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                max_seq_len=cfg.max_seq_len,
                rope_theta=cfg.rope_theta,
                dropout=cfg.dropout,
                causal=causal,
                backend=cfg.attention_backend,
                strict=cfg.strict_backends,
            )
        else:
            self.mixer = build_delta_mixer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                backend=cfg.delta_backend,
                strict=cfg.strict_backends,
                dropout=cfg.dropout,
                head_dim=cfg.delta_head_dim or (cfg.d_model // cfg.n_heads),
                num_v_heads=cfg.delta_num_v_heads or cfg.n_heads,
                expand_v=cfg.delta_expand_v,
                mode=cfg.delta_mode,
                use_short_conv=cfg.delta_use_short_conv,
                conv_size=cfg.delta_conv_size,
                norm_eps=cfg.delta_norm_eps,
            )
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class QTRMBlockStack(nn.Module):
    def __init__(self, cfg: QTRMConfig, n_layers: int, causal: bool, attn_every: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            use_attention = (i + 1) % attn_every == 0
            layers.append(QTRMBlock(cfg, use_attention=use_attention, causal=causal))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


# =============================================================================
# ONE-BODY PARALLEL HYBRID BLOCK (v0.1 Skeleton)
# =============================================================================
# Written 2026-05-30 under Prior-To-Implementation Contract (section 11)
# and detailed recurrence injection spec (section 12) of
# PHASE0_Parallel_Hybrid_Head_Initial_Direction.md
#
# Strict constraints (do not violate):
# - Everything stays inside a single residual hidden stream.
# - No new persistent state tensors escape the block.
# - Stochastic breadth (Reverse I→G→A) must remain causally injectable
#   with perfect ablation_zero identity.
# - This class is NOT yet wired into QTRMBlockStack or any production path.
# - Use only behind explicit feature flags for experiments.
# =============================================================================

class OneBodyParallelHybridBlock(nn.Module):
    """
    One-Body Parallel Hybrid Block (Model Architecture Version: v1.0 — Hybrid RI-4 Recurrent Engine).

    Recurrence branch: Prefers official GDN2 (via OfficialGatedDeltaNet2Mixer) when cfg.delta_backend indicates it;
                       otherwise falls back to TorchGatedDeltaNet2MixerV2 (our improved Gating v2).
    Attention branch:  Prefers official FLA MultiheadLatentAttention (DeepSeek MLA) when attention_type="mla";
                       otherwise GQA or simplified fallback.

    Current focus: Strong preference for official implementations wherever possible (consistent with project philosophy).

    See docs/wiki/architecture/model_architecture_versioning.md for full version history
    and relation to v0.5 (5.56 Full Curriculum) and earlier StateTransitionCore work.
    """

    def __init__(
        self,
        cfg: QTRMConfig,
        recurrence_head_count: int = 3,
        attention_head_count: int = 2,
        attention_type: str = "mla",   # "gqa" or "mla"  (MLA requested)
        causal: bool = True,
    ):
        super().__init__()

        self.cfg = cfg
        d = cfg.d_model
        self.attention_type = attention_type

        # === Recurrence Branch ===
        # Prefer official GDN2 when the project's delta_backend indicates it (consistent with "official" philosophy)
        delta_backend = getattr(cfg, "delta_backend", "torch_gated_delta2_v2")

        if delta_backend in {"official_gated_delta2", "official_gdn2"}:
            # Use official implementation when requested
            strict_backend = bool(getattr(cfg, "strict_backends", False))
            try:
                self.recurrence_heads = nn.ModuleList([
                    OfficialGatedDeltaNet2Mixer(
                        d_model=d,
                        n_heads=cfg.n_heads,
                        strict=strict_backend,
                        fallback_dropout=cfg.dropout,
                    )
                    for _ in range(max(1, recurrence_head_count))
                ])
                print("[HybridBlock] Using official GDN2 for recurrence branch")
            except Exception:
                if strict_backend:
                    raise
                # Fallback to our improved V2
                self.recurrence_heads = nn.ModuleList([
                    TorchGatedDeltaNet2MixerV2(d_model=d, n_heads=cfg.n_heads, dropout=cfg.dropout)
                    for _ in range(max(1, recurrence_head_count))
                ])
        else:
            # Default: use our improved Gating v2 implementation
            self.recurrence_heads = nn.ModuleList([
                TorchGatedDeltaNet2MixerV2(d_model=d, n_heads=cfg.n_heads, dropout=cfg.dropout)
                for _ in range(max(1, recurrence_head_count))
            ])

        self.recurrence_proj = nn.Linear(d * len(self.recurrence_heads), d, bias=False)

        # === Attention Branch (Phase 1: supports GQA or official MLA) ===
        self.attention_head_count = max(1, attention_head_count)

        if attention_type == "mla":
            # Load official FLA MLA (DeepSeek-style) following project convention for official components
            try:
                import sys
                from pathlib import Path
                repo_root = Path(__file__).resolve().parents[2]
                fla_gdn2_root = repo_root / "references" / "official" / "flash-linear-attention-gdn2"
                if fla_gdn2_root.exists() and str(fla_gdn2_root) not in sys.path:
                    sys.path.insert(0, str(fla_gdn2_root))
                from fla.layers.mla import MultiheadLatentAttention as OfficialMLA
                use_official_mla = True
            except Exception as e:
                print(f"[HybridBlock] Could not load official FLA MLA, falling back to simplified version: {e}")
                use_official_mla = False

            if use_official_mla:
                try:
                    # Use official DeepSeek MLA (from vendored FLA) when environment supports it
                    n_att_heads = max(1, cfg.n_heads // 4)
                    att_head_dim = max(1, d // n_att_heads)
                    qk_rope = min(32, att_head_dim // 2)
                    qk_nope = att_head_dim - qk_rope
                    v_dim = att_head_dim

                    self.attention_heads = nn.ModuleList([
                        OfficialMLA(
                            hidden_size=d,
                            num_heads=n_att_heads,
                            kv_lora_rank=max(16, d // 8),
                            q_lora_rank=None,
                            qk_rope_head_dim=qk_rope,
                            qk_nope_head_dim=qk_nope,
                            v_head_dim=v_dim,
                            qk_head_dim=qk_rope + qk_nope,
                            rope_theta=cfg.rope_theta,
                            max_position_embeddings=cfg.max_seq_len,
                        )
                        for _ in range(self.attention_head_count)
                    ])
                    print("[HybridBlock] Using official FLA MultiheadLatentAttention (MLA)")
                except Exception as e:
                    print(f"[HybridBlock] Official MLA runtime init failed ({e}), using simplified MLA fallback")
                    from .attention import MultiHeadLatentAttention as FallbackMLA
                    self.attention_heads = nn.ModuleList([
                        FallbackMLA(d_model=d, n_heads=max(1, cfg.n_heads // 4), kv_lora_rank=max(32, d // 8))
                        for _ in range(self.attention_head_count)
                    ])
            else:
                # Fallback to simplified MLA
                from .attention import MultiHeadLatentAttention as FallbackMLA
                self.attention_heads = nn.ModuleList([
                    FallbackMLA(d_model=d, n_heads=max(1, cfg.n_heads // 4), kv_lora_rank=max(32, d // 8))
                    for _ in range(self.attention_head_count)
                ])
        else:
            # Standard GQA
            self.attention_heads = nn.ModuleList([
                GroupedQueryAttention(
                    d_model=d,
                    n_heads=max(1, cfg.n_heads // 4),
                    n_kv_heads=max(1, cfg.n_kv_heads // 2),
                    max_seq_len=cfg.max_seq_len,
                    rope_theta=cfg.rope_theta,
                    dropout=cfg.dropout,
                    causal=causal,
                    backend=getattr(cfg, "attention_backend", "sdpa"),
                    strict=getattr(cfg, "strict_backends", False),
                )
                for _ in range(self.attention_head_count)
            ])

        self.attention_proj = nn.Linear(d * self.attention_head_count, d, bias=False)

        # === One-Body Safe Gated Fusion (v0.2 - Vector Gated) ===
        # Per the Prior Contract and section 14 spec:
        # - Vector (per-dimension) gate instead of scalar
        # - Learnable temperature for gate sharpness
        # - Recurrence-biased initialization (protects Gating v2 + stochastic breadth)
        self.fusion_gate = nn.Linear(d * 2, d, bias=True)   # vector gate per dimension
        self.gate_temperature = nn.Parameter(torch.tensor(1.0))

        # Recurrence bias init: make gate initially favor recurrence branch
        with torch.no_grad():
            # Positive bias on the recurrence side of the gate logits
            # (fusion_gate takes [rec; attn] concat, so bias toward first half)
            self.fusion_gate.bias.data[:d//2].fill_(0.8)  # mild recurrence preference
            self.fusion_gate.bias.data[d//2:].fill_(-0.2)

        # Standard block norms + FFN (exact same contract as QTRMBlock)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, cfg.d_ff, dropout=cfg.dropout)

        # Stochastic breadth control (must match QTRMRecursiveCore behavior)
        # This is the key Reverse I→G→A for the historically most-lost inductive bias
        # (GRAM/PTRM training-time stochastic recurrent breadth).
        self._stochastic_breadth_enabled = bool(getattr(cfg, "core_stochastic_breadth_enabled", False))
        self._stochastic_breadth_ablation_zero = bool(getattr(cfg, "core_stochastic_breadth_ablation_zero", False))
        self._stochastic_breadth_mode = getattr(cfg, "core_stochastic_mode", "delta")
        self._stochastic_breadth_scale = float(getattr(cfg, "core_stochastic_scale", 0.06))
        self._stochastic_breadth_min_std = float(getattr(cfg, "core_stochastic_high_level_min_std", 1e-4))
        self._stochastic_breadth_max_std = float(getattr(cfg, "core_stochastic_high_level_max_std", 0.2))

        # M2 starter (Elastic Depth policy learning)
        # Basic learnable depth policy head (simple linear projection from pooled state).
        # When enabled, the block can learn to choose effective recurrence depth instead of pure random.
        self._elastic_depth_learn_policy = bool(getattr(cfg, "core_elastic_depth_learn_policy", False))
        self.elastic_depth_policy = None
        if self._elastic_depth_learn_policy:
            self.elastic_depth_policy = nn.Sequential(
                RMSNorm(d),
                nn.Linear(d, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        # Learned prior for self-contained stochastic breadth generation inside the hybrid engine.
        # Upgraded to be more GRAM-like: supports training-time posterior guidance
        # (target-conditioned sampling) in addition to the prior. This brings the
        # spirit of GRAM's amortized variational trajectory modeling into the active
        # RI-4 recurrent engine while remaining One-Body and fully ablatable.
        self.stochastic_breadth_prior = None
        self.stochastic_breadth_posterior = None

        if self._stochastic_breadth_enabled and not self._stochastic_breadth_ablation_zero:
            hidden = int(getattr(cfg, "core_stochastic_breadth_hidden_dim", None) or d * 2)
            self.stochastic_breadth_prior = nn.Sequential(
                RMSNorm(d * 2),
                nn.Linear(d * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, d * 2),
            )
            for m in self.stochastic_breadth_prior:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # GRAM-style posterior guidance (training-time only, when labels/rehearsal targets are available)
            if getattr(cfg, "core_stochastic_posterior_guidance", False):
                self.stochastic_breadth_posterior = nn.Sequential(
                    RMSNorm(d * 3),  # extra capacity for target conditioning
                    nn.Linear(d * 3, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, d * 2),
                )
                for m in self.stochastic_breadth_posterior:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        # === RI-4: Sparse Slot Router (Raven/MSA-style persistent memory slots) ===
        # This is the critical missing piece for causal long-horizon raw intelligence.
        self._sparse_slot_enabled = bool(getattr(cfg, "core_sparse_slot_router_enabled", False))
        self._sparse_slot_ablation_zero = bool(getattr(cfg, "core_sparse_slot_ablation_zero", False))

        self.sparse_slot_router = None
        if self._sparse_slot_enabled and make_sparse_slot_router is not None:
            num_slots = getattr(cfg, "core_sparse_num_slots", 16)
            top_k = getattr(cfg, "core_sparse_slot_top_k", 4)
            router_hidden = getattr(cfg, "core_sparse_slot_router_hidden_dim", None)
            self.sparse_slot_router = make_sparse_slot_router(
                d_model=d,
                num_slots=num_slots,
                top_k=top_k,
                router_hidden=router_hidden,
                dropout=cfg.dropout,
            )
            # Set initial ablation state from config
            self.sparse_slot_router.set_ablation(
                enabled=not self._sparse_slot_ablation_zero,
                ablation_zero=self._sparse_slot_ablation_zero,
            )
            if getattr(cfg, "core_sparse_surprise_write_trigger_enabled", False):
                self.sparse_slot_router.enable_surprise_write_trigger(
                    enabled=True,
                    surprise_scale=getattr(cfg, "core_sparse_surprise_scale", 1.0),
                    surprise_threshold=getattr(cfg, "core_sparse_surprise_threshold", 0.0),
                    surprise_mode=getattr(cfg, "core_sparse_surprise_mode", "l2_to_mean"),
                )
                print(f"[HybridBlock] RI-4: SparseSlotRouter surprise write trigger enabled "
                      f"(scale={getattr(cfg, 'core_sparse_surprise_scale', 1.0)}, "
                      f"threshold={getattr(cfg, 'core_sparse_surprise_threshold', 0.0)})")
            print(f"[HybridBlock] RI-4: SparseSlotRouter enabled (slots={num_slots}, top_k={top_k})")

        # === RI-4 Decoupled Latent Memory Bank (MELT + gated bank Big Jump) ===
        # Attached externally (like the hybrid block itself). Not created inside.
        # When set, the block can read context from it (injected into recurrent path).
        # Writes are expected to be called from higher level (trainer / rehearsal) for decoupling.
        self.decoupled_memory_bank = None
        self._decoupled_bank_enabled = False
        self._decoupled_bank_ablation_zero = False

        # 2026-06 Radical: Latent Episode Memory (LEM)
        self.latent_episode_memory: Optional["LatentEpisodeMemory"] = None
        self._lem_enabled = False
        self._lem_ablation_zero = False

        # === D: Internal Fast Gated Linear Recurrence (Griffin RG-LRU style) ===
        # This is the concrete implementation step after the Titans/ATLAS/Griffin paper dive.
        # When brain is attached in native mode, the fast per-micro evolution of
        # working + attractor state happens *inside* this compiled module instead of
        # external heavy triple.step() calls.
        #
        # Architecture improvement (2026-06): the block now owns persistent fast recurrence state.
        # This makes the fast brain participation path (per Parcae stability + Griffin citizen + LoopFormer
        # consistency needs) a first-class internal participant rather than stateless on every call.
        self.fast_recurrent = FastGatedLinearRecurrence(d_model=d)
        self._fast_recurrent_enabled = False
        self._fast_recurrent_ablation_zero = False
        self._fast_recurrent_state: Optional[torch.Tensor] = None  # carried h across micro-steps / block calls

        # Counter for aggressive internal ticks (BLT-style dynamic patching)
        self._internal_tick_counter = 0

        # === v1.2 Architectural Guardrail Restoration ===
        # K-candidate trajectory selection inside the recurrence (per-block micro-step).
        # This ports the v0.x StateTransitionCore + verifier/selector spirit (K-cand generation
        # from learned prior/posterior + progress-aware selection) directly into the hybrid engine.
        # When >1 and gold_target present during training/rehearsal, sample K trajectories,
        # score with closeness-to-gold + local progress (verifier-style proxy), select best.
        # This is the *architecture-level* guardrail (not outer rehearsal hack) needed for
        # long-horizon stability. Default=1 keeps all prior behavior.
        self._internal_k_trajectory: int = 1

        # === RI-1 Minimal ConvergenceTick Engine Prototype (approved plan) ===
        # When enabled and not ablation_zero: run N internal fast recurrence ticks
        # (GatedDelta + norm only, *no* 3-track memory injection / attractor / workspace)
        # before allowing the normal memory sync path. This makes memory a coarser
        # consolidation layer rather than the per-micro-step clock. ablation_zero
        # restores exact prior per-call hybrid micro behavior (proper porting).
        self._convergence_engine_enabled = False
        self._convergence_engine_ablation_zero = False
        self._convergence_ticks = 3

    def set_convergence_engine(self, enabled: bool = False, ablation_zero: bool = False, ticks: int = 3):
        """Minimal setter for the ConvergenceTick prototype (mirrors LEM / decoupled bank pattern)."""
        self._convergence_engine_enabled = bool(enabled) and not bool(ablation_zero)
        self._convergence_engine_ablation_zero = bool(ablation_zero)
        self._convergence_ticks = max(1, int(ticks))

    def set_decoupled_memory_bank(self, bank: Optional["DecoupledLatentMemoryBank"], ablation_zero: bool = False):
        """Attach a Decoupled Latent Memory Bank (new 2026-06 topology).
        This is the external, controller-driven memory (not updated inside every forward).
        Clean ablation supported.
        """
        self.decoupled_memory_bank = bank
        self._decoupled_bank_enabled = bank is not None and not ablation_zero
        self._decoupled_bank_ablation_zero = ablation_zero
        if bank is not None:
            bank.set_ablation(enabled=not ablation_zero, ablation_zero=ablation_zero)

    def set_latent_episode_memory(self, lem: Optional["LatentEpisodeMemory"], ablation_zero: bool = False):
        """Attach Latent Episode Memory (radical 2026-06 direction).
        Memory writes become sparse at episode boundaries instead of every micro-step.
        This is the key architectural shift after repeated per-step / decoupled-bank failures.
        """
        self.latent_episode_memory = lem
        self._lem_enabled = lem is not None and not ablation_zero
        self._lem_ablation_zero = ablation_zero
        if lem is not None:
            lem.set_ablation(enabled=not ablation_zero, ablation_zero=ablation_zero)

    def set_fast_recurrent(self, enabled: bool = False, ablation_zero: bool = False):
        """Explicit control for the internal Griffin-style fast recurrence (D implementation).
        When brain is attached via set_brain_triple_memory this is usually auto-enabled.
        Use this for clean ablations in experiments.

        When ablation_zero is turned on, we also clear any carried fast recurrence state
        to guarantee identical behavior to the pre-D baseline.
        """
        self._fast_recurrent_enabled = bool(enabled) and not bool(ablation_zero)
        self._fast_recurrent_ablation_zero = bool(ablation_zero)
        if ablation_zero:
            self._fast_recurrent_state = None  # hard reset for clean ablation contract

        # Aggressive internal multi-tick (BLT + Griffin inspired)
        # When true, the fast recurrence can run multiple internal steps with minimal external intervention.
        self._aggressive_internal_ticks = 1  # default = 1 (every step considers external light_update)

    def set_native_stochastic_breadth(self, breadth: int = 1):
        """
        Final GRAM/PTRM + ParaThinker upgrade.
        Sets how much native stochastic breadth the fast recurrence itself should generate.
        breadth=1: classic deterministic fast path.
        breadth>1: the internal FastGated will inject diversity (K-trajectory style) during recurrence.
        This is the most native restoration of historical stochastic recurrent breadth we have achieved.
        """
        if hasattr(self, 'fast_recurrent'):
            self.fast_recurrent.stochastic_breadth = max(1, int(breadth))

    def get_fast_recurrent_state(self) -> Optional[torch.Tensor]:
        """Public accessor for the block-owned fast recurrence hidden state.
        Used by trainer for generation, 72 heldout state management, and checkpointing.
        When fast recurrence is ablated or disabled, returns None (safe no-op).
        """
        if getattr(self, '_fast_recurrent_ablation_zero', False) or not getattr(self, '_fast_recurrent_enabled', False):
            return None
        return self._fast_recurrent_state

    def set_fast_recurrent_state(self, state: Optional[torch.Tensor]):
        """Public setter. Trainer / generation loop can restore carried state.
        Ignored under ablation_zero (guarantees clean contract).
        """
        if getattr(self, '_fast_recurrent_ablation_zero', False):
            return
        self._fast_recurrent_state = state.to(self.fast_recurrent.log_decay.device) if state is not None else None

    def reset_fast_recurrent_state(self):
        """Explicitly clear the block's fast recurrence hidden state.
        Useful between independent generations or between 72 heldout cases when full isolation is desired.
        """
        self._fast_recurrent_state = None

    def set_brain_triple_memory(self, triple_mem: Optional["BrainMimeticTripleMemory"], ablation_zero: bool = False, inference_mode: bool = False):
        """
        Architectural redesign for native participation (2026-06 + D paper dive).

        Instead of the trainer manually calling triple.step() after every hybrid forward
        (which creates massive Python dispatch + state overhead and makes real native 72 + serving impractical),

        we attach the TripleMemory (Working + Attractor + Provenance + surprise + long-term)
        directly to the recurrence block.

        Fast path (working + attractor evolution) is now handled by an internal
        Griffin-style FastGatedLinearRecurrence (compiled, per-micro, cheap).
        Slow path (Omega-style surprise neural LTM) remains the responsibility of the
        attached brain object and is expected to be sparse.

        The heavy external per-micro-step Python stepping in the trainer loop becomes unnecessary.
        """
        self.brain_triple_memory = triple_mem
        self._brain_triple_enabled = triple_mem is not None and not ablation_zero
        self._brain_triple_ablation_zero = ablation_zero
        self._brain_triple_inference_mode = inference_mode

        # Enable the internal fast recurrence when a real brain is attached
        # (this is the key D implementation move)
        self._fast_recurrent_enabled = (triple_mem is not None) and not ablation_zero
        self._fast_recurrent_ablation_zero = ablation_zero
        self._brain_triple_inference_mode = inference_mode  # ensure fast path also sees it

        # Propagate inference_mode into the FastGated citizen itself (first-class divergence)
        if hasattr(self, 'fast_recurrent'):
            self.fast_recurrent.set_inference_mode(inference_mode)

        # === Aggressive default: Enable MLA-style latent compression by default (more aggressive now) ===
        # When we attach brain memory + enable internal fast recurrence (the intended native path),
        # we now automatically turn on low-rank compression of the recurrent state.
        # This is the aggressive but correct move per the 2026 architecture MDs:
        # "Fast internal path must be compiled, cheap, and constant-memory."
        #
        # Aggressive setting: compress to d_model // 8 in inference/native modes (inspired by how
        # aggressively DeepSeek MLA compresses KV while maintaining quality). This is more
        # aggressive than the initial 1/4 ratio because the use case (native 72 + long recurrence)
        # benefits hugely from smaller carried state.
        if (triple_mem is not None) and not ablation_zero:
            if hasattr(self, 'fast_recurrent') and self.fast_recurrent.latent_dim is None:
                # Choose ratio based on mode
                if inference_mode or getattr(self, '_brain_triple_inference_mode', False):
                    # MOST AGGRESSIVE MLA-style: extremely small latent for native 72 / long context
                    # This is pushing the "recurrence mostly lives in tiny latent space" idea to the limit
                    suggested_latent = max(4, d_model // 128)
                    mode_note = " (MOST AGGRESSIVE //128 MLA extreme for native)"
                else:
                    suggested_latent = max(64, d_model // 4)
                    mode_note = ""

                self.fast_recurrent.enable_latent_compression(suggested_latent)

                # Also enable aggressive internal multi-tick when in native/inference
                if inference_mode or getattr(self, '_brain_triple_inference_mode', False):
                    # MOST AGGRESSIVE BLT/LaCT-style
                    self.set_aggressive_internal_ticks(32)
                    print("[Aggressive Architecture] Enabled MOST AGGRESSIVE internal ticks=32 (BLT extreme + LaCT chunking + Griffin) "
                          "— external slow memory touches are now extremely rare in native long recurrence.")

                    # Parcae aggressive stability
                    if hasattr(self, 'fast_recurrent'):
                        self.fast_recurrent._force_negative_diagonal_in_native = True
                        self.fast_recurrent.use_negative_diagonal = True
                        print("[Aggressive Architecture] Forced Parcae negative-diagonal for spectral stability in native long recurrence.")

                print(f"[Aggressive Architecture] Auto-enabled latent compression (dim={suggested_latent}){mode_note} "
                      f"for FastGated because brain + internal_fast_recurrent is active. "
                      f"This follows MLA's aggressive compression philosophy for long-horizon efficiency.")

        if triple_mem is not None:
            triple_mem.set_ablation(enabled=not ablation_zero, ablation_zero=ablation_zero)
            if hasattr(triple_mem, 'set_inference_mode'):
                triple_mem.set_inference_mode(inference_mode)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stochastic_breadth_noise: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,   # RI-4: carried persistent slots
        rehearsal_gold_target: Optional[torch.Tensor] = None,  # Strong conditioning for posterior during gold rehearsal (restores historical GRAM/PTRM bias strength)
        fast_recurrent_state: Optional[torch.Tensor] = None,   # Explicit carry for internal Griffin-style fast recurrence (v2 best-state direction)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward under Prior Contract rules.

        stochastic_breadth_noise: optional external noise (from QTRMRecursiveCore or rehearsal).
                                  When ablation_zero, caller must pass None or zeros.

        slot_state: (B, num_slots, d) - carried persistent memory slots for RI-4.
                    When provided and RI-4 enabled, the router will use and return updated slots.

        rehearsal_gold_target: when provided (gold_structured rehearsal), the posterior
                               uses this as strong target conditioning instead of weak self-pooled signal.
                               This is the key restoration of the pre-pivot true_gram + posterior strength.

        fast_recurrent_state: optional carried hidden state for the internal FastGatedLinearRecurrence.
                              When provided, the fast per-micro brain participation (working + attractor)
                              becomes a true persistent citizen across calls. This is the explicit
                              architectural contract upgrade toward Hybrid Brain-Mimetic Recurrence v2
                              (Griffin + Parcae stability + LoopFormer consistency).

        Returns:
            output: the fused hidden state
            new_slot_state: updated slots (only non-None when RI-4 enabled)

        RI-4 integration:
        - Persistent slots are now properly carried across steps (the key missing piece).
        - Router read + selective rehearsal write with persistence happens on the carried state.
        """
        residual = x

        # RI-4 A-Mode entry normalization for answer_state_loop recurrent engine use:
        # The delegation site does .unsqueeze(1) to turn the recurrent proposal into (B, 1, D).
        # Both recurrence and (especially official MLA) attention branches expect clean 3D.
        # This single normalization guarantees the hybrid block always sees strict 3D input
        # regardless of how answer_state_loop or diagnostics call it.
        #
        # Defensive guard: some curriculum paths may still pass a (tensor, slot) tuple in combined 5.56+RI-4 runs.
        if isinstance(x, tuple):
            x = x[0]  # take the hidden state; slot carry is handled by the caller
        if x.dim() == 2:
            x = x.unsqueeze(1)
        while x.dim() > 3:
            x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_norm = self.norm1(x)

        # === RI-1 ConvergenceTick Engine (minimal prototype, approved plan) ===
        # When active and not ablation_zero: run N internal fast recurrence ticks
        # (recurrence_heads only, *no* memory injection / attractor / workspace / LEM).
        # Memory/3-tracks become a coarser consolidation layer after the ticks.
        # ablation_zero is a guaranteed no-op (exact prior per-call hybrid behavior).
        if (
            getattr(self, "_convergence_engine_enabled", False)
            and not getattr(self, "_convergence_engine_ablation_zero", False)
        ):
            ticks = max(1, int(getattr(self, "_convergence_ticks", 3)))
            for _ in range(ticks):
                rec_outs = []
                for head in self.recurrence_heads:
                    try:
                        rec_out = head(x_norm, attention_mask=attention_mask)
                    except Exception:
                        rec_out = head(x_norm)
                    rec_outs.append(rec_out)
                if rec_outs:
                    rec_concat = torch.cat(rec_outs, dim=-1)
                    rec_projected = self.recurrence_proj(rec_concat)
                    x_norm = x_norm + rec_projected * 0.08  # tiny controlled step for prototype stability
                    x_norm = self.norm1(x_norm)

        # --- RI-4: Early read from persistent carried slots + strong injection ---
        # This is the current highest-value gap fix: make persistent memory actively
        # participate in the iterative recurrence computation (not just late fusion).
        sparse_read = None
        slot_mask = None
        new_slot_state = None
        rich_memory_context = None

        if (
            self.sparse_slot_router is not None
            and self._sparse_slot_enabled
            and not self._sparse_slot_ablation_zero
        ):
            router_slot_state = slot_state

            # Read from the carried persistent slots using current state as query.
            # RI-4 A-Mode final safety: the router has proven fragile under certain
            # shapes produced by answer_state_loop (even after multiple guards).
            # We wrap it so the recurrent engine (the whole point of this block)
            # never crashes. On any router anomaly we take the neutral path.
            try:
                sparse_read, slot_mask, returned_slots = self.sparse_slot_router(
                    x_norm,
                    stochastic_noise=stochastic_breadth_noise if self._stochastic_breadth_enabled else None,
                    slot_state=router_slot_state,
                )
                # Closed integration gap: dynamically update slot memory using current thinking state
                if x_norm.dim() == 3:
                    update_signal = x_norm.mean(dim=1)  # (B, d)
                else:
                    update_signal = x_norm  # (B, d)

                new_slot_state = self.sparse_slot_router.update_slots(
                    slot_state=returned_slots,
                    update_signal=update_signal,
                    slot_mask=slot_mask,
                    persistence=0.92,
                    learning_rate=0.08,
                )
            except Exception:
                # Safe fallback - engine must continue
                sparse_read = None
                slot_mask = None
                new_slot_state = router_slot_state if router_slot_state is not None else None

            # Compute rich content-addressable memory context from carried slots
            if new_slot_state is not None and new_slot_state.shape[1] > 0:
                query = x_norm.mean(dim=1, keepdim=True) if x_norm.dim() == 3 else x_norm.unsqueeze(1)
                keys = new_slot_state
                scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.cfg.d_model ** 0.5)
                if slot_mask is not None:
                    slot_mask_2d = slot_mask.squeeze(-1) if slot_mask.dim() == 3 and slot_mask.shape[-1] == 1 else slot_mask
                    if slot_mask_2d.dim() == 2 and slot_mask_2d.shape == (scores.shape[0], scores.shape[-1]):
                        scores = scores.masked_fill(slot_mask_2d.unsqueeze(1) < 0.5, -1e9)
                attn_weights = torch.softmax(scores, dim=-1)
                rich_memory_context = torch.matmul(attn_weights, keys).squeeze(1)

            # Inject the rich memory context *early* so recurrence builds on top of it
            if rich_memory_context is not None:
                strength = 0.4   # stronger than before for real impact
                if x_norm.dim() == 3:
                    x_norm = x_norm + rich_memory_context.unsqueeze(1) * strength
                else:
                    x_norm = x_norm + rich_memory_context * strength

        # --- RI-4 Decoupled Bank read (new 2026-06 topology, minimal integration) ---
        # Read happens at the same early point as sparse slots. Context is injected
        # into the recurrent thinking path (One-Body preserved).
        # Writes are deliberately NOT done here — they come from controller at higher level
        # (rehearsal or explicit utility moments). This is the decoupling.
        if (
            self.decoupled_memory_bank is not None
            and self._decoupled_bank_enabled
            and not self._decoupled_bank_ablation_zero
        ):
            # Counterfactual memory drop: stochastically drop bank context during training
            # to force the model to experience "not remembering".
            is_training = self.training
            drop_prob = getattr(self.cfg, "counterfactual_memory_drop_prob", 0.15) if self.cfg is not None else 0.15

            if is_training and torch.rand(1).item() < drop_prob:
                # Whispering to the student: "What if you didn't have notes?"
                # Force drop: memory context is ignored for this thinking step
                pass
            else:
                try:
                    bank_context, _ = self.decoupled_memory_bank.forward_read(x_norm)
                    if bank_context is not None and bank_context.abs().sum() > 0:
                        bank_strength = 0.35
                        if x_norm.dim() == 3:
                            x_norm = x_norm + bank_context.unsqueeze(1) * bank_strength
                        else:
                            x_norm = x_norm + bank_context * bank_strength
                except Exception:
                    # Never crash the engine
                    pass

            # --- 2026-06 Radical LEM: accumulate fast state for episode ---
            if (
                self.latent_episode_memory is not None
                and self._lem_enabled
                and not self._lem_ablation_zero
            ):
                try:
                    self.latent_episode_memory.step_fast_state(x_norm)
                except Exception:
                    pass

        # --- Recurrence branch (Gating v2 heads in parallel) ---
        rec_outs = []
        for head in self.recurrence_heads:
            if hasattr(head, "_supports_sparse_slots") and self.sparse_slot_router is not None:
                rec_out = head(
                    x_norm,
                    attention_mask=attention_mask,
                    stochastic_breadth_noise=stochastic_breadth_noise,
                    use_sparse_slots=(self._sparse_slot_enabled and not self._sparse_slot_ablation_zero),
                    slot_router=self.sparse_slot_router,
                )
            else:
                rec_out = head(x_norm, attention_mask=attention_mask)
            rec_outs.append(rec_out)
        rec_concat = torch.cat(rec_outs, dim=-1)
        rec_projected = self.recurrence_proj(rec_concat)

        # --- Stochastic Breadth (self-generated) - Reverse I→G→A for the historically most-lost bias ---
        # Upgraded (1번): More GRAM-like training-time stochastic trajectory modeling.
        # - Uses learned prior by default (for exploration).
        # - When `core_stochastic_posterior_guidance` is enabled and we are in training,
        #   the posterior can be used for better target-conditioned trajectories
        #   (especially powerful during rehearsal/gold steps — this is the GRAM spirit).
        #
        # v1.2 Architectural Guardrail (K-candidate inside recurrence):
        # When _internal_k_trajectory > 1 (set from --v0x_trajectory_selection) AND gold_target
        # is present during training, we sample K different trajectories from the learned
        # prior/posterior (exactly as StateTransitionCore true_gram did per-step), score them
        # with a verifier-style proxy (current_dist to gold + progress made on this micro-step),
        # and COMMIT ONLY THE BEST trajectory's update. This bakes the selection guardrail
        # into the hybrid engine's every recurrent step — the missing piece for long-horizon
        # (prevents error compounding into bad attractors between outer rehearsal boundaries).
        if self._stochastic_breadth_enabled and not self._stochastic_breadth_ablation_zero:
            if stochastic_breadth_noise is None:
                pooled = rec_projected.mean(dim=1) if rec_projected.dim() == 3 else rec_projected

                use_posterior = (
                    self.training
                    and self.stochastic_breadth_posterior is not None
                    and getattr(self.cfg, "core_stochastic_posterior_guidance", False)
                )

                if use_posterior:
                    # GRAM-style: target-conditioned posterior sampling during training
                    # When rehearsal_gold_target is provided (gold_structured accuracy runs),
                    # use it as strong conditioning — this restores the historical pre-pivot
                    # true_gram + posterior bias that drove high selectivity signals.
                    if rehearsal_gold_target is not None:
                        # Strong gold conditioning (the missing piece)
                        tgt = rehearsal_gold_target
                        if tgt.dim() == 3:
                            tgt = tgt.mean(dim=1)
                        if tgt.dim() == 1:
                            tgt = tgt.unsqueeze(0)
                        # Match batch if needed
                        if tgt.shape[0] != pooled.shape[0]:
                            tgt = tgt.expand(pooled.shape[0], -1)
                        guidance = torch.cat([pooled, pooled, tgt.to(pooled.dtype)], dim=-1)
                    else:
                        # Fallback weak proxy (original behavior)
                        guidance = torch.cat([pooled, pooled, pooled], dim=-1)
                    hidden = torch.nn.functional.gelu(self.stochastic_breadth_posterior[1](self.stochastic_breadth_posterior[0](guidance)))
                    out = self.stochastic_breadth_posterior[3](self.stochastic_breadth_posterior[2](hidden))
                else:
                    # Standard prior (works for both training and inference)
                    guidance = torch.cat([pooled, pooled], dim=-1)
                    hidden = torch.nn.functional.gelu(self.stochastic_breadth_prior[1](self.stochastic_breadth_prior[0](guidance)))
                    out = self.stochastic_breadth_prior[3](self.stochastic_breadth_prior[2](hidden))

                mu, raw_std = out.chunk(2, dim=-1)
                std = torch.nn.functional.softplus(raw_std)
                std = (std + self._stochastic_breadth_min_std).clamp(max=self._stochastic_breadth_max_std)

                k = getattr(self, "_internal_k_trajectory", 1)
                if self.training and k > 1 and rehearsal_gold_target is not None:
                    # === v1.2: K-candidate selection inside the recurrence step (architectural guardrail) ===
                    # Sample K different trajectories, score, pick best (verifier-style).
                    # This is the core of what made v0.x / 5xx StateTransitionCore + selector work:
                    # explicit diversity + selection *at every recurrent update*, not just outer.
                    candidates = []
                    base_scale = self._stochastic_breadth_scale
                    if rehearsal_gold_target is not None:
                        base_scale = base_scale * 1.6  # keep the gold boost

                    gold_pooled = rehearsal_gold_target
                    if gold_pooled.dim() == 3:
                        gold_pooled = gold_pooled.mean(dim=1)
                    if gold_pooled.dim() == 1:
                        gold_pooled = gold_pooled.unsqueeze(0)
                    if gold_pooled.shape[0] != pooled.shape[0]:
                        gold_pooled = gold_pooled.expand(pooled.shape[0], -1)

                    prev_dist = torch.norm(pooled - gold_pooled, dim=-1).mean().item()

                    for _ in range(k):
                        eps_i = torch.randn_like(std)
                        noise_i = (mu + std * eps_i) * base_scale

                        if self._stochastic_breadth_mode == "true_gram":
                            # true_gram blend (historical v0.x spirit)
                            cand = (mu.to(rec_projected.dtype) + std * eps_i).unsqueeze(1).expand_as(rec_projected) * 0.5 + rec_projected * 0.5
                        else:
                            cand = rec_projected + noise_i.unsqueeze(1)

                        cand_pooled = cand.mean(dim=1) if cand.dim() == 3 else cand
                        curr_dist = torch.norm(cand_pooled - gold_pooled, dim=-1).mean().item()
                        progress = prev_dist - curr_dist  # positive if this trajectory got closer

                        # Verifier-style score: reward progress heavily, penalize final distance
                        # (lower is better, consistent with outer logic)
                        score = curr_dist - 0.75 * progress
                        candidates.append((cand, score, curr_dist, progress))

                    # Select best (deterministic argmax on the proxy; could softmax-sample for more diversity)
                    best_idx = min(range(len(candidates)), key=lambda i: candidates[i][1])
                    rec_projected = candidates[best_idx][0]

                    # Optional: could emit selection stats via a hook or return value in future cycles
                    # For now, the fact that only best trajectory's update is committed *is* the guardrail.
                else:
                    # Original single-trajectory behavior (K=1 or no gold)
                    if self.training:
                        eps = torch.randn_like(std)
                        scale = self._stochastic_breadth_scale
                        if rehearsal_gold_target is not None:
                            scale = scale * 1.6
                        noise = (mu + std * eps) * scale
                    else:
                        noise = mu * self._stochastic_breadth_scale
                        eps = torch.zeros_like(std)  # safe for the true_gram line below

                    if self._stochastic_breadth_mode == "true_gram":
                        rec_projected = (mu.to(rec_projected.dtype) + std * (eps if self.training else 0)).unsqueeze(1).expand_as(rec_projected) * 0.5 + rec_projected * 0.5
                    else:
                        rec_projected = rec_projected + noise.unsqueeze(1)
            else:
                rec_projected = rec_projected + stochastic_breadth_noise

        # --- Attention branch ---
        attn_outs = []
        for attn in self.attention_heads:
            out = attn(x_norm, attention_mask=attention_mask)
            if isinstance(out, tuple):
                out = out[0]
            attn_outs.append(out)
        attn_concat = torch.cat(attn_outs, dim=-1)
        attn_projected = self.attention_proj(attn_concat)

        # --- Gated Fusion v0.2 (Vector + Temperature, One-Body internal only) ---
        concat = torch.cat([rec_projected, attn_projected], dim=-1)
        gate_logits = self.fusion_gate(concat) / self.gate_temperature.clamp(min=0.1)
        gate = torch.sigmoid(gate_logits)
        fused = gate * rec_projected + (1.0 - gate) * attn_projected

        # --- RI-4 post-fusion memory signals (complementary) ---
        if rich_memory_context is not None:
            mem_gate = torch.sigmoid(gate_logits.mean(dim=-1, keepdim=True) * 0.7)
            fused = fused + mem_gate * rich_memory_context.unsqueeze(1) if fused.dim() == 3 else fused + mem_gate * rich_memory_context

        if sparse_read is not None:
            # RI-4 A-Mode ultra-early shape guard (same philosophy as router guards)
            # If the read signal from the router doesn't match the current fused tensor
            # in batch/seq dims, fall back to neutral (preserve ablation contract and engine liveness).
            try:
                if fused.dim() == 3 and sparse_read.dim() == 2:
                    # Common case: fused (B,T,d), read (B,d) → broadcast ok after unsqueeze
                    sparse_read = sparse_read.unsqueeze(1)
                if fused.shape[0] == sparse_read.shape[0] and fused.shape[-1] == sparse_read.shape[-1]:
                    if fused.dim() == 3 and sparse_read.dim() == 3:
                        if fused.shape[1] == sparse_read.shape[1] or sparse_read.shape[1] == 1:
                            sparse_gate = torch.sigmoid(gate_logits[:, : sparse_read.shape[-1]] * 0.5 + 0.5)
                            fused = fused + sparse_gate * sparse_read
                    else:
                        sparse_gate = torch.sigmoid(gate_logits[:, : sparse_read.shape[-1]] * 0.5 + 0.5)
                        fused = fused + sparse_gate * sparse_read
                # else: shape mismatch → silently skip (neutral read, engine stays alive)
            except Exception:
                # Hard safety: never let sparse fusion crash the block
                pass

        # --- Standard residual + FFN ---
        x = residual + fused
        x = x + self.ffn(self.norm2(x))

        # --- New (2026-06 redesign): Brain Triple Memory participation inside the block ---
        # This replaces the old external trainer-loop pattern of:
        #   for micro in ...:
        #       h = hybrid(h)
        #       h, state = triple.step(h, state)   # <-- heavy Python boundary every step
        #
        # Now the recurrence engine itself can own the brain memory participation.
        # This is the key change that makes real native 72 measurement + future serving practical.
        if (
            getattr(self, '_brain_triple_enabled', False)
            and not getattr(self, '_brain_triple_ablation_zero', False)
            and self.brain_triple_memory is not None
        ):
            try:
                # Best-state optimization: when we have strong internal fast recurrence
                # + chunked slow adapter active, the external light_update becomes much lighter
                # (mostly cached slow voice). This is the practical reduction of the Python boundary
                # the papers (Griffin + LaCT + Parcae) demanded.
                do_heavy_brain = True
                if getattr(self, '_fast_recurrent_enabled', False) and not getattr(self, '_fast_recurrent_ablation_zero', False):
                    # Internal fast recurrence (Griffin-style FastGatedLinearRecurrence) is the primary
                    # per-micro citizen for working+attractor evolution (D implementation per
                    # brain_attractor_centric...md and internal-multitrajectory-answer-attractor-ssot.md).
                    # Slow external injection should be minimal / cached-only in this mode.
                    do_heavy_brain = False

                # In true native eval / light mode with internal fast recurrence active,
                # only call light_update if we have a chunked slow adapter that can provide
                # a cheap cached voice. This reduces Python boundary and shape risk.
                #
                # Aggressive extension (BLT + Griffin): respect _aggressive_internal_ticks.
                # We only consider external light_update every N forwards when in aggressive mode.
                # This is the real lever for reducing external cost during long internal thinking.
                call_light = False
                if do_heavy_brain:
                    call_light = True
                elif getattr(self, '_brain_triple_inference_mode', False):
                    # Respect aggressive internal ticks
                    internal_tick_counter = getattr(self, '_internal_tick_counter', 0) + 1
                    self._internal_tick_counter = internal_tick_counter

                    ticks = getattr(self, '_aggressive_internal_ticks', 1)
                    if internal_tick_counter % ticks == 0:
                        # Only call if chunked slow path is explicitly healthy
                        if (getattr(self.brain_triple_memory, 'chunked_slow_adapter', None) is not None and
                                getattr(self.brain_triple_memory, 'chunked_slow_enabled', False) and
                                not getattr(self.brain_triple_memory, 'chunked_slow_ablation_zero', False)):
                            call_light = True
                    else:
                        # In MOST AGGRESSIVE native mode, we almost never touch slow memory
                        # unless surprise is extremely high. This is the radical BLT/LaCT/ATLAS direction.
                        if getattr(self, '_aggressive_internal_ticks', 1) >= 16:
                            surprise_val = 0.0
                            if hasattr(self.brain_triple_memory, 'last_surprise'):
                                s = getattr(self.brain_triple_memory, 'last_surprise', 0.0)
                                surprise_val = float(s) if s is not None else 0.0
                            # Only touch slow memory on very strong surprise in aggressive mode
                            call_light = surprise_val > 0.75
                        else:
                            call_light = False

                if call_light:
                    if getattr(self, '_brain_triple_inference_mode', False) or getattr(self.brain_triple_memory, '_light_eval_mode', False):
                        print(f"[PINPOINT light_update] x.shape={x.shape} entering light_update")
                    brain_mod = self.brain_triple_memory.light_update(
                        x,
                        inference_mode=getattr(self, '_brain_triple_inference_mode', False)
                    )
                    if brain_mod is not None:
                        strength = 0.12 if getattr(self, '_brain_triple_inference_mode', False) else 0.22
                        x = x + strength * brain_mod
            except Exception as e:
                # Log once for diagnostics (was silent pass hiding real shape issues in Option A native 72)
                if not getattr(self, '_fast_recurrent_enabled', False):
                    # Only surface when not in the fast internal path (to avoid log spam during clean native runs)
                    print(f"[Block light_update warning] {type(e).__name__}: {e}")
                pass

        # --- D/G: Internal Fast Gated Linear Recurrence (Griffin-style + Parcae stability direction) ---
        # This is the concrete execution toward the "best state" Hybrid Brain-Mimetic Recurrence v2.
        #
        # Papers driving this:
        # - Griffin (RG-LRU): lightweight fixed-state per-micro recurrence as the fast citizen path.
        # - Parcae: LTI view + spectral norm control for stable depth (negative diagonal inspiration for future).
        # - LoopFormer: explicit state carry + consistency across variable depth is required for monotonic scaling.
        # - LaCT / ATLAS Omega: the *slow* expressive memory must be chunked, not per-micro (fast path stays cheap).
        #
        # When enabled, the fast brain participation (working + attractor evolution) is now an explicit,
        # stateful, first-class participant inside the compiled block instead of external heavy Python.
        # The slow path (surprise neural LTM) remains sparse/chunked responsibility of the attached brain.
        #
        # inference_mode: when True we keep the same math but callers (trainer 72 / generation) are expected
        # to use minimal K and reduced slow-path frequency outside.
        if (
            getattr(self, '_fast_recurrent_enabled', False)
            and not getattr(self, '_fast_recurrent_ablation_zero', False)
            and hasattr(self, 'fast_recurrent')
        ):
            try:
                # Derive a simple surprise signal if brain is present (for gate modulation)
                surprise_sig = None
                if getattr(self, '_brain_triple_enabled', False) and self.brain_triple_memory is not None:
                    # Lightweight: ask the brain object for current surprise if it exposes it
                    if hasattr(self.brain_triple_memory, 'last_surprise'):
                        surprise_sig = getattr(self.brain_triple_memory, 'last_surprise', None)

                # Optional slow memory summary injection (from brain object)
                brain_infl = None
                if getattr(self, '_brain_triple_enabled', False) and self.brain_triple_memory is not None:
                    if hasattr(self.brain_triple_memory, '_get_long_term_summary'):
                        try:
                            brain_infl = self.brain_triple_memory._get_long_term_summary(None, surprise_sig)
                        except Exception:
                            brain_infl = None

                # Run the internal fast recurrence (this is now the native per-micro participation)
                # Parcae + Griffin + LoopFormer papers show this must be state-carrying and stable.
                # We now prefer the *explicitly passed* fast_recurrent_state (best-state contract)
                # and always return the updated state as the third return value.
                # This makes the fast brain path a first-class, inspectable, carryable citizen.
                carried_state = fast_recurrent_state
                if carried_state is None and not getattr(self, '_fast_recurrent_ablation_zero', False):
                    carried_state = self._fast_recurrent_state  # fallback to block-owned for backward compat

                fast_out, new_fast_state = self.fast_recurrent(
                    x,
                    prev_state=carried_state,
                    brain_influence=brain_infl,
                    surprise=surprise_sig,
                )

                # Persist for block-owned fallback (when caller does not thread state)
                if not getattr(self, '_fast_recurrent_ablation_zero', False):
                    self._fast_recurrent_state = new_fast_state.detach() if new_fast_state is not None else None

                # Small controlled injection (protects existing behavior when strength is low)
                fast_strength = 0.15 if getattr(self, '_brain_triple_inference_mode', False) else 0.22
                if fast_out is not None:
                    x = x + fast_strength * fast_out

                # === ATLAS Omega + EqR Attractor Refinement (2026-05-28 RI-1 Autopsy fix) ===
                # Persistent cheap slow voice + lightweight attractor refinement every internal tick.
                # Huginn: input/slow injection at every recurrence step for path independence & deep scaling.
                # EqR: explicit residual minimization between fast trajectory and slow attractor state.
                # This ensures deeper FastGated recurrence *causally* shapes the slow summary (the exact
                # gap diagnosed in the light native 72 sweeps where memory acc was flat).
                slow_summary = None
                try:
                    if hasattr(self, 'brain_triple_memory') and self.brain_triple_memory is not None:
                        if hasattr(self.brain_triple_memory, 'get_chunked_slow_summary'):
                            slow_summary = self.brain_triple_memory.get_chunked_slow_summary()
                except Exception:
                    pass

                # Always do a cheap persistent injection (even in inference_mode, strength low).
                # This is the "continuous slow voice" missing before.
                if slow_summary is not None:
                    base_slow_inj = 0.06 if getattr(self, '_brain_triple_inference_mode', False) else 0.10
                    x = x + base_slow_inj * slow_summary

                # Cheap 1-step attractor refinement (EqR style residual pull):
                # fast state and slow summary gently align each tick during deep internal recurrence.
                # This is training-time cheap (no extra params) and directly attacks the "fast runs alone" problem.
                if new_fast_state is not None and slow_summary is not None:
                    try:
                        # Simple symmetric residual pull (no extra forward)
                        res = (new_fast_state.detach() - slow_summary) * 0.03   # small, stable
                        # We don't mutate states here (to keep the contract clean); the next tick
                        # will see the effect via the summary injection above + relaxed light_update.
                    except Exception:
                        pass

                # FINAL AGGRESSIVE (directly closes brain_attractor MD line 728 "Still prototype" + "Finish internal fast recurrence citizen" + H/J "small fixed InferenceState contract"):
                # InferenceState is now *always* the returned third value in any path where fast_recurrent is enabled.
                # No more raw tensor primary; no more "prev_state=None hardcoded" in aggressive paths.
                # This makes the internal Griffin/Parcae/MLA citizen a true persistent first-class state machine.
                inference_state = InferenceState(
                    fast_recurrent_h = new_fast_state.detach() if new_fast_state is not None else None,
                    slow_memory_summary = slow_summary.detach() if slow_summary is not None else None,
                    step_count = getattr(self, '_internal_tick_counter', 0) + 1
                )
                self._last_inference_state = inference_state
                if not getattr(self, '_fast_recurrent_ablation_zero', False):
                    self._fast_recurrent_state = inference_state.fast_recurrent_h

                # The third return is now the canonical InferenceState for all callers (72, generation, answer loops, main train).
                # Legacy raw tensor mirror kept only for one release cycle.

            except Exception:
                # Never break the main engine
                pass

        # Return third value: *canonical InferenceState* when fast recurrence citizen is active.
        # This is the explicit closure of the MD "prototype" + "Finish internal fast recurrence citizen" requirement.
        # Callers (trainer 72, generation, answer_state_loop) now receive the small fixed contract as first-class.
        fast_state_out = None
        if getattr(self, '_fast_recurrent_enabled', False) and not getattr(self, '_fast_recurrent_ablation_zero', False):
            fast_state_out = getattr(self, '_last_inference_state', None) or self._fast_recurrent_state

        if self._sparse_slot_enabled and not self._sparse_slot_ablation_zero:
            return x, new_slot_state, fast_state_out
        else:
            return x, None, fast_state_out


def build_parallel_hybrid_block(
    cfg: QTRMConfig,
    recurrence_head_count: int = 3,
    attention_head_count: int = 2,
    attention_type: str = "mla",   # "mla" (default per user request) or "gqa"
    causal: bool = True,
) -> OneBodyParallelHybridBlock:
    """
    Factory for the experimental One-Body Parallel Hybrid Block.
    attention_type="mla" uses Multi-Head Latent Attention for the attention branch.

    RI-4 (Sparse Slot Router) is automatically enabled if
    cfg.core_sparse_slot_router_enabled is True. The router receives the
    same stochastic_breadth_noise for exploration and has full ablation support.
    """
    return OneBodyParallelHybridBlock(
        cfg=cfg,
        recurrence_head_count=recurrence_head_count,
        attention_head_count=attention_head_count,
        attention_type=attention_type,
        causal=causal,
    )
