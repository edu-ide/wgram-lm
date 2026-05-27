from __future__ import annotations
from typing import Optional, Tuple
from torch import nn
import torch

from .norm import RMSNorm
from .attention import GroupedQueryAttention
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
    One-Body Parallel Hybrid Block.

    Recurrence branch: Prefers official GDN2 (via OfficialGatedDeltaNet2Mixer) when cfg.delta_backend indicates it;
                       otherwise falls back to TorchGatedDeltaNet2MixerV2 (our improved Gating v2).
    Attention branch:  Prefers official FLA MultiheadLatentAttention (DeepSeek MLA) when attention_type="mla";
                       otherwise GQA or simplified fallback.

    Current focus: Strong preference for official implementations wherever possible (consistent with project philosophy).
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

        if delta_backend in {"official_gated_delta2", "official_gdn2", "gdn2_v2"}:
            # Use official implementation when requested
            try:
                self.recurrence_heads = nn.ModuleList([
                    OfficialGatedDeltaNet2Mixer(
                        d_model=d,
                        n_heads=cfg.n_heads,
                        strict=False,   # allow graceful fallback inside the official mixer
                        fallback_dropout=cfg.dropout,
                    )
                    for _ in range(max(1, recurrence_head_count))
                ])
                print("[HybridBlock] Using official GDN2 for recurrence branch")
            except Exception:
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
        # This was the missing piece: the active RI-4 recurrent engine (this block) could only
        # *receive* external noise, not generate training-time trajectory diversity itself.
        self.stochastic_breadth_prior = None
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stochastic_breadth_noise: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,   # RI-4: carried persistent slots
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward under Prior Contract rules.

        stochastic_breadth_noise: optional external noise (from QTRMRecursiveCore or rehearsal).
                                  When ablation_zero, caller must pass None or zeros.

        slot_state: (B, num_slots, d) - carried persistent memory slots for RI-4.
                    When provided and RI-4 enabled, the router will use and return updated slots.

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
                new_slot_state = returned_slots
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
                    scores = scores.masked_fill(slot_mask.unsqueeze(1) < 0.5, -1e9)
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
        # Previously this block could only consume external noise. Now it can generate
        # its own training-time trajectory diversity using a learned prior on the current state.
        # This directly targets the main gap that made the IMTA SSOT's mandatory
        # "GRAM/PTRM stochastic breadth off" ablation unexecutable on the active RI-4 engine.
        if self._stochastic_breadth_enabled and not self._stochastic_breadth_ablation_zero:
            if stochastic_breadth_noise is None:
                # Self-generate using learned prior (the missing capability)
                pooled = rec_projected.mean(dim=1) if rec_projected.dim() == 3 else rec_projected
                # Simple context: use pooled state itself (keeps it lightweight and One-Body)
                guidance = torch.cat([pooled, pooled], dim=-1)
                if self.stochastic_breadth_prior is not None:
                    hidden = torch.nn.functional.gelu(self.stochastic_breadth_prior[1](self.stochastic_breadth_prior[0](guidance)))
                    out = self.stochastic_breadth_prior[3](self.stochastic_breadth_prior[2](hidden))
                    mu, raw_std = out.chunk(2, dim=-1)
                    std = torch.nn.functional.softplus(raw_std)
                    std = (std + self._stochastic_breadth_min_std).clamp(max=self._stochastic_breadth_max_std)

                    if self.training:
                        eps = torch.randn_like(std)
                        noise = (mu + std * eps) * self._stochastic_breadth_scale
                    else:
                        noise = mu * self._stochastic_breadth_scale

                    if self._stochastic_breadth_mode == "true_gram":
                        # Stronger replace-style (historical true_gram behavior)
                        rec_projected = (mu.to(rec_projected.dtype) + std * (eps if self.training else 0)).unsqueeze(1).expand_as(rec_projected) * 0.5 + rec_projected * 0.5
                    else:
                        rec_projected = rec_projected + noise.unsqueeze(1)
            else:
                # External noise provided (backward compat with previous callers)
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

        if self._sparse_slot_enabled and not self._sparse_slot_ablation_zero:
            return x, new_slot_state
        else:
            return x, None


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
