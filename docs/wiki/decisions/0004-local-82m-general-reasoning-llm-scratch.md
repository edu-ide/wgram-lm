# 0004 Local 82M General Reasoning LLM Scratch Pretrain (Dynamic-BLT + One-Body)

**Date**: 2026-05-30
**Status**: ✅ Phase 1 completed successfully (2000 steps)
**Branch**: s041-donor-preserving-freegen (local 4090)

## Goal
Train a compact, **tokenizer-free "compact reasoning engine"** (~82M–210M parameter regime with dynamic patching) entirely on local RTX 4090 using high-quality math / code / instruction data. Prove that the Dynamic-BLT + recurrent one-body core can serve as a practical local general LLM substrate when data is sufficiently dense.

This is the direct continuation of the RI-gate-validated 82M Dynamic-BLT work (see 0003).

## Architecture (Phase 1 — Foundation)
- **Byte-level Dynamic BLT**: `patch-boundary-mode=hnet_dechunk`, target compression ~2.0–2.8×
- **Recurrent global core**: `decoder-latent-mode=one_body` + `think-structure=trm_dual_z` + `backbone=trm_qwen35_3to1`
- **DeltaNet-2 (GDN2)** official runtime with strict backends
- **Light attractor**: `answer-attractor-ce-weight=0.0` during main pretrain (memory safety). Attractor depths will be re-introduced in Phase 2 finetune on checkpoints.
- Model width: `d_model=384, n_heads=6, d_ff=1024, hybrid_layers=4, local_layers=2`
- Data: `/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled` (UltraData-style math/code/SFT rehearsal mix, ~4.4M tokens)

## Current Run
- **Directory**: `local_eval/20260530_82M_GENERAL_LLM_SCRATCH`
- **Launch script**: `scripts/558_local_82m_general_llm_scratch.sh`
- **Command line highlights**:
  - `--batch-size 3 --seq-len 256 --steps 2000`
  - `--patch-boundary-mode hnet_dechunk --decoder-latent-mode one_body`
  - `--train-think-steps 2 --answer-attractor-ce-weight 0.0` (Phase 1)
  - Activation of dynamic boundary + full GDN2 + one_body recurrent core all verified working.

## Early Results (as of ~step 200)
- `clean_loss`: ~11.25 (step 1) → **5.24** (step 200 eval)
- Compression ratio: stably 2.0–2.8× with active learned boundary probability ~0.34–0.48
- `hnet_dechunked_tokens` and `boundary_prob_rate` both non-zero and evolving → dynamic patching is learning
- Throughput after Triton compile: 1300–1400+ tokens/sec on 4090 (excellent)
- All custom kernels using `official_runtime` (no torch fallback)
- No shape-mismatch spam, no OOM after attractor regularization was disabled for this phase

## Monitoring
- Log: `tail -f local_eval/20260530_82M_GENERAL_LLM_SCRATCH/train.log`
- TensorBoard: http://localhost:6006 (all `local_eval/` runs)
- Checkpoints every 200 steps + `best_eval_model.pt`

## Phase Plan
1. **Phase 1 (current)**: 2000-step foundation with strong dynamic boundary + recurrent core. Goal: drive clean_loss well below 3.5–4.0 while boundary mechanism matures.
2. **Phase 2 (planned)**: Take best checkpoint → short attractor-enabled finetune (`--answer-attractor-ce-weight 0.05 --answer-attractor-depths 1 2 4 --train-think-steps 4`) + possible SFT on the same high-quality mix. This is where the "thinking" capability (RI-1 style fixed-point convergence) will be explicitly strengthened.
3. **Phase 3 (optional)**: Tiny chat/instruction SFT + local CLI playground for qualitative "does it reason?" checks on math/code problems.

## Relation to Prior Work
- Builds directly on the successful 82M RI-gate validation (Dynamic-BLT beat Fixed-BLT by >1.0 loss, perfect attractor fixed-point convergence at depth 8–12).
- Uses the exact same high-quality sampled data track that powered the RI experiments.
- Avoids the memory explosion of full attractor regularization during the long pretrain by deferring it — a pragmatic local-4090 adaptation.

## Next Immediate Actions
- Let Phase 1 run to completion (or at least 800–1000 steps) and record the loss curve + boundary evolution in this document.
- On a strong checkpoint, launch a 200–400 step attractor "thinking" finetune pass.
- Produce a small qualitative eval set (GSM8K-style arithmetic, simple Python synthesis, logic puzzles) for human inspection.

**Verdict target**: A local 82M-scale model that, while not a general knowledge oracle, is a genuinely useful compact reasoning engine for structured domains (math, code, formal logic) thanks to dynamic patching + recurrent state + (later) explicit attractor training.

## Phase 1 Completion Report (2026-05-30)

**Status**: ✅ **SUCCESSFULLY COMPLETED**

- Total steps: **2000 / 2000**
- Wall time: ~3 hours on single RTX 4090
- Final / Best eval clean_loss: **3.903** (achieved at step 2000 — the run kept improving until the very end)
- Initial eval clean_loss: 11.254
- Total improvement: **-7.35** absolute (roughly 65%+ relative reduction)
- Best checkpoint: `best_eval_model.pt` (also `last_model.pt`) — 803 MB

### Dynamic Boundary Behavior (hnet_dechunk)
- Remained active and healthy throughout the entire 2000 steps.
- Early (step ~200): `boundary_prob_rate` ~0.34–0.48, compression 2.0–2.8×
- Late (step 2000): `boundary_prob_rate` ~0.195, compression still ~2.0–2.5×
- The model learned to make sharper, more decisive patch boundaries as training progressed (rate decreased while still producing meaningful latent patches of ~35–60 tokens on average). This is the expected and desired behavior.

### Other Observations
- All training used official GDN2 Triton kernels (`actual_delta_runtime: "official_runtime"`) — zero fallback.
- No OOM, no NaN, no shape mismatch spam after the initial config stabilization.
- Throughput after kernel autotune: consistently 1300–1400+ tokens/sec.
- Checkpoints saved every 200 steps + final best/last.

### Artifacts
- `local_eval/20260530_82M_GENERAL_LLM_SCRATCH/`
  - `best_eval_model.pt` (primary checkpoint for Phase 2)
  - `report.json` (full training + final eval history)
  - `train.log` (~992 KB)
  - `tensorboard/` events

## Recommended Immediate Next Step: Phase 2 Attractor "Thinking" Finetune

The foundation (dynamic patching + recurrent one-body core) is now solid. The next logical move is to take the best checkpoint and explicitly train the **Attractor / multi-step thinking** capability that was intentionally kept light during Phase 1 for memory reasons.

Typical Phase 2 command skeleton (on the same data):
```bash
... 557_train... \
  --resume local_eval/20260530_82M_GENERAL_LLM_SCRATCH/best_eval_model.pt \
  --no-resume-strict \   # or strict, depending on head changes
  --steps 300-500 \
  --batch-size 3 --seq-len 256 \
  --train-think-steps 4 \
  --answer-attractor-depths 1 2 4 \
  --answer-attractor-ce-weight 0.05 \
  --answer-attractor-monotonic-weight 0.01 \
  ...
```

This is where we expect to see the RI-1 style behavior (deeper think steps producing meaningfully lower CE than shallow).

---
*Phase 1 completed successfully — 2026-05-30 16:03 KST*
