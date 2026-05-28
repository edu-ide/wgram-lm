# Diagnostic Run for Best-State Architecture (v2 direction)

## Goal
Small, safe, fast diagnostic continuation (10~20 steps) from a good anchor checkpoint,
then immediate 72 heldout strict-B measurement, using the improved architecture:

- Internal FastGatedLinearRecurrence (`--internal_fast_recurrent`)
- BrainMimeticTripleMemory with surprise-driven long-term
- Chunked Slow Memory Adapter (automatically enabled when long-term is on)
- Reduced external Python boundary for the fast path

This is the recommended first measurement after the "아키텍처 개선" phase.

## Recommended Command (Small & Safe)

```bash
PYTHONPATH=. python scripts/train_hybrid_ri4_real_continuation_minimal.py \
  --resume_from experiments/checkpoints/YOUR_BEST_ANCHOR.pt \
  --steps 15 \
  --brain_triple_memory \
  --internal_fast_recurrent \
  --data_intuition_loss_weight 0.04 \
  --batch_size 2 \
  --d_model 256 \                # match your anchor
  --device cuda \
  --dtype float32 \
  --heldout_max_cases 6 \
  --run_72_heldout_only \
  --accept-freeze-risk           # only if you really want native brain in 72
```

### Key Flags Explained

| Flag | Purpose | Recommendation for diagnostic |
|------|---------|-------------------------------|
| `--resume_from` | Start from previous best checkpoint | Required. Use your strongest previous brain-mimetic or hybrid anchor |
| `--steps` | How many training steps to run | 10 ~ 20 is ideal for first diagnostic |
| `--brain_triple_memory` | Activate Working + Attractor + Provenance + surprise | Must use |
| `--internal_fast_recurrent` | Use the new compiled fast path inside the block | Must use (this is the main architecture win) |
| `--data_intuition_loss_weight` | Train the surprise prediction | 0.03 ~ 0.06 |
| `--run_72_heldout_only` | After the short continuation, immediately run 72 heldout | Use for quick feedback |
| `--heldout_max_cases` | Safety cap | 4~8 strongly recommended when brain is active |
| `--batch_size` | Smaller = safer for native brain | 1~4 |

## Expected Behavior (Best State Config)

- Fast thinking (per micro-step) mostly happens inside `FastGatedLinearRecurrence` (compiled, state carrying).
- Slow long-term consolidation happens via `ChunkedSlowMemoryAdapter` (accumulates then commits in chunks).
- External `light_update` / triple.step calls are significantly reduced compared to old versions.
- In 72 heldout mode, it should still force light mode (K=1, writes blocked) for safety.

## After the Run

1. Look at the 72 accuracy numbers (strict-B pure_72 depth sweep).
2. Compare especially d=8 (and d=4, d=12 if available) against your previous best anchor.
3. Check logs for:
   - How often the chunked adapter actually committed
   - Whether internal fast recurrence was active
   - Any "Python boundary" warnings

## If You Want to Do Training First (Not Pure 72)

Remove `--run_72_heldout_only` and just run the short continuation:

```bash
... same flags without --run_72_heldout_only and without --heldout_max_cases ...
```

Then manually run the 72 measurement later using the saved checkpoint.

## Safety Notes

- Always start with small `--steps` (10-20) the first time.
- Use `--heldout_max_cases 4~8` when brain memory is active.
- Monitor GPU memory and temperature.
- If it still feels too slow, fall back to `--internal_fast_recurrent` only (without full brain) for a baseline.

## Next After This Diagnostic

If the short run + 72 shows promising signals (especially improvement or stability at d=8+), then we can consider:
- Longer continuation (100+ steps)
- More aggressive chunk sizes
- Full Parcae upgrade on FastGated
- Proper InferenceState-based generation path

---

**Current date of this recipe**: 2026-06 (after the big "다 진행해" architecture push)
