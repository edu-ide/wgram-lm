# S041 Donor-Preserving Free Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document, design, and smoke-test a donor-preserving path for repairing S040 free-generation collapse without claiming promotion.

**Architecture:** Keep Qwen donor logits as the fluent autoregressive mouth and let QTRM/LoRA contribute a bounded residual delta. The first S041 probe is inference-only: depth and residual scale are swept under a donor/QTRM conflict gate before any new training recipe is promoted.

**Tech Stack:** PyTorch, existing `scripts/192_eval_raw_intelligence.py`, JSONL evaluation outputs, wiki decision records.

---

### Task 1: Research And Wiki Grounding

**Files:**
- Create: `docs/wiki/sources/2026-donor-preserving-looplm-freegen-repair.md`
- Modify: `docs/wiki/decisions/2026-05-30-lora-rank8-depth-sweep-clarification.md`

- [x] **Step 1: Record the paper-backed design axes**

Summarize Relaxed Recursive Transformers, LoopUS, Ouro/LoopLM, LoopFormer, Parcae, ReFT/LoReFT, Proxy-Tuning/ThinkLogit, unlikelihood training, and scheduled/on-policy generation repair.

- [x] **Step 2: Tie the papers to the local failure**

Record that S040 has forced-choice signal but free generation remains collapsed, so S041 must preserve donor logits and train/evaluate the free-running path.

### Task 2: Add The S041 Smoke Runner

**Files:**
- Create: `scripts/262_run_s041_donor_preserving_freegen_sweep.sh`
- Create: `scripts/analyze_s041_freegen_sweep.py`

- [x] **Step 1: Add an executable runner**

Use the accepted S040 rank-8 checkpoint and run donor/core-off/canonical depth plus donor-preserving alpha sweep modes under `--donor-qtrm-conflict-gate`.

- [x] **Step 2: Add a deterministic summarizer**

Summarize per-mode hits, exact matches, average generated token count, dominant completion, and collapse flags.

### Task 3: Execute The Local Smoke

**Files:**
- Create: `reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.jsonl`
- Create: `reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.summary.json`
- Create: `reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.summary.md`

- [x] **Step 1: Run the smoke**

Run:

```bash
bash scripts/262_run_s041_donor_preserving_freegen_sweep.sh
```

- [x] **Step 2: Inspect the summary**

Promote nothing unless a donor-preserving mode beats donor-only on free generation without hiding behind a private renderer.

### Task 4: Record The Result And Commit

**Files:**
- Modify: `docs/wiki/log.md`
- Modify: `docs/wiki/index.md`
- Modify: `docs/wiki/decisions/0001-active-decision-index.md`

- [x] **Step 1: Update wiki result sections**

Record the exact smoke result and next S041 training implication.

- [x] **Step 2: Verify and commit**

Run:

```bash
git diff --check
git status --short
git commit
```
