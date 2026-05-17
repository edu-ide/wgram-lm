# QTRM-Native 2B/3B vs Qwen3.6-27B Milestones

Date: 2026-05-15.

Status: canonical target contract.

## Goal

Build a QTRM-Native 2B-3B model that beats Qwen3.6-27B on reasoning and memory
benchmarks with roughly 10x fewer parameters.

This is a hard target, not a current claim.

```text
target model:
  QTRM-Native-2B or QTRM-Native-3B

target baseline:
  Qwen3.6-27B

allowed win claim:
  QTRM-native token->embedding->backbone->recursive core->LM logits
  beats Qwen3.6-27B under matched benchmark scoring
```

## Non-Negotiable Boundary

The current Qwen-backbone bridge result is diagnostic only. It does not satisfy
the final goal because Qwen is still called as the runtime backbone.

```text
Qwen runtime backbone + QTRM adapter/residual = QTRM-Bridge diagnostic
Qwen weights copied only as initialization + native QTRM runtime = QTRM-Native
```

The bridge result can justify architecture transfer into QTRM-Native, but it
cannot be counted as a 27B-beating native model.

## Benchmark Targets

Target scores are preserved in
[Qwen3.6-27B Benchmark Targets](../sources/qwen36-27b-benchmarks.md) and in the
local milestone script:

```text
scripts/372_qtrm_native_27b_milestone_status.py
```

The first public target set includes SWE-bench, Terminal-Bench, SkillsBench,
QwenWebBench, NL2Repo, Claw-Eval, MMLU-Pro, GPQA Diamond, AIME 2026, HMMT Feb
2026, and HLE.

## Milestones

## Core-to-LM-to-Healing Dependency

These three milestones are mandatory ordering constraints for the whole 27B
target. They override any shortcut that tries to make language look good before
the recursive loop has been proven causal.

```text
M-A Recursive-core reasoning proof:
  prove the QTRM/TRM recursive core actually improves a held-out reasoning
  score. Passing requires core_off/think0, shallow-depth, and state
  reset/corruption ablations to reduce the same score.

M-B Core-to-LM attachment:
  attach the proven core to the normal LM path. The answer must be produced by
  token embeddings -> native backbone -> recursive core -> core-dependent
  readout -> LM logits -> autoregressive text.

M-C Language healing:
  only after M-A and M-B, run language healing, pretrained initialization,
  partial unfreeze, or larger corpus training. Healing is accepted only if it
  preserves the recursive-core causal gain.
```

Mapping to the public milestone ladder:

```text
M2/M3 must establish M-A.
M3/M4 must establish M-B.
M4+ may do M-C.
M7/M8 cannot be claimed until M-A, M-B, and M-C all hold under benchmark
scoring.
```

Executable status gate:

```text
scripts/372_qtrm_native_27b_milestone_status.py
```

The gate now records the dependency chain explicitly:

```text
core_to_lm_to_healing_dependencies:
  M_A_RECURSIVE_CORE_REASONING_PROOF
  M_B_CORE_TO_LM_ATTACHMENT
  M_C_LANGUAGE_HEALING_AFTER_CORE
```

Refresh wrappers pass the native L5 multifamily reasoning report as the M-A/M-B
evidence:

```text
scripts/380_refresh_m6_status.sh
scripts/385_refresh_m7_status.sh
```

## Current M7 Blocker 2026-05-16

The M7 public benchmark scorer was tightened to reject prompt echo. The current
checkpoint is therefore lower than the earlier optimistic report:

```text
strict report:
  local_eval/m7_qtrm_native_qwen35pre_mmlu_pro_balanced256_eval_strict_20260516/report.json

accuracy:
  1 / 256 = 0.00390625

prompt_echo_rate:
  1.0

invalid_pred_rate:
  0.98046875

pred histogram:
  <empty>: 251
  A: 5
```

Interpretation:

```text
M7 is not currently a knowledge-parity problem. It is first an answer-only
instruction-following problem: the model reconstructs the MCQ prompt instead
of emitting one option letter after Assistant:.
```

Next M7 repair gate:

```text
M7A_PUBLIC_MCQ_ANSWER_ONLY_HEALING:
  public-style MCQ prompt -> single option letter
  reject prompt echo
  reject empty/invalid answer
  reject A-dominated histogram
  preserve M-A/M-B recursive-core causality
```

## Fast-Path Schedule

This is the aggressive execution plan. The dates are not success promises; they
are kill-switches for avoiding endless architecture shopping.

| Milestone | Fast estimate | Max before pivot | Actual duration |
|---|---:|---:|---|
| M0 Target Contract | <= 0.5 day | none | same day on 2026-05-15 |
| M1 Bridge Causal Signal | 0.5 day | 1 day | same day on 2026-05-15; rejected by 3-seed stability gate |
| M2 Native Tiny LM | 0.5-1 day | 1 day | same day on 2026-05-15; accepted from existing native language bootstrap report |
| M3 Native Core Causality | 1 day | 2 days | same day on 2026-05-15; accepted by state-reset ablation |
| M4 Native Language Bootstrap | 2-4 days local, 1-2 days DGX | 4 days | same day on 2026-05-15 for small bootstrap; scale run still pending |
| M5 Qwen3.6 Public-Target Eval Harness | 0.5-1 day | 1 day | same day on 2026-05-15; accepted public-target manifest |
| M6 Scoped Raw-Reasoning Win | 3-7 days after M3/M4 | 7 days | same day on 2026-05-15; rejected until matched Qwen3.6 scoped baseline exists |
| M7 Public Benchmark Parity | 2-4 weeks after stable native language | 4 weeks | started on 2026-05-16; current checkpoint rejected |
| M8 Public Benchmark Win | 1-3 months after parity evidence | 3 months | not started |

## Fastest Strategy

1. Stop treating bridge tuning as the main path. Bridge is only a causal-signal
   detector.
2. After the M1 stability rejection, pivot to native instead of spending more
   time trying to tune one bridge checkpoint into all eval seeds.
3. Use automatic gates only. A run that misses family floor, stability, or
   language non-regression is rejected without discussion.
4. Prefer Qwen tokenizer, vocab, config shape, and pretrained initialization
   wherever compatible with native QTRM. Random-init novelty is a fallback, not
   the fast path.
5. Keep one canonical native stack at a time. Architecture shopping is allowed
   only when a gate identifies a specific bottleneck.
6. Use published Qwen3.6 benchmark scores as the fastest baseline; direct DGX
   rerun is optional for custom/scoped suites.
7. Use DGX only for steps that are compute-bound after the local gate passes.

## Compute Plan

| Milestone | 4090 feasibility | Preferred compute | Fastest path |
|---|---|---|---|
| M0 | yes | local 4090 or CPU | documentation plus executable status gate |
| M1 | yes | local 4090 | short bridge gates only; pivot after stability rejection |
| M2 | yes | local 4090 | Qwen tokenizer/config shape, tiny native path |
| M3 | yes | local 4090 | mandatory core-on/core-off/depth/reset ablations |
| M4 | partial | 4090 smoke, DGX real bootstrap | pretrained Qwen-compatible initialization, short context first |
| M5 | yes for public-target mode; direct Qwen rerun is optional | local 4090 for QTRM plus published Qwen3.6 benchmark targets | use official/public Qwen3.6 scores as fixed targets; run QTRM on matching public tasks/scorers |
| M6 | possible for scoped gates | 4090 first, DGX confirmation | small held-out raw-reasoning suite with strict ablations |
| M7 | no for serious 2B/3B training | DGX | scale only the architecture that passed M2-M6 |
| M8 | no for serious 2B/3B training | DGX plus reproducible eval harness | focus one benchmark win first, not all targets at once |

### M0 Target Contract

Accepted when this document and the executable status gate exist.

Current status: accepted.

### M1 Bridge Causal Signal

Purpose: prove the QTRM recurrent core can causally improve a Qwen-family
logit path before native transfer.

Acceptance:

```text
QTRM-Bridge core_on gain >= 0.05
min family gain >= 0.01
min family core accuracy >= 0.10
language top1 agreement >= 0.50
```

Single-seed evidence:

```text
local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_adapteronly_stepcond_ad128_s400_selectpair_repair_20260515/report.json

gain: 0.064453125
min_family_gain: 0.011764705882352955
min_family_core_accuracy: 0.0935672514619883
language_top1: 1.0
```

Stability evidence:

```text
local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_adapteronly_stepcond_ad128_checksum_repair_stability_20260515/report.json

num_seeds: 3
num_accepted: 1
min_gain: 0.037109375
mean_gain: 0.0546875
min_family_gain: -0.01764705882352942
min_family_core_accuracy: 0.07602339181286549
language_top1: 1.0
```

Current status: rejected as a bridge milestone. The bridge has a real but
unstable causal signal; the fastest path is to transfer the useful constraints
into QTRM-Native instead of spending more time tuning this bridge.

### M2 Native Tiny LM

Purpose: prove donorless QTRM-Native can learn non-degenerate language.

Acceptance:

```text
donor disabled
native tokenizer/embedding/backbone/core/logits path
English and Korean greedy samples are non-repetitive
next-token loss decreases versus random initialization
think0/core-off comparison recorded
```

### M3 Native Core Causality

Purpose: prove the recursive core is not cosmetic.

Acceptance:

```text
core_on > core_off
deeper core > shallow core
state reset/corruption reduces the same score
LM logits remain the only answer path
```

Current evidence:

```text
local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv3_core_ablation_20260515/report.json

think_eval_loss: 0.06253067243833912
think0_loss: 4.403806257955005
thinking_block_off_loss: 4.403806257955005
state_reset_loss: 1.0610178813987357
full_vs_best_shallow_depth: 0.39065462690292
```

Current status: accepted for the small native language checkpoint.

### M4 Native Language Bootstrap

Purpose: scale from tiny text to useful bilingual language behavior.

Acceptance:

```text
English and Korean text non-regression gate passes
generation history saved
repeat/degeneration guard passes
small held-out public text sample perplexity improves
```

### M5 Qwen3.6 Public-Target Eval Harness

Purpose: avoid invalid comparison.

Acceptance:

```text
official/public Qwen3.6-27B target scores saved
public task/scorer mapping saved
QTRM-Native outputs saved
QTRM scoring script saved
direct Qwen3.6 rerun marked optional, not required for public benchmark targets
QTRM-Native result saved
```

Correction:

```text
DGX is not required just to use Qwen3.6-27B public benchmark numbers.
DGX/server is needed only when we want a direct baseline rerun on a custom
suite or a benchmark whose public prompt/scorer setup cannot be reproduced
from published artifacts alone.
```

Current evidence:

```text
local_eval/qwen36_public_target_manifest/report.json

decision: accepted_qwen36_public_target_manifest
comparison_mode: public_qwen36_target_scores
direct_qwen36_rerun_required: false
benchmark_count: 13
artifact_count: 2
source: https://huggingface.co/Qwen/Qwen3.6-27B
```

Limitation:

```text
M5 does not claim a QTRM public benchmark win. It only fixes the public Qwen3.6
targets, scorer mapping, and QTRM artifact manifest so future wins cannot be
claimed against a moving or invalid baseline.
```

### M6 Scoped Raw-Reasoning Win

Purpose: first real 10x-smaller win, but scoped.

Acceptance:

```text
QTRM-Native-2B/3B > Qwen3.6-27B on a held-out raw reasoning/memory gate
core/depth/memory ablations reduce QTRM score
no retrieval, symbolic solver, or hidden answer path
```

Current evidence:

```text
manifest:
  local_eval/m6_scoped_raw_reasoning_manifest/report.json

suite:
  local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl

suite metadata:
  local_eval/m6_scoped_raw_reasoning_suite/metadata.json

Qwen3.6 runner:
  scripts/378_eval_qwen36_scoped_raw_reasoning_baseline.py

Qwen3.6 wrapper:
  scripts/379_run_m6_qwen36_baseline.sh

Qwen3.6 MTP GGUF proxy runner:
  scripts/381_eval_openai_compatible_scoped_reasoning_baseline.py

Qwen3.6 MTP GGUF proxy wrapper:
  scripts/382_run_m6_qwen36_mtp_proxy_baseline.sh

M6 status refresher:
  scripts/380_refresh_m6_status.sh

best QTRM-native candidate:
  local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/report.json

suite_id:
  qtrm_native_text_reasoning_modchain_revchain_checksum_program4_mod32

prompt_protocol:
  operation_definitions_v1

QTRM score:
  full_generation_exact: 0.6067708333333334
  think0_generation_exact: 0.020833333333333332
  core_gain: 0.5859375
  ablation_drop: 0.5716145833333334
  min_family_generation_exact: 0.4140625

Qwen3.6-27B-MTP-GGUF proxy baseline:
  report: local_eval/m6_qwen36_mtp_proxy_baseline/report.json
  score: 0.15364583333333334
  cases: 768
  checksum: 0.38671875
  modchain: 0.046875
  revchain: 0.02734375

decision:
  accepted_m6_scoped_raw_reasoning_win
```

Why accepted:

```text
The matched Qwen3.6-27B-MTP GGUF proxy baseline was measured on the exact
deterministic suite and prompt protocol. QTRM-Native beats it by a large margin,
and core-off/state-reset ablations still collapse the score, so this is a
scoped raw-reasoning win.

Limitation:
  this is a scoped custom-suite win over a quantized local Qwen3.6 GGUF proxy;
  it is not public benchmark parity and not a full-precision Qwen3.6 rerun.
```

Local GGUF proxy command:

```bash
LOG_EVERY=64 OUT_DIR=local_eval/m6_qwen36_mtp_proxy_baseline \
  bash scripts/382_run_m6_qwen36_mtp_proxy_baseline.sh
QWEN36_BASELINE_REPORT=local_eval/m6_qwen36_mtp_proxy_baseline/report.json \
  bash scripts/380_refresh_m6_status.sh
```

DGX llama-server proxy, 2026-05-17:

```text
server script, local control:
  scripts/407_dgx_qwen36_mtp_llama_server.sh

server script, on DGX:
  /mnt/data4tb/qwen36-mtp-llama-server.sh
  /mnt/data4tb/qtrm_multimodal_memoryos/scripts/407_qwen36_mtp_llama_server_dgx_local.sh

endpoint:
  http://192.168.219.113:18082/v1

server:
  /mnt/data4tb/llama-cpp-turboquant-cuda/build/bin/llama-server

model:
  /mnt/data4tb/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf

runtime:
  ctx 131072
  reasoning off
  MTP draft on
  ngram-mod on
  V cache turbo4
  K cache requested turbo4 but auto-upgraded to q8_0 for GQA quality
```

DGX proxy rerun results:

```text
64-case report:
  local_eval/dgx_qwen36_mtp_proxy_baseline_64_20260517/report.json
  score: 7 / 64 = 0.109375

256-case report:
  local_eval/dgx_qwen36_mtp_proxy_baseline_256_20260517/report.json
  score: 35 / 256 = 0.13671875
  checksum: 31 / 85 = 0.36470588235294116
  modchain: 4 / 86 = 0.046511627906976744
  revchain: 0 / 85 = 0.0

512-case report:
  local_eval/dgx_qwen36_mtp_proxy_baseline_512_20260517/report.json
  score: 75 / 512 = 0.146484375
  checksum: 59 / 170 = 0.34705882352941175
  modchain: 12 / 171 = 0.07017543859649122
  revchain: 4 / 171 = 0.023391812865497075

DGX 512 manifest:
  local_eval/m6_scoped_raw_reasoning_manifest_dgx512_20260517/report.json
  decision: accepted_m6_scoped_raw_reasoning_win
```

Interpretation:

```text
The DGX llama-server baseline reproduces the earlier local GGUF-proxy
conclusion: Qwen3.6-27B-MTP-GGUF remains weak on this deterministic scoped
modchain/revchain/checksum suite, while the accepted QTRM-native L5 report is
0.6067708333333334 over 768 cases with large core/state ablation drops.

This keeps M6 accepted as a scoped raw-reasoning win. It still does not imply
public benchmark parity or full-precision Qwen3.6 parity.
```

DGX full-precision command when reachable:

```bash
cd /mnt/data4tb/qtrm_multimodal_memoryos
PYTHONPATH=src MODEL_PATH=/home/sk/ws/llm/models/Qwen3.6-27B \
  bash scripts/379_run_m6_qwen36_baseline.sh
PYTHONPATH=src python scripts/376_build_m6_scoped_raw_reasoning_manifest.py \
  --qtrm-report local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/report.json \
  --qwen36-baseline-report local_eval/m6_qwen36_scoped_baseline/report.json \
  --out-json local_eval/m6_scoped_raw_reasoning_manifest/report.json \
  --out-md local_eval/m6_scoped_raw_reasoning_manifest/report.md
PYTHONPATH=src bash scripts/380_refresh_m6_status.sh
```

### M7 Public Benchmark Parity

Purpose: show that the native model is not only a synthetic-task specialist.

Acceptance:

```text
selected public benchmark scores enter Qwen3.6-27B parity band
language quality does not collapse
seed/checkpoint reproducibility package exists
```

First public smoke, 2026-05-16:

```text
suite materializer:
  scripts/383_materialize_m7_public_reasoning_suite.py

evaluator:
  scripts/384_eval_qtrm_native_public_mcq.py

status refresher:
  scripts/385_refresh_m7_status.sh

dataset:
  TIGER-Lab/MMLU-Pro

suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl

suite limitation:
  category-balanced 256-case subset generated through the local `datasets`
  backend. It is still a subset, not the full 12032-case MMLU-Pro test split.

checkpoint:
  local_eval/qtrm_native_pretrained_init_qwen35_compact_external4500_s3600_20260515/last.pt

checkpoint type:
  QTRM-Native with Qwen3.5 compact tokenizer/pretrained initialization
  runtime_donor: false

report:
  local_eval/m7_qtrm_native_qwen35pre_mmlu_pro_balanced256_eval/report.json

score:
  QTRM-Native Qwen3.5-preinit: 37 / 256 = 0.14453125
  previous byte-BPE tiny native: 16 / 256 = 0.0625
  Qwen3.6 target: 0.862
  parity floor with tolerance 0.02: 0.842

decision:
  rejected_m7_public_benchmark_parity
```

Interpretation:

```text
M6 proved a scoped recursive-core raw-reasoning signal. M7 shows Qwen3.5
pretrained initialization is materially better than the byte-BPE tiny checkpoint
on public MCQ, but it is still nowhere near public benchmark parity. The next
architecture/training step is not another synthetic reasoning tweak; it is
larger native language/knowledge bootstrap with Qwen-compatible pretrained
initialization, then balanced/full MMLU-Pro and GPQA-style evaluation.
```

Strict scorer correction and M7A surface gate, 2026-05-16:

```text
problem:
  The first M7 score was contaminated by prompt/options echo. After scorer
  correction, the compact Qwen-preinit checkpoint dropped to 1/256 with
  invalid_pred_rate 0.98046875 and prompt_echo_rate 1.0.

additional scorer correction:
  The evaluator now decodes only newly generated suffix token ids. This avoids
  mistaking lossy compact-tokenizer prompt reconstruction for generated echo.

compact-vocab diagnosis:
  compact Qwen tokenizer had unk_compact_id == eos_compact_id, so public MCQ
  OOV prompt pieces became EOS/UNK and answer generation collapsed to empty.
  Compact vocab is demoted for public MCQ parity work.

M7A accepted gate:
  script:
    scripts/401_run_qtrm_native_m7a_final_token_healing.sh

  trainer:
    scripts/400_train_qtrm_native_public_mcq_final_token.py

  accepted report:
    local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/m7a_gate_report.json

  checkpoint:
    local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt

  strict eval:
    local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/strict_eval/report.json

  result:
    accepted_m7a_public_mcq_answer_only_gate
    cases: 64
    accuracy: 0.140625
    invalid_pred_rate: 0.0
    prompt_echo_rate: 0.0
    max_pred_fraction: 0.234375
```

Meaning:

```text
M7A closes only the public-MCQ answer-surface bottleneck. The model can now
generate one option letter without echo/empty/single-label collapse on a
held-out public-style slice. This is not M7 parity and not evidence that the
model knows MMLU-Pro. The next M7 work must improve correctness while preserving
M7A and adding core-off/depth ablations.
```

M7B core-depth gate, 2026-05-16:

```text
script:
  scripts/403_run_qtrm_native_m7b_core_depth_gate.sh

scorer:
  scripts/402_score_m7b_core_depth_gate.py

report:
  local_eval/qtrm_native_m7b_core_depth_gate_m7a_s300_20260516/m7b_gate_report.json

decision:
  accepted_m7b_public_mcq_core_depth_gate

checkpoint:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt

strict depth sweep:
  depth0: 6 / 64 = 0.09375, A-only collapse
  depth1: 7 / 64 = 0.109375
  depth2: 7 / 64 = 0.109375
  depth4: 9 / 64 = 0.140625

gain:
  depth4 - depth0: +0.046875
  depth4 - best shallow: +0.03125

surface at depth4:
  invalid_pred_rate: 0.0
  prompt_echo_rate: 0.0
  max_pred_fraction: 0.234375
```

Meaning:

```text
M7B is a small but important public-style causal-depth result. It proves that
the accepted M7A answer path is not only formatting: full recursive depth
improves strict greedy option-letter accuracy over no/shallow thinking on the
same held-out public-style slice. It is not Qwen3.6-27B parity and not broad
MMLU-Pro competence.

Next M7C target:
  raise correctness while preserving M7A/M7B:
    invalid/echo stay near 0
    prediction histogram stays non-collapsed
    full depth keeps beating depth0 and best shallow
  then expand from 64 to 256/512/full public slices.
```

M7B DGX scale-out rerun, 2026-05-17:

```text
64-case DGX rerun:
  local_eval/dgx_m7b_core_depth_gate_m7a_s300_20260517/m7b_gate_report.json
  decision: accepted_m7b_public_mcq_core_depth_gate
  depth0: 6 / 64 = 0.09375
  depth1: 7 / 64 = 0.109375
  depth2: 7 / 64 = 0.109375
  depth4: 9 / 64 = 0.140625
  gain_vs_baseline: +0.046875
  gain_vs_best_shallow: +0.03125

256-case DGX scale-out:
  local_eval/dgx_m7b_core_depth_gate_m7a_s300_256_20260517/m7b_gate_report.json
  decision: rejected_m7b_public_mcq_core_depth_gate
  depth0: 39 / 256 = 0.15234375
  depth1: 23 / 256 = 0.08984375
  depth2: 21 / 256 = 0.08203125
  depth4: 24 / 256 = 0.09375
  gain_vs_baseline: -0.05859375
  gain_vs_best_shallow: -0.05859375
```

Scale-out decision:

```text
The 64-case M7B core-depth gain is reproducible, but it does not scale to 256
held-out public MCQ cases. Treat M7B as a small-slice causal-depth diagnostic,
not a promoted public benchmark result. The next public-MCQ work must repair
core-depth scale-out before any M7C/M7 parity claim.
```

M7C rejected answer-CE variants, 2026-05-16:

```text
new support:
  scripts/404_materialize_aj_remap_mcq.py
  scripts/400_train_qtrm_native_public_mcq_final_token.py
    --preserve-jsonl / --preserve-kl-weight
    --trainable-name-regex

all variants started from:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt

tested:
  A-J remap full tuning:
    7 / 64
  A-J remap low-LR short tuning:
    5 / 64
  natural external MCQ low-LR:
    7 / 64
  external MCQ + base option-KL:
    8 / 64
  MMLU auxiliary + base option-KL:
    8 / 64
  MMLU auxiliary + base option-KL + think/core-only:
    8 / 64
  A-J remap + base option-KL + think/core-only:
    7 / 64
  core-only depth-gain trajectory loss:
    5 / 64
  accepted M7A checkpoint with deeper loops:
    depth4 9 / 64, depth6 6 / 64, depth8 5 / 64, depth12 2 / 64

accepted baseline remains:
  M7A/M7B checkpoint at 9 / 64 strict generation
```

Decision:

```text
Final-answer CE is now a rejected M7C promotion path unless a new objective
changes the core trajectory itself. The next promotion candidate must be a
TRM-style recursive curriculum or latent-state objective that improves the
reasoning path before touching broad MCQ answer tuning again.

Blind loop-depth scaling is also rejected for this checkpoint. Depth4 is the
current sweet spot; deeper recurrent execution degrades held-out strict MCQ
accuracy.
```

M7D rejected knowledge-bootstrap shortcut, 2026-05-16:

```text
new support:
  scripts/405_eval_m7c_checkpoint_soup_in_memory.py
  scripts/406_build_mcq_knowledge_text_corpus.py

corpus:
  local_eval/m7_public_reasoning_suite/mcq_knowledge_text_aux_mmlu_external_20260516.jsonl
  3531 non-test MCQ answer-content records

results:
  checkpoint soup triage:
    best 9 / 64, no baseline improvement
  knowledge-only CE:
    rejected, repeated-word language degradation
  preservation-mix low-LR knowledge CE:
    language sample non-degenerate but semantic gate rejected
  knowledge mix + answer repair + KL continuation:
    next-token diagnostic 9 / 64
    strict M7B depth4 2 / 64 with invalid_pred_rate 0.796875
```

Decision:

```text
Do not promote trainer next-token diagnostics. M7 promotion requires strict
greedy generation. Small MCQ-derived knowledge CE is not enough for the current
64M native backbone; the next serious milestone needs larger native pretraining
or a truly integrated pretrained backbone, then the same M7A/M7B gates.
```

Qwen-integrated bridge revalidation, 2026-05-16:

```text
This remains a QTRM-native candidate only in the single-graph sense:
Qwen tokenizer/backbone -> mandatory QTRM core -> Qwen LM head.

Standalone SSOT public-MCQ reruns did not reproduce previous accepted-looking
M4 reports:
  256 core-only rerun:
    core_off 93 / 256
    core_on  93 / 256
    gain 0.0
    rejected
  256 l23open seed20260520 rerun:
    core_off 93 / 256
    core_on  94 / 256
    gain 0.00390625
    rejected
  512 residual_scale=0.06 rerun:
    core_off 194 / 512
    core_on  196 / 512
    gain 0.00390625
    rejected

Consequence:
  Do not count Qwen-integrated public-MCQ M4 as accepted. It is a diagnostic
  bridge until standalone reruns show repeatable core_on > core_off beyond the
  configured threshold.
```

### M8 Public Benchmark Win

Purpose: final target.

Acceptance:

```text
QTRM-Native-2B/3B exceeds Qwen3.6-27B target score
on at least one selected public benchmark
and preserves core-causality ablation evidence
```

## Immediate Next Actions

1. Close M1 by repairing the family core accuracy floor.
2. Transfer the accepted bridge recipe into a donorless QTRM-Native pretrained
   initialization run.
3. Move to M7 public benchmark parity with the accepted M6 scoped win as a
   diagnostic foothold, not as a public benchmark claim.
4. Keep bridge, teacher, and MemoryOS work diagnostic until the native path
   reproduces the gain.
5. For M7, prioritize native language/knowledge scale and public MCQ answer
   correctness after the accepted answer-only surface gate, before further
   recursive-core architecture shopping.
6. For M7C, do not repeat plain final-token answer CE. Train or constrain the
   recursive trajectory first, then rerun the same M7A/M7B gates.
7. For integrated pretrained backbones, treat training-periodic or old accepted
   reports as non-canonical until a standalone SSOT rerun reproduces the gain.

## Executable Status

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/372_qtrm_native_27b_milestone_status.py \
  --bridge-report local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_adapteronly_stepcond_ad128_s400_selectpair_repair_20260515/report.json
```

The generated report is a target-status ledger, not a training result.
