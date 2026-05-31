# QTRM Renderer Root Redesign: Latent Lookahead

Date: 2026-05-07

## Failure

The accepted Ouro answer-halt checkpoint has a real raw-recursive
forced-choice signal, but it still fails greedy generation.

Existing rejected renderer patches:

```text
low-rank LM adapter
greedy-token margin adapter
causal-prefix self-rollout
donor-unembedding head surgery
LM-head-only tuning
hidden bridge tuning
donor-preserving bounded logit delta smoke
```

New generation smoke on the donor-preserving core-forced readout checkpoint:

```text
artifact:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_readout_pref_s160_generation_smoke8.jsonl

result:
  hits: 0/48

mode samples:
  donor_only:        "100002"
  core_off:          "100002"
  core_steps_1:      "24992"
  core_steps_4:      "20000"
  core_steps_8:      "50006"
  delta_off:         "100002"
```

## Gold-Token Rank Probe

Added diagnostic:

```text
scripts/247_probe_qtrm_gold_token_ranks.py
```

Probe artifact:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/gold_token_rank_probe4.jsonl
```

Important correction:

```text
scripts/247_probe_qtrm_gold_token_ranks.py now reports strict rank and
unique_top1 separately. All-zero or tied logits are no longer counted as a
real unique top-1 hit.
```

Tie-aware smoke artifact:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/gold_token_rank_probe2_tieaware.jsonl
```

Tie-aware result:

```text
qtrm_core_off:
  first@1:        2/2
  first_unique@1: 0/2
  all_unique@1:   0/2

qtrm_core_steps_4:
  first@1:        2/2
  first_unique@1: 2/2
  all_unique@1:   0/2
  max-rank mean:  1440
```

Accepted halt-head checkpoint, 4 held-out cases:

```text
qtrm_core_steps_4:
  first@1: 4/4
  all@1:   0/4
  all<=10: 0/4
  max-rank mean: 740

qtrm_core_steps_8:
  first@1: 4/4
  all@1:   0/4
  all<=10: 0/4
  max-rank mean: 740

halt_gate_off:
  first-rank mean: 228008.75
```

Example:

```text
answer: 300015
target tokens: [" ", "3", "0", "0", "0", "1", "5"]
core_steps_4 ranks: [1, 14, 5, 5, 5, 4, 1222]
```

Interpretation:

```text
The halt gate correctly creates an answer-ready first-token state, but the
multi-token numeric continuation is not locally stable. Forced-choice succeeds
because sequence scoring can prefer the correct full answer over a small
distractor set. Greedy generation fails because later gold tokens are not
locally top-ranked.
```

## Big-Structure Doubt

The root architecture would be wrong if we keep trying to patch only:

```text
LM head geometry
single-token CE
token margin
small hidden bridge
donor logit scaling
```

Those patches do not create causal pressure for the answer state to encode the
next several tokens. The model needs a local future-token training signal.

## Prior-Backed Candidates

### Candidate A: Same-Prefix Latent Lookahead Auxiliary

Sources:

```text
docs/wiki/sources/latent-lookahead-renderer.md
```

Architecture:

```text
prompt prefix
-> donor hidden states
-> QTRM recurrent answer loop
-> answer hidden at current prefix
-> auxiliary future-token decoder for next K answer tokens
-> normal LM logits remain the runtime answer path
```

Training:

```text
future-token CE on next K answer tokens from the same prefix
forced-choice preservation KL/margin
halt-gate preservation
no typed/register answer channel
```

Acceptance:

```text
generation smoke4/8 improves above 0
forced-choice smoke8 does not regress from accepted baseline
halt_gate_off or future_aux_off loses the gain
donor/core_off remain lower
```

Status:

```text
implemented as a falsifier scaffold
```

Implementation:

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/248_run_qtrm_ouro_future_token_lookahead_s040.sh
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_future_token_lookahead_s040.yaml
```

Runtime contract:

```text
answer_state_loop_future_token_logits is auxiliary-only.
It does not replace outputs["logits"] and does not create a hidden answer
channel. The canonical answer path remains autoregressive LM logits.
```

Training knob:

```text
--answer-state-loop-future-token-ce-weight
--answer-state-loop-future-token-max-target-tokens
```

Leakage rule:

```text
future-token CE requires --causal-prefix-supervision, so the input prefix does
not already contain the answer tokens being predicted.
```

### Candidate B: Sequence-Energy Self-Reranker

Architecture:

```text
generate multiple candidate continuations
score candidates with QTRM causal forced-choice / sequence energy
select best candidate
```

Interpretation:

```text
useful diagnostic/runtime scaffold, but not enough as canonical architecture
unless the generator itself improves. Rank probe suggests the correct later
tokens can be too far down for ordinary beam search.
```

### Candidate C: Donor Residual-Stream Hook

Architecture:

```text
Qwen donor decoder remains the text renderer
QTRM recurrent core writes small bounded residuals into donor hidden layers
```

Interpretation:

```text
more invasive but better aligned with ReFT/steering-style prior. Try only if
Candidate A fails, because it requires donor forward hooks and layer sweeps.
```

## Decision

Proceed with Candidate A first.

The smallest useful implementation is not another private LM-head patch. It is
a same-prefix future-token auxiliary that forces the accepted answer-state loop
to encode the local continuation needed for greedy generation, while preserving
the universal LM path at runtime.

## QTRM Port Attempt 2026-05-12

Implementation added for the current pure-recursive QTRM branch:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_mandatory_next_token_decoder_future_aux_s040.yaml
scripts/333_run_nexttok_future_aux_smoke.sh
scripts/196_train_pure_recursive_depth_supervised.py
tests/test_pure_recursive_depth_supervised_train_script.py
```

Evaluator correction:

```text
scripts/192_eval_raw_intelligence.py now records:
  gold_answer
  canonical_completion

The older canonical_answer field is kept for compatibility but refers to the
model completion, not the case gold answer. This avoids misreading a failed
completion like 00000000 as the target answer.
```

Alignment correction:

```text
answer_state_loop_future_token_targets now uses the same
causal_prefix_answer_token_ids path as causal-prefix training, including:
  --causal-prefix-skip-leading-whitespace-targets
  --causal-prefix-append-eos-target

This prevents the future-token auxiliary from training on " answer" tokens
while the final path trains on stripped answer tokens.
```

Baseline regate:

```text
artifact:
  local_eval/orthodox_mandatory_nexttok_regate_goldfields

gold_answer: 600054
full completion: 00000000
answer_next_token_decoder_off completion: 00000000
core_state_zero completion: !!!!!!!!
decision: rejected_noncopy_lm_gate
```

Interpretation:

```text
The recurrent core is causal at the surface level because core_state_zero
changes the output. The next-token decoder is not causal in greedy generation
because decoder_off ties full.
```

Future-token S5 smoke:

```text
command:
  OUT_BASE=local_eval/nexttok_future_aux_smoke_s5 \
  STEPS=5 FUTURE_TOKEN_WEIGHT=0.5 SAVE_EVERY=1 \
  bash scripts/333_run_nexttok_future_aux_smoke.sh

gate artifact:
  local_eval/nexttok_future_aux_smoke_s5/gate_1case_s5_future0.5/report.json

full completion: 66666666
answer_recurrent_off completion: 00000000
answer_next_token_decoder_off completion: 66666666
decision: rejected_noncopy_lm_gate
```

Rank probes:

```text
step_000001 full ranks: 6@2,0@1,0@1,0@1,5@4,4@3,EOS@18
step_000003 full ranks: 6@2,0@1,0@1,0@1,5@4,4@3,EOS@17
step_000004 full ranks: 6@1,0@2,0@2,0@2,5@4,4@3,EOS@17
step_000005 full ranks: 6@1,0@2,0@2,0@2,5@4,4@3,EOS@16
```

Interpretation:

```text
The auxiliary can move the first answer token from 0 to 6, but it does not
stabilize the continuation. The result becomes 66666666 instead of 00000000.
This is a partial readout-direction signal, not an L4 answer-synthesis pass.
```

T4 self-rollout continuation:

```text
source:
  local_eval/nexttok_future_aux_smoke_s5/train_s5_future0.5/step_000005.pt

command:
  OUT_BASE=local_eval/nexttok_future_aux_stage_s5_to_t4_selfroll_s3 \
  INIT_CHECKPOINT=local_eval/nexttok_future_aux_smoke_s5/train_s5_future0.5/step_000005.pt \
  STEPS=3 MAX_TARGET_TOKENS=4 FUTURE_TOKEN_WEIGHT=0.25 \
  SELF_ROLLOUT_WEIGHT=1.0 SAVE_EVERY=1 RUN_GATE=0 \
  bash scripts/333_run_nexttok_future_aux_smoke.sh

best training step by teacher-forced/self-rollout telemetry:
  step_000002 final_path_acc=0.75
  step_000002 final_greedy_token_win_rate=0.75
  step_000002 causal_prefix_self_rollout_prefix_mismatch_rate=0.3333

rank probe:
  step_000001 full ranks: 6@2,0@1,0@1,0@1,5@4,4@3,EOS@17
  step_000002 full ranks: 6@2,0@1,0@1,0@1,5@6,4@3,EOS@19
  step_000003 full ranks: 6@2,0@1,0@1,0@1,5@7,4@3,EOS@30
```

Decision:

```text
Rejected as L4/general-LM promotion.

The future-token auxiliary is useful as a diagnostic because it proves the
answer loop can be pushed away from 00000000 and because recurrent_off changes
the output. However, the current next-token decoder is not a faithful port of
the accepted L1 latent-readout reproduction: it does not explicitly consume
the previous generated answer token as a decoder input. It is mostly a hidden
state transformer before the LM head.

Next orthodox candidate:
  port the L1 latent-readout contract more faithfully:
    previous token embedding or BOS
    + core/answer latent state
    -> recurrent decoder cell or small shared recurrent block
    -> LM logits

Reject further future-token weight/self-rollout tuning until that explicit
previous-token-conditioned readout exists and is ablated.
```

## 2026-05-12 Previous-Token Latent Readout

ETD correction:

```text
Encode, Think, Decode (arXiv:2510.07358) is the closest `thinking block`
reference for this bottleneck. Its lesson is not to add a hidden answer
channel, but to keep recursive latent computation inside the ordinary
encode -> think -> decode LM path.
```

Implemented the smallest faithful readout repair:

```text
previous input/generated token embedding
+ core/answer latent hidden
-> answer_state_loop_next_token_decoder_prev_token_fuse
-> answer_state_loop_next_token_decoder_stack
-> lm_head
```

Files:

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_prev_token_readout_s040.yaml
scripts/334_run_prev_token_readout_smoke.sh
tests/test_prev_token_latent_readout.py
```

Shortcut exclusion:

```text
The previous-token path uses the same visible token stream that the LM already
uses for autoregressive next-token prediction. It does not receive the gold
future answer token, a candidate list, or a side solver output. In selected
logit-position training, prev_token_ids are exactly input_ids at the selected
logit positions, matching the standard "logit at position t predicts token
t+1" LM contract.
```

Initialization:

```text
The fusion layer starts as identity on the latent hidden half and zero on the
previous-token half, so loading an older checkpoint does not immediately trash
the readout. Training can then open the previous-token weights.
```

Verification so far:

```text
.venv/bin/python -m py_compile src/wgram_lm/wgram_model.py src/wgram_lm/config.py \
  scripts/196_train_pure_recursive_depth_supervised.py \
  scripts/330_run_mixed_noncopy_lm_gate.py

PYTHONPATH=src .venv/bin/python -m unittest tests.test_prev_token_latent_readout
Ran 3 tests: OK
```

Claim level:

```text
Implementation scaffold only. This is not yet an accepted L4 result. Promotion
requires the smoke gate to show strict greedy generation improvement and a
drop under answer_next_token_decoder_off/core_state_zero/recurrent_off.
```

S5 readout-only smoke:

```text
command:
  HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT_BASE=local_eval/prev_token_readout_smoke_s5 STEPS=5 SAVE_EVERY=0 \
  bash scripts/334_run_prev_token_readout_smoke.sh

checkpoint:
  local_eval/prev_token_readout_smoke_s5/train_s5/last.pt

gate:
  local_eval/prev_token_readout_smoke_s5/gate_1case_s5/report.json

gold_answer: 600054
full completion: 00000000
donor_only completion: 10000
core_off completion: !!!!!!!!
core_state_zero completion: 6054
answer_recurrent_off completion: 00000000
answer_next_token_decoder_off completion: 00000000
decision: rejected_noncopy_lm_gate
```

Rank probe:

```text
full ranks:             6@2,0@1,0@1,0@1,5@4,4@3,EOS@15
decoder_off ranks:      6@2,0@1,0@1,0@1,5@6,4@4,EOS@12
recurrent_off ranks:    6@4,0@1,0@1,0@1,5@8,4@6,EOS@16
core_state_zero ranks:  6@1,0@1,0@3,0@3,5@1,4@1,EOS@1
```

Interpretation:

```text
The previous-token readout is wired and ablatable, but readout-only training is
not enough. The decisive failure is worse: zeroing the core trajectory gives a
better gold-token rank profile than the full recurrent core. That means the
current core state is injecting the wrong first-token bias, so more
decoder-only training would be tuning around a harmful latent state rather
than proving recursive reasoning.

Next causal hypothesis:
  unlock the recurrent core together with the answer loop under the same
  previous-token readout, then require full > core_state_zero. If that still
  fails, stop answer-loop patches and move toward an ETD-style in-path
  encode -> repeated thinking block -> decode experiment.
```

Core+answer S5 smoke:

```text
command:
  HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT_BASE=local_eval/prev_token_readout_core_answer_s5 \
  STEPS=5 SAVE_EVERY=0 TRAINABLE_PARAM_POLICY=core_and_answer_state_loop \
  bash scripts/334_run_prev_token_readout_smoke.sh

gold_answer: 600054
full completion: 60000066
core_state_zero completion: 6054
answer_recurrent_off completion: 00000000
answer_next_token_decoder_off completion: 66666666
decision: rejected_noncopy_lm_gate
```

Rank probe:

```text
full ranks:             6@1,0@1,0@1,0@1,5@4,4@3,EOS@16
decoder_off ranks:      6@1,0@2,0@2,0@2,5@5,4@3,EOS@13
recurrent_off ranks:    6@3,0@1,0@1,0@1,5@8,4@6,EOS@22
core_state_zero ranks:  6@1,0@1,0@2,0@2,5@1,4@1,EOS@1
```

Interpretation:

```text
Unlocking core+answer loop improves the surface from 00000000 to 60000066 and
makes the first four gold tokens top-1. It still fails the suffix and EOS, and
core_state_zero remains better on the rank profile. This is a partial signal,
not a promotion.
```

Suffix-pressure continuation:

```text
command:
  HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT_BASE=local_eval/prev_token_readout_core_answer_t5_later2 \
  INIT_CHECKPOINT=local_eval/prev_token_readout_core_answer_s5/train_s5/last.pt \
  STEPS=5 SAVE_EVERY=0 TRAINABLE_PARAM_POLICY=core_and_answer_state_loop \
  LATER_TOKEN_WEIGHT=2.0 SELF_ROLLOUT_WEIGHT=0.5 \
  bash scripts/334_run_prev_token_readout_smoke.sh

full completion: 00000000
core_state_zero completion: 6054
decision: rejected_noncopy_lm_gate
```

Decision:

```text
Do not continue this as a weight sweep. The targeted suffix-pressure
continuation regressed the useful partial signal. The current answer-loop
family is blocked because the core trajectory is not reliably helpful. Next
orthodox architecture step is an ETD-style in-path thinking-block probe.
```
