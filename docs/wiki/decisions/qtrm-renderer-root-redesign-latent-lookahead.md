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
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
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
