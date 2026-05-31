# Pure Recursive Full-Sequence KISS Failure Ledger

Status: rejected, 2026-05-02.

## Failure

The pure recursive core still does not pass the raw-intelligence gate. The
latest runs removed several non-causal shortcuts and training bugs, but no
checkpoint shows held-out depth scaling where `core_steps=4/8` beats donor and
`core_off`.

## Fixed During This Pass

```text
1. Final logits now use qtrm_residual_logits even when donor logits are absent.
2. Depth supervision now trains full answer-token spans, not only the first answer token.
3. Training schedule now cycles each row through every requested depth.
4. Raw KISS config disables donor-logit fusion, answer bottleneck, residual clamp,
   residual governor, and identity core-output blend for pure recursive tests.
5. Added core_only trainable policy to stop prelude/coda/lm paths from memorizing
   when testing whether the recursive core itself can learn.
6. Added mandatory `core_loop_readout`: final answer logits can now be produced
   directly from the recursive loop state, and `disable_core=true` zeroes that
   path when `core_loop_readout_requires_core=true`.
7. Added `core_and_loop_readout` trainable policy so the next gate can train
   only the recursive core and its direct loop readout while freezing donor,
   prelude, workspace, coda, and the LM embedding table.
```

## Key Results

```text
fullseq + residual path overfit8 s160:
  heldout: donor 3/8, core_off 3/8, core_steps_1/2/4/8 all 3/8
  diversity: 0/8

KISS no-blend staged overfit8 s320:
  heldout: donor 3/8, core_off 6/8, core8 2/8
  diversity: 2/8 changed
  train8: donor 5/8, core_off 6/8, core1 2/8, core8 4/8

KISS no-blend final-target overfit8 s320:
  heldout: donor 3/8, core_off 6/8, core8 2/8
  diversity: 1/8 changed
  train8: donor 5/8, core_off 6/8, core1 3/8, core8 4/8

KISS plain-coda final-target overfit8 s320:
  heldout: donor 3/8, core_off 2/8, core8 2/8
  diversity: 0/8
  train8: donor 5/8, core_off 7/8, core8 7/8

core_only plain-coda final-target overfit8 s320:
  heldout: donor 3/8, core_off 3/8, core8 3/8
  diversity: 0/8
  train8: donor 5/8, core_off 2/8, core8 3/8
```

## Interpretation

The corrected KISS/plain-coda path can overfit 7/8 train examples, but
`core_off` also reaches 7/8. That means the gain is not a recursive-core gain;
it is mostly a non-core answer path memorizing the tiny training set.

The stricter `core_only` run prevents that leak. It slightly improves train8
from `core_off=2/8` to `core8=3/8`, but it does not beat donor, does not
generalize, and does not show depth diversity. This is the cleanest current
diagnosis: the present recursive core is connected, but it is not yet a strong
raw reasoning engine.

## Formal CoT/Latent Separation Update

`A Formal Comparison Between Chain of Thought and Latent Thought`
(`arXiv:2509.25239`) is now part of the recurrent-depth reference set. Its
impact on this ledger is important:

- Failure on small serial arithmetic/list gates does not by itself disprove
  latent recurrence, because some tasks are CoT-favorable or
  stochastic/sequential.
- It does disprove any broad claim that the current QTRM loop is already a
  general raw-reasoning engine.
- The next depth gate must separate latent-favorable parallel tasks from
  CoT-favorable/stochastic tasks.

Required next data split:

```text
parallelizable/DAG-like:
  graph reachability
  circuit propagation
  local constraint propagation

sequential/stochastic:
  serial arithmetic traces
  approximate counting
  sampling-style decisions
```

The architecture should remain mandatory-core, but evaluation should no longer
expect one all-latent path to win uniformly across these families.

## Root Architecture Hypothesis

The current z_L/z_H core can perturb answer logits, but it does not learn a
reliable algorithmic state transition from these small supervised targets. The
next architecture must make the recurrent state itself the main computation
surface instead of letting coda/readout paths solve the task around it.

## Implemented Next Candidate

```text
Candidate: mandatory recursive loop readout
Raw-intelligence axis: recursive reasoning
SSOT source: prompt token stream only
Smallest path: prompt -> frozen encoder/prelude/workspace -> trainable
  recurrent z_L/z_H transition -> direct loop-state readout -> answer logits
Needed because: current core-only cannot overfit 8 reasoning cases with
  depth scaling
Duplicated logic avoided: no answer bottleneck, no donor fusion, no MemoryOS,
  no coda/prelude training in the first gate
Canonical gate: train8 overfit must reach core8 > core_off and heldout depth
  sweep must have changed outputs before scaling up
Kill criterion: core_only still cannot overfit 8 cases after readout is made
  direct and coda/lm leakage is frozen
```

Implementation:

```text
model fields:
  core_loop_readout_enabled
  core_loop_readout_requires_core
  core_loop_readout_logits
  core_loop_readout_hidden

code:
  src/wgram_lm/wgram_model.py
  src/wgram_lm/config.py
  src/wgram_lm/training/train.py

canonical config:
  configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml
```

Smoke verification:

```text
run: pure_recursive_loop_readout_smoke_s001
checkpoint: runs/qwen35_2b_4090_pure_recursive_loop_readout_smoke_s001/last.pt
report: docs/wiki/decisions/pure-recursive-loop-readout-smoke-s001-depth-gate-1.md
status: rejected, as expected for a 1-step smoke
purpose: verify the new checkpoint keys, donor path, trainable policy, and
  raw-depth runner execute end to end
```

S160 loop-readout results:

```text
final-target S160, tie-aware scorer:
  heldout8: rejected
  donor 2/8, core_off 0/8, core8 4/8
  failed: no depth scaling, depth outputs identical across steps
  train8: rejected
  donor 2/8, core_off 0/8, core8 3/8

staged-target S160, tie-aware scorer:
  heldout8: accepted
  donor 2/8, core_off 0/8, core8 3/8
  depth outputs changed on 2/8 cases
  heldout16: rejected
  donor 5/16, core_off 0/16, core8 4/16
  depth outputs changed on 3/16 cases
```

Interpretation:

```text
The mandatory loop readout fixed the old core-off leakage. Tie-aware scoring
shows core_off is now 0 on these gates because disabling the core zeroes the
answer path. Staged depth targets produced the first accepted 8-case
raw-recursive gate, but the signal is still weak and does not scale to 16
cases. The next checkpoint should not be promoted beyond "first positive
falsification signal"; it needs longer/cleaner staged training or a stronger
explicit recurrent state transition.
```

Next experiment:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
CONFIG=configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml \
INIT_CHECKPOINT=runs/qwen35_2b_4090_pure_recursive_reasoning_core_s160/last.pt \
OUT_DIR=runs/qwen35_2b_4090_pure_recursive_loop_readout_s160 \
RUN_NAME=pure_recursive_loop_readout_s160 \
STEPS=160 TARGET_MODE=final DEPTH_STEPS=1,2,4,8 \
bash scripts/197_run_pure_recursive_depth_supervised_train.sh
```

## Explicit Answer-State Loop Candidate

The next architecture candidate is closer to TRM than the previous
`core_loop_readout` path:

```text
Candidate: explicit recurrent answer-state loop
Raw-intelligence axis: recursive reasoning
SSOT source: prompt token stream only
Smallest path:
  prompt -> frozen encoder/prelude/workspace
  -> recurrent z_L/z_H transition
  -> y_0 text answer state
  -> for each depth t: y_t = update(y_{t-1}, z_H_t)
  -> LM head over y_T
Needed now because:
  core_loop_readout reads only the final loop state; it does not force the
  answer state itself to be iteratively revised.
Duplicated logic avoided:
  no coda, no donor fusion, no MemoryOS, no answer bottleneck, no verifier.
Canonical gate:
  train8/heldout depth sweep must show core8 > core_off and changed outputs.
Kill criterion:
  if answer-state loop cannot overfit tiny train8 or still has no depth
  diversity, reject this TRM adaptation and move to persistent y/z carry.
```

Implementation:

```text
model fields:
  answer_state_loop_enabled
  answer_state_loop_requires_core
  answer_state_loop_gate_init_bias
  answer_state_loop_gate_min
  answer_state_loop_logits
  answer_state_loop_hidden
  answer_state_loop_depth_hidden

trainable policy:
  core_and_answer_state_loop

canonical config:
  configs/qwen35_2b_4090_pure_recursive_answer_state_loop_s160.yaml
```

Smoke verification:

```text
run: pure_recursive_answer_state_loop_smoke_s001
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_smoke_s001/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-smoke-s001-depth-gate-1.md
status: rejected, as expected for a 1-step smoke
purpose: verify checkpoint initialization, new answer-state modules, and
  strict core-off behavior
```

S160 answer-state loop results:

```text
staged-target S160, tie-aware forced-choice scorer:
  heldout16: rejected
  donor 5/16, core_off 0/16, core8 5/16
  depth ladder: core1 3/16, core2 4/16, core4 4/16, core8 5/16
  depth outputs changed on 6/16 cases
  passed:
    deep_core_beats_core_off
    depth_scaling_gain_present
    depth_outputs_not_all_identical
    no_retrieval_or_memoryos_shortcut
  failed:
    deep_core_does_not_beat_donor
```

Interpretation:

```text
This is a partial architectural win but not a promoted checkpoint. The explicit
answer-state loop fixes the old non-causal issue: with core disabled, the final
answer path is zeroed and the gate falls to 0/16. The loop also changes outputs
with depth, and deeper steps help arithmetic examples such as arith-chain-000
and arith-chain-002.

The failure is still decisive: core8 only ties donor-only at 5/16. List
transform tasks collapse to EMPTY, and some boolean cases are correct at shallow
depth but become wrong at deeper depth. The next fix should not tune answer
formatting. It should train the recurrent state transition on hard failure
families and add a stability objective so deeper steps preserve a correct
answer instead of overwriting it.
```

Next experiment:

```text
Candidate: answer-state loop with depth-consistency and hard-family curriculum
Keep:
  prompt-only SSOT
  mandatory core path
  no MemoryOS, no retrieval, no donor fusion
Add:
  hard-family oversampling for list_transform and late-flip boolean cases
  depth-consistency preference: if a shallow depth is correct, deeper depths
    must not lower the correct choice margin
  staged arithmetic/list intermediate targets with enough examples to avoid
    EMPTY collapse
Kill criterion:
  core8 still does not beat donor-only on heldout16 after hard-family curriculum
  and depth-consistency training
```

Hard curriculum result:

```text
run: pure_recursive_answer_state_loop_hard_s320
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_hard_s320/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-hard-s320-causal-depth-gate-16.md
status: rejected
donor 5/16, core_off 0/16, core8 4/16
failed:
  deep_core_does_not_beat_donor
  no_depth_scaling_gain
```

Interpretation:

```text
Hard-family oversampling and first-token chosen/rejected margin did not solve
the root issue while training still used full prompt+answer inputs. It moved
some boolean cases from always-FALSE to always-TRUE and reduced arithmetic
accuracy. This is a local-loss failure, not a promoted architecture result.
```

## Causal Prefix Gate Fix

Failure:
the raw gate and depth-supervised training were not strict enough for QTRM.
They evaluated or trained on `prompt + full answer` in one forward pass.

Evidence:
QTRM workspace/core can attend across the full provided sequence, unlike the
donor's causal token prediction path. Candidate answer tokens are therefore on
the QTRM information path during forced-choice scoring.

Known limitation class:
future-answer leakage through non-causal latent workspace scoring.

Root architecture hypothesis:
the answer-state loop is meaningful only if both training and evaluation use
causal prefix inputs.

Information path needed:

```text
prompt tokens only
-> donor causal hidden states for current prefix
-> QTRM workspace/core
-> answer-state loop
-> next answer-token logits
```

Implemented:

```text
eval:
  scoring=causal_forced_choice
  candidate token i is scored from prompt + candidate tokens < i only

training:
  --causal-prefix-supervision
  prompt-only input predicts the first target answer token
```

Accepted result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_s160
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-s160-depth-gate-16.md
status: accepted
donor 5/16, core_off 0/16, core8 6/16
depth ladder: core1 3/16, core2 4/16, core4 6/16, core8 6/16
passed:
  deep_core_beats_core_off
  deep_core_beats_donor
  depth_scaling_gain_present
  depth_outputs_not_all_identical
  no_retrieval_or_memoryos_shortcut
```

Interpretation:

```text
This is the first strict causal-prefix raw-intelligence acceptance. It is
evidence that the mandatory recursive answer-state path can improve a
held-out prompt-only reasoning gate when future answer tokens are removed from
both training and evaluation.

The result remains narrow. Arithmetic improves strongly at deeper depth
(3/4 at core8 versus donor 2/4), but list_transform remains 0/4 and symbolic
binding remains 1/4. Do not claim robust general reasoning yet.
```

Next experiment:

```text
Candidate: causal-prefix multi-token depth training
Needed because:
  causal-prefix S160 trains only the first answer token, so list transforms and
  comma-separated multi-token answers remain unsolved.
Smallest path:
  autoregressive prefix loop over answer tokens
  each token forward sees prompt + previous answer tokens only
  same answer-state loop/core path
Acceptance gate:
  heldout16 still accepted and list_transform improves above 0/4
Kill criterion:
  list_transform remains 0/4 or core8 no longer beats donor
```

Result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_multitoken_s080
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-multitoken-s080-depth-gate-16.md
status: rejected
donor 5/16, core_off 0/16, core8 4/16
depth ladder: core1 3/16, core2 4/16, core4 4/16, core8 4/16
failed:
  deep_core_does_not_beat_donor
```

Interpretation:

```text
The implementation is causally cleaner than full-answer training because each
token sees only prompt + previous answer tokens. However, the naive
multi-token objective diluted the accepted first-token depth signal:
canonical causal_prefix_s160 core8 was 6/16, while multitoken_s080 core8 fell
to 4/16.

Do not scale this experiment by simply increasing steps. It did not improve
list_transform above 0/4 and it broke the donor-beating acceptance condition.
```

Big-structure doubt gate:

```text
Root architecture claim:
  mandatory recursive answer-state core can improve prompt-only raw reasoning.
Falsifying observation:
  deeper core no longer beats donor or later-token training collapses the
  depth gain.
Observed:
  later-token loss collapsed the donor-beating gate.
Likely issue:
  a single next-token CE objective treats sequence continuation and latent
  reasoning as the same pressure. For these tiny probes, later-token losses
  dominate formatting/continuation instead of strengthening the recursive
  decision state.
Replacement candidate:
  keep first-token causal-prefix as the canonical raw-reasoning gate, then add
  a separate sequence-readout or teacher-latent objective that is evaluated by
  its own ablation before promotion.
```

Next candidate:

```text
Candidate: split objective for recursive decision state vs sequence readout
Raw-intelligence axis:
  recursive reasoning first, sequence continuation second
SSOT source:
  one canonical prompt token stream; no hidden evidence or MemoryOS
Smallest path:
  preserve first-token depth CE/progress loss
  add optional later-token CE with a small weight or delayed schedule
  require core8 >= canonical baseline and list_transform > 0/4
Needed now because:
  naive multi-token causal-prefix failed the donor-beating gate
Duplicated logic removed or avoided:
  no separate answer channel, no retrieval path, no post-hoc formatting gate
Canonical gate:
  heldout16 causal_forced_choice plus family-level list_transform check
```

Result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_split_s080
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_split_s080.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_split_s080/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-split-s080-depth-gate-16.md
status: rejected
donor 5/16, core_off 0/16, core8 5/16
depth ladder: core1 3/16, core2 5/16, core4 5/16, core8 5/16
failed:
  deep_core_does_not_beat_donor
```

Interpretation:

```text
Lowering later-token loss from equal weight to 0.1 reduced damage but did not
preserve the accepted baseline. The split run ties donor instead of beating it
and still leaves list_transform at 0/4.

This is the second failed local fix for later-token/list behavior. Escalate
from CE-weight tuning to root architecture review.
```

Escalated architecture candidates:

```text
Candidate A: state-transition distillation
  Train recursive z-state transitions from a stronger teacher trace or
  solved latent target, then use answer CE only as readout validation.
  Raw-intelligence axis: recursive reasoning
  SSOT source: prompt-only token stream
  Smallest path: z_t -> z_{t+1} target plus first-token gate
  Kill criterion: no core-depth gain over causal_prefix_s160

Candidate B: explicit recurrent scratchpad state without visible CoT
  Add a small hidden state tuple that must update each recursive step and is
  decoded only at the end.
  Raw-intelligence axis: recursive reasoning
  SSOT source: prompt-only token stream
  Smallest path: recurrent state read/write ablation
  Kill criterion: core_off or state_stop matches full model

Candidate C: trainable memory composition gate
  Defer sequence continuation and test MSA/LM2-style memory separately on
  distractor recall and composition tasks.
  Raw-intelligence axis: trainable memory, then reasoning+memory composition
  SSOT source: token stream plus trainable memory state
  Smallest path: memory_on/off length sweep
  Kill criterion: memory_on does not beat memory_off
```

Recommended next:

```text
Candidate A. It attacks the current failure directly: the core should learn
better latent state transitions, not merely later answer-token continuation.
```
