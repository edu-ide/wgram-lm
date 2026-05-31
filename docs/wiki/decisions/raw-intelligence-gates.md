# Raw Intelligence Gates (Historical — Pre-2026-06)

> **Superseded**: See the canonical 2026-06 SSOT: [Raw Intelligence / Actual Reasoning Necessary Conditions](raw-intelligence-necessary-conditions-2026-06.md).
> This document is preserved as historical context (defined the original pure recursive depth + memory on/off gates on the old QTRMRecursiveCore substrate, before OneBodyParallelHybridBlock, before MSA elevation as first-class memory, and before deep 5.56 rehearsal integration on the hybrid).

**Status (original)**: implemented gate scaffold, 2026-05-02.

## Decision

QTRM ASI progress is now judged first by raw intelligence, not by answer
cleanliness, RAG correctness, MemoryOS retrieval, SSOT hygiene, or verifier
calibration.

Primary gates:

```text
Pure Recursive Reasoning:
  no MemoryOS, no retrieval, no hidden evidence
  donor_only_no_evidence vs qtrm_core_off_no_evidence
  vs qtrm_core_steps_1/2/4/8_no_evidence
  pass only if deeper core depth beats donor and core_off
  reject if depth 1/2/4/8 produce identical outputs on all comparable cases

Trainable Memory Intelligence:
  no MemoryOS retrieval shortcut
  qtrm_memory_on_no_evidence vs qtrm_memory_off_no_evidence
  pass only if memory-on improves recall/use over memory-off

Reasoning + Memory Composition:
  qtrm_core_memory_on_no_evidence
  vs qtrm_core_off_memory_on_no_evidence
  vs qtrm_core_on_memory_off_no_evidence
  pass only if both core-off and memory-off lose
```

SSOT, KISS, DRY, and YAGNI remain required engineering hygiene, but they are
subordinate. A change that only improves formatting, grounding, thresholds, or
retrieval is not ASI architecture progress unless it causes a measured
recursive-depth gain or memory-on/off gain.

## Task-Family Split

The 2026 formal CoT/latent-thought comparison changes how raw-intelligence
gates should be interpreted:

```text
source: docs/wiki/sources/formal-cot-vs-latent-thought.md
paper: https://arxiv.org/abs/2509.25239
```

Do not treat "latent loop beats CoT on every task" as the target. The target is
more precise:

```text
parallelizable tasks:
  latent/recurrent core should improve with depth

sequential or stochastic counting/sampling tasks:
  hybrid trace/latent mode may be required
  pure latent mode is allowed to lose if the task is formally CoT-favorable
```

Future raw gates should label each case with:

```text
reasoning_family
expected_paradigm
requires_stochasticity
parallel_depth_estimate
serial_trace_length_estimate
```

Promotion still requires measured gain, but rejection must identify whether the
failure is a QTRM architecture failure or a task-family mismatch.

Implemented fields:

```text
scripts/190_build_pure_recursive_reasoning_cases.py
scripts/192_eval_raw_intelligence.py
scripts/194_build_pure_recursive_reasoning_preferences.py
src/wgram_lm/eval/raw_intelligence_gate.py
data/eval/pure_recursive_reasoning_heldout_72.jsonl
data/filtered/pure_recursive_reasoning_train256_cases.jsonl
data/filtered/pure_recursive_reasoning_preferences_train.jsonl
```

The raw gate JSON now includes:

```text
by_task_family
by_reasoning_family
by_expected_paradigm
```

## Why This Exists

The previous 72-case strict mandatory-core result is useful but not enough:

```text
full mandatory core: 50/72
donor/core_off/workspace_off: 39/72
```

That proves the accepted checkpoint has a causal answer path on the current
MemoryOS-style heldout gate. It does not prove TRM-like pure reasoning. It also
does not prove MSA/LM2-style trainable memory intelligence. Those claims need
no-retrieval depth and memory ablations.

## Implemented Artifacts

```text
src/wgram_lm/eval/raw_intelligence_gate.py
scripts/190_build_pure_recursive_reasoning_cases.py
scripts/191_build_raw_intelligence_gate.py
scripts/192_eval_raw_intelligence.py
scripts/193_run_pure_recursive_reasoning_depth_gate.sh
scripts/194_build_pure_recursive_reasoning_preferences.py
scripts/195_run_pure_recursive_reasoning_core_train.sh
scripts/196_train_pure_recursive_depth_supervised.py
scripts/197_run_pure_recursive_depth_supervised_train.sh
configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml
configs/qwen35_2b_4090_pure_recursive_depth_supervised_s080.yaml
data/eval/pure_recursive_reasoning_heldout_72.jsonl
data/filtered/pure_recursive_reasoning_train256_cases.jsonl
data/filtered/pure_recursive_reasoning_preferences_train.jsonl
```

## First Smoke Result

The first 4-case interleaved smoke used forced-choice logprob scoring, not
greedy generation, because the base donor frequently emitted EOS immediately
after `Answer:`.

```text
report: docs/wiki/decisions/pure-recursive-reasoning-depth-gate-smoke4.md
summary: docs/wiki/decisions/pure-recursive-reasoning-depth-gate-smoke4-summary.json
eval: runs/eval/pure_recursive_reasoning_depth_sweep_smoke4.jsonl
status: rejected
donor_only_no_evidence: 1/4
qtrm_core_off_no_evidence: 1/4
qtrm_core_steps_1_no_evidence: 1/4
qtrm_core_steps_2_no_evidence: 1/4
qtrm_core_steps_4_no_evidence: 1/4
qtrm_core_steps_8_no_evidence: 1/4
shortcut records: 0
```

Interpretation:
the current accepted mandatory-core checkpoint has a causal answer path on the
MemoryOS-style gate, but it does not yet show raw recursive reasoning gain. The
core-depth outputs tie donor and core-off on this smoke, so the next fix must
train or redesign the recursive core itself.

## Raw-Core Training Path

The first corrective training path uses no MemoryOS evidence. It trains from
pure reasoning prompt/chosen/rejected rows and adds a `core_off` canonical
causal loss so the full path must prefer the chosen answer more than the
core-off ablation.

```text
train cases: data/filtered/pure_recursive_reasoning_train256_cases.jsonl
preference rows: data/filtered/pure_recursive_reasoning_preferences_train.jsonl
rows: 640
config: configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml
runner: scripts/195_run_pure_recursive_reasoning_core_train.sh
init: runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt
```

Canonical command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  bash scripts/195_run_pure_recursive_reasoning_core_train.sh
```

Promotion rule after training:

```text
Do not promote if qtrm_core_steps_8_no_evidence ties donor/core_off.
Do not promote if depth 1/2/4/8 has no positive scaling gain.
Do not promote if a MemoryOS/retrieval/workspace-evidence shortcut appears.
```

S160 result:

```text
report: docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8.md
failure ledger: docs/wiki/decisions/pure-recursive-reasoning-core-s160-failure-ledger.md
status: rejected
donor_only_no_evidence: 3/8
qtrm_core_off_no_evidence: 3/8
qtrm_core_steps_1/2/4/8_no_evidence: all 3/8
depth output diversity: 0/8 cases changed by depth
failed check: depth_outputs_identical_across_steps
```

Interpretation:
the core path is now active enough to change answers, but it does not improve
net score and does not scale with depth. The stricter gate shows that every
held-out case produced the same selected answer for depth 1, 2, 4, and 8. This
points to a root architecture issue: the current recursive core is not being
trained as a progressive state-update reasoning loop.

## Depth-Supervised Follow-Up

The next corrective path trained on prompt-only inputs so the answer token
could not leak through workspace construction. It first supervised recursive
core depth logits, then added final prompt-only answer CE so the answer path
also received direct pressure.

```text
runner: scripts/197_run_pure_recursive_depth_supervised_train.sh
internal-probe checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_s080/last.pt
final-path checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_finalpath_s080/last.pt
failure ledger: docs/wiki/decisions/pure-recursive-depth-supervised-s080-failure-ledger.md
```

Both S080 variants were rejected on the same 8-case raw-depth gate:

```text
donor_only_no_evidence: 3/8
qtrm_core_off_no_evidence: 3/8
qtrm_core_steps_1/2/4/8_no_evidence: all 3/8
depth output diversity: 0/8 cases changed by depth
failed check: depth_outputs_identical_across_steps
```

Interpretation:
the problem is not just missing loss on the final answer token. The recurrent
core reaches a fixed-point-like answer state after one step. The next
architecture candidate must make depth itself causal, for example through
depth-conditioned state transitions or explicit intermediate-state targets.

## Full-Sequence KISS Follow-Up

The next pass removed several training and routing defects:

```text
final logits now use qtrm_residual_logits even without donor logits
depth supervision now uses full answer-token spans
row/depth scheduling now cycles every row through every requested depth
pure recursive KISS config removes donor fusion, clamp, answer governor,
identity core-output blend, and answer bottleneck
core_only trainable policy was added for stricter causal-core probes
failure ledger: docs/wiki/decisions/pure-recursive-depth-fullseq-kiss-failure-ledger.md
```

Best diagnostic results:

```text
KISS no-blend staged overfit8 s320 train8:
  donor 5/8, core_off 6/8, core1 2/8, core8 4/8

KISS plain-coda final-target overfit8 s320 train8:
  donor 5/8, core_off 7/8, core8 7/8

core_only plain-coda final-target overfit8 s320 train8:
  donor 5/8, core_off 2/8, core8 3/8
```

Interpretation:
the model can overfit tiny train examples when non-core answer paths are
trainable, but the improvement is not causal to the recursive core because
`core_off` also improves. When training is restricted to `core_only`, the core
barely improves over `core_off` and still does not beat donor or show depth
diversity. The current raw-core design remains rejected.

## Mandatory Loop-Readout Follow-Up

The next architecture change makes the loop causal by construction rather than
only present as a latent prefix. The final QTRM residual answer path can now be:

```text
prompt tokens
-> frozen QTRM token embedding / prelude / workspace setup
-> recursive z_L/z_H core loop
-> core_loop_readout_cross(text queries, final loop state)
-> LM head
-> final text logits
```

`core_loop_readout_requires_core=true` means `disable_core=true` zeroes this
answer path. This is not an ASI proof. It is the next falsification probe:
if loop readout cannot overfit the tiny train8 gate and show `core8 > core_off`,
we reject this version and move to an even more explicit recurrent
state-machine core.

Artifacts:

```text
src/wgram_lm/wgram_model.py
src/wgram_lm/config.py
src/wgram_lm/training/train.py
configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml
tests/test_core_halting.py
tests/test_training_checkpoint_init.py
```

## Forced-Choice Tie Fix

Raw forced-choice scoring now treats top-choice logprob ties as non-answers:

```text
completion: __FORCED_CHOICE_TIE__
hit: false
record field: choice_tied
```

This matters because strict mandatory-loop configs intentionally zero logits
when `disable_core=true`. Before the fix, all-zero logits could select the
first candidate and falsely inflate `core_off`. After the fix, `core_off`
correctly scores 0 on the loop-readout gates when it has no answer path.

## Forced-Choice Length Normalization

Raw forced-choice scoring now records and, by default, ranks choices by
mean answer-token logprob:

```text
--choice-score-normalization mean
record fields:
  choice_score_normalization
  choice_scores[].logprob       # selected score used for ranking
  choice_scores[].logprob_sum   # raw summed answer-token logprob
  choice_scores[].token_count
```

Reason:
multi-token answers such as `208,204` were structurally disadvantaged against
short distractors like `EMPTY` when the gate ranked candidates by summed
logprob. For raw-intelligence gates with variable-length choices, `mean`
normalization is now the canonical eval contract. Legacy reports without this
field must not be compared directly to mean-normalized reports.

## Semantic Transition-State Contract

The code-only recursive state path must not use tokenization artifacts as the
main state target. The canonical hard-family generator now emits
`transition_state_codes`:

```text
arithmetic_chain:
  sum -> product -> final
list_transform:
  filtered -> doubled/final
symbolic_binding:
  one-hop -> two-hop/final
boolean_logic:
  not_q -> and -> final
```

Training uses these explicit semantic codes before falling back to the legacy
first-token-id hash. This keeps the causal state path small, but makes train
and heldout cases share the same operation-stage code instead of unrelated
token ids.

The preference margin also has a sequence mode:

```text
--choice-margin-mode sequence
```

Use sequence mode when the eval contract is mean forced-choice scoring,
especially for multi-token answers such as comma-separated lists. First-token
margin remains a legacy diagnostic, not the preferred hard-family training
contract.

## Full-State Depth Readout Contract

After semantic codes and continuous first-token transition state failed the
heldout200 hard-family gate, the next canonical pure-recursive training
contract is full-state depth readout:

```text
--staged-internal-sequence-ce-weight
--staged-internal-sequence-max-target-tokens
```

This trains each labelled depth readout against the full token sequence from
`depth_targets[depth]`, not only the first token. It is still prompt-only:
there is no retrieval, MemoryOS shortcut, or hidden evidence path. The claim is
accepted only if deeper recursive core outputs beat donor-only and the
transition-state-off ablation on heldout cases.

The first S240 full-state run failed because core8 tied both donor-only and
transition_state_off. Therefore full-state readout is now diagnostic, not a
canonical solution. Further work should move the primary objective to an
explicit recurrent state-machine core trained on solver traces, then evaluate
state prediction and final answer separately.

## First Positive Loop-Readout Signal

The first staged-target mandatory-loop run produced a small accepted 8-case
raw-recursive gate:

```text
checkpoint: runs/qwen35_2b_4090_pure_recursive_loop_readout_staged_s160/last.pt
report: docs/wiki/decisions/pure-recursive-loop-readout-staged-s160-depth-gate-8.md
status: accepted
donor: 2/8
core_off: 0/8
core8: 3/8
depth changed: 2/8 cases
```

The same checkpoint did not scale to 16 held-out cases:

```text
report: docs/wiki/decisions/pure-recursive-loop-readout-staged-s160-depth-gate-16.md
status: rejected
donor: 5/16
core_off: 0/16
core8: 4/16
depth changed: 3/16 cases
failed: deep core did not beat donor; no depth scaling gain
```

Decision:
this is the first real raw-recursive foothold, not a solved architecture.
Promote it only as evidence that mandatory loop readout plus staged targets can
make the recursive path causal on a small gate.

Tests:

```text
tests/test_raw_intelligence_gate.py
tests/test_pure_recursive_reasoning_cases.py
tests/test_raw_intelligence_gate_script.py
tests/test_raw_intelligence_eval_script.py
tests/test_pure_recursive_reasoning_gate_runner.py
tests/test_pure_recursive_reasoning_preferences.py
tests/test_pure_recursive_reasoning_core_train_script.py
```

## Canonical Command

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  bash scripts/193_run_pure_recursive_reasoning_depth_gate.sh
```

Useful overrides:

```bash
CONFIG=configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml
CHECKPOINT=runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt
MAX_CASES=16
MAX_NEW_TOKENS=12
QTRM_LOGITS_SCALE=0.75
DONOR_LOGITS_SCALE=1.0
```

Outputs:

```text
runs/eval/pure_recursive_reasoning_depth_sweep.jsonl
docs/wiki/decisions/pure-recursive-reasoning-depth-gate.md
docs/wiki/decisions/pure-recursive-reasoning-depth-gate-summary.json
```

## Acceptance Rule

Accepted pure recursive reasoning requires all of:

```text
deep_core_beats_core_off
deep_core_beats_donor
depth_scaling_gain_present
no_retrieval_or_memoryos_shortcut
```

Reject immediately if:

```text
core_off ties or beats deep core
donor-only ties or beats deep core
depth 1/2/4/8 does not show a positive scaling gain
depth 1/2/4/8 produce identical outputs on all comparable cases
any record reports MemoryOS, retrieval, evidence tokens, or workspace-memory evidence
```

## Next Required Work

1. Implement a depth-conditioned state-transition core with explicit
   intermediate targets.
2. Add trainable-memory cases for MSA/LM2 memory-on/off length and distractor
   sweeps.
3. Only after both gates pass, run the composition gate where memory retains
   facts and the recursive core composes them.
