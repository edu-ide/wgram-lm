# Pure Recursive Reasoning Core S160 Failure Ledger

Status: rejected, 2026-05-02.

## Failure

Raw-core S160 training made the QTRM core path change some answers, but it did
not create a net raw-reasoning gain or depth-scaling behavior.

## Evidence

```text
config: configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml
runner: scripts/195_run_pure_recursive_reasoning_core_train.sh
checkpoint: runs/qwen35_2b_4090_pure_recursive_reasoning_core_s160/last.pt
eval: runs/eval/pure_recursive_reasoning_core_s160_depth_gate_8.jsonl
report: docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8.md
summary: docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8-summary.json
status: rejected
```

Gate result:

```text
donor_only_no_evidence: 3/8
qtrm_core_off_no_evidence: 3/8
qtrm_core_steps_1_no_evidence: 3/8
qtrm_core_steps_2_no_evidence: 3/8
qtrm_core_steps_4_no_evidence: 3/8
qtrm_core_steps_8_no_evidence: 3/8
depth output diversity: 0/8 cases changed by depth
failed check: depth_outputs_identical_across_steps
shortcut records: 0
```

Observed qualitative change:

```text
core path fixed symbolic-binding-000:
  core_off: green, expected violet
  core_on:  violet, expected violet

core path broke boolean-logic-001:
  core_off: FALSE, expected FALSE
  core_on:  TRUE, expected FALSE
```

So the core is not merely invisible anymore. But it is not better, and depth
1/2/4/8 is identical for every comparable held-out case.

## Root Architecture Hypothesis

The current recursive core is acting like a single static core-present
transform, not like a depth-progressive reasoning process.

The likely reason is that training optimizes the final answer path, while no
loss requires intermediate depths to become progressively better or different.
Changing `outer_steps` at eval time therefore repeats the same settled behavior
instead of creating additional computation.

## Falsified Local Hypothesis

```text
Preference SFT + core_off canonical causal loss is enough to make deeper
latent recursion improve no-evidence reasoning.
```

Rejected. It made the core path active but did not improve net score or produce
depth scaling.

## Replacement Candidates

1. Depth-supervised recursive core

```text
Train core_depth_last_logits at every depth.
Require depth 1 < depth 2 < depth 4 on hard cases.
Use teacher or exact symbolic labels to supervise per-depth answer logits.
```

2. Iterative state transition objective

```text
Each core step predicts a next latent state, error/correction token, or
candidate answer distribution.
Loss rewards monotonic improvement and penalizes unchanged depth states.
```

3. Actual loop controller

```text
Core emits answer proposal + error signal.
Next core step consumes the previous proposal/error and updates state.
This is closer to TRM/looped-LM behavior than a repeated block with only final
answer supervision.
```

## Next Gate

The next candidate must pass a stricter raw-depth gate:

```text
qtrm_core_steps_2 beats qtrm_core_steps_1 on at least one held-out case.
qtrm_core_steps_4 or 8 beats qtrm_core_off and donor-only in hit count.
Depth outputs must not be identical across 1/2/4/8.
No MemoryOS/retrieval/workspace-evidence shortcuts.
```

If this fails again, stop SFT-style local tuning and redesign the core as a
stateful loop with explicit per-depth supervision.
