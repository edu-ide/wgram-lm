# Pure Recursive Depth-Supervised S080 Failure Ledger

Status: rejected, 2026-05-02.

## Failure

Prompt-only depth supervision did not create held-out recursive-depth gains.

Two variants were tested:

```text
internal depth probe only:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_s080/last.pt
  eval: runs/eval/pure_recursive_depth_supervised_s080_depth_gate_8.jsonl
  report: docs/wiki/decisions/pure-recursive-depth-supervised-s080-depth-gate-8.md

internal depth probe + final prompt-only answer CE:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_finalpath_s080/last.pt
  eval: runs/eval/pure_recursive_depth_supervised_finalpath_s080_depth_gate_8.jsonl
  report: docs/wiki/decisions/pure-recursive-depth-supervised-finalpath-s080-depth-gate-8.md
```

Both variants produced the same held-out 8-case gate:

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

## What This Means

The new training signal is valid as a diagnostic, but it is not enough as an
architecture fix.

During training, the prompt-only losses decreased:

```text
depth_final_ce: about 11.23 -> 5.25
final_path_ce: about 11.07 -> 5.06
```

However, the held-out answer choices were unchanged from S160. The recursive
core is still behaving like a quickly settled transform: once the core is on,
steps 1, 2, 4, and 8 select the same answer.

## Root Architecture Hypothesis

The current recurrent core has no strong reason to make each outer step a
different state transition. It can learn a one-step correction and then reach a
fixed point. Increasing `outer_steps` therefore repeats the same computation
instead of producing additional reasoning.

This falsifies the local hypothesis:

```text
Prompt-only depth answer CE plus a progress margin is enough to make deeper
latent recursion improve held-out no-evidence reasoning.
```

## Replacement Candidates

1. Depth-conditioned transition core

```text
Inject explicit step/depth state into the answer-relevant transition, not only
as a weak stability embedding. Each depth must learn a different transition
role: observe -> transform -> verify -> answer.
```

2. State-transition target dataset

```text
Use synthetic cases with exact intermediate states:
  arithmetic: subexpression after each operation
  symbolic binding: first mapping, second mapping
  boolean: NOT/AND/OR stages
  list transform: filter stage, transform stage
Train depth k to predict stage k, not only the final answer.
```

3. Depth-readout answer path

```text
Make the final answer path consume the selected depth state/readout directly.
Current final path can ignore depth-specific readouts; this must become causal
or the gate remains insensitive to recursive computation.
```

## Next Gate

Do not promote another run unless it satisfies all of:

```text
depth 2 output differs from depth 1 on at least one held-out case
depth 4 or 8 beats donor and core_off in hit count
depth output diversity changed_case_count > 0
no MemoryOS/retrieval/workspace-evidence shortcut
```

If another local loss fails, stop SFT-style tuning and implement a
depth-conditioned state-transition core.
