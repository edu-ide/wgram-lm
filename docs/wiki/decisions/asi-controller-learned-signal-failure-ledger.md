# ASI Controller Learned-Signal Failure Ledger

Status: rejected experiment, 2026-05-01.

## What Was Tested

The Stage-1.5 controller-signal scaffold worked only because the trace dataset
provided an oracle two-bit signal:

```text
[0, 0] -> RETRIEVE_MEMORY
[1, 0] -> VERIFY_EVIDENCE
[1, 1] -> ANSWER
```

This experiment removed that oracle input at inference time and asked QTRM to
predict the signal from its own latent core.

Implemented knobs:

- `controller_signal_source: learned_core`
- `controller_signal_base_scale: 0.0`
- `loss_controller_signal_weight`
- `trainable_param_policy: controller_only`
- `trainable_param_policy: controller_signal_head_only`

## Results

| Variant | Checkpoint | Held-Out Accuracy | Collapse |
| --- | --- | ---: | --- |
| Full learned-core signal | `runs/qwen35_2b_4090_controller_learned_signal_s300/last.pt` | 0.3333 | predicts `VERIFY_EVIDENCE` for every row |
| Head-only signal | `runs/qwen35_2b_4090_controller_learned_signal_head_s300/last.pt` | 0.3333 | predicts `ANSWER` for every row |
| Learned-readout diagnostic | `runs/qwen35_2b_4090_controller_learned_signal_readout_s300/last.pt` | 0.3704 | mostly predicts `ANSWER`; only 8/72 `VERIFY_EVIDENCE` rows correct |

Both variants fail the ASI causal-loop gate:

```text
qtrm_does_not_beat_donor_harness
qtrm_does_not_beat_scripted_harness
latent_core_not_causal
world_model_not_causal
verifier_not_causal
```

Reports:

- `docs/wiki/decisions/controller-learned-signal-s300-heldout-eval-summary.json`
- `docs/wiki/decisions/asi-controller-learned-signal-causal-loop-s300-summary.json`
- `docs/wiki/decisions/controller-learned-signal-head-s300-heldout-eval-summary.json`
- `docs/wiki/decisions/asi-controller-learned-signal-head-causal-loop-s300-summary.json`
- `docs/wiki/decisions/asi-controller-learned-signal-head-causal-loop-s300.md`
- `docs/wiki/decisions/controller-learned-signal-readout-s300-heldout-eval-summary.json`
- `docs/wiki/decisions/asi-controller-learned-signal-readout-causal-loop-s300-summary.json`
- `docs/wiki/decisions/asi-controller-learned-signal-readout-causal-loop-s300.md`

## Structural Diagnosis

This failed for a reason that matters:

```text
oracle signal path causal != learned reasoning path causal
```

The explicit signal projection can drive the action controller. That only
proves the action head can consume a correct scaffold. It does not prove the
world model, verifier, or recurrent core can produce that scaffold.

The learned replacement was a stateless per-row classifier:

```text
current prompt/state row -> frozen latent core -> two-bit signal -> action
```

That is weaker than the intended cognitive loop:

```text
state_t -> action_t -> observation_t+1 -> verification_t+1
-> state_t+1 -> next action
```

A one-row latent/readout classifier can memorize or collapse to the most
convenient action surface without learning transition dynamics. It also does
not force the world model and verifier to be causal causes of later decisions.

The `learned_readout` diagnostic is important because it moves the signal source
from `z_h[:, -1, :]` to the final generation/coda readout. It still fails. This
means the failure is broader than a bad latent pooling choice. The current
two-bit bottleneck and frozen oracle action mapping are too brittle to serve as
the learned planning path.

One additional warning signal: in the readout diagnostic, `latent_core_off` and
`workspace_off` score higher than the full model. That means the component is
not just non-causal; in this setup it can interfere with the learned readout.

## Consequence

Do not treat the oracle `controller_signal` result as a learned world-model or
verifier result.

Do not keep adding scalar losses to the same per-row bottleneck and call it
planning. The next architecture has to make the trace transition itself the
training object.

Do not promote `learned_readout` as the final path. It is only a diagnostic that
shows prompt/coda readout can recover a small part of the oracle signal but does
not solve the planner.

## Next Architecture Constraint

The next controller experiment should train and evaluate a recurrent transition
policy:

```text
TypedContextTape state_t
+ previous action
+ observation
+ verifier result
-> recurrent controller state_t+1
-> next action
```

Acceptance criteria:

- held-out action accuracy beats scripted/donor baselines on nontrivial traces;
- disabling recurrent state or core depth reduces score;
- disabling world-model prediction reduces cases that require consequence
  prediction;
- disabling verifier state reduces cases that require support/refute decisions;
- generated action traces improve answer-level task score, not only action
  imitation.
