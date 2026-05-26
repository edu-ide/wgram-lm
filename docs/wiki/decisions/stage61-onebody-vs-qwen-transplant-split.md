# Stage61 One-Body vs Qwen-Transplant Split

Date: 2026-05-23

## Decision

Split the work into two tracks.

```text
DGX:
  from-scratch one-body model
  reader -> recurrent thought -> LM-token speaker

Local:
  Qwen transplant diagnostic
  Qwen reader/speaker + added recurrent working-memory organ
```

This is not two competing random experiments. It separates two causes that were
previously mixed together:

```text
1. Can the architecture learn when it is born as one body?
2. If yes, can that same working-memory organ be surgically connected to Qwen's
   pretrained mouth?
```

## Plain-Language Model

The DGX track is like training a child from the beginning to read, think, and
speak with one nervous system.

The local Qwen track is like attaching a new working-memory organ to an adult
speaker. If it fails, that does not immediately prove the organ is wrong. It may
mean the organ never learned the speaker's language, or that the organ itself is
not yet writing the correct working note.

Therefore the local track must not be judged only by final answer accuracy. It
must answer:

```text
Does the ledger contain the right answer before Qwen speaks?
```

## Implemented Contracts

DGX one-body runner:

```text
scripts/launch_stage61a_dgx_onebody_fromscratch.sh
```

This wraps the existing donorless runner:

```text
scripts/411_dgx_trm_raw_scaleout_gate.sh
```

with the plain-language contract:

```text
tokens -> recurrent core -> LM logits
```

Local transplant diagnostic runner:

```text
scripts/launch_stage61b_local_qwen_transplant_diagnostic.sh
```

This keeps the Qwen LM-mouth path, but also evaluates a fixed direct ledger
renderer:

```text
typed digit ledger -> fixed numeric/list renderer
```

The direct renderer is diagnostic only. It is not a promoted answer path. It
tells us whether numeric/list failure comes from the working ledger or from
Qwen-mouth alignment.

## Code Changes

```text
scripts/530_train_final_typed_register_answerer.py
  Added qwen_lm_mouth_direct_ledger_renderer eval path.
  Added --eval-qwen-lm-mouth-direct-ledger-renderer.
  Ensured digit-transition logits are returned when this diagnostic is active.

tests/test_stage59_typed_value_trace_supervision.py
  Added a test proving the direct renderer can recover a numeric answer from
  digit_transition_executor logits before Qwen-mouth decoding.
```

## Local Stage61B Evidence

Run:

```text
/tmp/stage61b_local_qwen_transplant_directledger_e5
```

Best Qwen-mouth eval:

```text
6/32 = 0.1875 at epoch 2
```

Best family pattern:

```text
arithmetic_chain = 0/8
list_transform    = 0/8
boolean_logic     = 5/8 at best epoch
symbolic_binding  = 1/8 at best epoch
```

Direct ledger diagnostic:

```text
qwen_lm_mouth_direct_ledger_renderer_accuracy = 0.0
arithmetic_chain = 0/8
list_transform    = 0/8
```

Interpretation:

```text
This is not only a Qwen-mouth alignment failure.

The current numeric/list ledger is not yet writing a directly renderable answer.
Qwen-mouth can still learn some boolean/symbolic surface responses, but the
numeric working desk itself is wrong under the fixed ledger renderer.
```

## Gate

Promote local Qwen transplant only if both are true:

```text
direct ledger numeric/list > 0/16
Qwen-mouth full accuracy > direct-ledger-off / graft-off by a real margin
```

Reject more Qwen-mouth-only tuning if:

```text
direct ledger numeric/list stays 0/16
```

because then the speaker is being asked to pronounce a note that has not been
correctly written.

Promote DGX one-body only if:

```text
full depth > think0
full depth > destructive ablations
normal LM-token answer path is the only evaluated answer path
```

## Next Action

DGX:

```bash
bash scripts/launch_stage61a_dgx_onebody_fromscratch.sh run
```

Local:

```bash
EPOCHS=5 bash scripts/launch_stage61b_local_qwen_transplant_diagnostic.sh
```

If DGX one-body succeeds while local transplant fails, keep the architecture and
focus on alignment. If both fail, the recurrent working-memory architecture is
still not HRM-Text-like enough.
