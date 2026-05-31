# W-GRAM Reasoning LM V2

V2 is the clean canonical path for the W-GRAM-LM reasoning-language model.

```text
byte input
-> dynamic BLT causal chunk summary
-> 3:1 gated-delta/attention recurrent core
-> same-body IMTA latent trajectories
-> same-body own-latent prediction auxiliary
-> answer-contract and self-rollout LM losses
-> answer-prefix memory planner
-> causal token maturation
-> hnet causal speaker
-> same LM head
-> free autoregressive generation
```

Forbidden as promotion evidence:

- forced-choice scoring;
- candidate reranking;
- external GRAM/PTRM answer selection;
- LeWM as a separate answer path;
- boundary-byte-only core input;
- multiple answer heads.

Tiny smoke tests may use the torch core in `recurrent_core.py`. Promotion runs
must use the official GatedDeltaNet-2 backend and pass
`validate_v2_contract(..., require_promotion_ready=True)`.

`core_implementation="official_gated_delta2"` wires the existing
`OfficialGatedDeltaNet2Mixer` adapter into V2's 3:1 recurrent/attention core
and refuses CPU forward passes because the official kernels require CUDA/Triton.

Own-latent prediction targets the next causal chunk state. It is not allowed to
become a shortcut answer head. Repetition handling is explicit: the primary
fastlane records loop statistics and leaves old anti-repeat decoding knobs off
by default. Any optional repetition penalty or repeat unlikelihood setting is a
diagnostic switch, not promotion evidence.

Public code should import `WGRAMReasoningLMV2` and `WGRAMV2Config`. Legacy
aliases `QTRMReasoningLMV2` and `QTRMV2Config` remain only so older checkpoints
and historical experiment scripts can still be inspected.
