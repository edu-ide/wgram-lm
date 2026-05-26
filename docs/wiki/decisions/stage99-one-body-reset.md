# Stage99 One-Body Reset

Date: 2026-05-24

Architecture SSOT:

- [One-Body Architecture SSOT](../architecture/one-body-architecture-ssot.md)

Decision:

Stage99E/F/G/H ends the anchor/readback/selector bridge line as a main-path
architecture proposal. Bridge modules may remain only as diagnostic ablations.
The next general-LM/raw-intelligence architecture must be HRM-Text-style
one-body:

```text
input bytes/tokens
-> native reader
-> mandatory recurrent thought/core/search
-> same decoder hidden state
-> same LM head
-> text
```

Why:

- Stage99E learned inner-speech anchors but worsened the main held-out loss.
- Stage99F wired selected workspace readback but the selector stayed diffuse.
- Stage99G trained a sharp selector, but it selected low anchor CE rather than
  better final answers.
- Stage99H used final speaker CE as the scorecard, but candidate broadcasts
  were almost indistinguishable: 8-candidate target confidence ended at `0.126`,
  essentially the `0.125` uniform baseline.
- Depth probes still chose depth 2; deeper residual convergence made the state
  quieter, not more correct.

Plain-language read:

HRM-Text works because the model grows up as one body: the state that reads and
thinks is the state the LM head learns to speak from. Stage99 was a split-body
line: a latent thought organ, a bridge/interpreter, and a separate byte speaker.
The interpreter learned a small dictionary, but the speaker did not treat it as
its own thought.

Hard rejects after this decision:

- Qwen-pretrained convenience before a born-one-body baseline.
- Adapter, bridge, anchor, readback, selector, or sidecar as the proposed main
  solution for from-scratch/general-LM reasoning.
- Any byte/token shortcut that lets the final LM head lower loss while ignoring
  recurrent thought.
- Any promotion where recurrent-core-off, depth-off, bridge-off, or one-body
  state-off ties the full model.

Allowed uses:

- Reproducing Stage99 diagnostics.
- Ablating whether a bridge is non-causal.
- Comparing against the one-body path as a negative control.

Implementation guard:

`src/qtrm_mm/architecture/one_body_contract.py` is the code SSOT for blocking
Stage99-style answer-readback/anchor/selector losses by default. Trainer
scripts should import this guard rather than reimplementing bridge checks.
`scripts/557_train_blt_d_prefixlm_dataio.py` now delegates
`validate_architecture_contract` to that module. To reproduce a diagnostic
bridge run, the command must explicitly pass
`--allow-diagnostic-bridge-experiment`.

First implementation evidence:

`20260524_STAGE99I_LOCAL_ONE_BODY_GATE400` added
`--decoder-latent-mode one_body`, which removes the direct grouped-byte decoder
shortcut from the clean decoder input. This is the first code-level removal of
the split-body inertia.

Result:

```text
eval loss: 2.5603
accepted: false

depth 1 loss/residual: 2.5620 / 0.6703
depth 2 loss/residual: 2.5604 / 0.1890
depth 4 loss/residual: 2.5784 / 0.0994
depth 8 loss/residual: 2.6086 / 0.0544

best adaptive threshold: 0.3
best adaptive loss: 2.5604
selected depth: 2
```

Interpretation:

One-body routing is now enforced for the promoted path, but the first short
local gate did not create a deeper answer attractor. The direct shortcut is
gone; the remaining problem is teaching the recurrent state to become the
answer state under the same decoder/LM head.

Next architecture work:

Create a one-body decoder mode where recurrent thought is not an optional
conditioning side input. The final language path must depend on the recurrent
state directly:

```text
reader embeddings
-> recurrent core
-> decoder hidden
-> LM head
```

Promotion gate:

- Same data, same token budget, same eval rows.
- One-body mode beats or matches the best Stage99 bridge loss.
- Core/depth/state-off ablation removes the gain.
- Depth 4/adaptive depth improves over depth 1/2 or has a measured reason not
  to be used.
