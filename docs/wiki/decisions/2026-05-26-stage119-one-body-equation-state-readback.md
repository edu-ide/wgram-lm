# 2026-05-26 Stage119 Direction: One-Body Equation-State Readback

## Context (from Stage117/118 evaluation)
After Stage113-114 established that preference pressure works, and Stage117/118 showed that targeted "fixed-parrot" curriculum can push GD accuracy to 1.0 on the algebra trap gate, we observed:
- Clear win on the specific bottleneck (misleading repeated answer + small algebra).
- But regression on direct generation quality.

This matches the diagnosis in the Stage118 handoff:
> "The model does not need more exposure to the exact wrong answer token. It needs a stronger normal answer path for binding the final equation and performing the small calculation before the LM head speaks."

We are hitting the limit of what additional scalar contrast on the final answer can achieve. We need the **recurrent state itself** to do more of the work of representing and manipulating the equation.

## Proposed Direction: One-Body Equation-State Readback

### Core Idea
Instead of training the model only to prefer the correct final numeric answer over the parrot answer at the LM head, we add explicit pressure for the recurrent latent state to:
1. Bind / reconstruct the key fields of the final equation from the prompt (operands, operation, result variable).
2. Maintain or update an internal representation of the "current solved state" of that equation across recurrent steps.
3. Use that state to influence the final answer generation through the normal one-body LM path.

The answer must still come from the same LM head, with no side renderer or external calculator at inference.

### Minimal Architecture Sketch (to be refined)
- Same BPE PrefixLM reader.
- Recurrent core (current dual-state or improved) receives not only the usual tokens but also has an explicit "equation binding" pathway or auxiliary objective during training on algebra traps.
- Training objective includes:
  - The existing preference loss (intelligence answer > parrot answer).
  - An additional term that supervises or contrasts the recurrent state's representation of the equation fields (e.g., via a lightweight readout or contrastive loss on whether the state "knows" the correct operand values and the operation to apply).
  - Final answer still generated autoregressively from the recurrent state + normal decoder path.

Key constraint (per project doctrine):
- Everything must remain inside the normal token → recurrent state → same LM head causal path.
- Destructive ablations (state binding off, recurrent core off, etc.) must remove the gain on the algebra trap gate.

### Fast Falsification Gate (proposed)
1. Start from the current best anchor (Stage117 or Stage118 if we later promote it).
2. On generated non-heldout algebra traps, add a training signal that forces the recurrent state to carry explicit information about the final equation (e.g., operand binding, operation identity, expected result structure).
3. Evaluate on the same heldout 44-row GD smoke + algebra variants.
4. Require:
   - Algebra variants to improve or stay strong.
   - No regression on already-passed families (CRT, code tracing, etc.).
   - Language and direct generation preservation (or at least no worse than current anchor).
5. Strongest evidence: a state ablation (zeroing or randomizing the equation-binding component of the recurrent state) drops performance specifically on the hard algebra cases.

### Why This Over More Preference Engineering?
- Multiple rounds of "stronger preference / more targeted negative examples" have given us big local wins but also exposed the ceiling (generation regression).
- The problem has moved from "the model doesn't know which answer is better" to "the model cannot reliably maintain and compute with the actual equation inside its thinking state."
- This calls for a change in what the recurrent state is responsible for, not just how loudly we tell the LM head which final token to prefer.

## Open Questions to Resolve Before Implementation
- What is the minimal supervision signal on the recurrent state? (Span prediction of operands? Operation classification? Contrastive "solved vs unsolved" state?)
- How do we avoid turning this into a side head that the normal answer path can ignore?
- Can we do this purely with contrastive / preference-style losses on the state (no heavy intermediate supervision)?
- Does this compose with the existing dual-state (z_L / z_H) design, or does it require a new typed register / belief state for equations?

## References
- Stage118 handoff: docs/wiki/handoffs/2026-05-26-stage118-local-gd-preference-handoff.md
- Stage117/118 decision record: docs/wiki/decisions/2026-05-26-stage117-stage118-generated-algebra-traps.md
- Active Decision Index update (Stage101/117 line)

## Status
Initial direction sketch. Not yet a full experimental contract.
Next step: flesh out the minimal training objective and a cheap local probe (small data, short training) to test whether state-level equation binding moves the hard algebra cases more cleanly than additional final-answer preference pressure.
