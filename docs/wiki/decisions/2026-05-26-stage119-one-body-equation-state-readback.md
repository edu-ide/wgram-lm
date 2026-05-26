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

## Stage119 Experimental Contract (Minimal Viable Probe)

### Hypothesis (Plain Language)
The persistent algebra-under-misleading-demonstration failures are caused by the recurrent state not being forced to explicitly represent and solve the final equation. Additional final-answer preference pressure can mask the symptom on narrow gates but damages the normal one-body generation path. Forcing the recurrent state itself to bind the equation operands/operation and maintain a "solved" representation will produce a cleaner causal gain on hard algebra cases without regressing generation.

### Minimal Change (Causal Route)
Add a lightweight **equation-binding auxiliary objective** only on generated algebra trap data during training:

- At the step corresponding to the final equation in the prompt, require the recurrent state (z_L or a summary of the trajectory) to carry explicit information about:
  - Left operand value
  - Right operand value
  - Operator
  - Result variable being solved for

This can be implemented as:
- A small linear readout from the recurrent state → predict the above fields (or a contrastive "this state knows the correct binding" signal).
- Or by extending the existing typed register machinery (typed_value_registers / typed_digit_registers) with an explicit "equation_operand" role on algebra trap examples.

The final answer must still be generated through the normal LM head from the recurrent state. No side head is allowed to produce the promoted answer.

### Training Data
Reuse the exact generated non-heldout algebra trap data from Stage117/118 (misleading repeated demos + equations that require actual calculation).

### Loss Structure (on trap data only)
- Existing pairwise preference loss on (intelligence_answer vs parrot_answer) at the LM head.
- New auxiliary term: state-level equation binding loss (can start as simple regression or classification on the operand values + operator, or a contrastive "solved state" vs "unsolved state" signal).
- Small weight (e.g. 0.1–0.3) so it does not dominate the language modeling signal.

### Falsification Gate (Cheap Local Probe)
Train a short continuation (50–100 steps) from the current best anchor (Stage117 last) on the algebra trap data with the new auxiliary term.

Evaluate on:
1. The same heldout 44-row GD smoke (official_gdsuite_choice_probe).
2. Language heldout (8 cases).
3. Direct generation (12 cases).

**Promotion criteria** (must beat current anchor on the primary gate without destroying others):
- Hard algebra variants (original, v2fmt, instruction, numbered) show clear accuracy or margin improvement.
- No regression on already-passed families (CRT, code tracing, letter counting, etc.).
- Language loss/token accuracy does not materially regress vs anchor.
- Direct generation (exact + prefix accuracy) does not materially regress vs anchor.
- Strongest evidence: ablating the equation-binding component of the recurrent state (zeroing the relevant register channels or the auxiliary readout) specifically damages the hard algebra cases.

**Kill criteria**:
- No improvement on algebra variants, or
- Clear regression on language or direct generation, or
- The gain disappears when the auxiliary term is disabled (proving it's just extra scalar pressure, not state-level binding).

### Why This Is the Right Level of Change
- It directly attacks the new bottleneck identified after two rounds of preference engineering ("the state isn't doing the calculation").
- It stays strictly inside the one-body causal path (token → recurrent state → same LM head).
- It is falsifiable with a small local run + clean ablations.
- It builds on existing infrastructure (typed registers, state supervision patterns already present in the core).

## Open Implementation Questions (to resolve in the first probe)
- Use existing typed_value/typed_digit registers with a new role, or a dedicated lightweight "equation binding" vector?
- Supervise the state at every step, only at the final equation step, or via a contrastive "before vs after solving" signal?
- Keep the auxiliary loss only on trap data, or try to make it more general?

## Next Immediate Actions
1. Design the exact auxiliary loss / readout (smallest possible addition).
2. Add a `--stage119_equation_binding_weight` style flag to the BPE preference trainer (or a small wrapper).
3. Run a 60–80 step local probe from Stage117 last on the existing algebra trap data.
4. Run the standard 44-row GD smoke + language + generation gates.
5. Perform the state ablation and record the delta specifically on hard algebra.

## Status
Experimental contract defined. Ready for minimal implementation and local probe.

References: Stage118 handoff, Stage117/118 decision record, updated Active Decision Index (Stage101/117 line now points here for the next structural move).
