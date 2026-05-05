# Paper Diagram Prompts

Status: working prompt bank, 2026-04-29.

Use these prompts with a raster image model when a polished paper-style figure
is needed. Keep the generated image as an illustration only; the source of
truth remains the architecture docs and Mermaid diagrams.

## Figure 0A: QTRM Model Architecture Only

```text
Create a clean research-paper style architecture diagram titled
"QTRM Model Architecture".

Style:
- white background, thin gray/black arrows, muted colors, arXiv / NeurIPS
  figure style.
- use labeled rectangular modules, arrows, and small callout boxes.
- no decorative gradients, no marketing style, no 3D.
- landscape 16:9, readable labels.

Important boundary:
Do not draw MemoryOS, retrieval, reranking, tools, browser, history store, or
external documents inside this figure. This is the model-only architecture.

Main flow:
1. "Canonical Token Stream"
   - already compiled before model forward
2. "Tokenizer / input_ids"
3. "Frozen Qwen3.5 Donor"
   - donor hidden states
   - optional donor logits
4. "QTRM Token Embedding"
5. "Donor-State Projector"
6. "Latent Workspace"
7. "Looped Recursive Core z_L / z_H"
8. "Coda / Core-to-Text Injection"
9. "Residual LM Head"
10. "Verifier / Answer Heads"
    - evidence bottleneck
    - answer decision
    - optional span reader over canonical tokens
11. "Bounded Donor-Logit Fusion"
12. "Final Logits / Answer Channel"

Add a thick outer boundary labeled:
"MODEL BOUNDARY: starts at canonical token stream"

Add a small note:
"MemoryOS is a runtime system outside this boundary."
```

## Figure 0B: QTRM Runtime With Optional MemoryOS

```text
Create a clean research-paper style architecture diagram titled
"QTRM Runtime With Optional MemoryOS".

Style:
- white background, thin gray/black arrows, muted colors, arXiv / NeurIPS
  figure style.
- use labeled rectangular modules, arrows, and small callout boxes.
- no decorative gradients, no marketing style, no 3D.
- landscape 16:9, readable labels.

Main runtime flow must be one vertical pipeline before entering the model:

1. "User Chat Prompt"
2. "Retrieval Query Derivation"
   - derived from the user prompt
   - not an independent semantic input
3. "MemoryOS Retrieval / Rerank"
   - retrieves candidate evidence records:
     signed/current source, anonymous/stale source, distractor source
4. "Context Compiler / Chat Template Builder"
   - merges user prompt, retrieved evidence, tool results, and memory records
   - produces one canonical chat-template text
   - label: "SSOT: all semantic information becomes one canonical token stream"
5. "Tokenizer + Frozen Qwen3.5-2B Donor"
   - encodes the same canonical token stream
   - outputs donor hidden states and donor logits
6. "QTRM Cognitive Core"
   - latent workspace
   - looped reasoning core
   - residual adapter
   - evidence bottleneck gate
   - truth / causality heads
7. "Answer Formation"
   - evidence span reader over token-aligned canonical context
   - span boundary revision
   - answer decision / abstention
8. "Final Answer"
   - examples: "Answer: stone-arch" and "Answer: UNKNOWN"

Show token-aligned metadata attached to the canonical token stream:
- source_id
- evidence_type
- trust_score
- selected_source_mask

Important callout:
"Evidence masks and source selectors annotate the SSOT token stream.
They do not create a second semantic context."

Add a red crossed-out mini-diagram:
"Wrong: Prompt path + separate workspace evidence path"

Add a green mini-diagram:
"Correct: Retrieval -> Context Compiler -> one canonical token stream"

Make MemoryOS Retrieval return into the Context Compiler, not into the model as
a second independent input path.

Draw a vertical boundary after "Context Compiler / Chat Template Builder":
left side label "Runtime / System Layer"; right side label "QTRM Model Layer".
MemoryOS must stay on the runtime side.
```

## Figure 1: QTRM Limitation-Mitigation Architecture

```text
Create a clean academic paper architecture diagram for a system named "QTRM:
Qwen-backed Tiny Reasoning Memory Adapter".

Style:
- arXiv / NeurIPS paper figure style, white background, thin black outlines,
  restrained accent colors only: muted blue for frozen Qwen donor, muted green
  for trainable QTRM cognitive core, muted orange for MemoryOS evidence, muted
  red for risk/telemetry gates.
- vector-like, crisp lines, high readability, no glossy 3D, no decorative
  gradients, no background texture.
- use rectangular modules with small rounded corners, arrows, and concise labels.
- landscape 16:9, enough margin, labels must be readable.

Diagram content:
1. Left column: "SSOT Context Compiler" with boxes for User Prompt, MemoryOS
   Retrieval, Tool Results, and Optional Visual/Text Records, all merging into
   one "Canonical Chat-Template Token Stream". This left column must be outside
   the model boundary and labeled "Runtime / System Layer".
2. Upper middle: large frozen module "Frozen Qwen Donor" consuming the canonical
   token stream and producing "hidden
   states" and "donor logits". Mark it with a snowflake/frozen symbol if
   available.
3. Lower middle: compact trainable module "Tiny QTRM Cognitive Core" containing
   four internal blocks: Projector, Latent Workspace, Recursive Core z_L/z_H,
   Residual LM Head.
4. Right middle: "Bounded Residual Fusion" combining donor logits and QTRM
   residual logits with a formula label:
   final logits = donor logits + gate * scale * residual.
5. Far right: "Answer / Action" with two outputs: Natural-language answer and
   NEEDS_SEARCH action.
6. Bottom row: "Mitigation & Evidence Gates" with four small boxes:
   donor-only baseline, workspace-off ablation, core-off ablation, KL-to-donor
   preservation, repetition/entropy telemetry.
7. Add a right-side vertical panel titled "Limitations -> Mitigations" with
   five compact rows:
   - donor dependency -> donor-only baseline + residual delta scoring
   - residual harms fluency -> gated residual + KL-to-donor preservation
   - latent reasoning unproven -> workspace_off/core_off ablations
   - retrieved evidence can mislead -> reranker + verifier + conflict tests
   - long-context cost -> MemoryOS retrieval + compact evidence window
8. Add a caption-like footer: "The donor provides fluency; the small QTRM core
   learns evidence-sensitive residual corrections. Claims require ablation
   evidence."

Important:
- Do not draw QTRM as replacing Qwen.
- Do not draw MemoryOS inside the model boundary.
- Make the tiny cognitive core visually smaller than the donor.
- Show that MemoryOS/retrieval is external before the context compiler, then
  returns into one canonical token stream before model forward.
- Make the "Limitations -> Mitigations" panel visible inside the figure, not
  just implied by the architecture.
- Avoid marketing style. Make it look like a serious ML systems paper figure.
```

## Figure 1B: Limitation-Mitigation Overlay

```text
Create a publication-quality overlay figure for the QTRM architecture showing
how each known limitation is addressed by a concrete mitigation.

Style:
- academic ML paper figure, white background, crisp vector-like rendering,
  readable 9-11 pt labels, thin black connector lines, restrained colors.
- use red outline callouts for limitations and green outline callouts for
  mitigations.
- no marketing style, no glowing effects, no 3D, no abstract decoration.

Main architecture:
User Prompt and MemoryOS Retrieval first merge in an "SSOT Context Compiler",
which emits one canonical token stream. That single stream then enters:
1. Frozen Qwen Donor -> hidden states + donor logits.
2. Tiny QTRM Cognitive Core -> projector -> latent workspace -> recursive
   z_L/z_H core -> residual LM head.
Both paths merge in Bounded Residual Fusion:
final logits = donor logits + gate * scale * residual.
Output goes to Answer or NEEDS_SEARCH.

Overlay five numbered limitation callouts directly on the architecture:
L1 "Donor dependency": place near Frozen Qwen Donor.
M1 "Mitigation: donor-only baseline; residual delta must improve held-out
tasks without fluency loss."

L2 "Residual may damage fluency": place near Bounded Residual Fusion.
M2 "Mitigation: gated residual, residual clamp, KL-to-donor preservation,
entropy/repetition telemetry."

L3 "Latent reasoning not proven": place near Latent Workspace and Recursive
Core.
M3 "Mitigation: workspace_off and core_off ablations; depth sweeps; causal
performance delta."

L4 "Retrieved evidence may be wrong or distracting": place near MemoryOS
Retrieval.
M4 "Mitigation: reranker, verifier, source/time metadata, conflict and
distractor tests."

L5 "Long context and inference cost": place around the whole pipeline.
M5 "Mitigation: external MemoryOS index, compact evidence window, donor-cache,
run residual only on evidence-sensitive steps."

Bottom footer:
"Claim boundary: QTRM does not replace the donor. It aims to improve
evidence-sensitive reasoning per compute through retrieval, verification, and
bounded residual correction."

Important:
- The five L/M pairs must be visible as explicit text in the image.
- Do not hide mitigations as tiny footnotes.
- Keep QTRM visually smaller than the donor.
- Make the diagram suitable for a paper section titled "Limitations and
Mitigations".
```

## Figure 2: Ablation Matrix

```text
Create a clean academic ablation diagram for QTRM evaluation.

Style:
- white background, publication-quality vector figure, black text, thin lines.
- use four horizontal lanes with consistent spacing.
- muted color accents only.

Content:
Title: "QTRM Component Ablation Modes".

Lane 1: "donor_only"
Prompt -> Frozen Qwen Donor -> Donor Logits -> Output.
Show QTRM disabled.

Lane 2: "residual"
Prompt -> Frozen Qwen Donor -> donor logits.
Donor hidden states -> Projector -> Latent Workspace -> Recursive Core ->
Residual LM Head -> Bounded Residual Fusion -> Output.

Lane 3: "workspace_off"
Prompt -> Frozen Qwen Donor -> donor logits.
Donor hidden states -> Projector -> Prelude/Coda residual path -> Fusion ->
Output. Mark Latent Workspace and Recursive Core as crossed out.

Lane 4: "core_off"
Prompt -> Frozen Qwen Donor -> donor logits.
Donor hidden states -> Projector -> Latent Workspace -> Residual LM Head ->
Fusion -> Output. Mark Recursive Core z_L/z_H update as bypassed.

Bottom strip: metrics collected for every lane:
target-token rank, top-k, argmax changed, KL(fused || donor), residual norm,
entropy, repetition, answer correctness.

Important:
- The diagram should make causal isolation obvious.
- Use exact labels: donor_only, residual, workspace_off, core_off.
- Avoid dense equations except the small fusion formula:
final logits = donor logits + QTRM residual.
```

## Figure 3: Realistic Claim Boundary

```text
Create a paper-style conceptual figure explaining the realistic claim boundary
for a small cognitive-core LLM system.

Style:
- clean two-column academic diagram, white background, simple icons, restrained
  colors, readable labels.

Left column title: "Unsupported claim"
Show a tiny core directly replacing a giant 1T-parameter model for all tasks.
Use a red warning marker and label: "not proven".

Right column title: "Testable claim"
Show a tiny QTRM cognitive core inside the model boundary, plus an external
runtime layer coordinating MemoryOS Retrieval, Verifier, Failure Memory, and
Tool/Search Loop. Output label:
"better accuracy-per-compute on evidence-sensitive tasks".

Bottom: "Required proof"
Use five checkboxes:
donor-only baseline, held-out retrieval tasks, conflict/distractor tests,
workspace/core ablations, cost and latency comparison.

Important:
- The figure must be sober and technical, not promotional.
- Emphasize that the small core can win through orchestration, evidence, and
  verification, not by magically storing all knowledge internally.
```

## Figure 4: Logical-Causal Evidence Bottleneck

```text
Create a clean academic paper architecture diagram for "QTRM Logical-Causal
Evidence Bottleneck".

Style:
- NeurIPS/arXiv paper figure, white background, crisp vector-like rendering,
  thin black outlines, readable labels.
- restrained colors: muted blue for frozen Qwen donor, muted green for trainable
  QTRM core, muted orange for MemoryOS evidence, muted red for refute/missing
  gates, gray for ablations.
- no glossy effects, no 3D, no decorative gradients, no background texture.
- landscape 16:9 with enough spacing for all text.

Main diagram:
1. Left runtime side: "User Prompt" plus optional "MemoryOS Retrieval" enter
   "Context Compiler", which emits one "Canonical Token Stream".
2. Model side: the canonical token stream goes into "Frozen Qwen Donor".
3. Donor has two outputs:
   - "Donor Hidden States" into QTRM Projector.
   - "Donor Logits" into final fusion.
4. Add a dashed optional probe path labeled:
   "workspace/dual ablation only: hidden evidence states".
   Make clear this is not the canonical model input.
5. Middle trainable QTRM stack:
   "Prelude" -> "Latent Workspace" -> "Recursive Core z_L / z_H" -> "Coda".
6. From the final workspace state, branch into four verifier heads:
   "Support", "Refute", "Missing / NEI", "Causal Evidence Gate".
7. Show the bottleneck formula near the gate:
   gate = sigmoid(causal + support - refute - missing).
8. The gate controls "Bounded QTRM Residual Logits".
9. Final fusion:
   "Final Logits = Donor Logits + Evidence Gate * Residual".
10. Right: output box "Answer / NEEDS_SEARCH".

Limitations and mitigations panel inside the same figure:
- L1: Retrieved target found but answer wrong.
  M1: Evidence gate must open only for supported canonical evidence.
- L2: Fake or counterfactual evidence can distract.
  M2: Counterfactual workspace loss plus refute/missing heads.
- L3: Residual may damage donor fluency.
  M3: Residual clamp, residual gate, donor KL.
- L4: Latent workspace causality unproven.
  M4: workspace_memory_off and evidence_bottleneck_off ablations.

Bottom ablation strip:
"full", "workspace_memory_off", "core_context_off",
"evidence_bottleneck_off", "counterfactual evidence".

Important:
- Do not show QTRM as replacing the donor.
- Do not draw MemoryOS inside the model boundary.
- Make the bottleneck visibly between recursive workspace state and residual
  logits.
- Make "workspace-only evidence context" visually distinct as an ablation probe,
  not as canonical model input.
- The figure should communicate that the innovation is a measurable causal
  evidence-to-answer path, not just a larger prompt.
```

## Figure 5: Canonical QTRM Active-Vs-Disabled Matrix

```text
Create a publication-quality architecture figure titled
"QTRM Canonical Architecture: Active Path and Implemented Scaffolds".

Style:
- serious ML paper figure, white background, crisp vector-like rendering,
  readable labels, thin outlines.
- use muted blue for frozen Qwen donor, muted green for active trainable QTRM
  modules, muted orange for MemoryOS evidence, muted gray for implemented but
  disabled/scaffold modules, muted red for ablation gates.
- no 3D, no glossy effects, no decorative gradients.
- landscape 16:9.

Main active path:
1. Left runtime side: "User Prompt" and optional "MemoryOS Retrieval/Rerank"
   enter "Context Compiler", which emits one "Canonical Token Stream".
   Draw this runtime side outside the model boundary.
2. Model side: "Canonical Token Stream" enters "Frozen Qwen Donor".
3. Donor outputs:
   - "Donor Hidden States" to "QTRM Projector".
   - "Donor Logits" to "Donor-Residual Fusion".
4. Optional dashed probe path:
   "workspace/dual hidden evidence states" -> "QTRM Projector".
   Label: "ablation probe, not canonical model input".
5. Active QTRM stack:
   Projector -> Prelude -> Latent Workspace with Memory Gate ->
   Recursive Core z_L/z_H with Gated Core Context -> Coda -> Residual LM Head.
6. Evidence bottleneck branch from recursive z_H:
   Support, Refute, Missing/NEI, Causal Evidence Gate.
   Formula: evidence_gate = sigmoid(causal + support - refute - missing).
7. Residual governor:
   Residual Clamp + Residual Gate + Evidence Gate.
8. Fusion:
   final text logits = donor logits + gated bounded residual.
9. Output: Answer or NEEDS_SEARCH.
10. Core LeWM auxiliary branch:
   z_H trajectory + action trace(RETRIEVE/VERIFY/ANSWER) ->
   LeWM Predictor -> next z_H loss + SIGReg.

Right-side status panel:
Active in canonical probe:
- donor logits
- bounded residual gate
- canonical token stream
- evidence bottleneck
- counterfactual workspace loss
- core LeWM trajectory loss

Implemented but not canonical:
- token-level JEPA loss
- controller aux heads
- core halting / early exit
- donor annealing
- multimodal image scaffold
- external reranker / symbolic verifier

Bottom ablation strip:
donor_only, workspace_memory_off, core_context_off, evidence_bottleneck_off,
core_world_model_off, counterfactual evidence.

Important:
- Do not imply QTRM replaces the donor.
- Do not imply MemoryOS is inside the model.
- Show disabled/scaffold modules in gray and visually separate them from the
  active green path.
- Make the donor-fusion boundary explicit: donor-free QTRM is not the canonical
  architecture yet.
```

## Figure 6: Agentic Closed-Loop Planner

```text
Create a clean academic paper architecture diagram titled
"QTRM/MemoryOS Agentic Closed-Loop Planner".

Style:
- arXiv / NeurIPS systems figure, white background, crisp vector-like shapes,
  thin black outlines, readable 9-11 pt labels.
- restrained colors: muted blue for frozen donor/QTRM policy, muted orange for
  MemoryOS/tools/environment, muted green for verifier/reward/training, muted
  purple for trace/skill memory, muted red for collapse diagnostics.
- no 3D, no glossy effects, no decorative gradients, no marketing style.
- landscape 16:9.

Main loop:
1. Left: "Task / User Goal" enters "AgentHarness".
2. Inside AgentHarness show a circular closed loop with numbered stages:
   Observe State -> QTRM/Donor Policy -> Choose Action -> Execute Tool/MemoryOS
   -> Observation -> Verifier Reward -> TraceStore -> Next Step or Stop.
3. QTRM/Donor Policy box contains:
   Frozen Qwen donor, QTRM residual adapter, LatentWorkspace, Recursive Core,
   ControllerHeads.
   Mark ControllerHeads as "to train".
4. Action space box:
   OBSERVE, RETRIEVE_MEMORY, SEARCH_WEB, VERIFY_EVIDENCE, WRITE_MEMORY,
   WRITE_SKILL, SIMULATE, ANSWER, STOP.
5. Environment box:
   MemoryOS retrieval, web/search tools, executable sandbox, optional symbolic
   world model simulator.
6. Verifier box:
   support/refute/missing, executable tests, citation/source checks, cost and
   budget gates.
7. TraceStore / Replay Buffer box:
   state, action, observation, verifier result, reward, memory writes, skill
   writes, checkpoint.
8. Training branch from TraceStore:
   Trace SFT -> Preference / SimPO -> Turn-level RL credit assignment ->
   updated ControllerHeads / QTRM residual.
9. Model-based planning branch:
   candidate actions -> LeWM-style latent world model -> predicted next z_H ->
   value/evidence gate -> action selection.
10. Right-side red diagnostics panel:
   echo trap, template collapse, interaction collapse, reward hacking, memory
   contamination. Add guards: action repeat rate, MI/input-dependence proxy,
   tool/evidence density, hidden replay tests, typed memory gates.

Bottom claim boundary:
"Current status: scaffold. QTRM is not yet an autonomous agent. First build
replayable traces; then train controller heads; then add turn-level RL and
latent world-model planning."

Important:
- Do not draw all tools as differentiable neural modules.
- Make MemoryOS/tools clearly external environments.
- Show the loop returning from verifier/trace store back to action selection.
- Make collapse diagnostics visible in the figure, not hidden in a footnote.
```

## Figure 7: 2026 Prompt-Conditioned Memory Reader

```text
Create a clean academic paper architecture diagram titled
"QTRM 2026: Prompt-Conditioned Memory Reader".

Style:
- arXiv / NeurIPS paper figure, white background, crisp vector-like rendering,
  thin black outlines, readable labels.
- restrained colors: muted blue for frozen Qwen donor, muted orange for
  MemoryOS hidden evidence, muted green for trainable QTRM reader/core, muted
  red for UNKNOWN/ablation gates, muted gray for disabled bypass paths.
- no glossy effects, no 3D, no decorative gradients, no marketing style.
- landscape 16:9 with enough spacing.

Main diagram:
1. Left top: "Visible Prompt / Question" enters "Frozen Qwen3.5 Donor".
2. Donor outputs "Prompt Hidden States" and "Donor Logits".
3. Left bottom: "MemoryOS Retrieval" outputs "Hidden Workspace Evidence
   Tokens". Label: "not visible prompt text".
4. Middle: "Prompt-Conditioned Memory Reader" with two inputs:
   - query projection from prompt/question states
   - key/value projection from hidden evidence token states
5. Inside the reader show four heads:
   "Evidence Selector", "Start Span", "End Span", "No-Answer / UNKNOWN".
6. Reader outputs "Selected Evidence Span or UNKNOWN".
7. Selected span enters "Answer-Only Copy/Decoder Channel".
8. The answer channel controls "Bounded QTRM Residual Logits".
9. Final fusion:
   "Final Logits = Donor Logits + Evidence-Gated Residual".
10. Right output: "Short Answer / UNKNOWN / NEEDS_SEARCH".

Show the rejected old path as a gray dashed bypass below:
"Hidden Evidence -> Latent Workspace -> Free-form Residual Logits".
Put a red label on it: "rejected by workspace-swap gate".

Bottom ablation strip:
"full", "workspace_memory_off", "span_reader_off", "evidence_gate_off",
"workspace_swap".

Right-side source panel:
- LM2: auxiliary memory + cross-attention/gates
- G-MemLLM: frozen backbone + gated latent memory
- MemCoT: task-conditioned memory/evidence localization
- MSA: trainable sparse memory at 100M-token scale

Caption footer:
"2026 correction: memory can be a separate lane, but the question must
condition memory selection. QTRM should prove causal evidence use before
claiming latent reasoning."
```
