# Paper Diagram Prompts

Status: working prompt bank, 2026-04-29.

Use these prompts with a raster image model when a polished paper-style figure
is needed. Keep the generated image as an illustration only; the source of
truth remains the architecture docs and Mermaid diagrams.

## Figure 1: QTRM Limitation-Mitigation Architecture

```text
Create a clean academic paper architecture diagram for a model named "QTRM:
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
1. Left column: "Inputs" with boxes for Prompt, Retrieved Evidence, Optional
   Visual/Text States.
2. Upper middle: large frozen module "Frozen Qwen Donor" producing "hidden
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
- Make the tiny cognitive core visually smaller than the donor.
- Show that MemoryOS/retrieval is external, not a giant prompt context.
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
Prompt and retrieved evidence enter two paths:
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
Show a tiny QTRM cognitive core coordinating Frozen Donor, MemoryOS Retrieval,
Verifier, Failure Memory, and Tool/Search Loop. Output label:
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
