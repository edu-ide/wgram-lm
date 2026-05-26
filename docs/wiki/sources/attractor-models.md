# Attractor Models

Primary source:

- `2605.12466` - Solve the Loop: Attractor Models for Language and Reasoning
  - arXiv: https://arxiv.org/abs/2605.12466
  - Submitted: 2026-05-12

Core mechanism:

- A backbone proposes output embeddings.
- An attractor module refines those embeddings toward a fixed point.
- Iteration count is chosen by convergence, not by a fixed recurrence depth.
- Training uses fixed-point solving and implicit differentiation so effective
  depth can grow without ordinary unrolled-memory cost.

Why it matters for QTRM/PV-GRAM:

- This is closer to the current failure than a simple halt head.
- A halt head answers "when should I stop?"
- An attractor model answers "what stable answer state should the loop converge
  to?"

Local mapping:

```text
BLT/PV-GRAM reader
-> recurrent thought proposal
-> attractor refinement toward fixed answer state
-> convergence-based adaptive stop
-> normal byte/token speaker
```

Current local evidence:

- Stage98C/98D/98E all show the same pattern:
  fixed-point residual falls with deeper `think_steps`, but held-out loss is
  best at depth 2.
- Offline convergence stopping from depth-probe rows does not beat the best
  fixed depth. It chooses depth 2 when the residual threshold is loose enough
  to get the best loss, and chooses depth 4/8 only when the answer loss worsens.

Local result snapshot:

```text
run      eval loss  best depth-probe loss/depth  best adaptive stop
Stage98C 2.5655     2.5655 @ depth 2             2.5655 @ threshold 0.3
Stage98D 2.9902     2.9564 @ depth 2             2.9564 @ threshold 0.3
Stage98E 2.5676     2.5675 @ depth 2             2.5675 @ threshold 0.3
Stage99B 2.5504     2.5503 @ depth 2             2.5503 @ threshold 0.3
Stage99C 2.5504     2.5503 @ depth 2             2.5503 @ threshold 0.3
Stage99D 2.5479     2.5486 @ depth 2             2.5486 @ threshold 0.3
Stage99E 2.5554     2.5558 @ depth 2             2.5558 @ threshold 0.2
Stage99F 2.5563     2.5566 @ depth 2             2.5566 @ threshold 0.3
Stage99G 2.5581     2.5588 @ depth 2             2.5588 @ threshold 0.3
Stage99H 2.5556     2.5560 @ depth 2             2.5560 @ threshold 0.3
Stage99I 2.5603     2.5604 @ depth 2             2.5604 @ threshold 0.3
```

Interpretation:

- Residual convergence is currently correlated with "less movement," not with
  "more correct answer."
- A convergence stop rule alone therefore collapses to either depth 2 (best
  loss) or depth 4/8 (lower residual but worse answer).
- Stage99B shows that direct answer-attractor training can improve the short
  eval gate, but it still does not make deeper depth the best answer state.
- Stage99C shows that pulling hidden states toward fixed gold speaker
  embeddings is trainable and moves the state geometry, but the normal speaker
  still does not read deeper states as better answers.
- Stage99D shows that a minimal self-embedding readback can slightly improve
  eval loss and confirms the normal speaker path can be made answer-causal, but
  it still does not make depth 4/8 better than depth 2.
- Stage99E shows that a latent-to-language inner-speech anchor is learnable,
  but naive anchor readback turns into a side task and worsens the main eval
  gate. A corpus-callosum-like bridge needs selected/global-workspace routing,
  not only a language anchor head.
- Stage99F shows that selected workspace readback is wired correctly, but an
  untrained selector becomes a diffuse averaging operation. The selector needs
  a verifier/critic target; otherwise the model has a broadcast channel without
  an editor.
- Stage99G shows that the selector can learn a sharp editor target, but
  anchor-CE is the wrong scorecard. The selector learned to choose a low-anchor
  candidate without improving the normal final speaker.
- Stage99H shows that final-speaker CE is the right scorecard but the current
  bridge has almost no candidate contrast. With 8 candidates, target confidence
  ended at 0.126, essentially the 0.125 uniform baseline. The final speaker
  therefore sees the candidate broadcasts as nearly interchangeable.
- Stage99H ends the bridge-first line as a main-path proposal. The project now
  needs HRM-Text-style one-body text learning, not another selected
  workspace/readback/anchor/selector variant.
- Stage99I implements the first one-body decoder gate by removing the direct
  grouped-byte decoder shortcut, but this alone does not make deeper recurrent
  thought more correct. It is a necessary routing correction, not a solved
  attractor.

Plain-language lesson:

```text
The current loop can become quiet, but it is not yet becoming correct.
Therefore the next architecture change should make recurrent thought the same
state that the normal speaker reads, not add another bridge beside the speaker.
```

Stage99B lesson:

```text
The first answer-attractor loss made the exam score slightly better, but the
valley is still at depth 2. The model has a shallow correct basin, not yet a
deep answer basin.
```

Stage99C lesson:

```text
The hidden state can be turned toward the right answer embedding, but that is
not enough. The speaker must be trained to read the attractor state itself as
the answer, otherwise deeper thought can be more "semantically pointed" while
still producing worse logits.
```

Stage99D lesson:

```text
The answer path can reread its own speaker expectation and get a small eval
gain, but self-readback is still circular. It makes the mouth listen to itself,
not to a verifier-selected thought. The next readback must choose which latent
state enters the global workspace before the final speaker produces logits.
```

Stage99E lesson:

```text
The inner-speech anchor learns quickly, but a learned anchor is not yet a
callosal bridge. The model can make a small verbal note from latent state, but
the note is not selected, verified, or globally broadcast before the final
speaker. In plain Korean: 언어 앵커는 배웠지만, 그 말풍선이 정답을 더 잘
말하게 만드는 뇌량은 아직 아니다.
```

Stage99F lesson:

```text
The callosal workspace path now exists, but it has no trained editor. The
selector keeps averaging many positions instead of confidently broadcasting one
useful candidate. In plain Korean: 방송국 건물은 지었지만 편집장이 없다.
```

Stage99G lesson:

```text
The editor can be trained, but it must read the right scorecard. Stage99G made
the selector follow low anchor CE, not low final answer CE. In plain Korean:
편집장은 생겼지만 시험 점수표가 아니라 말풍선 점수표를 보고 편집했다.
```

Stage99H lesson:

```text
The final speaker scorecard is correct, but the candidate broadcasts barely
change the final speaker. The bridge is not yet answer-causal. In plain Korean:
통역사는 단어를 조금 배웠지만, 말하는 쪽은 그 신호를 답안지 점수로 읽지
못한다.
```

Stage99I lesson:

```text
The direct byte shortcut can be removed, but that only forces the mouth to hear
the recurrent state. It does not by itself teach the recurrent state to walk
toward the answer valley. In plain Korean: 통역사를 빼고 한 몸으로 묶기 시작한
것은 맞지만, 그 몸이 오래 생각할수록 정답으로 가는 버릇은 아직 안 생겼다.
```

First local falsification gate:

```text
fixed depth 1/2/4/8
vs
convergence-selected adaptive depth
vs
attractor-trained adaptive depth

Promote only if convergence-selected or attractor-trained depth beats fixed
depth 2 on held-out loss/generation while using equal or lower average depth.
```

Next local training contract:

```text
native byte/token reader
-> mandatory recurrent thought/core/search
-> same decoder hidden state
-> same LM head
-> text

Hard requirements:
  no answer path that can bypass recurrent thought through raw byte/local
  shortcut alone
  recurrent-core-off must reduce the same held-out language/generation metric
  deeper/adaptive recurrent compute must beat or match shallow depth with a
  causal explanation
  same LM head must read the recurrent state directly

Required ablation:
  recurrent thought off, depth off, or one-body state off should remove the gain.

Stage99C narrows the next contract:
  fixed answer-state target alone is insufficient. The next candidate must add
  answer-causal readback, for example an attractor-refined state that is fed
  back through the same byte/token speaker and whose readback-off ablation
  destroys the gain.

Stage99D narrows the next contract:
  minimal self-embedding readback is insufficient. The next candidate must
  replace circular readback with selected workspace readback:
  candidate latent states -> verifier/selector -> selected readback vector ->
  same byte/token speaker. Promote only if depth 4 or adaptive selected depth
  beats fixed depth 2 on held-out loss or generation.

Stage99E narrows the next contract:
  naive inner-speech anchor readback is insufficient. The next candidate should
  either train the anchor with readback closed before opening the bridge, or use
  confidence/verifier-selected workspace readback. Promote only if the readback
  path improves held-out loss/generation and a readback-off ablation removes
  the gain.

Stage99F narrows the next contract:
  selected readback without selector supervision is insufficient. The next
  candidate must train the selector with a real critic signal:
  candidate depths/positions -> answer CE or verifier score -> selector target
  -> broadcast selected vector -> same byte/token speaker. Promote only if the
  selector becomes meaningfully sharper and the readback-off ablation removes
  the gain.

Stage99G narrows the next contract:
  selector supervision is insufficient if the scorecard is anchor CE. The
  selector must be trained against the final speaker path, not a side anchor
  head. Promote only if final answer loss/generation improves and the
  readback-off ablation removes the gain.

Stage99H narrows the next contract:
  final-speaker scoring is insufficient if candidate broadcasts are
  indistinguishable. Do not keep refining bridge/readback/anchor/selector as
  the main path. The next candidate must remove the split-body shortcut and use
  HRM-Text-style one-body routing: input -> recurrent thought -> same decoder
  hidden -> same LM head.

Stage99I narrows the next contract:
  one-body routing is now mandatory, but the first 400-step local gate rejected.
  Keep the bridge/selector ban. The next candidate must preserve one-body
  routing while restoring the strongest HRM-Text-like training contract and
  adding core-off/depth-off ablations. Promote only if depth/adaptive compute
  improves held-out generation or loss and removing recurrent state removes the
  gain.
```
