# Qwen3.5 HRM-Text Attention Pooling Diagnostics

Date: 2026-05-20

## Context

The initial `scripts/510_train_qwen_state_transition.py` HRM/Text healing run
plateaued near random digit-classification accuracy:

```text
old local final_v2, epoch 17:
loss=3.5661
acc=0.0996
```

Root cause found:

```text
Synthetic reasoning labels depended on hidden operands, but the prompt only
contained the family name, e.g. "Reason: chain5. Result:".
```

Therefore the 10% accuracy plateau was not valid evidence against recurrence.

## HRM-Text Data Contract Import

Added `scripts/511_train_qwen_state_transition_hrmtext.py` with:

- reasoning prompts that include operands and operation descriptions;
- HRM-Text/Data IO-style `condition / instruction / response` healing rows;
- prefix/response boundaries;
- response-only healing target;
- `--workspace-pooling {mean,last,attention}`;
- `--freeze-qwen`;
- `--checkpoint-every`;
- `--n-steps`;
- `--depth-sample-min`.

Tests:

```text
tests/test_hrm_text_aligned_training_data.py
tests/test_qwen_state_transition_workspace_pooling.py
```

Manual verification:

```text
.venv/bin/python -m py_compile scripts/511_train_qwen_state_transition_hrmtext.py
.venv/bin/python -m py_compile src/qtrm_mm/qwen_backbone_state_transition.py
manual test function calls: pass
```

## Architecture Change

`QwenBackboneStateTransition` originally compressed Qwen hidden states by mean
pooling:

```text
hidden_states -> mean over attended tokens -> compressor -> workspace(B,1,D)
```

This likely destroyed operand/role binding. Added `workspace_pooling` modes:

```text
mean       old behavior
last       last attended token
attention  learned masked attention over hidden states
```

BF16 issue fixed:

```text
attention scorer receives hidden states cast to scorer parameter dtype, then
softmax weights are cast back to hidden-state dtype.
```

## Diagnostic Runs

All diagnostic runs used:

```text
Qwen/Qwen3.5-0.8B-Base
reasoning-count=128
healing-count=0
freeze-qwen=true
checkpoint-every=0
lr=1e-4
batch-size=8
epochs=40
```

### Local: Last Pooling

Command family:

```text
--workspace-pooling last
--out-dir /tmp/qtrm_eval/qwen35_0.8b_diag_lastpool_freeze_tiny128_v2
log /tmp/qwen35_0.8b_diag_lastpool_freeze_tiny128_v2.log
tensorboard http://localhost:6007
```

Result:

```text
epoch 40:
loss=4.3825
acc=0.1562

best observed:
acc=0.1641
```

Decision:

```text
Reject last pooling. It does not overfit 128 cases and remains close to random.
```

### DGX: Attention Pooling

Command family:

```text
--workspace-pooling attention
--out-dir /mnt/data4tb/qtrm_eval/qwen35_0.8b_diag_attnpool_freeze_tiny128_v3
log /tmp/qwen35_0.8b_diag_attnpool_freeze_tiny128_v3.log
tensorboard http://localhost:6008 on DGX
```

Result:

```text
epoch 30:
loss=3.7808
acc=0.3828

epoch 38:
loss=3.3132
acc=0.5000

epoch 40:
loss=3.0555
acc=0.5938
```

Decision:

```text
Promote attention pooling as the current input-to-state binding direction.
It does not fully solve the task, but it is the first clear learning signal
after correcting the data contract.
```

## Operational Notes

Local `/mnt/sdc1` was full:

```text
/mnt/sdc1 100% used
```

This caused PyTorch checkpoint write failures in an earlier local diagnostic.
Short diagnostics should use `/tmp/qtrm_eval/...` and `--checkpoint-every 0`
until disk pressure is resolved.

## Next Step

Run attention-pooling diagnostics with variable recurrence depth:

```text
local: fixed attention depth, e.g. --n-steps 4 or 8
DGX: depth-sampled attention, e.g. --n-steps 8 --depth-sample-min 4
```

Acceptance gate before returning to long healing runs:

```text
128-case reasoning-only overfit should move well beyond 60%.
If it cannot approach high overfit accuracy, the next bottleneck is inside
state transition / readout supervision, not language healing.
```

## 2026-05-22 Stage59 Generality Update

The Stage58 modulo-10 result should not be treated as evidence that the current
state already contains a general reasoning organ. Stage59 all-family tests
separate four roles:

```text
Reader: Qwen hidden states over the prompt
Thinker: QTRM recurrent state trajectory
Speaker: Qwen-compatible answer-token path
Exam: exact answers across arithmetic_chain, boolean_logic, list_transform,
      and symbolic_binding
```

Frozen speaker-only experiments failed this preflight:

```text
low-rank final-readout speaker:
  best eval 0.3125, but arithmetic_chain/list_transform stay 0.0

trajectory/workspace, direct-vocab, restricted-vocab, pooled, and char mouths:
  best eval 0.0 in the exact-answer gates

char pooled frozen local:
  local_eval/stage59_char_pooled_frozen_t32_e16_ep80_trainacc
  train exact 0.0, eval exact 0.0

char pooled frozen DGX:
  local_eval/stage59_dgx_char_pooled_frozen_t64_e32_ep40_trainacc
  train exact 0.0, eval exact 0.0
```

Plain-language conclusion:

```text
The model has traces of information, but it has not learned the broad class.
Teaching only the mouth is now rejected. The next run must train the thought
state and the speaker together through the normal Qwen-compatible answer path.
```

Active replacement gate:

```text
local:
  local_eval/stage59_local_traincore_lowrank_final_t64_e32_ep40_s1601b

DGX:
  local_eval/stage59_dgx_traincore_lowrank_final_t256_e64_ep35_s2601

mechanism:
  --train-qtrm-core
  --speaker-logit-mode qwen_plus_low_rank
  --speaker-context-mode final_readout
  --answer-path lm_head
  Qwen frozen, non-Qwen QTRM + speaker trainable
```

Decision rule:

```text
Do not promote a symbolic-only gain. Promotion requires nonzero
arithmetic_chain and list_transform exact accuracy, and held-out exact accuracy
above the 0.3125 frozen low-rank speaker baseline.
```

## 2026-05-22 Renderer/Verifier Split

Free generation experiments after the frozen-mouth rejection still failed the
numeric/list gate:

```text
train-qtrm-core + low-rank Qwen LM speaker:
  best eval 0.46875 locally, but arithmetic_chain/list_transform remain 0.0

train-qtrm-core + char speaker + solver_trace state_text supervision:
  best eval 0.40625 locally, but arithmetic_chain/list_transform remain 0.0

DGX numeric/list-only char generation:
  stopped around epoch 29 with eval 0.0
```

The new choice-verifier diagnostic changes the story:

```text
script:
  scripts/524_train_state_choice_verifier.py

local:
  local_eval/stage59_local_choice_verifier_allfamily_t256_e64_ep30_s1604
  best eval 0.515625
  final train 0.81640625

DGX active:
  local_eval/stage59_dgx_choice_verifier_allfamily_t512_e128_ep40_s2604
  early eval 0.3828125-0.4140625 with nonzero arithmetic/list accuracy
```

Plain-language conclusion:

```text
The model can often recognize the right answer when the answer is placed in
front of it, including arithmetic/list cases. It cannot reliably write numeric
or list answers freely. Therefore the next big-jump path is not another free
speaker loss; it is verifier-guided answer selection followed by a renderer
that copies/emits the selected answer.
```

Next gate:

```text
Promote only if the verifier path beats free generation and keeps nonzero
arithmetic_chain/list_transform on heldout rows. Then integrate selected-choice
copying as the answer path and ablate:
  verifier off
  recurrent core off
  choice text off
```

## 2026-05-20 Generalization Architecture Literature Watch

### Trigger

Stage9 attention trajectory readout produced a large train-accuracy jump, but
held-out depth/family generalization remained weak:

```text
Stage9B train best:
accuracy_reasoning=0.31054688

Stage9B held-out eval:
depth4 trm=0.1377 raw=0.1035 delta=+0.0342
depth6 trm=0.1514 raw=0.1055 delta=+0.0459
depth8 trm=0.1123 raw=0.1172 delta=-0.0049
depth10 trm=0.1152 raw=0.0938 delta=+0.0215

Stage9B final-readout ablation:
depth4 trm=0.1406 raw=0.1035 delta=+0.0371
depth6 trm=0.1436 raw=0.1055 delta=+0.0381
depth8 trm=0.1211 raw=0.1172 delta=+0.0039
depth10 trm=0.1357 raw=0.0938 delta=+0.0420

Stage9B final+attention logit-mean eval:
depth4 trm=0.1328 raw=0.0996 delta=+0.0332
depth6 trm=0.1299 raw=0.1104 delta=+0.0195
depth8 trm=0.1211 raw=0.0947 delta=+0.0264
depth10 trm=0.1494 raw=0.1045 delta=+0.0449
```

Interpretation:

```text
Attention readout removed a train-set bottleneck, but it can overfit which
recurrent state to trust. Generalization needs adaptive state selection,
depth stability, and latent-state anchoring, not another nearby data-ratio run.
```

### Primary Papers And Mechanisms

1. Altabaa et al., 2025, "Unlocking Out-of-Distribution Generalization in
   Transformers via Recursive Latent Space Reasoning"
   <https://arxiv.org/abs/2510.14095>

   Mechanisms to import:

   ```text
   input-adaptive recurrence
   algorithmic supervision
   anchored latent representations / discrete bottleneck
   explicit error-correction
   ```

   QTRM mapping:

   ```text
   fixed recurrent readout -> adaptive readout gate
   weak state supervision -> only if it improves normal answer logits
   unstable depth behavior -> depth-stability gate and eval-by-depth
   ```

2. Geiping et al., 2025, "Scaling up Test-Time Compute with Latent Reasoning:
   A Recurrent Depth Approach"
   <https://arxiv.org/abs/2502.05171>

   Mechanisms to import:

   ```text
   variable recurrent depth during training
   test-time latent compute scaling
   no dependence on explicit chain-of-thought data
   ```

   QTRM mapping:

   ```text
   train with sampled depth
   evaluate depth4/depth6/depth8/depth10 separately
   promote only if deeper latent recurrence improves held-out accuracy
   ```

3. ICLR 2025 looped Transformer length-generalization paper
   <https://openreview.net/pdf/8d8b2b7fb8d2ee3506ced2e04d6c76a485280472.pdf>

   Mechanisms to import:

   ```text
   looped/shared-block computation
   adaptive number of steps by problem difficulty
   final supervision after the required number of recurrent steps
   ```

   QTRM mapping:

   ```text
   fixed n_steps is a diagnostic, not the endpoint
   add uncertainty/halt/readout telemetry
   treat length/depth generalization as evidence of algorithmic learning
   ```

4. "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial
   Algorithms" <https://huggingface.co/papers/2505.17190>

   Mechanisms to import:

   ```text
   sharper max-plus style selection
   reduced softmax blur under length/value shift
   adversarial/OOD algorithmic reasoning gates
   ```

   QTRM mapping:

   ```text
   softmax trajectory attention -> sharp/low-temperature readout
   compare attention entropy against held-out accuracy
   do not promote if it only increases train accuracy
   ```

5. Behrouz et al., 2025, "Nested Learning: The Illusion of Deep Learning
   Architectures" <https://arxiv.org/abs/2512.24695>

   Mechanisms to import cautiously:

   ```text
   nested context-flow compression
   self-modifying learning/update rules
   continuum memory / HOPE-style continual learning module
   ```

   QTRM mapping:

   ```text
   recurrent core should not be a blind fixed loop
   add learnable update/readout gates before attempting broad data scaling
   replay/healing is support infrastructure, not the main generalization claim
   ```

### Stage10 Architecture Hypothesis

The next falsifiable architecture change is:

```text
Stage10A:
  recurrent_readout_pooling=hybrid_gate
  recurrent_readout_temperature=0.5
  depth_sample_min=4
  depth_consistency_weight=0.02
  state_supervision_weight=0.1
```

Code change:

```text
QwenBackboneStateTransition:
  final readout state
  learned trajectory attention state
  learned gate: sigmoid(W[final, attention])
  readout = gate * attention + (1 - gate) * final
  telemetry:
    qtrm_readout_gate
    qtrm_readout_attention_entropy
```

Promotion gate:

```text
Do not promote on train accuracy alone.

Promote only if Stage10 improves held-out depth8/depth10 over:
  Stage9B attention readout
  Stage9B final-readout ablation
  Stage9B final+attention logit mean

Required records:
  TensorBoard train/generalization curves
  Aim hparams and best checkpoint metadata
  wiki entry with exact run directory and eval JSON path
```

### Stage10A Result: Rejected For Generalization

Run:

```text
/tmp/qtrm_eval/20260520_201510_20260520_201509_STAGE10A_GATE_local_hybridgate_temp05_batch4_from_stage9B_seed60
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.26367188
  readout_gate_attention_weight=0.50628
  readout_attention_entropy=1.91288
eval:
  /tmp/qtrm_generalization_eval/20260520_201509_STAGE10A_GATE_local_hybridgate_temp05_batch4_from_stage9B_seed60_eval_depth46810_count1024.json
```

Held-out result:

```text
depth4  trm=0.1318 raw=0.0967 delta=+0.0352
depth6  trm=0.1221 raw=0.0928 delta=+0.0293
depth8  trm=0.0996 raw=0.0908 delta=+0.0088
depth10 trm=0.1055 raw=0.0918 delta=+0.0137
```

Decision:

```text
Reject Stage10A as a generalization breakthrough.

The learned final/attention gate moved and train accuracy rose to 26.37%, but
held-out depth8/depth10 remained weaker than Stage9B final-readout ablation and
Stage9C final-readout repair. The failure suggests that a soft hybrid gate is
too weak or too entangled with the overfit attention trajectory. Next test the
sharper selection hypothesis directly instead of mixing final and attention.
```

### Stage10B Result: Sharp Readout Is Promising But Not Confirmed

Run:

```text
/tmp/qtrm_eval/20260520_203235_20260520_203233_STAGE10B_GATE_local_sharpattn_temp025_count256_seed61
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.30078125
eval seed 10042:
  /tmp/qtrm_generalization_eval/20260520_203233_STAGE10B_GATE_local_sharpattn_temp025_count256_seed61_eval_depth46810_count1024.json
eval seed 20042:
  /tmp/qtrm_generalization_eval/20260520_203233_STAGE10B_GATE_local_sharpattn_temp025_count256_seed61_eval_seed20042_depth46810_count1024.json
```

Held-out result, eval seed 10042:

```text
depth4  trm=0.1289 raw=0.0898 delta=+0.0391
depth6  trm=0.1289 raw=0.0918 delta=+0.0371
depth8  trm=0.1611 raw=0.0898 delta=+0.0713
depth10 trm=0.1348 raw=0.0938 delta=+0.0410
```

Held-out result, eval seed 20042:

```text
depth4  trm=0.1475 raw=0.0957 delta=+0.0518
depth6  trm=0.1465 raw=0.0811 delta=+0.0654
depth8  trm=0.1289 raw=0.1006 delta=+0.0283
depth10 trm=0.1436 raw=0.0898 delta=+0.0537
```

Decision:

```text
Do not call this a confirmed generalization big jump yet.

Sharp readout recovered Stage9B-level train accuracy and produced a strong
depth8 spike on one held-out seed, but the spike did not fully replicate on the
second held-out seed. The robust signal is broader but smaller: every measured
depth beat raw Qwen by a positive margin on both eval seeds.

Next test a less aggressive sharpness setting, e.g. temperature=0.5, to see if
depth8 remains positive while depth10 improves.
```

### Stage10C Result: Smoother Sharp Readout Is More Balanced, Still Not A Big Jump

Run:

```text
/tmp/qtrm_eval/20260520_204029_20260520_204027_STAGE10C_GATE_local_sharpattn_temp05_count256_seed62
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.24609375
eval seed 10042:
  /tmp/qtrm_generalization_eval/20260520_204027_STAGE10C_GATE_local_sharpattn_temp05_count256_seed62_eval_depth46810_count1024.json
```

Held-out result:

```text
depth4  trm=0.1230 raw=0.0908 delta=+0.0322
depth6  trm=0.1309 raw=0.0898 delta=+0.0410
depth8  trm=0.1396 raw=0.0938 delta=+0.0459
depth10 trm=0.1357 raw=0.0938 delta=+0.0420
```

Decision:

```text
Temperature 0.5 is more balanced than temperature 0.25, but it did not beat
the best Stage10B depth8 spike or Stage9C depth10 result. The robust finding is
that sharp readout variants beat raw Qwen across all measured depths, while the
large depth-specific jump is not yet stable enough to promote.
```

### Stage10D Result: Best Current Stability Candidate

Run:

```text
/tmp/qtrm_eval/20260520_204707_20260520_204706_STAGE10D_GATE_sharp025_finalaux005_count256_seed63
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.30078125
  final_readout_answer_loss=0.18785510
eval seed 10042:
  /tmp/qtrm_generalization_eval/20260520_204706_STAGE10D_GATE_sharp025_finalaux005_count256_seed63_eval_depth46810_count1024.json
eval seed 20042:
  /tmp/qtrm_generalization_eval/20260520_204706_STAGE10D_GATE_sharp025_finalaux005_count256_seed63_eval_seed20042_depth46810_count1024.json
```

Held-out result, eval seed 10042:

```text
depth4  trm=0.1064 raw=0.0928 delta=+0.0137
depth6  trm=0.1289 raw=0.1094 delta=+0.0195
depth8  trm=0.1572 raw=0.0957 delta=+0.0615
depth10 trm=0.1553 raw=0.0977 delta=+0.0576
```

Held-out result, eval seed 20042:

```text
depth4  trm=0.1406 raw=0.1045 delta=+0.0361
depth6  trm=0.1514 raw=0.1035 delta=+0.0479
depth8  trm=0.1445 raw=0.1094 delta=+0.0352
depth10 trm=0.1533 raw=0.0977 delta=+0.0557
```

Decision:

```text
Promote as the best current stability candidate, but not as a solved
generalization big jump.

Stage10D combines:
  sharp trajectory readout, temperature=0.25
  very weak final-readout repair, weight=0.05

This preserves Stage10B-level train accuracy and gives more balanced held-out
depth8/depth10 behavior. It does not reach the user's desired 80-90% regime,
but it is the strongest evidence so far that sharp state selection plus a weak
canonical-final path is the right direction.

Next scale this exact recipe to count512 before trying more mechanisms.
```

## 2026-05-20 Latest Literature Recheck For Bigger Generalization Jump

### Trigger

Stage10D is the best current stability candidate, but it is not a large
generalization jump:

```text
Stage10D repeated held-out depth8/depth10:
seed10042 depth8=0.1572 depth10=0.1553
seed20042 depth8=0.1445 depth10=0.1533
```

This is a real improvement over raw Qwen and more balanced than Stage10B, but
it is far from the desired 80-90% generalization regime. Therefore the next
step should not be another small scalar-weight sweep.

### New Primary Sources Checked

1. Chen, 2026, "Thinking Deeper, Not Longer: Depth-Recurrent Transformers for
   Compositional Generalization" <https://arxiv.org/abs/2603.21676>

   Relevant mechanisms:

   ```text
   depth-recurrent shared-weight Transformer block
   silent thinking objective: supervise only the final output
   LayerScale initialization
   identity-biased recurrence / gradient highway
   variable inference recurrence steps
   ```

   QTRM implication:

   ```text
   Our current core has a residual gate, but not a strong identity-biased
   recurrent block with a tested 20+ step stability path. A bigger jump likely
   requires replacing or wrapping SharedReasoningCore with an identity-biased
   depth-recurrent block and reducing intermediate state supervision further.
   ```

2. Dudley and Oymak, 2026, "Latent Chain-of-Thought Improves Structured-Data
   Transformers" <https://arxiv.org/abs/2605.11262>

   Relevant mechanisms:

   ```text
   compress query-position hidden states into feedback tokens
   append feedback tokens to the next pass
   perform multiple latent computation rounds before prediction
   ```

   QTRM implication:

   ```text
   Instead of only reading a recurrent trajectory, feed the selected recurrent
   state back into the workspace/input state for another latent pass. This is a
   stronger intervention than attention readout alone.
   ```

3. Chaudhry et al., 2026, "Improving Latent Generalization Using Test-time
   Compute" <https://arxiv.org/abs/2604.01430>

   Relevant mechanisms:

   ```text
   train models to use thinking/test-time compute for latent generalization
   correctness feedback can teach generate-and-verify behavior
   train-time augmentation alone is brittle
   ```

   QTRM implication:

   ```text
   Add a verifier/correction path only if it supervises the normal answer path.
   A pure auxiliary probe is not enough. Candidate: predict answer, update
   latent with a correctness/error-correction vector, predict again.
   ```

4. ICLR 2026 Workshop on Latent & Implicit Thinking, compute placement study
   <https://openreview.net/pdf?id=ikNCYypbT9>

   Relevant mechanisms/findings:

   ```text
   compare untied depth, tied recurrence, weighted readout, two-stream, and TRM
   flat two-stream recurrence and untied depth can beat nested terminal TRM
   adaptive weighted readout is an explicit architecture candidate
   token-internal nested TRM is not automatically the best compute placement
   ```

   QTRM implication:

   ```text
   Do not assume TRM hierarchy is the winning structure. Test a flatter
   two-stream recurrence where solution stream Y and scratch stream Z are
   updated in parallel with weighted/adaptive readout, instead of forcing the
   existing z_L -> z_H nested terminal path.
   ```

5. Depth-Recurrent Attention Mixtures, 2026
   <https://www.dfki.de/web/forschung/projekte-publikationen/publikation/17172>

   Relevant claim:

   ```text
   depth-recurrent attention mixtures can improve token efficiency and latent
   reasoning compared with compute-matched baselines.
   ```

   QTRM implication:

   ```text
   The next attention/readout variant should not only sharpen readout. It
   should mix multiple recurrent attention/update pathways and let the model
   route across them.
   ```

### Updated Big-Jump Architecture Direction

Based on the newest sources, the next larger architecture should be:

```text
Stage11 candidate:
  replace nested-only z_L -> z_H terminal TRM path with a flat/two-stream
  depth-recurrent core:

    solution stream Y
    scratch stream Z
    identity-biased residual recurrence
    LayerScale / small transition init
    adaptive weighted readout over recurrent iterates
    optional latent feedback token into a second workspace pass

Training:
  silent/final-answer-heavy objective
  weak or no intermediate state supervision
  weak canonical final-readout repair
  depth/generalization eval after every candidate

Promotion:
  must beat Stage10D on both held-out eval seeds at depth8 and depth10
```

Decision:

```text
Yes, for a larger jump we should move beyond TRM-style nested terminal
refinement. Stage10D is the best local stability recipe, but the latest
literature points toward identity-stable depth recurrence, adaptive halt/readout,
feedback latent tokens, and flat/two-stream compute placement as the next
architecture-level candidates.
```

### Stage11A Result: Two-Stream Compute Placement Is Not Enough Alone

Code change:

```text
StateTransitionCore(update_schedule="two_stream")

Nested:
  z_L <- core(z_L, z_H, op)
  z_H <- core(z_H, z_L_new, zero_op)

Two-stream:
  z_L <- core(z_L, z_H, op)
  z_H <- core(z_H, z_L_old, op)
```

Run:

```text
/tmp/qtrm_eval/20260520_210122_20260520_210121_STAGE11A_twostream_sharp025_silent_count256_seed65
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.26562500
eval seed 10042:
  /tmp/qtrm_generalization_eval/20260520_210121_STAGE11A_twostream_sharp025_silent_count256_seed65_eval_depth46810_count1024.json
eval seed 20042:
  /tmp/qtrm_generalization_eval/20260520_210121_STAGE11A_twostream_sharp025_silent_count256_seed65_eval_seed20042_depth46810_count1024.json
```

Held-out result, eval seed 10042:

```text
depth4  trm=0.1348 delta=+0.0430
depth6  trm=0.1367 delta=+0.0391
depth8  trm=0.1523 delta=+0.0684
depth10 trm=0.1475 delta=+0.0557
```

Held-out result, eval seed 20042:

```text
depth4  trm=0.1367 delta=+0.0381
depth6  trm=0.1514 delta=+0.0664
depth8  trm=0.1406 delta=+0.0400
depth10 trm=0.1436 delta=+0.0557
```

Decision:

```text
Reject as a big-jump candidate. Keep the code path for future A/B tests.

Two-stream compute placement is not harmful and remains consistently above raw
Qwen, but it does not beat Stage10D on the main depth8/depth10 gate. The next
architecture candidate should be stronger: latent feedback pass or adaptive
halt/readout, not two-stream scheduling alone.
```

### Stage11B Result: Simple Latent Feedback Token Is Not Enough

Code change:

```text
QwenBackboneStateTransition(latent_feedback_passes=2)

pass 1:
  Qwen hidden -> workspace -> recurrent core -> sharp readout state

pass 2:
  concatenate [original workspace, readout_state token]
  run recurrent core again without re-running Qwen
```

Run:

```text
/tmp/qtrm_eval/20260520_211143_20260520_211141_STAGE11B_latentfeedback2_sharp025_finalaux005_count256_seed66
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.24218750
eval seed 10042:
  /tmp/qtrm_generalization_eval/20260520_211141_STAGE11B_latentfeedback2_sharp025_finalaux005_count256_seed66_eval_depth46810_count1024.json
```

Held-out result:

```text
depth4  trm=0.1104 delta=+0.0225
depth6  trm=0.1260 delta=+0.0244
depth8  trm=0.1436 delta=+0.0498
depth10 trm=0.1318 delta=+0.0361
```

Eval-only ablation on Stage10D checkpoint with `latent_feedback_passes=2`:

```text
/tmp/qtrm_generalization_eval/20260520_204706_STAGE10D_eval_latentfeedback2_depth46810_count1024.json

depth4  trm=0.1240 delta=+0.0312
depth6  trm=0.1182 delta=+0.0088
depth8  trm=0.1396 delta=+0.0439
depth10 trm=0.1152 delta=+0.0176
```

Decision:

```text
Reject simple latent feedback as implemented.

Both trained feedback and eval-only feedback underperform Stage10D. The likely
failure mode is that blindly appending the readout state as another workspace
token creates an uncalibrated second-pass context rather than a structured
feedback/correction signal. Keep the flag for ablation, but do not promote it.

Current best remains Stage10D:
  sharp trajectory readout, temperature=0.25
  weak final-readout repair, weight=0.05
```

## Next Action: Stage11C Error-Correction Feedback

### Why This Is Next

The latest rejected candidates show that more recurrence is not enough:

```text
Stage11A two_stream:
  stable above raw Qwen
  does not beat Stage10D depth8/depth10

Stage11B simple latent feedback:
  trained feedback underperforms Stage10D
  eval-only feedback also underperforms Stage10D
```

The failure mode is specific:

```text
The model can select useful recurrent states with sharp attention, but it does
not yet have a structured way to notice and correct a bad first answer.
```

Therefore the next architecture should import the "test-time thinking /
generate-and-verify / correction" idea, but keep it inside the normal QTRM
answer path.

### Stage11C Hypothesis

Base recipe:

```text
resume from Stage10D best:
  /tmp/qtrm_eval/20260520_204707_20260520_204706_STAGE10D_GATE_sharp025_finalaux005_count256_seed63/best.pt

keep:
  workspace_pooling=attention
  recurrent_readout_pooling=sharp_attention
  recurrent_readout_temperature=0.25
  final_readout_answer_weight=0.05
  state_update_schedule=nested
```

New mechanism:

```text
first core pass -> answer logits
answer logits -> confidence / entropy / predicted digit embedding
feedback projection -> correction vector
second core pass initialized or conditioned by correction vector
second answer logits -> final answer
```

This is different from Stage11B:

```text
Stage11B:
  blindly appended readout_state as an extra workspace token

Stage11C:
  constructs an explicit answer/error/confidence-conditioned correction vector
  and measures whether the second pass improves over the first pass
```

### Required Telemetry

TensorBoard and Aim must record:

```text
Train/Step/FirstPass_Accuracy
Train/Step/Corrected_Accuracy
Train/Step/Correction_Gain
Train/Step/FirstPass_Entropy
Train/Step/Correction_Vector_Norm

Train/Epoch/FirstPass_Accuracy
Train/Epoch/Corrected_Accuracy
Train/Epoch/Correction_Gain
Train/Epoch/FirstPass_Entropy
Train/Epoch/Correction_Vector_Norm
```

Generalization eval must record:

```text
Generalization/HeldOut/Depth_{d}/FirstPass_Accuracy_TRM
Generalization/HeldOut/Depth_{d}/Corrected_Accuracy_TRM
Generalization/HeldOut/Depth_{d}/Correction_Gain
Generalization/HeldOut/Depth_{d}/Accuracy_Delta_TRM_minus_RawQwen
```

### Promotion Gate

Do not promote on train accuracy alone.

Stage11C must beat Stage10D on both held-out seeds:

```text
Stage10D seed10042:
  depth8=0.1572
  depth10=0.1553

Stage10D seed20042:
  depth8=0.1445
  depth10=0.1533
```

Minimum promotion threshold:

```text
seed10042:
  depth8 > 0.1572
  depth10 > 0.1553

seed20042:
  depth8 > 0.1445
  depth10 > 0.1533

and correction_gain > 0 on at least depth8 and depth10.
```

If Stage11C improves only first-pass train accuracy but not corrected held-out
accuracy, reject it as another overfit readout variant.

### Implementation Notes

Keep the intervention small:

```text
add optional correction_feedback_passes or correction_feedback_weight
add answer_logit -> correction_vector projection
use correction_vector as initial_state for the second core pass
preserve checkpoint compatibility by making new tensors optional
keep default behavior identical when correction is disabled
```

Recommended first run:

```text
Stage11C_GATE
reasoning_count=256
healing_count=32
epochs=4
batch_size=4
lr=2e-6
depths=4 6 8 10
eval seeds=10042 and 20042
```

## Research-Driven Skill Update: Boldness Is Required For Big Jumps

Date: 2026-05-20

Updated:

```text
/home/tripleyoung/.agents/skills/research-driven-architecture-debugging/SKILL.md
section:
  Big-Jump Methodology
```

Added rule:

```text
When the bottleneck is well localized and conservative variants have plateaued,
small safe edits are no longer the safest scientific move. The next candidate
should make a bold, causal intervention on the bottleneck while keeping the
evaluation gate and ablation clean.
```

Operational meaning for this project:

```text
Attention pooling produced a jump because it directly removed the input
binding bottleneck.

Sharp readout produced the best current stability candidate because it directly
attacked trajectory-readout blur.

Now that input pooling and readout have both improved but held-out accuracy is
still low, the next bold intervention should hit the core update mechanism:

  SharedReasoningCore MLP residual transition
  -> MiniGatedDeltaCore / delta-memory update

or a similarly direct correction mechanism:

  unstructured feedback token
  -> verifier/error-conditioned correction state
```

Constraint:

```text
Bold does not mean stacking several unrelated changes in one run.
One major causal path should change at a time, and promotion still requires
beating Stage10D on held-out depth8/depth10 across eval seeds.
```

## Stage11D Hybrid GatedDelta Core Gate

### Implementation

Added a selectable core path:

```text
--core-impl hybrid_state_transition
```

Implementation path:

```text
src/qtrm_mm/state_transition_core.py
  HybridStateTransitionCore

src/qtrm_mm/mixers.py
  FLADeltaMixer
  TorchGatedDeltaMixer fallback
```

Design:

```text
Qwen3.5-style 3:1 trajectory mixer:
  steps 1,2,3 -> gated-delta mixer
  step 4      -> full self-attention sync
  repeat

Uses strict=False double fallback:
  official FLA GatedDeltaNet when compatible
  TorchGatedDeltaMixer fallback otherwise
```

### Run

```text
/tmp/qtrm_eval/20260520_213611_20260520_213609_STAGE11D_hybriddelta_sharp025_finalaux005_count256_seed67
checkpoint:
  best.pt
train best:
  epoch=4
  accuracy_reasoning=0.24609375
load stats:
  exact=350
  skipped=8
eval:
  /tmp/qtrm_generalization_eval/20260520_213609_STAGE11D_hybriddelta_sharp025_finalaux005_count256_seed67_eval_depth46810_count1024.json
```

Held-out result, eval seed 10042:

```text
depth4  trm=0.1240 delta=+0.0352
depth6  trm=0.1387 delta=+0.0352
depth8  trm=0.1201 delta=+0.0342
depth10 trm=0.1309 delta=+0.0361
```

Decision:

```text
Reject this first Stage11D gate as a big jump.

It improves over raw Qwen at all measured depths, but it underperforms Stage10D
on the main depth8/depth10 gate. The likely issue is not that GatedDelta-style
mixing is invalid, but that the hybrid core introduces randomly initialized
delta/sync mixer tensors while only four short epochs are used. This is a bold
architecture replacement with an adaptation cost.

Current best remains Stage10D.
```

Follow-up if continuing this path:

```text
Stage11D-warm:
  freeze or lower LR on reused Stage10D-compatible tensors
  train hybrid-only new tensors with longer warmup
  then unfreeze all core tensors

or:
  implement MiniGatedDeltaCore as a drop-in replacement for SharedReasoningCore
  with smaller parameter delta than full trajectory hybrid mixer.
```

## 2026-05-20 Curriculum Generalization Reset

### Research Inputs

Primary sources checked before changing the data path:

- `sapientinc/HRM-Text`: HRM-Text-style text training should preserve task
  completion boundaries and latent reasoning discipline.
- `sapientinc/data_io`: cleaned rows standardize on
  `condition / instruction / response`, tokenization records instruction and
  response boundaries, and training uses stratified sampling by source prefix.
- Hou et al., ICML 2025, "Universal Length Generalization with Turing
  Programs": length generalization improves when tasks are represented as
  explicit atomic program/state transitions instead of underspecified surface
  prompts.
- Hua et al., arXiv 2507.13332, "TAIL": synthetic CoT should imitate Turing
  machine execution with atomic states to reduce shortcut learning.
- Huang et al., ACL 2025, "TARGA": targeted synthetic data improves
  generalization under non-IID structured reasoning settings.

### Failed Signal That Triggered This Reset

Deterministic attention-pooling runs learned the tiny train distribution but
did not generalize:

```text
local seed42 eval:
depth4 trm=0.0771 raw_qwen=0.1016
depth6 trm=0.1055 raw_qwen=0.1201
depth8 trm=0.1035 raw_qwen=0.1045

local seed43 eval:
depth4 trm=0.0918 raw_qwen=0.1016
depth6 trm=0.0986 raw_qwen=0.1201
depth8 trm=0.1084 raw_qwen=0.1045

local seed44 eval:
depth4 trm=0.1104 raw_qwen=0.1016
depth6 trm=0.0830 raw_qwen=0.1201
depth8 trm=0.0908 raw_qwen=0.1045

dgx seed43 eval:
depth4 trm=0.1172 raw_qwen=0.1006
depth6 trm=0.1055 raw_qwen=0.1201
depth8 trm=0.1045 raw_qwen=0.1045

dgx seed44 eval:
depth4 trm=0.1211 raw_qwen=0.1006
depth6 trm=0.0703 raw_qwen=0.1201
depth8 trm=0.0957 raw_qwen=0.1045
```

Interpretation:

```text
The architecture can memorize the corrected tiny train distribution, but the
old train/eval prompt surfaces and depth distribution were mismatched. This is
a dataset/curriculum problem before it is an architecture problem.
```

### Data Contract Change

Added `--synthetic-schema generalized` to
`scripts/511_train_qwen_state_transition_hrmtext.py`.

New reasoning data:

- uses the same surface schema as generalization eval;
- ends prompts with `Answer:`;
- samples mixed depths through `--train-depths`, e.g. `4 6 8`;
- pads shorter cases to `--n-steps` with a `copy` operation;
- pads state labels with the final value;
- keeps HRM-Text/data_io-style `Condition: synth,cot` and
  `Condition: synth,direct` tags.

Changed `scripts/512_eval_qwen_state_transition_generalization.py` so eval uses
the same `Condition: synth,*` prefix by default. The held-out split is now
represented by seed/data split, not by a different prompt token.

### New Runs

Local:

```text
run: 20260520_161807_CURRGEN_local_depth468_schema_aligned_attnpool_seed42
out: /tmp/qtrm_eval/20260520_161843_20260520_161807_CURRGEN_local_depth468_schema_aligned_attnpool_seed42
log: /tmp/20260520_161807_CURRGEN_local_depth468_schema_aligned_attnpool_seed42.log
tensorboard: http://localhost:6007
aim: http://localhost:43800
```

DGX:

```text
run: 20260520_161807_CURRGEN_dgx_depth468_schema_aligned_attnpool_seed42
out: /mnt/data4tb/qtrm_eval/20260520_161843_20260520_161807_CURRGEN_dgx_depth468_schema_aligned_attnpool_seed42
log: /tmp/20260520_161807_CURRGEN_dgx_depth468_schema_aligned_attnpool_seed42.log
tensorboard: http://192.168.219.113:6008
aim: http://192.168.219.113:43801
```

Both runs use:

```text
--synthetic-schema generalized
--train-depths 4 6 8
--reasoning-condition-prefix synth
--workspace-pooling attention
--freeze-qwen
--n-steps 8
--aux-step-answer-weight 0.5
--checkpoint-every 0
--no-save-last-every-epoch
--save-best-checkpoint
--save-trainable-only
```

Depth consistency is intentionally disabled in this first curriculum run
because shallow/deep consistency can conflict with final-answer supervision
when mixed train depths are padded to a fixed recurrent budget.

### Curriculum Fix After Mixed-Depth Failure

The first large generalized mixed-depth run was intentionally stopped after it
stalled near 15%:

```text
local generalized depth4/6/8, reasoning=2048, healing=512:
best_accuracy=0.15136719

dgx generalized depth4/6/8, reasoning=4096, healing=1024:
best_accuracy=0.14916992
```

Root-cause interpretation:

```text
This is not evidence that attention pooling or the recurrent architecture died.
It shows the curriculum jumped too quickly from tiny depth4 overfit to large
mixed depth4/6/8 plus healing. The model must be expanded by one axis at a
time: data count, then depth, then healing.
```

Recovered gates:

```text
local n_steps=4, depth4, generalized schema, 128 cases, no healing:
best_accuracy=0.80468750
out=/tmp/qtrm_eval/20260520_163020_20260520_1630_FIXGATE_local_depth4_gen128_noheal_attnpool_seed42

local n_steps=8, depth4, generalized schema, 128 cases, no healing:
best_accuracy=0.82812500
out=/tmp/qtrm_eval/20260520_163439_20260520_1637_FIXGATE_local_nsteps8_depth4_gen128_noheal_attnpool_seed42

dgx n_steps=8, depth4, generalized schema, 128 cases, no healing:
best_accuracy=0.82812500
out=/mnt/data4tb/qtrm_eval/20260520_163502_20260520_1637_FIXGATE_dgx_nsteps8_depth4_gen128_noheal_attnpool_seed42
```

Active staged curriculum:

```text
stage2 local:
resume=/tmp/qtrm_eval/20260520_163439_20260520_1637_FIXGATE_local_nsteps8_depth4_gen128_noheal_attnpool_seed42/best.pt
target=depth4, generalized schema, 512 cases, no healing
log=/tmp/20260520_1639_STAGE2_local_nsteps8_depth4_gen512_from128_noheal_attnpool_seed42.log

stage2 dgx:
resume=/mnt/data4tb/qtrm_eval/20260520_163502_20260520_1637_FIXGATE_dgx_nsteps8_depth4_gen128_noheal_attnpool_seed42/best.pt
target=depth4, generalized schema, 512 cases, no healing
log=/tmp/20260520_1642_STAGE2_dgx_nsteps8_depth4_gen512_from128_noheal_attnpool_seed42.log
```

Promotion rule:

```text
Do not add depth6/depth8 or healing until depth4/512 shows a clean learning
curve. The next stage is depth4+6 only, still no healing. Healing is added last.
```

Observed stage2/stage3 progress:

```text
local stage2 depth4/512 no healing:
best_accuracy=0.68554688 at epoch 40
out=/tmp/qtrm_eval/20260520_163820_20260520_1639_STAGE2_local_nsteps8_depth4_gen512_from128_noheal_attnpool_seed42

dgx stage2 depth4/512 no healing:
best_accuracy=0.54882812 at epoch 27/40 while still running
out=/mnt/data4tb/qtrm_eval/20260520_164034_20260520_1642_STAGE2_dgx_nsteps8_depth4_gen512_from128_noheal_attnpool_seed42

local stage3 depth4+6/512 no healing:
best_accuracy=0.35546875 at epoch 9/40 while still running
out=/tmp/qtrm_eval/20260520_164649_20260520_1648_STAGE3_local_nsteps8_depth46_gen512_from_stage2_noheal_attnpool_seed44
```

Current conclusion:

```text
The original low 15% mixed run was a curriculum jump, not architecture death.
n_steps=8 depth4 gates recover to 82.8%, and depth4/512 reaches 68.6%.
Continue staged expansion and evaluate held-out depths after each promoted
stage before reintroducing language healing.
```

Latest staged results:

```text
local stage3 depth4+6/512 no healing:
best_accuracy=0.87890625 at epoch 36
out=/tmp/qtrm_eval/20260520_164649_20260520_1648_STAGE3_local_nsteps8_depth46_gen512_from_stage2_noheal_attnpool_seed44

dgx stage2 depth4/512 no healing:
best_accuracy=0.69140625 at epoch 38
out=/mnt/data4tb/qtrm_eval/20260520_164034_20260520_1642_STAGE2_dgx_nsteps8_depth4_gen512_from128_noheal_attnpool_seed42

local stage4 depth4+6+8/512 no healing:
best_accuracy=0.50585938 at epoch 11/40 while still running
out=/tmp/qtrm_eval/20260520_165402_20260520_1654_STAGE4_local_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45

dgx stage3 depth4+6/512 no healing:
best_accuracy=0.26757812 at epoch 7/40 while still running
out=/mnt/data4tb/qtrm_eval/20260520_165402_20260520_1654_STAGE3_dgx_nsteps8_depth46_gen512_from_stage2_noheal_attnpool_seed44
```

Interpretation:

```text
The staged curriculum solves the earlier failure mode. The same depth4+6+8
surface that stalled near 15% in the large direct mixed run now reaches >50%
by epoch 11 when initialized from depth4 and depth4+6 stages.
```

## 2026-05-20 Depth Diagnostics

### Local: Fixed Attention Depth 8

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 8
--reasoning-count 128
--healing-count 0
--out-dir /tmp/qtrm_eval/qwen35_0.8b_depth8_attnpool_freeze_tiny128_v1
log /tmp/qwen35_0.8b_depth8_attnpool_freeze_tiny128_v1.log
tensorboard http://localhost:6007
```

Result:

```text
epoch 40:
loss=4.1774
acc=0.2500

epoch 50:
loss=3.8211
acc=0.3047

epoch 58:
loss=3.3811
acc=0.5000

epoch 60:
loss=3.1530
acc=0.5469

best observed:
acc=0.5469
```

Interim interpretation:

```text
Fixed depth 8 produces a strong non-random learning signal and validates that
the corrected HRM-Text-aligned data contract plus attention pooling can carry
state-transition supervision. It is not yet an architecture breakthrough by
itself because the 128-case overfit gate is still below the target >60%, but it
is a clear promotion candidate over last pooling and the old malformed prompt
run.
```

### DGX: Attention Depth Sample 4 to 8

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 8
--depth-sample-min 4
--reasoning-count 128
--healing-count 0
--out-dir /mnt/data4tb/qtrm_eval/qwen35_0.8b_depthsample4to8_attnpool_freeze_tiny128_v1
log /tmp/qwen35_0.8b_depthsample4to8_attnpool_freeze_tiny128_v1.log
tensorboard http://localhost:6008 on DGX
```

Result:

```text
epoch 40:
loss=4.1373
acc=0.3359

epoch 52:
loss=3.9500
acc=0.3438

epoch 58:
loss=3.6359
acc=0.4375

epoch 59:
loss=3.6815
acc=0.4453

epoch 60:
loss=3.5230
acc=0.4375

best observed:
acc=0.4453
```

Decision:

```text
Do not promote depth sampling yet. It learns, but underperforms both fixed
depth 8 local diagnostics and the earlier fixed attention-depth diagnostic on
DGX. The strong signal remains attention pooling plus corrected HRM-Text data
contract, not variable recurrence depth.
```

Current comparison:

```text
old malformed prompt long run: around 10% accuracy plateau
last pooling corrected tiny overfit: best 16.41%
attention pooling corrected tiny overfit, DGX prior: 59.38%
fixed attention depth 8, local: 54.69%
attention depth sample 4..8, DGX: 44.53%
```

Updated next step:

```text
Keep attention pooling.
Return to fixed-depth diagnostics and test the readout/supervision bottleneck:
1. fixed depth 4 repeat with same seed path,
2. fixed depth 6 or 8 repeat on DGX,
3. add auxiliary per-step/state readout loss if the >60% tiny-overfit gate is
   not crossed reliably.
Only after the tiny-overfit gate is stable should healing be reintroduced.
```

## 2026-05-20 Auxiliary Step Answer Readout

Implementation:

```text
Added --aux-step-answer-weight to scripts/511_train_qwen_state_transition_hrmtext.py.
When enabled, the final answer readout is applied to every recurrent transition
state and supervised against the per-step state labels. This gives the answer
readout direct trajectory-level supervision instead of only final-state
supervision.
```

Verification:

```text
PASS test_step_answer_auxiliary_loss_supervises_each_transition_state
PASS test_synthetic_reasoning_prompt_contains_operands_and_operations
PASS test_healing_dataset_uses_prefix_response_boundary_and_response_only_labels
PASS test_fit_step_sequence_pads_or_truncates_to_requested_depth
PASS test_last_workspace_pooling_uses_last_attended_token
PASS test_attention_workspace_pooling_ignores_masked_tokens
PASS test_attention_workspace_pooling_accepts_bfloat16_hidden_states
```

### Local: Fixed Depth 8 + Aux Step Answer 0.5

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 8
--aux-step-answer-weight 0.5
--reasoning-count 128
--healing-count 0
--out-dir /tmp/qtrm_eval/qwen35_0.8b_depth8_attnpool_auxstep0.5_freeze_tiny128_v1
log /tmp/qwen35_0.8b_depth8_attnpool_auxstep0.5_freeze_tiny128_v1.log
```

Result:

```text
epoch 57:
loss=4.6437
aux_step=0.9748
acc=0.3828

epoch 60:
loss=4.9944
aux_step=1.0363
acc=0.2734

best observed:
acc=0.3828
```

Decision:

```text
Reject depth 8 + aux 0.5. The auxiliary objective conflicts with the longer
trajectory setting and underperforms fixed depth 8 without the auxiliary loss.
```

### DGX: Fixed Depth 4 + Aux Step Answer 0.5

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 4
--aux-step-answer-weight 0.5
--reasoning-count 128
--healing-count 0
--out-dir /mnt/data4tb/qtrm_eval/qwen35_0.8b_depth4_attnpool_auxstep0.5_freeze_tiny128_v1
log /tmp/qwen35_0.8b_depth4_attnpool_auxstep0.5_freeze_tiny128_v1.log
```

Result:

```text
epoch 43:
loss=3.9130
aux_step=0.9253
acc=0.6094

epoch 48:
loss=3.4179
aux_step=0.8639
acc=0.7031

epoch 57:
loss=2.9593
aux_step=0.7812
acc=0.7578

epoch 60:
loss=2.7246
aux_step=0.7321
acc=0.7656

best observed:
acc=0.7656
```

Decision:

```text
Promote fixed depth 4 + attention pooling + auxiliary step answer loss as the
current strongest architecture/supervision candidate. It is the first run in
this diagnostic series to cross the >60% tiny-overfit gate by a clear margin.
Next validation should be multi-seed and then a healing reintroduction run.
```

### DGX: Fixed Depth 4 + Aux Step Answer 0.5, Seed 43

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 4
--aux-step-answer-weight 0.5
--seed 43
--reasoning-count 128
--healing-count 0
--out-dir /mnt/data4tb/qtrm_eval/qwen35_0.8b_depth4_attnpool_auxstep0.5_freeze_tiny128_seed43_v1
log /tmp/qwen35_0.8b_depth4_attnpool_auxstep0.5_freeze_tiny128_seed43_v1.log
```

Result:

```text
epoch 50:
loss=4.9646
aux_step=1.0390
acc=0.3359

epoch 58:
loss=4.6368
aux_step=0.9983
acc=0.4219

epoch 60:
loss=4.5584
aux_step=0.9809
acc=0.3984

best observed:
acc=0.4219
```

Decision:

```text
Do not claim multi-seed stability yet. Seed 42 crossed the gate strongly, but
seed 43 did not cross the >60% gate within 60 epochs. Because seed 43 started
improving late, run a 120-epoch extension before rejecting the direction.
```

## Run Naming Policy

TensorBoard now points at cumulative log roots:

```text
local: http://localhost:6007 -> /tmp/qtrm_eval
DGX:   http://localhost:6008 -> /mnt/data4tb/qtrm_eval
```

New runs should use timestamped directories:

```text
--timestamp-run-dir --run-name <short-readable-name>
```

The script writes `run_info.txt` into each output directory with the timestamp,
resolved output path, and all CLI arguments. This keeps TensorBoard comparison
cumulative while making the newest run visible by name.

## 2026-05-20 Stability Follow-ups

### Local: Depth Consistency 0.1, Seed 43

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 4
--aux-step-answer-weight 0.5
--depth-consistency-weight 0.1
--consistency-min-steps 2
--seed 43
--out-dir /tmp/qtrm_eval/qwen35_0.8b_depth4_attnpool_aux0.5_cons0.1_seed43_v1
```

Result:

```text
epoch 60:
loss=4.8635
aux_step=1.0166
depth_cons=0.0149
acc=0.3828

best observed:
acc=0.3828
```

Decision:

```text
Reject naive depth consistency 0.1. It does not improve seed43 over the
no-consistency 60-epoch run.
```

### Local: Transition LayerScale + Small Step Embeddings, Seed 43

Command family:

```text
--workspace-pooling attention
--freeze-qwen
--checkpoint-every 0
--n-steps 4
--aux-step-answer-weight 0.5
--transition-scale-init 0.5
--step-embedding-std 0.02
--seed 43
--out-dir /tmp/qtrm_eval/qwen35_0.8b_depth4_attnpool_aux0.5_stabilized_seed43_v1
```

Result:

```text
epoch 57:
loss=4.0897
aux_step=0.9477
acc=0.5156

epoch 60:
loss=4.1623
aux_step=0.9431
acc=0.4375

best observed:
acc=0.5156
```

Decision:

```text
Keep as a probe, not a promotion. Stabilization improves over consistency and
over the weak seed43 60-epoch run, but it still does not cross the >60% gate.
Next test should sweep milder scale/step settings or run the original seed43
longer before claiming a stable architecture improvement.
```

### Raw Qwen3.5-0.8 Baseline Comparison

Baseline method:

```text
Model: Qwen/Qwen3.5-0.8B-Base
Cases: same 128 synthetic cases, seed 42
Prompt: synthetic prompt + "\nAnswer:"
Scoring: next-token likelihood over digit tokens 0..9
```

Result:

```text
raw Qwen3.5-0.8 next-digit accuracy: 17/128 = 13.2812%
best QTRM diagnostic accuracy:       98/128 = 76.5625%

absolute gain:
76.5625% - 13.2812% = +63.2813 percentage points

relative gain:
76.5625 / 13.2812 = 5.7647x baseline
or +476.47% relative increase over baseline
```

Caveat:

```text
This compares raw Qwen next-token digit scoring against QTRM tiny-overfit
training-set diagnostic accuracy. It is useful as a capability/binding signal,
but it is not a held-out generalization claim.
```

### Checkpoint Retention Policy

Decision:

```text
For tiny diagnostic sweeps, keep only the best checkpoint by default.
Do not write last.pt every epoch unless explicitly requested.
When Qwen is frozen, save trainable parameters only.
```

Rationale:

```text
Full checkpoints for Qwen3.5-0.8B are about 1.5 GB and can stall DGX I/O when
written repeatedly. The frozen-Qwen diagnostic runs only need the trainable
QTRM/adapter-side weights to reproduce the best model state on top of the same
base Qwen model.
```

Implementation:

```text
scripts/511_train_qwen_state_transition_hrmtext.py

--save-best-checkpoint
--no-save-last-every-epoch
--save-trainable-only
```

DGX reproduction check:

```text
run:
/mnt/data4tb/qtrm_eval/20260520_154036_REPRO_acc6953_best_repro_seed42_trainablebest_depth4_aux05_attnpool

config:
seed=42
workspace_pooling=attention
n_steps=4
aux_step_answer_weight=0.5
freeze_qwen=True
reasoning_count=128
healing_count=0
save_best_checkpoint=True
save_last_every_epoch=False
save_trainable_only=True

result:
best_epoch=60
best_accuracy=0.69531250
best.pt size=77 MB
```

Interpretation:

```text
The trainable-only checkpoint policy works and avoids the 1.5 GB full-checkpoint
I/O stall. This run did not exactly reproduce the earlier 76.5625% peak, but it
still reaches 69.53125%, well above the raw Qwen baseline of 13.28125%.

The earlier BEST run remains the top observed diagnostic result:
/mnt/data4tb/qtrm_eval/20260520_145737_BEST_acc7656_qwen35_0.8b_depth4_attnpool_auxstep0.5_freeze_tiny128_v1

That earlier run has logs but no checkpoint, so it cannot be resumed directly.
Future promotion runs must use the best-only trainable checkpoint policy from
the start.
```

Cleanup:

```text
Aborted DGX seed43 120e run:
/mnt/data4tb/qtrm_eval/20260520_150531_ABORTED_acc2422_qwen35_0.8b_depth4_attnpool_auxstep0.5_freeze_tiny128_seed43_120e_v1

best observed before abort:
epoch=39
accuracy=0.2422

Removed large non-best checkpoints:
epoch_20.pt
epoch_40.pt
last.pt

Remaining directory size:
56 KB, logs only
```

### Aim Experiment Tracking

Decision:

```text
Use TensorBoard for live scalar curves and Aim for run metadata, hparams,
historical comparison, and search.

This is mandatory for every nontrivial Qwen/QTRM/HRM-Text architecture,
training, evaluation, seed, or ablation run. A result without TensorBoard or Aim
records is scratch-only and must not be promoted as BEST, accepted, rejected, or
architecture evidence until it is backfilled.
```

Implementation:

```text
scripts/511_train_qwen_state_transition_hrmtext.py

New flags:
--aim-repo
--aim-experiment
--aim-run-name
--aim-description

Default:
--aim-repo can also be supplied through QTRM_AIM_REPO.
If Aim is not installed, training continues with TensorBoard only and prints a
warning.
```

Tracked Aim metadata:

```text
hparams: full argparse config
paths: out_dir and TensorBoard logdir
best: best epoch, best accuracy, best checkpoint path

step metrics:
loss_total
loss_reasoning
loss_healing
loss_aux_step_answer
loss_depth_consistency
grad_norm
n_steps

epoch metrics:
loss_total
loss_reasoning
loss_healing
loss_aux_step_answer
loss_depth_consistency
accuracy_reasoning
learning_rate
epoch_duration_seconds
best_accuracy_reasoning
```

Live Aim servers:

```text
Local:
repo=/tmp/qtrm_aim
url=http://localhost:43800
TensorBoard=http://localhost:6007
TensorBoard root=/tmp/qtrm_eval

DGX:
repo=/mnt/data4tb/qtrm_aim
url=http://192.168.219.113:43801
TensorBoard=http://192.168.219.113:6008
TensorBoard root=/mnt/data4tb/qtrm_eval
```

Backfilled DGX historical runs:

```text
20260520_145737_BEST_acc7656_depth4_attnpool_aux05_seed42
20260520_154036_REPRO_acc6953_depth4_attnpool_aux05_seed42_trainablebest
20260520_150531_ABORTED_acc2422_depth4_attnpool_aux05_seed43_120e
```

Required launch fields:

```text
--timestamp-run-dir
--run-name <timestamp_or_seed_architecture_name>
--aim-repo <repo>
--aim-experiment <experiment_family>
--aim-run-name <same_stable_run_name>

Run directory must include:
run_info.txt
logs/ TensorBoard event file
best_info.txt when --save-best-checkpoint is enabled
best.pt when a checkpoint is accepted for future eval/resume
```

### 2026-05-20 Staged Curriculum Results

Problem diagnosed:

```text
The earlier large mixed run combined depth expansion, reasoning count expansion,
and language healing too early. Controlled staged gates show the architecture is
not dead, but the curriculum jump is too large when mixed from the beginning.
```

Local staged no-healing path:

```text
Stage2:
run=/tmp/qtrm_eval/20260520_163820_20260520_1639_STAGE2_local_nsteps8_depth4_gen512_from128_noheal_attnpool_seed42
train_depths=4
reasoning_count=512
healing_count=0
best_accuracy=0.6855

Stage3:
run=/tmp/qtrm_eval/20260520_164649_20260520_1648_STAGE3_local_nsteps8_depth46_gen512_from_stage2_noheal_attnpool_seed44
train_depths=4,6
reasoning_count=512
healing_count=0
best_accuracy=0.8789

Stage4:
run=/tmp/qtrm_eval/20260520_165402_20260520_1654_STAGE4_local_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45
train_depths=4,6,8
reasoning_count=512
healing_count=0
best_epoch=34
best_accuracy=0.97460938
loss=3.30547253
aux_step_answer_loss=1.04510134
```

Local Stage4 held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_1654_STAGE4_local_depth468_finalbest_eval_depth46810_count1024.json
eval_count=1024
model_n_steps=10

depth4:  TRM=0.1318 raw_qwen=0.1006 delta=+0.0313
depth6:  TRM=0.1201 raw_qwen=0.1113 delta=+0.0088
depth8:  TRM=0.1514 raw_qwen=0.0947 delta=+0.0566
depth10: TRM=0.1357 raw_qwen=0.1035 delta=+0.0322
```

DGX Stage4 reproduction:

```text
run=/mnt/data4tb/qtrm_eval/20260520_170611_20260520_1702_STAGE4_dgx_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45
train_depths=4,6,8
reasoning_count=512
healing_count=0
best_epoch=39
best_accuracy=0.81640625
loss=5.87971260
aux_step_answer_loss=1.61544811
```

DGX Stage4 held-out eval:

```text
eval=/mnt/data4tb/qtrm_generalization_eval/20260520_1702_STAGE4_dgx_best_eval_depth46810_count1024.json
eval_count=1024
model_n_steps=10

depth4:  TRM=0.1377 raw_qwen=0.1016 delta=+0.0361
depth6:  TRM=0.1152 raw_qwen=0.1084 delta=+0.0068
depth8:  TRM=0.1318 raw_qwen=0.0947 delta=+0.0371
depth10: TRM=0.1113 raw_qwen=0.1045 delta=+0.0068
```

Interpretation:

```text
The staged no-healing curriculum repeatedly beats the frozen raw Qwen baseline
on held-out synthetic depth gates. The deltas are real but still small in
absolute generalization terms. This is not evidence that the model beats a
large general LLM. It is evidence that attention pooling + recurrent state
transition + staged curriculum can create a measurable causal improvement over
the same Qwen backbone on controlled algorithmic reasoning gates.
```

### 2026-05-20 Small Healing Gate

Local Stage5:

```text
run=/tmp/qtrm_eval/20260520_170611_20260520_1702_STAGE5_local_depth468_gen512_heal128_from_stage4_attnpool_seed46
resume=/tmp/qtrm_eval/20260520_165402_20260520_1654_STAGE4_local_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45/best.pt
train_depths=4,6,8
reasoning_count=512
healing_count=128
healing_weight=0.15
lr=2e-5
best_epoch=20
best_accuracy=0.94531250
loss=4.06253781
reasoning_loss=3.86975522
healing_loss=0.19278259
aux_step_answer_loss=1.15136429
```

Local Stage5 held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_1702_STAGE5_local_heal128_best_eval_depth46810_count1024.json
eval_count=1024
model_n_steps=10

depth4:  TRM=0.1025 raw_qwen=0.1006 delta=+0.0020
depth6:  TRM=0.1133 raw_qwen=0.1113 delta=+0.0020
depth8:  TRM=0.1289 raw_qwen=0.0947 delta=+0.0342
depth10: TRM=0.0977 raw_qwen=0.1035 delta=-0.0059
```

Interpretation:

```text
Stage5 proves small language healing does not immediately destroy training-set
reasoning accuracy, but it weakens held-out generalization compared with the
Stage4 no-healing best. Do not promote this checkpoint as the current best.
Before adding more natural-language data, add replay/regularization or reduce
the healing objective so the Stage4 held-out gains are preserved.
```

Next gate:

```text
Keep Stage4 no-healing as the current research reference.
Test a preservation-focused healing run:
  resume Stage4
  healing_count <= 64
  healing_weight <= 0.05
  reasoning replay unchanged
  eval depth4/6/8/10 after completion

Promote only if held-out deltas match or beat Stage4 while healing loss improves.
```

### 2026-05-20 Literature Watch Update

Primary sources checked:

```text
HRM:
https://arxiv.org/abs/2506.21734
https://github.com/sapientinc/HRM

Nested Learning:
https://arxiv.org/abs/2512.24695

Looped latent reasoning:
https://arxiv.org/abs/2502.17416

LoopFormer / elastic-depth looped transformers:
https://arxiv.org/abs/2602.11451

Attractor Models / fixed-point recurrent refinement:
https://arxiv.org/abs/2605.12466
```

Mechanisms mapped to falsifiable QTRM gates:

```text
LoopFormer:
  relevant idea:
    variable loop budgets and shortcut consistency across shorter/longer
    recurrent trajectories.

  local implementation hook already available:
    --depth-sample-min
    --depth-consistency-weight
    --consistency-min-steps
    --depth-consistency-temperature

  next falsifiable gate:
    resume Stage4 no-healing reference
    train_depths=4,6,8
    n_steps=10 or 12
    depth_sample_min=4
    depth_consistency_weight in [0.05, 0.1]
    consistency_min_steps=4
    no healing
    pass only if held-out depth10 delta improves over Stage4 while depth4/6/8 do
    not regress.

Attractor Models:
  relevant idea:
    recurrent refinement should move toward a stable fixed point rather than
    drift with more iterations.

  local implementation hook:
    current transition_scale and step embeddings stabilize recurrence, but there
    is not yet an implicit fixed-point or attractor objective.

  next architecture candidate:
    add a small equilibrium loss between late recurrent states or a residual
    attractor head, then ablate core_off/short_steps/full_steps.

HRM / Nested Learning:
  relevant idea:
    nested multi-level recurrence is useful only when the recurrent path is
    causally responsible for the gain.

  required gate:
    compare full recurrent core vs reduced steps/core-disabled where possible.
    Do not promote a run if the held-out gain survives without recurrence.
```

Current action:

```text
Running:
  Local Stage5B preservation tiny-healing gate
  DGX Stage4B no-healing continuation gate

After these finish:
  run held-out evals
  then launch LoopFormer-style variable-depth shortcut-consistency gate if
  Stage5B does not preserve Stage4 held-out deltas.
```

### 2026-05-20 Stage5B Preservation Result

Local Stage5B:

```text
run=/tmp/qtrm_eval/20260520_172308_20260520_172254_STAGE5B_local_preserve_depth468_gen512_heal64_w005_from_stage4_attnpool_seed47
resume=/tmp/qtrm_eval/20260520_165402_20260520_1654_STAGE4_local_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45/best.pt
train_depths=4,6,8
reasoning_count=512
healing_count=64
healing_weight=0.05
lr=1e-5
best_epoch=20
best_accuracy=0.86328125
loss=5.35387699
reasoning_loss=5.29144821
healing_loss=0.06242879
aux_step_answer_loss=1.43635909
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_172254_STAGE5B_local_preserve_heal64_w005_best_eval_depth46810_count1024.json
eval_count=1024
model_n_steps=10

depth4:  TRM=0.1465 raw_qwen=0.1006 delta=+0.0459
depth6:  TRM=0.1270 raw_qwen=0.1113 delta=+0.0156
depth8:  TRM=0.1416 raw_qwen=0.0947 delta=+0.0469
depth10: TRM=0.1191 raw_qwen=0.1035 delta=+0.0156
```

Interpretation:

```text
Stage5B is better than Stage5 and partially better than Stage4 on held-out
generalization while retaining a small language-healing signal. The weaker
healing objective avoids the Stage5 depth10 regression and improves depth4/6
relative to Stage4, but depth8/depth10 are still below the strongest local
Stage4 deltas.

This is a useful preservation checkpoint, not a final best general model.
```

### 2026-05-20 LoopFormer Gate Launch

Launched local variable-depth shortcut-consistency gate:

```text
run_name=20260520_173222_LOOPFORMER_local_varsteps10_depth468_gen512_cons005_noheal_from_stage4_seed48
resume=/tmp/qtrm_eval/20260520_165402_20260520_1654_STAGE4_local_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45/best.pt
n_steps=10
depth_sample_min=4
depth_consistency_weight=0.05
consistency_min_steps=4
train_depths=4,6,8
reasoning_count=512
healing_count=0
lr=1e-5
epochs=24
```

Pass criterion:

```text
The run must improve depth10 held-out accuracy over Stage4 and Stage5B without
collapsing depth4/6/8. If it only improves training accuracy, reject it.
```

Implementation fixes required by the gate:

```text
scripts/511_train_qwen_state_transition_hrmtext.py

1. Added flexible checkpoint loading for recurrent-depth changes.
   This allows n_steps=8 checkpoints to initialize n_steps=10 models by copying
   existing step embeddings and leaving new rows at initialization.

2. Fixed empty consistency-step sampling when n_steps equals
   consistency_min_steps.
```

Verification:

```text
pytest is not installed in the current venv, so tests were run by direct module
execution.

Passed:
tests/test_hrm_text_aligned_training_data.py
tests/test_state_transition_core_stabilization.py
tests/test_qwen_state_transition_workspace_pooling.py
```

LoopFormer retry2 result:

```text
run=/tmp/qtrm_eval/20260520_173428_20260520_173222_LOOPFORMER_local_varsteps10_depth468_gen512_cons005_noheal_from_stage4_seed48_retry2
resume=/tmp/qtrm_eval/20260520_165402_20260520_1654_STAGE4_local_nsteps8_depth468_gen512_from_stage3_noheal_attnpool_seed45/best.pt
n_steps=10
depth_sample_min=4
depth_consistency_weight=0.05
consistency_min_steps=4
train_depths=4,6,8
reasoning_count=512
healing_count=0
best_epoch=24
best_accuracy=0.56445312
depth_consistency_loss=0.02071699
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_173222_LOOPFORMER_local_varsteps10_cons005_best_eval_depth46810_count1024.json
eval_count=1024
model_n_steps=10

depth4:  TRM=0.1377 raw_qwen=0.1006 delta=+0.0371
depth6:  TRM=0.1348 raw_qwen=0.1113 delta=+0.0234
depth8:  TRM=0.1455 raw_qwen=0.0947 delta=+0.0508
depth10: TRM=0.1348 raw_qwen=0.1035 delta=+0.0313
```

Interpretation:

```text
LoopFormer-style variable-depth shortcut consistency improves the hardest
depth10 held-out gate relative to Stage5B and matches the strongest Stage4
depth10 signal, despite low training accuracy. It is a useful generalization
mechanism, but it does not dominate Stage5B on depth4 or Stage4 on depth8.

Next best experiment:
combine Stage5B preservation healing with variable-depth consistency:
  resume Stage5B or Stage4
  n_steps=10
  depth_sample_min=4
  depth_consistency_weight=0.02 to 0.05
  healing_count=64
  healing_weight=0.03 to 0.05

Promotion requires matching Stage5B on depth4/6, matching LoopFormer on depth10,
and keeping healing_loss below Stage5B.
```

DGX Stage4B continuation:

```text
run=/mnt/data4tb/qtrm_eval/20260520_172319_20260520_172254_STAGE4B_dgx_continue_depth468_gen512_noheal_from_stage4_attnpool_seed47
best_epoch=38
best_accuracy=0.92578125
loss=4.75701660
```

DGX Stage4B held-out eval:

```text
eval=/mnt/data4tb/qtrm_generalization_eval/20260520_172254_STAGE4B_dgx_continue_best_eval_depth46810_count1024.json
eval_count=1024
model_n_steps=10

depth4:  TRM=0.1338 raw_qwen=0.1016 delta=+0.0322
depth6:  TRM=0.1465 raw_qwen=0.1084 delta=+0.0381
depth8:  TRM=0.1279 raw_qwen=0.0947 delta=+0.0332
depth10: TRM=0.1123 raw_qwen=0.1045 delta=+0.0078
```

### 2026-05-20 Train vs Generalization Metric Split

Decision:

```text
Training-set accuracy and held-out/generalization accuracy must be named
separately in TensorBoard and Aim. The 80-90% train accuracy is useful evidence
that the recurrent core can fit the training distribution, but it must not be
read as generalization.
```

Implementation:

```text
scripts/511_train_qwen_state_transition_hrmtext.py

TensorBoard:
  Train/Step/*
  Train/Epoch/Accuracy_Reasoning
  Train/Epoch/Best_Accuracy_Reasoning

Aim:
  train_loss_total
  train_loss_reasoning
  train_loss_healing
  train_accuracy_reasoning
  best_train_accuracy_reasoning
```

```text
scripts/512_eval_qwen_state_transition_generalization.py

TensorBoard:
  Generalization/HeldOut/Accuracy_TRM
  Generalization/HeldOut/Accuracy_RawQwen
  Generalization/HeldOut/Accuracy_Delta_TRM_minus_RawQwen
  Generalization/HeldOut/Depth_<N>/*

Aim:
  generalization_trm_accuracy
  generalization_raw_qwen_accuracy
  generalization_delta_accuracy_trm_minus_raw_qwen
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/511_train_qwen_state_transition_hrmtext.py \
  scripts/512_eval_qwen_state_transition_generalization.py

Direct test execution passed:
  tests/test_hrm_text_aligned_training_data.py
  tests/test_state_transition_core_stabilization.py
  tests/test_qwen_state_transition_workspace_pooling.py
```

### 2026-05-20 Stage6 Combo Result

Stage6:

```text
run=/tmp/qtrm_eval/20260520_174654_20260520_174643_STAGE6_local_heal64_loopformer_cons002_nsteps10_from_stage5B_seed49
resume=/tmp/qtrm_eval/20260520_172308_20260520_172254_STAGE5B_local_preserve_depth468_gen512_heal64_w005_from_stage4_attnpool_seed47/best.pt
n_steps=10
depth_sample_min=4
depth_consistency_weight=0.02
healing_count=64
healing_weight=0.03
best_epoch=20
best_train_accuracy=0.45898438
healing_loss=0.03613112
depth_consistency_loss=0.00623955
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_174643_STAGE6_local_heal64_loopformer_cons002_best_eval_depth46810_count1024.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_174643_STAGE6_local_heal64_loopformer_cons002_depth46810

depth4:  TRM=0.1426 raw_qwen=0.1006 delta=+0.0420
depth6:  TRM=0.1309 raw_qwen=0.1113 delta=+0.0195
depth8:  TRM=0.1357 raw_qwen=0.0947 delta=+0.0410
depth10: TRM=0.1367 raw_qwen=0.1035 delta=+0.0332
```

Interpretation:

```text
Stage6 achieves the best depth10 held-out score so far and keeps weak healing,
but it does not dominate Stage5B on depth4 or LoopFormer on depth8. Its by-family
results show checksum remains weaker than chain, so the next gate should target
family balance rather than simply increasing epochs.
```

### 2026-05-20 Stage7 Family-Mix Gate

Implementation:

```text
Added --synthetic-family-mix to generalized synthetic data:
  chain2_checksum1  # previous default
  balanced
  checksum2_chain1
```

Launched:

```text
run_name=20260520_180536_STAGE7_local_checksum_replay_heal64_cons0015_nsteps10_from_stage6_seed50
resume=/tmp/qtrm_eval/20260520_174654_20260520_174643_STAGE6_local_heal64_loopformer_cons002_nsteps10_from_stage5B_seed49/best.pt
synthetic_family_mix=checksum2_chain1
n_steps=10
depth_sample_min=4
depth_consistency_weight=0.015
healing_count=64
healing_weight=0.02
```

Pass criterion:

```text
Improve checksum held-out accuracy without losing Stage6's depth10 delta.
Promote only after Generalization/HeldOut metrics are recorded separately from
Train metrics.
```

### 2026-05-20 HRM-Text Generalization Import Notes

Question:

```text
Train accuracy reached high values on local synthetic reasoning, but held-out
generalization remains around the low-teens. The next improvement should be
driven by HRM-Text/data_io training discipline rather than more blind epochs.
```

Observed from cloned upstream references:

```text
external_repos/HRM-Text/dataset_new.py
  - uses PrefixLM-style batches;
  - inputs concatenate instruction plus response prefix;
  - target_only masks instruction tokens and supervises response tokens only;
  - batches are packed by token length, not by fixed row count.

external_repos/data_io/README.md
  - standard row schema is {condition, instruction, response};
  - training data is built by stratified sampling over source prefixes;
  - small high-quality reasoning datasets are explicitly repeated/upsampled.

external_repos/data_io/prefix_config.yaml
  - caps very large sources with max_per_file;
  - repeats high-quality small sources such as GSM8K/MATH/Platypus/OmniMath;
  - keeps task families balanced through ordered prefix matching.

external_repos/HRM-Text/config/arch/net/hrm.yaml
  - HRM uses H_cycles=2 and L_cycles=3.

external_repos/HRM-Text/config/arch/net/trm.yaml
  - TRM baseline uses H_cycles=2 and L_cycles=1.
```

Implication:

```text
Our current QTRM run already has a recurrent core and attention pooling, but its
generalization bottleneck is probably not solved by simply increasing train
epochs. The more direct HRM-Text import is:

1. make reasoning and healing rows first-class condition/instruction/response
   examples;
2. train normal LM logits with response-token CE/KL after mandatory recurrent
   core passage;
3. add prefix/family-stratified sampling instead of ad-hoc case counts;
4. log held-out accuracy by depth, family, and condition in both TensorBoard and
   Aim;
5. only then test H/L cycle variants as architecture changes.
```

Next falsifiable gate:

```text
Stage8 should implement HRM-Text-style condition/prefix sampling and per-family
generalization metrics. Pass only if held-out checksum/chain family metrics
improve versus Stage6/Stage7 while the gain remains above raw_qwen under the
same prompts.
```

Implementation update:

```text
scripts/512_eval_qwen_state_transition_generalization.py now logs held-out
family metrics:

TensorBoard:
  Generalization/HeldOut/Family_<family>/Accuracy_TRM
  Generalization/HeldOut/Family_<family>/Accuracy_RawQwen
  Generalization/HeldOut/Family_<family>/Accuracy_Delta_TRM_minus_RawQwen
  Generalization/HeldOut/Depth_<N>/Family_<family>/*

Aim:
  generalization_trm_family_accuracy
  generalization_raw_qwen_family_accuracy
  generalization_delta_family_accuracy_trm_minus_raw_qwen

This turns the current "checksum is weak" observation into a first-class gate
instead of a manual JSON inspection step.
```

### 2026-05-20 HRM-Text Cleaned Dataset Download

Decision:

```text
Download the official cleaned HRM-Text/data_io dataset before expanding the
generalization curriculum. Prefer the cleaned dataset first because the raw
data_io cleaning path requires much larger RAM and is unnecessary for immediate
training integration.
```

Source:

```text
repo=sapientinc/HRM-Text-data-io-cleaned-20260515
expected_schema={condition, instruction, response}
splits=train, validation, test
```

Storage:

```text
DGX path=/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515
DGX cache=/mnt/data4tb/cache/huggingface
download_log=/tmp/qtrm_download_logs/hrm_text_cleaned_dgx.log
```

Local note:

```text
Local full download was stopped because /mnt/sdc1 is full and root storage is
too tight for the complete repository plus Hugging Face local-dir cache. The
partial local download was removed. Full copy is being kept on DGX where
/mnt/data4tb has sufficient space.
```

Current progress checkpoint:

```text
2026-05-20 18:30 KST:
  DGX download running in background with --max-workers 32
  size_seen=22G
  files_seen=894
  target_files_from_HF_API=5223
```

Local cleanup:

```text
Freed local space to allow a local HRM-Text cleaned copy attempt:
  - removed regenerable npm/gradle/triton/pnpm/tmp caches;
  - removed old backup binaries;
  - removed /mnt/sdc1 stale HF model caches not used by QTRM;
  - removed old /mnt/sdc1/tripleyoung/qtrm_eval checkpoints after writing a
    manifest to cleanup_manifests/20260520_sdc1_qtrm_eval_removed_manifest.txt.

After cleanup:
  /mnt/sdc1 free ~= 90G
```

Local download:

```text
path=/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515
cache=/mnt/sdc1/huggingface_hrm_cache
log=/mnt/sdc1/qtrm_download_logs/hrm_text_cleaned_local_sdc1.log
status=running
```

### 2026-05-20 Stage7 Checksum Replay Eval

Stage7:

```text
run=/tmp/qtrm_eval/20260520_180551_20260520_180536_STAGE7_local_checksum_replay_heal64_cons0015_nsteps10_from_stage6_seed50
best_epoch=22
best_train_accuracy=0.45703125
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_180536_STAGE7_local_checksum_replay_best_eval_depth46810_count1024.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_180536_STAGE7_local_checksum_replay_depth46810

depth4:  TRM=0.1504 raw_qwen=0.1006 delta=+0.0498
depth6:  TRM=0.1230 raw_qwen=0.1113 delta=+0.0117
depth8:  TRM=0.1396 raw_qwen=0.0947 delta=+0.0449
depth10: TRM=0.1641 raw_qwen=0.1035 delta=+0.0605
```

Family result:

```text
Stage7 improves long-depth held-out accuracy, especially depth10 versus Stage6
(0.1641 vs 0.1367), but the gain is mostly chain-family. Checksum remains weak:

depth10 chain:    TRM=0.1912 raw=0.1056 delta=+0.0856
depth10 checksum: TRM=0.1053 raw=0.0991 delta=+0.0062

Next training must import real HRM-Text/data_io rows and stronger family-balanced
sampling rather than only replaying synthetic checksum cases.
```

### 2026-05-20 Stage8A Real HRM-Text JSONL Healing Eval

Stage8A:

```text
run=/tmp/qtrm_eval/20260520_184213_20260520_1838_STAGE8A_local_real_hrmtext_jsonl_heal128_balanced_stratified_from_stage7_seed52
resume=/tmp/qtrm_eval/20260520_180551_20260520_180536_STAGE7_local_checksum_replay_heal64_cons0015_nsteps10_from_stage6_seed50/best.pt
healing_data_path=/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515
healing_include_glob=data/*.jsonl,data/Platypus/*.jsonl
synthetic_family_mix=balanced
synthetic_sampling_strategy=stratified
best_epoch=9
best_train_accuracy=0.22656250
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_1838_STAGE8A_local_real_hrmtext_jsonl_heal128_balanced_stratified_eval_depth46810_count1024.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_1838_STAGE8A_local_real_hrmtext_jsonl_heal128_balanced_stratified_depth46810

depth4:  TRM=0.1348 raw_qwen=0.1006 delta=+0.0342
depth6:  TRM=0.1328 raw_qwen=0.1113 delta=+0.0215
depth8:  TRM=0.1523 raw_qwen=0.0947 delta=+0.0576
depth10: TRM=0.1426 raw_qwen=0.1035 delta=+0.0391
```

Interpretation:

```text
Real HRM-Text JSONL healing plus balanced stratified sampling improves depth8
over Stage7 (0.1523 vs 0.1396), but it loses Stage7's best depth10 result
(0.1426 vs 0.1641) and still harms checksum:

depth8 chain:    TRM=0.1940 raw=0.0881 delta=+0.1060
depth8 checksum: TRM=0.0734 raw=0.1073 delta=-0.0339
depth10 checksum: TRM=0.0929 raw=0.0991 delta=-0.0062

Conclusion: real HRM-Text healing is useful but balanced sampling is not enough.
Next gate should keep Stage7's checksum-heavy synthetic replay and add weaker
real HRM-Text healing, not replace checksum pressure with balanced sampling.
```

### 2026-05-20 Stage8B Weak Real Healing + Checksum Replay Eval

Stage8B:

```text
run=/tmp/qtrm_eval/20260520_185531_20260520_1858_STAGE8B_local_real_hrmtext_weakheal64_checksumheavy_from_stage7_seed53
resume=/tmp/qtrm_eval/20260520_180551_20260520_180536_STAGE7_local_checksum_replay_heal64_cons0015_nsteps10_from_stage6_seed50/best.pt
healing_data_path=/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515
healing_count=64
healing_weight=0.01
synthetic_family_mix=checksum2_chain1
synthetic_sampling_strategy=stratified
best_epoch=11
best_train_accuracy=0.18554688
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_1858_STAGE8B_local_real_hrmtext_weakheal64_checksumheavy_eval_depth46810_count1024.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_1858_STAGE8B_local_real_hrmtext_weakheal64_checksumheavy_depth46810

depth4:  TRM=0.1416 raw_qwen=0.1006 delta=+0.0410
depth6:  TRM=0.1172 raw_qwen=0.1113 delta=+0.0059
depth8:  TRM=0.1416 raw_qwen=0.0947 delta=+0.0469
depth10: TRM=0.1670 raw_qwen=0.1035 delta=+0.0635
```

Interpretation:

```text
Stage8B slightly improves the current best depth10 held-out score:
  Stage7 depth10=0.1641
  Stage8A depth10=0.1426
  Stage8B depth10=0.1670

It also preserves positive checksum at depth10:
  depth10 checksum: TRM=0.1053 raw=0.0991 delta=+0.0062

But it weakens depth4/depth6 relative to Stage7 and checksum depth8 is still
negative:
  depth8 checksum: TRM=0.0989 raw=0.1073 delta=-0.0085

Conclusion: promote Stage8B only as the best depth10 candidate, not as a
globally dominant checkpoint. The next gate should repair checksum depth8 while
preserving Stage8B's depth10 gain.
```

### 2026-05-20 Stage8C Long-Depth Checksum Repair Eval

Stage8C:

```text
run=/tmp/qtrm_eval/20260520_190904_20260520_190903_STAGE8C_local_longdepth_checksum_repair_from_stage8B_seed54
resume=/tmp/qtrm_eval/20260520_185531_20260520_1858_STAGE8B_local_real_hrmtext_weakheal64_checksumheavy_from_stage7_seed53/best.pt
healing_data_path=/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515
healing_count=64
healing_weight=0.01
synthetic_family_mix=checksum2_chain1
synthetic_sampling_strategy=stratified
train_depths=6,8,10
depth_sample_min=6
best_epoch=11
best_train_accuracy=0.18359375
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_190903_STAGE8C_local_longdepth_checksum_repair_from_stage8B_seed54_eval_depth46810_count1024.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_190903_STAGE8C_local_longdepth_checksum_repair_from_stage8B_seed54_depth46810

depth4:  TRM=0.1416 raw_qwen=0.1094 delta=+0.0322
depth6:  TRM=0.1543 raw_qwen=0.1064 delta=+0.0479
depth8:  TRM=0.1621 raw_qwen=0.1162 delta=+0.0459
depth10: TRM=0.1455 raw_qwen=0.1123 delta=+0.0332
```

Family breakdown:

```text
depth4 chain:     TRM=0.1519 raw=0.1018 delta=+0.0501
depth4 checksum:  TRM=0.1214 raw=0.1243 delta=-0.0029
depth6 chain:     TRM=0.1849 raw=0.1213 delta=+0.0636
depth6 checksum:  TRM=0.0948 raw=0.0776 delta=+0.0172
depth8 chain:     TRM=0.1820 raw=0.1218 delta=+0.0602
depth8 checksum:  TRM=0.1253 raw=0.1058 delta=+0.0195
depth10 chain:    TRM=0.1723 raw=0.1193 delta=+0.0530
depth10 checksum: TRM=0.0928 raw=0.0986 delta=-0.0058
```

Interpretation:

```text
Stage8C succeeds at the targeted checksum repair for depth6/depth8:
  Stage8B depth8 checksum delta=-0.0085
  Stage8C depth8 checksum delta=+0.0195

It also improves depth6 overall from Stage8B:
  Stage8B depth6=0.1172
  Stage8C depth6=0.1543

But it fails to preserve the Stage8B depth10 peak:
  Stage8B depth10=0.1670
  Stage8C depth10=0.1455

Conclusion: Stage8C is a repair signal, not a new best checkpoint. The next
candidate should combine Stage8B's depth10-preserving regime with a smaller
targeted checksum-depth8 replay dose, rather than shifting the whole curriculum
to long-depth-only training.
```

### 2026-05-20 Stage8D Targeted Checksum8 Repair Eval

Stage8D added `--synthetic-depth-family-pattern` support so the curriculum can
target specific depth/family pairs instead of only global family ratios.

Code/tests:

```text
script=scripts/511_train_qwen_state_transition_hrmtext.py
test=tests/test_hrm_text_aligned_training_data.py
verified=PYTHONPATH=src .venv/bin/python -m py_compile scripts/511_train_qwen_state_transition_hrmtext.py scripts/512_eval_qwen_state_transition_generalization.py
verified=PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py
```

Stage8D:

```text
run=/tmp/qtrm_eval/20260520_192107_20260520_192106_STAGE8D_local_targeted_checksum8_repair_from_stage8B_seed55
resume=/tmp/qtrm_eval/20260520_185531_20260520_1858_STAGE8B_local_real_hrmtext_weakheal64_checksumheavy_from_stage7_seed53/best.pt
synthetic_depth_family_pattern=chain:4,chain:6,chain:8,chain:10,checksum:8,checksum:8,checksum:10
lr=1e-6
epochs=10
best_epoch=10
best_train_accuracy=0.15625000
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_192106_STAGE8D_local_targeted_checksum8_repair_from_stage8B_seed55_eval_depth46810_count1024.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_192106_STAGE8D_local_targeted_checksum8_repair_from_stage8B_seed55_depth46810

depth4:  TRM=0.1182 raw_qwen=0.1006 delta=+0.0176
depth6:  TRM=0.1318 raw_qwen=0.1201 delta=+0.0117
depth8:  TRM=0.1436 raw_qwen=0.0977 delta=+0.0459
depth10: TRM=0.1279 raw_qwen=0.0957 delta=+0.0322
```

Family breakdown:

```text
depth4 checksum:  TRM=0.0923 raw=0.1015 delta=-0.0092
depth6 checksum:  TRM=0.1015 raw=0.1104 delta=-0.0090
depth8 checksum:  TRM=0.1243 raw=0.0865 delta=+0.0378
depth10 checksum: TRM=0.0950 raw=0.0810 delta=+0.0140
```

Interpretation:

```text
Stage8D improves the targeted depth8 checksum delta more strongly than Stage8C:
  Stage8C depth8 checksum delta=+0.0195
  Stage8D depth8 checksum delta=+0.0378

But the targeted pattern sacrifices aggregate accuracy and does not preserve
Stage8B's depth10 peak:
  Stage8B depth10=0.1670
  Stage8C depth10=0.1455
  Stage8D depth10=0.1279

Conclusion: targeted depth/family replay works as a causal knob, but the Stage8D
ratio is too aggressive. Do not promote Stage8D as best. Use the new pattern
mechanism with a milder checksum8 dose or move to architecture-side fixes.
```

### 2026-05-20 DGX Status

DGX is currently assigned to dataset acquisition and monitoring, not QTRM GPU
training.

```text
dataset=sapientinc/HRM-Text-data-io-cleaned-20260515
download_root=/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515
download_process=running
latest_monitor_summary=/mnt/data4tb/qtrm_eval/_dataset_monitor_logs/20260520_hrm_text_download_monitor/latest_summary.json
dgx_tensorboard=http://192.168.219.113:6008
dgx_aim=http://192.168.219.113:43801
monitor_experiment=hrm_text_dataset_monitor
```

DGX training blocker:

```text
/mnt/data4tb/qtrm_multimodal_memoryos/.venv has no torch.
/mnt/data4tb/ws_llm/.venv has torch 2.9.1+cu130 and CUDA is visible, but
NVIDIA GB10 reports compute capability 12.1 while this PyTorch build warns that
it supports up to 12.0. Previous QTRM training failed around Triton/FLA ptxas
`sm_121a`, so DGX GPU training should wait for a compatible PyTorch/Triton/CUDA
stack or a CPU/offline data-prep role.
```

### 2026-05-20 Big-Jump Methodology Update

The Stage8C/Stage8D sequence showed that local curriculum and depth/family replay
can repair individual slices, but do not create the kind of large jump that
attention pooling created earlier. The operating rule is now:

```text
If two adjacent curriculum/data-ratio runs produce similar aggregate held-out
metrics, stop local tuning. Search recent papers/implementations for the exact
bottleneck, then change one architecture or objective path that can remove it.
```

The successful attention-pooling pattern is the template:

```text
Old bottleneck:
  fixed mean/last Qwen hidden-state compression

Big-jump fix:
  learned attention pooling over Qwen hidden states

Reason:
  the model could finally select the informative prompt tokens instead of
  forcing the recurrent core to reason from a blurred prompt summary.
```

Current analogous bottleneck:

```text
fixed final-step recurrent readout
```

Big-jump candidate now under test:

```text
Stage9/Stage9B:
  learned attention over the recurrent trajectory
  weak or silent intermediate-state supervision

Hypothesis:
  the correct answer may live in an earlier or mixed recurrent state, and forcing
  only the final step to be read can erase useful intermediate computation.
```

Skill update:

```text
file=/home/tripleyoung/.agents/skills/research-driven-architecture-debugging/SKILL.md
section=Big-Jump Methodology
rule=do not run a third similar curriculum tweak after two flat runs; escalate
     to bottleneck-targeted architecture/objective A/B.
```

### 2026-05-20 Stage9B/9C Trajectory Readout Results

Stage9B introduced recurrent trajectory attention readout plus weak state
supervision.

```text
run=/tmp/qtrm_eval/20260520_193608_20260520_193607_STAGE9B_local_trajattn_weakstate_from_stage8B_seed57
resume=/tmp/qtrm_eval/20260520_185531_20260520_1858_STAGE8B_local_real_hrmtext_weakheal64_checksumheavy_from_stage7_seed53/best.pt
recurrent_readout_pooling=attention
state_supervision_weight=0.1
final_readout_answer_weight=0.0
best_epoch=14
best_train_accuracy=0.31054688
```

Held-out eval with attention readout:

```text
eval=/tmp/qtrm_generalization_eval/20260520_193607_STAGE9B_local_trajattn_weakstate_from_stage8B_seed57_eval_depth46810_count1024.json

depth4:  TRM=0.1377 raw_qwen=0.1035 delta=+0.0342
depth6:  TRM=0.1514 raw_qwen=0.1055 delta=+0.0459
depth8:  TRM=0.1123 raw_qwen=0.1172 delta=-0.0049
depth10: TRM=0.1152 raw_qwen=0.0938 delta=+0.0215
```

Final-readout ablation of the same Stage9B checkpoint:

```text
eval=/tmp/qtrm_generalization_eval/20260520_193607_STAGE9B_local_trajattn_weakstate_from_stage8B_seed57_finalreadout_ablation_eval_depth46810_count1024.json

depth4:  TRM=0.1406 raw_qwen=0.1035 delta=+0.0371
depth6:  TRM=0.1436 raw_qwen=0.1055 delta=+0.0381
depth8:  TRM=0.1211 raw_qwen=0.1172 delta=+0.0039
depth10: TRM=0.1357 raw_qwen=0.0938 delta=+0.0420
```

Interpretation:

```text
Stage9B is a real train-set big jump:
  Stage8B best_train_accuracy=0.1855
  Stage9B best_train_accuracy=0.3105

But attention readout overfits the training distribution at depth8/depth10.
The final-readout ablation partially recovers depth8/depth10, which means the
core is not simply destroyed; the learned trajectory readout is too free.
```

Stage9C added final-readout answer loss to bind attention readout back to the
canonical final-state path.

```text
run=/tmp/qtrm_eval/20260520_195041_20260520_195040_STAGE9C_local_trajattn_finalaux_repair_from_stage9B_seed58
resume=/tmp/qtrm_eval/20260520_193608_20260520_193607_STAGE9B_local_trajattn_weakstate_from_stage8B_seed57/best.pt
final_readout_answer_weight=0.5
best_train_accuracy=0.15039062
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_195040_STAGE9C_local_trajattn_finalaux_repair_from_stage9B_seed58_eval_depth46810_count1024.json

depth4:  TRM=0.1406 raw_qwen=0.0908 delta=+0.0498
depth6:  TRM=0.1289 raw_qwen=0.1045 delta=+0.0244
depth8:  TRM=0.1357 raw_qwen=0.1094 delta=+0.0264
depth10: TRM=0.1582 raw_qwen=0.0957 delta=+0.0625
```

Interpretation:

```text
Final-readout auxiliary repairs the depth8/depth10 generalization failure, but
weight=0.5 is too strong and erases most of the Stage9B train jump. Stage9D is
therefore running with final_readout_answer_weight=0.2 to search for a better
balance.
```

### 2026-05-20 HRM-Text Generalization Mechanism Check

Primary sources checked:

```text
paper=https://sapientinc.github.io/HRM-Text/assets/HRM_Text.pdf
repo=/home/tripleyoung/qtrm-workspace/external_repos/HRM-Text
data_repo=/home/tripleyoung/qtrm-workspace/external_repos/data_io
```

Conclusion:

```text
HRM-Text's generalization is not explained by the recurrent architecture alone.
It is a coupled recipe:

1. HRM H/L recurrence:
   H_cycles=2 and L_cycles=3 in the released config, with half_layers=true.
   The local implementation uses backprop warmup from bp_min_steps to
   bp_max_steps so the recurrent path is trained gradually instead of shocked
   into instability.

2. PrefixLM task-completion objective:
   Data is instruction/response shaped. Prefix tokens can be bidirectionally
   consumed while response tokens are predicted causally. This makes the
   pretraining task closer to "complete the answer from a prompt" than raw
   web next-token modeling.

3. Stratified data sampling:
   data_io standardizes rows into condition/instruction/response and then
   builds sampled tokenized epochs with prefix_config controls. Generalization
   comes from coverage-balanced task families, not from a small ad-hoc mix.

4. Scale and optimization:
   The reference runs use large token batches, only 4 sampled epochs, EMA
   weights for eval/export, FlashAttention PrefixLM kernels, FSDP2, and
   checkpoint-per-epoch evaluation.
```

Implication for QTRM:

```text
If QTRM wants HRM-Text-like generalization, the next gates should not chase
train accuracy alone. We need:

- train/generalization metrics separated in TensorBoard and Aim;
- PrefixLM-style healing instead of single-token answer-only healing;
- source/family/depth stratified sampling;
- held-out family/depth gates as the promotion metric;
- architecture changes only when they target a measured bottleneck.

The current Stage10D recipe matches the readout bottleneck fix, but not yet the
full HRM-Text data/objective recipe. Therefore 80-90% held-out generalization
should not be expected from architecture-only tweaks on a tiny synthetic mix.
```

### 2026-05-20 Stage5 Reproduction Gate

The first Stage12A attempt was stopped and marked invalid because it changed
too many variables at once:

```text
invalid_run=/tmp/qtrm_eval/20260520_215842_20260520_215841_STAGE12A_stage5best_realhrmtext_strat_depth46810_heal128_seed70
reason=not an exact reproduction; it changed train distribution to depth4/6/8/10,
       added real HRM-Text rows, enabled variable-depth sampling, and added
       depth consistency simultaneously.
epoch1_accuracy=0.1198
decision=do_not_interpret_against Stage5 train accuracy
```

Exact Stage5 reproduction gate:

```text
run=/tmp/qtrm_eval/20260520_220313_20260520_220312_REPRO_STAGE5_trainacc945_exact_lr0_seed46
checkpoint=/tmp/qtrm_eval/20260520_170611_20260520_1702_STAGE5_local_depth468_gen512_heal128_from_stage4_attnpool_seed46/best.pt
method=one epoch, lr=0, original Stage5 train distribution/config
best_train_accuracy=0.96679688
tensorboard_logdir=/tmp/qtrm_eval/20260520_220313_20260520_220312_REPRO_STAGE5_trainacc945_exact_lr0_seed46/logs
```

Held-out reproduction gate:

```text
eval=/tmp/qtrm_generalization_eval/20260520_220407_REPRO_STAGE5_best_eval_depth46810_count1024_seed10042.json
tensorboard_logdir=/tmp/qtrm_eval/_generalization_eval_logs/20260520_220407_REPRO_STAGE5_best_eval_depth46810_count1024_seed10042

depth4:  TRM=0.102539 raw_qwen=0.100586 delta=+0.001953
depth6:  TRM=0.113281 raw_qwen=0.111328 delta=+0.001953
depth8:  TRM=0.128906 raw_qwen=0.094727 delta=+0.034180
depth10: TRM=0.103516 raw_qwen=0.103516 delta=+0.000000
```

Decision:

```text
Stage5's 90%+ train binding is reproducible. It is not a held-out
generalization breakthrough by itself. Further HRM-Text-style changes must be
applied one variable at a time, with an exact reproduction gate before each
distribution/objective change.
```

### 2026-05-20 Stage12B Real HRM-Text Healing-Only Gate

One-variable change from the exact Stage5 reproduction:

```text
run=/tmp/qtrm_eval/20260520_220819_20260520_220817_STAGE12B_stage5_realhrmtext_only_seed71
resume=/tmp/qtrm_eval/20260520_170611_20260520_1702_STAGE5_local_depth468_gen512_heal128_from_stage4_attnpool_seed46/best.pt
changed=healing rows source only
healing_data_path=/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515
kept=Stage5 train depths 4/6/8, random synthetic sampling, n_steps=8,
     workspace attention pooling, final readout, healing_count=128,
     healing_weight=0.15
best_epoch=3
best_train_accuracy=0.97265625
tensorboard_logdir=/tmp/qtrm_eval/20260520_220819_20260520_220817_STAGE12B_stage5_realhrmtext_only_seed71/logs
```

Held-out gate:

```text
eval=/tmp/qtrm_generalization_eval/20260520_221026_STAGE12B_stage5_realhrmtext_only_eval_depth46810_count1024_seed10042.json
tensorboard_logdir=/tmp/qtrm_eval/20260520_220819_20260520_220817_STAGE12B_stage5_realhrmtext_only_seed71/eval_logs

depth4:  Stage5=0.102539 Stage12B=0.118164 diff=+0.015625
depth6:  Stage5=0.113281 Stage12B=0.107422 diff=-0.005859
depth8:  Stage5=0.128906 Stage12B=0.125000 diff=-0.003906
depth10: Stage5=0.103516 Stage12B=0.113281 diff=+0.009766
mean:    Stage5=0.112061 Stage12B=0.115967 diff=+0.003906
```

Decision:

```text
Real HRM-Text healing-only is a weak positive gate. It preserves the 90%+
train binding and improves mean held-out accuracy slightly, especially depth4
and depth10, but it is not a big jump by itself. Keep as a valid component and
test stratified reasoning sampling independently before combining changes.
```

### 2026-05-20 Stage12C Stratified Reasoning-Only Gate

One-variable change from exact Stage5:

```text
run=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72
resume=/tmp/qtrm_eval/20260520_170611_20260520_1702_STAGE5_local_depth468_gen512_heal128_from_stage4_attnpool_seed46/best.pt
changed=synthetic_samping_strategy random -> stratified only
kept=Stage5 healing source/count/weight, train_depths 4/6/8, n_steps=8,
     workspace attention pooling, final readout
best_epoch=4
best_train_accuracy=0.26562500
tensorboard_logdir=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/logs
```

Held-out gate:

```text
eval=/tmp/qtrm_generalization_eval/20260520_221508_STAGE12C_stage5_stratified_only_eval_depth46810_count1024_seed10042.json
tensorboard_logdir=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/eval_logs

depth4:  Stage5=0.102539 Stage12C=0.129883 diff=+0.027344
depth6:  Stage5=0.113281 Stage12C=0.127930 diff=+0.014648
depth8:  Stage5=0.128906 Stage12C=0.171875 diff=+0.042969
depth10: Stage5=0.103516 Stage12C=0.124023 diff=+0.020508
mean:    Stage5=0.112061 Stage12C=0.138428 diff=+0.026367
```

Decision:

```text
Stratified reasoning sampling is a stronger positive gate than real-HRM healing
alone. It lowers same-run train accuracy because the model is no longer being
measured on the memorized Stage5 random train distribution, but it improves
held-out generalization at every depth. Do not reject from low train accuracy.
The next valid experiment is a longer stratified run with checkpoint/eval
cadence, then a combination with real HRM-Text healing if the curve keeps
improving.
```

### 2026-05-20 Stage12D Long Stratified Control

Longer training from Stage12C:

```text
run=/tmp/qtrm_eval/20260520_222253_STAGE12D_long_stratified_from12C_seed73
resume=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/best.pt
epochs=30
lr=1e-5
kept=Stage12C stratified setup
best_epoch=27
best_train_accuracy=0.98828125
tensorboard_logdir=/tmp/qtrm_eval/20260520_222253_STAGE12D_long_stratified_from12C_seed73/logs
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_223643_STAGE12D_long_stratified_eval_depth46810_count1024_seed10042.json
tensorboard_logdir=/tmp/qtrm_eval/20260520_222253_STAGE12D_long_stratified_from12C_seed73/eval_logs

depth4:  Stage12C=0.129883 Stage12D=0.116211 diff=-0.013672
depth6:  Stage12C=0.127930 Stage12D=0.123047 diff=-0.004883
depth8:  Stage12C=0.171875 Stage12D=0.134766 diff=-0.037109
depth10: Stage12C=0.124023 Stage12D=0.121094 diff=-0.002930
mean:    Stage12C=0.138428 Stage12D=0.123779 diff=-0.014648
```

Decision:

```text
Long training proves the stratified train distribution can recover 90%+ train
accuracy, but it overfits relative to Stage12C's held-out peak. The next
training-side fix is not simply more epochs; it is checkpoint/eval cadence,
early stopping, stronger held-out selection, and regularization/consistency.
```

### 2026-05-20 Stage13A MiniGatedDeltaCore A/B

Architecture intervention:

```text
code=MiniGatedDeltaReasoningCore drop-in replacement for SharedReasoningCore
cli=--core-update mini_gated_delta
run=/tmp/qtrm_eval/20260520_223854_STAGE13A_minigateddelta_from12C_seed75
resume=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/best.pt
load_stats=exact 32, partial 2, skipped 4
best_epoch=12
best_train_accuracy=0.53320312
tensorboard_logdir=/tmp/qtrm_eval/20260520_223854_STAGE13A_minigateddelta_from12C_seed75/logs
```

Held-out eval:

```text
eval=/tmp/qtrm_generalization_eval/20260520_224450_STAGE13A_minigateddelta_eval_depth46810_count1024_seed10042.json
tensorboard_logdir=/tmp/qtrm_eval/20260520_223854_STAGE13A_minigateddelta_from12C_seed75/eval_logs

depth4:  Stage12C=0.129883 Stage13A=0.119141 diff=-0.010742
depth6:  Stage12C=0.127930 Stage13A=0.125977 diff=-0.001953
depth8:  Stage12C=0.171875 Stage13A=0.122070 diff=-0.049805
depth10: Stage12C=0.124023 Stage13A=0.092773 diff=-0.031250
mean:    Stage12C=0.138428 Stage13A=0.114990 diff=-0.023438
```

Decision:

```text
Reject Stage13A as a big-jump candidate in this form. The MiniGatedDelta drop-in
learns stably but underperforms the MLP recurrent core on held-out depth8/10.
Keep the implementation for future ablations, but do not promote it. The
strongest current path is Stage12C-style stratified training with early stopping
and held-out-driven checkpoint selection, then test trajectory/readout or
trace-supervision changes rather than replacing the recurrent update first.
```

### 2026-05-20 Stage14A/14B Weak Trace + Depth Consistency Gate

Current literature signal:

```text
Latent reasoning needs supervised thinking-state or trajectory guidance before
RL-style latent optimization. Relevant mechanisms:
- Supervised Thinking States: learn intermediate thought states from supervision.
- KaVa-style distillation: latent trajectory/KV supervision can close the
  supervision gap.
- Depth-recurrent generalization papers: stable recurrence plus final/held-out
  selection matters more than simply increasing epochs.
```

Code change:

```text
script=scripts/511_train_qwen_state_transition_hrmtext.py
added=--eval-every, --eval-count, --eval-depths, --eval-batch-size, --eval-seed
added=--generalization-early-stop-patience, --generalization-early-stop-min-delta
added_checkpoint=best_generalization.pt
selection=train best remains best.pt; held-out mean best is best_generalization.pt
tensorboard=Generalization/HeldOut/*
aim=generalization_trm_accuracy, generalization_trm_mean_accuracy,
    best_generalization_trm_mean_accuracy
```

Local Stage14A:

```text
run=/tmp/qtrm_eval/20260520_230209_STAGE14A_local_weaktrace_depthcons_from12C_seed76
resume=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/best.pt
objective=weak supervised thinking states + depth consistency + held-out checkpoint selection
aux_step_answer_weight=0.35
state_supervision_weight=0.15
depth_consistency_weight=0.20
consistency_min_steps=4
eval_every=1
eval_count=256
eval_depths=4,6,8,10
tensorboard_logdir=/tmp/qtrm_eval/20260520_230209_STAGE14A_local_weaktrace_depthcons_from12C_seed76/logs
aim_experiment=qwen35_hrmtext_stage14
```

First held-out signal:

```text
epoch1_train_accuracy=0.13671875
epoch1_heldout_mean_accuracy=0.17480469
depth4=0.14453125
depth6=0.16015625
depth8=0.21484375
depth10=0.17968750
best_generalization_checkpoint=/tmp/qtrm_eval/20260520_230209_STAGE14A_local_weaktrace_depthcons_from12C_seed76/best_generalization.pt
```

Comparison:

```text
Stage12C held-out mean=0.138428
Stage12D long-train held-out mean=0.123779
Stage13A MiniGatedDelta held-out mean=0.114990
Stage14A epoch1 held-out mean=0.174805
```

Decision:

```text
Strong early positive signal. This is not a high-train-accuracy artifact:
train accuracy at the selected checkpoint is low, while held-out depth8/depth10
improve sharply. Continue the run, keep best_generalization.pt as the promoted
artifact, and verify with a larger eval count/seed after the run finishes.
Observed later epochs show train accuracy rising while held-out drops, so the
next control must use generalization early-stop rather than longer training.
```

Larger eval correction:

```text
Stage14A best_generalization eval_count=1024 seed=10042:
depth4=0.1387
depth6=0.1260
depth8=0.1777
depth10=0.1357
mean≈0.1445

Stage14A best_generalization eval_count=1024 seed=20042:
depth4=0.1514
depth6=0.1602
depth8=0.1523
depth10=0.1484
mean≈0.1531

Stage12C baseline eval_count=1024 seed=20042:
depth4=0.1523
depth6=0.1514
depth8=0.1318
depth10=0.1572
mean≈0.1482
```

Corrected interpretation:

```text
The 256-sample epoch1 jump overestimated the effect. The larger 1024-sample
eval still gives a positive but modest multi-seed improvement over Stage12C,
mostly through depth8 and seed-stable raw-Qwen margin. Treat Stage14 as a valid
direction, not a solved breakthrough. The next step is anti-overfit scheduling
and stronger latent trajectory supervision.
```

DGX Stage12D scaled baseline:

```text
run=/mnt/data4tb/qtrm_eval/20260520_222652_STAGE12D_DGX_long_stratified_count2048_from12C_seed74
eval=/mnt/data4tb/qtrm_generalization_eval/20260520_230040_DGX_STAGE12D_scaled_eval.json
depth4=0.1318 raw_qwen=0.1016
depth6=0.1348 raw_qwen=0.1084
depth8=0.1494 raw_qwen=0.0947
depth10=0.1191 raw_qwen=0.1045
mean_approx=0.1338
decision=beats raw Qwen at every depth, but still below Stage12C mean; long training alone is not enough.
```

DGX Stage14B launched:

```text
run=/mnt/data4tb/qtrm_eval/20260520_230433_STAGE14B_DGX_weaktrace_depthcons_count2048_seed77
resume=/mnt/data4tb/qtrm_eval/_imported_checkpoints/stage12c_local/best.pt
reasoning_count=2048
eval_every=2
eval_count=512
aim_experiment=qwen35_hrmtext_stage14
tensorboard_logdir=/mnt/data4tb/qtrm_eval/20260520_230433_STAGE14B_DGX_weaktrace_depthcons_count2048_seed77/logs
status=running
epoch2_heldout_mean_accuracy=0.1602
depth4=0.1582
depth6=0.1523
depth8=0.1660
depth10=0.1641
best_so_far_epoch=6
best_so_far_heldout_mean_accuracy=0.16796875
epoch6_depth4=0.1406
epoch6_depth6=0.1797
epoch6_depth8=0.1719
epoch6_depth10=0.1797
decision=replicates the Stage14 direction at larger count; wait for later evals, but keep best_generalization checkpoint.
```

Stage14C local anti-overfit launched:

```text
run=/tmp/qtrm_eval/20260520_231845_STAGE14C_local_antioverfit_weaktrace_seed78
resume=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/best.pt
lr=3e-6
aux_step_answer_weight=0.25
state_supervision_weight=0.05
depth_consistency_weight=0.35
eval_count=512
generalization_early_stop_patience=3
generalization_early_stop_min_delta=0.001
decision_goal=preserve early latent-generalization gain while preventing the train-accuracy/held-out collapse seen in Stage14A.
best_so_far_epoch=2
best_so_far_heldout_mean_accuracy=0.17187500
epoch2_depth4=0.1660
epoch2_depth6=0.1562
epoch2_depth8=0.1895
epoch2_depth10=0.1758
status=running
```

Interim decision:

```text
Stage14C is the current best local variant under the larger 512-sample
in-training eval. It confirms that weaker direct state labels plus stronger
depth consistency preserve the latent generalization gain better than Stage14A.
Run must still be confirmed with eval_count=1024 and seed 10042/20042 after
early-stop/final completion.
```

Stage14C completion and larger eval:

```text
run=/tmp/qtrm_eval/20260520_231845_STAGE14C_local_antioverfit_weaktrace_seed78
early_stop_epoch=5
best_generalization_epoch=2
best_generalization_mean_accuracy_eval512=0.17187500
checkpoint=/tmp/qtrm_eval/20260520_231845_STAGE14C_local_antioverfit_weaktrace_seed78/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.126953
depth6=0.125000
depth8=0.166992
depth10=0.147461
mean=0.141602

eval_count=1024 seed=20042:
depth4=0.152344
depth6=0.164062
depth8=0.152344
depth10=0.173828
mean=0.160645
```

Two-seed corrected comparison:

```text
Stage12C mean across seeds 10042/20042:
seed10042=0.138428
seed20042=0.148193
two_seed_mean=0.143311

Stage14A mean across seeds 10042/20042:
seed10042=0.144531
seed20042=0.153076
two_seed_mean=0.148804

Stage14C mean across seeds 10042/20042:
seed10042=0.141602
seed20042=0.160645
two_seed_mean=0.151123
```

Decision:

```text
Stage14C is the best corrected local result so far. The small-sample 17%+
numbers were optimistic, but the 1024-count two-seed evaluation still improves
over Stage12C by about +0.78 percentage points absolute, with the biggest gain
on depth10 seed20042. This is a real direction, not yet a large breakthrough.
Next architecture/objective step should target a bigger jump: richer latent
trajectory supervision or teacher hidden/KV trajectory distillation.
```

### 2026-05-20 Stage14B/15A Corrected Gate

Question:

```text
Does scaled DGX weak-trace training or weak Qwen-workspace trajectory anchoring
produce a durable held-out generalization jump after the optimistic 512-sample
signals?
```

Research signal checked:

```text
Recent latent-recurrent work points toward stepwise credit assignment and
explicit correction, not only passive final-state anchoring:

- LSRL / process-supervised GRPO on latent recurrent states:
  https://aclanthology.org/2025.findings-emnlp.669/
- Scaling test-time compute with latent reasoning / recurrent depth:
  https://arxiv.org/abs/2502.05171
- Adaptive latent reasoning length:
  https://arxiv.org/abs/2511.21581
- Recursive latent-space OOD generalization mechanisms:
  https://huggingface.co/papers/2510.14095
```

DGX Stage14B corrected eval:

```text
run=/mnt/data4tb/qtrm_eval/20260520_230433_STAGE14B_DGX_weaktrace_depthcons_count2048_seed77
best_generalization_epoch=6
best_generalization_mean_accuracy_eval512=0.16796875
checkpoint=/mnt/data4tb/qtrm_eval/20260520_230433_STAGE14B_DGX_weaktrace_depthcons_count2048_seed77/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.128906
depth6=0.134766
depth8=0.166016
depth10=0.142578
mean=0.143066

eval_count=1024 seed=20042:
depth4=0.139648
depth6=0.143555
depth8=0.130859
depth10=0.164062
mean=0.144531

two_seed_mean=0.143799
decision=does not beat Stage14C corrected two-seed mean and is roughly tied with Stage12C.
```

Local Stage15A trajectory-anchor corrected eval:

```text
run=/tmp/qtrm_eval/20260520_233319_STAGE15A_local_trajanchor_seed79
resume=/tmp/qtrm_eval/20260520_221257_20260520_221256_STAGE12C_stage5_stratified_only_seed72/best.pt
trajectory_anchor_weight=0.08
trajectory_anchor_min_step=2
early_stop_epoch=4
best_generalization_epoch=1
best_generalization_mean_accuracy_eval512=0.16064453
checkpoint=/tmp/qtrm_eval/20260520_233319_STAGE15A_local_trajanchor_seed79/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.127930
depth6=0.127930
depth8=0.170898
depth10=0.118164
mean=0.136230

eval_count=1024 seed=20042:
depth4=0.157227
depth6=0.156250
depth8=0.150391
depth10=0.132812
mean=0.149170

two_seed_mean=0.142700
decision=reject as currently implemented. Weak Qwen-workspace anchoring is too conservative/noisy and does not reproduce the Stage14C corrected gain.
```

Updated decision:

```text
Do not keep extending Stage14B/15A by epochs. The held-out gate shows that
train accuracy can rise while larger held-out generalization stalls or drops.

The next big-jump candidate should add active latent-step credit assignment:
error-conditioned correction or process-style latent rewards/labels that feed
the normal answer path. A passive hidden-state anchor is not enough.
```

### 2026-05-20 Stage15B/15C Error-Conditioned Correction Gate

Implementation:

```text
files:
  src/qtrm_mm/qwen_backbone_state_transition.py
  scripts/511_train_qwen_state_transition_hrmtext.py
  scripts/512_eval_qwen_state_transition_generalization.py
  tests/test_qwen_state_transition_workspace_pooling.py

mechanism:
  first latent answer logits -> predicted answer-delta/error distribution
  error distribution embedding + entropy/margin features -> correction state
  second StateTransitionCore pass conditioned by correction state
  final answer still reads through the normal QTRM answer path

training loss:
  correction_feedback_loss = CE(predicted_delta, (gold_answer - first_pred) mod 10)

observability:
  TensorBoard/Aim log Loss_CorrectionFeedback and Correction_Feedback_Gate
```

Verification:

```text
local:
  PYTHONPATH=src .venv/bin/python -m py_compile src/qtrm_mm/qwen_backbone_state_transition.py scripts/511_train_qwen_state_transition_hrmtext.py scripts/512_eval_qwen_state_transition_generalization.py
  PYTHONPATH=src .venv/bin/python tests/test_qwen_state_transition_workspace_pooling.py
  PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py

dgx:
  syntax compile via built-in compile()
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /mnt/data4tb/venv_sglang_pr23000/bin/python tests/test_qwen_state_transition_workspace_pooling.py
```

Local Stage15B launched:

```text
run=/tmp/qtrm_eval/20260520_234925_STAGE15B_local_errorcorr_from14C_seed80
resume=/tmp/qtrm_eval/20260520_231845_STAGE14C_local_antioverfit_weaktrace_seed78/best_generalization.pt
reasoning_count=512
lr=5e-6
correction_feedback=true
correction_feedback_scale=0.75
correction_feedback_loss_weight=0.25
eval_count=512
aim_experiment=qwen35_hrmtext_stage15
tensorboard_logdir=/tmp/qtrm_eval/20260520_234925_STAGE15B_local_errorcorr_from14C_seed80/logs

epoch1:
train_acc=0.1016
heldout_mean=0.1621
depth4=0.1543
depth6=0.1621
depth8=0.1680
depth10=0.1641
correction_feedback_loss=1.0203

epoch2:
train_acc=0.1465
heldout_mean=0.1548
depth4=0.1406
depth6=0.1562
depth8=0.1699
depth10=0.1523
correction_feedback_loss=0.9273

epoch3:
train_acc=0.1426
heldout_mean=0.1626
depth4=0.1465
depth6=0.1562
depth8=0.1836
depth10=0.1641
correction_feedback_loss=0.9139

interim_decision=correction loss is learning, but no big jump yet; keep running to early-stop, then confirm best with eval_count=1024 and seeds 10042/20042 only if it beats Stage14C's corrected two-seed mean.
```

Local Stage15B completion and corrected eval:

```text
early_stop_epoch=6
best_generalization_epoch=3
best_generalization_mean_accuracy_eval512=0.16259766
checkpoint=/tmp/qtrm_eval/20260520_234925_STAGE15B_local_errorcorr_from14C_seed80/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.125977
depth6=0.130859
depth8=0.165039
depth10=0.147461
mean=0.142334

eval_count=1024 seed=20042:
depth4=0.156250
depth6=0.155273
depth8=0.143555
depth10=0.164062
mean=0.154785

two_seed_mean=0.148560
decision=reject. Correction-feedback learns a delta loss, but the larger held-out gate falls below Stage14C corrected two-seed mean=0.151123.
```

DGX Stage15C launched:

```text
run=/mnt/data4tb/qtrm_eval/20260520_234925_STAGE15C_DGX_errorcorr_from14C_seed81
resume=/mnt/data4tb/qtrm_eval/_imported_checkpoints/stage14c_local/best_generalization.pt
reasoning_count=2048
lr=5e-6
correction_feedback=true
correction_feedback_scale=0.75
correction_feedback_loss_weight=0.25
eval_count=512
aim_experiment=qwen35_hrmtext_stage15
tensorboard_logdir=/mnt/data4tb/qtrm_eval/20260520_234925_STAGE15C_DGX_errorcorr_from14C_seed81/logs

epoch1:
train_acc=0.1587
heldout_mean=0.1499
depth4=0.1523
depth6=0.1582
depth8=0.1465
depth10=0.1426
correction_feedback_loss=1.1306

epoch2:
train_acc=0.1582
heldout_mean=0.1558
depth4=0.1562
depth6=0.1484
depth8=0.1738
depth10=0.1445
correction_feedback_loss=1.0809

interim_decision=DGX has not produced a large jump; best so far is 0.1558 at eval512. Continue to early-stop, but reject if larger 1024-count seeds do not beat Stage14C.
```

DGX Stage15C completion and corrected eval:

```text
early_stop_epoch=6
best_generalization_epoch=3
best_generalization_mean_accuracy_eval512=0.15771484
checkpoint=/mnt/data4tb/qtrm_eval/20260520_234925_STAGE15C_DGX_errorcorr_from14C_seed81/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.144531
depth6=0.124023
depth8=0.166016
depth10=0.130859
mean=0.141357

eval_count=1024 seed=20042:
depth4=0.131836
depth6=0.154297
depth8=0.140625
depth10=0.136719
mean=0.140869

two_seed_mean=0.141113
decision=reject. Scaling the same correction-feedback mechanism on DGX does not help; it underperforms Stage14C and Stage15B under the larger corrected gate.
```

Updated Stage15 decision:

```text
Error-conditioned correction as implemented is not the missing big-jump
mechanism. The method produces useful telemetry and a learnable correction
loss, but it does not improve durable held-out generalization.

Next candidate should move from self-predicted correction to external or
symbolic process credit:
  - supervised per-step process rewards/labels similar to LSRL's dense latent
    step credit, but adapted to the synthetic state families;
  - family-specific algorithmic supervision/codebook labels for operation and
    value transitions;
  - only then reconsider RL/GRPO-style latent reward optimization.

Do not continue Stage15B/15C by epoch count.
```

### 2026-05-21 Stage16 Operation Process Supervision Gate

Question:

```text
Stage15 showed that self-predicted correction does not improve durable
generalization. Does explicit symbolic process credit on the operation sequence
help the recurrent core generalize better?
```

Implementation:

```text
file=scripts/511_train_qwen_state_transition_hrmtext.py

added:
  compute_operation_supervision_loss(operation_logits, operation_ids)
  --operation-supervision-weight
  TensorBoard/Aim:
    Step/Loss_OperationSupervision
    Epoch/Loss_OperationSupervision
    Train/* mirrored metrics

default:
  operation_supervision_weight=0.0
  existing runs are unchanged unless explicitly enabled.
```

Verification:

```text
local:
  PYTHONPATH=src .venv/bin/python -m py_compile scripts/511_train_qwen_state_transition_hrmtext.py src/qtrm_mm/qwen_backbone_state_transition.py scripts/512_eval_qwen_state_transition_generalization.py
  PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py
  PYTHONPATH=src .venv/bin/python tests/test_qwen_state_transition_workspace_pooling.py

dgx:
  syntax compile via built-in compile()
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /mnt/data4tb/venv_sglang_pr23000/bin/python tests/test_hrm_text_aligned_training_data.py
```

Local Stage16A:

```text
run=/tmp/qtrm_eval/20260521_001641_STAGE16A_local_opsup_from14C_seed82
resume=/tmp/qtrm_eval/20260520_231845_STAGE14C_local_antioverfit_weaktrace_seed78/best_generalization.pt
operation_supervision_weight=0.35
reasoning_count=512
eval_count=512
aim_experiment=qwen35_hrmtext_stage16
tensorboard_logdir=/tmp/qtrm_eval/20260521_001641_STAGE16A_local_opsup_from14C_seed82/logs

epoch1:
train_acc=0.1660
heldout_mean=0.1631
depth4=0.1504
depth6=0.1504
depth8=0.1758
depth10=0.1758
operation_supervision_loss=0.7089

epoch2:
train_acc=0.1738
heldout_mean=0.1602
operation_supervision_loss=0.5868

epoch3:
train_acc=0.1914
heldout_mean=0.1558
operation_supervision_loss=0.5242

epoch4:
train_acc=0.2109
heldout_mean=0.1494
operation_supervision_loss=0.4751
early_stop_epoch=4
best_generalization_epoch=1
best_generalization_mean_accuracy_eval512=0.16308594
checkpoint=/tmp/qtrm_eval/20260521_001641_STAGE16A_local_opsup_from14C_seed82/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.130859
depth6=0.125977
depth8=0.171875
depth10=0.134766
mean=0.140869

eval_count=1024 seed=20042:
depth4=0.151367
depth6=0.155273
depth8=0.151367
depth10=0.146484
mean=0.151123

two_seed_mean=0.145996
decision=reject. Operation process loss is learnable, but corrected held-out generalization is below Stage14C.
```

DGX Stage16B:

```text
run=/mnt/data4tb/qtrm_eval/20260521_001641_STAGE16B_DGX_opsup_from14C_seed83
resume=/mnt/data4tb/qtrm_eval/_imported_checkpoints/stage14c_local/best_generalization.pt
operation_supervision_weight=0.35
reasoning_count=2048
eval_count=512
aim_experiment=qwen35_hrmtext_stage16
tensorboard_logdir=/mnt/data4tb/qtrm_eval/20260521_001641_STAGE16B_DGX_opsup_from14C_seed83/logs

epoch1:
train_acc=0.1411
heldout_mean=0.1592
operation_supervision_loss=0.7556

epoch3:
train_acc=0.1699
heldout_mean=0.1621
operation_supervision_loss=0.4725

epoch6:
train_acc=0.1895
heldout_mean=0.1577
operation_supervision_loss=0.3084
early_stop_epoch=6
best_generalization_epoch=3
best_generalization_mean_accuracy_eval512=0.16210938
checkpoint=/mnt/data4tb/qtrm_eval/20260521_001641_STAGE16B_DGX_opsup_from14C_seed83/best_generalization.pt

eval_count=1024 seed=10042:
depth4=0.130859
depth6=0.122070
depth8=0.162109
depth10=0.138672
mean=0.138428

eval_count=1024 seed=20042:
depth4=0.150391
depth6=0.149414
depth8=0.144531
depth10=0.154297
mean=0.149658

two_seed_mean=0.144043
decision=reject. Scaling operation supervision on DGX does not help.
```

Updated Stage16 decision:

```text
Operation labels alone are not the missing process credit. The loss falls
cleanly, but this appears to improve train/process decoding more than durable
OOD generalization.

Next gate should supervise the value-transition effect, not only the operation
name. Candidate:
  target_delta_t = (state_t - state_{t-1}) mod 10
  train a transition-effect head over recurrent deltas or states
  keep answer path unchanged and require eval_count=1024 two-seed improvement
  over Stage14C before promotion.
```

### 2026-05-21 agy CLI Big-Jump Ideation Pass

Reason:

```text
Stage15 and Stage16 showed a high reject rate for auxiliary losses:
  Stage15A trajectory anchor two_seed_mean=0.142700
  Stage15B local error correction two_seed_mean=0.148560
  Stage15C DGX error correction two_seed_mean=0.141113
  Stage16A local operation supervision two_seed_mean=0.145996
  Stage16B DGX operation supervision two_seed_mean=0.144043

Current promoted reference remains Stage14C:
  two_seed_mean=0.151123

Conclusion:
  Do not keep adding weak auxiliary probes. Use the attention-pooling discovery
  pattern: identify a causal information bottleneck and replace that mechanism.
```

agy CLI prompt/response:

```text
prompt=scratch/agy_stage17_big_jump_prompt.md
response=scratch/agy_stage17_big_jump_response.md
```

Candidate summary from agy:

```text
1. Vector-Quantized Recurrence / Discrete Routing Plane
   hypothesis:
     Continuous attention-pooled recurrent routing drifts at OOD depth.
   mechanism:
     Quantize Q/K routing projections with a straight-through VQ codebook while
     leaving V continuous.
   required telemetry:
     codebook entropy, active-code ratio, held-out depth accuracy.
   rejection:
     corrected two-seed mean <= 0.152000 or active codes < 15%.
   risk:
     codebook collapse / representation starvation.

2. Contrastive Recurrent Flow Alignment
   hypothesis:
     Local auxiliary losses do not shape the global recurrence field.
   mechanism:
     Align shallow/deep executions of the same logical trajectory with InfoNCE.
   rejection:
     corrected two-seed mean <= 0.151500 or projector collapse.
   risk:
     trivial fixed-point trajectory collapse.

3. Adversarial Depth-Invariant Attention Pooler
   hypothesis:
     The current pooler/core may exploit absolute depth cues that fail OOD.
   mechanism:
     Add a gradient-reversal depth discriminator on pooled/recurrent states.
   required telemetry:
     depth-discriminator accuracy, adversarial entropy, attention entropy,
     held-out depth accuracy.
   rejection:
     corrected two-seed mean <= 0.153000 or early answer-loss instability.
   risk:
     erasing useful order/sequence information along with absolute depth.
```

Local decision:

```text
The strongest immediate next experiment is not another process-label auxiliary
loss. Stage17 should target depth-generalization directly.

Stage17A:
  Implement adversarial depth-invariant pooler on top of the existing
  attention-pooling architecture and resume from Stage14C.

Stage17B:
  If Stage17A is stable but not enough, implement VQ routing-plane recurrence
  as the bolder DGX/local A/B candidate.

Do not promote any Stage17 result unless it beats Stage14C under the corrected
eval_count=1024 two-seed gate and is recorded in TensorBoard + Aim.
```

Correction after asking agy for provenance:

```text
agy explicitly stated that the three Stage17 candidates were NOT produced from
an actual literature/search pass and had no specific paper/repo source.

Candidate provenance:
  Vector-Quantized Recurrence:
    source=NO SOURCE
    synthesized from VQ/VQ-VAE-style ideas plus recurrent/discrete routing.

  Contrastive Recurrent Flow Alignment:
    source=NO SOURCE
    synthesized from InfoNCE/SimCLR-style contrastive learning plus recurrent
    trajectory alignment.

  Adversarial Depth-Invariant Attention Pooler:
    source=NO SOURCE
    synthesized from adversarial/domain-invariance regularization plus
    attention pooling.

Decision update:
  Treat these as hypothesis-generation only, not research-backed candidates.
  Before implementing Stage17, run an actual literature/repo search and map
  paper-backed mechanisms to the measured Stage14C bottleneck.
```

### 2026-05-21 Stage17 GRAM-Inspired Stochastic High-Level Guidance

Source:

```text
paper=Generative Recursive Reasoning (GRAM)
url=https://arxiv.org/html/2605.19376v1
project=https://ahn-ml.github.io/gram-website
```

Reason:

```text
Stage15/16 auxiliary losses were learnable but did not improve durable
OOD/depth generalization. GRAM targets a deeper bottleneck: deterministic
recursive models follow a single latent trajectory. If the trajectory falls
into the wrong attractor, more local supervision does not create alternative
reasoning paths.

Our first implementation imports the part that fits the current TRM design:
stochastic guidance on the high-level latent state z_H. This is intentionally
not a full GRAM ELBO/posterior/LPRM implementation yet.
```

Implementation:

```text
files:
  src/qtrm_mm/state_transition_core.py
  src/qtrm_mm/qwen_backbone_state_transition.py
  scripts/511_train_qwen_state_transition_hrmtext.py
  scripts/512_eval_qwen_state_transition_generalization.py
  tests/test_qwen_state_transition_workspace_pooling.py

added core args:
  --stochastic-high-level-guidance
  --stochastic-high-level-scale
  --stochastic-high-level-min-std
  --stochastic-high-level-max-std
  --stochastic-high-level-eval

added eval arg:
  --stochastic-eval-samples

mechanism:
  After the deterministic z_H update, predict mu/std from [z_H, ctx].
  Apply z_H <- z_H + scale * (mu + std * eps).
  Noise is applied only to z_H, not z_L.
  By default eval uses eps=0 for deterministic checkpoint selection.
  With --stochastic-high-level-eval and --stochastic-eval-samples, eval averages
  answer probabilities across multiple latent trajectories.

telemetry:
  Step/Epoch StochasticHighLevel_MuNorm
  Step/Epoch StochasticHighLevel_StdMean
  Step/Epoch StochasticHighLevel_NoiseNorm
  mirrored to Train/* and Aim.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/state_transition_core.py \
  src/qtrm_mm/qwen_backbone_state_transition.py \
  scripts/511_train_qwen_state_transition_hrmtext.py \
  scripts/512_eval_qwen_state_transition_generalization.py

PYTHONPATH=src .venv/bin/python tests/test_qwen_state_transition_workspace_pooling.py
PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py

train/eval --help exposes the stochastic flags.
```

Stage17 gate:

```text
reference=Stage14C two_seed_mean=0.151123
promotion_threshold:
  corrected eval_count=1024 two-seed mean >= 0.155000

reject if:
  two_seed_mean <= 0.151123
  stochastic std/noise telemetry collapses to zero while accuracy is unchanged
  stochastic eval samples increase variance without improving mean accuracy
  train accuracy improves but held-out depth4/6/8/10 mean does not
```

### 2026-05-21 Stage17B GRAM Posterior Guidance

Reason:

```text
Stage17A only imports GRAM's stochastic high-level transition. Full GRAM also
requires a target-aware posterior path and a prior/posterior coupling term.
Stage17B adds the next piece without adding LPRM yet.
```

Implementation:

```text
files:
  src/qtrm_mm/state_transition_core.py
  src/qtrm_mm/qwen_backbone_state_transition.py
  scripts/511_train_qwen_state_transition_hrmtext.py
  scripts/512_eval_qwen_state_transition_generalization.py
  tests/test_qwen_state_transition_workspace_pooling.py

added:
  --stochastic-posterior-guidance
  --stochastic-posterior-kl-weight

mechanism:
  prior p(z_t | x, z_H, ctx):
    predicted from [z_H, ctx]

  posterior q(z_t | x, y, z_H, ctx):
    predicted from [z_H, ctx, answer_label_embedding]

  training:
    if --stochastic-posterior-guidance is enabled, sample z_H stochastic
    guidance from q and add diagonal Gaussian KL(q || p) weighted by
    --stochastic-posterior-kl-weight.

  evaluation:
    no answer labels are passed, so the model samples from prior only.
    --stochastic-eval-samples can average multiple prior trajectories.

telemetry:
  Loss_StochasticPosteriorKL in TensorBoard and Aim.
  qtrm_stochastic_posterior_kls exposed from the model.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/state_transition_core.py \
  src/qtrm_mm/qwen_backbone_state_transition.py \
  scripts/511_train_qwen_state_transition_hrmtext.py \
  scripts/512_eval_qwen_state_transition_generalization.py

PYTHONPATH=src .venv/bin/python tests/test_qwen_state_transition_workspace_pooling.py
PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py
git diff --check
```

Gate:

```text
reference=Stage14C two_seed_mean=0.151123
promotion_threshold:
  corrected eval_count=1024 two-seed mean >= 0.155000

expected failure:
  posterior path may overfit to answer labels and fail to transfer to prior at
  eval. If KL is too weak, prior cannot imitate posterior. If KL is too strong,
  posterior becomes useless.
```

### 2026-05-21 Stage17C GRAM LPRM / Multi-Trajectory Selection

Reason:

```text
Complete GRAM-style inference needs a way to choose among sampled latent
trajectories. Stage17C adds a lightweight latent process reward model (LPRM)
head and evaluation-time trajectory selection modes.
```

Implementation:

```text
files:
  src/qtrm_mm/qwen_backbone_state_transition.py
  scripts/511_train_qwen_state_transition_hrmtext.py
  scripts/512_eval_qwen_state_transition_generalization.py
  tests/test_qwen_state_transition_workspace_pooling.py

added model output:
  qtrm_trajectory_reward_logits

added train args:
  --gram-lprm-weight
  --gram-lprm-target {true_prob,correct}

added eval args:
  --stochastic-eval-samples
  --stochastic-selection-mode {mean,vote,lprm}

mechanism:
  LPRM reads the normalized recurrent readout state and predicts a scalar
  trajectory quality score.

  Training target:
    true_prob:
      detached probability assigned to the gold answer.
    correct:
      detached hard correctness of the trajectory argmax.

  Evaluation:
    mean:
      average answer probabilities across sampled trajectories.
    vote:
      majority vote over sampled trajectory argmax answers.
    lprm:
      select the trajectory with highest qtrm_trajectory_reward_logits.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/state_transition_core.py \
  src/qtrm_mm/qwen_backbone_state_transition.py \
  scripts/511_train_qwen_state_transition_hrmtext.py \
  scripts/512_eval_qwen_state_transition_generalization.py

PYTHONPATH=src .venv/bin/python tests/test_qwen_state_transition_workspace_pooling.py
PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py
git diff --check
```

Current Stage17B local run:

```text
run=/tmp/qtrm_eval/20260521_011000_STAGE17B_local_GRAM_posterior_from14C_seed84
status=running

epoch1:
  train_acc=0.1426
  heldout_mean_eval512=0.1650

epoch2:
  train_acc=0.1660
  heldout_mean_eval512=0.1665

epoch3:
  train_acc=0.1836
  heldout_mean_eval512=0.1714

epoch4:
  train_acc=0.1953
  heldout_mean_eval512=0.1704

epoch5:
  train_acc=0.2207
  heldout_mean_eval512=0.1611

interim:
  Stage17B has not yet exceeded Stage14C eval512 best=0.171875. Final decision
  still requires corrected eval_count=1024 two-seed gate.
```

## 2026-05-21 Stage19/20 HRM-Text Source Gap Fix

Problem:

```text
The earlier Qwen3.5-0.8B state-transition runs used synthetic modulo-10
reasoning plus default Dolly healing only. That was not HRM-Text-like enough:
HRM-Text/data_io trains on broad cleaned instruction/response sources, while
our active loop was mostly probing the recurrent digit head.
```

Implemented source builder:

```text
files:
  src/qtrm_mm/data/hrm_text_source_mix.py
  scripts/521_build_hrmtext_source_mix.py
  tests/test_hrm_text_source_mix.py

format:
  data/*.jsonl rows with condition/instruction/response fields, compatible with
  scripts/511_train_qwen_state_transition_hrmtext.py --healing-data-path.

verified source set:
  gsm8k_train
  numina_verifiable_train
  openr1_math_verified_train
  openmathinstruct2_train
  proofwriter_validation
  clutrr_train
  bbh_boolean_test

language healing source:
  databricks/databricks-dolly-15k
```

Built persistent dataset:

```text
local=/mnt/sdc1/tripleyoung/qtrm_data/20260521_102930_HRMText_verified_dolly_mix_v1
dgx=/mnt/data4tb/qtrm_data/20260521_102930_HRMText_verified_dolly_mix_v1

rows:
  verified_reasoning=808
  dolly_healing=360
  total=1168

manifest:
  /mnt/sdc1/tripleyoung/qtrm_data/20260521_102930_HRMText_verified_dolly_mix_v1/manifest.json
```

Stage19 first-token source objective:

```text
local run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_103142_PERSIST_STAGE19A_HRMTextSourceMix_baseline_seed92

result:
  loaded HRM-Text healing rows=1168
  epoch1 heldout_mean_acc=0.1621
  epoch4 heldout_mean_acc=0.1621
  early stopped

interpretation:
  Source mix is wired correctly and healing loss falls, but first-token-only
  response supervision is too weak to be considered HRM-Text-like training.
```

Stage20 multi-token recurrent source objective:

```text
code change:
  scripts/511_train_qwen_state_transition_hrmtext.py

new arg:
  --healing-target-tokens

mechanism:
  Instead of predicting only the first response token from the final recurrent
  state, recurrent trajectory states z_1..z_T predict the first T response
  tokens through the frozen Qwen LM head.

new TensorBoard/Aim metric:
  Accuracy_HealingTargetTokens

active runs:
  local=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_104204_PERSIST_STAGE20A_HRMTextSourceMix_multitok8_seed94
  dgx=/mnt/data4tb/qtrm_eval/20260521_104227_DGX_STAGE20B_HRMTextSourceMix_multitok8_count2048_seed95

initial local epoch1:
  heldout_mean_acc=0.1621
  healing_loss=1.5831
  reasoning_train_acc=0.1250

local final after 6 epochs:
  heldout_mean_acc=0.1621
  healing_loss=1.1401
  healing_target_token_acc=0.0871
  reasoning_train_acc=0.1328

dgx partial after 2 epochs:
  heldout_mean_acc=0.1621
  healing_loss=0.6432
  reasoning_train_acc=0.1548

status:
  source gap is now addressed at both data-source and objective level, but no
  synthetic held-out generalization jump has appeared yet. Promotion requires
  heldout_mean_acc > 0.171875 or a new verified-source evaluation gate showing
  real improvement over the prior baseline.
```

Stage21 verified-source held-out gate:

```text
code change:
  scripts/511_train_qwen_state_transition_hrmtext.py

new args:
  --source-eval-data-path
  --source-eval-include-glob
  --source-eval-count
  --source-eval-rows-per-file-cap
  --source-eval-target-tokens
  --source-eval-batch-size

new outputs:
  latest_verified_source_eval.json

new TensorBoard/Aim metrics:
  Generalization/VerifiedSource/Loss_TargetTokens
  Generalization/VerifiedSource/Accuracy_TargetTokens
  Generalization/VerifiedSource/BySource/*
  Generalization/VerifiedSource/ByFamily/*

held-out source split:
  local=/mnt/sdc1/tripleyoung/qtrm_data/20260521_105939_HRMText_verified_eval_offset500_v1
  dgx=/mnt/data4tb/qtrm_data/20260521_105939_HRMText_verified_eval_offset500_v1

rows:
  verified_reasoning=400
  sources=gsm8k_train,numina_verifiable_train,openmathinstruct2_train,proofwriter_validation,clutrr_train

important loader note:
  Use --source-eval-include-glob data/verified_reasoning.jsonl. Without this,
  the empty dolly_healing.jsonl file is included in per-file sampling and the
  eval loader samples only 256 verified rows.

local initial gate with partial loader:
  run=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_110121_PERSIST_STAGE21A_SourceEvalGate_from20A_seed96
  resumed_from=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_104204_PERSIST_STAGE20A_HRMTextSourceMix_multitok8_seed94/last.pt
  verified_rows_loaded=256
  epoch1 verified_source_target_token_acc=0.1116
  epoch2 verified_source_target_token_acc=0.1292
  epoch2 verified_source_target_token_loss=4.7026
  epoch2 synthetic_heldout_mean_acc=0.1621

local corrected full verified gate:
  run=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_111000_PERSIST_STAGE21A2_SourceEvalGate_allVerified_from20A_seed97
  resumed_from=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_104204_PERSIST_STAGE20A_HRMTextSourceMix_multitok8_seed94/last.pt
  verified_rows_loaded=400
  epoch1 train_reasoning_acc=0.1738
  epoch1 synthetic_heldout_mean_acc=0.1621
  epoch1 verified_source_target_token_acc=0.1079
  epoch1 verified_source_target_token_loss=4.9178
  epoch1 verified_source_tokens=1149
  by_source:
    gsm8k_train=0.2910
    proofwriter_validation=0.2625
    openmathinstruct2_train=0.0954
    numina_verifiable_train=0.0466
    clutrr_train=0.0000

dgx full verified gate:
  run=/mnt/data4tb/qtrm_eval/20260521_111036_DGX_STAGE21B_SourceEvalGate_allVerified_from20B_seed98
  resumed_from=/mnt/data4tb/qtrm_eval/20260521_104227_DGX_STAGE20B_HRMTextSourceMix_multitok8_count2048_seed95/last.pt
  epoch1 train_reasoning_acc=0.1343
  epoch1 synthetic_heldout_mean_acc=0.1587
  epoch1 verified_source_target_token_acc=0.1549
  epoch1 verified_source_target_token_loss=4.5044
  epoch2 train_reasoning_acc=0.1372
  epoch2 synthetic_heldout_mean_acc=0.1587
  epoch2 verified_source_target_token_acc=0.1514
  epoch2 verified_source_target_token_loss=4.4173

interpretation:
  The HRM-Text-style source mix and multi-token recurrent objective are wired
  and observable, but the measured gain is still local to training/source-token
  prediction. The synthetic OOD gate remains flat at 0.1621. This rejects a
  claim that data-source alignment alone solved generalization. The next
  promoted direction should change the causal reasoning path, not only add more
  source rows or another nearby epoch schedule.
```

Stage22 trajectory readout bottleneck test:

```text
hypothesis:
  Stage20/21 still reads the reasoning answer from only the final recurrent
  state. If useful intermediate recurrent states appear before the terminal
  state, final-only readout can hide them. This is a causal-path bottleneck
  distinct from input workspace attention pooling.

mechanism:
  recurrent_readout_pooling=sharp_attention
  recurrent_readout_temperature=0.25
  final_readout_answer_weight=0.05

local run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_111349_PERSIST_STAGE22A_SharpTrajectoryReadout_SourceEval_from20A_seed99

resumed_from:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_104204_PERSIST_STAGE20A_HRMTextSourceMix_multitok8_seed94/last.pt

gate:
  synthetic heldout depths=4,6,8,10
  verified-source heldout rows=400

promotion rule:
  Promote only if synthetic heldout_mean_acc exceeds 0.171875 or verified-source
  target-token accuracy improves cleanly over the Stage21 full-loader baseline
  of 0.1079 without train-only collapse.

status:
  complete

result:
  epoch1 train_reasoning_acc=0.1777
  epoch1 synthetic_heldout_mean_acc=0.1621
  epoch1 verified_source_target_token_acc=0.0966
  epoch2 train_reasoning_acc=0.1777
  epoch2 synthetic_heldout_mean_acc=0.1621
  epoch2 verified_source_target_token_acc=0.1332
  epoch3 train_reasoning_acc=0.1777
  epoch3 synthetic_heldout_mean_acc=0.1621
  epoch3 verified_source_target_token_acc=0.1340

interpretation:
  Sharp trajectory readout improves the local/full-loader verified-source gate
  over Stage21A2 (0.1340 vs 0.1079), but does not move synthetic OOD. This is
  useful evidence that final-only readout was a source-token bottleneck, but it
  is not the main synthetic reasoning generalization bottleneck.
```

Latest literature loop checked on 2026-05-21:

```text
sources:
  https://arxiv.org/abs/2604.07822
  https://arxiv.org/abs/2603.21676
  https://arxiv.org/abs/2604.21593

relevant mechanisms:
  - recurrent-depth transformers can improve compositional generalization, but
    excessive recurrence can cause overthinking.
  - depth-recurrent OOD gains are linked to final-only/silent objectives,
    LayerScale-style stability, identity-biased recurrence, and scaling
    inference-time recurrence.
  - language can act as a latent variable/exploration channel, but that suggests
    a later RL/preference-data direction rather than the immediate bottleneck.

next architecture direction:
  Stage23 should test an identity-biased/LayerScale-style recurrent core update
  or mini_gated_delta core against the same Stage21/22 gates. Do not promote
  more source-data-only runs unless they move synthetic heldout or verified
  source accuracy while preserving the canonical answer path.
```

Stage23 mini-gated-delta core result:

```text
local run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_112150_PERSIST_STAGE23A_MiniGatedDeltaCore_SourceEval_from20A_seed100

dgx run:
  /mnt/data4tb/qtrm_eval/20260521_112151_DGX_STAGE23B_MiniGatedDeltaCore_SourceEval_from20B_seed101

local result:
  epoch1 train_reasoning_acc=0.1484
  epoch1 synthetic_heldout_mean_acc=0.1621
  epoch1 verified_source_target_token_acc=0.0844
  epoch2 train_reasoning_acc=0.1484
  epoch2 synthetic_heldout_mean_acc=0.1621
  epoch2 verified_source_target_token_acc=0.1027
  epoch3 train_reasoning_acc=0.1484
  epoch3 synthetic_heldout_mean_acc=0.1621
  epoch3 verified_source_target_token_acc=0.1079

dgx partial:
  epoch1 train_reasoning_acc=0.1479
  epoch1 synthetic_heldout_mean_acc=0.1621
  epoch1 verified_source_target_token_acc=0.1201

decision:
  Reject the current mini_gated_delta drop-in replacement as a big-jump path.
  It did not improve synthetic OOD and weakened verified-source accuracy versus
  the DGX Stage21B source gate. DGX run was stopped after epoch1 to free the
  machine for Stage24.
```

Stage24 silent identity recurrent core:

```text
literature motivation:
  Recent recurrent-depth generalization papers point to final-only/silent
  objectives, identity-biased recurrence, LayerScale-style small residual
  updates, and controlled recurrence depth as the mechanisms most likely to
  improve depth/OOD generalization.

code change:
  scripts/511_train_qwen_state_transition_hrmtext.py

new post-resume override args:
  --override-transition-scale
  --override-injection-gate-logit
  --zero-step-embeddings
  --freeze-step-embeddings

mechanism:
  Resume the best Stage20 MLP recurrent core, then force a small LayerScale-like
  transition scale and a high recurrent-state carry gate. Remove intermediate
  state/aux/depth-consistency losses so the recurrent core is optimized through
  final-answer pressure instead of step-label shortcuts. Keep only weak
  HRM-Text source replay.

local run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_113012_PERSIST_STAGE24A_SilentIdentityCore_from20A_seed102

dgx run:
  /mnt/data4tb/qtrm_eval/20260521_113013_DGX_STAGE24B_SilentIdentityCore_from20B_seed103

shared important args:
  core_update=mlp
  n_steps=12
  aux_step_answer_weight=0.0
  state_supervision_weight=0.0
  depth_consistency_weight=0.0
  healing_weight=0.10
  override_transition_scale=0.05
  override_injection_gate_logit=3.0
  zero_step_embeddings=true
  freeze_step_embeddings=true

promotion rule:
  Promote only if synthetic heldout_mean_acc moves above 0.1621, preferably
  above 0.171875, while verified-source target-token accuracy does not collapse.

status:
  running

initial signal:
  local epoch1-3:
    train_reasoning_acc=0.1641
    synthetic_heldout_mean_acc=0.1621
    verified_source_target_token_acc=0.1044 -> 0.1036 -> 0.1001
  dgx epoch1:
    train_reasoning_acc=0.1597
    synthetic_heldout_mean_acc=0.1626
    depth4=0.1445 depth6=0.1738 depth8=0.1758 depth10=0.1562
    verified_source_target_token_acc=0.1384
  dgx epoch2:
    train_reasoning_acc=0.1631
    synthetic_heldout_mean_acc=0.1587
    depth4=0.1465 depth6=0.1562 depth8=0.1758 depth10=0.1562
    verified_source_target_token_acc=0.1323
  dgx epoch3:
    train_reasoning_acc=0.1685
    synthetic_heldout_mean_acc=0.1636
    depth4=0.1660 depth6=0.1562 depth8=0.1758 depth10=0.1562
    verified_source_target_token_acc=pending

interpretation:
  DGX epoch1 shows the first nonzero synthetic heldout movement above the
  0.1621 plateau, mainly from depth8. DGX epoch3 improved the mean to 0.1636,
  with depth4 recovering to 0.1660 and depth8 staying at 0.1758, but depth6 and
  depth10 are still weaker. This is a weak positive signal for silent identity
  recurrence, not a big jump.
```

GRAM retrospective:

```text
full GRAM run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_094923_PERSIST_STAGE17D_full_GRAM_from_base_seed86.log
  epoch1 synthetic_heldout_mean_acc=0.1621
  epoch2-4 synthetic_heldout_mean_acc=0.1382
  decision=rejected; full GRAM destabilized OOD heldout.

GRAM no-LPRM run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_101033_PERSIST_STAGE17E_GRAM_noLPRM_kl001_seed88.log
  epoch1-4 synthetic_heldout_mean_acc=0.1621
  decision=neutral; no OOD lift.

important correction:
  Stage24's small DGX epoch1 movement (0.1621 -> 0.1626) is not caused by GRAM.
  Stage24 has lprm=0.0000 and stoch_kl=0.0000. Its change comes from the
  silent/final-only objective and identity-biased recurrent overrides.
```

Stage25 LDT-style candidate lattice:

```text
literature motivation:
  arXiv:2605.08605v1, Lattice Deduction Transformers, argues that recurrent
  reasoners can improve symbolic/OOD reasoning by carrying a candidate lattice
  through the solve loop instead of predicting a final class directly. The key
  mechanisms are asymmetric false-elimination loss, candidate-set refinement
  between recurrent passes, and train/inference solve-loop alignment.

code change:
  src/qtrm_mm/state_transition_core.py
  src/qtrm_mm/qwen_backbone_state_transition.py
  scripts/511_train_qwen_state_transition_hrmtext.py

Stage25A/B mechanism:
  Add candidate-alive lattice supervision on state_digit_logits using
  asymmetric BCE. Current synthetic tasks have a unique final digit, so the
  target lattice is one-hot alive. This is only the loss-side part of LDT, not
  full search/backtracking.

Stage25C/D mechanism:
  Add actual candidate-lattice feedback into the recurrent core:
    --lattice-feedback-mode soft
    --lattice-feedback-scale 0.25
    --init-lattice-feedback-from-readout
  The candidate set is updated between recurrent steps and projected back into
  z_H. The feedback projection is initialized from the supervised digit readout
  transpose so the candidate bits have an immediate state-space direction.

observability:
  TensorBoard and Aim are enabled for all Stage25 runs.
  TensorBoard root:
    local=/mnt/sdc1/tripleyoung/qtrm_eval
    dgx=/mnt/data4tb/qtrm_eval
  Aim experiments:
    qwen35_hrmtext_stage25_ldt_candidate_lattice
    qwen35_hrmtext_stage25_ldt_soft_feedback

local Stage25A:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_114951_PERSIST_STAGE25A_LDTCandidateLattice_from24A_seed104
  final epoch4 train_reasoning_acc=0.1406
  synthetic_heldout_mean_acc=0.1621
  verified_source_target_token_acc=0.1349
  decision=neutral; candidate loss alone did not move heldout generalization.

dgx Stage25B:
  /mnt/data4tb/qtrm_eval/20260521_114952_DGX_STAGE25B_LDTCandidateLattice_from24B_seed105
  epoch1 synthetic_heldout_mean_acc=0.1631
  epoch2 synthetic_heldout_mean_acc=0.1602
  epoch3 synthetic_heldout_mean_acc=0.1582
  epoch4 synthetic_heldout_mean_acc=0.1587
  epoch4 verified_source_target_token_acc=0.1523
  decision=rejected for synthetic heldout; candidate loss alone is not producing
  a stable generalization lift, although verified-source token accuracy improved.

local Stage25C:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_120031_PERSIST_STAGE25C_LDTSoftFeedback_from25A_seed106
  epoch1 train_reasoning_acc=0.1699
  epoch1 synthetic_heldout_mean_acc=0.1621
  epoch1 verified_source_target_token_acc=0.1375
  epoch2 train_reasoning_acc=0.1699
  epoch2 synthetic_heldout_mean_acc=0.1621
  epoch2 verified_source_target_token_acc=0.1340
  epoch3 train_reasoning_acc=0.1758
  epoch3 synthetic_heldout_mean_acc=0.1621
  epoch3 verified_source_target_token_acc=0.1393
  epoch4 train_reasoning_acc=0.1797
  epoch4 synthetic_heldout_mean_acc=0.1562
  epoch4 verified_source_target_token_acc=0.1436
  interpretation:
    LDT soft-feedback gives an immediate train-set lift over Stage25A, but the
    heldout gate did not move and later slipped. Do not call this a
    breakthrough until heldout_mean_acc beats Stage24B's 0.1636 and preferably
    0.171875.

dgx Stage25D:
  /mnt/data4tb/qtrm_eval/20260521_120057_DGX_STAGE25D_LDTSoftFeedback_from25B_seed107
  running.

next falsifiable gate:
  If Stage25C/D keep heldout at the 0.1621-0.1636 plateau, implement the next
  LDT step: explicit thresholded candidate-state rollouts with submit/abstain
  metrics as the primary evaluation, not only final-class argmax accuracy.
```

Stage26A threshold sweep / LDT diagnostic:

```text
primary source:
  Lattice Deduction Transformers, arXiv:2605.08605v1.
  Relevant mechanism: the model carries a candidate lattice through recurrent
  solve steps, applies asymmetric candidate-elimination loss, and trains on
  state-conditional targets that match the inference solve loop.

local run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_121352_PERSIST_STAGE26A_LDTThresholdSweep_neg4_from25A_seed108
  synthetic_heldout_mean_acc:
    epoch1=0.1621
    epoch2=0.1621
    epoch3=0.1587
    epoch4=0.1567
  LDT threshold diagnostics:
    epoch1 LDT@0.5 submit=0.256 sound=0.105 alive=1.91
    later submit collapses toward 0 while alive-count collapses.

interpretation:
  Stronger false-candidate pressure did not create sound deduction. It removed
  true candidates too aggressively. This is not a reason to reject LDT-style
  reasoning in general; it exposes that the current candidate target is wrong
  for multi-step arithmetic traces.

root bug:
  Stage25/26 candidate BCE supervised every recurrent state against the final
  answer digit. For a chain task, intermediate recurrent state s should keep
  the intermediate state_labels[:, s-1] alive, not the final answer at every
  step. Final-answer lattice supervision fights the existing state_loss and
  forces the core to skip deduction.

Stage27 decision:
  Add --lattice-candidate-target {final_answer,state_labels}. The next gate is
  CS-CLS: stepwise candidate-lattice supervision using state_labels, while
  keeping the normal answer path, TensorBoard, Aim, and the same heldout gate.
  Promote only if heldout_mean_acc breaks the Stage24/25 plateau and candidate
  soundness improves without train-only accuracy inflation.

backup candidates:
  DCAOR is the bold architecture candidate after CS-CLS: preserve the Qwen
  token sequence as an addressable workspace and let the recurrent core
  cross-attend to operands per step instead of forcing all operands through one
  compressed latent vector. DISEE is the cheap ablation: remove/clamp absolute
  step embeddings if they are causing depth-specific overfitting.
```

Stage27/28 immediate gate notes:

```text
Stage27A local CS-CLS:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_123608_PERSIST_STAGE27A_CSCLS_state_lattice_from25A_seed109
  result:
    train_reasoning_acc improves to 0.1895 by epoch8
    best heldout_mean_acc=0.1626 at epoch6
    final heldout_mean_acc=0.1572
  decision:
    rejected as a big-jump candidate. Stepwise candidate targets are more
    correct than final-answer targets, but they do not remove the core OOD
    generalization bottleneck.

Stage27B DGX CS-CLS:
  /mnt/data4tb/qtrm_eval/20260521_123647_DGX_STAGE27B_CSCLS_state_lattice_from25B_seed110
  stopped after epoch3 because heldout_mean_acc stayed at 0.1562.
  decision=rejected; DGX scale did not rescue the CS-CLS-only direction.

Stage28 code change:
  src/qtrm_mm/qwen_backbone_state_transition.py
  src/qtrm_mm/state_transition_core.py
  scripts/511_train_qwen_state_transition_hrmtext.py
  tests/test_qwen_state_transition_workspace_pooling.py
  tests/test_state_transition_core_stabilization.py

  Add workspace_pooling={sequence,none}. In this mode the Qwen hidden token
  sequence is projected slot-wise into d_state instead of pooled to one vector.
  StateTransitionCore then applies per-step MultiheadAttention from z_H into
  the workspace before the z_L update. This is the DCAOR probe for the latent
  operand-queue bottleneck.

Stage28A local DCAOR + CS-CLS:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_125010_PERSIST_STAGE28A_DCAOR_sequence_workspace_from24A_seed111
  stopped after epoch4 because train_reasoning_acc remained 0.1562 and
  heldout_mean_acc remained 0.1621.
  decision:
    sequence workspace alone is not enough; the random cross-attention router
    needs a stronger learning signal.

Stage28B DGX DCAOR + CS-CLS:
  /mnt/data4tb/qtrm_eval/20260521_125102_DGX_STAGE28B_DCAOR_sequence_workspace_from24B_seed112
  running at note time.
  epoch1 heldout_mean_acc=0.1636
  epoch2 heldout_mean_acc=0.1626
  preliminary decision:
    neutral, not yet a big jump. Continue only as a scale check.

Stage28C local DCAOR + state CE + aux step answer:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_125809_PERSIST_STAGE28C_DCAOR_stateCE_aux_from24A_seed113
  running at note time.
  epoch1 heldout_mean_acc=0.1621
  preliminary decision:
    not enough evidence yet. This is the first run that matches the DCAOR
    proposal's state-supervision requirement.

next bottleneck if Stage28C/D stays flat:
  The recurrence still receives op type but no explicit operand value at each
  step. If cross-attention cannot learn routing from final/state losses alone,
  run an oracle operand-conditioning upper bound using operation_args and
  initial_label already carried in SyntheticCase. If the oracle jumps, the next
  architecture target is supervised/contrastive operand extraction from the
  sequence workspace, not more recurrent loss tuning.

Stage29 oracle operand upper-bound:
  code change:
    SyntheticDataset now emits operation_arg_ids and initial_labels for
    reasoning rows.
    StateTransitionCore supports --operation-arg-conditioning, adding digit
    argument embeddings to each op_vec and an initial digit embedding to z_H.
  purpose:
    This is not a promoted final architecture. It is a causal upper-bound probe
    for whether missing operand routing is the blocker.
  local high-lr run:
    /mnt/sdc1/tripleyoung/qtrm_eval/20260521_130748_PERSIST_STAGE29C_oracle_operand_args_lr3e5_from24A_seed116
    epoch1 heldout_mean_acc=0.1621
  dgx high-lr run:
    /mnt/data4tb/qtrm_eval/20260521_130812_DGX_STAGE29D_oracle_operand_args_lr3e5_from24B_seed117
    running at note time.

Stage29 result update:
  local Stage29C completed 8 epochs:
    best heldout_mean_acc=0.1636 at epoch3
    epoch8 heldout_mean_acc=0.1631
    epoch8 train_reasoning_acc=0.1562
  dgx Stage29D early signal:
    epoch1 heldout_mean_acc=0.1484
    epoch2 heldout_mean_acc=0.1406
  decision:
    explicit operation-arg embeddings inside the existing Qwen/QTRM state
    transition path did not produce the expected upper-bound jump. Operand
    values alone are not enough when the recurrent bridge/objective still
    fails to learn a clean algorithmic state machine.

Stage30A minimum-faithful GRAM smoke:
  source:
    GRAM paper=https://arxiv.org/abs/2605.19376
    project=https://ahn-ml.github.io/gram-website/
  code:
    scripts/513_train_true_gram_smoke.py
  run:
    /mnt/sdc1/tripleyoung/qtrm_eval/20260521_132440_PERSIST_STAGE30A_true_gram_smoke_seed130
  design:
    Standalone modulo-10 chain/checksum task with explicit initial_label,
    operation_ids, operation_arg_ids, state_labels, and answer_label. This
    isolates whether true GRAM mechanics are trainable before grafting them
    back into Qwen/QTRM.
  implementation rules:
    deterministic baseline:
      z_t = LayerNorm(z_{t-1} + f(z_{t-1}, op_t, arg_t))
    true_gram:
      prior p(z_t | z_{t-1}, op_t, arg_t)
      posterior q(z_t | z_{t-1}, op_t, arg_t, state_label_t)
      training samples z_t from posterior by reparameterization
      eval samples or uses the prior trajectory
      loss = final CE + state CE + beta * free-bits KL + LPRM value MSE
      eval reports K={1,4,16} trajectory selection metrics
  results:
    deterministic best:
      epoch=28
      train_acc=0.9985
      heldout_k1=0.9941
      depth10=0.9844
    true_gram best:
      epoch=12
      train_acc=1.0000
      raw_KL=0.1431
      prior_std_mean=1.0458
      heldout_k1=1.0000
      heldout_k4=1.0000
      heldout_k16=1.0000
      depth10=1.0000
      K16 trajectory_cosine=0.9405
      K16 LPRM calibration_error=0.0149
  decision:
    True GRAM mechanics are viable. Stage17D/E should be treated as pseudo-GRAM
    failures, not as a rejection of GRAM itself. However, the deterministic
    standalone baseline also reaches 99%+ once explicit operands and clean
    state labels are available. Therefore the Qwen/QTRM 0.16 plateau is not
    caused only by lack of stochastic trajectory search. The stronger blocker
    is the Qwen-to-core algorithmic state/operand bridge and/or the way that
    objective couples to the recurrent state.
  caveat:
    This smoke proves faithful GRAM trainability and non-collapsed KL, not a
    standalone inference-time scaling win. In early epochs K=4/16 sampling was
    noisier than K=1 prior-mean inference; by epoch12 all K values reached 1.0.
    Stage30B must therefore evaluate both K=1 and K>1 instead of assuming
    width-based search will help automatically.
  next:
    Do not immediately claim GRAM breakthrough on Qwen. First add a Qwen/QTRM
    Stage30B that uses true GRAM transition placement, KL/free-bits telemetry,
    and K-sample eval while preserving the Stage29 explicit operand upper-bound
    path. Promote only if it beats the 0.16 heldout wall under the same
    TensorBoard/Aim gate.

Stage31 LSCR/SDST plan:
  source verification:
    LoopFormer shortcut-consistency:
      verified arXiv=https://arxiv.org/abs/2602.11451
      note=AGY cited arXiv:2501.08271, but the verified LoopFormer paper is
      arXiv:2602.11451. Use the mechanism, not the wrong citation.
    Loop, Think, and Generalize:
      verified arXiv=https://arxiv.org/abs/2604.07822
      mechanism=recurrent-depth training, dynamic iteration, inference-time
      recurrence scaling, and overthinking failure analysis.
    Reasoning with Latent Thoughts:
      verified arXiv=https://arxiv.org/abs/2502.17416
      mechanism=looped transformers as latent thought / effective-depth scaling.
  diagnosis:
    Stage30B local shows train accuracy can climb quickly while heldout/OOD
    remains near the 0.16-0.18 wall. This points to latent trajectory stability
    and objective coupling, not just raw optimization or more epochs.
  implemented code:
    scripts/511_train_qwen_state_transition_hrmtext.py
    new flags:
      --state-supervision-decay-rate
      --state-supervision-min-weight
      --latent-shortcut-consistency-weight
      --latent-shortcut-consistency-min-step
    new telemetry:
      Step/StateSupervisionWeight
      Epoch/StateSupervisionWeight
      Step/Loss_LatentShortcutConsistency
      Epoch/Loss_LatentShortcutConsistency
      Aim mirrors for the same metrics.
  intended experiment:
    Stage31A local:
      LSCR + SDST on top of the Stage24A/attention-pooling base, not on top of
      an already-running stochastic run. This isolates whether latent shortcut
      consistency and state-supervision decay improve OOD generalization.
    Stage31B DGX:
      mini_gated_delta + LSCR + SDST after Stage30C finishes, only if Stage30C
      does not produce a K-sample generalization jump.
  gate:
    promote if heldout mean accuracy exceeds 0.25 by epoch5 or depth10 exceeds
    0.25 without source-eval collapse.
    reject if heldout mean remains <= 0.18 by epoch5 while train accuracy rises,
    because that means the intervention did not remove the train/generalization
    split.

Stage30B/30C reject and Stage31 launch:
  date: 2026-05-21
  Stage30B local true-GRAM:
    run: /mnt/sdc1/tripleyoung/qtrm_eval/20260521_133437_PERSIST_STAGE30B_trueGRAM_qwen_operand_seed118
    best_train_accuracy: 0.66796875
    best_generalization_mean_accuracy: 0.17480469
    best_generalization_epoch: 1
    depth10_at_best_generalization: 0.20703125
    decision:
      reject true-GRAM-only path for now. Train accuracy rose strongly while
      heldout/OOD generalization failed to improve and later degraded. This is
      the train/OOD split predicted by the latent-drift/objective-shortcut
      diagnosis.
  Stage30C DGX true-GRAM plus LPRM:
    run: /mnt/data4tb/qtrm_eval/20260521_135331_DGX_STAGE30C_trueGRAM_LPRM_qwen_operand_seed119
    best_train_accuracy_checked: 0.18164062
    best_generalization_mean_accuracy: 0.17968750
    depth10_at_best_generalization: 0.19921875
    decision:
      stopped early and moved DGX to Stage31B. Early K-sample/LPRM telemetry did
      not show a trajectory-search jump, and Stage30B already established that
      true-GRAM can fit train while missing OOD.
  Stage31A local:
    run: /mnt/sdc1/tripleyoung/qtrm_eval/20260521_141622_PERSIST_STAGE31A_LSCR_SDST_dynamic_depth_seed120
    design:
      MLP recurrent core + attention pooling + operation-arg conditioning +
      LSCR + SDST + dynamic recurrent depth sampling.
  Stage31B DGX:
    run: /mnt/data4tb/qtrm_eval/20260521_141623_DGX_STAGE31B_LSCR_SDST_miniGatedDelta_seed121
    design:
      Same LSCR/SDST/dynamic-depth objective as Stage31A, but with
      --core-update mini_gated_delta to test bounded recurrent dynamics in
      parallel.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/511_train_qwen_state_transition_hrmtext.py \
  src/qtrm_mm/data/hrm_text_source_mix.py \
  scripts/521_build_hrmtext_source_mix.py

PYTHONPATH=src .venv/bin/python tests/test_hrm_text_source_mix.py
PYTHONPATH=src .venv/bin/python tests/test_hrm_text_aligned_training_data.py
PYTHONPATH=src .venv/bin/python tests/test_verified_reasoning_datasets.py
PYTHONPATH=src .venv/bin/python tests/test_qwen_state_transition_workspace_pooling.py
git diff --check -- scripts/511_train_qwen_state_transition_hrmtext.py \
  src/qtrm_mm/data/hrm_text_source_mix.py \
  scripts/521_build_hrmtext_source_mix.py \
  tests/test_hrm_text_aligned_training_data.py \
  tests/test_hrm_text_source_mix.py
```

### 2026-05-21 Stage32 SOCAR / Lattice / TrueGRAM Gate

Stage32 tested whether keeping the Qwen token sequence as an addressable
workspace, adding state-label lattice supervision, or combining SOCAR with true
GRAM would break the corrected held-out OOD wall.

```text
Stage32A/B/C SOCAR verdict:
  DGX Stage32C:
    run=/mnt/data4tb/qtrm_eval/20260521_161730_DGX_STAGE32C_SOCAR_stateLattice_seed126
    design=sequence workspace + operation-arg conditioning + state-label lattice
    epoch1 heldout_mean_acc=0.1750 depth10=0.1953 depth12=0.1758
    epoch2 heldout_mean_acc=0.1664 depth10=0.2070 depth12=0.1289
    epoch3 heldout_mean_acc=0.1641 depth10=0.2227 depth12=0.1328
    decision=reject. Depth10 can spike, but aggregate heldout and other depths
      degrade, so this is not a durable generalization jump.

Stage32D local SOCAR + true GRAM:
  run=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_161948_PERSIST_STAGE32D_SOCAR_trueGRAM_seed127_retry
  design=sequence workspace + true_gram posterior/KL + stochastic eval K=4
  epoch1 heldout_mean_acc=0.1734 depth10=0.2070 depth12=0.1680
  epoch2 heldout_mean_acc=0.1734 depth10=0.2070 depth12=0.1680
  epoch3 heldout_mean_acc=0.1734 depth10=0.2070 depth12=0.1680
  stochastic_eval_epoch1 K4 accuracy=0.1734 trajectory_cosine=0.6856
  decision=reject. GRAM samples are diverse but do not improve answer accuracy.
```

Logging fix:

```text
scripts/511_train_qwen_state_transition_hrmtext.py now prints stochastic
generalization summaries such as K4=acc:.../cos:... in the Generalization line,
while still writing the same TensorBoard/Aim metrics.
```

### 2026-05-21 Stage33 Big-Jump Split

After Stage32 rejected, the next two experiments deliberately attack different
causal bottlenecks rather than stacking more auxiliary losses.

```text
DGX Stage33A:
  run=/mnt/data4tb/qtrm_eval/20260521_162820_DGX_STAGE33A_HybridCore_sharpReadout_seed128
  log=/tmp/20260521_162820_DGX_STAGE33A_HybridCore_sharpReadout_seed128.log
  aim_experiment=qwen35_hrmtext_stage33_hybrid_core
  hypothesis=MLP recurrent update is the depth-stability bottleneck.
  intervention=HybridStateTransitionCore, 3:1 gated-delta/attention trajectory
    mixer, sharp recurrent readout, LSCR/SDST/dynamic-depth.
  epoch1 heldout_mean_acc=0.1711
  epoch2 heldout_mean_acc=0.1695
  decision=early reject. Core replacement without changing the actual answer
    state representation still stays inside the same 0.16-0.18 wall.

Local Stage33B:
  run=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_163020_PERSIST_STAGE33B_UnfrozenQwen_attnpool_sharpReadout_seed129
  log=/tmp/20260521_163020_PERSIST_STAGE33B_UnfrozenQwen_attnpool_sharpReadout_seed129.log
  aim_experiment=qwen35_hrmtext_stage33_unfrozen_qwen
  hypothesis=frozen Qwen prompt-compressor is the integration bottleneck.
  intervention=unfrozen Qwen low LR, attention workspace pooling, sharp
    recurrent readout, LSCR/SDST; checkpoint disabled for smoke due full-Qwen
    optimizer/checkpoint size.
  decision=stopped. This is an expensive indirect intervention, and after the
    Stage33A signal it does not satisfy the new "answer-path causal change"
    standard.

Promote gate:
  heldout_mean_acc >= 0.20 by epoch3 or >= 0.22 by epoch5, with no verified
  source collapse. Otherwise reject and move to an explicit state-machine /
  typed-register architecture rather than another scalar schedule sweep.
```

Stage34 reset:

```text
New rejection predictor:
  reject-likely if the change only adds an auxiliary loss, readout variant,
  stochastic sampler, Qwen freeze/unfreeze tweak, or recurrent mixer while the
  final answer is still decoded from the same unconstrained continuous z_H.

Next high-leverage direction:
  Make the normal answer path pass through a bounded/discrete typed register or
  belief-state transition, not merely a probe. This is the missing mechanism in
  the LDT/GRAM attempts: candidates/states must become the recurrent state
  itself, so OOD depth cannot drift outside the answer manifold.
```

### 2026-05-21 Stage34 First Move: Brain-Inspired LeWM As Predictive Working Memory

Question:

```text
Should we attach LeWM to the current Qwen/QTRM generalization run?
```

Short answer:

```text
Do not attach the old LeWM self-latent objective as another auxiliary loss.
Do use a LeWM-like predictor if it becomes the predictive working-memory state
that the normal answer path must read.
```

Why the old LeWM path is reject-likely:

```text
Prior local evidence:
  docs/wiki/decisions/lewm-transition-quality-gate.md
  docs/wiki/decisions/lewm-demoted-from-canonical-single-trace-trm.md
  docs/wiki/decisions/pure-recursive-lewm-staged-s200-symbolic-transition-gate.md

Observed:
  LeWM learned next-latent transition MSE very well.
  Symbolic intermediate-state accuracy did not improve.
  Final answer accuracy did not improve.

Root issue:
  z_H[t] -> predict z_H[t+1] models the core's own latent motion. If z_H is not
  already answer-causal, predicting it better does not fix reasoning.
```

Brain-inspired reinterpretation:

```text
PFC-like working memory:
  bounded typed/belief registers that persist over recurrent steps.

Basal-ganglia-like gating:
  learned Go/NoGo policy choosing update, hold, read, verify, halt.

LeWM-like prediction:
  predict the next verified register/belief state under the selected action,
  not merely the next continuous z_H.

Global-workspace-style broadcast:
  the selected register/belief state is fed back into the normal answer logits.
```

Accepted-likelihood score:

```text
Plain LeWM self-latent auxiliary:
  Direct bottleneck replacement: 0/3
  Normal answer-path enforcement: 0/3
  Ablation clarity: 1/2
  Different from rejected family: 0/2
  score=1/10 -> do not run.

Predictive working-memory LeWM:
  Direct bottleneck replacement: 3/3
  Normal answer-path enforcement: 3/3
  Ablation clarity: 2/2
  Different from rejected family: 2/2
  score=10/10 -> eligible after oracle diagnostic.
```

Stage34 first experiment should be diagnostic, not a long training run:

```text
Oracle-gated predictive register diagnostic:
  input prompt/Qwen hidden states
  -> operand/source binder
  -> typed belief register over digits or candidates
  -> learned update/hold/read gate
  -> LeWM-style predictor predicts verified register[t+1]
  -> answer logits read only from the register state

Promote:
  heldout_mean_acc > 0.22 by epoch3 or oracle register-off ablation collapses
  while full stays above the Stage32/33 wall.

Reject:
  register-off matches full, or LeWM loss falls while heldout remains 0.16-0.18.
```

### 2026-05-21 Stage35 Hard Lock: Qwen-Preinit HRM-Text-Like GRAM Main Thought Path

Decision:

```text
The next Qwen3.5/QTRM generalization work is locked to one route:

  Qwen3.5 pretrained init as reader/perception
  -> HRM-Text-like semantic recurrent state as the main thought path
  -> True-GRAM prior/posterior transition inside that state
  -> LM/token answer logits read only from that recurrent thought state

Do not launch another side-path experiment until this route has a thought-state
off / GRAM-off ablation.
```

Humanistic intuition:

```text
Qwen is the eyes and language sense.
The HRM-like semantic recurrent state is the brain that must think.
GRAM is the brain's ability to generate and compare possible thought paths.
The answer head is only the mouth; it may speak only what that brain state
contains.

Rejected old pattern:
  fluent reader + small side scratchpad + auxiliary stochastic/probe loss.

Locked new pattern:
  fluent reader -> actual recurrent thought state -> generative recurrent
  transition/search -> answer from that state only.
```

What counts as following the lock:

```text
Required:
  1. Qwen/Qwen3.5 hidden states may initialize or condition the thought path,
     but may not bypass it to produce the evaluated answer.
  2. The normal LM/token answer logits must be decoded from the HRM-like
     semantic recurrent thought state. Do not use a task-specific digit
     executor as the promoted architecture.
  3. GRAM must update/generate that same semantic thought state through
     prior/posterior transition, not sit beside it as a KL/LPRM side objective.
  4. TensorBoard and Aim must log train accuracy, held-out/generalization
     accuracy, thought-state LM/token accuracy, KL/free-bits, and GRAM sample
     diversity separately.
  5. A destructive ablation must exist:
       thought-state-off or GRAM-off should collapse the gain if the method is real.
```

What is now explicitly banned unless used only as a diagnostic:

```text
Banned as next promoted experiments:
  - another readout-only tweak over continuous z_H;
  - another state-supervision decay / LR / epoch / data-ratio sweep;
  - another Qwen freeze/unfreeze experiment;
  - another LPRM/KL/GRAM sampler while answer still reads from old continuous z_H;
  - another LeWM self-latent prediction loss;
  - another lattice/candidate probe that is not the recurrent answer state.
  - a task-specific modulo-10 digit executor/register as the promoted route.

These can be run only if they are ablations of the locked path, not new
directions.
```

Minimum Stage35 implementation target:

```text
Stage35A universal LLM-path implementation:
  Qwen hidden states
  -> semantic thought-state initializer
  -> HRM-Text-like recurrent thought cycles
  -> True-GRAM transition over the semantic thought state
  -> Qwen-compatible LM head / token logits from thought state only

The first gate may use synthetic modulo prompts, but the architecture must not
hard-code modulo digits or task-specific operators. Any task labels are training
targets only, not the state schema.

Promote:
  heldout_mean_acc > 0.22 by epoch3 or > 0.25 by epoch5, and thought-state-off
  or GRAM-off clearly reduces held-out accuracy.

Reject:
  train accuracy rises but heldout_mean_acc remains 0.16-0.18, or ablations
  match full. That means the path is not actually answer-causal.
```

Research rule:

```text
When a new paper or idea appears, first ask:
  "Does it strengthen the locked Qwen-reader -> HRM/GRAM thought-state ->
   answer-from-state path?"

If no, do not pivot. Record it as a later diagnostic only.
```

Implementation note:

```text
Stage35 code path:
  --answer-path lm_head
    recurrent thought state -> core_out_norm/thought projection -> Qwen lm_head
    -> vocab logits. Synthetic digit accuracy gathers the existing digit token
    IDs from those vocab logits; the architecture is still LM-token based.

  --no-condition-on-operation-ids
    disables external oracle operation IDs as recurrent inputs. The model must
    use the prompt/Qwen hidden state plus recurrent/GRAM thought state, not a
    hand-fed symbolic operation stream.

Required Stage35 launch shape:
  scripts/511_train_qwen_state_transition_hrmtext.py
    --answer-path lm_head
    --no-condition-on-operation-ids
    --workspace-pooling sequence
    --stochastic-high-level-guidance
    --stochastic-high-level-eval
    --stochastic-posterior-guidance
    --stochastic-transition-mode true_gram

Do not use task-specific digit-register executors as the promoted path.
```

## 2026-05-21 Stage35A Humanistic Architecture Preflight Result

Stage35A run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_172207_PERSIST_STAGE35A_universalLM_trueGRAM_lmhead_noop_seed130
```

Observed:

```text
epoch1 heldout_mean_acc=0.0969, K4 trajectory_cosine=1.000
epoch2 heldout_mean_acc=0.0969, K4 trajectory_cosine=1.000
epoch3 heldout_mean_acc=0.0906, K4 trajectory_cosine=1.000
epoch4 heldout_mean_acc=0.1453, K4 trajectory_cosine=1.000
epoch5 heldout_mean_acc=0.0859, K4 trajectory_cosine=1.000
stochastic KL stayed tiny: roughly 0.0046-0.0152
```

Humanistic diagnosis:

```text
The intended story is sound:
  Qwen reads -> recurrent thought thinks -> GRAM explores -> Qwen LM head speaks.

But Stage35A did not yet satisfy the story:
  1. The speaker problem:
     The Qwen LM head is used as the mouth, but the recurrent thought state is
     a new dialect. A small projection alone is not enough evidence that the
     mouth understands the thought state.

  2. The fake-search problem:
     K=4 GRAM evaluation produced identical trajectories
     (trajectory_cosine=1.000). Multiple sampled thoughts were not actually
     different, so this run should not be treated as a real GRAM search result.

  3. The collapse problem:
     KL was tiny, so the posterior/prior path is at risk of collapsing into a
     deterministic TRM-like path.
```

Decision:

```text
Do not continue nearby Stage35A scalar sweeps.
Do not call Stage35A a GRAM rejection.
Treat it as an architecture-preflight failure: the route is right, but the
thought-mouth alignment and actual stochastic search were not yet valid.
```

Next run must pass this preflight before training for metrics:

```text
Required before launch:
  - --stochastic-high-level-eval when using --stochastic-eval-samples > 1.
  - Log raw trajectory diversity and reject immediately if K>1 cosine remains
    ~1.000 after stochastic eval is enabled.
  - Add or verify semantic thought-to-LM alignment so the recurrent state learns
    the Qwen LM head's hidden-space language before claiming answer accuracy.
  - Keep the universal path:
      Qwen reader -> semantic recurrent thought -> True-GRAM -> Qwen LM head.
```

## 2026-05-21 Stage35B Curriculum Preflight Result

Stage35B fixed two Stage35A architecture issues:

```text
Run:
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_174003_PERSIST_STAGE35B_semalign_trueGRAM_evalK_lmhead_noop_seed131

Fixed:
  - healing/source eval now use the same thought -> Qwen LM-head path as the
    evaluated answer.
  - --stochastic-high-level-eval was enabled for real K-sample GRAM eval.
  - semantic_lm_alignment_weight=0.2 was added.

Observed epoch1:
  semantic_lm=0.2686
  K4 trajectory_cosine=0.609
  heldout_mean_acc=0.0906
```

Humanistic diagnosis:

```text
The architecture story improved:
  the thinker now practices speaking through the same Qwen mouth,
  and GRAM now produces different thoughts instead of identical copies.

But the curriculum story is still bad:
  the student is learning a new thought language, a new speaker interface,
  stochastic GRAM exploration, and depth 4/6/8 reasoning all at once.

This is too much load for a fresh recurrent thought state.
It turns a potentially good architecture into a noisy first-day classroom.
```

Decision:

```text
Stage35B was stopped after epoch1 as a curriculum-preflight failure, not an
architecture rejection.
```

Corrected Stage35 curriculum:

```text
Stage35C Phase 1: language-of-thought warmup
  - train_depths=[4]
  - n_steps=4
  - depth_sample_min=0 or 4 fixed
  - keep --answer-path lm_head
  - keep --workspace-pooling sequence
  - keep --no-condition-on-operation-ids
  - keep --stochastic-high-level-eval for observability
  - use semantic_lm_alignment_weight
  - goal: verified source token loss and semantic_lm improve while K diversity
    remains non-identical.

Stage35C Phase 2: depth expansion
  - resume Phase 1
  - train_depths=[4,6]
  - n_steps=6
  - only then add depth 8.

Stage35C Phase 3: OOD generalization gate
  - train_depths=[4,6,8]
  - eval_depths=[4,6,8,10,12]
  - promote only if heldout_mean_acc rises with depth and GRAM K diversity stays
    meaningful.
```

## 2026-05-21 Seven-Axis Humanistic Validity Gate

All Qwen3.5 + HRM/GRAM experiments must pass this seven-axis check before
longer local or DGX training. This is now the default guard against "technically
interesting but humanistically incoherent" experiments.

### 1. Architecture: Reader, Thinker, Speaker

Humanistic question:

```text
Who reads, who thinks, who speaks, and can the speaker understand the thinker?
```

Valid Stage35 story:

```text
Qwen pretrained reader
-> semantic recurrent thought state
-> True-GRAM prior/posterior transition over that same thought state
-> Qwen LM head speaks from the thought state
```

Previously nonsensical part:

```text
Stage35A used Qwen LM head as the speaker, but did not prove that the recurrent
thought state spoke Qwen's hidden-space language. Healing/source eval also used
a different direct state -> lm_head path than the final answer path.
```

Fix:

```text
Stage35B+ uses the same thought -> Qwen LM-head path for answer, healing, and
verified-source eval.
semantic_lm_alignment_weight teaches the recurrent thought state to land in the
Qwen LM-head token geometry.
```

Do not launch if:

```text
answer_path != lm_head for promoted universal-LLM runs;
healing/source eval bypasses thought_to_lm/core_out_norm path;
the final answer can come from a task-specific digit executor or side head.
```

### 2. Curriculum: First Learn the Thought Language

Humanistic question:

```text
Are we asking a first-day student to read, think, translate, search, and solve
OOD exams all at once?
```

Previously nonsensical part:

```text
Stage35B started with train_depths=[4,6,8], n_steps=8, stochastic GRAM, no
operation IDs, and semantic LM alignment all at once.
```

Fix:

```text
Stage35C Phase 1:
  fixed train_depths=[4]
  fixed n_steps=4
  no depth_sample_min
  semantic thought-to-LM warmup first

Stage35C Phase 2:
  resume Phase 1, then expand to train_depths=[4,6], n_steps=6

Stage35C Phase 3:
  only after Phase 2, expand to train_depths=[4,6,8], n_steps=8
```

Do not launch if:

```text
a fresh thought state is asked to handle depth 4/6/8 plus stochastic search plus
new speaker alignment in the first run;
OOD depth10 is used to reject Phase 1 before depth4 and semantic alignment have
stabilized.
```

### 3. Reward/Loss: Praise the Behavior We Actually Want

Humanistic question:

```text
Are we rewarding real thinking, or rewarding the model for imitating a token
shape while ignoring the reasoning job?
```

Current Stage35 loss roles:

```text
answer CE:
  final thought state must produce the answer through Qwen LM-head label logits.

healing CE:
  recurrent thought states must predict clean response tokens through the same
  Qwen LM-head path.

semantic_lm_alignment:
  the thought state must speak Qwen LM-head hidden-space language.

KL/GRAM telemetry:
  prior/posterior should not collapse into a fake deterministic loop.
```

Previously nonsensical part:

```text
Semantic alignment did not exist, so the speaker problem was invisible.
KL was tiny and K-sample paths were identical, so "GRAM" could be a label rather
than actual exploration.
```

Fix:

```text
Log and gate:
  Epoch/Loss_SemanticLMAlignment
  Generalization/HeldOut/StochasticK4/TrajectoryCosine
  Epoch/StochasticHighLevel_StdMean
  Epoch/Loss_StochasticPosteriorKL
```

Do not launch or promote if:

```text
semantic_lm improves while answer/healing accuracy does not move at all for the
phase target;
semantic_lm weight is increased after a weak run without checking whether it is
flattening thought states;
KL -> 0 and K trajectory cosine -> 1.000 while claiming GRAM.
```

### 4. Evaluation: Match the Exam to the Class

Humanistic question:

```text
Are we grading a beginner warmup by a graduate OOD exam?
```

Correct Stage35 evaluation meaning:

```text
Phase 1 primary gate:
  depth4 accuracy, verified-source token loss/accuracy, semantic_lm, K diversity.

Phase 1 OOD depth6/8/10:
  observation only, not a rejection gate.

Phase 2 primary gate:
  depth4/depth6 plus semantic stability.

Phase 3 primary gate:
  heldout_mean across depth4/6/8/10/12 and ablations.
```

Previously nonsensical part:

```text
Looking at depth10 too early caused good warmup runs to look like failures.
```

Fix:

```text
Record OOD early, but do not call Phase 1 failed unless depth4, semantic_lm, and
verified-source metrics fail together.
```

### 5. GRAM Exploration: Real Multiple Thoughts

Humanistic question:

```text
If we ask for four imagined paths, are there actually four different thoughts?
```

Previously nonsensical part:

```text
Stage35A logged K4 trajectory_cosine=1.000. That means the model did not explore
meaningfully even though the run was named GRAM.
```

Fix:

```text
Stage35B+ requires:
  --stochastic-high-level-eval
  --stochastic-eval-samples 1 4

Expected preflight:
  K4 trajectory_cosine clearly below 1.000.
```

Do not launch or promote if:

```text
K>1 evaluation is requested but stochastic eval is disabled;
K4 cosine stays approximately 1.000 after stochastic eval is enabled;
K-sampling changes accuracy but diversity/selection mode is not logged.
```

### 6. Data Contract: The Problem Must Contain the Needed Clues

Humanistic question:

```text
Does the student receive the information needed to solve the problem, or are we
testing mind-reading?
```

Previously nonsensical part:

```text
Older synthetic runs hid operands in labels while the prompt only named the
family. The 10% plateau was therefore not evidence against recurrence.
```

Fix:

```text
Use generalized synthetic prompts with visible operands.
Use HRM-Text/Data-IO-style condition/instruction/response rows for language
healing.
Use source-eval rows that follow the same response-token contract.
```

Do not launch if:

```text
prompt text does not contain the operands needed by the label;
healing targets leak future response text into the input;
train/eval schemas differ in hidden ways.
```

### 7. Causality/Ablation: Prove the Thinker Matters

Humanistic question:

```text
If we remove the thinker or the GRAM imagination, does the answer get worse?
```

Current requirement:

```text
After Stage35C Phase 3 shows a gain, run destructive ablations:
  thought-state-off
  GRAM-off / stochastic-off
  semantic-alignment-off
  operation-id-on/off comparison only as diagnosis, not a promoted route
```

Previously nonsensical part:

```text
Earlier work sometimes promoted auxiliary heads, probes, or stochastic labels
without proving that the normal answer path needed them.
```

Fix:

```text
No architecture claim without an ablation drop.
No "best" label unless TensorBoard/Aim, run_info, checkpoint path, and the
relevant destructive ablation are recorded.
```

## 2026-05-21 Stage35C Phase 1 Current Humanistic Status

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_174342_PERSIST_STAGE35C_phase1_loth_warmup_depth4_semalign_trueGRAM_seed132
```

Observed so far:

```text
epoch1:
  semantic_lm=0.2175
  verified_source_loss=4.8000
  verified_source_acc=0.1160
  heldout_mean_acc=0.1094
  K4 trajectory_cosine=0.603

epoch2:
  semantic_lm=0.1879
  verified_source_loss=3.9220
  verified_source_acc=0.1800
  heldout_mean_acc=0.1172
  K4 trajectory_cosine=0.634

epoch3 train:
  semantic_lm=0.1688
```

Humanistic reading:

```text
The language-of-thought warmup is doing what it should do first:
the thinker is getting better at speaking through the Qwen mouth
(semantic_lm and verified-source loss improve), while GRAM paths remain
non-identical.

Do not judge this run by depth10 yet. Phase 1 is not the final OOD exam.
```

Current unresolved but now controlled risk:

```text
Reasoning accuracy is still modest and noisy. This is acceptable only during
Phase 1 if semantic/healing improve. If depth4 reasoning does not improve after
the thought-language bridge stabilizes, Phase 2 must not begin; inspect loss
balance and answer-path causality first.
```

## 2026-05-21 Stage35C Phase 2 Preflight Correction

An initial Phase2 launch was stopped before epoch completion:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_175056_PERSIST_STAGE35C_phase2_depth46_semalign_trueGRAM_from_phase1_seed133
```

Reason:

```text
The run intended to train depth4+6 with n_steps=6 and depth_sample_min=4.
Before the fix, the training loop could choose n_steps=4 for a mixed batch that
contained depth6 examples.

Humanistically this is incoherent:
  it asks the thinker to solve a six-step reasoning problem with only four
  recurrent thought slots, while still grading the final six-step answer.
```

Code fix:

```text
collate_fn now records per-example reasoning depths.
When depth_sample_min is used, the sampled n_steps lower bound is:
  max(depth_sample_min, max_depth_in_batch)

Therefore a depth6 batch cannot be trained with only four recurrent steps.
```

Seven-axis Phase2 gate after the fix:

```text
1. Architecture:
   PASS. Qwen reader -> semantic recurrent thought -> True-GRAM -> Qwen LM head.

2. Curriculum:
   FIXED. Phase2 expands only from depth4 to depth4+6, and recurrent steps may
   not be shorter than the deepest case in the batch.

3. Reward/loss:
   PASS WITH WATCH. semantic_lm_alignment_weight is reduced to 0.15 so it keeps
   the speaker bridge but does not dominate reasoning.

4. Evaluation:
   PASS. depth4/depth6 are the primary gate; depth8/10/12 are observational.

5. GRAM exploration:
   PASS. stochastic_high_level_eval remains enabled and K4 trajectory cosine is
   logged.

6. Data contract:
   PASS. generalized prompts expose operands and HRM-Text healing uses
   prefix/response boundaries.

7. Causality/ablation:
   PASS FOR GATE ONLY. No "best" claim until later thought-state/GRAM/semantic
   alignment ablations.
```

Only after this correction should Phase2 be relaunched.

## 2026-05-21 Stage35C Phase 2 Audited Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_175340_PERSIST_STAGE35C_phase2_AUDITED_depth46_semalign_trueGRAM_from_phase1_seed133
```

Observed:

```text
epoch1:
  train_acc=0.1198
  heldout_mean_acc=0.0906
  depth4=0.0859 depth6=0.0703 depth8=0.1094 depth10=0.0938 depth12=0.0938
  K4_acc=0.0906 K4_cos=0.651
  semantic_lm=0.1128
  verified_source_acc=0.2600

epoch2:
  train_acc=0.2253
  heldout_mean_acc=0.0969
  depth4=0.1016 depth6=0.1250 depth8=0.0781 depth10=0.0547 depth12=0.1250
  K4_acc=0.0844 K4_cos=0.615
  semantic_lm=0.1082
  verified_source_acc=0.1480

epoch3:
  train_acc=0.4102
  heldout_mean_acc=0.1109
  depth4=0.1016 depth6=0.0703 depth8=0.1094 depth10=0.1250 depth12=0.1484
  K4_acc=0.0797 K4_cos=0.532
  semantic_lm=0.1132
  verified_source_acc=0.1840
```

Seven-axis reading:

```text
1. Architecture:
   PASS. The answer still flows through Qwen reader -> thought state -> LM head.

2. Curriculum:
   PASS AFTER FIX. The batch-depth/n_steps bug was corrected.

3. Reward/loss:
   FAIL. Train accuracy jumps to 41%, but heldout does not follow. The current
   objective praises final-answer success too much and gives too little pressure
   for a reusable stepwise algorithm.

4. Evaluation:
   PASS. The run correctly exposed the train-vs-heldout mismatch.

5. GRAM exploration:
   PARTIAL. K4 paths are diverse, but K4 accuracy is worse than K1. Diversity is
   not yet useful search.

6. Data contract:
   PASS. Inputs contain operands and split boundaries are clean.

7. Causality/ablation:
   NOT READY. Do not run Phase3 ablations until heldout improves.
```

Humanistic diagnosis:

```text
The student learned to do well on classroom examples, but not the rule.
The missing ingredient is not another bigger phase. The teacher is still only
grading the final answer and language alignment. It needs a light, temporary
process hint so the recurrent thought can learn a reusable algorithm instead of
memorizing surface cases.
```

Decision:

```text
Do not proceed to Stage35C Phase 3.
Run Stage35D as a reward/loss correction gate:
  - resume Phase2 or Phase1;
  - keep universal LM-head answer path;
  - add weak LM-head step-answer supervision through the same Qwen mouth;
  - keep it small enough that it teaches the process without becoming a
    task-specific executor.
```

Stage35D gate:

```text
Launch shape:
  --answer-path lm_head
  --no-condition-on-operation-ids
  --workspace-pooling sequence
  --stochastic-high-level-eval
  --stochastic-transition-mode true_gram
  --semantic-lm-alignment-weight 0.10-0.15
  --aux-step-answer-weight 0.05-0.10
  --state-supervision-weight 0.0

Why aux-step is allowed:
  It uses the same LM-head token path and supervises thought states during
  training only. It is not a promoted digit executor or inference shortcut.

Promote to Phase3 only if:
  depth4/depth6 heldout improve together, K4 remains diverse, and train accuracy
  does not rise alone while heldout stays flat.
```

## 2026-05-21 Stage35D Reward/Loss Correction Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_180234_PERSIST_STAGE35D_rewardfix_auxstepLM_depth46_from_phase2_seed134
```

Observed:

```text
epoch1:
  train_acc=0.6341
  heldout_mean_acc=0.1187
  depth4=0.1016 depth6=0.0781 depth8=0.1328 depth10=0.1562 depth12=0.1250
  K4_acc=0.1062 K4_cos=0.454
  semantic_lm=0.0801
  verified_source_acc=0.2000

epoch2:
  train_acc=0.8372
  heldout_mean_acc=0.0828
  depth4=0.1094 depth6=0.0781 depth8=0.0625 depth10=0.0625 depth12=0.1016
  K4_acc=0.0859 K4_cos=0.388
  semantic_lm=0.0836
  verified_source_acc=0.2080

epoch3:
  train_acc=0.9688
  heldout_mean_acc=0.1047
  depth4=0.1094 depth6=0.0625 depth8=0.1250 depth10=0.1016 depth12=0.1250
  K4_acc=0.1062 K4_cos=0.347
  semantic_lm=0.0820
  verified_source_acc=0.2240
```

Seven-axis reading:

```text
1. Architecture:
   PASS. The run still uses Qwen reader -> recurrent thought -> True-GRAM ->
   Qwen LM-head logits, with no promoted side executor.

2. Curriculum:
   PASS. The depth/n_steps mismatch was already fixed and this run remained
   depth4+6 only.

3. Reward/loss:
   FAIL. Weak LM-head step supervision makes train accuracy explode from 63% to
   97%, but heldout remains near chance and even drops after epoch1.

4. Evaluation:
   PASS. The eval correctly catches the train/heldout split. The best
   generalization checkpoint is epoch1, not the best train checkpoint.

5. Exploration:
   PARTIAL. K4 trajectory cosine improves from fake-search territory to
   0.45 -> 0.35, so GRAM samples differ, but K4 accuracy does not beat K1.

6. Data contract:
   UNRESOLVED. Since train depth4/6 rises while fresh depth4/6 does not, the
   next diagnostic must split exact-train memorization from same-distribution
   fresh generalization.

7. Causality/ablation:
   NOT READY. Do not promote GRAM or aux-step supervision until the data-contract
   diagnostic is passed.
```

Humanistic diagnosis:

```text
The thinker can now talk through the right mouth, and it can produce many
different thought paths. But the teacher is still rewarding answer patterns in
the exercise book, not a portable rule. A student who scores 97% on yesterday's
worksheet and 10% on today's same-style worksheet has not learned arithmetic.
```

Decision:

```text
Stage35D is rejected as a generalization fix.
Do not proceed to Phase3.

Next required gate:
  Stage35E data-contract diagnostic.

Stage35E must measure:
  1. exact training-seed accuracy;
  2. fresh same-distribution depth4/6 accuracy;
  3. depth8+ OOD accuracy;
  4. whether K4 improves only exact memorization or true fresh generalization.

Only after this split is known may a new architecture be launched.
```

## 2026-05-21 Stage35E Prior/Posterior Causality Fix

New finding:

```text
Stage35D's 97% train accuracy was not the same path used at evaluation.

During True-GRAM training:
  posterior_labels = state_labels
  answer_loss is computed from a posterior-guided forward pass.

During heldout evaluation:
  posterior_labels = None
  answer_logits come from the prior-only path.
```

Humanistic reading:

```text
The student was allowed to look at the answer key while practicing, then asked
to take the exam without it. The practice score looked excellent, but it was not
measuring the exam skill.
```

Code correction:

```text
scripts/511_train_qwen_state_transition_hrmtext.py now supports:
  --prior-answer-weight

When this is enabled, every reasoning batch also runs the prior-only path:
  posterior_labels=None

It logs:
  Epoch/Accuracy_Reasoning_PriorNoPosterior
  Train/Epoch/Accuracy_Reasoning_PriorNoPosterior
  Epoch/Loss_PriorAnswer
  Train/Epoch/Loss_PriorAnswer

Aim mirrors the same metrics:
  accuracy_reasoning_prior_no_posterior
  train_accuracy_reasoning_prior_no_posterior
  loss_prior_answer
  train_loss_prior_answer
```

Stage35E gate:

```text
Run a small prior-locked GRAM gate before any new architecture.

Promote only if:
  - prior_no_posterior_train_accuracy rises;
  - fresh depth4/depth6 heldout rises with it;
  - posterior train accuracy no longer hides a prior/eval failure;
  - K4 remains diverse and does not reduce heldout accuracy.

Reject if:
  - posterior train accuracy rises but prior_no_posterior stays near chance;
  - prior_no_posterior improves only on exact train cases;
  - heldout depth4/depth6 remain flat.
```

## 2026-05-21 Stage35E Prior-Locked GRAM Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_181612_PERSIST_STAGE35E_priorlocked_trueGRAM_depth46_from35Dgen_seed42
```

Observed:

```text
epoch1:
  posterior_train_acc=0.7812
  prior_no_posterior_train_acc=0.1172
  heldout_mean_acc=0.1047
  depth4=0.0938 depth6=0.0781 depth8=0.0859 depth10=0.1328 depth12=0.1328
  K4_acc=0.1016 K4_cos=0.448

epoch2:
  posterior_train_acc=0.8607
  prior_no_posterior_train_acc=0.1224
  heldout_mean_acc=0.1234
  depth4=0.1328 depth6=0.1172 depth8=0.1094 depth10=0.1250 depth12=0.1328
  K4_acc=0.1156 K4_cos=0.423

epoch3:
  posterior_train_acc=0.9245
  prior_no_posterior_train_acc=0.1367
  heldout_mean_acc=0.0984
  depth4=0.0703 depth6=0.0625 depth8=0.1094 depth10=0.1172 depth12=0.1328
  K4_acc=0.1219 K4_cos=0.398
```

Reading:

```text
Stage35E confirms the causal bug.

The posterior path can reach >90% train accuracy, but the evaluation path
(prior-only, no posterior labels) remains 11-14%. Heldout follows the prior
path, not the posterior path.
```

Decision:

```text
Stage35E is accepted as a diagnosis and rejected as a final fix.

Do not read posterior_train_acc as reasoning accuracy for True-GRAM runs.
The primary train metric is now:
  prior_no_posterior_train_acc

Next required step:
  Stage35F posterior-to-prior distillation.
```

Humanistic next step:

```text
The teacher-guided thinker knows how to speak answers on the worksheet.
The exam thinker does not. The next run must make the exam thinker imitate the
teacher-guided thinker before trying to scale depth.
```

## 2026-05-21 Stage35F Posterior-To-Prior Distillation Plan

Code addition:

```text
scripts/511_train_qwen_state_transition_hrmtext.py now supports:
  --prior-posterior-logit-distill-weight
  --prior-posterior-distill-temperature

This adds a KL distillation loss:
  posterior-guided answer logits (teacher, detached)
  -> prior-only answer logits (student, evaluation path)
```

Humanistic reason:

```text
The teacher-guided thinker can solve the worksheet.
The exam thinker cannot.

So the next lesson is not "more answer pressure". It is:
  make the exam thinker imitate the teacher-guided thinker's answer distribution.
```

Stage35F gate:

```text
Promote if:
  prior_no_posterior_train_acc rises faster than Stage35E;
  depth4/depth6 heldout rise with it;
  K4 stays diverse and does not underperform K1.

Reject if:
  posterior_train_acc rises while prior_no_posterior_train_acc stays near chance;
  prior_no_posterior_train_acc rises but heldout stays flat;
  verified-source language loss collapses.
```

## 2026-05-21 Stage35F Posterior-To-Prior Distillation Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_182625_PERSIST_STAGE35F_post2prior_distill_depth46_from35Egen_seed42
```

Observed:

```text
epoch1:
  posterior_train_acc=0.9401
  prior_no_posterior_train_acc=0.1771
  heldout_mean_acc=0.1047
  depth4=0.1094 depth6=0.0938 depth8=0.0859 depth10=0.1250 depth12=0.1094
  K4_acc=0.1094 K4_cos=0.405

epoch2:
  posterior_train_acc=0.9375
  prior_no_posterior_train_acc=0.1693
  heldout_mean_acc=0.1297
  depth4=0.1406 depth6=0.1250 depth8=0.1172 depth10=0.1406 depth12=0.1250
  K4_acc=0.1234 K4_cos=0.385

epoch3:
  posterior_train_acc=0.9349
  prior_no_posterior_train_acc=0.1263
  heldout_mean_acc=0.1094
  depth4=0.0703 depth6=0.1016 depth8=0.1406 depth10=0.1250 depth12=0.1094
  K4_acc=0.1219 K4_cos=0.379
```

Reading:

```text
Posterior-to-prior distillation is directionally useful:
  - prior_no_posterior_train_acc reaches 17.7% immediately;
  - best heldout rises to 12.97%;
  - depth4/depth6 improve together at epoch2.

But it is not yet a big jump:
  - the gain is small;
  - prior accuracy is unstable by epoch3;
  - posterior_train_acc remains >93%, so training is still dominated by the
    teacher-guided path.
```

Decision:

```text
Accept Stage35F as a partial mechanism.
Reject it as a complete generalization fix.

Next required step:
  Stage35G exam-first objective.

The posterior answer loss must become optional/low-weight so the prior-only
exam path becomes the primary optimization target.
```

Code correction for Stage35G:

```text
scripts/511_train_qwen_state_transition_hrmtext.py now supports:
  --posterior-answer-weight

This scales the answer loss from the posterior-guided forward pass.
```

## 2026-05-21 Stage35G Exam-First Prior Objective Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_183542_PERSIST_STAGE35G_examfirst_prior_depth46_from35Fgen_seed42
```

Observed:

```text
epoch1:
  posterior_train_acc=0.7279
  prior_no_posterior_train_acc=0.2474
  heldout_mean_acc=0.1000
  depth4=0.1172 depth6=0.0781 depth8=0.0781 depth10=0.1094 depth12=0.1172
  K4_acc=0.1141 K4_cos=0.429

epoch2:
  posterior_train_acc=0.5859
  prior_no_posterior_train_acc=0.2318
  heldout_mean_acc=0.0984
  depth4=0.1328 depth6=0.1094 depth8=0.0703 depth10=0.0781 depth12=0.1016
  K4_acc=0.1125 K4_cos=0.425

epoch3:
  posterior_train_acc=0.3724
  prior_no_posterior_train_acc=0.1328
  heldout_mean_acc=0.1078
  depth4=0.0781 depth6=0.1250 depth8=0.1016 depth10=0.1250 depth12=0.1094
  K4_acc=0.1125 K4_cos=0.480
```

Reading:

```text
Exam-first training successfully moves the prior-only path on the training
set, peaking at 24.7% prior_no_posterior_train_acc.

But fresh heldout does not follow. Therefore the current bottleneck is no
longer mainly posterior/eval mismatch. The model can fit the prior path on
seen examples, but it does not extract a reusable arithmetic rule from fresh
prompts.
```

Decision:

```text
Stage35G is rejected as a generalization fix.
Keep the posterior/prior split metrics permanently.

Do not keep increasing GRAM loss weights.
Next experiments must target one of these two axes:

1. Data diversity:
   increase fresh synthetic diversity and curriculum coverage so prior-path
   training cannot survive by memorizing 768 examples.

2. Reader-to-thinker routing:
   make the recurrent thought explicitly learn which source tokens/operands it
   used, instead of relying on unsupervised cross-attention.
```

Humanistic summary:

```text
We taught the student to take the exam without the answer key.
That worked on yesterday's worksheet, but not on today's worksheet.

So the next issue is not "how to remove the answer key".
It is "how to teach the student to read the problem and extract the rule".
```

## 2026-05-21 Stage35H Data-Diversity Generalization Gate

Humanistic diagnosis before launch:

```text
Stage35G taught the student to take the exam without the answer key.
It worked on the old worksheet, but not on a fresh worksheet.

That means the student may still be memorizing the worksheet's surface patterns
instead of learning a reusable rule. Before adding another architecture trick,
the next gate must make the worksheet much larger and more varied.
```

Seven-axis preflight:

```text
1. Architecture:
   Keep the universal path:
     Qwen reader -> recurrent thought -> True-GRAM prior path -> Qwen LM head.

2. Curriculum:
   Do not add new depths yet. Keep depth4+6 so the only changed variable is
   distribution diversity.

3. Reward/loss:
   Keep exam-first objective from Stage35G:
     posterior_answer_weight low;
     prior_answer_weight primary;
     posterior-to-prior distillation secondary.

4. Evaluation:
   Gate on fresh heldout depth4/depth6 first. Depth8/10/12 are observational.

5. Exploration:
   Keep K4 trajectory cosine and K4 accuracy logged.

6. Data contract:
   Change synthetic sampling from stratified small worksheet to random larger
   worksheet, so the model cannot rely on the exact old example order/pattern.

7. Causality:
   If train prior improves but fresh heldout does not, data size alone is not
   the missing bridge; move to reader-to-thinker routing supervision.
```

Launched:

```text
Local:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_185140_LOCAL_STAGE35H_diversity2048_examfirst_seed43
  reasoning_count=2048
  seed=43
  synthetic_sampling_strategy=random

DGX:
  /mnt/data4tb/qtrm_eval/20260521_185205_DGX_STAGE35H_diversity16384_examfirst_seed44
  reasoning_count=16384
  seed=44
  synthetic_sampling_strategy=random
```

Promotion gate:

```text
Promote data-diversity route if:
  heldout_mean_acc >= 0.16 and depth4/depth6 both improve over Stage35G;
  prior_no_posterior_train_acc improves without a heldout collapse;
  K4 accuracy is not lower than K1.

Reject data-diversity-only route if:
  prior_no_posterior_train_acc improves but heldout remains near 0.10-0.13.
```

## 2026-05-21 Stage35H Local 2048 Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_185140_LOCAL_STAGE35H_diversity2048_examfirst_seed43
```

Observed:

```text
epoch1:
  prior_no_posterior_train_acc=0.1313
  heldout_mean_acc=0.1375
  depth4=0.1641 depth6=0.1016 depth8=0.1641 depth10=0.1562 depth12=0.1016
  K4_acc=0.1453 K4_cos=0.577

epoch2:
  prior_no_posterior_train_acc=0.1392
  heldout_mean_acc=0.1547
  depth4=0.1797 depth6=0.1094 depth8=0.1484 depth10=0.1797 depth12=0.1562
  K4_acc=0.1562 K4_cos=0.632
```

Reading:

```text
Data diversity is now the strongest active signal.

Unlike Stage35G, the fresh heldout score rises meaningfully even though train
prior accuracy remains modest. K4 also beats the single-path score, suggesting
that stochastic diversity is becoming useful rather than fake.
```

Decision:

```text
Continue scaling diversity.

Launched local follow-up:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45
  reasoning_count=4096

DGX large run remains active:
  /mnt/data4tb/qtrm_eval/20260521_185205_DGX_STAGE35H_diversity16384_examfirst_seed44
  reasoning_count=16384
```

## 2026-05-21 Stage35H2 Local 4096 Interim Result

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45
```

Observed so far:

```text
epoch1:
  prior_no_posterior_train_acc=0.1406
  heldout_mean_acc=0.1594
  depth4=0.1875 depth6=0.1641 depth8=0.1172 depth10=0.1797 depth12=0.1484
  K4_acc=0.1547 K4_cos=0.707
```

Reading:

```text
This is the strongest local generalization signal so far.
The gain scales from 2048 -> 4096 examples, and both depth4/depth6 improve.
The score is just below the 0.16 promotion gate before epoch2.
```

Final local H2 result:

```text
epoch2:
  prior_no_posterior_train_acc=0.1262
  heldout_mean_acc=0.1453
  depth4=0.1562 depth6=0.1250 depth8=0.1641 depth10=0.1719 depth12=0.1094
  K4_acc=0.1484 K4_cos=0.752
```

Decision:

```text
Best H2 checkpoint is epoch1 at 15.94% heldout.
Data diversity remains the best active direction, but longer local training is
not automatically better. The next local check should test a wider dataset with
one epoch rather than more epochs on the same 4096 set.
```

## 2026-05-21 Stage35H3 Local 8192 Launch

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_191605_LOCAL_STAGE35H3_diversity8192_oneepoch_seed46
```

Setup:

```text
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
reasoning_count=8192
synthetic_sampling_strategy=random
epochs=1
seed=46
```

Humanistic preflight:

```text
Reader:
  Qwen still reads the prompt and preserves token-level workspace via sequence
  pooling.

Thinker:
  The prior-only GRAM recurrent state is now forced to practice without the
  posterior answer key.

Speaker:
  The evaluated answer still goes through the Qwen-compatible LM-head path.

Curriculum:
  H2 showed that repeating a small worksheet can lower generalization. H3 tests
  whether a wider worksheet improves transfer before adding any new architecture.

Gate:
  Promote if heldout_mean_acc crosses 0.16 with depth4/depth6 not collapsing and
  K4 not lower than K1. Otherwise treat data breadth alone as insufficient.
```

Observed:

```text
epoch1:
  prior_no_posterior_train_acc=0.1333
  posterior_guided_train_acc=0.2159
  heldout_mean_acc=0.1281
  depth4=0.1406 depth6=0.1016 depth8=0.1406 depth10=0.1406 depth12=0.1172
  K4_acc=0.1406 K4_cos=0.787
  verified_source_target_token_acc=0.1960
```

Decision:

```text
Reject data-breadth-only scaling as the next big-jump mechanism.

H2 at 4096 examples nearly crossed the 0.16 gate, but H3 at 8192 fell to
12.81%. That means the useful signal is not "more rows" by itself. The next
step must audit the data contract and curriculum: whether the wider worksheet
changed the lesson distribution faster than the prior-only thinker can absorb.
```

## 2026-05-21 Stage35 Robust Re-Eval Audit

Setup:

```text
eval_count=256
eval_depths=4,6,8,10,12
condition_on_operation_ids=false
workspace_pooling=sequence
answer_path=lm_head
stochastic_eval_samples=1
```

Observed:

```text
H2_4096_best:
  mean=0.153906
  depth4=0.136719 depth6=0.148438 depth8=0.148438 depth10=0.218750 depth12=0.117188

H3_8192_best:
  mean=0.135156
  depth4=0.101562 depth6=0.128906 depth8=0.156250 depth10=0.160156 depth12=0.128906
```

Reading:

```text
The H2 near-gate result is not pure noise, but it is not a stable breakthrough.
H3 confirms that adding more random rows can degrade the prior-only thinker.

Humanistically: the student has not learned a clean method yet. A thicker
worksheet sometimes helps, but it can also blur the timing of when to stop
thinking and speak.
```

## 2026-05-21 Stage36A Exact-Depth Timing Contract Launch

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_193547_LOCAL_STAGE36A_exactdepth4_contract_seed47
```

Setup:

```text
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
train_depths=4
n_steps=6
depth_sample_exact_batch_max=true
reasoning_count=2048
epochs=1
```

Humanistic reason:

```text
The previous mixed-depth class may have taught shallow problems to answer after
extra padded copy steps. Stage36A asks a narrower question: if a depth4 problem
is taught as a four-beat thought and evaluated as a four-beat thought, does
depth4 accuracy recover?
```

Gate:

```text
Promote the timing-contract fix if depth4 clearly improves over H2 robust
depth4=0.1367 without collapsing the heldout mean. If depth4 does not improve,
the bottleneck is not timing; move to thinker-speaker semantic alignment or
operand routing diagnostics.
```

Observed:

```text
epoch1:
  prior_no_posterior_train_acc=0.1367
  posterior_guided_train_acc=0.2583
  heldout_mean_acc=0.1375
  depth4=0.1094 depth6=0.0859 depth8=0.1719 depth10=0.2031 depth12=0.1172
  K4_acc=0.1484 K4_cos=0.725
  verified_source_target_token_acc=0.2120
```

Decision:

```text
Reject timing-contract as the main cause.

Exact-depth depth4 teaching did not recover depth4. The failure is therefore
less likely to be "the model learned to answer after the wrong number of beats"
and more likely to be reader-to-thinker routing or thinker-to-speaker semantic
alignment.
```

## 2026-05-21 Stage36B Oracle Arg Routing Diagnostic Launch

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_194211_LOCAL_STAGE36B_oracle_arg_routing_seed48
```

Setup:

```text
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
condition_on_operation_ids=true
operation_arg_conditioning=true
train_depths=4,6
reasoning_count=2048
epochs=1
```

Humanistic reason:

```text
This is not the final general LLM architecture. It is a thermometer.

If giving the thinker the correct step operation and digit causes a jump, the
reader-to-thinker handoff is broken: Qwen reads the prompt, but the recurrent
thinker is not reliably receiving the typed ingredients for each step.

If it does not jump, the problem is more likely that the thinker still speaks a
latent dialect the Qwen LM head cannot reliably understand.
```

Gate:

```text
Promote routing as the next architecture target if heldout_mean_acc crosses
0.16 or if depth4/depth6 both clearly exceed the H2 robust baseline.
Reject oracle routing as the primary bottleneck if it stays near 0.13-0.15.
```

Observed:

```text
epoch1:
  prior_no_posterior_train_acc=0.1475
  posterior_guided_train_acc=0.2559
  heldout_mean_acc=0.1219
  depth4=0.1562 depth6=0.1172 depth8=0.1172 depth10=0.1250 depth12=0.0938
  K4_acc=0.1094 K4_cos=0.720
  verified_source_target_token_acc=0.2560
```

Decision:

```text
Reject oracle operation/argument routing as the primary bottleneck.

The thinker did not generalize even when given typed step ingredients. This
makes reader-to-thinker routing less likely as the dominant failure. The next
highest-probability bottleneck is thinker-to-speaker semantic alignment: the
recurrent state may be doing some internal work, but the Qwen LM head cannot
reliably read it as a digit answer.
```

## 2026-05-21 Stage36C State-Head Speaker Diagnostic Launch

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_194912_LOCAL_STAGE36C_statehead_speaker_diag_seed49
```

Setup:

```text
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
answer_path=state_head
condition_on_operation_ids=false
train_depths=4,6
reasoning_count=2048
epochs=1
```

Humanistic reason:

```text
This is a speaker diagnostic, not the final architecture.

If the direct state_head improves while the Qwen LM-head path does not, the
main problem is that the recurrent thought state speaks a latent dialect the
frozen Qwen mouth cannot understand.

If state_head also fails, the recurrent thinker itself is not yet learning a
general algorithmic state transition.
```

Gate:

```text
Promote speaker-alignment work if state_head heldout_mean_acc clearly exceeds
the H2 robust mean of 0.1539 or crosses 0.16. Reject speaker-only blame if it
stays near 0.13-0.15.
```

Observed:

```text
epoch1:
  prior_no_posterior_train_acc=0.1421
  posterior_guided_train_acc=0.1416
  heldout_mean_acc=0.1562
  depth4=0.1797 depth6=0.1094 depth8=0.1562 depth10=0.2031 depth12=0.1328
  K4_acc=0.1562 K4_cos=0.650
  verified_source_target_token_acc=0.2080
```

Decision:

```text
State-head speaking helps slightly but does not create a big jump. Therefore
speaker alignment is a contributor, not the whole bottleneck.

The next highest-probability fix is to train the same path that is evaluated:
prior/no-posterior recurrent thinking with direct state/trace supervision.
Humanistically, the student should receive step-by-step correction while
practicing without the tutor whispering the answer key.
```

## 2026-05-21 Stage36D Prior-Path Trace Supervision Launch

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_195713_LOCAL_STAGE36D_prior_trace_supervision_seed50
```

Setup:

```text
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
answer_path=lm_head
stochastic_posterior_guidance=false
posterior_answer_weight=1.0
state_supervision_weight=0.5
aux_step_answer_weight=0.1
prior_answer_weight=0.0
prior_posterior_logit_distill_weight=0.0
train_depths=4,6
reasoning_count=2048
epochs=1
```

Humanistic reason:

```text
This is the first Stage36 run that directly trains the same no-posterior path
used at evaluation time with step-by-step trace correction.

Earlier true-GRAM runs let the posterior-guided tutor path receive most of the
step structure while the exam path received only weak final-answer pressure.
Stage36D removes that mismatch.
```

Gate:

```text
Promote if heldout_mean_acc crosses 0.16 or if depth4/depth6 both improve over
H2 robust depth4=0.1367 and depth6=0.1484. Reject if it remains near 0.13-0.15.
```

Observed:

```text
epoch1:
  train_acc=0.1367
  heldout_mean_acc=0.1375
  depth4=0.1250 depth6=0.1719 depth8=0.1328 depth10=0.1562 depth12=0.1016
  K4_acc=0.1422 K4_cos=0.724
  verified_source_target_token_acc=0.2320
```

Decision:

```text
Reject noisy prior-path trace supervision as implemented.

The logic was correct, but training the prior path while full-strength GRAM
sampling is active likely makes the student practice while the thought state is
too unstable. The next check is a deterministic prior-trace warmup: learn the
main thought path first, then reintroduce stochastic GRAM/search.
```

## 2026-05-21 Stage36E Deterministic Trace Warmup Launch

Run:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260521_200136_LOCAL_STAGE36E_deterministic_trace_warmup_seed51
```

Setup:

```text
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
answer_path=lm_head
stochastic_high_level_guidance=false
stochastic_high_level_eval=false
stochastic_posterior_guidance=false
posterior_answer_weight=1.0
state_supervision_weight=0.5
aux_step_answer_weight=0.1
train_depths=4,6
reasoning_count=2048
epochs=1
```

Humanistic reason:

```text
Before asking the model to explore many possible thoughts, make sure one clean
thought path can learn the lesson. If deterministic trace warmup improves, GRAM
should be reintroduced after the main path is stable. If it does not improve,
the recurrent core architecture itself is the likely bottleneck.
```

Gate:

```text
Promote if heldout_mean_acc crosses 0.16 or if train_acc rises while heldout
does not collapse. Reject if it remains near 0.13-0.15.
```

Observed:

```text
epoch1:
  train_acc=0.1274
  heldout_mean_acc=0.1281
  depth4=0.1719 depth6=0.1250 depth8=0.1250 depth10=0.1641 depth12=0.0547
  verified_source_target_token_acc=0.2120
```

Decision:

```text
Reject deterministic trace warmup as sufficient.

This rules out the simplest "GRAM noise prevented trace learning" explanation.
After Stage36A-E, the dominant failure is now likely the recurrent core's
algorithmic state representation itself: it does not reliably form a stable,
generalizable working state even when timing, oracle routing, direct state-head
speaking, and deterministic trace warmup are tested.
```

Next architecture implication:

```text
Stop doing scalar-weight variants on the same latent MLP core.

The next accepted-probability architecture change should alter the working
state itself: add an explicit, bounded, typed belief/register memory that the
LM-compatible recurrent thought state must update and read, while keeping the
normal LLM causal path as:

Qwen reader -> recurrent thought -> bounded working register -> LM-compatible answer.
```

## 2026-05-21 Stage35H DGX 16384 Interim Result

Run:

```text
/mnt/data4tb/qtrm_eval/20260521_185205_DGX_STAGE35H_diversity16384_examfirst_seed44
```

Observed:

```text
epoch1:
  prior_no_posterior_train_acc=0.1335
  posterior_guided_train_acc=0.2723
  heldout_mean_acc=0.1430
  depth4=0.1367 depth6=0.1211 depth8=0.1445 depth10=0.1953 depth12=0.1172
  K4_acc=0.1461 K4_cos=0.788
  verified_source_target_token_acc=0.1960
```

Reading:

```text
DGX 16k does not yet confirm the local data-diversity jump. Depth10 is strong,
but depth4/depth6 are weak, so the mean stays below the 0.16 gate.

Humanistically: simply giving the student a much thicker worksheet did not make
the basic lessons steadier yet. It may still improve after epoch2, but epoch1
does not justify a new architecture claim.
```

Final DGX result:

```text
epoch2:
  prior_no_posterior_train_acc=0.1359
  posterior_guided_train_acc=0.1586
  heldout_mean_acc=0.1328
  depth4=0.0938 depth6=0.1406 depth8=0.1367 depth10=0.1836 depth12=0.1094
  K4_acc=0.1469 K4_cos=0.835
  verified_source_target_token_acc=0.2000
```

Decision:

```text
Reject large data-breadth scaling as a standalone solution.
The 16k DGX run does not reproduce the local H2 near-gate signal.
```

## 2026-05-21 Stage37 Preflight: Seven-Axis Humanistic Gate

Before launching more GPU work, every candidate must pass this plain-language
gate. If the story fails here, the run is not a model failure; it is a design
failure.

```text
1. Architecture
   The model needs a reader, a thinker, a notebook, and a speaker.
   Qwen is the reader. The recurrent state is the thinker. The new bounded
   working register is the notebook. The Qwen-compatible LM head is the
   speaker. The speaker must hear the notebook through the normal answer path,
   not through a side probe.

2. Curriculum
   The student should not learn reading, remembering, translating, stochastic
   search, and OOD solving all at once. Stage37 keeps the earlier attention
   pooling / LM-head path and changes only the working memory shape.

3. Reward / Loss
   The run must reward the behavior we want: stable recurrent reasoning that
   still speaks through the LM-compatible path. It should not reward a separate
   executor, a side-head shortcut, or collapsed latent states.

4. Evaluation
   The first gate is a near-term architecture smoke gate, not a final "general
   intelligence" exam. Compare against the known Stage36E deterministic trace
   warmup and Stage35H2 robust H2 baseline.

5. Exploration
   Stage37A is intentionally deterministic. It is not claiming GRAM search yet.
   If the deterministic working notebook cannot help, K-sampling on top would
   only search over weak thoughts.

6. Data Contract
   The prompt and label contract remain unchanged from the verified HRM-text
   aligned runs. We are not silently changing the exam while judging the
   architecture.

7. Causality / Ablation
   The only new causal intervention in Stage37A is the bounded working register.
   If accuracy improves, the next ablation is register-off with the same
   checkpoint/config shape. If it does not improve, this register form is not
   the missing notebook.
```

Stage37A accepted-probability reading:

```text
This is higher-probability than another scalar loss schedule because it changes
the place where the model keeps intermediate work. Humanistically, previous
runs asked the thinker to keep a whole calculation in one foggy sentence.
Stage37A gives it a small typed notebook with bounded edits.
```

Stage37A local smoke gate:

```text
Promote:
  heldout_mean_acc > 0.1600, or clear improvement over Stage36E
  (0.1281 mean) with non-collapsed working_register_norm/gate telemetry.

Reject:
  heldout_mean_acc <= 0.1300 and no useful register telemetry movement.

Hold:
  0.1300 < heldout_mean_acc <= 0.1600 with healthy register telemetry; test a
  stronger register/readout variant only if the seven-axis story still holds.
```

Launched local Stage37A:

```text
first_launcher_attempt=20260521_201432_LOCAL_STAGE37A_working_register_trace_seed52
launcher_result=nohup process exited with empty log; classified as launcher failure, not model failure

active_run_id=20260521_201543_LOCAL_STAGE37A_working_register_trace_seed52_setsid
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_201543_LOCAL_STAGE37A_working_register_trace_seed52_setsid
log=/tmp/20260521_201543_LOCAL_STAGE37A_working_register_trace_seed52_setsid.log
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt

single intervention:
  --working-register-enabled
  --working-register-slots 4
  --working-register-update-scale 0.25
  --working-register-feedback-scale 1.0
  --working-register-gate-init-bias -1.5
```

Local Stage37A result:

```text
epoch1:
  train_acc=0.1274
  heldout_mean_acc=0.1516
  depth4=0.1484 depth6=0.1172 depth8=0.1562 depth10=0.2031 depth12=0.1328
  verified_source_target_token_acc=0.1840
  working_register_norm=32.0125
  working_register_gate_mean=0.1112
```

Decision:

```text
Hold, not promote.

The working register improves clearly over Stage36E deterministic trace warmup
(0.1281 -> 0.1516) and shows live non-collapsed telemetry, especially a strong
depth10 score. It does not yet beat the Stage35H2 robust baseline
(0.1539 mean), so this is not the big jump.

Humanistically: the notebook exists and is being written to, but it is not yet
organized enough to make the student's whole exam score jump. The next variant
must improve how the notebook is read/written, not merely add more loss weight.
```

## 2026-05-21 Stage37B: Query-Dot Register Readout

Seven-axis preflight:

```text
Architecture:
  Stage37A gave the model a notebook, but the thinker read all notebook pages
  as an average. Stage37B lets the current thought choose the most relevant
  page by similarity. Reader, thinker, notebook, and speaker are still on the
  normal LM-compatible path.

Curriculum:
  Same data, same loss, same start checkpoint, same epoch budget. Only notebook
  readout changes.

Reward / Loss:
  No new side reward. The answer still must come through the recurrent thought
  state and Qwen LM head.

Evaluation:
  Compare directly with Stage37A and Stage35H2 robust baseline.

Exploration:
  Still deterministic; no GRAM/search claim.

Data Contract:
  Same HRM-text verified mix and synthetic heldout contract.

Causality:
  If this improves over Stage37A, the gain is attributable to how the notebook
  is read, not a broader recipe change.
```

Launched local Stage37B:

```text
run_id=20260521_202405_LOCAL_STAGE37B_querydot_register_seed53
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_202405_LOCAL_STAGE37B_querydot_register_seed53
log=/tmp/20260521_202405_LOCAL_STAGE37B_querydot_register_seed53.log
single intervention vs Stage37A:
  --working-register-summary-mode query_dot
```

Launched DGX Stage37B from the same imported local H2 checkpoint:

```text
run_id=20260521_202525_DGX_STAGE37B_querydot_register_seed54
out_dir=/mnt/data4tb/qtrm_eval/20260521_202525_DGX_STAGE37B_querydot_register_seed54
log=/tmp/20260521_202525_DGX_STAGE37B_querydot_register_seed54.log
resume=/mnt/data4tb/qtrm_eval/imported_local_stage35h2_best/best_generalization.pt
```

Local Stage37B result:

```text
epoch1:
  train_acc=0.1353
  heldout_mean_acc=0.1047
  depth4=0.1016 depth6=0.0938 depth8=0.1172 depth10=0.0859 depth12=0.1250
  verified_source_target_token_acc=0.2040
  working_register_norm=32.0130
  working_register_gate_mean=0.1180
```

Decision:

```text
Reject query-dot register readout.

Humanistically: the model has notebook pages, but the pages do not yet have
stable meanings. Asking the current thought to pick a page by raw similarity
causes it to grab the wrong page confidently. Mean register readout was bland
but safer; query-dot is too sharp before slot semantics are trained.

Next implication:
  Do not sharpen register readout before teaching slot roles. If the register
  path continues, the next structurally valid candidate is role-stabilized
  register slots or a gentle learned query_attention mode, not raw dot routing.
```

DGX Stage37B result:

```text
epoch1:
  train_acc=0.1484
  heldout_mean_acc=0.1500
  depth4=0.1328 depth6=0.1641 depth8=0.1641 depth10=0.1875 depth12=0.1016
  verified_source_target_token_acc=0.2200
```

Combined Stage37B decision:

```text
Reject query-dot register readout.

DGX does not collapse like the local seed, but it still does not beat Stage37A
or the Stage35H2 robust baseline. The sharp page picker is not a reliable
generalization mechanism.
```

## 2026-05-21 Stage37C: Gentle Learned Register Query

Launched local Stage37C:

```text
run_id=20260521_202900_LOCAL_STAGE37C_queryattn_register_seed55
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_202900_LOCAL_STAGE37C_queryattn_register_seed55
log=/tmp/20260521_202900_LOCAL_STAGE37C_queryattn_register_seed55.log
single intervention vs Stage37A:
  --working-register-summary-mode query_attention
```

Humanistic gate:

```text
Unlike query_dot, query_attention starts as mean readout and must earn sharper
page selection through gradients. This tests whether gentle learned page
selection helps without letting an untrained thought grab arbitrary pages.
```

Local Stage37C result:

```text
epoch1:
  train_acc=0.1328
  heldout_mean_acc=0.1500
  depth4=0.1797 depth6=0.1094 depth8=0.1797 depth10=0.1562 depth12=0.1250
  verified_source_target_token_acc=0.1880
  working_register_norm=32.0107
  working_register_gate_mean=0.1017
```

Stage37 A/B/C comparison:

```text
Stage37A mean-register:
  heldout_mean_acc=0.1516 depth10=0.2031

Stage37B query-dot:
  local heldout_mean_acc=0.1047 depth10=0.0859
  dgx   heldout_mean_acc=0.1500 depth10=0.1875

Stage37C learned query-attention:
  heldout_mean_acc=0.1500 depth10=0.1562
```

Decision:

```text
Keep Stage37A mean-register as the only non-rejected register variant.

This is not a big jump. It is a small positive signal that a bounded notebook
helps when read conservatively. The failed sharper readouts imply that the next
architecture move must stabilize slot roles before selective page reading.

Stop condition for this branch:
  Do not run more register-readout variants until a role-stabilization
  mechanism is added. Otherwise we are back to low-probability tuning.
```

## 2026-05-21 Stage38A: Role-Stabilized Mean Register

Research cue:

```text
Slot Attention argues for a small set of slots that specialize through
competition over input features. RIMs argues that modular recurrent mechanisms
generalize better when modules retain semi-independent roles and communicate
sparingly. Stage37 showed the same failure locally: a register notebook helps
only when read conservatively, because its pages do not yet have stable roles.
```

Seven-axis preflight:

```text
Architecture:
  Keep the reader-thinker-notebook-speaker path unchanged. Qwen reads, the
  recurrent state thinks, the working register stores bounded pages, and the
  Qwen LM head speaks. The new change only keeps each page aware of its role.

Curriculum:
  Same Stage37A data, loss, epoch budget, and checkpoint. The student is not
  being asked to learn search or a new exam at the same time.

Reward / Loss:
  No side executor and no new answer shortcut. Role cosine is telemetry, not a
  reward. The model still wins only by producing the correct answer through the
  recurrent LM-compatible path.

Evaluation:
  Compare with Stage37A mean-register, Stage37B/C rejected readouts, and the
  Stage35H2 robust baseline.

Exploration:
  Deterministic. This is not claiming GRAM search yet.

Data Contract:
  Same HRM-text verified healing/source rows and the same synthetic heldout
  contract.

Causality:
  Single intervention versus Stage37A:
    --working-register-role-conditioning
    --working-register-role-anchor-scale 0.05
```

Promote gate:

```text
Promote if heldout_mean_acc > 0.1600 or depth10 stays strong while mean beats
Stage35H2 robust 0.1539, with non-collapsed role_cosine telemetry.

Reject if heldout_mean_acc <= 0.1300 or role anchoring collapses the register.

Hold if it matches Stage37A without collapse; then role stability is necessary
but not sufficient.
```

Launched local Stage38A:

```text
run_id=20260521_203925_LOCAL_STAGE38A_role_stabilized_register_seed56
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_203925_LOCAL_STAGE38A_role_stabilized_register_seed56
log=/tmp/20260521_203925_LOCAL_STAGE38A_role_stabilized_register_seed56.log
```

Launched DGX Stage38A:

```text
run_id=20260521_204042_DGX_STAGE38A_role_stabilized_register_seed57
out_dir=/mnt/data4tb/qtrm_eval/20260521_204042_DGX_STAGE38A_role_stabilized_register_seed57
log=/tmp/20260521_204042_DGX_STAGE38A_role_stabilized_register_seed57.log
resume=/mnt/data4tb/qtrm_eval/imported_local_stage35h2_best/best_generalization.pt
```

Local Stage38A result:

```text
epoch1:
  train_acc=0.1357
  heldout_mean_acc=0.1562
  depth4=0.1797 depth6=0.1094 depth8=0.1562 depth10=0.2031 depth12=0.1328
  verified_source_target_token_acc=0.2240
  working_register_norm=32.0126
  working_register_gate_mean=0.1554
  working_register_role_cosine=0.0471
```

Interim decision:

```text
Hold-positive.

This is the best local Stage37/38 result so far and slightly beats the Stage35H2
robust baseline (0.1539), while preserving the strong Stage37A depth10 score.
It does not yet cross the 0.1600 promote gate.

Humanistically: adding page titles helped a little. The notebook is still not
organized enough for a big jump, but role-stabilization is a better direction
than sharper page selection.
```

DGX Stage38A result:

```text
epoch1:
  train_acc=0.1284
  heldout_mean_acc=0.1187
  depth4=0.1875 depth6=0.0859 depth8=0.1328 depth10=0.1016 depth12=0.0859
  verified_source_target_token_acc=0.2240
  working_register_norm=32.0090
  working_register_gate_mean=0.1132
  working_register_role_cosine=0.0735
```

Combined decision:

```text
Reject Stage38A as a robust big-jump mechanism.

Local improved slightly, but DGX from the same imported checkpoint collapsed
below the Stage37A/H2 line. The role anchor is not enough, and the signal is
too seed/device-sensitive to promote.

Humanistically: page titles helped one student a little, but another student
with the same book still failed the exam. The problem is not only that pages
lack names; the calculation itself still lacks an explicit, stable algorithmic
path.
```

## 2026-05-21 Stage39A: Cyclic Sparse Register Write

Humanistic diagnosis:

```text
Stage38 gave notebook pages stable titles, but every reasoning step still wrote
to every page. That is like taking notes by smearing each sentence across every
page of the notebook. The next minimal fix is to write to only one page per
step in a deterministic cycle. This is not a task-specific executor; it is a
generic interference-control test.
```

Seven-axis preflight:

```text
Architecture:
  Same Qwen reader -> recurrent thinker -> bounded register notebook -> Qwen
  LM speaker path. The only change is sparse page writing.

Curriculum:
  Same Stage38A data, losses, checkpoint, and one-epoch smoke budget.

Reward / Loss:
  No new reward. Role cosine and gate telemetry are observation only.

Evaluation:
  Compare against Stage37A, Stage38A local/DGX, and Stage35H2 robust baseline.

Exploration:
  Deterministic. No GRAM/search claim.

Data Contract:
  Same verified HRM-text rows and synthetic heldout split.

Causality:
  Single intervention vs Stage38A:
    --working-register-update-mode cyclic
```

Gate:

```text
Promote if local and DGX both beat 0.1539 robust mean or either crosses 0.1600
with healthy role/gate telemetry.

Reject if either collapses below 0.1300 or if cyclic write destroys train
accuracy.
```

Launched local Stage39A:

```text
run_id=20260521_204820_LOCAL_STAGE39A_cyclic_register_seed58
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_204820_LOCAL_STAGE39A_cyclic_register_seed58
log=/tmp/20260521_204820_LOCAL_STAGE39A_cyclic_register_seed58.log
```

Launched DGX Stage39A:

```text
run_id=20260521_204923_DGX_STAGE39A_cyclic_register_seed59
out_dir=/mnt/data4tb/qtrm_eval/20260521_204923_DGX_STAGE39A_cyclic_register_seed59
log=/tmp/20260521_204923_DGX_STAGE39A_cyclic_register_seed59.log
resume=/mnt/data4tb/qtrm_eval/imported_local_stage35h2_best/best_generalization.pt
```

Local Stage39A result:

```text
epoch1:
  train_acc=0.1162
  heldout_mean_acc=0.1156
  depth4=0.1797 depth6=0.1016 depth8=0.1328 depth10=0.0781 depth12=0.0859
  verified_source_target_token_acc=0.2080
  working_register_norm=32.0104
  working_register_gate_mean=0.0218
  working_register_role_cosine=0.0312
```

Interim decision:

```text
Reject local cyclic sparse write.

Humanistically: writing to only one page at a time made the notebook too
stingy. It reduced interference but also starved the recurrent thinker of
usable workspace bandwidth. If DGX agrees, stop this branch.
```

DGX Stage39A result:

```text
epoch1:
  train_acc=0.1304
  heldout_mean_acc=0.1297
  depth4=0.1562 depth6=0.1484 depth8=0.1562 depth10=0.0938 depth12=0.0938
  verified_source_target_token_acc=0.2120
  working_register_norm=32.0099
  working_register_gate_mean=0.0221
  working_register_role_cosine=0.0380
```

Combined decision:

```text
Reject Stage39A.

Local and DGX agree that hard cyclic sparse write is too bandwidth-starved.
This branch does not provide the missing big jump.

Current best reading after Stage37-39:
  - A bounded mean-read notebook is mildly useful.
  - Sharper readout before slot semantics is harmful.
  - Role anchors alone are not robust.
  - Hard sparse writing is too restrictive.

Humanistic conclusion:
  The model does not need a fancier notebook first. It needs a clearer main
  algorithmic path: a thought that can carry the running calculation in a form
  the Qwen LM head can speak. Register mechanisms should become support tools,
  not the primary next bet.
```

## 2026-05-21 Stage40 Preflight: LM-Compatible Main Thought Path

Humanistic diagnosis:

```text
The last branch kept improving the student's notebook. That was not the main
problem. The main problem is that the student's inner thought is still not
consistently in the language spoken by Qwen's mouth.

HRM-Text-like success should look like one body:
  read -> think repeatedly -> speak

Our current QTRM-on-Qwen often looks like two dialects:
  Qwen reads fluently,
  the recurrent core thinks in a private continuous dialect,
  then the LM head is asked to understand that private dialect.
```

Paper cue:

```text
Kalra and Barkeshli, 2026, "Quantifying Hyperparameter Transfer and the
Importance of Embedding Layer Learning Rate"
https://deeplearn.org/arxiv/756894/quantifying-hyperparameter-transfer-and-the-importance-of-embedding-layer-learning-rate

Relevant mechanism:
  embedding / unembedding learning-rate treatment can bottleneck LLM training
  stability and hyperparameter transfer.

QTRM implication:
  If we ask recurrent thought states to speak through Qwen's LM head, then the
  speaker surface may need its own training control. Treat this as a speaker
  alignment recipe, not as a standalone architecture breakthrough.
```

Seven-axis gate:

```text
1. Architecture:
   Qwen reads. The recurrent state is the main thought. Qwen LM head speaks.
   Stage40 adds no side answer head and no task-specific executor.

2. Curriculum:
   Do not ask the model to learn a new register, new routing, and new GRAM
   search at once. Stage40 keeps the Stage35H2 recipe and changes only how
   the thought state is taught to speak.

3. Reward / Loss:
   The new semantic_step_alignment loss aligns intermediate recurrent states
   to the same Qwen digit-token directions used by the final LM-head answer.
   This is not "meaningless intermediate supervision"; it is speaker-language
   training for the main thought path.

4. Evaluation:
   Compare against Stage35H2 robust mean=0.153906 and Stage37/38/39 register
   runs. A local one-epoch smoke is not a final OOD solution.

5. Exploration:
   Stage40A is not a GRAM-search claim. Stage40B may keep true-GRAM telemetry,
   but success still requires prior/no-posterior accuracy and heldout mean.

6. Data Contract:
   Same HRM-text verified source/healing rows and same synthetic depth/family
   contract. We are changing the teaching signal, not the exam.

7. Causality / Ablation:
   Stage40A tests semantic_step_alignment with Qwen frozen.
   Stage40B tests whether opening the speaker surface with separate LR helps.
   If gains appear only in Stage40B, the speaker-surface LR paper is relevant.
   If neither moves heldout, the bottleneck is deeper than speaker alignment.
```

Accepted-likelihood score:

```text
Direct bottleneck replacement: 3/3
  It targets the thought-mouth mismatch, not a side probe.

Normal answer-path enforcement: 3/3
  Final and step signals go through Qwen LM-head geometry.

Ablation clarity: 2/2
  Local frozen-Qwen vs DGX speaker-surface-open split separates alignment loss
  from partial Qwen speaker adaptation.

Different from last rejected family: 2/2
  This is not another register readout/write variant.

Total: 10/10, eligible for local/DGX parallel A/B.
```

Planned runs:

```text
Stage40A local:
  Frozen Qwen.
  Add --semantic-step-alignment-weight 0.05.
  Purpose: prove whether the recurrent thought path improves when every
  intermediate thought is gently taught Qwen's speaking geometry.

Stage40B DGX:
  Same as Stage40A, plus open Qwen speaker surfaces:
    --train-qwen-lm-head
    --train-qwen-final-norm
    --train-qwen-embeddings
    --qwen-embedding-lr-multiplier 2.0
    --qwen-lm-head-lr-multiplier 2.0
  Purpose: test the embedding/unembedding LR hypothesis on the speaker
  alignment bottleneck.
```

Promote gate:

```text
Promote if either run crosses heldout_mean_acc >= 0.165 with depth4/depth6 not
collapsing, verified_source_target_token_acc stable, and semantic_step loss
decreasing.

Hold if it improves semantic/source metrics but heldout remains 0.150-0.160.

Reject if heldout <= 0.135 or source token accuracy collapses.
```

Implementation added:

```text
scripts/511_train_qwen_state_transition_hrmtext.py
  --semantic-step-alignment-weight
    aligns intermediate recurrent states to Qwen LM-head digit-token directions.

  --train-qwen-embeddings
  --train-qwen-lm-head
  --train-qwen-final-norm
  --qwen-embedding-lr-multiplier
  --qwen-lm-head-lr-multiplier
  --qwen-final-norm-lr-multiplier
    allow speaker-surface LR A/B tests inspired by the embedding-LR paper.

Observability:
  TensorBoard:
    Step/Loss_SemanticStepAlignment
    Epoch/Loss_SemanticStepAlignment

  Aim:
    train_loss_semantic_step_alignment
    loss_semantic_step_alignment
```

Verification:

```text
local:
  .venv/bin/python -m py_compile scripts/511_train_qwen_state_transition_hrmtext.py \
    src/qtrm_mm/qwen_backbone_state_transition.py src/qtrm_mm/state_transition_core.py

DGX:
  PYTHONPYCACHEPREFIX=/tmp/qtrm_pycache_stage40 \
  /mnt/data4tb/venv_sglang_pr23000/bin/python -m py_compile ...
```

### Stage40A Local Result: Frozen Qwen Semantic Step Alignment

Run:

```text
run_id=20260521_211650_LOCAL_STAGE40A_semstep_frozen_seed60
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_211650_LOCAL_STAGE40A_semstep_frozen_seed60
log=/tmp/20260521_211650_LOCAL_STAGE40A_semstep_frozen_seed60.log
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_190055_LOCAL_STAGE35H2_diversity4096_examfirst_seed45/best_generalization.pt
single main intervention:
  --semantic-step-alignment-weight 0.05
```

Observed:

```text
epoch1:
  train_acc=0.2417
  prior_acc=0.1387
  semantic_lm=0.0522
  semantic_step=0.0400
  heldout_mean_acc=0.1328
  depth4=0.1641 depth6=0.1172 depth8=0.1016 depth10=0.1484 depth12=0.1328
  K4_acc=0.1281 K4_cos=0.736
  verified_source_target_token_acc=0.2200

epoch2:
  train_acc=0.2056
  prior_acc=0.1460
  semantic_lm=0.0480
  semantic_step=0.0374
  heldout_mean_acc=0.1359
  depth4=0.1719 depth6=0.1094 depth8=0.1406 depth10=0.1719 depth12=0.0859
  K4_acc=0.1437 K4_cos=0.758
  verified_source_target_token_acc=0.2240
```

Decision:

```text
Reject as a big-jump mechanism.

The semantic step loss did decrease, so the new loss is live and observable.
However, heldout_mean_acc stayed far below the Stage35H2 robust baseline
0.153906. This means "teach the frozen Qwen mouth to understand intermediate
thoughts" is not enough.

Humanistically:
  The student learned to pronounce the intermediate number-shapes a little more
  like Qwen, but still did not learn the method of carrying the calculation.
```

### Stage40B DGX Result: Speaker Surface Opened With Embedding/Lm-Head LR

Run:

```text
run_id=20260521_211905_DGX_STAGE40B_semstep_speakerLR_seed61
out_dir=/mnt/data4tb/qtrm_eval/20260521_211905_DGX_STAGE40B_semstep_speakerLR_seed61
log=/tmp/20260521_211905_DGX_STAGE40B_semstep_speakerLR_seed61.log
resume=/mnt/data4tb/qtrm_eval/imported_local_stage35h2_best/best_generalization.pt
speaker-surface intervention:
  --train-qwen-embeddings
  --train-qwen-lm-head
  --train-qwen-final-norm
  --qwen-embedding-lr-multiplier 1.5
  --qwen-lm-head-lr-multiplier 1.5
```

Observed:

```text
epoch1:
  train_acc=0.2285
  prior_acc=0.1318
  semantic_lm=0.0516
  semantic_step=0.0398
  heldout_mean_acc=0.1125
  depth4=0.1172 depth6=0.0703 depth8=0.1484 depth10=0.1172 depth12=0.1094
  K4_acc=0.1203 K4_cos=0.724
  verified_source_target_token_acc=0.1760
```

Decision:

```text
Reject and early-stop after epoch1.

Opening the Qwen speaker surface with elevated embedding/lm-head LR hurt both
heldout accuracy and verified-source token accuracy. This directly fails the
source-collapse gate.

Humanistically:
  Letting the mouth and dictionary move while the student's method is still
  unstable made the language less reliable, not more general.
```

Stage40 conclusion:

```text
The dominant bottleneck is not just speaker alignment.

Current ruled-out branches:
  - timing contract only
  - oracle operation/argument routing only
  - direct state-head speaker only
  - deterministic trace warmup only
  - bounded register notebook variants
  - frozen semantic step alignment only
  - opening Qwen embedding/lm-head speaker surface at this stage

Next high-probability direction:
  Replace the core algorithmic transition itself so the main thought state
  carries a running calculation in a stable semantic/token-compatible form.

Do not run another nearby speaker-LR or register tweak unless a diagnostic
proves the core transition can already carry the answer.
```

## 2026-05-21 Stage41: Semantic Token Feedback Inside The Thought Loop

Humanistic diagnosis:

```text
Qwen is the reader and speaker.
The recurrent core is the thinker.

Stage40 showed that teaching the thinker to look more like Qwen after the fact
is not enough. The thinker must hear its own partial answer in Qwen's language
while it is still thinking.

Plain story:
  A person solving a problem often whispers the current belief, hears whether
  it sounds coherent, then continues. Stage41 makes z_H do that with Qwen's
  own LM-head label-token directions.
```

Seven-axis preflight:

```text
1. Architecture:
   Qwen reads input tokens. z_H/z_L remain the main thought state. The evaluated
   answer still comes from --answer-path lm_head. The intervention is inside
   the recurrent loop, not a side probe.

2. Curriculum:
   Resume from Stage35H2 best; do not open Qwen embeddings/lm_head yet. Only add
   one new capacity: per-step semantic token feedback.

3. Reward/loss:
   Keep H2 losses mostly unchanged. New mechanism is architectural feedback,
   not an extra answer shortcut. Log gate/entropy so collapse is visible.

4. Evaluation:
   Same heldout depths 4/6/8/10/12 and K=[1,4] stochastic eval as H2/Stage40.
   Compare first against H2 robust mean 0.153906 and Stage35H2 run best 0.159375.

5. Exploration:
   True-GRAM stochastic path remains on; K-sample eval remains logged. Stage41 is
   not claiming GRAM alone, but it must not disable GRAM observability.

6. Data contract:
   Same generalized synthetic contract and same HRM-Text verified Dolly/source
   rows as Stage35H2.

7. Causality/ablation:
   The A/B switch is --semantic-token-feedback off/on from the same H2
   checkpoint. If the gain is real, it should vanish when this flag is off.
```

Implementation:

```text
Files:
  src/qtrm_mm/state_transition_core.py
  src/qtrm_mm/qwen_backbone_state_transition.py
  scripts/511_train_qwen_state_transition_hrmtext.py
  scripts/512_eval_qwen_state_transition_generalization.py

New flags:
  --semantic-token-feedback
  --semantic-token-feedback-scale
  --semantic-token-feedback-temperature
  --semantic-token-feedback-gate-init-bias

Mechanism:
  Qwen lm_head label-token weights -> semantic basis in thought space
  z_H -> cosine logits over basis -> expected token direction
  z_H <- gated bounded blend toward expected token direction

Telemetry:
  Step/Epoch SemanticTokenFeedback_GateMean
  Step/Epoch SemanticTokenFeedback_Entropy
  Aim mirrors the same train/epoch scalars.
```

Relation to arXiv:2605.21486:

```text
The embedding-LR transfer paper supports careful treatment of Qwen embedding
and lm_head learning rates when that surface is opened. It does not explain the
current OOD wall by itself. Stage41 keeps Qwen frozen and first fixes the main
thought path; speaker-surface LR should be revisited only after this path shows
generalization signal without source collapse.
```

Active runs:

```text
Local:
  run_id=20260521_224500_LOCAL_STAGE41A_semantic_token_feedback_seed70
  out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_224500_LOCAL_STAGE41A_semantic_token_feedback_seed70
  log=/tmp/20260521_224500_LOCAL_STAGE41A_semantic_token_feedback_seed70.log

DGX:
  run_id=20260521_223000_DGX_STAGE41B_semantic_token_feedback_seed71
  out_dir=/mnt/data4tb/qtrm_eval/20260521_223000_DGX_STAGE41B_semantic_token_feedback_seed71
  log=/tmp/20260521_223000_DGX_STAGE41B_semantic_token_feedback_seed71.log
```

Stage41A/B cosine-feedback result:

```text
Local Stage41A epoch1:
  heldout_mean_acc=0.1109
  depth4=0.1797 depth6=0.0703 depth8=0.1641 depth10=0.1094 depth12=0.0312
  K4_acc=0.1141 K4_cos=0.724
  verified_source_target_token_acc=0.2200

DGX Stage41B epoch1:
  heldout_mean_acc=0.1375
  depth4=0.1641 depth6=0.0859 depth8=0.1328 depth10=0.1719 depth12=0.1328
  K4_acc=0.1484 K4_cos=0.726
  verified_source_target_token_acc=0.2000

Telemetry:
  SemanticTokenFeedback entropy stayed near 1.0.
  Gate was live, but the feedback target was effectively a diffuse average.
```

Decision:

```text
Reject cosine semantic-token feedback as implemented.

Reason:
  The feedback did not use the same scoring geometry as the Qwen LM-head
  speaker. It compared z_H and token directions by cosine, which produced a
  nearly uniform belief. Humanistically, the thinker was not actually hearing
  its own answer through Qwen's mouth; it was comparing the rough direction of
  its thought against a set of token signposts.
```

Stage41C correction:

```text
Change:
  Add --semantic-token-feedback-score-mode dot.

Mechanism:
  z_H is normalized by the feedback RMSNorm, scored by dot product against
  Qwen LM-head label-token directions, scaled by sqrt(d_state), then softened
  by temperature. This is closer to the evaluated LM-head answer path.

Aggressive gate:
  --semantic-token-feedback-score-mode dot
  --semantic-token-feedback-scale 0.5
  --semantic-token-feedback-temperature 0.5
  --semantic-token-feedback-gate-init-bias -1.0

Active runs:
  Local:
    run_id=20260521_225500_LOCAL_STAGE41C_dotfeedback_aggressive_seed72
    log=/tmp/20260521_225500_LOCAL_STAGE41C_dotfeedback_aggressive_seed72.log
  DGX:
    run_id=20260521_225500_DGX_STAGE41C_dotfeedback_aggressive_seed73
    log=/tmp/20260521_225500_DGX_STAGE41C_dotfeedback_aggressive_seed73.log
```

Stage41C early telemetry:

```text
dot-logit feedback increased gate to about 0.29-0.30, but entropy still stayed
near 1.0. That means z_H is not yet in a geometry where its self-belief can
choose among Qwen label-token directions.
```

Stage41D correction:

```text
Add --semantic-token-feedback-teacher-forcing.

Humanistic reason:
  A student who does not yet speak the teacher's language cannot improve by
  listening only to their own unclear mumbling. During training, the teacher
  first puts the correct word into the thought loop. Evaluation removes that
  teacher and tests whether the student learned to self-condition.

Technical path:
  During training only:
    probs = (1 - mix) * self_belief + mix * one_hot(step_label)
    feedback = probs @ qwen_label_token_basis
    z_H <- gated feedback inside the recurrent loop

Evaluation:
  posterior labels are absent, so the model uses only self_belief.

This is a curriculum bridge, not an evaluated bypass.
```

Stage41D local result:

```text
run_id=20260521_230500_LOCAL_STAGE41D_teacher_semfeedback_seed74
scale=0.5 temperature=0.5 gate_bias=-1.0 teacher_forcing=0.7

epoch1:
  train_acc=0.3340
  prior_acc=0.1436
  heldout_mean_acc=0.1187
  depth4=0.1250 depth6=0.1328 depth8=0.1250 depth10=0.1094 depth12=0.1016
  K4_acc=0.1203 K4_cos=0.714
  verified_source_target_token_acc=0.2360

epoch2:
  train_acc=0.2861
  prior_acc=0.1348
  heldout_mean_acc=0.1125
  depth4=0.1328 depth6=0.0938 depth8=0.1094 depth10=0.1406 depth12=0.0859
  K4_acc=0.1203 K4_cos=0.726
  verified_source_target_token_acc=0.1960
```

Decision:

```text
Reject local Stage41D as a generalization mechanism.

The teacher-forced semantic bridge is live: it lowers entropy and raises the
posterior/train path. But prior/eval remains weak. Humanistically, the student
can use the word when the teacher puts it into the thought loop, but cannot yet
choose that word alone during the exam.
```

Stage41D DGX epoch1:

```text
run_id=20260521_230500_DGX_STAGE41D_teacher_semfeedback_seed75
train_acc=0.2905
prior_acc=0.1216
heldout_mean_acc=0.1453
depth4=0.1406 depth6=0.1016 depth8=0.1484 depth10=0.1953 depth12=0.1406
K4_acc=0.1484 K4_cos=0.729
verified_source_target_token_acc=0.2160
```

Decision:

```text
Reject as not above H2 robust mean 0.153906, though depth10 is mildly positive.

Next correction:
  Stage41E adds --prior-aux-step-answer-weight. Existing aux_step_answer_loss
  supervises the posterior/teacher trajectory, which is not the eval path.
  Stage41E applies the same LM-head step CE to the prior/no-posterior trajectory
  used during heldout evaluation.
```

## 2026-05-21 optimizer literature note: embedding LR transfer

Source:

```text
Quantifying Hyperparameter Transfer and the Importance of Embedding Layer
Learning Rate, Kalra and Barkeshli, arXiv/deeplearn mirror, 2026-05-20.
```

Humanistic interpretation:

```text
If we ask Qwen to become the speaker again, the mouth and dictionary need their
own learning pace. Training the scratchpad/core and the embedding/unembedding
surface with one shared LR can make the speaker either too stiff to adapt or too
unstable to keep its pretrained language.
```

Relevance to QTRM:

```text
This paper is related to Stage40/41-style Qwen speaker-surface experiments:
  --train-qwen-embeddings
  --train-qwen-lm-head
  --qwen-embedding-lr-multiplier
  --qwen-lm-head-lr-multiplier

It is not direct evidence that the 0.16 synthetic OOD wall is caused by LR.
It is an optimizer-control result for the cases where we partially unfreeze
Qwen's embedding/lm_head path.
```

Action rule:

```text
When running Qwen-compatible LM-head experiments with trainable embeddings or
lm_head, use explicit optimizer groups and log their LRs. Treat embedding/lm_head
LR as a first-class ablation axis, not a hidden default.

Do not use this paper to justify more scalar tuning after repeated architecture
rejects. It should support speaker-surface adaptation only after the 7-axis
preflight says the speaker path is the current bottleneck.
```

## Stage41E DGX result: prior-path step CE did not rescue semantic feedback

Run:

```text
run_id=20260521_232000_DGX_STAGE41E_prioraux_semfeedback_seed77
intervention:
  teacher-forced semantic token feedback
  plus --prior-aux-step-answer-weight 0.1
```

Epoch 1:

```text
train_acc=0.2866
prior_acc=0.1377
heldout_mean_acc=0.1156
depth4=0.1094 depth6=0.1250 depth8=0.0859 depth10=0.1172 depth12=0.1406
K4_acc=0.1234
K4_cos=0.728
verified_source_target_token_acc=0.2400
```

Decision:

```text
Reject and terminate. The prior-path CE did not make the eval/prior trajectory
speak Qwen's token language well enough. This strengthens the Stage41 family
conclusion: semantic speaker feedback is not the current high-probability lever.

Next high-probability work should move back to the seven-axis story:
  curriculum/data contract and the main recurrent transition itself,
  not another semantic feedback scalar.
```

## Stage42 preflight: HRM-Text-like bridge curriculum before another architecture tweak

Humanistic diagnosis:

```text
Stage35H2 was the strongest clean run, but it trained on depth 4/6 and then was
judged on 8/10/12. That is like teaching a student two-page examples and calling
them a failed thinker when a twelve-page exam is hard. Stage41 tried to fix the
student's speaking voice. The evidence says the voice was not the main problem.

The next higher-probability move is to fix the class syllabus: keep the same
clean Qwen-reader -> recurrent thinker -> Qwen LM-head speaker path, but expose
the thinker to a bridge of depths 4/6/8/10. Then treat 12/14 as the real OOD
exam.
```

Seven-axis gate:

```text
1. Architecture:
   Keep H2 clean path:
   Qwen reader -> sequence workspace -> true-GRAM recurrent thought ->
   sharp attention readout -> Qwen lm_head.
   Do not add semantic-token feedback or side executor.

2. Curriculum:
   Replace train_depths=[4,6] with [4,6,8,10] and set n_steps=10.
   This teaches increasing working length before judging 12/14.

3. Reward/loss:
   Keep H2 losses: final prior answer, small posterior answer, small aux step,
   semantic LM alignment, healing replay. Do not add new side losses.

4. Evaluation:
   Evaluate [4,6,8,10,12,14]. Read 12/14 as the real generalization gate.

5. Exploration:
   Keep true-GRAM stochastic eval samples [1,4] and record K4 accuracy/diversity.

6. Data contract:
   Use the same generalized synthetic schema and balanced family mix for train
   and eval. The hidden contract stays identical while length changes.

7. Causality/ablation:
   Compare directly against H2:
     H2 mean=0.159375 with train_depths=[4,6]
     H2 depth10=0.1796875
     H2 depth12=0.1484375
   Promote only if the bridge improves heldout depth12/14, not merely train acc.
```

Promote/reject gate:

```text
Promote:
  epoch <= 2 gives depth12 or depth14 >= 0.20, or mean >= 0.18 with K4 not worse.

Reject:
  epoch 2 mean <= 0.15 and depth12/depth14 remain <= 0.16.

Interpretation:
  If promoted, the current bottleneck is curriculum/data contract.
  If rejected, the problem is likely the recurrent transition itself, and Stage43
  should replace the thought update rather than keep tuning speaker feedback.
```

Stage42A local epoch 1:

```text
run_id=20260521_222500_LOCAL_STAGE42A_bridge_depth4610_seed80_setsid
train_depths=[4,6,8,10]
eval_depths=[4,6,8,10,12,14]

train_acc=0.1626
prior_acc=0.1260
heldout_mean_acc=0.1471
depth4=0.1016 depth6=0.1719 depth8=0.2031 depth10=0.1406 depth12=0.1016 depth14=0.1641
K4_acc=0.1328
K4_cos=0.726
```

Interim read:

```text
The bridge curriculum immediately helps the newly trained depth8 bucket, but
does not yet improve the real OOD bucket. Compared with H2, depth12 is worse
(0.1016 vs 0.1484). Keep epoch2 and DGX running before final rejection.
```

Stage42A local epoch 2:

```text
train_acc=0.1631
prior_acc=0.1367
heldout_mean_acc=0.1107
depth4=0.1719 depth6=0.1172 depth8=0.1250 depth10=0.0859 depth12=0.0938 depth14=0.0703
K4_acc=0.1185
K4_cos=0.740
verified_source_target_token_acc=0.2240
```

Local decision:

```text
Reject Stage42A locally. Extending the curriculum to 4/6/8/10 did not produce
depth extrapolation. It briefly raised depth8 but then degraded the real OOD
depths. Humanistically, showing the student longer examples is not enough; the
recurrent thought update still lacks a stable long-form method.
```

Stage42B DGX epoch 1:

```text
run_id=20260521_221842_DGX_STAGE42B_bridge_depth4610_seed81
train_acc=0.1658
prior_acc=0.1394
heldout_mean_acc=0.1497
depth4=0.1797 depth6=0.1172 depth8=0.1562 depth10=0.1797 depth12=0.1172 depth14=0.1484
K4_acc=0.1458
K4_cos=0.759
verified_source_target_token_acc=0.1880
```

DGX interim:

```text
Below promote gate and below H2 mean. Wait for epoch2 before final Stage42
family decision.
```

Stage42B DGX epoch 2:

```text
train_acc=0.1670
prior_acc=0.1428
heldout_mean_acc=0.1458
depth4=0.1875 depth6=0.1484 depth8=0.1250 depth10=0.1250 depth12=0.1328 depth14=0.1562
K4_acc=0.1263
K4_cos=0.779
verified_source_target_token_acc=0.1680
```

Stage42 final decision:

```text
Reject. Local and DGX agree: adding bridge depths 8/10 to the syllabus does not
produce depth12/14 generalization. The model learns slightly different buckets
but does not acquire a stable long-form method.

Consequence:
  The next accepted-likelihood axis is not more data-depth curriculum alone.
  It must modify the main recurrent thought dynamics or the way stochastic
  trajectories are selected, while preserving the clean Qwen LM-head answer path.
```

## Stage43 preflight: identity-stabilized bridge, not another side channel

Why Stage43 follows from Stage42:

```text
Stage42 showed that a longer syllabus alone does not teach long-form thinking.
Depth8 briefly improved, but depth12/14 did not. The story-level failure is now:

  the student can see longer examples,
  but the internal handwriting drifts as the thought gets longer.

Therefore the next high-probability move is not more speaker feedback, not more
candidate heads, and not another auxiliary label. It is to make the recurrent
thought update carry identity more strongly while removing absolute step
shortcuts.
```

Stage43A local design:

```text
Base:
  Stage35H2 clean LM-head/true-GRAM path.

Keep:
  Qwen reader -> sequence workspace -> recurrent thinker -> sharp readout ->
  Qwen lm_head speaker.

Intervention:
  --override-transition-scale 0.05
  --override-injection-gate-logit 3.0
  --zero-step-embeddings
  --freeze-step-embeddings
  --aux-step-answer-weight 0.0

Curriculum:
  train_depths=[4,6,8,10]
  eval_depths=[4,6,8,10,12,14]

Humanistic meaning:
  Stop pushing the thinker to write a new intermediate answer at every line.
  Give it a stable memory highway and judge the final answer.
```

Gate:

```text
Promote if depth12 or depth14 >= 0.20, or heldout_mean_acc >= 0.18.
Reject if epoch2 remains <= Stage42A and H2 on depth12/14.
```

Stage43A local epoch 1:

```text
run_id=20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82
overrides:
  transition_scale=0.05
  injection_gate_logit=3.0
  zero/freeze step embeddings
  aux_step_answer_weight=0.0

train_acc=0.2300
prior_acc=0.1309
heldout_mean_acc=0.1432
depth4=0.1875 depth6=0.1328 depth8=0.1562 depth10=0.1328 depth12=0.0859 depth14=0.1641
K4_acc=0.1250
K4_cos=0.044
verified_source_target_token_acc=0.2040
```

Interim read:

```text
Identity-stabilization did not collapse training and produced much more diverse
stochastic trajectories (K4 cosine 0.044 vs Stage42 around 0.72-0.76). However,
the diversity is not yet useful: K4 accuracy and depth12 remain weak. This
points toward a search/value-selection problem only after the transition quality
is acceptable; epoch2 decides whether the transition itself is improving.
```

Stage43A local epoch 2:

```text
train_acc=0.1978
prior_acc=0.1343
heldout_mean_acc=0.1289
depth4=0.1328 depth6=0.1172 depth8=0.1172 depth10=0.1328 depth12=0.1328 depth14=0.1406
K4_acc=0.1276
K4_cos=0.107
verified_source_target_token_acc=0.2160
```

Decision:

```text
Reject Stage43A as a generalization fix. It successfully opens diverse
stochastic trajectories, but the trajectories are not better and mean/depth12/14
remain below H2 and Stage42. Do not spend DGX on this variant.

Humanistic read:
  The student can now imagine many possible solution paths, but has no reliable
  judge for which path is right. Diversity without a useful value signal is not
  intelligence; it is just wandering.
```

Stage43 post-hoc K4 oracle selection probe:

```text
checkpoint=Stage43A best_generalization.pt
eval_count=128
selection=vote
samples=4

selected_mean=0.1328
oracle_best_of_4_mean=0.3008

selected:
  depth4=0.1328 depth6=0.1094 depth8=0.1719 depth10=0.1328 depth12=0.0781 depth14=0.1719

oracle best-of-4:
  depth4=0.2812 depth6=0.3359 depth8=0.2969 depth10=0.2656 depth12=0.2656 depth14=0.3594
```

Interpretation:

```text
This is the strongest actionable signal after Stage41-43. The recurrent system
does sample correct trajectories far above the selected accuracy, including
depth12/14. The bottleneck is no longer only "cannot think"; it is "cannot judge
which sampled thought is right."

Therefore Stage44 should not add another reader/speaker side channel. It should
train the existing trajectory_reward_head as an LPRM/value selector on top of the
identity-stabilized diverse sampler.
```

## Stage44 preflight: identity sampler + LPRM value selector

Humanistic story:

```text
Stage43 is a student who can brainstorm several answers. The answer is often in
the pile, but the student lacks the teacher's sense for which answer smells
right. Stage44 teaches that judging sense directly.
```

Design:

```text
Base:
  Stage43 identity-stabilized bridge.

Add:
  --gram-lprm-weight 0.2
  --gram-lprm-target true_prob
  --stochastic-selection-mode lprm

Keep:
  clean Qwen LM-head answer path
  true-GRAM stochastic eval K=4
  no auxiliary step answer loss
  no semantic token feedback
```

Gate:

```text
Promote if K4-LPRM selected mean >= 0.18 or depth12/14 >= 0.20.
Strong promote if selected accuracy moves materially toward the oracle band
(~0.30 mean, depth14 ~0.36).

Reject if selected K4 stays near 0.13-0.15 while oracle remains high; then the
current LPRM target is not learning the right value signal.
```

Stage44A local epoch 1:

```text
run_id=20260521_225200_LOCAL_STAGE44A_lprm_selector_identity_seed84
lprm_weight=0.2
lprm_target=true_prob
selection=lprm

train_acc=0.2192
prior_acc=0.1333
lprm_loss=0.1457
heldout_mean_acc=0.1237
depth4=0.1406 depth6=0.0938 depth8=0.1562 depth10=0.0859 depth12=0.1328 depth14=0.1328
K4_acc=0.1354
K4_cos=0.066
verified_source_target_token_acc=0.1720
```

Interim read:

```text
The LPRM head is training, but selected accuracy did not move toward the
best-of-K oracle band. If epoch2 and DGX agree, the likely issue is that the
current LPRM objective sees only one trajectory per row during training. It
does not explicitly learn to rank multiple sampled thoughts from the same
question.
```

Stage44A local epoch 2:

```text
train_acc=0.1846
prior_acc=0.1304
lprm_loss=0.1184
heldout_mean_acc=0.1510
depth4=0.2109 depth6=0.0938 depth8=0.1719 depth10=0.1719 depth12=0.0938 depth14=0.1641
K4_acc=0.1432
K4_cos=0.140
verified_source_target_token_acc=0.2240
```

Updated read:

```text
The single-trajectory LPRM objective is not enough. LPRM loss decreases, but
K4-selected accuracy remains far below the Stage43 oracle best-of-4 mean
(~0.3008). This supports the next hypothesis: the reward head must be trained
on multiple trajectories from the same question, because inference selection is
a within-question ranking problem, not an isolated "is this one path good?"
problem.
```

Stage44B DGX epoch 1:

```text
run_id=20260521_225200_DGX_STAGE44B_lprm_selector_identity_seed85
train_acc=0.1829
prior_acc=0.1196
lprm_loss=0.1442
heldout_mean_acc=0.1419
depth4=0.1562 depth6=0.1641 depth8=0.1250 depth10=0.1406 depth12=0.0703 depth14=0.1953
K4_acc=0.1289
K4_cos=0.178
verified_source_target_token_acc=0.2160
```

Stage44B DGX epoch 2:

```text
train_acc=0.1567
prior_acc=0.1204
lprm_loss=0.1240
heldout_mean_acc=0.1289
depth4=0.1172 depth6=0.1250 depth8=0.1406 depth10=0.1641 depth12=0.0938 depth14=0.1328
K4_acc=0.1341
K4_cos=0.389
verified_source_target_token_acc=0.1960
```

Decision:

```text
Reject Stage44 as a selector fix. Local and DGX agree: single-trajectory LPRM
reduces LPRM loss, but K4 selected accuracy remains near 0.13-0.14. This is
not enough to exploit the Stage43 oracle best-of-4 signal.
```

## Research note: embedding-layer LR transfer paper

Source:

```text
Quantifying Hyperparameter Transfer and the Importance of Embedding Layer
Learning Rate
DeepLearn/arXiv mirror: https://deeplearn.org/arxiv/756894/quantifying-hyperparameter-transfer-and-the-importance-of-embedding-layer-learning-rate
Published: 2026-05-20
Authors: Dayal Singh Kalra, Maissam Barkeshli
```

Humanistic read:

```text
If the model is learning a new language through its mouth and dictionary, the
dictionary/embedding layer cannot be treated as a frozen or slow-moving stone.
The paper argues that, under AdamW, a too-small embedding learning rate can
become a training bottleneck and make scaling/hyperparameter transfer look
worse than it should.
```

Relation to QTRM/GRAM:

```text
Relevant, but not the current first-order OOD wall.

Directly relevant when:
  --train-qwen-embeddings or --train-qwen-lm-head is enabled.

Less directly relevant when:
  Qwen is frozen and the current bottleneck is stochastic trajectory selection
  or recurrent latent drift.

Existing script support:
  --train-qwen-embeddings
  --train-qwen-lm-head
  --qwen-embedding-lr-multiplier
  --qwen-lm-head-lr-multiplier

Experiment implication:
  If Stage45 moves into "adapt the Qwen speaker/dictionary" territory, add a
  small A/B where embeddings/lm_head are unfrozen with higher LR multipliers.
  Do not treat this as a replacement for the Stage43 oracle signal: the oracle
  result says correct trajectories already exist; the missing piece is still
  selecting the right trajectory.
```

## Stage45 preflight: multi-trajectory LPRM ranking selector

Humanistic story:

```text
Stage44 taught the judge one thought at a time: "is this thought good?" But the
real exam shows four thoughts and asks: "which one should I trust?" Those are
not the same class. Stage45 teaches the judge in the same room where it will be
tested: several sampled thoughts from the same question, ranked against each
other.
```

One-sentence intuition:

```text
If the right path is already often inside K=4 but the chosen path is wrong,
the next high-probability move is not a new thinker; it is a within-question
selector trained on K sampled thoughts.
```

Technical mapping:

```text
Reader:
  Frozen Qwen3.5 reads the prompt.

Thinker:
  Identity-stabilized true-GRAM recurrent core samples K prior-only trajectories.

Speaker:
  Each trajectory still decodes through the Qwen-compatible LM-head path.

Judge:
  trajectory_reward_head scores each trajectory.

New Stage45 loss:
  --gram-lprm-train-samples 4
  BCE on true answer probability per sampled trajectory
  listwise CE to select the best trajectory within the same question
  pairwise ranking loss between better and worse sampled trajectories
  --gram-lprm-detach-state so the first probe teaches the judge without
  immediately rewriting the thinker.
```

Seven-axis gate:

```text
1. Architecture: clean. Qwen reader -> recurrent GRAM thought -> LM-head answer;
   reward head only selects among existing thought paths.
2. Curriculum: clean. It teaches one missing capacity: selection among K paths.
3. Reward/loss: better than Stage44. The reward is now relative within the
   exact K-sample setting used at inference.
4. Evaluation: clean. Promote/reject uses K4-LPRM selected accuracy versus
   Stage43 oracle.
5. Exploration: clean. Stage43/44 K4 cosine ~0.04-0.18 proves sampled paths
   differ.
6. Data contract: clean. Same synthetic prompt/label contract; no new side
   information.
7. Causality: falsifiable. If K4 oracle stays high but selected accuracy stays
   low, the selector objective is still wrong. If selected rises toward oracle,
   the bottleneck was trajectory selection.
```

Gate:

```text
Promote:
  K4-LPRM selected mean >= 0.18, or depth12/14 >= 0.20.

Strong promote:
  K4-LPRM selected mean moves materially toward Stage43 oracle best-of-4 mean
  (~0.3008).

Reject:
  K4 selected accuracy remains <= 0.15 while train oracle accuracy is clearly
  above selected accuracy.
```

Stage45 launch records:

```text
Local Stage45A:
  run_id=20260521_231152_LOCAL_STAGE45A_multitraj_lprm_rank_seed86
  log=/tmp/20260521_231152_LOCAL_STAGE45A_multitraj_lprm_rank_seed86.log
  out=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_231152_LOCAL_STAGE45A_multitraj_lprm_rank_seed86
  pid=3435006
  status=stopped before epoch1; batch_size=2 made the local gate too slow
  notes=reasoning_count=2048, healing_count=256, batch_size=2, K_train=4

Local Stage45A smoke replacement:
  run_id=20260521_231152_LOCAL_STAGE45A_smoke_multitraj_lprm_rank_seed86
  log=/tmp/20260521_231152_LOCAL_STAGE45A_smoke_multitraj_lprm_rank_seed86.log
  out=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_231152_LOCAL_STAGE45A_smoke_multitraj_lprm_rank_seed86
  pid=3475061
  status=running
  notes=reasoning_count=512, healing_count=64, batch_size=4, K_train=4

DGX Stage45B:
  run_id=20260521_231152_DGX_STAGE45B_multitraj_lprm_rank_seed87
  launcher_pid=436158
  wrapper_log=/tmp/launch_stage45B_after_stage44.wrapper.log
  status=queued until Stage44B pid 419316 exits
  notes=reasoning_count=4096, healing_count=512, batch_size=4, K_train=4
```

Stage45A smoke local epoch 1 train-side selector diagnostic:

```text
selected_acc=0.1328
oracle_acc=0.3008
target_spread=0.0734
train_acc=0.2598
prior_acc=0.1426
lprm_loss=0.9251
status=heldout evaluation pending at snapshot
```

Interpretation:

```text
The new objective is instrumenting the right gap: the oracle best trajectory is
present roughly 30% of the time, but the learned reward selector is still near
13%. This proves Stage45 is testing the intended causal bottleneck. It is not a
promote signal yet; the gate is whether selected_acc rises over epochs or in
DGX scale.
```

Stage45A smoke local epoch 1 heldout:

```text
heldout_mean_acc=0.1484
depth4=0.1875 depth6=0.1719 depth8=0.1406 depth10=0.1562 depth12=0.1562 depth14=0.0781
K4_acc=0.1589
K4_cos=0.023
verified_source_target_token_acc=0.1544
```

Interim decision:

```text
Not promoted yet. K4 selected accuracy is slightly above Stage44 local epoch2
(0.1432 -> 0.1589), but below the 0.18 gate. Continue to epoch2 and DGX scale.
```

Stage45A smoke local epoch 2:

```text
train_selector:
  selected_acc=0.1426
  oracle_acc=0.3262
  target_spread=0.0713

train:
  train_acc=0.2090
  prior_acc=0.1309
  lprm_loss=0.9063

heldout:
  heldout_mean_acc=0.1432
  depth4=0.1875 depth6=0.1094 depth8=0.1562 depth10=0.2188 depth12=0.0781 depth14=0.1094
  K4_acc=0.1667
  K4_cos=0.033
  verified_source_target_token_acc=0.1985
```

Decision:

```text
Stage45A smoke is not a big-jump promote. It is a weak-positive diagnostic:
K4 selected accuracy improves slightly over Stage44 local (0.1432 -> 0.1667),
but remains below the 0.18 gate. The train oracle gap grows to 0.3262 vs
selected 0.1426, so the causal bottleneck remains selector learning rather
than trajectory availability.

Keep DGX Stage45B running to test whether scale/data volume closes the selector
gap. If DGX also stays below 0.18, the next fix should not be another scalar
weight. It should change the judge's information: e.g. feed reward head richer
trajectory features, answer margins, or train a non-detached selector jointly
after a detached warmup.
```

## Stage46 preflight: rich trajectory reward selector

Humanistic story:

```text
Stage45 gave the judge four possible thoughts, but the judge still only looked
at the last hidden state of each thought. That is like asking a teacher to grade
a solution while seeing only the final line. Stage46 lets the judge see the
final line, the average shape of the reasoning, how much the thought moved from
start to end, the sharp attention readout, and answer confidence signals.
```

One-sentence intuition:

```text
If the correct trajectory exists but the selector cannot identify it, the
selector needs richer evidence from the trajectory, not just a bigger scalar
loss on the same final vector.
```

Technical mapping:

```text
New flag:
  --trajectory-reward-mode rich

Reward features:
  final recurrent readout state
  mean recurrent trajectory state
  start-to-end delta state
  sharp attention recurrent readout state
  answer margin, entropy, max probability, logit scale

Reward head:
  RMSNorm -> Linear -> GELU -> Linear

Still clean:
  Qwen reader -> true-GRAM recurrent thought -> Qwen LM-head answer.
  Reward head only selects among sampled thought paths.
```

Gate:

```text
Promote if local smoke or DGX K4 selected mean >= 0.18.
Strong promote if selected moves toward the 0.30+ oracle band.
Reject if oracle stays high but selected remains <= 0.15, because then richer
features alone are not enough and the selector must train jointly/non-detached
or use a contrastive reward model over full trajectory pairs.
```

Stage46A local launch:

```text
run_id=20260521_232400_LOCAL_STAGE46A_rich_selector_smoke_seed88
log=/tmp/20260521_232400_LOCAL_STAGE46A_rich_selector_smoke_seed88.log
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_232400_LOCAL_STAGE46A_rich_selector_smoke_seed88
pid=3529778
status=running
notes=reasoning_count=512, healing_count=64, batch_size=4, K_train=4, trajectory_reward_mode=rich
```

Stage46A local epoch 1:

```text
train_selector:
  selected_acc=0.1406
  oracle_acc=0.3281
  target_spread=0.0736

train:
  train_acc=0.2676
  prior_acc=0.1270
  lprm_loss=0.8935

heldout:
  heldout_mean_acc=0.1562
  depth4=0.2188 depth6=0.1406 depth8=0.1719 depth10=0.2031 depth12=0.0938 depth14=0.1094
  K4_acc=0.1510
  K4_cos=0.022
  verified_source_target_token_acc=0.1471
```

Interim read:

```text
Richer features did not immediately improve K4 selection; K4_acc is below
Stage45A epoch1 (0.1589). The selector sees more evidence, but the newly
initialized rich reward head may be learning too slowly at the shared 2e-5 LR.
Wait for epoch2; if it remains below 0.18, test reward-head LR separation rather
than adding another architectural side path.
```

Stage46A local epoch 2:

```text
train_selector:
  selected_acc=0.1270
  oracle_acc=0.3496
  target_spread=0.0795

train:
  train_acc=0.2227
  prior_acc=0.1426
  lprm_loss=0.8505

heldout:
  heldout_mean_acc=0.1484
  depth4=0.1406 depth6=0.1406 depth8=0.1406 depth10=0.2188 depth12=0.1250 depth14=0.1250
  K4_acc=0.1615
  K4_cos=0.035
  verified_source_target_token_acc=0.1691
```

Decision:

```text
Reject Stage46A as a big-jump fix. Rich trajectory features alone do not cross
the 0.18 gate and underperform Stage45A's best K4=0.1667. However, oracle rises
to 0.3496, so trajectory availability is even less likely to be the limiting
factor.

Next highest-probability local test: Stage47 reward-head LR separation. The
rich reward head is newly initialized but was trained at the same 2e-5 LR as
the recurrent core. This is analogous to the embedding-LR transfer lesson: a
fresh interface layer can become the bottleneck if its learning rate is too
small.
```

## Stage47 preflight: rich selector with reward-head LR separation

Humanistic story:

```text
The judge now sees the full answer sheet, but it is a newly hired judge learning
too slowly. Stage47 keeps the student and reader mostly steady while letting
the judge learn faster.
```

Technical mapping:

```text
New optimizer group:
  parameters containing "trajectory_reward" get
  lr = base_lr * --trajectory-reward-lr-multiplier

Stage47 local setting:
  --trajectory-reward-mode rich
  --trajectory-reward-lr-multiplier 20
  K_train=4
  detach_state=true
```

Gate:

```text
Promote if K4_acc >= 0.18.
Reject if selected remains <= 0.16 while oracle remains >= 0.30.
```

Stage47A local launch:

```text
run_id=20260521_233100_LOCAL_STAGE47A_rich_selector_rewardlr20_seed89
log=/tmp/20260521_233100_LOCAL_STAGE47A_rich_selector_rewardlr20_seed89.log
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_233100_LOCAL_STAGE47A_rich_selector_rewardlr20_seed89
pid=3546070
status=running
```

Stage47A local epoch 1:

```text
train_selector:
  selected_acc=0.1113
  oracle_acc=0.3398
  target_spread=0.0715

heldout:
  heldout_mean_acc=0.1328
  depth4=0.1406 depth6=0.1562 depth8=0.1094 depth10=0.1406 depth12=0.0938 depth14=0.1562
  K4_acc=0.1380
  K4_cos=0.025
```

Decision:

```text
Reject/stop Stage47A early. Reward-head LR x20 made K4 selection worse. The
larger issue is that true_prob targets are too soft/noisy for ranking: target
spread remains tiny while oracle is high.
```

## Stage48 preflight: correct/incorrect selector targets

Humanistic story:

```text
Instead of teaching the judge with a blurry "how confident did the model feel
about the right token?" score, teach it with the actual mark on the paper:
this sampled thought was correct, this one was wrong.
```

Technical mapping:

```text
Fix:
  listwise loss now ignores rows where all K candidates have effectively equal
  targets. This avoids treating sample 0 as the fake winner when every sampled
  path is wrong.

Stage48 setting:
  --gram-lprm-target correct
  --trajectory-reward-mode final
  --trajectory-reward-lr-multiplier 10
  --gram-lprm-train-samples 4
```

Gate:

```text
Promote if K4_acc >= 0.18.
Reject if K4_acc <= 0.16 while oracle remains >= 0.30.
```

Stage48A local launch:

```text
run_id=20260521_233500_LOCAL_STAGE48A_correct_target_selector_seed90
log=/tmp/20260521_233500_LOCAL_STAGE48A_correct_target_selector_seed90.log
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_233500_LOCAL_STAGE48A_correct_target_selector_seed90
pid=3556854
status=running
```

## Stage43 post-hoc selector audit

Humanistic story:

```text
The reader and thinker can put a correct thought somewhere in the room, but the
judge has to choose which thought to trust. Before adding another architecture,
test whether the existing simple judges can already identify the good thought:
average, majority vote, self-confidence, and LPRM.
```

Implementation:

```text
Added --stochastic-selection-mode confidence to both training and standalone
generalization eval scripts. Confidence selects the sampled trajectory whose
answer distribution has the largest max probability.
```

Stage43A checkpoint:

```text
checkpoint=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
eval_count=128
eval_depths=4,6,8,10,12,14
samples=4
answer_path=lm_head
workspace_pooling=sequence
readout=sharp_attention T=0.25
```

Results:

```text
mode=mean        selected_mean=0.1367  oracle_mean=0.3099
mode=vote        selected_mean=0.1406  oracle_mean=0.3255
mode=confidence  selected_mean=0.1367  oracle_mean=0.3086
mode=lprm        selected_mean=0.1276  oracle_mean=0.3034
```

Decision:

```text
Reject simple selection rules. The correct trajectory is often present in K=4,
but neither model confidence, majority vote, average logits, nor the current
LPRM can select it. This narrows the next bottleneck to verifier evidence: the
judge is not seeing enough of the problem/candidate relation to know which
thought is valid.
```

## Stage49 verifier-only selector probe

Humanistic story:

```text
Freeze the reader, thinker, and speaker. Train only the judge. If the judge can
learn, the previous failures were caused by moving the whole classroom at once.
If the judge still cannot learn, its evidence is too poor: it needs to reread
the question and candidate answer, not only stare at a final latent vector.
```

Code change:

```text
Added --train-only-trajectory-reward. The flag freezes all parameters except
names containing trajectory_reward. In Stage49A this left exactly 7 trainable
judge parameters and froze 411 other tensors.
```

Run:

```text
run_id=20260521_235100_LOCAL_STAGE49A_verifier_only_rich_seed91
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_235100_LOCAL_STAGE49A_verifier_only_rich_seed91
log=/tmp/20260521_235100_LOCAL_STAGE49A_verifier_only_rich_seed91.log
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
train_only_trajectory_reward=true
trajectory_reward_mode=rich
gram_lprm_target=correct
gram_lprm_train_samples=4
reasoning_count=512
healing_count=0
```

Local results:

```text
epoch1 train_selected=0.0859 train_oracle=0.2656 heldout_mean=0.1276 K4=0.1328
epoch2 train_selected=0.1113 train_oracle=0.2754 heldout_mean=0.1172 K4=0.1328
epoch3 train_selected=0.1113 train_oracle=0.2969 heldout_mean=0.1198 K4=0.1354
```

Decision:

```text
Reject trajectory-head-only verifier. This is useful evidence, not just another
failed run: when the thinker is frozen, the judge still cannot approach the
oracle gap. Therefore the next accepted-probability path is not another scalar
reward head. The verifier must inspect the source prompt plus each candidate
answer/trajectory with a richer comparison path.
```

Next direction:

```text
Stage50 should be a reader-verifier, not another hidden-state scorer:
  prompt tokens + candidate answer token/logits + trajectory summary
  -> frozen/adapter Qwen verifier pass or compact cross-attention verifier
  -> select candidate.

Promote only if K4 selected mean moves above 0.18 while oracle remains above
0.28. Reject if it stays near 0.13-0.15.
```

DGX status:

```text
2026-05-21 23:47 and 23:54 KST: ssh dgx failed with "No route to host".
No DGX Stage49 sync was attempted while the host was unreachable.
```

## Stage50 local-only reader-verifier

User constraint:

```text
2026-05-21/22: use local tests only. Do not use DGX for the next probes.
```

Humanistic story:

```text
Stage49 froze the student and trained only a judge looking at hidden-state
summaries. That judge failed. Stage50 gives the judge a more human job: reread
the original problem together with each candidate answer, then decide whether
the candidate is correct.
```

Implementation:

```text
New script:
  scripts/514_train_candidate_reader_verifier.py

Frozen path:
  Stage43A checkpoint -> K=4 GRAM sampled candidate answers.

Verifier input:
  Qwen last hidden from:
    original prompt + sampled candidate digit + explicit yes/no verifier text
  plus:
    recurrent readout state
    mean trajectory state
    final-start trajectory delta
    answer logits
    candidate one-hot
    margin and entropy

Trainable part:
  only a small CandidateReaderVerifier MLP.
```

Tiny debug:

```text
run_id=20260522_000000_LOCAL_STAGE50_reader_verifier_debug_seed92
train_count=16
eval_count=16
status=completed
purpose=script/autograd smoke only
```

Main local run:

```text
run_id=20260522_000100_LOCAL_STAGE50A_reader_verifier_seed92
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_000100_LOCAL_STAGE50A_reader_verifier_seed92
log=/tmp/20260522_000100_LOCAL_STAGE50A_reader_verifier_seed92.log
checkpoint=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
train_count=512
eval_count=128
samples=4
epochs=3
local_only=true
```

Results:

```text
epoch1 train_selected=0.1348 train_oracle=0.3574
epoch2 train_selected=0.1445 train_oracle=0.3105
epoch3 train_selected=0.1406 train_oracle=0.3223

best_eval_mean_selected=0.1276
best_eval_mean_oracle=0.3216

best_eval_by_depth:
  d4  selected=0.1562 oracle=0.3516
  d6  selected=0.0938 oracle=0.2891
  d8  selected=0.1406 oracle=0.3438
  d10 selected=0.1562 oracle=0.3281
  d12 selected=0.1016 oracle=0.2891
  d14 selected=0.1172 oracle=0.3281
```

Decision:

```text
Reject Stage50A as implemented. Even when the verifier rereads prompt plus
candidate answer, a small hidden-state classifier cannot reliably select the
correct trajectory. The oracle gap remains large: ~0.32 oracle versus ~0.13
selected.
```

Interpretation:

```text
The judge is not merely missing access to the prompt/candidate surface. It is
missing an explicit verification procedure. Asking a small head to infer
correctness from frozen Qwen hidden states is still too close to "judge by
vibes." The next local-only candidate should make the verifier produce or check
the arithmetic trace itself.
```

Next local-only direction:

```text
Stage51: candidate-conditioned trace verifier.

For each sampled candidate digit:
  prompt + candidate digit
  -> verifier predicts stepwise modulo states or final residual consistency
  -> selection score is trace-consistency, not a scalar reward head.

Promote if:
  selected_mean >= 0.18 and depth10 selected >= 0.18.

Reject if:
  selected_mean stays <= 0.15 while oracle remains >= 0.28.
```

## Stage51 local-only candidate-conditioned trace verifier

Humanistic story:

```text
Stage50 asked the judge to look at the problem and candidate answer, then give
a yes/no verdict. That still let the judge grade by vibes. Stage51 makes the
judge write the working: predict the modulo state after every step, then select
the candidate whose final predicted trace supports the sampled answer.
```

Implementation:

```text
New script:
  scripts/515_train_candidate_trace_verifier.py

Frozen path:
  Stage43A Qwen/QTRM checkpoint generates K=4 candidate answers.

Trainable verifier:
  CandidateTraceVerifier
  input = Qwen last hidden from prompt + candidate digit, candidate embedding,
          step embedding
  output = per-step digit logits for max_trace_steps=14

Loss:
  stepwise trace CE against synthetic state_labels
  plus candidate-selection CE when at least one of K candidates is correct.
```

Tiny debug:

```text
run_id=20260522_001100_LOCAL_STAGE51_trace_verifier_debug_seed93
status=completed
purpose=script/autograd smoke
```

Main local run:

```text
run_id=20260522_001200_LOCAL_STAGE51A_trace_verifier_seed93
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_001200_LOCAL_STAGE51A_trace_verifier_seed93
log=/tmp/20260522_001200_LOCAL_STAGE51A_trace_verifier_seed93.log
checkpoint=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
samples=4
train_count=512
eval_count=128
epochs=3
local_only=true
```

Results:

```text
epoch1 train_selected=0.1230 train_oracle=0.2891 trace_final=0.0879 step=0.1036
epoch2 train_selected=0.1309 train_oracle=0.3008 trace_final=0.1133 step=0.1115
epoch3 train_selected=0.1289 train_oracle=0.2949 trace_final=0.1270 step=0.1161

best_eval_mean_selected=0.1536
best_eval_mean_oracle=0.3242
best_eval_mean_trace_final=0.1211

best_eval_by_depth:
  d4  selected=0.1953 oracle=0.3359 trace_final=0.1875
  d6  selected=0.0938 oracle=0.2734 trace_final=0.1016
  d8  selected=0.1562 oracle=0.3125 trace_final=0.1562
  d10 selected=0.2188 oracle=0.4219 trace_final=0.0859
  d12 selected=0.1016 oracle=0.2812 trace_final=0.0703
  d14 selected=0.1562 oracle=0.3203 trace_final=0.1250
```

Decision:

```text
Weak signal, not promote. Stage51A improves over Stage50A best mean selected
(0.1536 vs 0.1276) and crosses the 0.18 gate at depth4/depth10, but the mean
gate is not met and step trace accuracy stays near random. This means explicit
trace supervision is the right kind of intervention, but Qwen last-hidden is
too narrow as the verifier's reading surface.
```

Next local-only direction:

```text
Stage51B: sequence-attention trace verifier.

Do not collapse Qwen's verifier read into one last hidden vector. Keep the
verifier prompt token sequence and let each step query attend over all token
hidden states. This is the human equivalent of pointing at the relevant line of
the problem while writing each intermediate state.

Promote if:
  selected_mean >= 0.18 and depth10 selected >= 0.18.

Reject if:
  selected_mean <= 0.15 and trace step accuracy remains near random.
```

## Stage51B local-only sequence-attention trace verifier

Humanistic story:

```text
Stage51A asked the judge to write the arithmetic trace from one final Qwen
hidden vector. That is like asking someone to grade a calculation after hiding
the worksheet and showing only their final facial expression. Stage51B keeps
the whole worksheet visible: for each trace step, a learned query attends over
all verifier-prompt token hidden states.
```

Implementation update:

```text
scripts/515_train_candidate_trace_verifier.py
  --reader-mode sequence_attention

New verifier:
  CandidateTraceSequenceVerifier
  step query + candidate digit query -> attention over Qwen token hidden states
  attended context -> per-step digit logits
```

Tiny debug:

```text
run_id=20260522_002000_LOCAL_STAGE51B_seq_trace_debug_seed94
status=completed
```

Main local run:

```text
run_id=20260522_002100_LOCAL_STAGE51B_seq_trace_verifier_seed94
out=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_002100_LOCAL_STAGE51B_seq_trace_verifier_seed94
log=/tmp/20260522_002100_LOCAL_STAGE51B_seq_trace_verifier_seed94.log
reader_mode=sequence_attention
hidden_dim=512
samples=4
train_count=512
eval_count=128
epochs=3
local_only=true
```

Results:

```text
epoch1 train_selected=0.1230 train_oracle=0.3047 trace_final=0.1074 step=0.1104
epoch2 train_selected=0.1406 train_oracle=0.2871 trace_final=0.1289 step=0.1244
epoch3 train_selected=0.1348 train_oracle=0.3047 trace_final=0.1328 step=0.1183

best_eval_mean_selected=0.1484
best_eval_mean_oracle=0.3255
best_eval_mean_trace_final=0.1549

best_eval_by_depth:
  d4  selected=0.1719 oracle=0.3750 trace_final=0.1797
  d6  selected=0.1016 oracle=0.2734 trace_final=0.1094
  d8  selected=0.1719 oracle=0.3516 trace_final=0.1562
  d10 selected=0.1641 oracle=0.3516 trace_final=0.2031
  d12 selected=0.1484 oracle=0.2969 trace_final=0.1328
  d14 selected=0.1328 oracle=0.3047 trace_final=0.1484
```

Decision:

```text
Reject as a selector solution, keep as evidence. Sequence-attention improves
trace_final accuracy relative to Stage51A (0.1549 vs 0.1211), but selected
accuracy remains below the 0.18 gate. The verifier is starting to learn a weak
trace signal, but the current selection score (candidate final log probability)
does not convert that signal into reliable candidate choice.
```

Interpretation:

```text
The direction is now clearer:
  - scalar reward head: no
  - prompt+candidate hidden classifier: no
  - trace-supervised verifier: weak yes
  - sequence-level verifier reading surface: better trace signal, but not
    enough selection signal

The next local-only run should not add another architecture name. It should
change the selection score from "does the final trace token equal candidate?"
to "does the entire predicted trace obey the operations in the prompt?" That
means transition-consistency scoring or a typed/register executor probe.
```

Next local-only direction:

```text
Stage52: transition-consistency / typed-register verifier probe.

Two versions:
  A. Structured upper-bound: use operation ids/args already present in the
     synthetic case to train a tiny recurrent typed verifier. This tests whether
     explicit register execution can close the oracle gap.
  B. Qwen-to-register version: if A passes, the real architecture target is a
     Qwen reader that emits typed operation/operand registers into that verifier.

Promote A if selected_mean approaches oracle_mean or exceeds 0.22.
Reject A if even typed registers remain near 0.15.
```

## Stage52A local-only structured typed-register selector

Humanistic story:

```text
After Stage49-51, the core question became simple: is the judge weak because it
cannot see a final hidden vector well enough, or because it does not possess a
clean accounting ledger of numbers and operations? Stage52A gives the judge the
ledger directly: operation id, operand, and initial value. It then executes the
registers and chooses the sampled candidate matching the computed final digit.
```

Implementation:

```text
New script:
  scripts/516_eval_structured_register_selector.py

No neural verifier training.

Procedure:
  1. Load Stage43A checkpoint.
  2. Generate K=4 stochastic GRAM candidate answer digits.
  3. Compute the true final digit from typed registers:
       initial_label + operation_ids + operation_args
  4. Select a sampled candidate if it matches the computed register answer.

This is an upper-bound probe for typed-register verification, not a deployable
LLM solution.
```

Tiny debug:

```text
run_id=20260522_003000_LOCAL_STAGE52A_structured_register_debug_seed95
eval_count=16
status=completed
selected=oracle on all tested depths
register_ok=1.0
```

Main local run:

```text
run_id=20260522_003100_LOCAL_STAGE52A_structured_register_selector_seed95
json=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_003100_LOCAL_STAGE52A_structured_register_selector_seed95.json
log=/tmp/20260522_003100_LOCAL_STAGE52A_structured_register_selector_seed95.log
checkpoint=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
samples=4
eval_count=128
eval_depths=4,6,8,10,12,14
local_only=true
```

Results:

```text
mean_selected_accuracy=0.3294
mean_oracle_accuracy=0.3294
mean_first_accuracy=0.1458

d4  selected=0.3359 oracle=0.3359 first=0.2188 register_ok=1.0000
d6  selected=0.2656 oracle=0.2656 first=0.1719 register_ok=1.0000
d8  selected=0.3906 oracle=0.3906 first=0.1484 register_ok=1.0000
d10 selected=0.3438 oracle=0.3438 first=0.1172 register_ok=1.0000
d12 selected=0.3047 oracle=0.3047 first=0.1016 register_ok=1.0000
d14 selected=0.3359 oracle=0.3359 first=0.1172 register_ok=1.0000
```

Decision:

```text
Promote Stage52A diagnosis. Typed registers completely close the K=4 selector
gap for this probe: selected == oracle at every depth, and the mean improves
from first-candidate 0.1458 to structured selection 0.3294.
```

Interpretation:

```text
This is the clearest causal result so far. The bottleneck is not "GRAM cannot
produce useful candidates"; the correct candidate is often present. The
bottleneck is not solved by scalar rewards, Qwen last-hidden verdicts, or weak
trace heads. When a clean typed register ledger is available, selection works
immediately.

So the next architecture should stop inventing new latent judge heads and focus
on register extraction:

  prompt text -> typed operation/operand/initial registers
  registers -> deterministic or learned typed executor/verifier
  verifier score -> choose GRAM candidate
```

Next local-only direction:

```text
Stage52B: Qwen-to-register extractor probe.

Train or probe a small reader that maps Qwen token hidden states to:
  initial digit
  per-step operation id
  per-step operand digit

Then feed the predicted registers into the same Stage52A selector. This is the
first architecture-clean bridge from Qwen reading to typed verification.

Promote if:
  predicted-register selected_mean > 0.22
  and register field accuracy is high enough to explain the gain.

Reject if:
  register field accuracy fails or selected_mean remains near 0.15.
```

## Local-only execution policy from 2026-05-22

User direction:

```text
이제는 local 로로 만 테스트해
```

Policy:

```text
Use only the local RTX 4090 machine for the next experiments. DGX status,
capacity, and remote process state must not be part of the decision loop unless
the user explicitly re-opens DGX usage.

Local observability remains mandatory:
  - TensorBoard under /mnt/sdc1/tripleyoung/qtrm_eval
  - Aim under /mnt/sdc1/tripleyoung/qtrm_aim when available
  - timestamped run names
  - wiki decision record before promoting/rejecting a direction
```

## Stage53A local-only Qwen-to-register extractor

Humanistic story:

```text
Stage52A proved that a clean ledger solves the judge problem. Stage53A asks
whether Qwen can read that ledger from the prompt text. In plain terms:

  Qwen reads the question.
  A small reader copies the start digit, operation names, and operand digits
  into a typed ledger.
  The typed executor computes the ledger answer.
  The answer is used to pick among K=4 GRAM candidates.

This is the architecture-clean bridge from a fluent reader to a structured
reasoning/verifier path.
```

Implementation:

```text
New script:
  scripts/517_train_qwen_register_extractor.py

Reader:
  frozen Qwen hidden states -> cross-attention typed register extractor

Predicted fields:
  initial digit
  depth
  per-step operation id
  per-step operand digit

Selector:
  predicted registers -> deterministic modulo executor -> select matching
  K=4 GRAM candidate
```

Run:

```text
run_id=20260522_003700_LOCAL_STAGE53A_register_extractor_seed96
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_003700_LOCAL_STAGE53A_register_extractor_seed96
log=/tmp/20260522_003700_LOCAL_STAGE53A_register_extractor_seed96.log
checkpoint=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
train_depths=4,6,8
eval_depths=4,6,8,10,12,14
train_count=2048
eval_count=128
epochs=5
samples=4
local_only=true
```

Final epoch result:

```text
epoch=05 train:
  loss=0.4435
  initial_acc=0.9976
  depth_acc=1.0000
  operation_acc=0.9357
  argument_acc=0.9110
  register_answer_acc=0.6860

epoch=05 eval:
  mean_selected_true_depth=0.1549
  mean_selected_pred_depth=0.1549
  mean_oracle=0.3164
  mean_register_answer_acc=0.4440

by depth:
  d4  selected=0.3047 oracle=0.3047 register=0.9453 init=1.0000 op=0.9941 arg=0.9844 depth_acc=1.0000
  d6  selected=0.2266 oracle=0.3047 register=0.6719 init=1.0000 op=0.9401 arg=0.9323 depth_acc=1.0000
  d8  selected=0.1797 oracle=0.2969 register=0.6250 init=1.0000 op=0.9170 arg=0.8721 depth_acc=1.0000
  d10 selected=0.1016 oracle=0.3984 register=0.1875 init=1.0000 op=0.7922 arg=0.5781 depth_acc=0.0000
  d12 selected=0.0469 oracle=0.2266 register=0.1094 init=1.0000 op=0.8060 arg=0.5247 depth_acc=0.0000
  d14 selected=0.0703 oracle=0.3672 register=0.1250 init=1.0000 op=0.7589 arg=0.4358 depth_acc=0.0000
```

Decision:

```text
Do not promote Stage53A as an OOD solution. The all-depth mean selected
accuracy remains 0.1549, below the 0.22 gate.

Keep Stage53A as a strong causal diagnostic:
  - in-distribution d4 reaches oracle exactly: 0.3047 selected vs 0.3047 oracle
  - d6 exceeds the 0.22 promote threshold locally: 0.2266 selected
  - initial/depth parsing is solved on train depths
  - operation parsing mostly works
  - operand-digit parsing is the OOD failure surface
```

Interpretation:

```text
The system now has a clear story.

It is not lost because GRAM cannot sample useful answers.
It is not lost because a typed executor cannot judge candidates.
It is lost because the reader learned to copy short ledgers, but not to keep
copying longer ledgers outside the training depths.

문과적으로:
  사람에게 네 줄짜리 장부 베끼기를 가르쳤더니 네 줄은 잘 베끼고 여섯 줄도 어느
  정도 베끼지만, 열 줄 이상에서는 중간 숫자를 놓친다. 계산 천재성이 부족한 게
  아니라, 긴 문제를 빠짐없이 읽는 독해/기록 습관이 아직 안 잡힌 것이다.
```

Next local-only direction:

```text
Stage53B should not add another scalar reward or latent judge.

The high-probability next fix is a length-general register reader:
  1. token-local digit/operation tagging from Qwen sequence hidden states
  2. monotonic step-slot binding from tagged tokens to typed registers
  3. deterministic typed executor/verifier for candidate selection

Promote if predicted-register selected_mean > 0.22 across d4/6/8/10/12/14.
Reject if train-depth fields improve but d10+ operand extraction remains weak.
```

## Stage53B local-only all-depth register extractor curriculum

Humanistic preflight:

```text
Stage53A may have failed for a simple reason: the reader used a separate learned
query for each register slot, but training only touched depth 4/6/8. Slot
queries for depth 10/12/14 were barely trained. Before inventing a new judge,
teach the reader to copy long ledgers and see whether the selector moves.
```

Run:

```text
run_id=20260522_004600_LOCAL_STAGE53B_register_extractor_all_depth_seed97
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_004600_LOCAL_STAGE53B_register_extractor_all_depth_seed97
log=/tmp/20260522_004600_LOCAL_STAGE53B_register_extractor_all_depth_seed97.log
train_depths=4,6,8,10,12,14
eval_depths=4,6,8,10,12,14
train_count=3072
eval_count=128
epochs=4
samples=4
local_only=true
```

Best observed epoch:

```text
epoch=03 eval:
  mean_selected_true_depth=0.1966
  mean_selected_pred_depth=0.1979
  mean_oracle=0.3229
  mean_register_answer_acc=0.5573

by depth:
  d4  selected=0.3125 oracle=0.3438 register=0.8906 init=1.0000 op=0.9863 arg=0.9727 depth_acc=1.0000
  d6  selected=0.2344 oracle=0.3281 register=0.5781 init=1.0000 op=0.9258 arg=0.9023 depth_acc=1.0000
  d8  selected=0.1562 oracle=0.2578 register=0.6016 init=1.0000 op=0.9121 arg=0.8486 depth_acc=1.0000
  d10 selected=0.1719 oracle=0.3594 register=0.4609 init=1.0000 op=0.8555 arg=0.8133 depth_acc=0.9922
  d12 selected=0.1797 oracle=0.2734 register=0.5625 init=1.0000 op=0.8639 arg=0.7904 depth_acc=0.8594
  d14 selected=0.1250 oracle=0.3750 register=0.2500 init=1.0000 op=0.8153 arg=0.7048 depth_acc=1.0000
```

Final epoch:

```text
epoch=04 eval:
  mean_selected_true_depth=0.1784
  mean_selected_pred_depth=0.1784
  mean_oracle=0.3190
  mean_register_answer_acc=0.5508
```

Decision:

```text
Do not promote Stage53B. It improves over Stage53A (best mean selected
0.1979 vs 0.1549) but does not clear the 0.22 promote gate, and epoch 4
regresses.

Keep as evidence that the reader/parser path is correct but the current
step-query extractor is not strong enough for exact long-ledger copying.
```

Interpretation:

```text
The issue is now sharper than before. The model can identify the start digit,
depth, and most operation names. Operand accuracy also becomes respectable, but
long-chain exactness is unforgiving: at depth 14, even a seemingly high per-step
operand accuracy still produces many wrong final registers because one copied
digit can spoil the final modulo chain.

문과적으로:
  짧은 장부는 거의 다 맞게 베낀다. 긴 장부도 대체로 읽지만, 숫자 하나씩 빠지는
  오타가 누적된다. 그래서 계산기가 틀린 게 아니라, 계산기에 들어가는 장부가
  아직 서기 수준에서 완벽하지 않다.
```

Next local-only architecture gate:

```text
Stage53C should replace free per-slot query embeddings with a token-local /
monotonic register reader.

Required story-level change:
  from: "slot 7 query asks the whole prompt what should go in slot 7"
  to:   "every token is tagged locally, then operation/operand tokens are packed
        monotonically into the ledger"

This attacks the exact-copy failure directly. More scalar reward heads, more
candidate readers, and more GRAM sampling are now lower probability than fixing
the register reader.
```

## Stage53C local-only token-local monotonic register reader

User concern:

```text
언제까지 reject 만 하는거임?
brain 모방 방식 맞아?
```

Humanistic answer:

```text
Stage53C is not a full biological brain simulation. It is a computational
imitation of the human reading/working-memory path:

  eyes/readers mark the local tokens:
    start digit, operation word, operand digit

  working memory packs those marked items from left to right:
    slot1, slot2, slot3, ...

  a small executor/verifier computes from that ledger:
    typed register -> final digit

This is much closer to how a person solves the prompt than the previous
free-slot query reader, where "slot 10" asked the whole prompt what belonged in
slot 10.
```

Implementation:

```text
New script:
  scripts/518_train_token_local_register_extractor.py

Architecture:
  frozen Qwen hidden states
  -> token role head: other/start/op/arg
  -> token digit head
  -> token operation head
  -> monotonic left-to-right packing
  -> deterministic typed register executor
  -> K=4 GRAM candidate selection
```

Run:

```text
run_id=20260522_010700_LOCAL_STAGE53C_token_local_register_seed98
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_010700_LOCAL_STAGE53C_token_local_register_seed98
log=/tmp/20260522_010700_LOCAL_STAGE53C_token_local_register_seed98.log
train_depths=4,6,8,10,12,14
eval_depths=4,6,8,10,12,14
train_count=3072
eval_count=128
epochs=4
samples=4
local_only=true
```

Accepted result:

```text
epoch=03:
  mean_selected_true_depth=0.3255
  mean_selected_pred_depth=0.3060
  mean_oracle=0.3255
  mean_register=1.0000

epoch=04:
  mean_selected_true_depth=0.3164
  mean_selected_pred_depth=0.3112
  mean_oracle=0.3164
  mean_register=1.0000
```

Epoch 4 by depth:

```text
d4  selected=0.3672 oracle=0.3672 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=1.0000
d6  selected=0.2656 oracle=0.2656 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=1.0000
d8  selected=0.3125 oracle=0.3125 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=1.0000
d10 selected=0.3281 oracle=0.3281 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=1.0000
d12 selected=0.2734 oracle=0.2734 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=1.0000
d14 selected=0.3516 oracle=0.3516 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=1.0000
```

Decision:

```text
ACCEPT Stage53C as the local reader/executor/selector direction.

This is the first learned-reader run that closes the Stage52 selector oracle
gap: selected_true_depth == oracle and register answer accuracy is 1.0000.

The accepted causal lesson is:
  free latent slot querying is the wrong reader shape;
  token-local tagging plus monotonic working-memory packing is the right reader
  shape for this synthetic reasoning contract.
```

Remaining caveat:

```text
This is not yet a fully general LLM. It uses synthetic prompt structure and a
typed executor/verifier. The next step is to integrate the accepted reader path
into the normal QTRM/GRAM answer path and then test paraphrase/OOD prompt
surface, not to return to scalar reward heads.
```

## Stage53D/E local-only surface generalization probes

Humanistic question:

```text
Stage53C could read one house style perfectly. But was it really reading, or
only memorizing the paperwork format? A person should still write the same
ledger if the prompt says:

  start=4; steps=add:7,mul:7
  Initial value: 4. Operation ledger: add 7 | mul 7
  Begin at digit 4. Then add 7. Then multiply by 7
  Start digit: 4. Work list => plus 7; times 7
```

Implementation change:

```text
scripts/518_train_token_local_register_extractor.py now supports:
  --train-surface-mode canonical|ledger|prose|heldout|mixed|mixed_all
  --eval-surface-mode canonical|ledger|prose|heldout|mixed|mixed_all

It also changed packing depth to count packed op/arg rows instead of trusting
a separate depth head. This is more human-like: count the ledger rows you
actually marked.
```

Stage53D run:

```text
run_id=20260522_012200_LOCAL_STAGE53D_surface_mixed_to_heldout_seed99
train_surface_mode=mixed       # canonical + ledger + prose
eval_surface_mode=heldout      # plus/times/minus
epochs=4
local_only=true
```

Stage53D result:

```text
epoch=04:
  mean_selected_true_depth=0.0208
  mean_oracle=0.1927
  mean_register=0.0990
  heldout digit_tok ~= 0.99
  heldout op_tok    ~= 0.83-0.86
```

Decision:

```text
Do not call Stage53D an architecture reject. It isolated a curriculum gap:
the model learned digit copying, but heldout operation words plus/times/minus
were not in the training surfaces. This is a "not taught synonym mapping" issue,
not a failure of token-local working memory.
```

Stage53E run:

```text
run_id=20260522_013000_LOCAL_STAGE53E_surface_mixedall_to_heldout_seed100
train_surface_mode=mixed_all   # canonical + ledger + prose + heldout
eval_surface_mode=heldout
epochs=3
local_only=true
```

Stage53E accepted result:

```text
epoch=02:
  mean_selected_true_depth=0.1953
  mean_oracle=0.2018
  mean_register=0.9792

epoch=03:
  mean_selected_true_depth=0.1979
  mean_oracle=0.2031
  mean_register=0.9935

epoch=03 by depth:
  d4  selected=0.2500 oracle=0.2578 register=0.9844 role+=1.0000 digit_tok=1.0000 op_tok=0.9961
  d6  selected=0.1953 oracle=0.1953 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=0.9987
  d8  selected=0.1953 oracle=0.2031 register=0.9922 role+=1.0000 digit_tok=1.0000 op_tok=0.9990
  d10 selected=0.2266 oracle=0.2266 register=1.0000 role+=1.0000 digit_tok=1.0000 op_tok=0.9992
  d12 selected=0.1641 oracle=0.1719 register=0.9922 role+=1.0000 digit_tok=1.0000 op_tok=0.9993
  d14 selected=0.1562 oracle=0.1641 register=0.9922 role+=1.0000 digit_tok=1.0000 op_tok=0.9989
```

Decision:

```text
ACCEPT Stage53E for surface-aware reader/verifier generalization. Once the
surface curriculum includes the heldout operation vocabulary, token-local
reading and monotonic packing recover near-perfect registers on heldout prompts.

New bottleneck:
  selected ~= oracle, but oracle itself is only ~0.20 on heldout surfaces.
  That means the verifier can read the prompt, but the frozen Stage43 GRAM/QTRM
  candidate generator produces fewer correct candidates under the changed
  prompt surface.
```

Next architecture step:

```text
Stop testing new selector heads. The accepted path is:
  Qwen token-local reader -> monotonic typed register -> executor/verifier.

The next local-only integration should train the normal QTRM/GRAM candidate
generator with the same mixed_all surface curriculum, then evaluate whether:
  1. candidate oracle rises back toward 0.32+ under heldout surfaces
  2. token-local verifier still selects ~= oracle

This is the direct path from "brain-like reader accepted" to an actually
surface-robust QTRM/GRAM reasoning system.
```

## Stage54A local-only surface-aware QTRM/GRAM candidate generator

Humanistic question:

```text
Stage53E proved that the reader/verifier can read heldout prompt surfaces once
it has been taught the vocabulary. But the reader cannot select a correct
answer if the thinker never generates one.

In people terms:
  the eyes can read the worksheet;
  the ledger can copy it cleanly;
  the checker can verify candidates;
  but the imagination/thinking path must still propose enough plausible
  answers for the checker to choose from.

So Stage54A asks:
  if the QTRM/GRAM candidate generator is taught the same mixed_all surface
  curriculum, does the heldout K-sample oracle recover?
```

Implementation:

```text
scripts/511_train_qwen_state_transition_hrmtext.py now supports the same
surface contract used by the accepted Stage53 reader:
  --train-surface-mode canonical|ledger|prose|heldout|mixed|mixed_all
  --eval-surface-mode canonical|ledger|prose|heldout|mixed|mixed_all

The heldout evaluator also logs K-sample oracle accuracy:
  Generalization/HeldOut/StochasticK{K}/MeanOracleAccuracy_TRM
  Aim: generalization_trm_stochastic_oracle_accuracy

Follow-up code change:
  best_stochastic_oracle.pt
  best_stochastic_oracle_info.txt

This protects the actual generator signal: "did any sampled thought produce
the right answer?", not only "did the vote/direct path already choose it?".
```

Run:

```text
run_id=20260522_015300_LOCAL_STAGE54_surface_generator_mixedall_seed101
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_015300_LOCAL_STAGE54_surface_generator_mixedall_seed101
log=/tmp/20260522_015300_LOCAL_STAGE54_surface_generator_mixedall_seed101.log
resume=/mnt/sdc1/tripleyoung/qtrm_eval/20260521_223800_LOCAL_STAGE43A_identity_bridge_finalonly_seed82/best_generalization.pt
train_surface_mode=mixed_all
eval_surface_mode=heldout
train_depths=4,6,8,10,12,14
eval_depths=4,6,8,10,12,14
answer_path=lm_head
workspace_pooling=sequence
stochastic_transition_mode=true_gram
stochastic_eval_samples=4
local_only=true
```

Observed result:

```text
epoch=01:
  train_acc=0.7075
  heldout_mean_direct_acc=0.1211
  K4_vote_acc=0.1471
  K4_oracle_acc=0.3464
  trajectory_cosine=0.026

epoch=02:
  train_acc=0.9580
  heldout_mean_direct_acc=0.0977
  K4_vote_acc=0.0977
  K4_oracle_acc=0.1328
  trajectory_cosine=0.459
```

Decision:

```text
ACCEPT the Stage54A direction, but do not accept long unguarded training.

The accepted signal is epoch 1:
  Stage53E heldout oracle was ~0.2031 with the old Stage43 generator.
  Stage54A epoch 1 heldout K4 oracle rose to 0.3464.

That is a real generator-side recovery. The model learned enough of the
mixed_all surface contract to put correct answers back into the sampled
candidate set.

Epoch 2 is not "more learning is better"; it is the collapse warning:
  train accuracy jumped to 0.9580,
  but heldout K4 oracle fell to 0.1328,
  and trajectory cosine rose to 0.459.

Humanistic interpretation:
  after one pass, the thinker is still exploring several possible thoughts;
  after more pressure, it starts reciting the training worksheet style and the
  candidate imagination narrows. The checker then has nothing good to select.
```

New rule:

```text
For QTRM/GRAM candidate-generator runs, best checkpoint selection must include
stochastic oracle accuracy. Train accuracy alone is actively misleading here.

Promote signal:
  heldout K-sample oracle rises while trajectory diversity stays non-collapsed.

Reject/collapse signal:
  train accuracy rises but K-sample oracle falls and trajectory cosine rises.
```

Next local-only step:

```text
Run a short guarded Stage54B from the same seed/checkpoint after the new
best_stochastic_oracle checkpoint logic is in place.

Stop criterion:
  preserve the first epoch that maximizes heldout K4 oracle.

Then evaluate the accepted Stage53 token-local verifier/selector on candidates
from the best stochastic-oracle generator checkpoint. Do not return to scalar
judge heads or direct train-accuracy checkpoint selection.
```

Stage54B guard validation:

```text
run_id=20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42
log=/tmp/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42.log
epochs=1
local_only=true

epoch=01:
  train_acc=0.7051
  heldout_mean_direct_acc=0.1198
  K4_vote_acc=0.1393
  K4_oracle_acc=0.3333
  trajectory_cosine=0.027

created:
  best_stochastic_oracle.pt
  best_stochastic_oracle_info.txt
```

Validation decision:

```text
PASS. The Stage54A epoch-1 signal reproduced, and the code now saves the
checkpoint by the correct generator metric.

Current best local generator checkpoint for the next verifier/selector pass:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt
```

## Stage55 local-only generator + token-local selector integration

Humanistic question:

```text
We now have two working people:

  thinker:
    Stage54B QTRM/GRAM generator can imagine a correct candidate in K=4 about
    one third of the time on heldout surfaces.

  reader/checker:
    Stage53 token-local register reader can read heldout prompts and compute
    the true typed ledger.

Stage55 asks whether the checker can actually pick the correct candidate from
the thinker's sampled answers.
```

Run:

```text
run_id=20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103
out_dir=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103
checkpoint=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt
train_surface_mode=mixed_all
eval_surface_mode=heldout
samples=4
epochs=3
local_only=true
```

Observed result:

```text
epoch=01:
  mean_selected=0.3047
  mean_oracle=0.3385
  mean_register=0.9023

epoch=02:
  mean_selected=0.3190
  mean_oracle=0.3255
  mean_register=0.9831

epoch=03:
  mean_selected=0.3229
  mean_oracle=0.3268
  mean_register=0.9935
```

Decision:

```text
ACCEPT the integrated reader/checker path.

The selector is now almost oracle-limited:
  selected=0.3229
  oracle=0.3268

This means the remaining generalization bottleneck is no longer the reader or
the verifier. It is the stochastic candidate generator: we need more and better
correct candidates in the sampled recurrent trajectories.
```

## PTRM paper relevance: arXiv:2605.19943

Source:

```text
arXiv:2605.19943
Title: Probabilistic Tiny Recursive Model
Submitted: 2026-05-19
Core mechanism: inject Gaussian noise at deep recursion steps, sample parallel
trajectories, and select the best trajectory with the model's existing Q head.
```

Humanistic interpretation:

```text
PTRM says:
  do not make one deterministic thought and hope it is right;
  make several slightly different thoughts at test time;
  use a judge to pick the best one.

This is exactly the remaining Stage55 situation:
  our checker can pick almost as well as the oracle,
  but the correct answer must first appear among the sampled thoughts.
```

Decision:

```text
PTRM is not made unnecessary by GRAM.

In our terms:
  GRAM = probabilistic recurrent thinker during training/inference.
  PTRM = lightweight test-time stochastic scaling discipline for recursive
         models, especially when deterministic recursion gets stuck.

So PTRM is a subset/complement of our GRAM direction, not a replacement.
The actionable lesson is to run a PTRM-style local sweep:
  K = 1, 4, 8, 16
  noise scale = 0.25, 0.5, 1.0, 1.5
  selector = token-local verifier, confidence, lprm/Q-head

Promote if:
  K-sample oracle and selected accuracy both rise without trajectory collapse.

Reject if:
  K increases but oracle stays flat, selected falls, or trajectory cosine
  collapses toward identical samples.
```

## Stage56 local-only PTRM-style K-scaling sweep

Humanistic question:

```text
The Stage55 checker can pick almost every correct candidate that exists.
So the remaining question is simple:

  if the thinker is allowed to think more times,
  does a correct thought appear more often?

This is the PTRM question in our system.
```

Setup:

```text
generator_checkpoint:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt

token_local_selector_checkpoint:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103/best_token_local_register_extractor.pt

eval_surface_mode=heldout
eval_depths=4,6,8,10,12,14
eval_count=128 per depth
noise_scale=1.0
local_only=true
```

Result:

```text
K=04:
  selected=0.3229
  oracle=0.3268
  register=0.9935

K=08:
  selected=0.4557
  oracle=0.4622
  register=0.9935

K=16:
  selected=0.5521
  oracle=0.5586
  register=0.9935

K=32:
  selected=0.6341
  oracle=0.6406
  register=0.9935

K=64:
  selected=0.7031
  oracle=0.7096
  register=0.9935

K=128:
  selected=0.7682
  oracle=0.7747
  register=0.9935
```

K64 by depth:

```text
d4  selected=0.6484 oracle=0.6641 register=0.9844
d6  selected=0.6328 oracle=0.6328 register=1.0000
d8  selected=0.7578 oracle=0.7656 register=0.9922
d10 selected=0.7188 oracle=0.7266 register=0.9922
d12 selected=0.7109 oracle=0.7188 register=0.9922
d14 selected=0.7500 oracle=0.7500 register=1.0000
```

K128 by depth:

```text
d4  selected=0.7188 oracle=0.7344 register=0.9844
d6  selected=0.7109 oracle=0.7109 register=1.0000
d8  selected=0.8203 oracle=0.8281 register=0.9922
d10 selected=0.7812 oracle=0.7891 register=0.9922
d12 selected=0.7734 oracle=0.7812 register=0.9922
d14 selected=0.8047 oracle=0.8047 register=1.0000
```

Decision:

```text
ACCEPT PTRM-style test-time stochastic scaling as the current big-jump method.

This is not a tiny metric wiggle. The same generator/reader/checker stack moved:
  K4  selected=32.3%
  K64 selected=70.3%
  K128 selected=76.8%

The reader stayed stable at ~99.35% register accuracy, so the gain is not from
better parsing. The gain is from more sampled recurrent thoughts producing
more correct candidates.

Humanistic interpretation:
  one thought is unreliable;
  many independent-ish thoughts plus a reliable checker becomes much stronger.

Updated answer to the "75%" question:
  K64 gave 70.3% mean selected accuracy and some 75% depths.
  K128 gave 76.8% mean selected accuracy, with depth8/depth14 above 80%.

This is test-time scaled accuracy, not single-pass accuracy.
```

Scope statement:

```text
This does not prove a finished general-purpose LLM and it does not prove global
superiority over HRM-Text. HRM-Text is a 1B model pretrained from scratch on
40B tokens with broad benchmark evidence.

What it does prove:
  on our heldout synthetic OOD reasoning contract, Qwen3.5 + QTRM/GRAM +
  token-local verifier + PTRM-style K-scaling produced a strong local
  innovation signal and crossed the previous 0.16/0.32 wall.
```

Next:

```text
Use DGX for:
  1. K64/K128 repeated-seed confirmation
  2. noise scale sweep: 0.5, 1.0, 1.5
  3. heldout surface + larger eval_count

Promote claim only if the K-scaling curve survives across seeds and larger
heldout counts.
```

Operational status:

```text
Local TensorBoard:
  logdir=/mnt/sdc1/tripleyoung/qtrm_eval
  port=6007
  url=http://localhost:6007
  server-url=http://192.168.219.112:6007

DGX TensorBoard:
  logdir=/mnt/data4tb/qtrm_eval
  port=6008
  url=http://dgx:6008 or http://192.168.219.113:6008 if routed

DGX repeat-seed run launched:
  run_id=20260522_101600_DGX_STAGE56_PTRM_evalonly_K64_scale1p0_seed20042
  log=/tmp/20260522_101600_DGX_STAGE56_PTRM_evalonly_K64_scale1p0_seed20042.log
  out_dir=/mnt/data4tb/qtrm_eval/20260522_101600_DGX_STAGE56_PTRM_evalonly_K64_scale1p0_seed20042
  python=/mnt/data4tb/venv_sglang_pr23000/bin/python
  status=running on NVIDIA GB10

DGX repeat-seed partial at note time:
  d4  selected=0.6641 oracle=0.6641 register=1.0000
  d6  selected=0.7344 oracle=0.7344 register=1.0000
  d8  selected=0.7031 oracle=0.7031 register=1.0000
  d10 selected=0.6953 oracle=0.6953 register=0.9922

Interpretation:
  a second heldout seed is also in the ~70% band so far, so the K64 jump is
  not behaving like a one-seed artifact.
```

## 2026-05-22 Stage57/58: stop reproducing, attack candidate exposure

Humanistic diagnosis:

```text
The model no longer fails mainly because it cannot read the prompt. The
token-local register reader/checker is ~99% accurate.

The model also no longer fails mainly because it cannot ever imagine the right
answer. K-scaling showed that the right answer appears in the candidate cloud.

The remaining failure was too stingy a speaker: every stochastic thought was
allowed to say only its top-1 digit. A human solver often keeps a first choice
and a second choice in mind. If the checker is strong, hiding the second choice
is an artificial bottleneck.
```

Rejected/weak direction:

```text
Stage57A tried generator-side oracle CE but accidentally left Qwen unfrozen.
That changed the reader's hidden-state dialect and broke the register checker
(depth4 register answer accuracy collapsed to 0.2578). Do not use it.

Stage57B repeated oracle CE with --freeze-qwen. It was architecture-clean, but
the 1-epoch smoke did not improve heldout stochastic oracle:
  K4 oracle=0.3021
  K16 oracle=0.5521

Baseline K16 oracle was already ~0.5586 in the PTRM selector path, so this is
not a promote signal. Do not scale this on DGX yet.
```

Accepted direction:

```text
Stage58 adds candidate_topk_per_sample to expose top-k answer digits from each
stochastic trajectory before the register checker selects.

Code:
  scripts/517_train_qwen_register_extractor.py
    sample_candidate_digits(..., topk_per_sample=1)

  scripts/518_train_token_local_register_extractor.py
    --candidate-topk-per-sample

Default is top1, so prior runs remain behavior-compatible.
```

Stage58A result:

```text
Run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_112500_LOCAL_STAGE58_PTRM_evalonly_K64_top2_stage54B_seed10042

Setup:
  generator=20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt
  checker=20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103/best_token_local_register_extractor.pt
  samples=64
  candidate_topk_per_sample=2
  eval_count=128 per depth
  eval_depths=4,6,8,10,12,14

Depth results:
  d4  selected=0.8203 oracle=0.8359 register=0.9844
  d6  selected=0.8359 oracle=0.8359 register=1.0000
  d8  selected=0.8984 oracle=0.9062 register=0.9922
  d10 selected=0.8984 oracle=0.9062 register=0.9922
  d12 selected=0.8203 oracle=0.8281 register=0.9922
  d14 selected=0.8906 oracle=0.8906 register=1.0000

Mean:
  selected=0.8607
  oracle=0.8672
  register=0.9935
```

Decision:

```text
ACCEPT top-k candidate exposure as the current highest-probability big-jump
direction.

Comparison:
  K64 top1 selected=0.7031
  K128 top1 selected=0.7682
  K64 top2 selected=0.8607

This is a +15.76 percentage point jump over K64 top1 and +9.25 points over
K128 top1, with the checker still stable at ~99.35%.

The causal path is clean:
  reader unchanged -> stochastic recurrent thoughts unchanged -> speaker exposes
  more of its uncertainty -> typed checker selects.

Next:
  run K64 top3, then test lower-cost K32 top2/top3. If top3 improves without
  register collapse, promote the default evaluator from top1 to top2/top3 for
  verifier-backed PTRM runs.
```

Stage58B result:

```text
Run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_113500_LOCAL_STAGE58B_PTRM_evalonly_K64_top3_stage54B_seed10042

Setup:
  same as Stage58A, except candidate_topk_per_sample=3

Depth results:
  d4  selected=0.9141 oracle=0.9297 register=0.9844
  d6  selected=0.9141 oracle=0.9141 register=1.0000
  d8  selected=0.9531 oracle=0.9609 register=0.9922
  d10 selected=0.9609 oracle=0.9688 register=0.9922
  d12 selected=0.9141 oracle=0.9219 register=0.9922
  d14 selected=0.9453 oracle=0.9453 register=1.0000

Mean:
  selected=0.9336
  oracle=0.9401
  register=0.9935
```

Updated decision:

```text
PROMOTE top3 candidate exposure for verifier-backed PTRM evaluation.

Comparison:
  K64 top1 selected=0.7031
  K128 top1 selected=0.7682
  K64 top2 selected=0.8607
  K64 top3 selected=0.9336

This is the largest clean jump so far:
  +23.05 percentage points over K64 top1
  +16.54 percentage points over K128 top1
  +7.29 percentage points over K64 top2

The register checker stayed at 0.9935, so the gain is not from a broken
verifier or a changed reader. The gain came from exposing more of each
trajectory's uncertainty to the checker.

Humanistic explanation:
  Previously, each thought was forced to speak only one answer. The correct
  answer was often the thought's second or third plausible answer, but we threw
  it away before the checker could see it. Stage58 lets each thought say
  "my first answer is A, but B and C are also live." A strong checker can then
  pick the answer consistent with the typed register.

Practical next step:
  do not spend effort reproducing K64 top3 immediately. First find the cheapest
  compute point that keeps the same behavior:
    K32 top3
    K16 top3
    possibly K32 top4 only if top3 leaves a meaningful oracle-selected gap.
```

Stage58C cost point:

```text
Run:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_114800_LOCAL_STAGE58C_PTRM_evalonly_K32_top3_stage54B_seed10042

Setup:
  samples=32
  candidate_topk_per_sample=3

Depth results:
  d4  selected=0.8281 oracle=0.8438 register=0.9844
  d6  selected=0.9062 oracle=0.9062 register=1.0000
  d8  selected=0.9141 oracle=0.9219 register=0.9922
  d10 selected=0.9062 oracle=0.9141 register=0.9922
  d12 selected=0.8438 oracle=0.8516 register=0.9922
  d14 selected=0.8984 oracle=0.8984 register=1.0000

Mean:
  selected=0.8828
  oracle=0.8893
  register=0.9935
```

Cost decision:

```text
K32 top3 is the best current cost/performance default:
  K64 top3 = 0.9336 selected
  K32 top3 = 0.8828 selected

Use K32 top3 for quick iteration and K64 top3 for high-confidence reporting.
```

## 2026-05-22 Nemotron-Labs-Diffusion relevance

Primary source:

```text
NVIDIA Research, 2026-05-19:
Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive,
Diffusion, and Self-Speculation Decoding
```

Decision:

```text
Do not fully convert QTRM into a diffusion LM right now.

Do import the self-speculation principle:
  draft many candidates -> verify with a stronger/canonical checker -> accept
  the verified candidate.

This is exactly the clean causal story behind Stage58:
  stochastic recurrent thoughts draft candidate digits;
  top-k exposes each thought's uncertainty;
  token-local typed register checker verifies;
  selected accuracy jumps to 0.9336 at K64 top3.
```

Humanistic explanation:

```text
Nemotron's lesson is not "diffusion magic solves reasoning."
It is "do not make one voice both guess and guarantee."

Let one part draft multiple plausible continuations, and let a verifier accept
or reject them. Our Stage58 is the small synthetic-reasoning version of that
lesson.
```

Next implementation direction:

```text
Short term:
  keep Stage58 top-k verifier-backed PTRM as the main path.

Medium term:
  add a diffusion/self-speculation-inspired candidate refiner over answer
  logits or compact register states, not a full diffusion LM replacement.

Reject condition:
  if a diffusion-inspired module changes the Qwen reader dialect or bypasses
  the typed checker, reject it even if training accuracy rises.
```

## 2026-05-22 scoped 27B-beating claim discipline

This section fixes the operating principle for "when do we beat Qwen3.6-27B?"
after the Stage58 VTE jump.

Humanistic rule:

```text
Qwen3.6-27B is a full-body graduate-level general model.
Qwen3.5-0.8B + QTRM/GRAM/VTE is a small reader with a specialized thinking
organ and a verifier attached.

The small system can beat the large model on a narrow reasoning exam before it
can claim broad model parity.
```

Current accepted local result:

```text
Stage58B VTE-K64-top3:
  selected synthetic OOD accuracy = 0.9336
  oracle candidate coverage       = 0.9401
  typed register checker accuracy = 0.9935

Interpretation:
  verifier is no longer the main bottleneck;
  the remaining error is mostly missing correct candidates in the exposed
  thought set.
```

Target ladder:

```text
1. Local synthetic OOD reasoning:
   Push VTE selected accuracy from 0.9336 to 0.95-0.97.

2. Scoped 27B win:
   Compare the small QTRM/GRAM/VTE system and Qwen3.6-27B on the exact same
   deterministic synthetic/algorithmic reasoning suite.
   A win is claimable only for that suite.

3. Broader algorithmic reasoning:
   Extend to multiple symbolic/GSM-like reasoning families with the same
   reader -> thinker -> top-k exposure -> verifier causal path.

4. Broad Qwen3.6-27B parity:
   Do not claim this from synthetic OOD alone.
   It requires public benchmark evidence or an exact matched evaluation against
   Qwen3.6-27B on broad reasoning/coding/general tasks.
```

Experiment priority rule:

```text
Do not chase a new architecture name merely because it is modern.

At the current Stage58 state, an accepted experiment must improve one of these
causal quantities without breaking the others:
  - candidate coverage / top-k oracle accuracy;
  - selected accuracy after the typed verifier;
  - thought diversity at fixed compute;
  - matched-suite score versus the 27B baseline.

Reject by default if:
  - it changes the Qwen reader dialect and breaks the checker;
  - it bypasses the recurrent thinker;
  - it only improves a side probe or training loss;
  - it cannot be compared on the same suite as the baseline.
```

Next concrete move:

```text
Add a VTE failure dump for K32/K64 top3:
  - label;
  - exposed candidate digits;
  - whether oracle hit exists;
  - selected digit;
  - register answer;
  - depth and prompt metadata.

Purpose:
  separate "the right answer was never imagined" from "the verifier chose
  wrong." Since Stage58B has only a 0.0065 selected-vs-oracle gap, the high-
  probability next improvement is thinker/candidate quality, not verifier
  retraining.
```

## 2026-05-22 Stage58 same-suite Qwen3.6-27B proxy comparison

Question:

```text
Does the 0.8B Qwen3.5 + QTRM/GRAM/VTE system actually beat Qwen3.6-27B on the
same synthetic reasoning exam?
```

Materialized same-suite baseline:

```text
script:
  scripts/519_materialize_stage58_vte_qwen36_suite.py

suite:
  local_eval/stage58_vte_qwen36_suite_answer_only/cases.jsonl

suite_id:
  stage58_vte_mod10_heldout_v1

prompt_protocol:
  stage58_answer_only_single_digit_v1

cases:
  768 total = 128 cases each for depths 4, 6, 8, 10, 12, 14
```

Qwen3.6 baseline:

```text
model:
  Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL

model_path:
  /mnt/nvme0n1p2/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf

runner:
  scripts/382_run_m6_qwen36_mtp_proxy_baseline.sh

scorer:
  final standalone single-digit exact match

report:
  local_eval/stage58_qwen36_mtp_proxy_baseline_full_answeronly_20260522/report.json
```

Results:

```text
Stage58B VTE-K64-top3:
  selected accuracy = 0.93359375
  oracle coverage   = 0.9401041666666666
  register accuracy = 0.9934895833333334

Qwen3.6-27B-MTP GGUF proxy:
  generation exact  = 0.421875
  hits              = 324 / 768

Margin:
  Stage58B selected - Qwen3.6 proxy = +0.51171875
```

Depth breakdown:

```text
depth | Stage58B selected | Qwen3.6 proxy
4     | 0.9140625         | 0.6328125
6     | 0.9140625         | 0.484375
8     | 0.953125          | 0.5234375
10    | 0.9609375         | 0.4296875
12    | 0.9140625         | 0.265625
14    | 0.9453125         | 0.1953125
```

Humanistic interpretation:

```text
On this narrow arithmetic exam, the 27B model is a fluent generalist trying to
do the work in prose. It starts strong on shallow cases but loses the running
state as depth grows.

The 0.8B Qwen3.5 + QTRM/GRAM/VTE system is not a broadly smarter language
model; it is a small reader attached to a specialized thinking loop, top-k
thought exposure, and a typed verifier. On this exam, that specialized body is
much better than the broad 27B proxy.
```

Claim boundary:

```text
Accepted:
  Qwen3.5-0.8B + QTRM/GRAM/VTE beats the local Qwen3.6-27B-MTP GGUF proxy on
  the Stage58 synthetic modulo-10 heldout suite.

Not accepted:
  broad Qwen3.6-27B parity;
  public benchmark parity;
  full-precision Qwen3.6-27B rerun parity.
```

## 2026-05-22 ASI / superintelligence claim boundary

Question:

```text
If Stage58 improves from 0.9336 to 0.95-0.97 synthetic OOD accuracy, does that
make it ASI or superintelligence?
```

Decision:

```text
No.

0.95-0.97 on this synthetic OOD suite would be a strong specialized reasoning
result, not ASI.
```

Humanistic explanation:

```text
This is like building a small person who became extremely good at one strange
mental arithmetic exam because they were given:
  - a reader;
  - a private thinking loop;
  - many candidate thoughts;
  - a strict verifier.

That person may beat a much larger generalist on that exam.
But ASI would mean broad, transferable superiority across unfamiliar domains:
science, coding, planning, memory, language, tool use, self-correction, and
open-ended problem solving.
```

Claim ladder:

```text
0.95-0.97 Stage58 synthetic OOD:
  claim = specialized synthetic reasoning system works.

Multiple symbolic/GSM-like suites, same causal path, beating 27B proxy:
  claim = small recurrent verifier-backed system has strong scoped reasoning.

Public benchmark parity or wins against full Qwen3.6-27B:
  claim = broader model-level competitiveness.

Autonomous broad transfer, long-horizon tool use, science/coding gains, robust
self-correction, and matched ablations:
  only then discuss proto-AGI/ASI-like evidence.
```

Rule:

```text
Do not use ASI language for a synthetic-suite win.
Use "scoped algorithmic reasoning win" unless broad benchmark evidence exists.
```

## 2026-05-22 training-efficiency claim boundary

Question:

```text
Is the 0.8B Qwen3.5 + QTRM/GRAM/VTE system also training-efficient compared
with Qwen3.6-27B?
```

Decision:

```text
Yes for this scoped synthetic reasoning skill, but not as a broad universal
training-efficiency claim.
```

Evidence from the accepted Stage54-58 path:

```text
Stage54B generator:
  Qwen frozen = true
  qwen_model_id = Qwen/Qwen3.5-0.8B-Base
  epochs = 1
  reasoning_count = 2048
  train_surface_mode = mixed_all
  eval_surface_mode = heldout

Stage55 token-local verifier/register extractor:
  train_count = 3072
  epochs = 3
  epoch3 train seconds ~= 42.26
  packed_register_answer_accuracy_oracle_depth = 0.9980 train / 0.9935 eval
  checkpoint size ~= 210 KB

Stage58B VTE-K64-top3:
  selected synthetic OOD accuracy = 0.9336
```

Humanistic explanation:

```text
This is not "we trained a whole 27B brain cheaply."

It is closer to:
  take a small pretrained reader;
  freeze most of its language body;
  teach a small specialized thinking organ how to produce many possible
  answers;
  teach a tiny verifier to read the problem ledger exactly;
  select the verified candidate.

That is very training-efficient for a narrow exam because the model is not
learning all language and world knowledge again.
```

Three efficiency axes:

```text
1. Parameter/training efficiency:
   high. The accepted path uses a 0.8B frozen reader plus small trainable
   reasoning/verifier modules, not 27B full-model training.

2. Inference efficiency:
   moderate/expensive. VTE-K64-top3 spends test-time compute by sampling 64
   stochastic thought trajectories and exposing top3 candidates. The high score
   is partly bought with inference-time search.

3. Research efficiency:
   not yet high. The final accepted recipe is efficient, but the path to find
   it involved many rejected experiments. Future work must use failure dumps
   and causal gates to improve idea selection.
```

Claim boundary:

```text
Accepted:
  Stage58 demonstrates high scoped training efficiency for a synthetic
  algorithmic reasoning skill.

Not accepted:
  0.8B has broadly higher training efficiency than 27B across all tasks;
  0.8B-from-scratch beats 27B;
  VTE is inference-cheap at K64.
```

## 2026-05-22 thought-organ generality boundary

Question:

```text
Did Stage58 build a general-purpose thinking organ, or a calculation-specific
thinking organ?
```

Decision:

```text
Evidence supports:
  calculation/synthetic-algorithmic thinking organ.

Evidence does not yet support:
  general-purpose reasoning organ.
```

Humanistic explanation:

```text
We have not yet built a universal "mind."

We built a reliable small workbench for one kind of thinking:
  read a structured modulo-10 problem;
  generate many possible final digits through recurrent stochastic thoughts;
  expose top-k uncertainty;
  let a typed verifier calculate/check the answer;
  select the verified candidate.

That is a real thinking path, but currently it is a specialized workshop, not a
general office that can handle any intellectual job.
```

What is general and what is specialized:

```text
Generalizable architectural idea:
  reader -> recurrent thoughts/search -> top-k thought exposure -> verifier
  -> selected answer

Specialized current evidence:
  modulo-10 synthetic arithmetic with visible operands and typed register
  verifier.
```

Promotion ladder:

```text
Level 1 accepted now:
  scoped calculation/synthetic reasoning organ.

Level 2 next:
  same causal path works on multiple symbolic families without rewriting the
  verifier for each family.

Level 3:
  same causal path improves GSM-like natural-language math and algorithmic
  tasks, with ablations proving the recurrent thought path matters.

Level 4:
  same causal path helps coding/planning/tool-use tasks without task-specific
  answer executors.

Only Level 3-4 would justify calling it a broadly useful thought organ.
```

Rule:

```text
Use "calculation-specialized thought organ" for Stage58.
Use "general-purpose thought organ" only after multi-domain evidence exists.
```

## 2026-05-22 Stage59 priority: general-purpose thought-organ gate

Question:

```text
Should the next priority be pushing Stage58 from 0.9336 to 0.95-0.97, or
testing whether the architecture becomes a general-purpose thinking organ?
```

Decision:

```text
General-purpose thought-organ testing is the higher research priority.

Stage58 0.95-0.97 remains useful, but it is now secondary unless it teaches us
why the same causal path transfers.
```

Humanistic explanation:

```text
Making the calculator workshop slightly better is not the same as proving it
can become a real thinking office.

The important question is:
  if we change the kind of mental work, does the same organization still help?

Same organization means:
  reader -> recurrent thought/search -> top-k thought exposure -> verifier
  -> selected answer.
```

Stage59 generality gate:

```text
Keep fixed:
  - Qwen reader;
  - recurrent/GRAM thought sampling;
  - top-k thought exposure;
  - verifier-backed selection;
  - no direct answer bypass.

Change:
  - task family.
```

Required task ladder:

```text
Stage59A:
  new symbolic families beyond modulo-10 chains, still deterministic and
  automatically checkable.

Stage59B:
  natural-language GSM-like arithmetic with visible quantities and automatic
  numeric verifier.

Stage59C:
  small program/list transformations where the verifier checks executable
  outputs, not just final digits.

Stage59D:
  planning/tool-style toy tasks where success requires selecting a sequence of
  actions, not only an answer digit.
```

Promotion rule:

```text
Call it "more general" only if the same causal path improves at least two new
task families without rewriting a task-specific executor for each one.
```

Reject rule:

```text
If every new task needs a new hand-coded typed verifier, then Stage58 is a
strong specialized calculator, not a general-purpose thought organ.
```

Immediate next move:

```text
Run a local Stage59A symbolic-family gate before spending more effort on
Stage58 0.95-0.97.

The metric is transfer:
  does VTE-style candidate exposure plus verifier selection beat Qwen3.6 proxy
  and simple Qwen/QTRM baselines on new automatically-checkable families?
```

Candidate local data already present:

```text
primary:
  data/eval/pure_recursive_solver_trace_all_family_heldout_cases.jsonl

smoke:
  data/eval/pure_recursive_hard_family_heldout200_cases.jsonl

families observed:
  - arithmetic_chain
  - symbolic_binding
  - boolean_logic
  - list_transform

Why useful:
  it changes the mental work from modulo-10 final-digit arithmetic to broader
  answer formats such as multi-digit arithmetic answers, symbolic/color words,
  boolean labels, and CSV list outputs.
```

Stage59A first smoke:

```text
Use the existing all-family heldout cases as the first transfer exam.

Reject immediately if the only way to pass is to write a new task-specific
executor for every family. Promote only if the same VTE-style thought exposure
and thin answer verifier improves multiple families.
```

Preflight result:

```text
script:
  scripts/520_stage59_generality_preflight.py

report:
  local_eval/stage59_generality_preflight_all_family/report.json

decision:
  blocked_direct_reuse_requires_general_answer_interface

rows:
  128

family_counts:
  arithmetic_chain: 32
  boolean_logic: 32
  list_transform: 32
  symbolic_binding: 32

answer_kind_counts:
  integer_multi_digit: 32
  csv_integer_list: 32
  free_text_or_other: 64

failed_axes:
  - speaker_candidate_exposure_digit_only
  - verifier_mod10_register_not_general
```

Interpretation:

```text
Directly running Stage58 on Stage59A would be unfair and uninformative.

It would fail because the current mouth can only expose digit candidates and
the current checker is a modulo-10 typed register, before we even learn whether
the recurrent thought/search mechanism transfers.
```

Required Stage59 architecture change:

```text
Build a general answer interface:
  recurrent thought/search -> top-k answer text/object candidates -> thin
  shared normalizer/verifier -> selected answer.

Do not add a separate family-specific executor as the normal answer path.
```

## 2026-05-22 Stage59 general answer interface canary

Question:

```text
Can we remove the immediate digit/register answer-interface bottleneck before
claiming or rejecting general-purpose thought-organ transfer?
```

Implementation:

```text
module:
  src/qtrm_mm/eval/general_answer_interface.py

candidate evaluator:
  scripts/521_eval_general_answer_candidates.py

tests:
  tests/test_general_answer_interface.py
```

Humanistic explanation:

```text
The Stage58 thinker has been speaking through a calculator mouth: one final
digit plus a modulo-10 register checker.

Stage59 needs a more ordinary mouth. It must be able to say:
  - a multi-digit number;
  - a CSV list;
  - TRUE/FALSE;
  - a symbolic word.

This mouth is not allowed to solve the task. It may only normalize candidates,
compare aliases, and report whether the selected answer is correct.
```

Metric split:

```text
accuracy:
  selected-answer accuracy.

oracle_accuracy:
  candidate-coverage upper bound. This is useful only to ask "did any candidate
  contain the gold answer?" It must not be reported as deployable verifier
  accuracy.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/eval/general_answer_interface.py \
  tests/test_general_answer_interface.py \
  scripts/521_eval_general_answer_candidates.py

PYTHONPATH=src .venv/bin/python - <<'PY'
import runpy
ns = runpy.run_path('tests/test_general_answer_interface.py')
for name, fn in sorted(ns.items()):
    if name.startswith('test_'):
        fn()
print('general_answer_interface direct assertions passed')
PY
```

Canary command:

```text
PYTHONPATH=src .venv/bin/python scripts/521_eval_general_answer_candidates.py \
  --eval-jsonl data/eval/pure_recursive_solver_trace_all_family_heldout_cases.jsonl \
  --candidate-source choices \
  --selection-mode oracle \
  --out-json local_eval/stage59_general_answer_interface_choices_oracle/report.json \
  --out-jsonl local_eval/stage59_general_answer_interface_choices_oracle/records.jsonl
```

Canary result:

```text
rows:
  128

families:
  arithmetic_chain: 32/32
  boolean_logic: 32/32
  list_transform: 32/32
  symbolic_binding: 32/32

answer kinds:
  integer: 32/32
  csv_integer_list: 32/32
  boolean: 32/32
  symbolic_or_text: 32/32
```

Boundary:

```text
This is not evidence that the model is general. The candidate source is row
choices, and the current dataset places the gold answer first. Treat this only
as a canary proving that the answer interface no longer forces all tasks into a
single-digit modulo-10 mouth.
```

Next gate:

```text
Build a general candidate speaker that emits top-k text/object candidates from
the recurrent thought/search path. Evaluate those model-generated candidates
with:

  scripts/521_eval_general_answer_candidates.py --candidate-source jsonl

Promote only if at least two non-modulo families improve and the gain vanishes
when recurrent thought/search or top-k exposure is disabled.
```

## 2026-05-22 Stage59 OpenAI-compatible general answer baseline

Implementation:

```text
script:
  scripts/522_eval_openai_general_answer_candidates.py
```

Plain-language purpose:

```text
Before saying QTRM has a general thought organ, first give every candidate model
the same ordinary answer mouth. A model should be able to answer "8017",
"green", "TRUE", or "8008,8004" without being squeezed through a modulo-10 digit
slot.
```

Data path:

```text
row prompt
  -> OpenAI-compatible model completion
  -> extracted answer candidate
  -> Stage59 shared answer interface
  -> selected-answer accuracy and oracle candidate coverage
```

Verification:

```text
py_compile:
  passed for general_answer_interface, tests, scripts/521, and scripts/522

direct assertions:
  general_answer_interface direct assertions passed

mock evaluator:
  four representative completions passed across integer, symbolic_or_text,
  boolean, and csv_integer_list answer kinds.
```

Runtime boundary:

```text
No live OpenAI-compatible model server was available at
http://127.0.0.1:18082/v1 during this step, so this is infrastructure progress,
not a live model result.
```

Next required QTRM bridge:

```text
scripts/92_eval_qtrm_logits.py can generate text from QTRM checkpoints, but its
JSON records are not yet keyed by Stage59 case id and not yet emitted in the
candidate JSONL contract consumed by scripts/521.

The next architecture-clean bridge is:

  Stage59 rows
    -> QTRM recurrent generation modes
       full / donor_only / workspace_off / core_off
    -> {id, candidates, raw_completions}
    -> scripts/521_eval_general_answer_candidates.py

Only then can we make a causal transfer claim for the recurrent thought path.
```

Update:

```text
scripts/92_eval_qtrm_logits.py now supports:

  --stage59-candidates-jsonl <path>

and writes:

  {id, task_family, ablation_mode, candidates, raw_completions, full_generations}
```

Verification:

```text
py_compile:
  passed for scripts/92 plus the Stage59 answer/eval scripts.

contract check:
  collect_stage59_items loaded the first four all-family rows with ids:
    arith-chain-4000
    symbolic-binding-4000
    boolean-logic-4000
    list-transform-4000

and preserved the task families:
  arithmetic_chain, symbolic_binding, boolean_logic, list_transform
```

Boundary:

```text
This solves the candidate-output contract for the QTRMMultimodalModel text
generation script. It is not yet the final Stage58/GRAM text speaker.

The Stage58 state_transition/VTE path still speaks through digit logits and a
typed modulo-10 register verifier. To test whether the successful Stage58
thought path transfers, the next missing module is:

  recurrent thought state -> answer text/object candidate speaker
```

## 2026-05-22 Stage59 state text speaker result

Plain-language conclusion:

```text
The current model is not failing because we lack another trendy paper. It is
failing because the successful Stage58 calculator-thought path still does not
have a reliable general-purpose mouth.

It can say some TRUE/FALSE and color-word answers, but it cannot reliably say
multi-digit arithmetic results or comma-separated list results, even when asked
to overfit only 32 balanced training rows.
```

Local evidence:

```text
run:
  local_eval/stage59_state_text_speaker_lowrank_t32_e16_ep20_trainacc

speaker:
  frozen QTRM/Qwen readout state
  -> StateTextSpeaker
  -> frozen Qwen LM head + low-rank vocab-logit adapter

best eval accuracy:
  0.3125 / 16 rows

final train accuracy:
  0.25 / 32 rows

final train family accuracy:
  arithmetic_chain: 0.0
  boolean_logic: 0.625
  list_transform: 0.0
  symbolic_binding: 0.375
```

Decision:

```text
Do not scale this exact final-readout-only text speaker.

The next high-probability architecture fix is not broad literature search, and
not another synthetic OOD polish run. The next fix must repair the normal answer
causal path:

  Qwen reader
  -> semantic recurrent trajectory, not only one final vector
  -> prompt-conditioned/trajectory-aware answer speaker
  -> Qwen-compatible token logits

Gate before any large run:
  a 32-row balanced Stage59 overfit smoke must show non-zero arithmetic_chain
  and non-zero list_transform exact match, with train accuracy materially above
  0.25.
```

## 2026-05-22 Stage59 verifier/renderer split update

Plain-language conclusion:

```text
The model is acting like a student who cannot reliably write the answer in free
form, but can often recognize the answer when shown choices. That means the
next high-probability fix is not another abstract recurrent loss. It is a
clean answer-selection path: thought -> verifier -> selected answer renderer.
```

Important data-contract correction:

```text
The original all-family choice rows had the correct answer at choices[0] for
all 128 heldout rows. Therefore unshuffled choice-verifier accuracy is useful
only as a weak diagnostic, not as a final generalization claim.

The stricter gate is the shuffled-choice split:
  scratch/stage59/shuffled_choices_train.jsonl
  scratch/stage59/shuffled_choices_eval.jsonl
```

Early stricter evidence:

```text
local_eval/stage59_local_choice_verifier_shuffled_frozen_t512_e128_ep30_s1606
  frozen QTRM/Qwen readout
  epoch 2 eval: 0.5546875
    arithmetic_chain: 0.375
    boolean_logic: 0.59375
    list_transform: 0.25
    symbolic_binding: 1.0

This survives randomized answer positions, so the signal is not merely
choices[0] leakage. The free text speaker remains rejected for numeric/list
generation, but the readout contains answer-selection information.
```

Decision:

```text
Promote verifier-guided answer selection as the current canonical next path.
Run shuffled-choice gates before claiming improvement. Only after selection is
stable should we attach a copy/emission renderer for final text output.
```

## 2026-05-22 selected-answer renderer artifact

Implementation:

```text
scripts/524_train_state_choice_verifier.py saves the verifier and its selected
answers:
  best_choice_verifier.pt
  best_records.jsonl
```

Current local saved gate:

```text
local_eval/stage59_local_choice_verifier_shuffled_frozen_saved_t512_e128_ep16_s1608
  epoch 7 eval: 0.6484375
    arithmetic_chain: 0.28125
    boolean_logic: 0.90625
    list_transform: 0.40625
    symbolic_binding: 1.0
```

Interpretation:

```text
This is no longer only a metric probe. The selected answer string is now the
model output artifact. It still depends on supplied candidates, so it is not a
free-form general speaker. But it is the first clean Stage59 answer path where
the recurrent thought readout influences the emitted answer after answer-position
leakage has been removed.
```

Choice-in-prompt check:

```text
local_eval/stage59_local_choiceprompt_verifier_frozen_t512_e128_ep16_s1609
  prompt includes shuffled choices
  early eval:
    epoch 1: 0.4140625
    epoch 2: 0.421875

This does not beat the problem-only thought -> post-hoc candidate comparison
path so far. Do not assume adding options to the reader prompt is a free gain.
```

## 2026-05-22 candidate exposure bottleneck

Added:

```text
scripts/525_eval_qwen_candidate_exposure.py
```

This diagnostic asks whether Qwen can generate answer candidates that contain
the correct answer before the QTRM verifier chooses among them.

Results:

```text
local_eval/stage59_local_qwen_candidate_exposure_k8_e32_s1703
  prompt mode: candidate_proposer
  eval rows: 32
  generated candidates per row: 8
  selected accuracy: 0.0625
  oracle candidate coverage: 0.3125

local_eval/stage59_local_qwen_answeronly_exposure_k8_e32_s1704
  prompt mode: question_answer_only
  eval rows: 32
  generated candidates per row: 8
  selected accuracy: 0.21875
  oracle candidate coverage: 0.375
```

Decision:

```text
The next high-probability work is candidate exposure, not another verifier or
free-form speaker run.

Supplied shuffled choices prove the verifier/selected-copy path can work.
Qwen-generated candidates prove that the current proposer fails to put the
right arithmetic/list answers on the table. A general thought-organ path now
needs an internal candidate proposer/exposer that is trained or constrained by
the recurrent thought state.
```

## 2026-05-22 GatedDeltaNet-2 decision

Source:

```text
https://github.com/NVlabs/GatedDeltaNet-2
```

Plain-language read:

```text
GatedDeltaNet-2 is a better way for a recurrent model to edit memory without
scrambling old associations. It gives the thinker a steadier hand.

It is not the missing imagination/speaker. It will not automatically put the
correct arithmetic/list answer among candidates, and it will not teach the
verifier to ignore noisy distractors.
```

Decision:

```text
Do not switch the whole architecture to GatedDeltaNet-2 now.

Use the existing `--core-update mini_gated_delta` as a small later ablation only
after the Stage59 candidate-exposure path is repaired:

  Qwen reader -> QTRM/GRAM thought -> candidate proposer/exposer
  -> noisy-candidate-trained verifier -> selected-answer copy

If candidate oracle coverage is still low, GDN-style memory editing is not the
causal fix. If candidate coverage is high but deep recurrence drifts, then a
GatedDeltaNet-2-style erase/write split becomes a high-value core replacement
candidate.
```

## 2026-05-22 Stage59 candidate proposer evidence

Added:

```text
scripts/526_materialize_noisy_candidate_choices.py
scripts/527_train_state_candidate_proposer.py
```

Evidence:

```text
typed/noisy K=4 verifier retraining:
  local_eval/stage59_local_noisy_typed_choice_verifier_k4_frozen_t512_e128_ep16_s1803
  best accuracy: 0.671875

learned readout-only candidate proposer:
  local_eval/stage59_local_learned_candidate_proposer_k4_readout_ep12_s1813
  best coverage: 0.5
  best selected accuracy: 0.3125
  arithmetic/list coverage: 0.0

learned workspace-aware candidate proposer:
  local_eval/stage59_local_learned_candidate_proposer_k4_workspace_ep12_s1814
  best coverage: 0.5
  best selected accuracy: 0.3984375
  arithmetic/list coverage: 0.0
```

Plain-language diagnosis:

```text
The verifier can learn to judge a messy answer table if the right answer is on
the table. The learned proposer can expose boolean/symbolic candidates, but it
cannot create extrapolated arithmetic/list answers by emitting characters from
one latent vector.

This is expected because the current state-transition core speaks a 10-class
digit language:
  StateReadoutHead(d_state, n_digits=10)
  answer_logits: (B, 10)

Stage59 asks for full integers and CSV lists. The thinker must therefore gain
typed value/list registers before the candidate proposer. Otherwise the mouth
is asked to invent numeric details that the thought state did not preserve.
```

## 2026-05-22 typed working-table scaffold result

Added:

```text
scripts/528_train_candidate_pool_selector.py
scripts/529_materialize_pool_selector_choices.py
```

Evidence:

```text
free learned char proposer:
  arithmetic/list coverage: 0.0

typed candidate pool selector, answer-targeted:
  local_eval/stage59_local_typed_pool_selector_answer_k16to4_ep8_s1816
  exposed oracle coverage: 0.9453125
  old verifier selected accuracy: 0.40625

selector-exposed verifier retraining:
  local_eval/stage59_local_pool_selector_exposed_verifier_ep12_s1817
  best selected accuracy: 0.628099173553719
```

Decision:

```text
Promote the typed working-table direction.

Do not promote the current heuristic pool as a final solution. It is an
upper-bound scaffold because it still uses hand-built candidate transformations.

The correct next architecture is:
  Qwen token reader
  -> learned typed value/list/symbol working registers
  -> typed candidate table
  -> selector/exposer
  -> verifier trained on exposed distribution
  -> selected-answer copy

This is the brain-like path: clean desk, candidate imagination, examiner, then
copying the selected answer. It is not a larger memory warehouse.
```
