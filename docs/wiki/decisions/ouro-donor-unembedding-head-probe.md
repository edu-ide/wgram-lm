# Ouro Donor Unembedding Head Probe

Date: 2026-05-06

## Failure

The accepted Ouro answer-halt S080 checkpoint can score the right answer under
causal forced-choice, but greedy and beam autoregressive rendering still emit
invalid token priors such as `UNKNOWN`, `1 1 1`, and multilingual fragments.

Root hypothesis tested here:

```text
Maybe the answer-state loop is usable, but QTRM's LM head is poorly aligned to
the Qwen tokenizer vocabulary because it is tied to QTRM's randomly initialized
text embedding rather than to the donor output embedding geometry.
```

## Prior Checked

- Press and Wolf, "Using the Output Embedding to Improve Language Models"
  <https://arxiv.org/abs/1608.05859>
- Inan, Khosravi, Socher, "Tying Word Vectors and Word Classifiers"
  <https://arxiv.org/abs/1611.01462>
- PonderLM-2, "Pretraining LLM with Latent Thoughts in Continuous Space"
  <https://arxiv.org/abs/2509.23184>
- Coconut official implementation:
  <https://github.com/facebookresearch/coconut>
  local clone: `references/official/coconut`,
  commit `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`.
- PonderLM-2 official implementation:
  <https://github.com/LUMIA-Group/PonderLM-2>
  local clone: `references/official/PonderLM-2`,
  commit `fa784bece621b989fb008c59b0fd8d282fa9c73c`.

Mechanism extracted:

```text
Output/vocab geometry matters. Latent thought work also predicts real tokens
from an LM-compatible hidden state; it does not leave the final answer hidden
in an arbitrary private vector space.
```

## Implemented Probe

Script:

```text
scripts/246_build_donor_unembedding_aligned_checkpoint.py
```

Tests:

```text
tests/test_donor_unembedding_aligned_checkpoint_script.py
```

The script builds an eval checkpoint by replacing only `lm_head.weight` with
the donor output embedding projected into QTRM width through the checkpoint's
trained `projector.visual_proj.weight`.

Two mappings were tested:

```text
pinv:
  W_qtrm = W_donor_out @ pinv(P)

project:
  W_qtrm = W_donor_out @ P.T
```

The default probe keeps `text_embed.weight` unchanged and writes an untied
config, so the trained input-token path is not overwritten.

## Results

Pinv donor-output head:

```text
artifact:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_unembed_head_pinv_from_halt_s080/

generation smoke4:
  full/gate_off: 0/8
  sample: "statusestraemplaestra' status'对各' skestra status"

causal forced-choice smoke8:
  full:          6/8
  halt_gate_off: 0/8

decision:
  reject, checkpoint deleted
```

Direct projection donor-output head:

```text
artifact:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_unembed_head_project_from_halt_s080/

generation smoke4:
  full/gate_off: 0/8
  sample: "static久 dead dead替代替代代替替代径 status替代 status"

decision:
  reject, checkpoint deleted before forced-choice because generation remained 0/8
```

## Interpretation

Rejected:

```text
Vocab-head surgery alone is not enough. Better donor output geometry changes
the junk tokens but does not make answer-state hidden vectors land in a
tokenizer-compatible answer manifold.
```

New root diagnosis:

```text
The bottleneck is answer_state_hidden -> LM-compatible hidden alignment, not
only LM head geometry or greedy exposure bias.
```

## Follow-Up

The immediate LM-head-only decoder alignment follow-up was tested and rejected:

```text
docs/wiki/decisions/ouro-lm-head-only-decoder-alignment.md
```

Updated next architecture candidate:

```text
answer-state hidden
-> trainable hidden-state bridge / residual decoder adapter
-> LM-compatible hidden
-> tokenizer LM head
-> autoregressive text
```

Do not add more small margin losses, donor-head replacement, or LM-head-only
tuning before this hidden-bridge falsifier is tested.
