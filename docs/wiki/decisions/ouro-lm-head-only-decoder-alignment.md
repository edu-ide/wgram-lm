# Ouro LM-Head-Only Decoder Alignment

Date: 2026-05-06

## Failure

After donor-unembedding head surgery failed, the next minimal hypothesis was:

```text
Maybe the accepted answer-halt hidden state is usable, but the LM head needs a
short untied decoder-alignment phase rather than a one-shot donor head
replacement.
```

This tests the renderer bottleneck without changing the accepted recursive
answer-halt core.

## Prior Checked

- Press and Wolf, "Using the Output Embedding to Improve Language Models"
  <https://arxiv.org/abs/1608.05859>
- Inan, Khosravi, Socher, "Tying Word Vectors and Word Classifiers"
  <https://arxiv.org/abs/1611.01462>
- PonderLM-2, "Pretraining LLM with Latent Thoughts in Continuous Space"
  <https://arxiv.org/abs/2509.23184>
- Coconut official implementation:
  <https://github.com/facebookresearch/coconut>
- PonderLM-2 official implementation:
  <https://github.com/LUMIA-Group/PonderLM-2>

Shared lesson:

```text
Latent states need a real LM-compatible rendering path. A correct hidden
answer-state score is not enough if the autoregressive token path is not
trained to read that state.
```

## Implemented Probe

Added trainable policy:

```text
trainable_param_policy: lm_head_only
```

Files:

```text
src/wgram_lm/training/train.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_lm_head_decoder_s080.yaml
tests/test_training_checkpoint_init.py
```

The policy trains only untied `lm_head.weight` and rejects tied embedding
models, so it cannot silently update `text_embed.weight`.

Training started from the accepted answer-halt S080 checkpoint:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_lm_head_decoder_s080_from_halt_s080/

steps: 80
trainable tensors: 1
trainable params: 127,139,840
```

## Results

Generation smoke4:

```text
full/gate_off: 0/8
full sample: "100 0601  6"
gate-off sample: "Whether Whether Compared Whether Trying Whether..."
```

Causal forced-choice smoke8:

```text
full:          0/8
halt_gate_off: 0/5 observed before early stop
```

The forced-choice eval was stopped after the full mode had already produced
eight held-out misses. This was enough to reject the checkpoint because it
destroyed the accepted baseline's main signal.

Decision:

```text
reject, checkpoint deleted
```

## Interpretation

Rejected:

```text
Training only the final LM head is too shallow. It changes token priors but
does not preserve the accepted causal forced-choice reasoning signal, and it
still cannot render correct greedy answers.
```

Updated root diagnosis:

```text
The bottleneck is not only output vocabulary geometry. The answer-state hidden
trajectory itself must be aligned into an LM-compatible hidden manifold before
the final LM head.
```

## Follow-Up

The immediate hidden-state bridge follow-up was tested and rejected:

```text
docs/wiki/decisions/ouro-hidden-bridge-s080-reject.md
```

Updated next root candidate:

```text
canonical prompt tokens
-> frozen donor decoder hidden states
-> QTRM recursive cognitive controller
-> small residual/modulation into donor-compatible hidden path
-> donor-compatible LM head / donor logits
-> autoregressive text
```

The renderer direction should now shift from private QTRM head repair to a
donor-decoder-preserving residual cognitive controller.
