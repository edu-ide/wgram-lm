# Qwen-Scope

Status: source and implementation note, accessed 2026-05-01.

Primary sources:

- Hugging Face collection:
  <https://huggingface.co/collections/Qwen/qwen-scope>
- Qwen3.5-2B Base SAE repo:
  <https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100>
- Technical report link from the model card:
  <https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf>

## What It Is

Qwen-Scope is not a new generator architecture. It is a set of Sparse
Autoencoders (SAEs) trained on Qwen3 and Qwen3.5 residual streams.

For `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100`:

- base model: Qwen3.5-2B / Qwen3.5-2B-Base family;
- hook point: residual stream;
- layers covered: 0-23;
- hidden size: 2048;
- SAE width: 32768;
- Top-K: 100;
- file format: one `layer{n}.sae.pt` tensor dict per transformer layer;
- tensor keys: `W_enc`, `W_dec`, `b_enc`, `b_dec`.

## QTRM Use

Use Qwen-Scope first as a donor diagnostics instrument:

1. Log donor residual SAE features for normal prompts.
2. Log the same features for repeated-output prompts such as `Freeze Freeze`
   or `world of the world`.
3. Compare supported, refuted, unsupported, and `NEEDS_SEARCH` examples.
4. Use feature-frequency differences to design verifier targets or data
   filters.
5. Only after stable correlations are proven, consider feature steering.

Do not treat SAE steering as the first fix. It can interfere with model
capabilities, and the Qwen-Scope model card includes a caution against
non-scientific capability interference.

## Implemented Tooling

Code:

- `src/wgram_lm/qwen_scope.py`
- `scripts/136_qwen_scope_probe.py`
- `scripts/137_compare_qwen_scope_groups.py`
- `scripts/138_score_qwen_scope_repeat_candidates.py`
- `tests/test_qwen_scope.py`
- `tests/test_qwen_scope_probe_script.py`
- `tests/test_qwen_scope_compare_script.py`
- `tests/test_qwen_scope_score_script.py`

The module loads official Qwen-Scope SAE files, validates tensor shapes, and
returns JSON-ready top-k feature records without materializing dense sparse
activation tensors.

Implementation note: batched prompts may be padded with the Qwen EOS token. The
probe therefore uses `attention_mask` and records the last non-padding token
instead of blindly taking `token_position=-1`.

Example:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
python scripts/136_qwen_scope_probe.py \
  --device cuda \
  --load-in-4bit \
  --layer 0 \
  --layer 12 \
  --layer 23 \
  --prompt "Explain quantum entanglement in simple terms." \
  --prompt "양자 컴퓨팅이란 무엇인가요?" \
  --out runs/qwen_scope/qwen35_2b_base_layers_0_12_23.jsonl
```

GPU note: this loads the donor model plus large SAE tensors, so do not run it
while a QTRM training job is using the GPU unless memory has been checked.

## Acceptance Gates

Qwen-Scope is useful for QTRM only if it produces measurable signals:

- repeated generations show repeat-associated SAE feature patterns;
- unsupported/evidence-missing examples separate from supported examples;
- donor-only and QTRM-residual runs show interpretable feature shifts;
- feature-derived filters improve held-out repetition or evidence-use metrics.

If these do not hold, keep Qwen-Scope as optional interpretability tooling and
do not wire it into the architecture.

## First Repeat-Vs-Normal Probe

Artifact:

- raw records:
  `runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_layers_0_12_23.jsonl`
- summary:
  `runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_summary.json`

Groups:

- normal prompt indices: `0,1,2`;
- repeated-output prompt indices: `3,4,5`;
- repeated final tokens: `Freeze`, `world`, `of`;
- layers: `0`, `12`, `23`;
- top-k logged per prompt/layer: `50`;
- comparison uses top-20 features.

Observed candidates:

| Layer | Repeat common top features | Repeat-shared-not-normal examples |
| --- | --- | --- |
| 0 | none | `16910`, `7605` |
| 12 | `847`, `2761`, `22167`, `24725`, `25296`, `26397` | `24725`, `25296`, `26397`, `22167`, `2761` |
| 23 | `29838`, `30452`, `31860` | `2248`, `12018`, `5121` |

Interpretation boundary:

- These are not proven "repetition neurons" or causal features.
- They are feature candidates that separate a tiny curated repeated-text set
  from a tiny normal set.
- The next gate is a larger prompt set with generated QTRM failures, not only
  hand-written repeated strings.

## QTRM-Generated Repeat Probe

Artifact:

- QTRM generation records:
  `runs/qwen_scope/qtrm_v2_generated_for_scope.jsonl`
- generated text prompt file:
  `runs/qwen_scope/qtrm_v2_generated_texts_for_scope.txt`
- Qwen-Scope records:
  `runs/qwen_scope/qtrm_v2_generated_layers_0_12_23.jsonl`
- summary:
  `runs/qwen_scope/qtrm_v2_generated_repeat_vs_normal_summary.json`

Source checkpoint:

- `runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt`
- config:
  `configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml`

Generated repetition split:

| Prompt index | Label | repeated_2gram_rate | repeated_3gram_rate | Notes |
| --- | --- | ---: | ---: | --- |
| 0 | normal | 0.016 | 0.000 | English quantum answer, visible `<think>` leakage but low repetition |
| 1 | normal-ish | 0.127 | 0.081 | Korean answer with generated multiple-choice format |
| 2 | repeat | 0.762 | 0.742 | Claim/evidence prompt repeats `Claim: Eiffel Tower...` |
| 3 | repeat | 0.254 | 0.161 | Math prompt starts a second copied problem |
| 4 | repeat | 0.175 | 0.145 | Korean history prompt repeats answer choices |

Overlap with curated repeat candidates:

| Layer | Overlap |
| --- | --- |
| 0 | none |
| 12 | `847` |
| 23 | `29838`, `31860` |

Interpretation:

- The curated repeated-string probe and the actual QTRM-generated repeat probe
  share layer-12/layer-23 SAE candidates.
- This supports using Qwen-Scope as a repeat-state diagnostic, but still does
  not prove causal control.
- The next useful step is a larger generated-failure set and a stricter
  detector that predicts high `repeated_2gram_rate` from these SAE candidate
  activations.

## Repeat Candidate Score Probe

Artifact:

- score JSON:
  `runs/qwen_scope/qtrm_v2_repeat_candidate_scores.json`

Detector:

- layer `12`: feature `847`;
- layer `23`: features `29838`, `31860`;
- score: prompt-level hit count plus summed candidate activation values;
- generation metrics are attached from
  `runs/qwen_scope/qtrm_v2_generated_for_scope.jsonl`;
- repeat label threshold: `repeated_2gram_rate >= 0.15`.

Command:

```bash
PYTHONPATH=src uv run python scripts/138_score_qwen_scope_repeat_candidates.py \
  --input runs/qwen_scope/qtrm_v2_generated_layers_0_12_23.jsonl \
  --metrics-jsonl runs/qwen_scope/qtrm_v2_generated_for_scope.jsonl \
  --candidate 12:847 \
  --candidate 23:29838,31860 \
  --out runs/qwen_scope/qtrm_v2_repeat_candidate_scores.json
```

Result:

| Prompt index | repeated_2gram_rate | Label | Candidate value sum | Hit count |
| --- | ---: | --- | ---: | ---: |
| 3 | 0.254 | repeat | 78.820 | 3 |
| 2 | 0.762 | repeat | 46.713 | 3 |
| 0 | 0.016 | normal | 23.338 | 3 |
| 4 | 0.175 | repeat | 0.500 | 1 |
| 1 | 0.127 | normal | 0.424 | 1 |

Interpretation:

- The two strongest prompt-copy/repetition failures rank highest.
- Prompt `0` is a clear false-positive risk: it has low repeated-2gram rate but
  strong layer-23 candidate activation at lower ranks.
- Prompt `4` is a mild Korean repeated-choice failure but scores low.
- Therefore this is useful as a severe-loop smoke detector, not yet as a
  general repetition classifier or causal repetition control mechanism.

Next gate:

- run at least 50-100 generated samples;
- separate severe prompt-copy loops from mild format/choice repetition;
- fit only transparent thresholds first, for example layer-23 value/rank gates;
- accept only if it improves held-out repeat detection without flagging normal
  CoT-style completions.

## 50-Sample Repeat Gate

Artifact:

- prompt suite:
  `runs/qwen_scope/qtrm_repeat_gate_prompts_s50.jsonl`
- QTRM generation eval:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_eval.jsonl`
- generated text prompt file:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_generated_texts.txt`
- Qwen-Scope layer records:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_layers_12_23.jsonl`
- category summary:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_category_summary.json`
- overlap-candidate scores:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_candidate_scores_rep15.json`
  and
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_candidate_scores_severe25.json`
- sparse-candidate scores:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_sparse_candidate_scores_rep15.json`
  and
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_sparse_candidate_scores_severe25.json`
- train/holdout split check:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_train_holdout_severe25_summary.json`

Prompt categories:

| Category | n | avg rep2 | rep2 >= 0.15 | rep2 >= 0.25 |
| --- | ---: | ---: | ---: | ---: |
| normal_qa | 10 | 0.081 | 2 | 1 |
| math_reasoning | 10 | 0.238 | 8 | 4 |
| evidence_check | 10 | 0.189 | 3 | 3 |
| korean_qa | 10 | 0.110 | 3 | 1 |
| repeat_stress | 10 | 0.081 | 2 | 2 |

The failure is not one uniform repetition mode:

- math prompts often answer correctly, then restart or copy the problem;
- evidence-check prompts can enter claim/evidence copy loops;
- some direct-answer prompts stop cleanly after a few tokens;
- some prompts have low n-gram repetition but still fail to stop before the
  `max_new_tokens` cap.

Candidate detector result:

| Candidate set | Label rule | Best threshold result |
| --- | --- | --- |
| Original overlap `12:847`, `23:29838,31860` | rep2 >= 0.15 | F1 0.652, precision 0.536, recall 0.833 |
| Original overlap `12:847`, `23:29838,31860` | rep2 >= 0.25 | F1 0.625, precision 0.476, recall 0.909 |
| Sparse in-sample candidates | rep2 >= 0.15 | F1 0.733, precision 0.917, recall 0.611 |
| Sparse in-sample candidates | rep2 >= 0.25 | F1 1.000 in-sample |

The sparse severe detector was then checked with a crude split:

- discovery split: prompt indices `0-24`;
- holdout split: prompt indices `25-49`;
- discovered candidates:
  layer `12` features `22191`, `20732`, `14551`, `13802`, `4724`,
  `7397`, `16904`, `21237`, `4003`, `24725`;
  layer `23` features `6203`, `19972`, `5515`;
- train threshold: `0.359375`;
- train result: F1 `1.0`;
- holdout result at train threshold: F1 `0.0`, `tp=0`, `fp=3`, `fn=3`,
  `tn=19`.

Conclusion:

- Qwen-Scope SAE features are useful for diagnosing specific generated failure
  families.
- The current feature candidates do not generalize as a single universal
  repetition detector across category shifts.
- Do not wire Qwen-Scope candidates into decoding or training as a hard gate
  yet.
- The next architecture direction should be an on-policy output verifier or
  stop/format controller trained on QTRM-generated failures, with Qwen-Scope
  features used only as optional analysis covariates.
