# Qwen3.6 Teacher Distillation Roadmap

Date: 2026-04-30.

Status: revised 2026-05-03 after Subliminal Learning review. Qwen3.6-27B
teacher serving can be used for candidate generation and critique, but direct
teacher-answer imitation is no longer canonical.

## Goal

Use `Qwen/Qwen3.6-27B` as a strong teacher for QTRM/MSA without confusing three
different jobs:

```text
QTRM distillation: learn answer policy, evidence use, correction behavior, and
latent reasoning control.

MSA distillation: learn memory routing, donor healing, and doc-id sparse
attention behavior.

Joint validation: prove QTRM actually uses MSA/MemoryOS evidence through
ablation, not just better surface text.
```

Qwen3.6-27B is a good teacher candidate because it is an open-weight Qwen-family
model with a compatible tokenizer lineage, 27B parameters, 64 layers, and a
hybrid Gated DeltaNet/Gated Attention layout. Its model card reports native
262K context and extension to about 1M tokens.

Source:
`https://huggingface.co/Qwen/Qwen3.6-27B`

Safety source:
`https://arxiv.org/abs/2507.14805`

## Local Hardware Decision

DGX is the right machine for teacher serving:

```text
ssh dgx
host: edgexpert-5b20
GPU: NVIDIA GB10
system memory: 121 GiB available at check time
data disk: /mnt/data4tb, about 1.1T free
```

Current DGX gap:

- Python exists;
- `torch`, `transformers`, `vllm`, and `sglang` are not installed yet;
- teacher cache and generated distillation data should live under
  `/mnt/data4tb`, not the nearly full root disk.

## Distillation Types

### Safety Boundary: Verifier-Gated Online Only

Subliminal Learning shows that teacher-generated data can transmit behavioral
traits through hidden signals even when the visible data is unrelated to those
traits and filtered. The effect was reported for number sequences, code, and
reasoning traces, and is most relevant when teacher and student share or closely
match the base model.

For this roadmap, Qwen3.6 is not allowed to be the sole source of truth for
training labels.

Rejected default:

```text
Qwen3.6 answer / trace / logits
-> surface filter
-> direct SFT or KL target
-> QTRM
```

Accepted default:

```text
Qwen3.6 proposes answer candidates, critiques, hard negatives, or new problems
-> rule solver / unit test / symbolic verifier / retrieval evidence checker /
   human-approved gold label decides
-> QTRM trains on verified answer or explicit chosen/rejected preference
```

Qwen3.6 online calls remain useful, but the online loop must be a
`verifier-gated online distillation` loop, not direct teacher imitation.

### Public HF Dataset Warmup

Use before spending teacher-generation budget. Public datasets give cheap
baseline coverage for preference, reasoning, evidence, and hallucination gates.
The first-wave manifest is:
`configs/hf_distill_datasets.yaml`.

Source notes:
[HF Distillation Datasets](../sources/hf-distillation-datasets.md).

```text
verified HF datasets
-> prompt-only/gold-answer QTRM rows or executable-test rows
-> smoke train/eval
-> GPT-5.5 xhigh proposes missing cases, but gold/verifier decides labels
-> DGX Qwen3.6-27B handles candidate generation, critique, hard negatives,
   and bounded verifier-gated online updates
```

### Offline Teacher Data

Use only when the answer is externally verified. Teacher generates structured
records once; 4090 or DGX trains the student later.

```json
{
  "prompt": "...",
  "answer": "...",
  "evidence_ids": ["doc-1", "doc-7"],
  "evidence_spans": ["..."],
  "rejected_answer": "...",
  "trace_summary": "...",
  "teacher_model": "Qwen/Qwen3.6-27B"
}
```

Best for:

- QTRM warmup after verification;
- preference pairs;
- evidence/counterfactual cases;
- MSA routing labels;
- cheap repeated training on 4090.

Not accepted:

- single-teacher answer-only SFT without verifier;
- single-teacher CoT trace SFT;
- same-family Qwen synthetic data accepted by surface filtering alone.

### Online Top-k Logit Distillation

Use only as a bounded auxiliary after verified warmup. Teacher runs during
student training and returns only the useful token subset on verifier-approved
states.

```text
teacher(prompt + prefix) -> top-k token ids + logprobs
student(prompt + prefix) -> student logits on same token ids
loss += sparse/top-k KL
```

Full vocab KL over Qwen's large vocabulary is too expensive for the first
implementation. Start with top-k `32`, `64`, or `128`, then compare against a
Sparse Logit Sampling-style cache if calibration is poor.

Safety rule: top-k KL must not override gold labels. If teacher distribution
conflicts with a rule solver, unit test, or symbolic/evidence verifier, the
verified label wins and the row is logged for audit.

Prior:
`https://arxiv.org/abs/2503.16870`

### On-policy Correction Distillation

Use after the student can generate nontrivial outputs. Student generates its own
answers; teacher proposes judgments or corrections, but a verifier accepts or
rejects them before training.

```text
student rollout
-> teacher critique/correction candidates
-> verifier or gold label accepts/rejects
-> chosen/rejected pair
-> QTRM update with preference + KL + repeat guard
```

This directly targets the current failure mode where static pair preference can
pass while generation still repeats or ignores evidence.

Priors:

- `https://arxiv.org/abs/2306.08543`
- `https://arxiv.org/abs/2604.00626`
- `https://arxiv.org/abs/2602.22495`

## QTRM First Checklist

QTRM should be distilled before full MSA quality claims because the present
failure is generation-side: retrieval can find evidence, but the residual/core
path does not reliably use it.

- [x] Donor KL path exists for QTRM logits.
- [x] Student LM loss exists for QTRM-only logits.
- [x] SimPO-style preference loss exists.
- [x] Workspace/counterfactual evidence losses exist.
- [x] Create Qwen3.6 teacher record schema:
  `src/qtrm_mm/distill/teacher_schema.py`.
- [x] Add OpenAI-compatible teacher prompt/response parser:
  `src/qtrm_mm/distill/qwen36_teacher_client.py`.
- [x] Add HF public dataset intake manifest:
  `configs/hf_distill_datasets.yaml`.
- [x] Add HF dataset converters:
  `src/qtrm_mm/distill/hf_dataset_convert.py` and
  `scripts/131_convert_hf_distill_dataset.py`.
- [ ] Build verifier-gated offline/online teacher data generator CLI.
- [ ] Generate a 100-case verified smoke dataset:
  answer, evidence ids, rejected answer, trace summary.
- [ ] Train QTRM on verified teacher/gold data with:

```text
CE(chosen)
+ Qwen3.5 donor KL
+ SimPO/DPKD-style preference
+ workspace evidence contrastive
+ conservative repeat unlikelihood
```

- [ ] Eval gates:
  donor-only, current QTRM, offline-distilled QTRM.
- [ ] Required ablations:
  workspace off, core off, core-context off, evidence bottleneck off, donor
  hidden off.
- [ ] Accept only if:
  held-out answers improve, repetition drops, and at least one workspace/core
  ablation changes the completion or score.
- [ ] Add online top-k teacher KL.
- [ ] Add verifier-gated online top-k teacher KL.
- [ ] Add verifier-gated on-policy student rollout correction.
- [ ] Accept online distillation only if it improves generation, not just pair
  preference accuracy, and safety/trait regression gates do not worsen.

## MSA Second Checklist

MSA is a donor fork and must be validated separately. Teacher answer quality is
not enough; MSA needs document routing and healing supervision.

- [x] Qwen3.5 full-MSA fork conversion manifest exists.
- [x] Tiny Qwen3.5-native MSA forward pass exists.
- [x] Tiny healing smoke exists.
- [ ] Add checkpoint save/load roundtrip for the custom MSA model.
- [ ] Build real Qwen3.5-2B weight-copy dry run.
- [ ] Build MSA training records:

```json
{
  "prompt": "...",
  "memory_docs": [{"doc_id": 1, "text": "..."}, {"doc_id": 2, "text": "..."}],
  "target_doc_ids": [2],
  "answer": "..."
}
```

- [ ] MSA stage-1 healing:
  freeze embeddings, MLPs, norms, and LM head; train MSA attention/router.
- [ ] MSA stage-2 healing:
  unfreeze more text backbone if LM loss and KL are stable.
- [ ] MSA routing loss:
  selected doc ids should match teacher/ground-truth evidence ids.
- [ ] Eval gates:
  routing recall, answer exactness, donor KL, repetition stats, and latency.
- [ ] Accept only if:
  routing recall improves over external retrieval baseline and donor language
  quality does not collapse.

## Joint QTRM + MSA Checklist

Run only after QTRM and MSA each pass their own smoke gates.

- [ ] Plug healed MSA donor under QTRM.
- [ ] Compare:

```text
Qwen3.5 donor only
Qwen3.5 donor + QTRM
MSA donor only
MSA donor + QTRM
```

- [ ] Required ablations:
  MSA routing off, workspace off, core off, evidence bottleneck off, donor-only.
- [ ] Required metrics:
  retrieval/routing recall, answer score, repeated n-gram rate, donor KL,
  latency, memory use.
- [ ] Accept only if:
  MSA+QTRM beats both donor-only and QTRM-without-MSA on held-out memory cases,
  and turning off MSA routing or QTRM workspace measurably hurts.

## Work Order

1. Stabilize custom MSA checkpoint save/load. **Done for tiny custom
   checkpoint roundtrip.**
2. Define Qwen3.6 teacher data schema. **Done in
   `src/qtrm_mm/distill/teacher_schema.py`.**
3. Add teacher prompt/response parser. **Done in
   `src/qtrm_mm/distill/qwen36_teacher_client.py`.**
4. Convert first-wave HF datasets into teacher-record smoke files.
5. Configure DGX teacher environment under `/mnt/data4tb`.
6. Generate small GPT-5.5 xhigh gold fills only for missing domains.
7. Generate 100-case verified offline teacher/gold smoke data.
8. Train and evaluate QTRM verified-distill smoke.
9. Add verifier-gated online top-k teacher KL.
10. Add verifier-gated on-policy QTRM correction loop.
11. Add MSA routing-label loss and real 2B healing dry run.
12. Run joint QTRM+MSA ablations.

## Claim Boundary

Online distillation is expected to be stronger than static SFT for generation
failure modes, but it is not accepted by loss alone. The claim must be proven
with held-out generation and ablations that show the trained path is causally
used.
