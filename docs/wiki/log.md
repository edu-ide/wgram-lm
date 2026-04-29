# QTRM LLM Wiki Log

## [2026-04-29] decision | limitations and mitigation roadmap

Added `decisions/limitations-mitigation-roadmap.md` to keep the current QTRM
limits tied to concrete mitigations: Proxy-Tuning/DExperts for logit residuals,
Side-Tuning/Ladder Side-Tuning for the frozen-donor sidecar pattern,
KL/distillation for donor preservation, looped-transformer/Parcae references for
recursive stability, and Self-RAG/CRAG/RAPTOR-style gates for MemoryOS. The next
engineering order is telemetry, ablations, gated residual, and KL-to-donor loss
before longer training.

## [2026-04-29] plan | current architecture pretrain viability probe

Added `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` and
`scripts/105_run_current_arch_pretrain_probe.sh`. This locks the next
architecture gate before more reasoning-data expansion: Qwen donor logits remain
the base language policy, QTRM contributes only a bounded residual
(`qtrm_logits_scale=0.10`), workspace is enabled, and JEPA/aux losses stay off.

The probe trains 2000 steps on `data/filtered/qtrm_clean_pilot.jsonl`, runs live
prompt diagnostics, and writes `post_eval.jsonl` plus `post_eval_prompts.txt`
under `runs/qwen35_2b_4090_current_arch_pretrain_probe`.

## [2026-04-29] implementation | Bongakgyo critical synthesis case expansion

Added `src/qtrm_mm/training/bongak_critical_synthesis_cases.py` and
`scripts/104_build_bongak_critical_synthesis_cases.py`. Built 30
`bongak_critical_synthesis` cases from the local
`/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_요약.md`
and
`/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_매뉴얼.md`
sources.

Outputs:
`data/filtered/critical_synthesis_bongak_cases.jsonl` and
`data/filtered/critical_synthesis_bongak_traces.jsonl`, 30 rows each. These
rows intentionally treat the local doctrine as source material rather than
truth labels: the target answers must critique weak or controlling claims,
preserve useful practice values, bracket unverifiable metaphysics, and end with
a positive constructive conclusion.

## [2026-04-29] implementation | critical synthesis trace builder

Added `src/qtrm_mm/training/critical_synthesis_data.py` and
`scripts/103_build_critical_synthesis_traces.py`. Built
`data/filtered/critical_synthesis_traces.jsonl` from
`data/eval/critical_synthesis_probe.jsonl`. The generated rows preserve the
required structure: critique, preserve, risks, reframe, and positive
conclusion.

## [2026-04-29] implementation | fact verification and critical synthesis gates

Added a deterministic fact-verification gate:
`src/qtrm_mm/eval/fact_verification.py`,
`src/qtrm_mm/training/fact_verification_data.py`,
`scripts/102_eval_fact_verification_memoryos.py`, and
`data/eval/fact_verification_probe.jsonl`. The gate separates verdict accuracy,
action accuracy, retrieval recall, temporal priority, authority priority, and
conflict handling before generation.

Added the critical-synthesis axis for religion/value questions:
`src/qtrm_mm/eval/critical_synthesis.py` and
`data/eval/critical_synthesis_probe.jsonl`. The target is not blind skepticism;
it requires critique, preservation of value, risk checks, reframing, and a
constructive positive conclusion. Added wiki pages for value/critical synthesis
and the 본각교 handling contract.

## [2026-04-29] ingest | fact verification and fake-info references

Downloaded fact-verification and fake-info papers under
`references/papers/fact_verification`: FEVER, FActScore, LongFact/SAFE,
RAGTruth, OpenFActScore, CIBER, CONFACT, CARE-RAG, verifiable misinformation
agent, LiveFact, and ArbGraph. Cloned official repositories for FEVER baseline,
FActScore, LongFact/SAFE, RAGTruth, OpenFActScore, CONFACT, and ArbGraph under
`references/official`.

Added wiki pages for fact-verification reasoning. The design now separates
predictive latent intuition from factual truth: LeWorldModel remains the
world-model prior, while MemoryOS needs retrieval, source metadata, atomic claim
verification, conflict arbitration, temporal labels, and `NEEDS_SEARCH` routing.

## [2026-04-29] implementation | self-improvement preference rows

Downloaded current self-improvement/hallucination references under
`references/papers/self_improvement` and added wiki pages for the self-
improvement loop. The design now treats `UNKNOWN` as a closed-evidence eval
label, not the final open-world MemoryOS answer. In agentic mode, insufficient
evidence routes to `Action: NEEDS_SEARCH`, then retrieval/search expands the
evidence set before answering or reporting bounded non-verification.

Added `src/qtrm_mm/training/self_improvement_data.py` and
`scripts/101_build_self_improvement_preferences.py`. Built
`data/filtered/memory_self_improvement_preferences_analysis.jsonl` from the
held-out MemoryOS eval: 11 analysis-only rows, with 6 `needs_search` states and
5 answer-correction states. These rows are explicitly `analysis_only` to avoid
held-out train leakage.

## [2026-04-29] eval | held-out MemoryOS generalization

Ran the new held-out MemoryOS reasoning gate after synthetic trace expansion.
The checkpoint `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
evaluated on `data/eval/memory_reasoning_heldout_probe.jsonl` with Harrier
top-30, Qwen3-Reranker-0.6B, and top-5 evidence.

Result: donor-only stayed at 6/12, while QTRM residual improved to 9/12.
QTRM residual solved abstention 4/4 but missed two Korean conflict cases and
one English multi-hop case. This confirms the earlier 9-case trace fine-tune
was overfit, while the broader synthetic trace set generalizes partially.

Next gate: fix Korean temporal/authority conflict selection, the remaining
multi-hop miss, and UNKNOWN repetition artifacts before scaling MemoryOS.

## [2026-04-29] synthesis | long-horizon agent references

Added `sources/long-horizon-agent-references.md` and
`concepts/long-horizon-agent-architecture.md`. The wiki now records the current
agentic/long-context reference bundle: Externalization in LLM Agents, Memory for
Autonomous LLM Agents, MemGPT, Memex(RL), Recursive Language Models, ReAct,
Reflexion, Voyager, SWE-agent, Self-RAG, CRAG, and RAPTOR.

Decision captured: QTRM should not be redesigned as a standalone long-running
agent. Long-running behavior belongs in MemoryOS plus an agent harness with
mode routing, indexed evidence, trace memory, reflection memory, skill memory,
sandboxed execution, budgets, and verification gates. RLM is a future inference
mode, not the default training target.

## [2026-04-29] ingest | Titans neural memory

Added `references/official/titans-pytorch` at commit `714a14c` and downloaded
`references/papers/long_term_memory/titans_2501.00663.pdf`. Added wiki pages
for Titans and neural long-term memory. QTRM now treats Titans as a future
long-term/test-time memory ablation, separate from the current in-context
`LatentWorkspace` and donor-logit residual path.

Clarified architecture wording: QTRM performs looped latent-workspace
computation over donor representations, but it should be described as a
Qwen-backed looped latent-workspace residual adapter rather than a standalone
loop LM or a proven latent-reasoning LM.

## [2026-04-29] experiment | residual 0.10 stability gate

Finished `configs/qwen35_2b_4090_donor_residual_s010_1000.yaml` and saved
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`. Live diagnostics at
200/400/600/800/1000 steps remained stable, and independent reload evaluation
passed the same language-stability gate as residual `0.05`.

Metrics: Korean quantum prompt `loss=2.1777`, `ppl=8.83`,
`rank_mean=6.62`, `rep2=0.000`; English entanglement prompt `loss=2.8766`,
`ppl=17.75`, `rank_mean=15.40`, `rep2=0.032`; math prompt `loss=2.2573`,
`ppl=9.56`, `rank_mean=53.67`, `rep2=0.115`. The math prompt generated the
correct answer and stopped without continuing into another algebra example.

Decision: `qtrm_logits_scale=0.10` is inside the current safe residual range
and becomes the next working candidate. Do not scale residual strength further
until donor-only versus QTRM-residual memory/retrieval evals exist.

## [2026-04-29] experiment | residual 0.05 stability gate

Finished `configs/qwen35_2b_4090_donor_residual_s005_1000.yaml` and saved
`runs/qwen35_2b_4090_donor_residual_s005_1000/last.pt`. Independent reload
evaluation passed the language-stability gate: Korean and English generations
remained coherent with no `Freeze`, no `world of the world`, and no single-token
collapse. Metrics: Korean quantum prompt `loss=2.2096`, `ppl=9.11`,
`rank_mean=6.88`, `rep2=0.000`; English entanglement prompt `loss=2.9217`,
`ppl=18.57`, `rank_mean=16.40`, `rep2=0.048`. The math prompt answered `x=4`
then continued into another synthetic algebra example; this is template
continuation rather than the earlier collapse mode, but future math evals need
answer-only stop criteria.

Verification: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`
ran 36 tests successfully, and `py_compile` passed for the modified model,
training, donor, loss, and eval files.

## [2026-04-29] architecture | QTRM forward pass diagrams

Added `docs/wiki/architecture/qtrm-forward-pass.md` with Mermaid flowcharts,
generation sequence, tensor shape ledger, and current donor-logit residual
contract. The page explicitly separates latent-workspace computation from
claims of independent latent reasoning or standalone QTRM language generation.

## [2026-04-29] synthesis | Karpathy cognitive core and QTRM goal

Added corrected wiki notes for the Karpathy/Dwarkesh cognitive-core claim. The
wiki now distinguishes the supported hypothesis from the social-media overclaim
that a clean 1B model directly equals a 1.8T frontier model. Added QTRM goal and
scope decision: prioritize donor baseline, data quality, tiny-overfit,
target-token-rank, entropy/repetition diagnostics, memory traces, and ablations
before another long training run.

## [2026-04-29] implementation | QTRM training diagnostics

Added diagnostic utilities and scripts:
`scripts/91_donor_only_generate.sh`, `scripts/92_eval_qtrm_logits.py`,
`scripts/92_eval_qtrm_logits.sh`, and
`scripts/93_tiny_overfit_donor_adapter.sh`. The first checkpoint probe on
`runs/qwen35_2b_4090_extended/last.pt` shows `Freeze` as the top next token and
greedy completion collapse with donor on and off. The checkpoint also predates
the current LeWM/SIGReg JEPA module layout, causing 45 missing new JEPA keys and
4 unexpected old `jepa.net.*` keys during non-strict loading.

Follow-up diagnostics: Qwen donor-only generation is coherent on Korean prompts.
Current-code donor-on tiny-overfit reaches `rank=1.0` and `top1=1.0`, and a
saved tiny checkpoint continues `Quantum entanglement means` without `Freeze`.
Found and fixed a real data pipeline bug: HF-tokenized JSONL samples now carry
an `attention_mask` based on the tokenizer pad id instead of assuming pad id
`0` in `collate_jsonl`.

Ran `configs/qwen35_2b_4090_fixed_pilot.yaml` for 120 real-data steps after
the mask fix. The pilot loss moved from roughly `lm=12.51` to `lm=9.19`.
Evaluation on the Korean quantum prompt no longer selected `Freeze`; the top
tokens were high-frequency punctuation/function words and greedy decoding
formed `, and the.` loops. Interpretation: current code and fixed masking avoid
the old `Freeze` attractor in this short pilot, but the run is far below a
usable LM loss and still needs clean data, longer training, and live
entropy/repetition gates.

## [2026-04-29] ingest | OpenMythos and recurrent depth

Added `references/official/openmythos` at `8c68c1f` and
`references/official/parcae` at `dee8363`. Downloaded recurrent-depth papers:
Parcae stable looped LMs, Looped Transformers are Better at Learning Learning
Algorithms, Reasoning with Latent Thoughts, and a negative/limited latent-CoT
probing paper. Updated recursive-core docs with stable injection, depth-sweep,
and recurrence telemetry requirements.

## [2026-04-29] ingest | Architecture search and composition

Added architecture-search papers under `references/papers/architecture_search`:
Transformer modification transfer, NAS survey, RegNet/design spaces,
EfficientNet, ConvNeXt, task arithmetic, NAS search-phase evaluation, and No
Free Lunch. Added wiki pages for treating QTRM as a compositional architecture
that must be validated through ablations rather than assumed optimal.

## [2026-04-29] ingest | Training diagnostics

Added training-diagnostics papers under
`references/papers/training_diagnostics`: LLM scaling laws, Chinchilla
compute-optimal training, gradient noise scale, LC-PFN learning-curve
extrapolation, ACL 2025 LLM training dynamics, and Prechelt early stopping.
Added wiki pages for using these papers as QTRM run-failure probes.

## [2026-04-29] ingest | LeWorldModel

Added `references/official/le-wm` and `references/papers/leworldmodel_2603.19312.pdf`.
Changed the JEPA target from older stop-grad/EMA style to LeWM-style
end-to-end next-embedding prediction plus SIGReg.

## [2026-04-29] ingest | Tiny Recursive Models

Added `references/official/tiny-recursive-models` and
`references/papers/tiny_recursive_models_2510.04871.pdf`. Marked current QTRM
recursive core as needing comparison against TRM carry, no-grad cycles,
carry-detach, and ACT halting.

## [2026-04-29] ingest | Gated DeltaNet

Added `references/official/gated-delta-net` and
`references/papers/gated_delta_networks_2412.06464.pdf`. Marked current
`TorchGatedDeltaMixer` as a smoke/debug fallback, not official Gated DeltaNet.

## [2026-04-29] implementation | Gated DeltaNet adapter

Updated `src/qtrm_mm/mixers.py` to prefer the official FLA import path
`from fla.layers import GatedDeltaNet`. Added adapter tests for strict mode,
non-strict fallback, mask forwarding, and backend registry detection.
Updated the 4B adapter config and production backend docs to prefer
`delta_backend: fla_gated_delta`.

## [2026-04-29] ingest | Karpathy LLM Wiki

Added wiki schema under `docs/wiki`. Adopted the raw-source / synthesized-wiki /
schema split for QTRM architecture work.

## [2026-04-29] ingest | Qwen3.5 donor architecture

Downloaded Qwen3.5 2B Base and chat model cards/configs into
`references/model_configs`. Added Qwen3.5 Omni technical report
`references/papers/qwen35_omni_technical_report_2604.15804.pdf`. Created wiki
pages for Qwen3.5 architecture and donor integration constraints.

## [2026-04-29] ingest | Qwen donor lineage

Added `references/official/qwen35`,
`references/papers/qwen3_vl_technical_report_2511.21631.pdf`, and
`references/papers/qwen3_omni_technical_report_2509.17765.pdf`. These cover
Qwen3.5 release guidance plus Qwen3-VL/Qwen3-Omni design lineage.

## [2026-04-29] ingest | Transfer, merge, healing

Added model merging and healing-tune references under
`references/official/model-merging` and `references/papers/model_merging`.
Covered MergeKit, TIES, DARE, DELLA, Model Soups, Branch-Train-Merge,
continual/domain-adaptive pretraining, model-merging surveys, and
merge-friendly fine-tuning. Added QTRM wiki pages for the transfer/healing axis.

## [2026-04-29] diagnostics | QTRM collapse probes

Added donor-only, logit, and tiny-overfit diagnostics. The donor-only Qwen
baseline generates coherent text, and a 16-sample donor-on tiny-overfit run
reaches `rank=1.0`/`top1=1.0`, so the current path can learn. The old extended
checkpoint is invalid for current architecture comparisons because it has JEPA
state mismatch and collapses to `Freeze`.

Fixed HF padding-mask handling in `jsonl_dataset`: padding is now derived from
the tokenizer pad id instead of assuming `0`. Added a clean text-only pilot data
builder and built `data/filtered/qtrm_clean_pilot.jsonl` with 6000 accepted
rows. The 500-step clean pilot reached roughly `lm=6.86` from an initial
`lm=12.52`. It removed the `Freeze` top-token failure but still free-runs into
dialogue markers and repeated high-frequency phrases, so the next work is an
objective/data/architecture ablation rather than another blind long run.

Also fixed autoregressive donor handling in `scripts/90_infer_with_donor.sh`
and training diagnostics: donor states are refreshed against the full generated
sequence instead of being held at the initial prompt length. Re-running the clean
checkpoint with refreshed donor states still repeats, confirming this was a
real consistency bug but not the remaining collapse's primary cause.

Added configurable train loss weights and ran
`configs/qwen35_2b_4090_clean_lm_only_pilot.yaml` with `loss_jepa_weight=0` and
`loss_aux_weight=0`. The 300-step LM-only run reached about `lm=6.73` but still
generated `world of the world` loops, so JEPA/aux should not be treated as the
main collapse source.

## [2026-04-29] ingest | Workspace and memory references

Added official/near-official references for latent workspace and memory:
DeepMind Perceiver, Salesforce LAVIS/BLIP-2 Q-Former, OpenFlamingo
PerceiverResampler, and ARMT. Downloaded related PDFs under
`references/papers/memory_workspace`. The current QTRM workspace is now treated
as an in-context working-memory adapter, not a proven persistent memory system.

## [2026-04-29] implementation | Donor logits and Perceiver-style workspace

Added donor-logit residual generation through `donor_logits_scale` and
`qtrm_logits_scale`. `donor_logits_scale=1.0, qtrm_logits_scale=0.0` removes the
generation collapse and produces Qwen-quality completions, proving the previous
loop was caused by forcing a random QTRM LM head to learn the full language
distribution. Added Perceiver/OpenFlamingo-style workspace depth controls:
`workspace_layers`, `workspace_ff_mult`, and `workspace_include_latents_in_kv`.

Ran `configs/qwen35_2b_4090_donor_residual_workspace_pilot.yaml` for 120 steps.
The run stayed near donor-quality generation on both Korean and English prompts
with no `world of the world` collapse. This establishes the next baseline:
Qwen donor as base policy, QTRM as a small residual workspace/memory adapter.

## [2026-04-29] ingest | In-Place TTT

Added ByteDance Seed `In-Place-TTT` at `references/official/in-place-ttt` and
downloaded `references/papers/test_time_training/in_place_ttt_2604.06169.pdf`.
This is relevant to QTRM as a donor-side adaptive-memory axis: selected Qwen/LLaMA
MLP down-projection fast weights are updated during inference with a
next-token-prediction-aligned objective. It should be evaluated as a separate
ablation after donor-logit residual generation remains stable.

## [2026-04-29] plan | Residual ablation

Added `docs/wiki/decisions/residual-ablation-plan.md` to preserve the immediate
experiment plan. Added 1000-step residual configs for `qtrm_logits_scale=0.05`
and `0.10`. The first run is `0.05` with donor logits as the base policy,
Perceiver-style workspace depth enabled, and JEPA/aux disabled.

## [2026-04-29] eval | Memory retrieval probe

Added `data/eval/memory_retrieval_probe.jsonl`,
`src/qtrm_mm/eval/memory_retrieval.py`, and
`scripts/95_eval_memory_retrieval.py`. The script compares donor-only logits
against QTRM residual logits with and without fixed MemoryOS-style evidence, and
scores only generated completion text so answers present in the prompt are not
counted as hits.

Initial result on
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`: donor-only with
evidence `5/5`, QTRM residual with evidence `5/5`, donor-only no evidence `0/5`,
and QTRM residual no evidence `0/5`. Interpretation: evidence injection works,
but the behavior is already provided by Qwen donor logits. QTRM residual `0.10`
does not break the path, but this probe does not yet prove a distinct
QTRM-specific memory policy.

## [2026-04-29] eval | Harrier MemoryOS retrieval

Changed the MemoryOS text embedding default to
`microsoft/harrier-oss-v1-270m` and updated Harrier query encoding to use the
official SentenceTransformers `prompt_name="web_search_query"` path, with a
custom-instruction fallback for other embedders. Added
`scripts/96_build_memory_retrieval_probe_index.py` to build probe indexes that
preserve `case_id`, `evidence_role`, and `is_target` metadata.

Built `runs/eval/memory_retrieval_memoryos_harrier270m_index` from
`data/eval/memory_retrieval_distractor_probe.jsonl` and ran
`scripts/95_eval_memory_retrieval.py --evidence-mode memoryos`. Result:
retrieved target `10/10`, answer hit `10/10` across donor-only and QTRM
residual modes. This verifies the real MemoryOS retrieval path, but donor-only
still matches QTRM residual, so the next task is a memory trace or distractor
task where QTRM must improve over donor evidence copying.

## [2026-04-29] eval | Qwen3 reranked MemoryOS

Added `src/qtrm_mm/memoryos/rerank.py` with `none`, `lexical`, and
`cross_encoder` reranking backends. `cross_encoder` supports
`Qwen/Qwen3-Reranker-0.6B` through SentenceTransformers and caches the model
within the process. Updated `retrieve.py` and
`scripts/95_eval_memory_retrieval.py` to support
`retrieve_top_n -> rerank -> retrieval_top_k` evidence selection.

Smoke-tested Qwen3-Reranker-0.6B on two documents: it ranked the target archive
code document above the vault distractor (`10.75` vs `-4.6875`). Full
MemoryOS eval with Harrier top-20 and Qwen3 top-3 reranking scored retrieval
target `10/10` and answer hit `10/10` across donor-only and QTRM residual
modes. The reranker improves evidence ordering, but this probe still does not
show a QTRM-specific advantage over donor-only generation.

## [2026-04-29] eval | Hard MemoryOS reasoning probe

Added `data/eval/memory_reasoning_probe.jsonl` with temporal conflict,
authority conflict, English/Korean multi-hop, and negative missing-answer
cases. Added target recall/all-target retrieval metrics to the memory eval
helpers and JSONL output.

Built `runs/eval/memory_reasoning_harrier270m_index` and ran Harrier top-20 ->
Qwen3-Reranker-0.6B -> top-5 evidence generation. Result: donor-only `5/6`,
QTRM residual `5/6`, with all target evidence retrieved in both modes. The
single failure is the negative missing-answer case: both modes answered a seen
distractor passphrase instead of `UNKNOWN`. This isolates a real next weakness:
abstention/contradiction handling, not retrieval recall.

## [2026-04-29] plan | 100M MemoryOS and MSA

Cloned the MSA reference implementation to `references/official/msa` at
`30405b2`. Added `docs/wiki/sources/memory-sparse-attention.md` and
`docs/wiki/decisions/memoryos-100m-scale-plan.md`.

Added `src/qtrm_mm/memoryos/scale_plan.py` and
`scripts/97_plan_memoryos_scale.py` to estimate large MemoryOS builds before
ingestion. The default 100M-token estimate with 512-token chunks, 64-token
overlap, and Harrier 270M 640-dimensional embeddings is 223,215 chunks and
about 0.532 GiB of float32 embedding storage. Decision: 100M+ tokens is a
MemoryOS external-memory target, while active model context remains a much
smaller retrieved/reranked/compressed working set.

## [2026-04-29] eval | Abstention metrics before scale

Expanded `data/eval/memory_reasoning_probe.jsonl` from 6 to 9 cases with more
negative missing-answer, authority-conflict, and temporal missing-current-state
checks. Updated `src/qtrm_mm/eval/memory_retrieval.py` and
`scripts/95_eval_memory_retrieval.py` so records carry `category`,
`task_family`, and `expected_unknown`, and summaries report accuracy/retrieval
metrics by mode, category, and task family.

Added `src/qtrm_mm/memoryos/scale_benchmark.py` and
`scripts/98_benchmark_memoryos_scale.py` to write staged 1M/10M planning records
before attempting larger MemoryOS ingestion. Default output is
`runs/eval/memoryos_scale_plan_1m_10m.jsonl`. Current order: fix "retrieved but
answered wrong" on the small hard probe before treating 100M-scale architecture
as the main bottleneck.

Rebuilt `runs/eval/memory_reasoning_harrier270m_index` from the expanded
9-case probe, then reran Harrier top-20 -> Qwen3-Reranker-0.6B -> top-5
generation into `runs/eval/memory_reasoning_qwen3_rerank_32tok_expanded.jsonl`.
Result: donor-only `6/9`, QTRM residual `6/9`, all target evidence retrieved
`18/18`. By task family, conflict `8/8`, multi-hop `4/4`, abstention `0/6`.
This confirms the current blocker is not search recall; it is answer selection
when the correct response is `UNKNOWN`.

## [2026-04-29] train | Memory trace abstention fine-tune

Added `src/qtrm_mm/training/memory_trace_data.py` and
`scripts/99_build_memory_trace_data.py` to convert hard MemoryOS reasoning cases
into supervised traces. Wrote
`data/filtered/memory_abstention_traces.jsonl` with 27 rows: `target`, `all`,
and `lexical` evidence variants for each of the 9 probe cases.

Changed `JsonlTextVisionDataset` and `qtrm_smoke_loss` so rows with
`prompt`/`answer` train only answer tokens via `labels=-100` on prompt tokens.
Added `--init-checkpoint` to `qtrm_mm.training.train` so memory trace runs can
continue from the stable residual checkpoint.

Strict prompting alone did not fix the issue: with the original residual
checkpoint, donor-only and QTRM residual still failed abstention. A first
trace run at `qtrm_logits_scale=0.1` learned some UNKNOWN behavior but was too
weak; scale sweeps showed `0.5` was useful while `1.0` over-abstained.

Final run:

- Config: `configs/qwen35_2b_4090_memory_abstention_trace_s050.yaml`
- Init: `runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`
- Output: `runs/qwen35_2b_4090_memory_abstention_trace_s050/last.pt`
- Eval: `runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl`

Result on the expanded 9-case hard probe with all target evidence retrieved:
donor-only `5/9`, QTRM residual `9/9`. QTRM residual task-family accuracy:
conflict `4/4`, multi-hop `2/2`, abstention `3/3`. This fixes the current
small-scale "retrieved but answered wrong" blocker, but it is not yet proof of
general MemoryOS reasoning beyond this synthetic probe.
