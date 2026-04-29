# QTRM Reference Baseline

This project should treat these repositories and papers as read-only reference
material for the QTRM architecture reset. QTRM-specific changes should be
documented as explicit deviations from these baselines.

## Official Code References

| Area | Local path | Upstream | Commit | Purpose |
| --- | --- | --- | --- | --- |
| Qwen model family | `references/official/qwen3` | `https://github.com/QwenLM/Qwen3` | `7a2f61ffc7a2` | Qwen architecture, model usage, generation expectations |
| Qwen3.5 official repo | `references/official/qwen35` | `https://github.com/QwenLM/Qwen3.5` | `f1443092c299` | Qwen3.5 release notes, official usage, model list, serving/training guidance |
| Qwen3.5 2B config | `references/model_configs/qwen35_2b_base` | `https://huggingface.co/Qwen/Qwen3.5-2B-Base` | downloaded 2026-04-29 | Donor model card, tokenizer config, and nested `text_config`/`vision_config` |
| Qwen3.5 2B chat config | `references/model_configs/qwen35_2b_chat` | `https://huggingface.co/Qwen/Qwen3.5-2B` | downloaded 2026-04-29 | Chat/instruction serving reference and chat template |
| HF generation stack | `references/official/transformers` | `https://github.com/huggingface/transformers` | `f4fc6d013864` | `generate`, cache, tokenizer, attention mask conventions |
| PEFT/LoRA | `references/official/peft` | `https://github.com/huggingface/peft` | `9a20e07d347f` | LoRA/QLoRA adapter implementation patterns |
| DExperts | `references/official/dexperts` | `https://github.com/alisawuffles/DExperts` | `4ef198fe4cad` | Product-of-experts logit steering baseline |
| FUDGE | `references/official/fudge-controlled-generation` | `https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation` | `32c60893d9e0` | Future-discriminator logit adjustment baseline |
| GeDi | `references/official/gedi` | `https://github.com/salesforce/GeDi` | `2346c7ee99cd` | Generative-discriminator guided decoding baseline |
| Proxy-Tuning | `references/official/proxy-tuning` | `https://github.com/alisawuffles/proxy-tuning` | `5f2da2c2783b` | Logit-difference adaptation over a frozen large model |
| Side-Tuning | `references/official/side-tuning` | `https://github.com/jozhang97/side-tuning` | `dea345691fb7` | Frozen backbone plus side network reference |
| Ladder Side-Tuning | `references/official/ladder-side-tuning` | `https://github.com/ylsung/Ladder-Side-Tuning` | `1798e82e52f2` | Layer-wise side network reference |
| Training launcher | `references/official/accelerate` | `https://github.com/huggingface/accelerate` | `d1d2bba464de` | Distributed/mixed precision training patterns |
| SFT/DPO/RLHF | `references/official/trl` | `https://github.com/huggingface/trl` | `574ebe0503f5` | Post-training APIs and dataset/objective patterns |
| Gated DeltaNet | `references/official/gated-delta-net` | `https://github.com/NVlabs/GatedDeltaNet` | `b53d6d3a1612` | Official gated delta rule mixer and hybrid block reference |
| I-JEPA | `references/official/ijepa` | `https://github.com/facebookresearch/ijepa` | `52c1ae95d05f` | Joint-embedding predictive objective baseline |
| EB-JEPA | `references/official/eb_jepa` | `https://github.com/facebookresearch/eb_jepa` | `966e61e9285b` | Energy-based JEPA and planning/world-model examples |
| JEPA-WM | `references/official/jepa-wms` | `https://github.com/facebookresearch/jepa-wms` | `13cf1d9c7e47` | Action-conditioned JEPA world-model planning reference |
| LeWorldModel | `references/official/le-wm` | `https://github.com/lucas-maes/le-wm` | `bf04d3e8c375` | Latest end-to-end JEPA world-model objective and predictor reference |
| DreamerV3 | `references/official/dreamerv3` | `https://github.com/danijar/dreamerv3` | `b65cf81a6fb1` | Latent world-model transition and imagination reference |
| Tiny Recursive Models | `references/official/tiny-recursive-models` | `https://github.com/SamsungSAILMontreal/TinyRecursiveModels` | `c01103738605` | TRM recursive z_H/z_L, ACT halting, carry-detach reference |
| MergeKit | `references/official/model-merging/mergekit` | `https://github.com/arcee-ai/mergekit` | `71113608094f` | Practical LLM merge/frankenmerge/passthrough toolkit |
| TIES-Merging | `references/official/model-merging/ties-merging` | `https://github.com/prateeky2806/ties-merging` | `44e7891fc84f` | Task-vector merge interference reference |
| DELLA-Merging | `references/official/model-merging/della` | `https://github.com/declare-lab/della` | `80675156d674` | Magnitude-based sampling/pruning merge reference |
| Model Soups | `references/official/model-merging/model-soups` | `https://github.com/mlfoundations/model-soups` | `d5398f181ea5` | Checkpoint averaging / greedy soup reference |
| Branch-Train-Merge | `references/official/model-merging/btm` | `https://github.com/hadasah/btm` | `529b0f162bb6` | Branch-train-merge expert language model reference |
| Parcae | `references/official/parcae` | `https://github.com/sandyresearch/parcae` | `dee8363` | Paper-backed stable looped language model and recurrence-depth reference |

## Experimental / Speculative Code References

| Area | Local path | Upstream | Commit | Purpose |
| --- | --- | --- | --- | --- |
| OpenMythos | `references/official/openmythos` | `https://github.com/kyegomez/OpenMythos` | `8c68c1f` | Community theoretical recurrent-depth transformer sketch; not official Anthropic evidence |

## Paper References

| Area | Local PDF | Source |
| --- | --- | --- |
| Qwen | `references/papers/qwen3_technical_report_2505.09388.pdf` | `https://arxiv.org/abs/2505.09388` |
| Qwen3-VL | `references/papers/qwen3_vl_technical_report_2511.21631.pdf` | `https://arxiv.org/abs/2511.21631` |
| Qwen3-Omni | `references/papers/qwen3_omni_technical_report_2509.17765.pdf` | `https://arxiv.org/abs/2509.17765` |
| Qwen3.5 Omni | `references/papers/qwen35_omni_technical_report_2604.15804.pdf` | `https://arxiv.org/abs/2604.15804` |
| Gated DeltaNet | `references/papers/gated_delta_networks_2412.06464.pdf` | `https://arxiv.org/abs/2412.06464` |
| I-JEPA | `references/papers/ijepa_2301.08243.pdf` | `https://arxiv.org/abs/2301.08243` |
| EB-JEPA | `references/papers/eb_jepa_2602.03604.pdf` | `https://arxiv.org/abs/2602.03604` |
| JEPA world models | `references/papers/jepa_wms_physical_planning_2512.24497.pdf` | `https://arxiv.org/abs/2512.24497` |
| LeWorldModel | `references/papers/leworldmodel_2603.19312.pdf` | `https://arxiv.org/abs/2603.19312` |
| DreamerV3 | `references/papers/dreamerv3_2301.04104.pdf` | `https://arxiv.org/abs/2301.04104` |
| Tiny Recursive Models | `references/papers/tiny_recursive_models_2510.04871.pdf` | `https://arxiv.org/abs/2510.04871` |
| Model Soups | `references/papers/model_merging/model_soups_2203.05482.pdf` | `https://arxiv.org/abs/2203.05482` |
| TIES-Merging | `references/papers/model_merging/ties_merging_2306.01708.pdf` | `https://arxiv.org/abs/2306.01708` |
| DARE Merging | `references/papers/model_merging/dare_merging_2311.03099.pdf` | `https://arxiv.org/abs/2311.03099` |
| DELLA-Merging | `references/papers/model_merging/della_merging_2406.11617.pdf` | `https://arxiv.org/abs/2406.11617` |
| Branch-Train-Merge | `references/papers/model_merging/branch_train_merge_2208.03306.pdf` | `https://arxiv.org/abs/2208.03306` |
| Continual Pretraining | `references/papers/model_merging/continual_pretraining_lms_2302.03241.pdf` | `https://arxiv.org/abs/2302.03241` |
| Domain Adaptive Pretraining | `references/papers/model_merging/dont_stop_pretraining_2004.10964.pdf` | `https://arxiv.org/abs/2004.10964` |
| LLM Model Merging Survey | `references/papers/model_merging/model_merging_llm_survey_2603.09938.pdf` | `https://arxiv.org/abs/2603.09938` |
| LLM Model Merging Systematic Study | `references/papers/model_merging/systematic_llm_model_merging_2511.21437.pdf` | `https://arxiv.org/abs/2511.21437` |
| Merge-Friendly Fine-Tuning | `references/papers/model_merging/sharpness_aware_merge_finetuning_2504.14662.pdf` | `https://arxiv.org/abs/2504.14662` |
| LoRA | `references/papers/lora_2106.09685.pdf` | `https://arxiv.org/abs/2106.09685` |
| QLoRA | `references/papers/q_lora_2305.14314.pdf` | `https://arxiv.org/abs/2305.14314` |
| DExperts | `references/papers/logit_sidecar/dexperts_2021_acl_long_522.pdf` | `https://aclanthology.org/2021.acl-long.522/` |
| FUDGE | `references/papers/logit_sidecar/fudge_2104.05218.pdf` | `https://arxiv.org/abs/2104.05218` |
| GeDi | `references/papers/logit_sidecar/gedi_2009.06367.pdf` | `https://arxiv.org/abs/2009.06367` |
| Proxy-Tuning | `references/papers/logit_sidecar/proxy_tuning_2401.08565.pdf` | `https://arxiv.org/abs/2401.08565` |
| Side-Tuning | `references/papers/logit_sidecar/side_tuning_1912.13503.pdf` | `https://arxiv.org/abs/1912.13503` |
| Ladder Side-Tuning | `references/papers/logit_sidecar/ladder_side_tuning_2206.06522.pdf` | `https://arxiv.org/abs/2206.06522` |
| AdapterFusion | `references/papers/logit_sidecar/adapterfusion_2021_eacl_main_39.pdf` | `https://aclanthology.org/2021.eacl-main.39/` |
| Transformer baseline | `references/papers/attention_is_all_you_need_1706.03762.pdf` | `https://arxiv.org/abs/1706.03762` |
| HF Transformers | `references/papers/hf_transformers_library_1910.03771.pdf` | `https://arxiv.org/abs/1910.03771` |
| Parcae stable looped LMs | `references/papers/recurrent_depth/parcae_stable_looped_lm_2604.12946.pdf` | `https://arxiv.org/abs/2604.12946` |
| Looped transformers for learning algorithms | `references/papers/recurrent_depth/looped_transformers_learning_algorithms_2311.12424.pdf` | `https://arxiv.org/abs/2311.12424` |
| Reasoning with latent thoughts | `references/papers/recurrent_depth/reasoning_with_latent_thoughts_looped_transformers_2502.17416.pdf` | `https://arxiv.org/abs/2502.17416` |
| Latent CoT probing | `references/papers/recurrent_depth/latent_cot_depth_recurrent_transformer_2507.02199.pdf` | `https://arxiv.org/abs/2507.02199` |

## Web / Transcript References

| Area | Wiki page | Source |
| --- | --- | --- |
| Karpathy cognitive core and corrected 1B-vs-1.8T claim | `docs/wiki/sources/karpathy-cognitive-core.md` | `https://glasp.co/youtube/lXUZvyajciY` |
| GPT-4 model-size non-disclosure | `docs/wiki/sources/karpathy-cognitive-core.md` | `https://arxiv.org/abs/2303.08774` |
| Llama 3 high-token training note | `docs/wiki/sources/karpathy-cognitive-core.md` | `https://huggingface.co/blog/llama3` |
| TinyStories data-quality evidence | `docs/wiki/sources/karpathy-cognitive-core.md` | `https://arxiv.org/abs/2305.07759` |
| Textbook-quality data / phi-1 | `docs/wiki/sources/karpathy-cognitive-core.md` | `https://arxiv.org/abs/2306.11644` |
| Textbook-quality data / phi-1.5 | `docs/wiki/sources/karpathy-cognitive-core.md` | `https://arxiv.org/abs/2309.05463` |

## Review Rules

1. Keep Qwen/HF generation behavior as the default baseline.
2. Treat QTRM as an explicit adapter/controller/world-model extension, not as an
   unvalidated replacement for the donor generator.
3. Any Qwen3.5 donor integration must read the nested HF config (`text_config`,
   `vision_config`) and preserve the official tokenizer/chat-template behavior.
4. Any delta/recurrent token mixer must map to Gated DeltaNet/FLA/Qwen hybrid
   patterns or be marked experimental.
5. Any recursive latent workspace must map to TRM/HRM patterns or be marked
   experimental.
6. Any JEPA world-model behavior must state whether it follows I-JEPA/EB-JEPA,
   JEPA-WM, or the newer LeWorldModel contract. The current preferred QTRM
   target is LeWM-style end-to-end next-embedding prediction plus SIGReg.
7. Any transfer/merge/healing-tune plan must distinguish checkpoint merging,
   frankenmerge/passthrough, continued pretraining, SFT healing, and adapter
   merging. Do not call "healing tune" official unless mapped to a specific
   post-merge fine-tuning or continual-pretraining objective.
8. Maintain `docs/wiki` as a persistent LLM Wiki: raw sources stay in
   `references`, synthesized pages stay in the wiki, and new findings update
   related concept/component pages instead of staying only in chat.
9. Before long training runs, add regression tests for causality, tokenizer/data
   alignment, generation quality, and Qwen baseline preservation.
10. Any recurrent-depth claim must include depth-sweep validation and recurrent
    stability telemetry. Treat OpenMythos as a sketch, not as official evidence.
11. Any Karpathy-style "small cognitive core" claim must be treated as a
    data/memory-separation hypothesis, not as proof that QTRM can reach frontier
    capability by adding steps. Require donor baselines, clean trace audits,
    target-token-rank, entropy/repetition diagnostics, and ablations first.
