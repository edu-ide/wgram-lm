# Transfer, Merge, Healing Sources

Source repos:

| Area | Local path | Commit |
| --- | --- | --- |
| MergeKit | `references/official/model-merging/mergekit` | `71113608094f` |
| TIES-Merging | `references/official/model-merging/ties-merging` | `44e7891fc84f` |
| DELLA-Merging | `references/official/model-merging/della` | `80675156d674` |
| Model Soups | `references/official/model-merging/model-soups` | `d5398f181ea5` |
| Branch-Train-Merge | `references/official/model-merging/btm` | `529b0f162bb6` |

Paper PDFs:

| Area | Local PDF |
| --- | --- |
| Model Soups | `references/papers/model_merging/model_soups_2203.05482.pdf` |
| TIES-Merging | `references/papers/model_merging/ties_merging_2306.01708.pdf` |
| DARE | `references/papers/model_merging/dare_merging_2311.03099.pdf` |
| DELLA | `references/papers/model_merging/della_merging_2406.11617.pdf` |
| Branch-Train-Merge | `references/papers/model_merging/branch_train_merge_2208.03306.pdf` |
| Continual Pretraining | `references/papers/model_merging/continual_pretraining_lms_2302.03241.pdf` |
| Don't Stop Pretraining | `references/papers/model_merging/dont_stop_pretraining_2004.10964.pdf` |
| LLM model merging survey | `references/papers/model_merging/model_merging_llm_survey_2603.09938.pdf` |
| LLM model merging systematic study | `references/papers/model_merging/systematic_llm_model_merging_2511.21437.pdf` |
| Merge-friendly fine-tuning | `references/papers/model_merging/sharpness_aware_merge_finetuning_2504.14662.pdf` |

QTRM relevance:

- MergeKit gives the practical frankenmerge/passthrough vocabulary and tooling.
- TIES/DARE/DELLA explain task-vector interference and pruning/sampling methods.
- Model Soups explains checkpoint averaging within a common basin.
- Branch-Train-Merge covers training expert branches and merging them back.
- Continual pretraining/domain-adaptive pretraining covers "healing" as a
  training objective after architecture changes or domain shifts.
- Sharpness-aware merge fine-tuning is the closest formal reference for
  merge-friendly/healing-aware fine-tuning.
