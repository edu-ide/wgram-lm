# Architecture Search Sources

Paper PDFs:

| Area | Local PDF | Upstream |
| --- | --- | --- |
| Transformer modification transfer | `references/papers/architecture_search/transformer_modifications_transfer_2021.emnlp-main.465.pdf` | <https://aclanthology.org/2021.emnlp-main.465/> |
| Neural architecture search survey | `references/papers/architecture_search/neural_architecture_search_survey_jmlr_2019.pdf` | <https://www.jmlr.org/papers/v20/18-598.html> |
| Network design spaces / RegNet | `references/papers/architecture_search/designing_network_design_spaces_regnet_2003.13678.pdf` | <https://arxiv.org/abs/2003.13678> |
| EfficientNet / compound scaling | `references/papers/architecture_search/efficientnet_compound_scaling_1905.11946.pdf` | <https://arxiv.org/abs/1905.11946> |
| ConvNeXt / staged modernization | `references/papers/architecture_search/convnext_modernizing_convnet_2201.03545.pdf` | <https://arxiv.org/abs/2201.03545> |
| Task arithmetic | `references/papers/architecture_search/task_arithmetic_2212.04089.pdf` | <https://arxiv.org/abs/2212.04089> |
| NAS search-phase evaluation | `references/papers/architecture_search/evaluating_nas_search_phase_1902.08142.pdf` | <https://arxiv.org/abs/1902.08142> |
| No Free Lunch theorem | `references/papers/architecture_search/no_free_lunch_optimization_wolpert_macready_1997.pdf` | <https://www.cs.utexas.edu/~shivaram/readings/b2hd-WolpertMacready1997.html> |

Related PDFs already tracked under `references/papers/model_merging`:

- `model_soups_2203.05482.pdf`
- `ties_merging_2306.01708.pdf`
- `branch_train_merge_2208.03306.pdf`

QTRM relevance:

- Transformer modification transfer is the warning reference: a good result in
  one implementation may not transfer to QTRM without controlled replication.
- NAS and design-space papers define the right process: specify axes, search
  or sweep within those axes, and evaluate under fixed budgets.
- EfficientNet and ConvNeXt are positive examples of systematic combination:
  they combine ideas through balanced scaling or staged ablation, not by adding
  every successful component at once.
- Task arithmetic and model-merging references show capability composition can
  work, but interference must be measured.
- No Free Lunch prevents treating any single architecture recipe as universally
  best independent of task, data, compute, and evaluation.
