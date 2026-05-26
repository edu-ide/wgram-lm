# Past-Success Reproducibility Manifest

다시 돌려볼 수는 있지만, 아직 논문식 완전 재현 패키지는 아니다.

Overall status: `replayable_not_sealed`

| Run | Status | Can Replay | Fully Sealed | Selected | Oracle | Blocking Gaps |
| --- | --- | --- | --- | ---: | ---: | --- |
| Stage56_K128 | replayable_not_sealed | True | False | 0.7682291666666666 | 0.7747395833333334 | no immutable code commit/hash recorded; no materialized eval JSONL rows stored beside summary; stochastic eval enabled without deterministic replay proof; no one-command rerun script with tolerance compare recorded |
| Stage58B_K64_top3 | replayable_not_sealed | True | False | 0.93359375 | 0.9401041666666666 | no immutable code commit/hash recorded; no materialized eval JSONL rows stored beside summary; stochastic eval enabled without deterministic replay proof; no one-command rerun script with tolerance compare recorded |

## Artifact Hashes

| Run | Artifact | Exists | SHA256 | Path |
| --- | --- | --- | --- | --- |
| Stage56_K128 | summary | True | `530f75ec62a7daebdb98a9cf29906ffd0a91dd8ba0eb19a10e7d64477d766a10` | `/mnt/sdc1/tripleyoung/qtrm_eval/20260522_101900_LOCAL_STAGE56_PTRM_evalonly_K128_scale1p0/summary.json` |
| Stage56_K128 | generator_checkpoint | True | `ef171d5b9e49abff9c3ca4f2458a2798dcc67363b1b10726a192f02fdce54eb0` | `/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt` |
| Stage56_K128 | extractor_checkpoint | True | `cd8f3eea645ad04072242a4f1460d69281dcecd92bf6bb12fab48469949489fe` | `/mnt/sdc1/tripleyoung/qtrm_eval/20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103/best_token_local_register_extractor.pt` |
| Stage58B_K64_top3 | summary | True | `94162524b44fb78a8000ab37b77dbc28ebb71a56181f4ac8eeb3b8e77f461747` | `/mnt/sdc1/tripleyoung/qtrm_eval/20260522_113500_LOCAL_STAGE58B_PTRM_evalonly_K64_top3_stage54B_seed10042/summary.json` |
| Stage58B_K64_top3 | generator_checkpoint | True | `ef171d5b9e49abff9c3ca4f2458a2798dcc67363b1b10726a192f02fdce54eb0` | `/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt` |
| Stage58B_K64_top3 | extractor_checkpoint | True | `cd8f3eea645ad04072242a4f1460d69281dcecd92bf6bb12fab48469949489fe` | `/mnt/sdc1/tripleyoung/qtrm_eval/20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103/best_token_local_register_extractor.pt` |
