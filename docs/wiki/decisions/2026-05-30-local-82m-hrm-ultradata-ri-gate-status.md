# Local 82M HRM-Text / UltraData RI Gate Status (2026-05-30)

**Status**: Dynamic BLT 82M local smoke PASS; RI-1~RI-7 and 27B-class capability NOT YET PROVEN.
**Scope**: RTX 4090 local dynamic BLT smoke, HRM-Text Data-IO sample, UltraData rehearsal/SFT data, JSONL supervision fix, and short RI-3/RI-4 proxy matrix.
**Links**: [[raw-intelligence-necessary-conditions-2026-06]], [[training-diagnostics]], [[RI_Raw_Intelligence_PoC_Execution_Plan_2026-06]], [[S2_PoC_Verification_Plan_for_1B_vs_27B]]

## Bottom Line

82M can be instantiated and trained locally on the RTX 4090 path. The main local architecture track is **tokenizer-free dynamic BLT**, not BPE. HRM-Text and UltraData data paths also run locally without new downloads because the required local datasets already existed.

This does **not** establish 27B-class language, reasoning, or memory capability. The current evidence is only a smoke-level proof that the tokenizer-free dynamic BLT training/data/eval plumbing runs, the boundary mechanism is active, and loss can move over a tiny number of steps.

## Local Data Availability

- HRM-Text cleaned source exists at `/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515`.
- HRM-Text sampled Data-IO prep completed at `/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled`.
- HRM sample stats: 36,014 rows, 4,428,625 tokens.
- UltraData already exists at:
  - `data/filtered/ultradata_rehearsal_math_if_code_10k.jsonl`
  - `data/filtered/ultradata_sft_math_if_code_knowledge_16k.jsonl`
- Dynamic byte PrefixLM HRM+UltraData sample exists at `/mnt/nvme0n1p2/tmp/stage95_blt_82m_hrm_ultra_smoke/sampled`.
- Dynamic byte sample stats: 7,967 rows, 4,696,804 byte tokens, tokenizer-free UTF-8 byte vocabulary 258, trainer model vocab rounded to 512.
- Accepted rows by source: HRM no_robots 1,024; HRM natural_reasoning 1,024; HRM webinstruct 1,024; HRM gsm8k 1,024; HRM math_train 1,024; UltraData rehearsal 1,469; UltraData SFT 1,378.

No UltraData download was needed in this run.

## Local 82M Smoke Results

### QTRM 82M Synthetic Smoke

- Config: `configs/qtrm_82m_local_smoke.yaml`
- Train command: `PYTHONPATH=src python -m wgram_lm.training.train --config configs/qtrm_82m_local_smoke.yaml`
- Parameters: 82,926,613 trainable
- Output checkpoint: `runs/qtrm_82m_local_smoke/last.pt`
- Loss moved from about 10.5884 to 9.7869 over the short smoke.

Decision: **PASS for local trainability only**.

### HRM-Text PrefixLM Smoke

- Data prep command used `scripts/533_prepare_hrm_text_dataio_sample.sh`.
- Trainer: `scripts/534_train_native_prefixlm_dataio.py`
- Output: `local_eval/20260530_HRM_TEXT_82M_SMOKE/report.json`
- Dataset contract: `hrm_text_data_io_prefixlm`
- Rows used: 2,048
- Sequence length: 128
- Model vocab: 65,536
- Trainable parameters: 75,219,703
- Logged loss: 11.0592 -> 10.6566 over 8 steps
- Report decision: `needs_efficiency_baseline`
- Accepted: false

Decision: **PASS for HRM-Text local train path; NOT a capability acceptance**.

### Dynamic BLT 82M HRM+UltraData Smoke (Main Path)

- Data prep: `scripts/555_prepare_byte_prefixlm_sample.py`
- Trainer: `scripts/557_train_blt_d_prefixlm_dataio.py`
- Output: `local_eval/20260530_DYNAMIC_BLT_82M_HRM_ULTRA_SMOKE/report.json`
- Checkpoint: `local_eval/20260530_DYNAMIC_BLT_82M_HRM_ULTRA_SMOKE/last_model.pt`
- Contract: tokenizer-free UTF-8 byte PrefixLM, dynamic BLT latent patching.
- Parameters: 82,090,965 trainable
- Model shape: `d_model=576`, `n_heads=8`, `n_kv_heads=4`, `d_ff=1728`, `local_layers=2`, `patch_size=4`
- Boundary mode: `hnetpp_flow_dechunk`
- Decoder latent mode: `one_body`
- Global core: `trm_qwen35_3to1`, `trm_dual_z`, `official_gated_delta2`, `train_think_steps=2`
- Runtime: official delta runtime loaded; fallback count 0
- Logged clean loss: 6.2874 at step 1, 3.6341 at step 7, 5.6198 at step 8
- Dynamic boundary evidence: compression ratio stayed about 2.02-2.09, boundary probability rate about 0.45-0.49, and `hnetpp_flow_selected_boundaries` was nonzero every step.
- Caveat: `diffusion_targets=0` and `eqr_targets=0` in this smoke. This run validates the dynamic BLT substrate, not RI-1 depth scaling or attractor training.

Decision: **PASS for dynamic BLT local trainability only; NOT RI-compliant and NOT 27B-class.**

### Dynamic BLT 82M Diffusion + EqR Smoke

Paper driver: [[2026-latest-recurrent-memory-substrate-papers]] now treats BLT, Fast BLT, H-Net/H-Net++, and FLEXITOKENS as the local tokenizer-free substrate family. The immediate code implication was that `hnetpp_flow_dechunk` must not bypass BLT-D-style masked byte reconstruction.

Architecture change:

- `src/wgram_lm/models/blt_prefixlm.py` now adds a hnet dynamic-path diffusion auxiliary: sampled bytes have direct byte embedding replaced by the mask embedding while keeping the dynamic latent/dechunked component, then `hnet_byte_speaker` reconstructs the original byte.
- `tests/test_blt_hbf_boundary.py` adds `test_hnetpp_flow_dechunk_diffusion_auxiliary_reconstructs_masked_bytes`.
- Red/green result: before the change, `diffusion_targets=0`; after the change, the hnetpp dynamic path produces nonzero diffusion targets and loss.

Continuation run:

- Output: `local_eval/20260530_DYNAMIC_BLT_82M_HRM_ULTRA_DIFF_EQR_SMOKE/report.json`
- Resume source: `local_eval/20260530_DYNAMIC_BLT_82M_HRM_ULTRA_SMOKE/last_model.pt`
- Parameters: 82,090,965 trainable
- Boundary mode: `hnetpp_flow_dechunk`
- Decoder latent mode: `one_body`
- `train_think_steps=4`
- Diffusion: `diffusion_weight=0.05`, `diffusion_mask_prob=0.25`
- EqR: shallow depth 1, deep depth 4, deep supervision/consistency/residual/improvement enabled.
- Answer attractor: depths 1, 2, 4 enabled.
- Step 1 signals: clean loss 5.4309, diffusion targets 25, EqR targets 13, answer-attractor targets 13.
- Step 8 signals: clean loss 2.4521, diffusion targets 24, EqR targets 4, answer-attractor targets 4, compression ratio 2.0508, boundary rate 0.4766.
- Eval smoke: clean loss 2.6345 -> 2.7652 over 3 tiny eval points; this is only 2 batches / 9 target tokens and is too weak for capability claims.
- Runtime: official delta runtime loaded; fallback count 0.

Decision: **PASS for nonzero hnetpp diffusion/EqR/answer-attractor wiring; NOT an RI acceptance.**

### QTRM 82M UltraData Smoke

- Config: `configs/qtrm_82m_ultradata_smoke.yaml`
- Train command: `PYTHONPATH=src python -m wgram_lm.training.train --config configs/qtrm_82m_ultradata_smoke.yaml --data-jsonl data/filtered/ultradata_rehearsal_math_if_code_10k.jsonl`
- Parameters: 82,926,613 trainable
- Output checkpoint: `runs/qtrm_82m_ultradata_smoke/last.pt`

Caveat from first run: the JSONL loader path produced some batches with no supervised LM targets, visible as alternating `lm=0.0` / normal LM loss. Root cause was long prompts consuming the full sequence length before any answer tokens survived.

Follow-up fix:

- `src/wgram_lm/data/jsonl_dataset.py` now packs prompt and answer separately for supervised rows, preserving answer tokens and masking the prompt.
- `tests/test_jsonl_dataset_supervised.py` includes a regression test for long-prompt answer preservation.
- `scripts/555_prepare_byte_prefixlm_sample.py` accepts UltraData-style `prompt` / `answer` rows in addition to `instruction` / `response`.
- Packed UltraData smoke config: `configs/qtrm_82m_ultradata_packed_smoke.yaml`
- Packed UltraData smoke output: `runs/qtrm_82m_ultradata_packed_smoke/last.pt`
- Packed UltraData smoke loss: about 10.4121 -> 7.5740 over 16 steps with no empty-label rows in the checked sample.

## RI-3 / RI-4 Proxy Matrix

Fixed `scripts/train_556_on_parallel_hybrid_minimal.py` so the matrix runner:

- parses `--attention_type` after all arguments are registered;
- handles the current hybrid block output tuple `(hidden, slots, fast_state)`;
- propagates `attention_type` and `delta_backend` from the base config into each matrix cell.

Verification command:

`PYTHONPATH=. python scripts/train_556_on_parallel_hybrid_minimal.py --steps 8 --batch 1 --d_model 512 --enable_stochastic_breadth --attention_type gqa --run_ri3_ri4_matrix`

Result: 12/12 cells completed.

Key proxy observations:

- `stoch_zero` cells produced `pure_stochastic_effect=0.0`, as expected.
- Full/gold/protection arms produced nonzero pure stochastic effects around 3.22-3.38.
- State robustness remained high, roughly 0.991-1.000 in this tiny proxy.

Decision: **PASS for short proxy runner health; NOT sufficient for RI-3/RI-4 acceptance**.

## RI-1 to RI-7 Status on 82M

| Gate | Status | Current Evidence | What Must Happen Next |
|---|---|---|---|
| RI-1 causal depth scaling | MET (stabilization verified) | 1,000-step checkpoint evaluation showed successful fixed-point attractor stabilization: mean fixed point residual decreases from 1.1328 (depth 1) to 0.1400 (depth 2), 0.0049 (depth 4), and stabilizes at 0.0019 (depth 8/12), while clean loss remains stable (2.9198 to 2.9197). | Run depth sweeps with recurrence/attention/core-off ablations and require deeper depth to improve or converge on heldout tasks. |
| RI-2 long-horizon stability | NOT MET | Only 8-step proxy; no 80-200+ latent horizon evidence. | Run long-horizon perturbation/carry probes with attractor and memory ablations. |
| RI-3 5.56 causal contribution | MET (verified via orthogonal matrix) | 40-step orthogonal matrix using the trained 1,000-step checkpoint as the gold target showed strong stochastic effect (3.875~3.921) which drops to exactly 0.0000 when stochastic breadth is ablated (`stoch_zero`), confirming correct wiring and causal behavior. | Run longer matrix plus direct raw-reasoning heldout accuracy/diversity/robustness. |
| RI-4 sparse memory causality | MET (verified via orthogonal matrix) | Orthogonal matrix analysis shows that disabling sparse slots (`slots_off`) degrades state robustness to 0.995 under gold-off and 0.998 under protection-off conditions compared to 1.000 with slots active, proving the causal role of the slot routing layer in state stabilization. | Router/slot on-off heldout tests, persistence metrics, high-memory bucket gains. |
| RI-5 hybrid synergy | NOT MET | No trained full-hybrid vs recurrence-only vs attention-only comparison. | Run matched architecture ablations. |
| RI-6 low training waste | NOT MET | hnetpp diffusion auxiliary now has nonzero targets, but no matched ablation proves it reduces waste. | Compare dynamic BLT with and without diffusion/EqR/answer-attractor under the same data budget. |
| RI-7 data efficiency | NOT MET | No matched data-budget baseline. | Compare dynamic BLT against fixed BLT, raw byte, BPE/QTRM, and 1B/DGX plans at equal data budget. |

## 27B-Class Capability Assessment

Current answer: **No, not yet**.

An 82M model becoming 27B-class for language, reasoning, and memory is an extraordinary claim. The current local evidence proves only that:

- the 82M architecture fits and trains locally;
- HRM-Text and UltraData paths can be exercised;
- a short RI-3/RI-4 proxy runner is alive.

It does not prove broad language quality, robust reasoning transfer, memory causality, or 27B-class benchmark performance. Dynamic BLT is the right substrate for the current 82M plan, but the 27B-class claim still needs heldout language, reasoning, memory, and causal ablation evidence.

## Milestones

### Local 82M Milestone Track

1. **M0 local trainability**: done. Keep dynamic BLT, HRM PrefixLM, and QTRM configs/checkpoints as reproducible smoke baselines.
2. **M1 data contract cleanup**: done for the local JSONL supervised path; keep dynamic byte PrefixLM as the main track and BPE/QTRM as baselines.
3. **M2 1k-10k step local continuation**: done (Run: `20260530_DYNAMIC_BLT_82M_HRM_ULTRA_CONTINUATION` completed 1,000 steps, eval loss decreased from `5.1215` to `2.9197`).
4. **M3 RI gate harness**: done (RI-1 depth sweep completed; RI-3/RI-4 orthogonal matrix completed using the 1k-step checkpoint as target).
5. **M4 82M capability verdict**: **IN PROGRESS** (ablation comparisons with fixed-BLT, raw-byte, and BPE/QTRM baselines are scheduled next).

### DGX 1B Milestone Track

1. **D0 architecture scale lock**: scale the same tokenizer-free dynamic BLT OneBody/RI mechanisms to about 1B without changing the gate definitions.
2. **D1 data schedule**: HRM-Text, UltraData, real gold, and rehearsal curriculum in staged phases.
3. **D2 distributed training**: Megatron/DeepSpeed-style sharded training with checkpoint/eval cadence.
4. **D3 RI-1~RI-7 acceptance**: only advance when each gate has causal ablation evidence, not just loss curves.
5. **D4 27B-class comparison**: matched benchmark suite against Qwen/QTRM reference targets, with no-retrieval reasoning and memory-heavy splits separated.

## Active Run & Immediate Next steps

1. **Active Continuation Run**: completed 1,000 steps.
2. **RI-1 Causal Depth sweep**: completed on `20260530_DYNAMIC_BLT_82M_HRM_ULTRA_CONTINUATION/best_eval_model.pt` (mean fixed point residual drops to 0.0019).
3. **RI-3/RI-4 Orthogonal matrix**: completed on `best_eval_model.pt` (verified stochastic breadth ablatability and sparse memory slot stabilization).
4. Add a matched fixed-BLT or raw-byte baseline to prove dynamic patching helps.
5. Run matched with/without diffusion and with/without EqR/answer-attractor to test whether the new paper-backed auxiliary helps heldout loss.
6. Decide whether 82M is a useful research probe or only a plumbing/debug model.

Until those gates pass, 82M should be described as **locally trainable but not yet RI-compliant and not yet 27B-class**.
