from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "559_run_stage95_blt_partial_then_full_dgx.sh"


class Stage95BLTPartialThenFullLauncherTests(unittest.TestCase):
    def test_plan_documents_byte_partial_then_full_continuation(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "WORK_BASE": "/tmp/test_stage95_partial_then_full",
                "PARTIAL_STEPS": "12",
                "FULL_STEPS": "34",
                "REQUIRED_TRITON_PTXAS_PATH": "/usr/local/cuda-13.2/bin/ptxas",
                "TRITON_PTXAS_PATH": "/usr/local/cuda-13.2/bin/ptxas",
            }
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "plan"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Stage95 BLT partial-then-full automation", result.stdout)
        self.assertIn("tokenizer-free byte partial", result.stdout)
        self.assertIn("preflight", result.stdout)
        self.assertIn("Continue training on FULL_SAMPLE", result.stdout)
        self.assertIn("run-full", result.stdout)
        self.assertIn("launch-full", result.stdout)
        self.assertIn("Direct full action", result.stdout)
        self.assertIn("otherwise the best available run-local checkpoint", result.stdout)
        self.assertIn("--no-save-optimizer-checkpoint", result.stdout)
        self.assertIn("--strict-backends", result.stdout)
        self.assertIn("not Stage93 BPE", result.stdout)
        self.assertIn("PARTIAL_STEPS=12", result.stdout)
        self.assertIn("FULL_STEPS=34", result.stdout)
        self.assertIn("PARTIAL_BATCH_SIZE=8", result.stdout)
        self.assertIn("FULL_BATCH_SIZE=8", result.stdout)
        self.assertIn("FULL_RESUME_CKPT=<auto>", result.stdout)
        self.assertIn("RESUME_STRICT=1", result.stdout)
        self.assertIn("RESUME_LOAD_OPTIMIZER=auto", result.stdout)
        self.assertIn("20260525_STAGE95G_DGX_1B_OFFICIAL_GDN2_ONEBODY_PARTIAL_CLEAN", result.stdout)
        self.assertIn("20260525_STAGE95I_DGX_1B_OPUS_GD_OFFICIAL_GDN2_ONEBODY_FULL", result.stdout)
        self.assertIn("do not point at the legacy Stage95B/C", result.stdout)
        self.assertIn("REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas", result.stdout)
        self.assertIn("TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas", result.stdout)
        self.assertIn("OFFICIAL_GDN2_PREFLIGHT_SMOKE=forward_auto", result.stdout)
        self.assertIn("PATCH_BOUNDARY_MODE=hnet_dechunk", result.stdout)
        self.assertIn("not fixed BLT-2", result.stdout)
        self.assertIn("TEACHER_DISTILL_WEIGHT=0.0", result.stdout)
        self.assertIn("Teacher distillation is disabled by default", result.stdout)
        self.assertIn("QWEN_BOUNDARY_PRIOR_WEIGHT=0.0", result.stdout)
        self.assertIn("reader's underline", result.stdout)
        self.assertIn("START_FULL_BUILD_EARLY=0", result.stdout)
        self.assertIn("FULL_BUILD_LOCK=", result.stdout)
        self.assertIn("parquet I/O contention", result.stdout)
        self.assertIn("SOURCE_BUCKET_QUOTAS", result.stdout)
        self.assertIn("DECODER_LATENT_MODE=one_body", result.stdout)
        self.assertIn("PAST_SUCCESS_REPORT_JSON=", result.stdout)
        self.assertIn("ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP=0", result.stdout)
        self.assertIn("official GatedDeltaNet-2 is fail-fast", result.stdout)
        self.assertIn("promoted path defaults to DECODER_LATENT_MODE=one_body", result.stdout)
        self.assertIn("SELECTION_MODE=utility", result.stdout)
        self.assertIn("PARTIAL_SELECTION_MODE=first", result.stdout)
        self.assertIn("FULL_SELECTION_MODE=utility", result.stdout)
        self.assertIn("UTILITY_SCORE_JSONL=", result.stdout)
        self.assertIn("OPUS projected-utility", result.stdout)
        self.assertIn("OPUS_CHECKPOINT=<none>", result.stdout)
        self.assertIn("OPUS_PRECONDITIONER=adamw_state", result.stdout)
        self.assertIn("OPUS_PARAM_NAME_REGEX=", result.stdout)
        self.assertIn("SAVE_OPTIMIZER_CHECKPOINT=auto", result.stdout)
        self.assertIn("OPTIMIZER_CHECKPOINT_EVERY=0", result.stdout)
        self.assertIn("PARTIAL_OUT/last.pt", result.stdout)
        self.assertIn("optimizer state is not silently lost", result.stdout)
        self.assertIn("newer weights rather than rewinding progress", result.stdout)
        self.assertIn("--resume-load-optimizer", result.stdout)
        self.assertIn("generalization_dynamics_lite_probe.jsonl", result.stdout)
        self.assertIn("GD_LITE_ENABLED=1", result.stdout)
        self.assertIn("GD_LITE_REQUIRE_ACCEPT=0", result.stdout)

    def test_status_is_safe_when_no_files_exist(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "WORK_BASE": "/tmp/test_stage95_partial_then_full_missing",
                "REQUIRED_TRITON_PTXAS_PATH": "/usr/local/cuda-13.2/bin/ptxas",
                "TRITON_PTXAS_PATH": "/usr/local/cuda-13.2/bin/ptxas",
            }
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "status"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("partial_sample_ready=no", result.stdout)
        self.assertIn("full_sample_ready=no", result.stdout)
        self.assertIn("partial_checkpoint=missing", result.stdout)
        self.assertIn("partial_report=missing", result.stdout)
        self.assertIn("full_checkpoint=missing", result.stdout)
        self.assertIn("full_report=missing", result.stdout)
        self.assertIn("full_resume_ckpt=auto", result.stdout)
        self.assertIn("resume_load_optimizer=auto", result.stdout)
        self.assertIn("decoder_latent_mode=one_body", result.stdout)
        self.assertIn("past_success_report=", result.stdout)

    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_exports_src_pythonpath_for_dgx_direct_script_runs(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("PYTHONPATH", text)
        self.assertIn('${ROOT}/src', text)
        self.assertIn("scripts/557_train_blt_d_prefixlm_dataio.py", text)

    def test_launcher_preserves_batch_and_step_overrides_under_nohup(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PARTIAL_BATCH_SIZE="${PARTIAL_BATCH_SIZE}"', text)
        self.assertIn('FULL_BATCH_SIZE="${FULL_BATCH_SIZE}"', text)
        self.assertIn('PARTIAL_STEPS="${PARTIAL_STEPS}"', text)
        self.assertIn('FULL_STEPS="${FULL_STEPS}"', text)
        self.assertIn('FULL_RESUME_CKPT="${FULL_RESUME_CKPT}"', text)
        self.assertIn('RESUME_STRICT="${RESUME_STRICT}"', text)
        self.assertIn('RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER}"', text)
        self.assertIn("--strict-backends", text)
        self.assertIn('PATCH_BOUNDARY_MODE="${PATCH_BOUNDARY_MODE}"', text)
        self.assertIn('TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}"', text)
        self.assertIn('TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT}"', text)
        self.assertIn('TEACHER_DISTILL_MAX_TARGETS="${TEACHER_DISTILL_MAX_TARGETS}"', text)
        self.assertIn('QWEN_BOUNDARY_PRIOR_WEIGHT="${QWEN_BOUNDARY_PRIOR_WEIGHT}"', text)
        self.assertIn('QWEN_BOUNDARY_TOKENIZER_MODEL_ID="${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}"', text)
        self.assertIn('DECODER_LATENT_MODE="${DECODER_LATENT_MODE}"', text)
        self.assertIn('PAST_SUCCESS_REPORT_JSON="${PAST_SUCCESS_REPORT_JSON}"', text)
        self.assertIn('PAST_SUCCESS_RESTORATION_GATE_JSON="${PAST_SUCCESS_RESTORATION_GATE_JSON}"', text)
        self.assertIn('ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT="${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}"', text)
        self.assertIn(
            'ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP="${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}"',
            text,
        )
        self.assertIn('START_FULL_BUILD_EARLY="${START_FULL_BUILD_EARLY}"', text)
        self.assertIn('FULL_BUILD_LOCK="${FULL_BUILD_LOCK}"', text)
        self.assertIn('SOURCE_BUCKET_QUOTAS="${SOURCE_BUCKET_QUOTAS}"', text)
        self.assertIn('SELECTION_MODE="${SELECTION_MODE}"', text)
        self.assertIn('PARTIAL_SELECTION_MODE="${PARTIAL_SELECTION_MODE}"', text)
        self.assertIn('FULL_SELECTION_MODE="${FULL_SELECTION_MODE}"', text)
        self.assertIn('TRAIN_THINK_STEPS="${TRAIN_THINK_STEPS}"', text)
        self.assertIn('UTILITY_SCORE_JSONL="${UTILITY_SCORE_JSONL}"', text)
        self.assertIn('UTILITY_TEMPERATURE="${UTILITY_TEMPERATURE}"', text)
        self.assertIn('OPUS_CHECKPOINT="${OPUS_CHECKPOINT}"', text)
        self.assertIn('OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER}"', text)
        self.assertIn('OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM}"', text)
        self.assertIn('OPUS_PARAM_NAME_REGEX="${OPUS_PARAM_NAME_REGEX}"', text)
        self.assertIn('SAVE_OPTIMIZER_CHECKPOINT="${SAVE_OPTIMIZER_CHECKPOINT}"', text)
        self.assertIn('OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY}"', text)
        self.assertIn('GD_LITE_ENABLED="${GD_LITE_ENABLED}"', text)
        self.assertIn('GD_LITE_PROBE_JSONL="${GD_LITE_PROBE_JSONL}"', text)

    def test_launcher_disables_legacy_teacher_distillation_by_default(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("20260524_STAGE94C_LOCAL_BYTEFREE82M_LANGSAMPLE_RETRY/last_model.pt", text)
        self.assertIn('TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.0}"', text)
        self.assertIn('--teacher-checkpoint "${TEACHER_CHECKPOINT}"', text)
        self.assertIn('--teacher-distill-weight "${TEACHER_DISTILL_WEIGHT}"', text)
        self.assertIn('--teacher-distill-max-targets "${TEACHER_DISTILL_MAX_TARGETS}"', text)
        self.assertIn("ensure_teacher_distill_ready", text)
        self.assertIn("legacy fallback mixer keys", text)

    def test_launcher_exposes_qwen_tokenizer_boundary_prior_as_disabled_ablation(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('QWEN_BOUNDARY_PRIOR_WEIGHT="${QWEN_BOUNDARY_PRIOR_WEIGHT:-0.0}"', text)
        self.assertIn('QWEN_BOUNDARY_TOKENIZER_MODEL_ID="${QWEN_BOUNDARY_TOKENIZER_MODEL_ID:-Qwen/Qwen3.5-0.8B-Base}"', text)
        self.assertIn('--qwen-boundary-prior-weight "${QWEN_BOUNDARY_PRIOR_WEIGHT}"', text)
        self.assertIn('--qwen-boundary-tokenizer-model-id "${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}"', text)

    def test_launcher_delays_full_sample_build_by_default_to_avoid_io_contention(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('START_FULL_BUILD_EARLY="${START_FULL_BUILD_EARLY:-0}"', text)
        self.assertIn("full_sample_builder=delayed_until_after_partial_training", text)
        self.assertIn("START_FULL_BUILD_EARLY", text)

    def test_launcher_uses_partial_optimizer_checkpoint_for_proper_opus_full_window(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('SELECTION_MODE="${SELECTION_MODE:-utility}"', text)
        self.assertIn('PARTIAL_SELECTION_MODE="${PARTIAL_SELECTION_MODE:-first}"', text)
        self.assertIn('FULL_SELECTION_MODE="${FULL_SELECTION_MODE:-${SELECTION_MODE}}"', text)
        self.assertIn("resolve_full_opus_checkpoint", text)
        self.assertIn("full_sample_builder=waiting_for_partial_opus_checkpoint", text)
        self.assertIn('OPUS_CHECKPOINT="${effective_opus_checkpoint}"', text)
        self.assertIn("--save-optimizer-checkpoint", text)
        self.assertIn("--optimizer-checkpoint-every", text)
        self.assertIn("--no-save-optimizer-checkpoint", text)
        self.assertIn('SAVE_OPTIMIZER_CHECKPOINT}" == "auto"', text)
        self.assertIn('"${run_name}" == "partial_training"', text)
        self.assertIn('TRAINING_NAME_FOR_ARGS="${name}"', text)
        self.assertIn("full_sample_builder=already_locked", text)
        self.assertIn(') 8>"${FULL_BUILD_LOCK}"', text)
        self.assertIn("sample_ready_for_selection", text)
        self.assertIn("full_sample=present_but_not_opus_selected", text)
        self.assertIn('OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS:-256}"', text)
        self.assertIn('OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM:-2048}"', text)
        self.assertIn("clean_decoder|hnet_byte_speaker", text)

    def test_launcher_promoted_path_defaults_to_one_body_with_preflight(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('DECODER_LATENT_MODE="${DECODER_LATENT_MODE:-one_body}"', text)
        self.assertIn('--decoder-latent-mode "${DECODER_LATENT_MODE}"', text)
        self.assertIn("--past-success-report-json", text)
        self.assertIn("--past-success-restoration-gate-json", text)
        self.assertIn("--allow-missing-past-success-preflight", text)
        self.assertIn("--acknowledge-past-success-restoration-gap", text)
        self.assertNotIn("--decoder-latent-mode add", text)

    def test_launcher_requires_explicit_gb10_triton_ptxas(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("TRITON_PTXAS_PATH", text)
        self.assertIn("REQUIRED_TRITON_PTXAS_PATH", text)
        self.assertIn("20260525_STAGE95G_DGX_1B_OFFICIAL_GDN2_ONEBODY_PARTIAL_CLEAN", text)
        self.assertIn("20260525_STAGE95I_DGX_1B_OPUS_GD_OFFICIAL_GDN2_ONEBODY_FULL", text)
        self.assertNotIn("20260524_STAGE95B_DGX_1B_BLT2_PARTIAL_MODELONLY", text)
        self.assertNotIn("20260524_STAGE95C_DGX_1B_BLT2_FULL_CONTINUE_MODELONLY", text)
        self.assertIn("missing required ptxas", text)
        self.assertIn("missing required ptxas contract", text)
        self.assertIn("613_preflight_official_gdn2_contract.py", text)
        self.assertIn("--official-smoke", text)
        self.assertIn("OFFICIAL_GDN2_PREFLIGHT_SMOKE", text)
        self.assertNotIn('TRITON_PTXAS_PATH="/usr/local/cuda-13.2/bin/ptxas"', text)
        self.assertNotIn('REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-/usr/local/cuda-13.2/bin/ptxas}"', text)

    def test_training_completion_uses_report_not_intermediate_checkpoint(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('local report="${out_dir}/report.json"', text)
        self.assertIn('local optimizer_ckpt="${out_dir}/last.pt"', text)
        self.assertIn('local model_ckpt="${out_dir}/last_model.pt"', text)
        self.assertIn('! "${model_ckpt}" -nt "${optimizer_ckpt}"', text)
        self.assertIn("resume_load_optimizer_for_path", text)
        self.assertIn('${name}=complete report=${report}', text)
        self.assertIn("already_running_waiting_for_report", text)
        self.assertIn('--resume "${ckpt}"', text)
        self.assertIn("--resume-load-optimizer", text)
        self.assertIn("resuming_optimizer_state", text)
        self.assertIn("--resume-strict", text)
        self.assertIn("FULL_RESUME_CKPT", text)
        self.assertNotIn("checkpoint_ready", text)

    def test_run_performs_preflight_before_training_and_before_full_resume(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        run_body = text[text.index("run() {") :]

        self.assertIn("stage95_preflight=initial", run_body)
        self.assertIn("stage95_preflight=before_full_resume", run_body)
        self.assertIn("run_gd_lite_gate_if_enabled", run_body)
        self.assertLess(run_body.index("stage95_preflight=initial"), run_body.index("build_partial_sample_if_needed"))
        self.assertLess(run_body.index("stage95_preflight=before_full_resume"), run_body.index("full_continue_training"))

    def test_run_full_trains_ready_full_sample_without_partial_builder(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        run_full_body = text[text.index("run_full() {") : text.index("launch() {")]

        self.assertIn("full_sample=missing_or_wrong_selection", run_full_body)
        self.assertIn("stage95_preflight=full_direct", run_full_body)
        self.assertIn("full_direct_training", run_full_body)
        self.assertIn("run_gd_lite_gate_if_enabled", run_full_body)
        self.assertIn('"${FULL_SAMPLE}"', run_full_body)
        self.assertIn('"${FULL_OUT}"', run_full_body)
        self.assertNotIn("build_partial_sample_if_needed", run_full_body)
        self.assertNotIn("wait_for_full_sample", run_full_body)
        self.assertIn("launch_full", text)
        self.assertIn("run-full", text)


if __name__ == "__main__":
    unittest.main()
