#!/usr/bin/env python3
"""Plain-language preflight for official GDN2 one-body runs.

This is intentionally small and conservative.  It checks the runtime evidence
that can make an official-GDN2 experiment misleading before GPU time is spent:
pinned ptxas, legacy fallback checkpoint keys, and reported delta runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


LEGACY_DELTA_FALLBACK_MARKERS = (
    ".mixer.runtime_fallback.",
    ".mixer.impl.in_proj.",
    ".mixer.impl.gate_proj.",
    ".mixer.impl.out_proj.",
)


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dict: {path}")
    return payload


def _runtime_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    model = payload.get("model")
    if not isinstance(model, dict):
        return None
    global_core = model.get("global_core")
    if not isinstance(global_core, dict):
        return None
    runtime = global_core.get("delta_runtime")
    return runtime if isinstance(runtime, dict) else None


def _runtime_from_report(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not a dict: {path}")
    model = payload.get("model")
    if not isinstance(model, dict):
        return None
    global_core = model.get("global_core")
    if not isinstance(global_core, dict):
        return None
    runtime = global_core.get("delta_runtime")
    return runtime if isinstance(runtime, dict) else None


def _args_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args")
    return args if isinstance(args, dict) else {}


def _legacy_delta_fallback_keys(state_dict: dict[str, Any], *, limit: int = 8) -> list[str]:
    keys: list[str] = []
    for key in state_dict:
        if any(marker in str(key) for marker in LEGACY_DELTA_FALLBACK_MARKERS):
            keys.append(str(key))
            if len(keys) >= limit:
                break
    return keys


def _state_dict_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("model_state_dict", payload)
    return state if isinstance(state, dict) else {}


def _check_runtime(runtime: dict[str, Any] | None, *, label: str, blockers: list[str], warnings: list[str]) -> None:
    if runtime is None:
        warnings.append(f"{label}: delta runtime summary missing")
        return
    actual = str(runtime.get("actual_delta_runtime", ""))
    fallback_count = int(runtime.get("delta_runtime_fallback_active_count", 0) or 0)
    torch_direct_count = int(runtime.get("delta_runtime_torch_direct_count", 0) or 0)
    if actual != "official_runtime" or fallback_count != 0 or torch_direct_count != 0:
        blockers.append(
            f"{label}: not clean official runtime "
            f"(actual={actual}, fallback_count={fallback_count}, torch_direct_count={torch_direct_count})"
        )


def _run_official_gdn2_smoke(mode: str) -> dict[str, Any]:
    mode = str(mode or "none")
    if mode == "none":
        return {"mode": mode, "status": "skipped"}

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        import torch
        from qtrm_mm.mixers import OfficialGatedDeltaNet2Mixer

        mixer = OfficialGatedDeltaNet2Mixer(
            d_model=64,
            n_heads=4,
            strict=True,
            head_dim=16,
            num_v_heads=4,
            use_short_conv=False,
        )
        if mode == "import":
            return {
                "mode": mode,
                "status": "pass",
                "device": "none",
                "is_official_backend": bool(getattr(mixer, "is_official_backend", False)),
            }
        if mode == "forward_auto":
            device = "cuda" if torch.cuda.is_available() else "none"
            if device == "none":
                return {
                    "mode": mode,
                    "status": "pass",
                    "device": "none",
                    "note": "cuda unavailable; import smoke passed",
                }
        elif mode == "forward_cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for official GDN2 forward smoke")
            device = "cuda"
        elif mode == "forward_cpu":
            device = "cpu"
        else:
            raise ValueError(f"unknown official smoke mode: {mode}")

        mixer = mixer.to(device).eval()
        x = torch.randn(1, 8, 64, device=device)
        with torch.no_grad():
            y = mixer(x)
        if tuple(y.shape) != tuple(x.shape):
            raise RuntimeError(f"unexpected official GDN2 output shape: {tuple(y.shape)} != {tuple(x.shape)}")
        return {
            "mode": mode,
            "status": "pass",
            "device": device,
            "shape": list(y.shape),
            "is_official_backend": bool(getattr(mixer, "is_official_backend", False)),
        }
    except Exception as exc:
        return {
            "mode": mode,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_preflight(
    *,
    required_ptxas: str,
    triton_ptxas: str,
    checkpoint: str = "",
    report_json: str = "",
    target_backend: str = "official_gated_delta2",
    expect_decoder_latent_mode: str = "",
    official_smoke: str = "import",
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {
        "target_backend": target_backend,
        "required_ptxas": required_ptxas,
        "triton_ptxas": triton_ptxas,
        "checkpoint": checkpoint,
        "report_json": report_json,
        "expect_decoder_latent_mode": expect_decoder_latent_mode,
        "official_smoke": official_smoke,
    }

    if target_backend not in {"official_gated_delta2", "official_gdn2"}:
        warnings.append(f"target backend is not official GDN2: {target_backend}")

    if not required_ptxas:
        blockers.append("missing required ptxas contract: REQUIRED_TRITON_PTXAS_PATH is empty")
    if not triton_ptxas:
        blockers.append("missing required ptxas: TRITON_PTXAS_PATH is empty")
    if required_ptxas and triton_ptxas and required_ptxas != triton_ptxas:
        blockers.append(f"wrong ptxas: TRITON_PTXAS_PATH={triton_ptxas}, required={required_ptxas}")
    if triton_ptxas and not os.access(triton_ptxas, os.X_OK):
        blockers.append(f"missing required ptxas executable: {triton_ptxas}")

    smoke_result = _run_official_gdn2_smoke(official_smoke)
    evidence["official_gdn2_smoke"] = smoke_result
    if str(smoke_result.get("status", "")) == "fail":
        blockers.append(f"official GDN2 smoke failed: {smoke_result.get('error', '<unknown>')}")

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            blockers.append(f"checkpoint missing: {ckpt_path}")
        else:
            payload = _load_torch_checkpoint(ckpt_path)
            state_dict = _state_dict_from_payload(payload)
            legacy_keys = _legacy_delta_fallback_keys(state_dict)
            evidence["checkpoint_legacy_delta_fallback_key_examples"] = legacy_keys
            if legacy_keys:
                blockers.append(
                    "checkpoint contains legacy fallback delta-mixer keys: "
                    + ", ".join(legacy_keys[:3])
                )
            _check_runtime(_runtime_from_payload(payload), label="checkpoint", blockers=blockers, warnings=warnings)
            args = _args_from_payload(payload)
            decoder_latent_mode = str(args.get("decoder_latent_mode", ""))
            evidence["checkpoint_decoder_latent_mode"] = decoder_latent_mode
            if expect_decoder_latent_mode and decoder_latent_mode and decoder_latent_mode != expect_decoder_latent_mode:
                blockers.append(
                    f"checkpoint decoder_latent_mode={decoder_latent_mode}, expected={expect_decoder_latent_mode}"
                )
            elif expect_decoder_latent_mode and not decoder_latent_mode:
                warnings.append("checkpoint decoder_latent_mode missing")
    else:
        warnings.append("checkpoint not provided; resume cleanliness was not checked")

    if report_json:
        report_path = Path(report_json)
        if not report_path.exists():
            blockers.append(f"report missing: {report_path}")
        else:
            _check_runtime(_runtime_from_report(report_path), label="report", blockers=blockers, warnings=warnings)

    status = "pass" if not blockers else "fail"
    if status == "pass":
        plain = (
            "문과적 판정: 실행 장비와 성적표가 같은 실험을 가리킨다. "
            "official GDN2라고 부를 근거가 오염되지 않았다."
        )
    else:
        plain = (
            "문과적 판정: 지금 실행하면 성적표가 오염된다. "
            "official GDN2 실험 이름으로 다른 엔진이나 legacy checkpoint가 섞일 수 있다."
        )
    return {
        "status": status,
        "plain_language": plain,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-ptxas", default=os.environ.get("REQUIRED_TRITON_PTXAS_PATH", ""))
    parser.add_argument("--triton-ptxas", default=os.environ.get("TRITON_PTXAS_PATH", ""))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--target-backend", default="official_gated_delta2")
    parser.add_argument("--expect-decoder-latent-mode", default="")
    parser.add_argument(
        "--official-smoke",
        choices=("none", "import", "forward_auto", "forward_cuda", "forward_cpu"),
        default=os.environ.get("OFFICIAL_GDN2_PREFLIGHT_SMOKE", "import"),
    )
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_preflight(
        required_ptxas=str(args.required_ptxas),
        triton_ptxas=str(args.triton_ptxas),
        checkpoint=str(args.checkpoint),
        report_json=str(args.report_json),
        target_backend=str(args.target_backend),
        expect_decoder_latent_mode=str(args.expect_decoder_latent_mode),
        official_smoke=str(args.official_smoke),
    )
    if str(args.out):
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not bool(args.json_only):
        print(result["plain_language"])
        for blocker in result["blockers"]:
            print(f"BLOCKER: {blocker}")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
