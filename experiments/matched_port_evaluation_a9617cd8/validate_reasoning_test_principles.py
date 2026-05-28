#!/usr/bin/env python3
"""
REASONING TEST PRINCIPLES GATE — Always-On Verification Layer

This is the permanent, automatic enforcer for the user's explicit requirements
on all "추론 테스트" (pure recursive reasoning intelligence measurements):

Core principles (codified in FAIR_COMPARISON_PROTOCOL.md + SKILL.md):
1. conditions-matched (identical base checkpoint + continuation-only,
   same param/step/data/recipe) — non-negotiable for any cross-era claim.
2. Strict B only on pure_recursive_reasoning_heldout_72.jsonl
   (forced_choice or 4-way CandidatePoolSelector, no evidence, no retrieval).
3. One-Body Causal Path: answer must flow through the normal recurrent state
   (z_h etc.) → normal LM head / projection. No side renderer or external scorer.
4. Real behavioral restoration for restored mechanisms (GRAM/PTRM stochastic
   breadth, gated equation_binding readback, Answer Align Attractor depth-wise
   monotonic improvement must be actually measured when claimed).
5. Full honest ablation matrix + explicit "conditions-matched: yes|no + reason"
   tagging on every reported number.

This gate is designed to be called UNCONDITIONALLY at the start and end of
every pure_72 strict-B inference runner. It prints a canonical block and
writes a machine-readable report so that "원칙 지켜졌는지" is impossible to forget.

Usage in runners (minimal):
    from validate_reasoning_test_principles import run_principle_gate

    # at top of main()
    run_principle_gate(
        phase="start",
        benchmark="pure_recursive_reasoning_heldout_72",
        conditions_matched_declared="partial_synthetic_base",
        core_flags={"stochastic": True, "binding": True},
        ...
    )

    # at end, after accuracy printed
    run_principle_gate(phase="end", accuracy=21/72, ...)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time


@dataclass
class PrinciplesReport:
    """Structured compliance report."""
    overall_status: str = "UNKNOWN"          # PASS / PARTIAL / FAIL / EXPLORATORY
    conditions_matched: bool = False
    strict_b_used: bool = False
    benchmark: str = ""
    one_body_path: bool = True               # default optimistic
    real_behavior_restored: Dict[str, bool] = field(default_factory=dict)
    ablation_cleanliness: bool = False
    honest_labeling: bool = False
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self):
        print("\n" + "=" * 70)
        print("REASONING TEST PRINCIPLES COMPLIANCE REPORT")
        print("=" * 70)
        print(f"Overall Status: {self.overall_status}")
        print(f"Conditions-Matched: {'YES' if self.conditions_matched else 'NO'}")
        print(f"Strict B on pure_72: {'YES' if self.strict_b_used else 'NO'}")
        print(f"Benchmark: {self.benchmark or 'NOT SPECIFIED'}")

        print("\n--- Mechanism-Specific Behavioral Restoration ---")
        for mech, passed in self.real_behavior_restored.items():
            status = "PASS" if passed else "WEAK / NOT MEASURED"
            print(f"  {mech}: {status}")

        print(f"\nClean Ablations: {'YES' if self.ablation_cleanliness else 'NO / PARTIAL'}")
        print(f"Honest Labeling (I-stage, conditions-matched, etc.): {'YES' if self.honest_labeling else 'NO'}")

        if self.issues:
            print("\nIssues Found:")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.recommendations:
            print("\nRecommendations:")
            for rec in self.recommendations:
                print(f"  - {rec}")

        print("=" * 70 + "\n")


# =============================================================================
# ALWAYS-ON PRINCIPLE GATE (the part that actually gets called from runners)
# =============================================================================

CANONICAL_BENCHMARK = "data/eval/pure_recursive_reasoning_heldout_72.jsonl"
CANONICAL_STRICT_B_NAMES = ["forced_choice", "strict_b", "4-way", "candidatepoolselector", "core-state discrimination"]


def _detect_strict_b_from_context(benchmark: str, scoring: str = "", core_flags: Optional[Dict] = None) -> bool:
    if CANONICAL_BENCHMARK.split("/")[-1].replace(".jsonl", "") in benchmark.lower():
        return True
    if any(kw in (scoring or "").lower() for kw in CANONICAL_STRICT_B_NAMES):
        return True
    if core_flags and any(core_flags.get(k, False) for k in ["forced_choice", "strict_b"]):
        return True
    return False


def _conditions_matched_label(declared: Any) -> str:
    if declared is True:
        return "yes"
    if declared is False:
        return "no"
    if isinstance(declared, str):
        return declared
    return "unknown"


def run_principle_gate(
    phase: str = "start",                    # "start" or "end"
    benchmark: str = "",
    conditions_matched_declared: Any = "unknown",   # bool or "partial_synthetic_base" etc.
    strict_b_scoring: Optional[str] = None,
    core_flags: Optional[Dict[str, bool]] = None,   # e.g. {"stochastic": True, "binding": True}
    one_body_confirmed: bool = True,
    depth_behavior_measured: Optional[Dict[str, Any]] = None,
    ablation_matrix_reported: bool = False,
    honest_notes: Optional[List[str]] = None,
    accuracy: Optional[str] = None,                 # e.g. "21/72 (29.17%)"
    checkpoint: Optional[str] = None,
    extra_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    The single function that every pure_72 strict-B inference script should call
    unconditionally. Prints the canonical block the user asked for and writes
    a report artifact.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    issues: List[str] = []
    recommendations: List[str] = []

    bench_ok = CANONICAL_BENCHMARK.split("/")[-1].replace(".jsonl", "") in (benchmark or "").lower()
    if not bench_ok:
        issues.append(f"Benchmark is not the canonical pure_recursive_reasoning_heldout_72 (got: {benchmark})")

    strict_b_ok = _detect_strict_b_from_context(benchmark or "", strict_b_scoring or "", core_flags)
    if not strict_b_ok:
        issues.append("Strict B (forced_choice / 4-way no-evidence) not confirmed in call or flags")

    cm_label = _conditions_matched_label(conditions_matched_declared)
    if cm_label in ("no", "unknown", "false"):
        issues.append("conditions-matched is not explicitly 'yes'. Cross-era superiority claims are invalid.")

    # One-body
    if not one_body_confirmed:
        issues.append("One-body causal path not confirmed (answer must go through recurrent state → normal LM/projection)")

    # GRAM/PTRM restoration signals (I-stage port)
    stoch = (core_flags or {}).get("stochastic", (core_flags or {}).get("core_stochastic_breadth_enabled", False))
    bind = (core_flags or {}).get("binding", (core_flags or {}).get("core_equation_binding_enabled", False))

    restoration_notes = []
    if stoch:
        restoration_notes.append("stochastic_breadth=ON in forward path")
    if bind:
        restoration_notes.append("gated_equation_binding readback=ON (z_h injection)")

    # === Proper Porting of the three historical experiment tracks ===
    # (Workspaces + Attractor + Provenance) — now treated as first-class in the main pipeline
    ws = (core_flags or {}).get("workspaces", (core_flags or {}).get("core_thought_workspace_enabled", False))
    attr = (core_flags or {}).get("attractor", (core_flags or {}).get("core_answer_attractor_enabled", False))
    prov = (core_flags or {}).get("provenance", (core_flags or {}).get("core_provenance_register_enabled", False))

    three_tracks_proper = ws and attr and prov
    three_tracks_notes = []
    if ws:
        three_tracks_notes.append("Workspaces (importance selector)")
    if attr:
        three_tracks_notes.append("Answer Attractor (depth-wise monotonic)")
    if prov:
        three_tracks_notes.append("Provenance (Graph + WorldModel + Gated Register)")

    if three_tracks_proper:
        three_tracks_notes.append("→ PROPER PORTING ACTIVE (default in main RI-4 trainer)")

    # Attractor / depth behavior (the user's specific question about "순환 루프마다 정답률이 올라가는가")
    attractor_ok = False
    if depth_behavior_measured:
        attractor_ok = bool(depth_behavior_measured.get("improves_with_depth") or depth_behavior_measured.get("monotonic"))
        if not attractor_ok:
            issues.append("Depth-wise answer quality improvement not demonstrated (Answer Align Attractor core promise)")

    # Honest labeling
    if honest_notes:
        if any("I-stage" in n or "partial" in n.lower() or "synthetic" in n.lower() for n in honest_notes):
            recommendations.append("Honest I-stage / partial / synthetic labeling present — good.")
    else:
        issues.append("No honest_notes provided (must declare I-stage, synthetic base, conditions status etc.)")

    # Final verdict
    if "no" in cm_label or "unknown" in cm_label or len(issues) >= 3:
        verdict = "EXPLORATORY ONLY — DO NOT USE FOR CROSS-ERA '원시 추론 지능' CLAIMS"
    elif cm_label == "yes" and len(issues) == 0:
        verdict = "PASS (conditions-matched + strict B + one-body + honest)"
    else:
        verdict = "PARTIAL (see issues)"

    # === THE CANONICAL PRINT BLOCK THE USER REQUESTED ===
    print("\n" + "=" * 72)
    print("REASONING TEST PRINCIPLES GATE  (항상 실행되는 원칙 검증)")
    print("=" * 72)
    print(f"timestamp: {ts}")
    print(f"phase: {phase}")
    print(f"benchmark: {benchmark or 'NOT SPECIFIED'}")
    print(f"conditions-matched: {cm_label}")
    print(f"strict_b (pure_72, no-evidence): {'YES' if strict_b_ok else 'NO / NOT CONFIRMED'}")
    print(f"one_body_causal_path: {'YES' if one_body_confirmed else 'NO'}")
    print(f"GRAM/PTRM restoration live: {', '.join(restoration_notes) if restoration_notes else 'NOT DECLARED'}")
    print(f"Answer Attractor depth behavior measured: {'YES' if attractor_ok else 'NO / NOT PROVIDED'}")

    # Three historical tracks (proper porting status)
    if three_tracks_notes:
        print(f"Three historical tracks (Workspaces + Attractor + Provenance): {', '.join(three_tracks_notes)}")
    else:
        print("Three historical tracks (Workspaces + Attractor + Provenance): NOT DECLARED")
    print(f"ablation_matrix fully reported: {'YES' if ablation_matrix_reported else 'NO'}")
    print(f"honest labeling present: {'YES' if honest_notes else 'NO'}")
    if accuracy:
        print(f"reported_accuracy: {accuracy}")
    if checkpoint:
        print(f"checkpoint: {checkpoint}")
    print(f"\nVERDICT: {verdict}")
    if issues:
        print("\nISSUES:")
        for i in issues:
            print(f"  - {i}")
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for r in recommendations:
            print(f"  - {r}")
    print("=" * 72 + "\n")

    report = {
        "timestamp": ts,
        "phase": phase,
        "benchmark": benchmark,
        "conditions_matched": cm_label,
        "strict_b": strict_b_ok,
        "one_body": one_body_confirmed,
        "gram_ptrm_restoration": restoration_notes,
        "attractor_depth_behavior": attractor_ok,
        "three_historical_tracks": {
            "workspaces": ws,
            "attractor": attr,
            "provenance": prov,
            "all_properly_ported": three_tracks_proper,
            "notes": three_tracks_notes,
        },
        "ablation_matrix_reported": ablation_matrix_reported,
        "honest_notes": honest_notes or [],
        "issues": issues,
        "recommendations": recommendations,
        "verdict": verdict,
        "accuracy": accuracy,
        "checkpoint": checkpoint,
        "extra": extra_context or {},
    }

    # Always write artifact next to the script that called us
    try:
        out_dir = Path(__file__).parent
        report_path = out_dir / f"PRINCIPLES_GATE_{phase.upper()}_{int(time.time())}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[PRINCIPLE GATE] Report written: {report_path.name}")
    except Exception as e:
        print(f"[PRINCIPLE GATE] Could not write report artifact: {e}")

    return report


# Back-compat thin wrapper for old manual usage
class ReasoningPrinciplesValidator:
    def __init__(self):
        self.pure_72_path = Path(CANONICAL_BENCHMARK)

    def validate(self, **kwargs):
        # Delegate to the new always-on gate (end phase)
        return run_principle_gate(phase="manual", **kwargs)


if __name__ == "__main__":
    # Self-test of the gate
    run_principle_gate(
        phase="end",
        benchmark="pure_recursive_reasoning_heldout_72",
        conditions_matched_declared="partial_synthetic_base",
        strict_b_scoring="core-state forced_choice proxy",
        core_flags={"stochastic": True, "binding": True},
        one_body_confirmed=True,
        depth_behavior_measured={"improves_with_depth": False},
        ablation_matrix_reported=True,
        honest_notes=["I-stage port of QTRMRecursiveCore stochastic + gated binding (a9617cd8). Synthetic 10-step base. Not a real pretraining continuation."],
        accuracy="21/72 (29.17%)",
        checkpoint="base_for_matched_a9617cd8_port_test.pt",
    )
