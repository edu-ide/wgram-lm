#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class GateSpec:
    name: str
    target_level: str
    major_bottleneck: str
    script: str
    default_args: tuple[str, ...]
    report_name: str
    wiki_path: str
    accepted_decisions: tuple[str, ...]
    on_accept: str
    on_reject: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def profile_args(gate_name: str, profile: str) -> tuple[str, ...]:
    profile = str(profile)
    if profile == "triage" and gate_name not in {
        "qtrm_native_tiny_lm_first",
        "qtrm_native_tiny_lm_depth_ablation",
        "qtrm_native_l5_language_nonregression",
        "qtrm_native_broad_wiki_text_nonregression",
        "qtrm_native_broad_wiki_depth_ablation",
        "qtrm_native_dual_reverse_l4_baseline_compare",
        "qtrm_native_nested_dual_reverse_l4_baseline_compare",
        "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare",
    }:
        profile = "smoke"
    if gate_name == "donorless_recurrent_depth":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--train-cases",
                "32",
                "--eval-cases",
                "16",
                "--batch-size",
                "8",
                "--device",
                "cpu",
                "--log-every",
                "0",
            )
        if profile == "standard":
            return (
                "--steps",
                "1200",
                "--train-cases",
                "4096",
                "--eval-cases",
                "512",
                "--batch-size",
                "128",
                "--log-every",
                "200",
            )
    if gate_name == "ordered_list_state":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--train-cases",
                "32",
                "--eval-cases",
                "16",
                "--batch-size",
                "8",
                "--device",
                "cpu",
                "--log-every",
                "0",
            )
        if profile == "standard":
            return (
                "--steps",
                "1600",
                "--train-cases",
                "4096",
                "--eval-cases",
                "512",
                "--batch-size",
                "128",
                "--log-every",
                "100",
            )
    if gate_name == "prompt_source_position_binder":
        base = (
            "--train-jsonl",
            "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl",
            "--eval-jsonl",
            "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl",
            "--input-source",
            "token_embedding",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "16",
                "--eval-every",
                "5",
                "--log-every",
                "1",
                "--hidden-dim",
                "64",
                "--token-embedding-dim",
                "64",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "1000",
                "--batch-size",
                "64",
                "--eval-batch-size",
                "128",
                "--eval-every",
                "100",
                "--log-every",
                "100",
                "--hidden-dim",
                "512",
                "--token-embedding-dim",
                "256",
                "--transformer-layers",
                "2",
                "--transformer-heads",
                "8",
            )
    if gate_name == "prompt_source_position_binder_numeric":
        base = (
            "--train-jsonl",
            "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl",
            "--eval-jsonl",
            "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl",
            "--input-source",
            "numeric_value_embedding",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "16",
                "--eval-every",
                "5",
                "--log-every",
                "1",
                "--hidden-dim",
                "64",
                "--token-embedding-dim",
                "64",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "300",
                "--batch-size",
                "64",
                "--eval-batch-size",
                "128",
                "--eval-every",
                "50",
                "--log-every",
                "25",
                "--hidden-dim",
                "256",
                "--token-embedding-dim",
                "128",
                "--transformer-layers",
                "1",
                "--transformer-heads",
                "4",
            )
    if gate_name == "prompt_source_position_binder_token_plus_numeric":
        base = (
            "--train-jsonl",
            "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl",
            "--eval-jsonl",
            "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl",
            "--input-source",
            "token_plus_numeric_value",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "16",
                "--eval-every",
                "5",
                "--log-every",
                "1",
                "--hidden-dim",
                "64",
                "--token-embedding-dim",
                "64",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "300",
                "--batch-size",
                "64",
                "--eval-batch-size",
                "128",
                "--eval-every",
                "50",
                "--log-every",
                "25",
                "--hidden-dim",
                "256",
                "--token-embedding-dim",
                "128",
                "--transformer-layers",
                "1",
                "--transformer-heads",
                "4",
            )
    if gate_name == "qtrm_absolute_ordered_state":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "18",
                "--log-every",
                "25",
            )
    if gate_name == "qtrm_source_pointer_state":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "128",
                "--log-every",
                "25",
            )
    if gate_name == "qtrm_numeric_source_pointer_state":
        if profile == "smoke":
            return (
                "--numeric-source-features",
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
                "--min-numeric-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--numeric-source-features",
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "128",
                "--log-every",
                "25",
                "--min-numeric-value-drop",
                "0.25",
            )
    if gate_name == "qtrm_token_numeric_source_pointer_state":
        if profile == "smoke":
            return (
                "--token-numeric-value-features",
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
                "--min-token-numeric-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--token-numeric-value-features",
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "128",
                "--log-every",
                "25",
                "--min-token-numeric-value-drop",
                "0.25",
            )
    if gate_name == "qtrm_minimal_depth":
        if profile in {"smoke", "standard"}:
            return ()
    if gate_name == "renderer_canonical_lm":
        if profile in {"smoke", "standard"}:
            return ()
    if gate_name == "small_general_reasoning":
        if profile == "smoke":
            return (
                "--max-train-per-source",
                "1",
                "--max-eval-per-source",
                "1",
                "--max-train-cases",
                "2",
                "--max-eval-cases",
                "2",
                "--soft-prefix-steps",
                "2",
                "--max-new-tokens",
                "4",
                "--log-every",
                "1",
                "--no-require-family-full-hit",
                "--min-full-accuracy",
                "0.0",
            )
        if profile == "standard":
            return ()
    if gate_name == "qtrm_native_l1_mha":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "4",
                "--program-len",
                "2",
                "--modulus",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--accept-min-exact",
                "0.80",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
            )
        if profile == "standard":
            return (
                "--steps",
                "600",
                "--train-cases",
                "4096",
                "--eval-cases",
                "256",
                "--program-len",
                "2",
                "--modulus",
                "8",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "128",
                "--accept-min-exact",
                "0.90",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
            )
    if gate_name == "qtrm_native_l1_hybrid":
        base = (
            "--backbone",
            "qtrm_hybrid_3to1",
            "--n-kv-heads",
            "2",
            "--hybrid-layers",
            "4",
            "--attn-every",
            "4",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "4",
                "--program-len",
                "2",
                "--modulus",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--accept-min-exact",
                "0.80",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "600",
                "--train-cases",
                "4096",
                "--eval-cases",
                "256",
                "--program-len",
                "2",
                "--modulus",
                "8",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "128",
                "--accept-min-exact",
                "0.90",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
            )
    if gate_name == "qtrm_native_l2_curriculum_depth":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "4",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--accepted-decision",
                "accepted_l2_native_recursive_gain",
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
            )
        if profile == "standard":
            return (
                "--steps",
                "5000",
                "--train-cases",
                "16384",
                "--eval-cases",
                "512",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "96",
                "--n-heads",
                "4",
                "--d-ff",
                "192",
                "--batch-size",
                "128",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--accepted-decision",
                "accepted_l2_native_recursive_gain",
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--log-every",
                "1000",
            )
    if gate_name == "qtrm_native_tiny_lm_first":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--seq-len",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--target-level",
                "L1 native tiny LM first",
                "--accepted-decision",
                "accepted_native_tiny_lm_first",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "4",
                "--max-run-fraction",
                "0.70",
                "--max-full-vs-think0-loss-ratio",
                "1.50",
                "--max-full-vs-off-loss-ratio",
                "1.50",
            )
        if profile == "triage":
            return (
                "--steps",
                "120",
                "--seq-len",
                "48",
                "--d-model",
                "48",
                "--n-heads",
                "4",
                "--d-ff",
                "96",
                "--batch-size",
                "32",
                "--target-level",
                "L1 native tiny LM first",
                "--accepted-decision",
                "accepted_native_tiny_lm_first",
                "--max-random-loss-fraction",
                "0.95",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.35",
                "--max-full-vs-think0-loss-ratio",
                "1.35",
                "--max-full-vs-off-loss-ratio",
                "1.35",
                "--log-every",
                "60",
            )
        if profile == "standard":
            return (
                "--steps",
                "800",
                "--seq-len",
                "64",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--target-level",
                "L1 native tiny LM first",
                "--accepted-decision",
                "accepted_native_tiny_lm_first",
                "--max-random-loss-fraction",
                "0.70",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_tiny_lm_depth_ablation":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--seq-len",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--target-level",
                "L2 native tiny LM depth ablation",
                "--accepted-decision",
                "accepted_native_tiny_lm_depth_ablation",
                "--eval-depth-sweep",
                "0,1,2,4",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "4",
                "--max-run-fraction",
                "0.70",
                "--max-full-vs-think0-loss-ratio",
                "1.50",
                "--max-full-vs-off-loss-ratio",
                "1.50",
            )
        if profile == "triage":
            return (
                "--steps",
                "120",
                "--seq-len",
                "48",
                "--d-model",
                "48",
                "--n-heads",
                "4",
                "--d-ff",
                "96",
                "--batch-size",
                "32",
                "--target-level",
                "L2 native tiny LM depth ablation",
                "--accepted-decision",
                "accepted_native_tiny_lm_depth_ablation",
                "--eval-depth-sweep",
                "0,1,2,4",
                "--max-random-loss-fraction",
                "0.95",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.35",
                "--max-full-vs-think0-loss-ratio",
                "1.35",
                "--max-full-vs-off-loss-ratio",
                "1.35",
                "--max-full-vs-best-shallow-loss-ratio",
                "1.05",
                "--log-every",
                "60",
            )
        if profile == "standard":
            return (
                "--steps",
                "800",
                "--seq-len",
                "64",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--target-level",
                "L2 native tiny LM depth ablation",
                "--accepted-decision",
                "accepted_native_tiny_lm_depth_ablation",
                "--eval-depth-sweep",
                "0,1,2,4",
                "--max-random-loss-fraction",
                "0.70",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-best-shallow-loss-ratio",
                "0.98",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_l3_language_slice":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--seq-len",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--max-random-loss-fraction",
                "0.70",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.25",
            )
        if profile == "standard":
            return (
                "--steps",
                "800",
                "--seq-len",
                "64",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--max-random-loss-fraction",
                "0.70",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.25",
            )
    if gate_name == "qtrm_native_l5_language_nonregression":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--baseline-steps",
                "2",
                "--seq-len",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--target-level",
                "L5C QTRM-native language non-regression",
                "--accepted-decision",
                "accepted_l5_language_nonregression",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.30",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
            )
        if profile == "triage":
            return (
                "--steps",
                "120",
                "--baseline-steps",
                "120",
                "--text-file",
                "docs/wiki/decisions/qtrm-native-hard-lock.md",
                "--seq-len",
                "64",
                "--d-model",
                "48",
                "--n-heads",
                "4",
                "--d-ff",
                "96",
                "--batch-size",
                "32",
                "--target-level",
                "L5C QTRM-native language non-regression",
                "--accepted-decision",
                "accepted_l5_language_nonregression",
                "--max-random-loss-fraction",
                "0.95",
                "--min-unique-chars",
                "10",
                "--max-run-fraction",
                "0.30",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
                "--log-every",
                "60",
            )
        if profile == "standard":
            return (
                "--steps",
                "800",
                "--baseline-steps",
                "800",
                "--text-file",
                "docs/wiki/architecture/qtrm-native-first-roadmap.md",
                "--seq-len",
                "96",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--target-level",
                "L5C QTRM-native language non-regression",
                "--accepted-decision",
                "accepted_l5_language_nonregression",
                "--max-random-loss-fraction",
                "0.85",
                "--min-unique-chars",
                "12",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_broad_wiki_text_nonregression":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--baseline-steps",
                "2",
                "--text-glob",
                "docs/wiki/decisions/*.md",
                "--max-text-chars",
                "20000",
                "--seq-len",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--target-level",
                "L5C QTRM-native broad wiki text non-regression",
                "--accepted-decision",
                "accepted_broad_wiki_text_nonregression",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.35",
                "--max-full-vs-think0-loss-ratio",
                "1.35",
                "--max-full-vs-off-loss-ratio",
                "1.35",
                "--max-full-vs-baseline-loss-ratio",
                "1.50",
            )
        if profile == "triage":
            return (
                "--steps",
                "160",
                "--baseline-steps",
                "160",
                "--text-glob",
                "docs/wiki/decisions/*.md",
                "--text-glob",
                "docs/wiki/architecture/*.md",
                "--max-text-chars",
                "120000",
                "--seq-len",
                "96",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--target-level",
                "L5C QTRM-native broad wiki text non-regression",
                "--accepted-decision",
                "accepted_broad_wiki_text_nonregression",
                "--max-random-loss-fraction",
                "0.95",
                "--min-unique-chars",
                "12",
                "--max-run-fraction",
                "0.30",
                "--max-full-vs-think0-loss-ratio",
                "1.35",
                "--max-full-vs-off-loss-ratio",
                "1.35",
                "--max-full-vs-baseline-loss-ratio",
                "1.50",
                "--log-every",
                "80",
            )
        if profile == "standard":
            return (
                "--steps",
                "800",
                "--baseline-steps",
                "800",
                "--text-glob",
                "docs/wiki/**/*.md",
                "--max-text-chars",
                "300000",
                "--seq-len",
                "128",
                "--d-model",
                "96",
                "--n-heads",
                "4",
                "--d-ff",
                "192",
                "--batch-size",
                "64",
                "--target-level",
                "L5C QTRM-native broad wiki text non-regression",
                "--accepted-decision",
                "accepted_broad_wiki_text_nonregression",
                "--max-random-loss-fraction",
                "0.90",
                "--min-unique-chars",
                "14",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.30",
                "--max-full-vs-off-loss-ratio",
                "1.30",
                "--max-full-vs-baseline-loss-ratio",
                "1.45",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_broad_wiki_depth_ablation":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--baseline-steps",
                "2",
                "--text-glob",
                "docs/wiki/decisions/*.md",
                "--max-text-chars",
                "20000",
                "--seq-len",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--target-level",
                "L5C QTRM-native broad wiki depth ablation",
                "--accepted-decision",
                "accepted_broad_wiki_depth_ablation",
                "--eval-depth-sweep",
                "0,1,2,4",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.35",
                "--max-full-vs-think0-loss-ratio",
                "1.35",
                "--max-full-vs-off-loss-ratio",
                "1.35",
                "--max-full-vs-baseline-loss-ratio",
                "1.50",
                "--max-full-vs-best-shallow-loss-ratio",
                "1.10",
            )
        if profile == "triage":
            return (
                "--steps",
                "240",
                "--baseline-steps",
                "240",
                "--text-glob",
                "docs/wiki/decisions/*.md",
                "--text-glob",
                "docs/wiki/architecture/*.md",
                "--max-text-chars",
                "120000",
                "--seq-len",
                "96",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--target-level",
                "L5C QTRM-native broad wiki depth ablation",
                "--accepted-decision",
                "accepted_broad_wiki_depth_ablation",
                "--eval-depth-sweep",
                "0,1,2,4",
                "--max-random-loss-fraction",
                "0.95",
                "--min-unique-chars",
                "12",
                "--max-run-fraction",
                "0.30",
                "--max-full-vs-think0-loss-ratio",
                "1.35",
                "--max-full-vs-off-loss-ratio",
                "1.35",
                "--max-full-vs-baseline-loss-ratio",
                "1.50",
                "--max-full-vs-best-shallow-loss-ratio",
                "1.05",
                "--log-every",
                "80",
            )
        if profile == "standard":
            return (
                "--steps",
                "800",
                "--baseline-steps",
                "800",
                "--text-glob",
                "docs/wiki/**/*.md",
                "--max-text-chars",
                "300000",
                "--seq-len",
                "128",
                "--d-model",
                "96",
                "--n-heads",
                "4",
                "--d-ff",
                "192",
                "--batch-size",
                "64",
                "--target-level",
                "L5C QTRM-native broad wiki depth ablation",
                "--accepted-decision",
                "accepted_broad_wiki_depth_ablation",
                "--eval-depth-sweep",
                "0,1,2,4",
                "--max-random-loss-fraction",
                "0.90",
                "--min-unique-chars",
                "14",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.30",
                "--max-full-vs-off-loss-ratio",
                "1.30",
                "--max-full-vs-baseline-loss-ratio",
                "1.45",
                "--max-full-vs-best-shallow-loss-ratio",
                "0.98",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_l5d_official_fla_runtime":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--train-cases",
                "18",
                "--eval-cases",
                "6",
                "--task-families",
                "modchain,revchain,modchain,revchain,checksum",
                "--eval-task-families",
                "modchain,revchain,checksum",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "32",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "64",
                "--batch-size",
                "6",
                "--device",
                "cuda",
                "--backbone",
                "qtrm_hybrid_3to1",
                "--hybrid-layers",
                "4",
                "--attn-every",
                "4",
                "--delta-backend",
                "fla_gated_delta",
                "--strict-backends",
                "--delta-head-dim",
                "8",
                "--delta-num-v-heads",
                "4",
                "--delta-expand-v",
                "1.0",
                "--delta-mode",
                "chunk",
                "--delta-conv-size",
                "4",
                "--delta-norm-eps",
                "1e-6",
                "--log-every",
                "0",
                "--accept-min-exact",
                "0.0",
                "--accept-min-depth-gain",
                "-1.0",
                "--accept-min-ablation-drop",
                "-1.0",
                "--accept-min-family-exact",
                "0.0",
                "--accepted-decision",
                "accepted_l5d_official_fla_runtime",
            )
        if profile == "standard":
            return (
                "--steps",
                "200",
                "--train-cases",
                "1024",
                "--eval-cases",
                "96",
                "--task-families",
                "modchain,revchain,modchain,revchain,checksum",
                "--eval-task-families",
                "modchain,revchain,checksum",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "128",
                "--batch-size",
                "32",
                "--device",
                "cuda",
                "--backbone",
                "qtrm_hybrid_3to1",
                "--hybrid-layers",
                "4",
                "--attn-every",
                "4",
                "--delta-backend",
                "fla_gated_delta",
                "--strict-backends",
                "--delta-head-dim",
                "16",
                "--delta-num-v-heads",
                "4",
                "--delta-expand-v",
                "1.0",
                "--delta-mode",
                "chunk",
                "--delta-conv-size",
                "4",
                "--delta-norm-eps",
                "1e-6",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--log-every",
                "50",
                "--accept-min-exact",
                "0.0",
                "--accept-min-depth-gain",
                "-1.0",
                "--accept-min-ablation-drop",
                "-1.0",
                "--accept-min-family-exact",
                "0.0",
                "--accepted-decision",
                "accepted_l5d_official_fla_runtime",
            )
    if gate_name == "qtrm_native_l5d_placement_seed_stability":
        if profile == "smoke":
            return (
                "--profile",
                "smoke",
                "--candidates",
                "mha_etd,official_fla_think",
                "--target-candidate",
                "official_fla_think",
                "--seeds",
                "337",
                "--min-seeds",
                "1",
                "--min-promoted-rate",
                "1.0",
                "--min-delta-vs-mha",
                "0.0",
            )
        if profile == "standard":
            return (
                "--profile",
                "short",
                "--candidates",
                "mha_etd,official_fla_think",
                "--target-candidate",
                "official_fla_think",
                "--seeds",
                "337",
                "338",
                "339",
                "--min-seeds",
                "3",
                "--min-promoted-rate",
                "1.0",
                "--min-delta-vs-mha",
                "0.0",
            )
    if gate_name == "qtrm_native_l5d_placement_language_nonregression":
        base = (
            "--backbone",
            "qtrm_hybrid_3to1",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "qtrm_hybrid_3to1",
            "--decode-backbone",
            "mha_etd",
            "--hybrid-layers",
            "4",
            "--attn-every",
            "4",
            "--delta-backend",
            "fla_gated_delta",
            "--strict-backends",
            "--delta-head-dim",
            "16",
            "--delta-num-v-heads",
            "4",
            "--delta-expand-v",
            "1.0",
            "--delta-mode",
            "chunk",
            "--delta-conv-size",
            "4",
            "--delta-norm-eps",
            "1e-6",
            "--target-level",
            "L5D QTRM-native placement language non-regression",
            "--accepted-decision",
            "accepted_l5d_placement_language_nonregression",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "2",
                "--baseline-steps",
                "2",
                "--seq-len",
                "32",
                "--d-model",
                "32",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "64",
                "--batch-size",
                "4",
                "--device",
                "cuda",
                "--log-every",
                "0",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.30",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "800",
                "--baseline-steps",
                "800",
                "--text-file",
                "docs/wiki/architecture/qtrm-native-first-roadmap.md",
                "--seq-len",
                "96",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--max-random-loss-fraction",
                "0.85",
                "--min-unique-chars",
                "12",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_l5d_mamba3_placement_language_nonregression":
        base = (
            "--backbone",
            "mamba3",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "mamba3",
            "--decode-backbone",
            "mha_etd",
            "--strict-backends",
            "--target-level",
            "L5D QTRM-native Mamba3 placement language non-regression",
            "--accepted-decision",
            "accepted_l5d_mamba3_placement_language_nonregression",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "2",
                "--baseline-steps",
                "2",
                "--seq-len",
                "32",
                "--d-model",
                "32",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "64",
                "--batch-size",
                "4",
                "--device",
                "cuda",
                "--log-every",
                "0",
                "--max-random-loss-fraction",
                "1.10",
                "--min-unique-chars",
                "8",
                "--max-run-fraction",
                "0.30",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "800",
                "--baseline-steps",
                "800",
                "--text-file",
                "docs/wiki/architecture/qtrm-native-first-roadmap.md",
                "--seq-len",
                "96",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--max-random-loss-fraction",
                "0.85",
                "--min-unique-chars",
                "12",
                "--max-run-fraction",
                "0.25",
                "--max-full-vs-think0-loss-ratio",
                "1.25",
                "--max-full-vs-off-loss-ratio",
                "1.25",
                "--max-full-vs-baseline-loss-ratio",
                "1.35",
                "--log-every",
                "100",
            )
    if gate_name == "qtrm_native_l5d_placement_scaled_reasoning":
        base = (
            "--task-families",
            "modchain,revchain,modchain,revchain,checksum",
            "--eval-task-families",
            "modchain,revchain,checksum",
            "--program-len",
            "4",
            "--modulus",
            "32",
            "--backbone",
            "qtrm_hybrid_3to1",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "qtrm_hybrid_3to1",
            "--decode-backbone",
            "mha_etd",
            "--hybrid-layers",
            "4",
            "--attn-every",
            "4",
            "--delta-backend",
            "fla_gated_delta",
            "--strict-backends",
            "--delta-head-dim",
            "16",
            "--delta-num-v-heads",
            "4",
            "--delta-expand-v",
            "1.0",
            "--delta-mode",
            "chunk",
            "--delta-conv-size",
            "4",
            "--delta-norm-eps",
            "1e-6",
            "--depth-intermediate-loss-weight",
            "0.5",
            "--active-len-curriculum",
            "--accepted-decision",
            "accepted_l5d_placement_scaled_reasoning",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "2",
                "--train-cases",
                "18",
                "--eval-cases",
                "6",
                "--d-model",
                "32",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "64",
                "--batch-size",
                "6",
                "--device",
                "cuda",
                "--log-every",
                "0",
                "--accept-min-exact",
                "0.0",
                "--accept-min-depth-gain",
                "0.001",
                "--accept-min-ablation-drop",
                "0.001",
                "--accept-min-family-exact",
                "0.0",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "1200",
                "--train-cases",
                "4096",
                "--eval-cases",
                "384",
                "--d-model",
                "96",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "192",
                "--batch-size",
                "64",
                "--device",
                "cuda",
                "--log-every",
                "200",
                "--accept-min-exact",
                "0.06",
                "--accept-min-depth-gain",
                "0.01",
                "--accept-min-ablation-drop",
                "0.005",
                "--accept-min-family-exact",
                "0.015",
            )
    if gate_name == "qtrm_native_l5d_mamba3_placement_scaled_reasoning":
        base = (
            "--task-families",
            "modchain,revchain,modchain,revchain,checksum",
            "--eval-task-families",
            "modchain,revchain,checksum",
            "--program-len",
            "4",
            "--modulus",
            "32",
            "--backbone",
            "mamba3",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "mamba3",
            "--decode-backbone",
            "mha_etd",
            "--strict-backends",
            "--target-level",
            "L5D QTRM-native Mamba3 placement scaled reasoning",
            "--depth-intermediate-loss-weight",
            "0.5",
            "--active-len-curriculum",
            "--accepted-decision",
            "accepted_l5d_mamba3_placement_scaled_reasoning",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "2",
                "--train-cases",
                "18",
                "--eval-cases",
                "6",
                "--d-model",
                "32",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "64",
                "--batch-size",
                "6",
                "--device",
                "cuda",
                "--log-every",
                "0",
                "--accept-min-exact",
                "0.0",
                "--accept-min-depth-gain",
                "0.001",
                "--accept-min-ablation-drop",
                "0.001",
                "--accept-min-family-exact",
                "0.0",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "1200",
                "--train-cases",
                "4096",
                "--eval-cases",
                "384",
                "--d-model",
                "64",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "128",
                "--batch-size",
                "64",
                "--device",
                "cuda",
                "--log-every",
                "200",
                "--accept-min-exact",
                "0.06",
                "--accept-min-depth-gain",
                "0.01",
                "--accept-min-ablation-drop",
                "0.005",
                "--accept-min-family-exact",
                "0.015",
            )
    if gate_name == "qtrm_native_dual_path_reverse_length_gate":
        if profile == "smoke":
            return (
                "--profile",
                "smoke",
                "--lengths",
                "4",
                "--candidates",
                "official,dual_path_reverse",
                "--device",
                "cpu",
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "8",
                "--batch-size",
                "4",
                "--d-model",
                "16",
                "--d-ff",
                "32",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--log-every",
                "0",
            )
        if profile == "standard":
            return (
                "--profile",
                "short",
                "--lengths",
                "4,6,8",
                "--candidates",
                "official,dual_path_reverse",
                "--device",
                "cuda",
            )
    if gate_name == "qtrm_native_l4_mixed_text_reasoning":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "4",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
            )
        if profile == "standard":
            return (
                "--steps",
                "8000",
                "--train-cases",
                "16384",
                "--eval-cases",
                "512",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "128",
                "--n-heads",
                "8",
                "--d-ff",
                "256",
                "--batch-size",
                "128",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--log-every",
                "1000",
            )
    if gate_name in {
        "qtrm_native_dual_reverse_l4_baseline_compare",
        "qtrm_native_nested_dual_reverse_l4_baseline_compare",
        "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare",
        "qtrm_native_fast_slow_latent_update_l4_repair",
    }:
        nested_update = gate_name == "qtrm_native_nested_dual_reverse_l4_baseline_compare"
        nested_split_mixer_update = (
            gate_name
            == "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare"
        )
        fast_slow_update = gate_name == "qtrm_native_fast_slow_latent_update_l4_repair"
        think_structure = (
            "trm_dual_z_nested_official_schedule_split_mixer_3to1"
            if nested_split_mixer_update or fast_slow_update
            else (
                "trm_dual_z_nested_reversed_mha_etd"
                if nested_update
                else "trm_dual_z_reversed_mha_etd"
            )
        )
        accepted_decision = (
            "accepted_fast_slow_latent_update_l4_repair"
            if fast_slow_update
            else (
                "accepted_nested_official_schedule_split_mixer_3to1_l4_baseline_compare"
                if nested_split_mixer_update
                else (
                    "accepted_nested_dual_reverse_l4_baseline_compare"
                    if nested_update
                    else "accepted_dual_reverse_l4_baseline_compare"
                )
            )
        )
        smoke_decision = (
            "smoke_passed_fast_slow_latent_update_l4_repair"
            if fast_slow_update
            else (
                "smoke_passed_nested_official_schedule_split_mixer_3to1_l4_baseline_compare"
                if nested_split_mixer_update
                else (
                    "smoke_passed_nested_dual_reverse_l4_baseline_compare"
                    if nested_update
                    else "smoke_passed_dual_reverse_l4_baseline_compare"
                )
            )
        )
        z_l_counterfactual_weight = (
            "0.10"
            if nested_update or nested_split_mixer_update or fast_slow_update
            else "0.05"
        )
        z_l_counterfactual_margin = (
            "0.15"
            if nested_update or nested_split_mixer_update or fast_slow_update
            else "0.10"
        )
        trm_l_cycles = "6" if nested_split_mixer_update or fast_slow_update else "2"
        official_schedule_depth_args = (
            (
                "--train-think-steps",
                "3",
                "--eval-think-steps",
                "3",
            )
            if nested_split_mixer_update or fast_slow_update
            else ()
        )
        fast_slow_args = (
            (
                "--fast-slow-latent-loss-weight",
                "0.15",
                "--fast-slow-latent-every",
                "1",
                "--fast-slow-z-l-margin",
                "0.20",
                "--fast-slow-z-h-margin",
                "0.05",
                "--fast-slow-z-l-weight",
                "2.0",
                "--fast-slow-z-h-weight",
                "0.5",
            )
            if fast_slow_update
            else ()
        )
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "4",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--backbone",
                "mha_etd",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "mha_etd",
                "--decode-backbone",
                "mha_etd",
                "--think-structure",
                think_structure,
                "--trm-l-cycles",
                trm_l_cycles,
                *official_schedule_depth_args,
                "--halt-pooling",
                "last",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--depth-intermediate-loss-weight",
                "0.5",
                *fast_slow_args,
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.0",
                "--accept-min-depth-gain",
                "-1.0",
                "--accept-min-ablation-drop",
                "-1.0",
                "--accepted-decision",
                smoke_decision,
            )
        if profile == "triage":
            return (
                "--steps",
                "240",
                "--train-cases",
                "2048",
                "--eval-cases",
                "96",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "128",
                "--n-heads",
                "8",
                "--d-ff",
                "256",
                "--backbone",
                "mha_etd",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "mha_etd",
                "--decode-backbone",
                "mha_etd",
                "--think-structure",
                think_structure,
                "--trm-l-cycles",
                trm_l_cycles,
                *official_schedule_depth_args,
                "--halt-pooling",
                "last",
                "--resume-from",
                "local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/last.pt",
                "--resume-allow-missing",
                "--train-only-resume-missing-params",
                "--batch-size",
                "64",
                "--lr",
                "5e-4",
                "--weight-decay",
                "0.0",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--z-l-counterfactual-loss-weight",
                z_l_counterfactual_weight,
                "--z-l-counterfactual-margin",
                z_l_counterfactual_margin,
                "--z-l-counterfactual-every",
                "1",
                *fast_slow_args,
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.665",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--accepted-decision",
                accepted_decision,
                "--log-every",
                "80",
            )
        if profile == "standard":
            return (
                "--steps",
                "1000",
                "--train-cases",
                "4096",
                "--eval-cases",
                "256",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "128",
                "--n-heads",
                "8",
                "--d-ff",
                "256",
                "--backbone",
                "mha_etd",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "mha_etd",
                "--decode-backbone",
                "mha_etd",
                "--think-structure",
                think_structure,
                "--trm-l-cycles",
                trm_l_cycles,
                *official_schedule_depth_args,
                "--halt-pooling",
                "last",
                "--resume-from",
                "local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/last.pt",
                "--resume-allow-missing",
                "--train-only-resume-missing-params",
                "--batch-size",
                "64",
                "--lr",
                "5e-4",
                "--weight-decay",
                "0.0",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--z-l-counterfactual-loss-weight",
                z_l_counterfactual_weight,
                "--z-l-counterfactual-margin",
                z_l_counterfactual_margin,
                "--z-l-counterfactual-every",
                "1",
                *fast_slow_args,
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.665",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--accepted-decision",
                accepted_decision,
                "--log-every",
                "1000",
            )
    if gate_name == "qtrm_native_l5_multifamily":
        if profile == "smoke":
            return (
                "--steps",
                "2",
                "--train-cases",
                "18",
                "--eval-cases",
                "6",
                "--task-families",
                "modchain,revchain,modchain,revchain,checksum",
                "--eval-task-families",
                "modchain,revchain,checksum",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "6",
                "--device",
                "cpu",
                "--log-every",
                "0",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--accept-min-family-exact",
                "0.30",
                "--accepted-decision",
                "accepted_l5_multifamily",
            )
        if profile == "standard":
            return (
                "--steps",
                "12000",
                "--train-cases",
                "24576",
                "--eval-cases",
                "768",
                "--task-families",
                "modchain,revchain,modchain,revchain,checksum",
                "--eval-task-families",
                "modchain,revchain,checksum",
                "--program-len",
                "4",
                "--modulus",
                "32",
                "--d-model",
                "128",
                "--n-heads",
                "8",
                "--d-ff",
                "256",
                "--batch-size",
                "128",
                "--depth-intermediate-loss-weight",
                "0.5",
                "--active-len-curriculum",
                "--accept-min-exact",
                "0.60",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--accept-min-family-exact",
                "0.40",
                "--accepted-decision",
                "accepted_l5_multifamily",
                "--log-every",
                "1000",
            )
    raise ValueError(f"unsupported profile for {gate_name}: {profile}")


def gate_specs(profile: str) -> dict[str, GateSpec]:
    return {
        "donorless_recurrent_depth": GateSpec(
            name="donorless_recurrent_depth",
            target_level="L1 scaffold",
            major_bottleneck="reset prerequisite for bottleneck 2 recursive depth scaling",
            script="scripts/260_train_donorless_recurrent_depth_probe.py",
            default_args=profile_args("donorless_recurrent_depth", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/donorless-recurrent-depth-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "open qtrm_minimal_depth gate: port the same recurrence pressure "
                "into QTRM and require donor-only < QTRM plus core_off < QTRM"
            ),
            on_reject=(
                "stop integrated donor-QTRM tuning; redesign the donorless "
                "recurrence/task until an isolated depth gain is accepted"
            ),
        ),
        "ordered_list_state": GateSpec(
            name="ordered_list_state",
            target_level="L1 scaffold",
            major_bottleneck=(
                "ordered select/map/copy recurrent state before canonical LM "
                "renderer integration"
            ),
            script="scripts/315_train_ordered_list_state_probe.py",
            default_args=profile_args("ordered_list_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/ordered-list-state-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "port the ordered-slot transition into QTRM so final LM logits "
                "depend on the ordered recurrent state; require source/state-off "
                "ablation drop before L3"
            ),
            on_reject=(
                "do not tune answer bridges; redesign the ordered recurrent "
                "state until filter->double composition is accepted in isolation"
            ),
        ),
        "prompt_source_position_binder": GateSpec(
            name="prompt_source_position_binder",
            target_level="L1 scaffold",
            major_bottleneck=(
                "prompt-token numeric source-position binding before QTRM "
                "recurrent pointer-state integration"
            ),
            script="scripts/320_train_prompt_source_position_binder_probe.py",
            default_args=profile_args("prompt_source_position_binder", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/prompt-source-position-binder-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "port the source-position binder into QTRM and require "
                "numeric-feature/binder-off ablation drop"
            ),
            on_reject=(
                "add numeric-aware input representation or digit/value features "
                "before retrying recurrent pointer-state QTRM L2"
            ),
        ),
        "prompt_source_position_binder_numeric": GateSpec(
            name="prompt_source_position_binder_numeric",
            target_level="L1 scaffold",
            major_bottleneck=(
                "numeric-aware source-position binding before QTRM recurrent "
                "pointer-state integration"
            ),
            script="scripts/320_train_prompt_source_position_binder_probe.py",
            default_args=profile_args("prompt_source_position_binder_numeric", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/prompt-source-position-binder-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "port numeric-aware source-slot embeddings into QTRM and require "
                "numeric-feature-off plus core-off ablation drops"
            ),
            on_reject=(
                "redesign numeric-aware input representation before retrying "
                "QTRM source-pointer L2"
            ),
        ),
        "prompt_source_position_binder_token_plus_numeric": GateSpec(
            name="prompt_source_position_binder_token_plus_numeric",
            target_level="L1 scaffold",
            major_bottleneck=(
                "canonical token-path value-aware source-position binding before "
                "QTRM recurrent pointer-state integration"
            ),
            script="scripts/320_train_prompt_source_position_binder_probe.py",
            default_args=profile_args(
                "prompt_source_position_binder_token_plus_numeric",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/prompt-source-position-binder-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "replace side-channel numeric source features with token-path "
                "value-aware embeddings in QTRM source-pointer L2"
            ),
            on_reject=(
                "canonical token-path numeric binding is still insufficient; "
                "improve token-aligned value representation before QTRM L2"
            ),
        ),
        "qtrm_absolute_ordered_state": GateSpec(
            name="qtrm_absolute_ordered_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "port accepted ordered-list recurrent state into QTRM primitive "
                "role/value state with absolute value targets"
            ),
            script="scripts/316_run_qtrm_absolute_ordered_state_gate.py",
            default_args=profile_args("qtrm_absolute_ordered_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-absolute-ordered-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open canonical LM renderer gate: require final LM logits or "
                "generation to improve and drop under ordered-state-off ablation"
            ),
            on_reject=(
                "do not add answer bridges; fix QTRM ordered state learning or "
                "port the donorless ordered-slot transition more directly"
            ),
        ),
        "qtrm_source_pointer_state": GateSpec(
            name="qtrm_source_pointer_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "replace brittle absolute value classes with source-position "
                "pointer state on the corrected list combination split"
            ),
            script="scripts/319_run_qtrm_source_pointer_state_gate.py",
            default_args=profile_args("qtrm_source_pointer_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-source-pointer-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open copy/edit LM renderer gate: require final autoregressive "
                "text to depend causally on source-pointer state"
            ),
            on_reject=(
                "do not add renderer complexity; fix prompt-position binding "
                "or recurrent pointer updates before claiming L2 state progress"
            ),
        ),
        "qtrm_numeric_source_pointer_state": GateSpec(
            name="qtrm_numeric_source_pointer_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "numeric-aware prompt source binding must become causal inside "
                "QTRM source-position recurrent state"
            ),
            script="scripts/319_run_qtrm_source_pointer_state_gate.py",
            default_args=profile_args("qtrm_numeric_source_pointer_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-source-pointer-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open copy/edit LM renderer gate: require autoregressive text "
                "to depend on numeric-aware source-pointer state, primitive-off, "
                "and numeric-feature-off ablations"
            ),
            on_reject=(
                "numeric-aware L1 does not yet route causally through QTRM; "
                "inspect projector/core binding and recurrent pointer update"
            ),
        ),
        "qtrm_token_numeric_source_pointer_state": GateSpec(
            name="qtrm_token_numeric_source_pointer_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "token-path value-aware numeric binding must become causal "
                "inside QTRM source-position recurrent state"
            ),
            script="scripts/319_run_qtrm_source_pointer_state_gate.py",
            default_args=profile_args(
                "qtrm_token_numeric_source_pointer_state",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-source-pointer-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open copy/edit LM renderer gate using canonical token-path "
                "numeric source-pointer state"
            ),
            on_reject=(
                "token-path L1 binding has not yet become QTRM recurrent L2; "
                "inspect token numeric embedding load/training and pointer update"
            ),
        ),
        "qtrm_minimal_depth": GateSpec(
            name="qtrm_minimal_depth",
            target_level="L2 local gate",
            major_bottleneck="minimal QTRM depth scaffold after donorless recurrence L1",
            script="scripts/301_build_qtrm_minimal_depth_gate.py",
            default_args=profile_args("qtrm_minimal_depth", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-minimal-depth-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open renderer/canonical-LLM-path gate; primitive executor success "
                "is not yet normal autoregressive text generation"
            ),
            on_reject=(
                "redesign QTRM minimal depth path before renderer, memory, or "
                "metacognition work"
            ),
        ),
        "renderer_canonical_lm": GateSpec(
            name="renderer_canonical_lm",
            target_level="L3 candidate",
            major_bottleneck="bottleneck 4 latent-state to autoregressive text renderer",
            script="scripts/302_build_renderer_canonical_lm_gate.py",
            default_args=profile_args("renderer_canonical_lm", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/renderer-canonical-lm-gate.md",
            accepted_decisions=("accepted_l3_candidate", "accepted"),
            on_accept="promote renderer candidate to broader held-out generation gate",
            on_reject=(
                "renderer remains bottleneck; design a donor-compatible text "
                "renderer before memory/metacognition expansion"
            ),
        ),
        "small_general_reasoning": GateSpec(
            name="small_general_reasoning",
            target_level="L2 local gate / L3 candidate",
            major_bottleneck=(
                "recursive core + state codec + autoregressive final answer path "
                "must beat donor-only on a mixed small reasoning gate"
            ),
            script="scripts/308_run_small_general_reasoning_gate.py",
            default_args=profile_args("small_general_reasoning", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/small-general-reasoning-gate.md",
            accepted_decisions=(
                "accepted_l3_candidate_small_general_reasoning",
                "accepted",
            ),
            on_accept=(
                "promote to broader universal-LLM causal-path gate with more "
                "families, donor-preservation checks, and harder ablations"
            ),
            on_reject=(
                "inspect whether failure is donor-only tie, core_off tie, "
                "state_off tie, or family coverage; fix that axis before "
                "claiming general LLM progress"
            ),
        ),
        "qtrm_native_l1_mha": GateSpec(
            name="qtrm_native_l1_mha",
            target_level="L1 native LM path",
            major_bottleneck=(
                "donorless token->native backbone->mandatory recurrent core->LM "
                "logits viability with a plain MHA ETD backbone"
            ),
            script="scripts/335_train_qtrm_native_etd_probe.py",
            default_args=profile_args("qtrm_native_l1_mha", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l1_native_etd", "accepted"),
            on_accept=(
                "compare the Qwen3.5-style hybrid backbone under the same L1 "
                "gate, then promote only if strict ablations remain causal"
            ),
            on_reject=(
                "fix native answer/EOS decoding or core-to-logits path before "
                "adding memory, donor, or larger language data"
            ),
        ),
        "qtrm_native_l1_hybrid": GateSpec(
            name="qtrm_native_l1_hybrid",
            target_level="L1 native LM path",
            major_bottleneck=(
                "Qwen3.5-style Delta/Delta/Delta/Attention hybrid must preserve "
                "the same native recurrent causal path as MHA ETD"
            ),
            script="scripts/335_train_qtrm_native_etd_probe.py",
            default_args=profile_args("qtrm_native_l1_hybrid", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l1_native_etd", "accepted"),
            on_accept=(
                "treat the hybrid as a candidate backbone; keep MHA as baseline "
                "until speed and seed-stability comparisons are complete"
            ),
            on_reject=(
                "do not promote DeltaNet/MSA complexity yet; keep MHA ETD as the "
                "canonical baseline and inspect reset/op-zero ablations"
            ),
        ),
        "qtrm_native_l2_curriculum_depth": GateSpec(
            name="qtrm_native_l2_curriculum_depth",
            target_level="L2 native recursive gain",
            major_bottleneck=(
                "harder program_len=4/mod32 reasoning must require recurrent "
                "depth through the ordinary LM-generation answer path"
            ),
            script="scripts/335_train_qtrm_native_etd_probe.py",
            default_args=profile_args("qtrm_native_l2_curriculum_depth", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_l2_native_recursive_gain",
                "accepted_l1_native_etd",
                "accepted",
            ),
            on_accept=(
                "advance to L3 language-slice non-regression and L4 mixed text "
                "reasoning; keep curriculum/depth targets as training scaffolds"
            ),
            on_reject=(
                "inspect active-length curriculum, step-wise depth supervision, "
                "and core-state ablations before scaling to natural text"
            ),
        ),
        "qtrm_native_tiny_lm_first": GateSpec(
            name="qtrm_native_tiny_lm_first",
            target_level="L1 native tiny LM first",
            major_bottleneck=(
                "QTRM-native is hard-locked; prove donorless token->native "
                "backbone->mandatory recurrent core->LM logits viability before "
                "synthetic reasoning, donor, MemoryOS, or MSA work"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args("qtrm_native_tiny_lm_first", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-native-hard-lock.md",
            accepted_decisions=("accepted_native_tiny_lm_first", "accepted"),
            on_accept=(
                "continue native-only: add a small recursive-depth LM ablation "
                "or proceed to qtrm_native_l3_language_slice; do not switch to "
                "donor/residual QTRM"
            ),
            on_reject=(
                "stay native-only and fix tokenizer, decoder, recurrence "
                "placement, loss, or tiny corpus before any reasoning gate"
            ),
        ),
        "qtrm_native_tiny_lm_depth_ablation": GateSpec(
            name="qtrm_native_tiny_lm_depth_ablation",
            target_level="L2 native tiny LM depth ablation",
            major_bottleneck=(
                "after native tiny LM viability, the mandatory recurrent path "
                "must show a depth-dependent LM loss advantage over shallower "
                "native paths before synthetic reasoning promotion"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args("qtrm_native_tiny_lm_depth_ablation", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-native-hard-lock.md",
            accepted_decisions=("accepted_native_tiny_lm_depth_ablation", "accepted"),
            on_accept=(
                "continue native-only: broaden the text slice or reintroduce "
                "small synthetic reasoning while preserving the same native "
                "token->core->logits path"
            ),
            on_reject=(
                "stay native-only and fix recurrence placement, halting, depth "
                "schedule, or decoder readout before synthetic reasoning"
            ),
        ),
        "qtrm_native_l3_language_slice": GateSpec(
            name="qtrm_native_l3_language_slice",
            target_level="L3 native language slice",
            major_bottleneck=(
                "native recurrence must learn basic next-token text without "
                "repeated-token collapse or donor language borrowing"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args("qtrm_native_l3_language_slice", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l3_language_slice", "accepted"),
            on_accept=(
                "combine text prompts with the recursive reasoning gate in L4; "
                "do not claim broad language ability from this tiny slice alone"
            ),
            on_reject=(
                "fix tokenizer, decoder, recurrence placement, or loss before "
                "attempting mixed reasoning-language gates"
            ),
        ),
        "qtrm_native_l5_language_nonregression": GateSpec(
            name="qtrm_native_l5_language_nonregression",
            target_level="L5C native language non-regression",
            major_bottleneck=(
                "native recurrence must preserve larger text next-token loss "
                "instead of winning reasoning gates by damaging the ordinary LM path"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args("qtrm_native_l5_language_nonregression", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l5_language_nonregression", "accepted"),
            on_accept=(
                "stay QTRM-native: if this was triage, run the standard "
                "language non-regression gate; if standard passes, broaden the "
                "native text slice before any backbone comparison"
            ),
            on_reject=(
                "fix recurrence placement, tokenizer/text data, capacity, or "
                "LM loss before adding MSA, donor distillation, or larger memory"
            ),
        ),
        "qtrm_native_broad_wiki_text_nonregression": GateSpec(
            name="qtrm_native_broad_wiki_text_nonregression",
            target_level="L5C broad native wiki text non-regression",
            major_bottleneck=(
                "the native language non-regression signal must survive a "
                "multi-document wiki corpus, not only a single curated file"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args("qtrm_native_broad_wiki_text_nonregression", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-native-hard-lock.md",
            accepted_decisions=("accepted_broad_wiki_text_nonregression", "accepted"),
            on_accept=(
                "stay QTRM-native: add a broad-corpus depth sweep or only then "
                "return to native mixed reasoning under the same token->core->logits path"
            ),
            on_reject=(
                "treat the prior single-file language result as too narrow; "
                "fix corpus loading, capacity, tokenizer, or recurrence placement"
            ),
        ),
        "qtrm_native_broad_wiki_depth_ablation": GateSpec(
            name="qtrm_native_broad_wiki_depth_ablation",
            target_level="L5C broad native wiki depth ablation",
            major_bottleneck=(
                "the broad wiki text result must depend on recurrent depth, "
                "not just on turning any recurrent block on"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args("qtrm_native_broad_wiki_depth_ablation", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-native-hard-lock.md",
            accepted_decisions=("accepted_broad_wiki_depth_ablation", "accepted"),
            on_accept=(
                "stay QTRM-native: return to native mixed reasoning only with "
                "the broad corpus language/depth baselines kept as regression gates"
            ),
            on_reject=(
                "do not promote mixed reasoning yet; fix depth schedule, "
                "recursive-state coupling, or capacity until broad-corpus depth is causal"
            ),
        ),
        "qtrm_native_l5d_official_fla_runtime": GateSpec(
            name="qtrm_native_l5d_official_fla_runtime",
            target_level="L5D official FLA runtime",
            major_bottleneck=(
                "Qwen3.5-style hybrid must use the official FLA GatedDeltaNet "
                "backend in strict mode before any backbone comparison claim"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args("qtrm_native_l5d_official_fla_runtime", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l5d_official_fla_runtime", "accepted"),
            on_accept=(
                "run L5D performance comparison next: MHA ETD versus official "
                "FLA GatedDeltaNet under the accepted L4/L5/L5C gates"
            ),
            on_reject=(
                "do not call the hybrid official; fix FLA import/runtime, "
                "GatedDeltaNet parameters, CUDA kernels, or strict backend wiring"
            ),
        ),
        "qtrm_native_l5d_placement_seed_stability": GateSpec(
            name="qtrm_native_l5d_placement_seed_stability",
            target_level="L5D placement seed stability",
            major_bottleneck=(
                "the staged official-FLA thinking-core placement must beat MHA "
                "ETD and pass causal ablations across seeds before promotion"
            ),
            script="scripts/343_qtrm_native_l5d_placement_seed_sweep.py",
            default_args=profile_args(
                "qtrm_native_l5d_placement_seed_stability",
                profile,
            ),
            report_name="placement_seed_sweep_summary.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l5d_placement_seed_stability", "accepted"),
            on_accept=(
                "promote the staged L5D placement: MHA ETD encode -> official "
                "FLA GatedDeltaNet recurrent think -> MHA ETD decode; next test "
                "language non-regression and longer training for this exact path"
            ),
            on_reject=(
                "do not promote the placement; inspect seed-specific failures, "
                "causal ablations, and MHA baseline deltas before scaling"
            ),
        ),
        "qtrm_native_l5d_placement_language_nonregression": GateSpec(
            name="qtrm_native_l5d_placement_language_nonregression",
            target_level="L5D placement language non-regression",
            major_bottleneck=(
                "the accepted official-FLA thinking-core placement must not "
                "damage the native autoregressive language path"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args(
                "qtrm_native_l5d_placement_language_nonregression",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_l5d_placement_language_nonregression",
                "accepted",
            ),
            on_accept=(
                "the staged official-FLA thinking-core placement preserves the "
                "small native LM path; next scale longer reasoning/language "
                "runs before MSA/LM2 memory gates"
            ),
            on_reject=(
                "do not scale the placement; fix language loss regression, "
                "sample degeneracy, or FLA recurrent training before memory work"
            ),
        ),
        "qtrm_native_l5d_placement_scaled_reasoning": GateSpec(
            name="qtrm_native_l5d_placement_scaled_reasoning",
            target_level="L5D placement scaled reasoning",
            major_bottleneck=(
                "the accepted staged official-FLA thinking-core placement must "
                "retain causal reasoning gains under longer training and larger eval"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args(
                "qtrm_native_l5d_placement_scaled_reasoning",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_l5d_placement_scaled_reasoning",
                "accepted",
            ),
            on_accept=(
                "scale-up confirms the staged official-FLA thinking-core path; "
                "next compare memory/long-context mechanisms only after preserving "
                "this causal LM path"
            ),
            on_reject=(
                "do not move to MSA/LM2 yet; inspect whether scale failure is "
                "exactness, depth gain, ablation drop, or a specific family"
            ),
        ),
        "qtrm_native_l5d_mamba3_placement_language_nonregression": GateSpec(
            name="qtrm_native_l5d_mamba3_placement_language_nonregression",
            target_level="L5D Mamba3 placement language non-regression",
            major_bottleneck=(
                "the MHA encode/decode plus official Mamba3 thinking-core "
                "placement must preserve native autoregressive language behavior"
            ),
            script="scripts/336_train_qtrm_native_text_probe.py",
            default_args=profile_args(
                "qtrm_native_l5d_mamba3_placement_language_nonregression",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_l5d_mamba3_placement_language_nonregression",
                "accepted",
            ),
            on_accept=(
                "Mamba3 think-core preserves the small native LM path; next "
                "run scaled reasoning before replacing the FLA canonical path"
            ),
            on_reject=(
                "do not promote Mamba3 think-core; inspect language loss, "
                "sample degeneracy, strict backend wiring, and recurrence depth"
            ),
        ),
        "qtrm_native_l5d_mamba3_placement_scaled_reasoning": GateSpec(
            name="qtrm_native_l5d_mamba3_placement_scaled_reasoning",
            target_level="L5D Mamba3 placement scaled reasoning",
            major_bottleneck=(
                "the MHA encode/decode plus official Mamba3 thinking-core "
                "placement must retain causal reasoning gains under longer "
                "training and larger eval"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args(
                "qtrm_native_l5d_mamba3_placement_scaled_reasoning",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_l5d_mamba3_placement_scaled_reasoning",
                "accepted",
            ),
            on_accept=(
                "Mamba3 think-core now has both language preservation and "
                "scaled reasoning evidence; compare it against FLA as the next "
                "canonical placement candidate"
            ),
            on_reject=(
                "do not replace FLA yet; diagnose whether Mamba3 failed exact "
                "generation, depth gain, ablation drop, or backend strictness"
            ),
        ),
        "qtrm_native_dual_path_reverse_length_gate": GateSpec(
            name="qtrm_native_dual_path_reverse_length_gate",
            target_level="L5R fixed dual-path reverse length gate",
            major_bottleneck=(
                "the fixed dual-path reverse TRM core must scale len4->len6->len8 "
                "and beat the official TRM baseline through native LM logits"
            ),
            script="scripts/352_run_qtrm_native_dual_path_reverse_gate.py",
            default_args=profile_args(
                "qtrm_native_dual_path_reverse_length_gate",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-native-dual-path-reverse-active-architecture.md",
            accepted_decisions=(
                "accepted_dual_path_reverse_length_gate",
                "accepted",
            ),
            on_accept=(
                "freeze dual-path reverse as the active native core and run "
                "language non-regression before donor-integrated healing"
            ),
            on_reject=(
                "do not resume architecture shopping; diagnose the failed "
                "dual-path reverse length/depth/ablation axis and repair it"
            ),
        ),
        "qtrm_native_l4_mixed_text_reasoning": GateSpec(
            name="qtrm_native_l4_mixed_text_reasoning",
            target_level="L4 native reasoning + language",
            major_bottleneck=(
                "normal text-form prompts and text-form answers must improve "
                "through mandatory recurrent depth and destructive ablations"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args("qtrm_native_l4_mixed_text_reasoning", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l4_mixed_text_reasoning", "accepted"),
            on_accept=(
                "keep QTRM-native as the canonical scaffold; next run "
                "seed-stability and broad language/depth regression gates "
                "before any memory or backbone work"
            ),
            on_reject=(
                "do not add MSA or external retrieval; fix core-to-text "
                "generation, capacity, curriculum, or depth supervision first"
            ),
        ),
        "qtrm_native_dual_reverse_l4_baseline_compare": GateSpec(
            name="qtrm_native_dual_reverse_l4_baseline_compare",
            target_level="L4 dual reverse versus single baseline",
            major_bottleneck=(
                "dual reverse must beat the current single recurrent L4 "
                "baseline of 0.664 exact through the same donorless native "
                "token->core->logits path before it can be reconsidered"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args(
                "qtrm_native_dual_reverse_l4_baseline_compare", profile
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_dual_reverse_l4_baseline_compare", "accepted"),
            on_accept=(
                "dual reverse beat the latest single baseline; next compare "
                "against the full L4 0.70 acceptance threshold and broad language/depth gates"
            ),
            on_reject=(
                "keep single recurrent MHA ETD as the active baseline; do not "
                "promote dual reverse until it beats 0.664 with depth and ablation margins"
            ),
        ),
        "qtrm_native_nested_dual_reverse_l4_baseline_compare": GateSpec(
            name="qtrm_native_nested_dual_reverse_l4_baseline_compare",
            target_level="L4 nested dual reverse versus single baseline",
            major_bottleneck=(
                "Nested Learning inspired dual reverse must show that learned "
                "fast z_L and slow z_H update rules improve the same donorless "
                "native token->core->logits metric without hiding the answer in "
                "a side channel"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args(
                "qtrm_native_nested_dual_reverse_l4_baseline_compare", profile
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_nested_dual_reverse_l4_baseline_compare",
                "accepted",
            ),
            on_accept=(
                "promote the nested dual-reverse candidate to a seed sweep and "
                "run broad language/depth non-regression before memory work"
            ),
            on_reject=(
                "keep the single recurrent MHA ETD baseline; inspect whether "
                "the learned nested update improved exact accuracy, z_L/z_H "
                "causality, or neither"
            ),
        ),
        "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare": GateSpec(
            name="qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare",
            target_level="L4 nested official-schedule split-mixer 3:1 versus single baseline",
            major_bottleneck=(
                "the official TRM schedule must stay H=3 and L=6 while z_L uses "
                "Mamba3+Attention 3:1 and z_H uses GatedDelta+Attention 3:1 with "
                "nested learned updates, then beat the same donorless native "
                "token->core->logits baseline"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args(
                "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_nested_official_schedule_split_mixer_3to1_l4_baseline_compare",
                "accepted",
            ),
            on_accept=(
                "promote the nested official-schedule split-mixer 3:1 candidate "
                "to seed stability and broad language/depth non-regression"
            ),
            on_reject=(
                "do not replace the accepted nested MHA repair; inspect whether "
                "the official H=3/L=6 Mamba3/GatedDelta split-mixer failed exact "
                "accuracy, z_L/z_H causality, or both before any further "
                "architecture shopping"
            ),
        ),
        "qtrm_native_fast_slow_latent_update_l4_repair": GateSpec(
            name="qtrm_native_fast_slow_latent_update_l4_repair",
            target_level="L4 Fast-Slow latent update repair",
            major_bottleneck=(
                "the official H=3/L=6 split-mixer core must make both fast z_L "
                "and slow z_H causally necessary through the same native "
                "token->core->LM-logit path, without adding a side renderer"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args(
                "qtrm_native_fast_slow_latent_update_l4_repair",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=(
                "accepted_fast_slow_latent_update_l4_repair",
                "accepted",
            ),
            on_accept=(
                "promote the Fast-Slow latent update repair to seed stability "
                "and broad language/depth non-regression"
            ),
            on_reject=(
                "inspect whether z_L-zero, z_H-zero, exact accuracy, or language "
                "retention is the limiting factor before changing mixers again"
            ),
        ),
        "qtrm_native_l5_multifamily": GateSpec(
            name="qtrm_native_l5_multifamily",
            target_level="L5 broader reasoning families",
            major_bottleneck=(
                "the same native recurrent LM path must solve multiple tagged "
                "reasoning families, not only the modular forward-chain task"
            ),
            script="scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
            default_args=profile_args("qtrm_native_l5_multifamily", profile),
            report_name="report.json",
            wiki_path="docs/wiki/architecture/qtrm-native-first-roadmap.md",
            accepted_decisions=("accepted_l5_multifamily", "accepted"),
            on_accept=(
                "promote to L5 seed sweep and keep broad native language/depth "
                "regression gates before any backbone comparison"
            ),
            on_reject=(
                "inspect per-family failures; do not switch backbone or add MSA "
                "until the family bottleneck is understood"
            ),
        ),
    }


def default_out_dir(gate: GateSpec, profile: str, out_root: str | Path) -> Path:
    return Path(out_root) / f"{gate.name}_{profile}"


def gate_command(gate: GateSpec, out_dir: str | Path) -> list[str]:
    return [
        sys.executable,
        gate.script,
        "--out-dir",
        str(out_dir),
        *gate.default_args,
    ]


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"missing gate report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def normalize_decision(report: dict[str, Any]) -> str:
    decision = str(report.get("decision") or report.get("status") or "").strip().lower()
    return decision or "unknown"


def is_accepted(report: dict[str, Any], gate: GateSpec) -> bool:
    decision = normalize_decision(report)
    return decision in {item.lower() for item in gate.accepted_decisions}


def _get_nested(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def decisive_metrics(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "eval_metrics.depth8_final_exact",
        "eval_metrics.depth4_final_exact",
        "eval_metrics.depth1_final_exact",
        "eval_metrics.depth2_final_exact",
        "eval_metrics.depth2_state_exact",
        "ablations.state_reset.depth8_final_exact",
        "ablations.state_reset.depth4_final_exact",
        "ablations.state_reset.depth2_final_exact",
        "ablations.op_zero.depth8_final_exact",
        "ablations.op_zero.depth4_final_exact",
        "ablations.op_zero.depth2_final_exact",
        "ablations.op_shuffle.depth8_final_exact",
        "ablations.order_shuffle.depth4_final_exact",
        "ablations.order_shuffle.depth2_final_exact",
        "last_loss",
        "metrics.full_answer_accuracy",
        "metrics.core_off_answer_accuracy",
        "metrics.full_minus_core_off",
        "metrics.donor_forced_choice_accuracy",
        "metrics.donor_greedy_accuracy",
        "metrics.full_minus_donor",
        "metrics.full_generation_accuracy",
        "metrics.core_off_generation_accuracy",
        "metrics.donor_generation_accuracy",
        "metrics.state_off_generation_accuracy",
        "metrics.ablation_generation_accuracy",
        "metrics.full_minus_ablation",
        "metrics.full_minus_state_off",
        "metrics.eval_family_count",
        "eval_metrics.think4.generation_exact",
        "eval_metrics.think4.answer_token_accuracy",
        "eval_metrics.think4.first_token_eos_rate",
        "eval_metrics.think0.generation_exact",
        "eval_metrics.state_reset.generation_exact",
        "eval_metrics.op_zero.generation_exact",
        "eval_metrics.think_eval_loss",
        "eval_metrics.think0_loss",
        "eval_metrics.thinking_block_off_loss",
        "eval_metrics.think0_baseline_loss",
        "eval_metrics.loss_ratios.full_vs_think0",
        "eval_metrics.loss_ratios.full_vs_thinking_block_off",
        "eval_metrics.loss_ratios.full_vs_baseline",
        "eval_metrics.loss_ratios.full_vs_best_shallow_depth",
        "eval_metrics.best_shallow_depth_loss",
        "eval_metrics.sample_degeneracy.unique_chars",
        "eval_metrics.sample_degeneracy.max_run_fraction",
        "backend_summary.fla_delta_mixers",
        "backend_summary.official_fla_delta_mixers",
        "backend_summary.mamba3_mixers",
        "backend_summary.official_mamba3_mixers",
        "backend_summary.torch_delta_mixers",
        "backend_summary.all_fla_mixers_official",
        "backend_summary.all_mamba3_mixers_official",
        "full_minus_think0",
        "full_minus_worst_ablation",
        "decisive_metrics.full_generation_exact",
        "decisive_metrics.think0_generation_exact",
        "decisive_metrics.state_reset_generation_exact",
        "decisive_metrics.op_zero_generation_exact",
        "decisive_metrics.full_minus_think0",
        "decisive_metrics.full_minus_worst_ablation",
        "decisive_metrics.min_family_generation_exact",
        "decisive_metrics.active_rows",
        "decisive_metrics.min_active_full_generation_exact",
        "decisive_metrics.min_active_full_minus_think0",
        "decisive_metrics.min_active_full_minus_worst_ablation",
        "decisive_metrics.min_active_target_len_generation_exact",
        "decisive_metrics.min_active_minus_official",
        "decisive_metrics.min_active_target_len_minus_official",
        "promoted_count",
        "promoted_rate",
        "causal_ok_count",
        "backend_ok_count",
        "min_delta_vs_mha",
        "max_delta_vs_mha",
        "min_full_generation_exact",
        "max_full_generation_exact",
        "full_trace_exact_accuracy",
        "full_value_accuracy",
        "full_step_exact_accuracy",
        "ablation_trace_exact_accuracy",
        "ablation_value_accuracy",
        "ablation_step_exact_accuracy",
        "trace_drop",
        "value_drop",
        "numeric_ablation_value_accuracy",
        "numeric_value_drop",
        "token_numeric_ablation_value_accuracy",
        "token_numeric_value_drop",
        "best_exact_acc",
    )
    metrics: dict[str, Any] = {}
    for key in keys:
        value = _get_nested(report, key) if "." in key else report.get(key)
        if value is not None:
            metrics[key] = value
    return metrics


def primary_metric(metrics: dict[str, Any]) -> tuple[str, Any]:
    priority = (
        "decisive_metrics.min_active_full_generation_exact",
        "decisive_metrics.full_generation_exact",
        "eval_metrics.think4.generation_exact",
        "metrics.full_generation_accuracy",
        "metrics.full_answer_accuracy",
        "eval_metrics.depth8_final_exact",
        "eval_metrics.depth4_final_exact",
        "best_exact_acc",
    )
    for key in priority:
        if key in metrics:
            return key, metrics[key]
    return "", ""


def operational_status(summary: dict[str, Any]) -> str:
    decision = str(summary.get("decision") or "")
    if decision == "command_failed":
        return "crash"
    if decision == "dry_run":
        return "probe"
    if bool(summary.get("accepted")):
        return "keep"
    return "discard"


def append_operation_ledger(path: str | Path, summary: dict[str, Any]) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary.get("decisive_metrics") or {}
    metric_name, metric_value = primary_metric(metrics)
    header = (
        "timestamp\tgate\tprofile\tdecision\tstatus\tprimary_metric\t"
        "primary_value\tout_dir\treport_path\tnext_action\n"
    )
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        ledger_path.write_text(header, encoding="utf-8")
    row = [
        str(summary.get("timestamp", "")),
        str(summary.get("gate", "")),
        str(summary.get("profile", "")),
        str(summary.get("decision", "")),
        operational_status(summary),
        str(metric_name),
        str(metric_value),
        str(summary.get("out_dir", "")),
        str(summary.get("report_path", "")),
        str(summary.get("next_action", "")).replace("\t", " ").replace("\n", " "),
    ]
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def append_wiki_result(wiki_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(wiki_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = str(summary["timestamp"])
    metrics = summary.get("decisive_metrics") or {}
    lines = [
        "",
        f"## Runner Result {timestamp}",
        "",
        "```text",
        f"gate: {summary['gate']}",
        f"target_level: {summary['target_level']}",
        f"profile: {summary['profile']}",
        f"decision: {summary['decision']}",
        f"accepted: {summary['accepted']}",
        f"next_action: {summary['next_action']}",
        "```",
        "",
        "Decisive metrics:",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        f"Report: `{summary['report_path']}`",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_gate(
    *,
    gate_name: str,
    profile: str,
    out_root: str | Path,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    write_wiki: bool = False,
    operation_ledger: str | Path | None = None,
) -> dict[str, Any]:
    specs = gate_specs(profile)
    if gate_name not in specs:
        raise ValueError(f"unknown gate: {gate_name}")
    gate = specs[gate_name]
    run_dir = Path(out_dir) if out_dir is not None else default_out_dir(gate, profile, out_root)
    report_path = run_dir / gate.report_name
    command = gate_command(gate, run_dir)
    root = repo_root()
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    exit_code: int | None = None

    if dry_run:
        report = {
            "decision": "dry_run",
            "status": "dry_run",
            "target_level": gate.target_level,
        }
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if skip_existing and report_path.exists():
            report = load_report(report_path)
            exit_code = 0
        else:
            env = dict(os.environ)
            env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            exit_code = int(completed.returncode)
            (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if exit_code != 0 and report_path.exists():
                report = load_report(report_path)
            elif exit_code != 0:
                report = {
                    "decision": "command_failed",
                    "status": "failed",
                    "returncode": exit_code,
                }
            else:
                report = load_report(report_path)

    accepted = is_accepted(report, gate)
    decision = normalize_decision(report)
    next_action = gate.on_accept if accepted else gate.on_reject
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "gate": gate.name,
        "target_level": gate.target_level,
        "major_bottleneck": gate.major_bottleneck,
        "profile": profile,
        "command": command,
        "out_dir": str(run_dir),
        "report_path": str(report_path),
        "exit_code": exit_code,
        "decision": decision,
        "accepted": accepted,
        "next_action": next_action,
        "decisive_metrics": decisive_metrics(report),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if operation_ledger is not None:
        append_operation_ledger(root / operation_ledger, summary)
    if write_wiki and not dry_run:
        append_wiki_result(root / gate.wiki_path, summary)
    return summary


def list_gates(profile: str) -> list[dict[str, str]]:
    return [
        {
            "name": gate.name,
            "target_level": gate.target_level,
            "major_bottleneck": gate.major_bottleneck,
            "on_accept": gate.on_accept,
            "on_reject": gate.on_reject,
        }
        for gate in gate_specs(profile).values()
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-click research gate runner. Executes a falsifiable gate, parses "
            "report.json, writes gate_summary.json, and emits the next branch."
        )
    )
    parser.add_argument("--gate", default="donorless_recurrent_depth")
    parser.add_argument("--profile", choices=["smoke", "triage", "standard"], default="standard")
    parser.add_argument("--out-root", default="local_eval/research_gate_runner")
    parser.add_argument("--out-dir", default=None, help="Override the run directory for one gate.")
    parser.add_argument("--list-gates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-wiki", action="store_true")
    parser.add_argument(
        "--operation-ledger",
        default=None,
        help=(
            "Append an autoresearch-style TSV row with keep/discard/crash status. "
            "Use a repo-relative path such as local_eval/research_gate_runner/results.tsv."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.list_gates:
        print(json.dumps(list_gates(args.profile), ensure_ascii=False, indent=2))
        return 0
    summary = run_gate(
        gate_name=args.gate,
        profile=args.profile,
        out_root=args.out_root,
        out_dir=args.out_dir,
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
        write_wiki=bool(args.write_wiki),
        operation_ledger=args.operation_ledger,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision"] != "command_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
