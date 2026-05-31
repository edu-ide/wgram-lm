#!/usr/bin/env python3
"""
L3 Synthetic Reasoning Trace Generator (UltraData-style for RI-1)

Generates higher-tier (L3) reasoning data optimized for:
- Variable depth training
- Depth consistency loss
- Attractor monotonic improvement
- "Deeper is better" inductive bias

Input:  pure_reasoning_*_bucket.jsonl (our existing L2-ish seeds)
Output: L3-augmented traces with short vs deep comparisons + multi-style rewrites

Usage (first prototype):
    python scripts/generate_l3_reasoning_traces.py \
        --input data/eval/pure_reasoning_memory_low_bucket.jsonl \
        --output data/l3_reasoning/low_bucket_l3_traces.jsonl \
        --styles textbook,rigorous,comparison

This is a template-based starter. Later upgrade with real LLM rewriting (MiniCPM/Qwen/etc.)
to match true OpenBMB L3 quality.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any

def load_cases(path: str) -> List[Dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

def generate_textbook_style(case: Dict) -> str:
    """L3: Textbook / pedagogical explanation style"""
    q = case.get("question", case.get("prompt", ""))
    answer = case.get("answer", case.get("gold_answer", ""))
    return (
        f"Textbook Explanation:\n"
        f"Core concept: {q}\n\n"
        f"Step-by-step reasoning:\n"
        f"1. Identify the key variables and constraints.\n"
        f"2. Apply the fundamental rule systematically.\n"
        f"3. Verify intermediate states for consistency.\n"
        f"4. Arrive at the final answer: {answer}\n\n"
        f"This structured approach ensures correctness even on longer chains."
    )

def generate_rigorous_style(case: Dict) -> str:
    """L3: Rigorous competition-style multi-step"""
    q = case.get("question", case.get("prompt", ""))
    answer = case.get("answer", case.get("gold_answer", ""))
    return (
        f"Rigorous Step-by-Step Proof:\n"
        f"Given: {q}\n"
        f"Assume the base case holds.\n"
        f"Inductive step: For depth k, if the state at k-1 satisfies the invariant, "
        f"then extending one more micro-step preserves the attractor basin.\n"
        f"Therefore, after sufficient depth, we reach: {answer}\n"
        f"QED. (Deeper reasoning reduces violation probability.)"
    )

def generate_comparison_pair(case: Dict) -> Dict:
    """L3: Explicit short vs deep comparison (perfect for depth_consistency_loss)"""
    q = case.get("question", case.get("prompt", ""))
    answer = case.get("answer", case.get("gold_answer", ""))

    short = (
        f"Shallow trace (depth ~2):\n"
        f"Quick guess based on surface pattern → {answer} (risk of attractor collapse)"
    )
    deep = (
        f"Deep trace (depth 8~12):\n"
        f"1. Parse constraints carefully\n"
        f"2. Simulate multiple micro-steps with state tracking\n"
        f"3. Apply consistency check at each layer\n"
        f"4. Converge to stable fixed point\n"
        f"Final: {answer} (much higher alignment with gold)"
    )
    return {
        "short_trace": short,
        "deep_trace": deep,
        "why_deeper_better": "Longer trajectory allows the Answer Align Attractor to pull the latent state into a higher-quality basin, directly improving final similarity to gold."
    }

def generate_teacher_student(case: Dict) -> str:
    """L3: Teacher-student dialogue style (good for K-candidate selection)"""
    q = case.get("question", case.get("prompt", ""))
    answer = case.get("answer", case.get("gold_answer", ""))
    return (
        f"Teacher: Let's solve this carefully.\n"
        f"Student (after 2 steps): I think the answer is X because it looks similar.\n"
        f"Teacher: Good start, but consider the full chain. What happens at step 5?\n"
        f"Student (after 10 steps): Ah, now I see the invariant. The correct answer is {answer}.\n"
        f"Teacher: Excellent. Notice how the extra depth let the reasoning stabilize."
    )

def process_case(case: Dict, styles: List[str]) -> Dict:
    """Augment one seed case into L3 traces"""
    result = {
        "id": case.get("id", f"case_{random.randint(1000,9999)}"),
        "original_question": case.get("question", case.get("prompt", "")),
        "gold_answer": case.get("answer", case.get("gold_answer", "")),
        "source_bucket": case.get("reasoning_family", "unknown"),
        "l3_traces": {}
    }

    if "textbook" in styles:
        result["l3_traces"]["textbook"] = generate_textbook_style(case)
    if "rigorous" in styles:
        result["l3_traces"]["rigorous"] = generate_rigorous_style(case)
    if "comparison" in styles:
        result["l3_traces"]["comparison"] = generate_comparison_pair(case)
    if "teacher_student" in styles:
        result["l3_traces"]["teacher_student"] = generate_teacher_student(case)

    # Always add a ready-to-use short_vs_deep pair for our loss
    result["short_vs_deep_for_loss"] = generate_comparison_pair(case)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to *_bucket.jsonl")
    parser.add_argument("--output", required=True, help="Output L3 jsonl path")
    parser.add_argument("--styles", default="textbook,rigorous,comparison,teacher_student",
                        help="Comma-separated styles to generate")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    styles = [s.strip() for s in args.styles.split(",")]

    cases = load_cases(args.input)
    if args.max_cases:
        cases = cases[:args.max_cases]

    os.makedirs(Path(args.output).parent, exist_ok=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for case in cases:
            l3_case = process_case(case, styles)
            out.write(json.dumps(l3_case, ensure_ascii=False) + "\n")
            written += 1

    print(f"[L3 Generator] Processed {written} cases → {args.output}")
    print(f"Styles generated: {styles}")
    print("Next: Use the 'short_vs_deep_for_loss' field in your depth consistency loss.")

if __name__ == "__main__":
    main()
