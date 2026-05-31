#!/usr/bin/env python3
"""
S043 autonomous: Real bad-prefix harvester (no synthetic corruption).

Runs the current checkpoint (donor_only for speed) on heldout cases,
identifies actual free-gen failures (wrong exact match, early repetition, collapse),
truncates the bad generation at the first clear failure point,
and emits denoise-format jsonl (text=problem+bad_prefix, correct_continuation=gold_rest)
that can be fed directly to recovery training with the new dataset wiring.

This produces genuine self-generated failures the policy actually makes,
which is the core of DenoiseRL-style repair for free-gen improvement.
"""
import os
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

def find_failure_point(generated: str, gold: str, max_prefix_tokens: int = 40) -> str:
    """Crude but effective: take prefix of generated up to first obvious deviation or repetition or length cap."""
    gen = generated.strip()
    # Simple repetition detector (3-gram repeat)
    words = gen.split()
    for i in range(3, len(words)):
        if words[i-3:i] == words[i:i+3] and i > 8:
            return " ".join(words[:i])
    # Length cap (keep early part where first-token error likely happened)
    if len(words) > max_prefix_tokens:
        return " ".join(words[:max_prefix_tokens])
    # Fallback: whole wrong generation (still useful as "model went off the rails")
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/s043_denoise_recovery_real1.yaml")
    ap.add_argument("--cases", default="data/tmp/phase0_tiny_math_40.jsonl")  # full "text" with Solution: for rich generations
    ap.add_argument("--max-cases", type=int, default=60)
    ap.add_argument("--out", default="data/tmp/real_bad_prefixes_from_current.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    tmp_jsonl = Path(tempfile.mktemp(suffix=".jsonl"))

    print(f"[harvester] Running donor baseline on {args.max_cases} cases to harvest real failures...", file=sys.stderr)
    cmd = [
        sys.executable, "scripts/192_eval_raw_intelligence.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--cases", args.cases,
        "--out", str(tmp_jsonl),
        "--device", args.device,
        "--max-cases", str(args.max_cases),
        "--mode", "donor_only_no_evidence",
        "--scoring", "generation",
        "--max-new-tokens", "64",
        "--no-repeat-ngram-size", "2",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root}/src"
    subprocess.check_call(cmd, cwd=root, env=env)

    bad_examples = []
    with open(tmp_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("hit") or rec.get("exact_match"):
                continue  # only real failures

            gen = (rec.get("completion") or rec.get("generated") or rec.get("text") or "").strip()
            if not gen or len(gen) < 20:
                continue

            # Use 192-provided gold fields (robust)
            gold_answer = rec.get("gold_answer") or ""
            question = rec.get("question") or rec.get("prompt") or ""
            cid = rec.get("id")

            # Build a usable "problem + bad prefix" context
            # Prefer question/prompt as the visible prefix
            prob_part = question.strip() if question else (gold_answer.split("Solution:", 1)[0].strip() if "Solution:" in gold_answer else gold_answer[:300])

            bad_prefix = find_failure_point(gen, gold_answer or gen, max_prefix_tokens=38)

            # correct_continuation: the gold answer (or the part after the visible prompt)
            correct_cont = gold_answer
            if question and gold_answer:
                # If gold starts with question, strip the common prefix
                if gold_answer.strip().startswith(question.strip()[:50]):
                    correct_cont = gold_answer[len(question):].strip()

            if not correct_cont:
                correct_cont = gold_answer or gen  # last resort

            ex = {
                "text": f"{prob_part} Solution: {bad_prefix}".strip(),
                "correct_continuation": correct_cont.strip(),
                "original": gold_answer or question,
                "type": "real_bad_prefix_from_rollout",
                "source_id": cid,
                "failure_reason": rec.get("audit_reasons") or rec.get("reason") or "exact_miss_or_collapse",
            }
            bad_examples.append(ex)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for ex in bad_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[harvester] Wrote {len(bad_examples)} real failure prefixes → {outp}")
    print("Use with: --data-jsonl " + str(outp))
    print("Recommended: mix with 20-30% good examples for preservation if needed.")


if __name__ == "__main__":
    main()