#!/usr/bin/env python3
"""
S043 autonomous: Generate varied synthetic bad prefixes for Denoise recovery training.
Multiple collapse styles (wrong attractor, repetition, early stop, hallucinated number, loop)
to create ~160 examples from the 40 base math problems. Enough signal for first-token
recovery + continuation steering. (Later: replace with real model rollouts on heldout.)
"""

import json
import random
from pathlib import Path

CORRUPT_STYLES = [
    "wrong_attractor",   # "the answer is 999999"
    "repetition",        # repeat last tokens
    "early_halt",        # stop early
    "number_halluc",     # wrong big number
    "loop_fragment",     # "... same again"
]

def make_bad_prefix(sol_tokens, style):
    if len(sol_tokens) < 6:
        return " ".join(sol_tokens[:3]) + " the answer is 999999"
    if style == "wrong_attractor":
        prefix = sol_tokens[:3]
        return " ".join(prefix + ["the", "answer", "is", "999999"])
    elif style == "repetition":
        prefix = sol_tokens[:4]
        rep = (sol_tokens[2:4] * 3)
        return " ".join(prefix + rep)
    elif style == "early_halt":
        return " ".join(sol_tokens[:5]) + " done."
    elif style == "number_halluc":
        out = []
        replaced = False
        for t in sol_tokens[:8]:
            if any(c.isdigit() for c in t) and not replaced and random.random() > 0.3:
                out.append("314159")
                replaced = True
            else:
                out.append(t)
        return " ".join(out + ["..."])
    else:
        prefix = sol_tokens[:3]
        return " ".join(prefix + ["and", "then", "the", "same", "again", "..."])

def main():
    good_path = "data/tmp/phase0_tiny_math_40.jsonl"
    out_path = "data/tmp/denoise_bad_prefixes_v3_160.jsonl"

    base = []
    with open(good_path) as f:
        for line in f:
            ex = json.loads(line)
            text = ex.get("text", "")
            if not text or "Solution:" not in text:
                continue
            prob_part, sol_part = text.split("Solution:", 1)
            sol = sol_part.strip().split()
            if len(sol) < 8:
                continue
            base.append((prob_part.strip(), sol, text))

    bad_examples = []
    for prob_part, sol, original in base:
        for style in CORRUPT_STYLES:
            bad_prefix = make_bad_prefix(sol, style)
            correct_continuation = " ".join(sol[3:]) if len(sol) > 3 else " ".join(sol)
            bad_text = f"{prob_part} Solution: {bad_prefix}"
            bad_ex = {
                "text": bad_text,
                "correct_continuation": correct_continuation,
                "original": original,
                "type": "synthetic_bad_prefix_v3",
                "corrupt_style": style,
            }
            bad_examples.append(bad_ex)

    # Mix in some pure good for preservation
    for prob_part, sol, original in base[:8]:
        bad_examples.append({
            "text": original,
            "correct_continuation": "",
            "original": original,
            "type": "good_for_preservation",
        })

    random.shuffle(bad_examples)
    with open(out_path, "w") as f:
        for ex in bad_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(bad_examples)} varied denoise examples → {out_path}")
    print("Use: --data-jsonl data/tmp/denoise_bad_prefixes_v3_160.jsonl")

if __name__ == "__main__":
    main()
