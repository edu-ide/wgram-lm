#!/usr/bin/env python3
import json
import random
from pathlib import Path

def extract_rows(file_path, max_extract=50000):
    rows = []
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping missing file: {file_path}")
        return rows

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                messages = data.get("messages")
                if messages and len(messages) >= 2:
                    prompt = messages[0].get("content")
                    answer = messages[1].get("content")
                    if prompt and answer:
                        rows.append({"prompt": prompt, "answer": answer})
                        count += 1
                        if count >= max_extract:
                            break
            except Exception:
                pass
    print(f"✓ Extracted {len(rows)} rows from {path.name}")
    return rows

def main():
    print("=== Creating Balanced UltraData 16K Full SFT Dataset ===")

    raw_dir = Path("data/raw")
    out_dir = Path("data/filtered")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ultradata_sft_math_if_code_knowledge_16k.jsonl"

    # 1. Extract from Math -> target: 4,000
    math_files = [
        raw_dir / "ultradata_sft_2605_math_think.jsonl",
        raw_dir / "ultradata_sft_2605_math_no_think.jsonl"
    ]
    math_rows = []
    for f in math_files:
        math_rows.extend(extract_rows(f))

    # 2. Extract from IF -> target: 4,000
    if_files = [
        raw_dir / "ultradata_sft_2605_if_think.jsonl",
        raw_dir / "ultradata_sft_2605_if_no_think.jsonl"
    ]
    if_rows = []
    for f in if_files:
        if_rows.extend(extract_rows(f))

    # 3. Extract from Code -> target: 4,000
    code_files = [
        raw_dir / "ultradata_sft_2605_code_think.jsonl",
        raw_dir / "ultradata_sft_2605_code_no_think.jsonl"
    ]
    code_rows = []
    for f in code_files:
        code_rows.extend(extract_rows(f, max_extract=30000))

    # 4. Extract from Knowledge -> target: 4,000
    knowledge_files = [
        raw_dir / "ultradata_sft_2605_knowledge_think.jsonl",
        raw_dir / "ultradata_sft_2605_knowledge_no_think.jsonl"
    ]
    knowledge_rows = []
    for f in knowledge_files:
        knowledge_rows.extend(extract_rows(f, max_extract=30000))

    # Sample and mix
    random.seed(42)

    sampled_math = random.sample(math_rows, min(4000, len(math_rows)))
    sampled_if = random.sample(if_rows, min(4000, len(if_rows)))
    sampled_code = random.sample(code_rows, min(4000, len(code_rows)))
    sampled_knowledge = random.sample(knowledge_rows, min(4000, len(knowledge_rows)))

    combined = sampled_math + sampled_if + sampled_code + sampled_knowledge
    random.shuffle(combined)

    # Write output
    with out_path.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n==========================================")
    print(f"Successfully generated: {out_path}")
    print(f"Total Rows: {len(combined)} (Math: {len(sampled_math)}, IF: {len(sampled_if)}, Code: {len(sampled_code)}, Knowledge: {len(sampled_knowledge)})")
    print(f"==========================================")

if __name__ == "__main__":
    main()
