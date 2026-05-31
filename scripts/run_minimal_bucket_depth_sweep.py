#!/usr/bin/env python3
import subprocess
import json
import re
import sys

def run_eval(checkpoint, steps, slots_off=False, num_cases=16):
    cmd = [
        ".venv/bin/python", "scripts/measure_continuation_hybrid_192.py",
        "--checkpoint", checkpoint,
        "--num_cases", str(num_cases),
        "--steps_per_case", str(steps)
    ]
    if slots_off:
        cmd.append("--slots_off")

    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "."})
    if result.returncode != 0:
        print(f"Error running eval for steps={steps}, slots_off={slots_off}: {result.stderr}")
        return None

    # Extract JSON between ## CONTINUATION_192_PROXY_JSON_START and ## CONTINUATION_192_PROXY_JSON_END
    match = re.search(r"## CONTINUATION_192_PROXY_JSON_START\n(.*?)\n## CONTINUATION_192_PROXY_JSON_END", result.stdout, re.DOTALL)
    if not match:
        print(f"Failed to find JSON block in output for steps={steps}, slots_off={slots_off}")
        return None

    try:
        return json.loads(match.group(1))
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return None

def main():
    checkpoint = "checkpoints/ri1_substrate_minimal_v4_20260529_220943/ri1_substrate_step600.pt"
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]

    num_cases = 16
    depths = [1, 2, 4, 8]

    print("=" * 80)
    print("RI-1 Causal Evaluation: Standalone Standalone Substrate Bucket Depth Sweep")
    print(f"Checkpoint : {checkpoint}")
    print(f"Cases      : {num_cases}")
    print(f"Depths     : {depths}")
    print("=" * 80)

    rows = []
    for depth in depths:
        print(f"Sweeping Depth {depth}...")
        # Memory ON (slots_off=False)
        res_on = run_eval(checkpoint, depth, slots_off=False, num_cases=num_cases)
        # Memory OFF (slots_off=True)
        res_off = run_eval(checkpoint, depth, slots_off=True, num_cases=num_cases)

        if res_on and res_off:
            acc_on = res_on.get("forced_choice_accuracy", 0.0) * 100
            acc_off = res_off.get("forced_choice_accuracy", 0.0) * 100
            margin = acc_on - acc_off

            rows.append({
                "depth": depth,
                "slots_on_acc": f"{acc_on:.2f}% ({res_on.get('forced_choice_correct')}/{num_cases})",
                "slots_off_acc": f"{acc_off:.2f}% ({res_off.get('forced_choice_correct')}/{num_cases})",
                "margin": f"{margin:+.2f}%",
                "margin_raw": margin
            })

    print("\n" + "=" * 80)
    print("SWEEP RESULTS")
    print("=" * 80)
    print(f"| Depth | Slots-Off (Baseline) | Slots-On (Active Memory) | Ablation Margin |")
    print(f"| :---: | :---: | :---: | :---: |")
    for r in rows:
        print(f"| **d={r['depth']}** | {r['slots_off_acc']} | {r['slots_on_acc']} | **{r['margin']}** |")
    print("=" * 80)

if __name__ == "__main__":
    main()
