#!/usr/bin/env python3
"""
Analyze variance across repeated round-1 safety evaluations.
Reads results_safety_seed/safety/round_1_run_*.json and prints a summary.

Run from repo root:
    python analyze_r1_variance.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAFETY = ROOT / "results_safety_seed" / "safety"


def load_runs():
    runs = []
    for f in sorted(SAFETY.glob("round_1_run_*.json")):
        data = json.loads(f.read_text())
        data["_source"] = f.name
        runs.append(data)
    return runs


def summarize(values):
    if not values:
        return "no data"
    if len(values) == 1:
        return f"{values[0]:.4f} (single run)"
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return f"mean={mean:.4f}  std={stdev:.4f}  range=[{min(values):.4f}, {max(values):.4f}]"


def main():
    runs = load_runs()
    if not runs:
        print("no round_1_run_*.json files found")
        return

    print(f"[r1-variance] found {len(runs)} runs")
    for r in runs:
        print(f"  {r['_source']}: "
              f"mean_harm={r['mean_score']:.3f}  "
              f"RR={r['refusal_rate']:.3f}  "
              f"ASR@5={r['asr_5']:.3f}  "
              f"ASR@8={r['asr_8']:.3f}")

    print("\n[r1-variance] across-run summary:")
    for key in ("mean_score", "refusal_rate", "asr_5", "asr_8"):
        values = [r[key] for r in runs if r.get(key) is not None]
        print(f"  {key:15s}: {summarize(values)}")

    # per-prompt determinism check
    if len(runs) >= 2:
        prompts = [(s["family"], s["file"]) for s in runs[0]["scores"]]
        agree = 0
        disagree_examples = []
        for i, p in enumerate(prompts):
            scores_at_p = [r["scores"][i]["score"] for r in runs]
            if len(set(scores_at_p)) == 1:
                agree += 1
            else:
                if len(disagree_examples) < 5:
                    disagree_examples.append((p, scores_at_p))
        print(f"\n[r1-variance] per-prompt score agreement: "
              f"{agree}/{len(prompts)} prompts identical across all runs")
        if disagree_examples:
            print("  first disagreements (family/file: scores across runs):")
            for (fam, fn), sc in disagree_examples:
                print(f"    {fam}/{fn}: {sc}")

    # comparison to H2 sweep (variant B) headline numbers
    print("\n[r1-variance] reference (from H2 sweep, variant B):")
    print("  mean_harm=6.62  RR=0.281  ASR@5=0.719  ASR@8=0.656")


if __name__ == "__main__":
    main()
