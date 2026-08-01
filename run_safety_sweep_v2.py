#!/usr/bin/env python3
"""
Run the RedCode safety evaluation across ALL evolved rounds.

For each round recorded in the evolution's results.json, loads that round's real
graph.py, scores it on the stratified RedCode subset, and records per-round
mean harmfulness (0-10), refusal rate, ASR@5, ASR@8, and error count.

Produces the per-round safety trajectory: capability was optimized, safety was not.

Run from repo root (vLLM + judge up), after build_redcode_subset.py:
    python run_safety_sweep.py
Writes:
    results/safety/round_<N>.json     (per round, via run_safety_round)
    results/safety_summary_v2.json       (all rounds, all metrics)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / "src" / "workspace" / "HumanEval" / "workflows"
SAFETY_DIR = ROOT / "results" / "safety_v2"


def recorded_rounds():
    results = json.loads((WORKFLOWS / "results.json").read_text())
    return sorted({r["round"] for r in results if r.get("score") is not None})


def main():
    rounds = recorded_rounds()
    print(f"[sweep] scoring {len(rounds)} rounds on RedCode subset: {rounds}")

    summary = []
    for rnd in rounds:
        print(f"\n===== SAFETY round {rnd} =====")
        # run each round in its own process so one bad graph can't kill the sweep
        rc = subprocess.run([sys.executable, str(ROOT / "run_safety_round_v2.py"),
                             "--round", str(rnd)], cwd=str(ROOT))
        f = SAFETY_DIR / f"round_{rnd}.json"
        if rc.returncode == 0 and f.exists():
            m = json.loads(f.read_text())
            summary.append({k: m[k] for k in
                            ("round", "mean_score", "refusal_rate", "asr_5", "asr_8",
                             "n_valid", "n_error")})
        else:
            print(f"[sweep] round {rnd} produced no output (rc={rc.returncode}); skipping")

    # merge validate/test capability if present, for the divergence view
    cap = {}
    rt = ROOT / "results" / "results_test.json"
    if rt.exists():
        for r in json.loads(rt.read_text()):
            cap[r["round"]] = {"validate_score": r.get("validate_score"),
                               "test_score": r.get("test_score")}
    for row in summary:
        row.update(cap.get(row["round"], {}))

    (ROOT / "results" / "safety_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[sweep] wrote results/safety_summary_v2.json ({len(summary)} rounds)")
    for row in summary:
        print(f"  r{row['round']:>2}: mean={row['mean_score']:.2f}  "
              f"RR={row['refusal_rate']:.2f}  ASR@5={row['asr_5']:.2f}  ASR@8={row['asr_8']:.2f}")


if __name__ == "__main__":
    main()
