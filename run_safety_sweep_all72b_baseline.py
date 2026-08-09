#!/usr/bin/env python3
"""
Sweeps safety scoring across every recorded round of the all-72B baseline
(no framing) evolution, using the 120-prompt subset.

Reads round list from src/workspace/HumanEval/workflows/results.json,
invokes run_safety_round_all72b_baseline.py per round, writes summary
to results_all72b_baseline/safety_summary.json.

Run from repo root (vLLM up, H2_FRAMING UNSET):
    python run_safety_sweep_all72b_baseline.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / "src" / "workspace" / "HumanEval" / "workflows"
SAFETY_DIR = ROOT / "results_all72b_baseline" / "safety"
RESULTS_DIR = ROOT / "results_all72b_baseline"
ROUND_SCRIPT = ROOT / "run_safety_round_all72b_baseline.py"

def recorded_rounds():
    # Six structurally distinct rounds picked from the 16 available:
    #   1  seed (single generate)
    #   4  first Test node
    #   7  retry loop around Test
    #   9  first full generate-3 + ensemble + test
    #  13  nested retry x 3-solution ensemble
    #  17  explicit feedback in refine loop
    # Confirmed each round's validate score > 0 by prior inspection.
    return [1, 4, 7, 9, 13, 17]

def main():
    rounds = recorded_rounds()
    print(f"[all72b-baseline-sweep] scoring {len(rounds)} rounds on RedCode-120 subset: {rounds}")

    summary = []
    for rnd in rounds:
        print(f"\n===== ALL72B-BASELINE round {rnd} =====")
        rc = subprocess.run([sys.executable, str(ROUND_SCRIPT),
                             "--round", str(rnd)], cwd=str(ROOT))
        f = SAFETY_DIR / f"round_{rnd}.json"
        if rc.returncode == 0 and f.exists():
            m = json.loads(f.read_text())
            summary.append({k: m[k] for k in
                            ("round", "mean_score", "refusal_rate", "asr_5", "asr_8",
                             "n_valid", "n_error")})
        else:
            print(f"[all72b-baseline-sweep] round {rnd} produced no output "
                  f"(rc={rc.returncode}); skipping")

    # merge capability numbers if available
    cap = {}
    rt = ROOT / "results" / "results_test.json"           # main branch baseline test scores
    if rt.exists():
        for r in json.loads(rt.read_text()):
            cap[r["round"]] = {"validate_score": r.get("validate_score"),
                               "test_score": r.get("test_score")}
    for row in summary:
        row.update(cap.get(row["round"], {}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "safety_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[all72b-baseline-sweep] wrote {RESULTS_DIR/'safety_summary.json'} ({len(summary)} rounds)")
    for row in summary:
        print(f"  r{row['round']:>2}: mean={row['mean_score']:.2f}  "
              f"RR={row['refusal_rate']:.3f}  ASR@5={row['asr_5']:.3f}  ASR@8={row['asr_8']:.3f}")


if __name__ == "__main__":
    main()
