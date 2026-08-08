#!/usr/bin/env python3
"""
Sweeps safety scoring across every recorded round of the qwen-32b-executor
(no framing) run. Invokes run_safety_round_qwen32b_executor.py per round.

Writes summary to
    results_qwen32b_executor/safety_summary.json

Run from repo root (both vLLM up, H2_FRAMING UNSET):
    python run_safety_sweep_qwen32b_executor.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / "src" / "workspace_qwen32b_executor" / "HumanEval" / "workflows"
SAFETY_DIR = ROOT / "results_qwen32b_executor" / "safety"
RESULTS_DIR = ROOT / "results_qwen32b_executor"
ROUND_SCRIPT = ROOT / "run_safety_round_qwen32b_executor.py"


def recorded_rounds():
    results = json.loads((WORKFLOWS / "results.json").read_text())
    return sorted({r["round"] for r in results if r.get("score") is not None})


def main():
    rounds = recorded_rounds()
    print(f"[qwen32b-exec-sweep] scoring {len(rounds)} rounds on RedCode subset: {rounds}")

    summary = []
    for rnd in rounds:
        print(f"\n===== QWEN32B-EXEC round {rnd} =====")
        rc = subprocess.run([sys.executable, str(ROUND_SCRIPT),
                             "--round", str(rnd)], cwd=str(ROOT))
        f = SAFETY_DIR / f"round_{rnd}.json"
        if rc.returncode == 0 and f.exists():
            m = json.loads(f.read_text())
            summary.append({k: m[k] for k in
                            ("round", "mean_score", "refusal_rate", "asr_5", "asr_8",
                             "n_valid", "n_error")})
        else:
            print(f"[qwen32b-exec-sweep] round {rnd} produced no output (rc={rc.returncode}); skipping")

    # merge capability numbers (validate + test) from existing results_test.json
    cap = {}
    rt = RESULTS_DIR / "results_test.json"
    if rt.exists():
        for r in json.loads(rt.read_text()):
            cap[r["round"]] = {"validate_score": r.get("validate_score"),
                               "test_score": r.get("test_score")}
    for row in summary:
        row.update(cap.get(row["round"], {}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "safety_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[qwen32b-exec-sweep] wrote {RESULTS_DIR/'safety_summary.json'} ({len(summary)} rounds)")
    for row in summary:
        print(f"  r{row['round']:>2}: mean={row['mean_score']:.2f}  "
              f"RR={row['refusal_rate']:.2f}  ASR@5={row['asr_5']:.2f}  ASR@8={row['asr_8']:.2f}")


if __name__ == "__main__":
    main()
