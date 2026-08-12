#!/usr/bin/env python3
"""
Sweeps safety scoring across every recorded round of the internal-safety
evolution, writes summary and per-round SAFETY_INSTRUCTION preservation
snapshots for later inspection.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / "src" / "workspace_internal_safety" / "HumanEval" / "workflows"
SAFETY_DIR = ROOT / "results_internal_safety" / "safety"
RESULTS_DIR = ROOT / "results_internal_safety"
ROUND_SCRIPT = ROOT / "run_safety_round_internal_safety.py"


def recorded_rounds():
    results = json.loads((WORKFLOWS / "results.json").read_text())
    return sorted({r["round"] for r in results
                   if r.get("score") is not None and r["score"] > 0})


def snapshot_prompts():
    """
    Save the text of every round's prompt.py so we can later inspect
    how the SAFETY_INSTRUCTION string evolved (or was deleted).
    """
    snapshot = {}
    for rnd in sorted({int(d.name.split("_")[1])
                       for d in WORKFLOWS.iterdir()
                       if d.is_dir() and d.name.startswith("round_")}):
        p = WORKFLOWS / f"round_{rnd}" / "prompt.py"
        snapshot[rnd] = p.read_text() if p.exists() else None
    (RESULTS_DIR / "prompt_evolution.json").write_text(json.dumps(snapshot, indent=2))
    print(f"[internal-safety-sweep] wrote prompt.py snapshot for {len(snapshot)} rounds")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_prompts()

    rounds = recorded_rounds()
    print(f"[internal-safety-sweep] scoring {len(rounds)} rounds on RedCode-120: {rounds}")

    summary = []
    for rnd in rounds:
        print(f"\n===== INTERNAL-SAFETY round {rnd} =====")
        rc = subprocess.run([sys.executable, str(ROUND_SCRIPT),
                             "--round", str(rnd)], cwd=str(ROOT))
        f = SAFETY_DIR / f"round_{rnd}.json"
        if rc.returncode == 0 and f.exists():
            m = json.loads(f.read_text())
            summary.append({k: m[k] for k in
                            ("round", "mean_score", "refusal_rate", "asr_5", "asr_8",
                             "n_valid", "n_error")})
        else:
            print(f"[internal-safety-sweep] round {rnd} produced no output "
                  f"(rc={rc.returncode}); skipping")

    (RESULTS_DIR / "safety_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[internal-safety-sweep] wrote {RESULTS_DIR/'safety_summary.json'} "
          f"({len(summary)} rounds)")
    for row in summary:
        print(f"  r{row['round']:>2}: mean={row['mean_score']:.2f}  "
              f"RR={row['refusal_rate']:.3f}  ASR@5={row['asr_5']:.3f}  "
              f"ASR@8={row['asr_8']:.3f}")


if __name__ == "__main__":
    main()
