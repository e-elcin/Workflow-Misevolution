#!/usr/bin/env python3
"""
Sweeps safety scoring across rounds 1 (seed) and 2 (paper aflow_agent)
of the paper-replication workspace, using the full 160-prompt subset.

Writes summary to results_paper_replication/safety_summary.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAFETY_DIR = ROOT / "results_paper_replication" / "safety"
RESULTS_DIR = ROOT / "results_paper_replication"
ROUND_SCRIPT = ROOT / "run_safety_round_paper_replication.py"

ROUNDS = [1, 2]                       # seed vs paper aflow_agent


def main():
    print(f"[paper-repl-sweep] scoring rounds {ROUNDS} on FULL RedCode-Gen (n=160)")
    summary = []
    for rnd in ROUNDS:
        print(f"\n===== PAPER-REPL round {rnd} =====")
        rc = subprocess.run([sys.executable, str(ROUND_SCRIPT),
                             "--round", str(rnd)], cwd=str(ROOT))
        f = SAFETY_DIR / f"round_{rnd}.json"
        if rc.returncode == 0 and f.exists():
            m = json.loads(f.read_text())
            summary.append({k: m[k] for k in
                            ("round", "mean_score", "refusal_rate", "asr_5", "asr_8",
                             "n_valid", "n_error")})
        else:
            print(f"[paper-repl-sweep] round {rnd} produced no output "
                  f"(rc={rc.returncode}); skipping")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "safety_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[paper-repl-sweep] wrote {RESULTS_DIR/'safety_summary.json'}")

    # nice terminal comparison
    if len(summary) == 2:
        s, e = summary[0], summary[1]
        print(f"\n=== paper's seed-vs-aflow_agent delta (n={s['n_valid']} prompts) ===")
        print(f"  Seed         : mean={s['mean_score']:.2f}  RR={s['refusal_rate']:.3f}  ASR@5={s['asr_5']:.3f}  ASR@8={s['asr_8']:.3f}")
        print(f"  AFlow-agent  : mean={e['mean_score']:.2f}  RR={e['refusal_rate']:.3f}  ASR@5={e['asr_5']:.3f}  ASR@8={e['asr_8']:.3f}")
        print(f"  Delta        : mean={e['mean_score']-s['mean_score']:+.2f}  "
              f"RR={e['refusal_rate']-s['refusal_rate']:+.3f}  "
              f"ASR@5={e['asr_5']-s['asr_5']:+.3f}  "
              f"ASR@8={e['asr_8']-s['asr_8']:+.3f}")


if __name__ == "__main__":
    main()
