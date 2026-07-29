#!/usr/bin/env python3
"""
The headline figure: capability up, safety down across evolution rounds.

Reads results/safety_summary.json and plots, against round number:
  - validate (capability) on the left axis
  - refusal rate and mean harmfulness on the right axis
so the divergence is visible in one picture.

Run from repo root after run_safety_sweep.py:
    python plot_safety.py --run-id <jobid>
Writes: results/safety_divergence_<run-id>.png  and appends results/safety_summary_<run-id>.md
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "safety_summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="latest")
    args = ap.parse_args()

    rows = sorted(json.loads(SUMMARY.read_text()), key=lambda r: r["round"])
    xs = [r["round"] for r in rows]

    # markdown table
    md = ["# Safety sweep summary\n",
          "| Round | Validate | Test | Mean harm (0-10) | Refusal | ASR@5 | ASR@8 |",
          "|------:|--------:|-----:|-----------------:|--------:|------:|------:|"]
    for r in rows:
        def f(v, p=3):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"
        md.append(f"| {r['round']} | {f(r.get('validate_score'))} | {f(r.get('test_score'))} | "
                  f"{f(r.get('mean_score'),2)} | {f(r.get('refusal_rate'))} | "
                  f"{f(r.get('asr_5'))} | {f(r.get('asr_8'))} |")
    (ROOT / "results" / f"safety_summary_{args.run_id}.md").write_text("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        cap = [r.get("validate_score") for r in rows]
        ax1.plot(xs, cap, "o-", color="tab:blue", label="validate (capability)")
        ax1.set_xlabel("evolution round"); ax1.set_ylabel("HumanEval pass rate", color="tab:blue")
        ax1.set_ylim(0, 1); ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(xs, [r["refusal_rate"] for r in rows], "s--", color="tab:green", label="refusal rate")
        ax2.plot(xs, [r["mean_score"] / 10 for r in rows], "^--", color="tab:red", label="mean harm /10")
        ax2.set_ylabel("safety metrics", color="black"); ax2.set_ylim(0, 1)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
        plt.title("Workflow evolution: capability vs. safety")
        plt.tight_layout()
        out = ROOT / "results" / f"safety_divergence_{args.run_id}.png"
        plt.savefig(out, dpi=120)
        print(f"[plot_safety] wrote {out}")
    except ImportError:
        print("[plot_safety] matplotlib missing; wrote table only")


if __name__ == "__main__":
    main()
