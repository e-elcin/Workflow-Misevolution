#!/usr/bin/env python3
"""
Plot refusal rate trajectories for all four AFlow evolution experiments
on a single figure. Reads from committed safety_summary.json files
across the different results directories.

Usage:
    python plot_all4_refusal.py
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_summary(path, label):
    """Load a safety_summary.json and return sorted [(round, refusal_rate), ...]."""
    p = ROOT / path
    if not p.exists():
        print(f"WARN: {p} missing, skipping {label}")
        return []
    data = json.loads(p.read_text())
    return sorted([(r["round"], r["refusal_rate"]) for r in data if r.get("refusal_rate") is not None])


# Load all four experiments
experiments = [
    ("results_safety_seed/safety_summary.json",
     "72B + safety framing at seed",  "tab:red",    "o"),
    ("results_all72b_baseline/safety_summary.json",
     "72B baseline, no framing",      "tab:blue",   "s"),
    ("results_qwen32b_executor/safety_summary.json",
     "Split-role (32B exec + 72B opt/judge), no framing",
                                       "tab:green",  "^"),
    # 32B baseline had only 4 rounds and no safety sweep completed --- skip if missing
    ("results_qwen32b_baseline/safety_summary.json",
     "32B baseline, no framing",      "tab:orange", "D"),
]

fig, ax = plt.subplots(figsize=(8, 5))

for path, label, color, marker in experiments:
    pts = load_summary(path, label)
    if not pts:
        continue
    xs, ys = zip(*pts)
    ax.plot(xs, [y * 100 for y in ys],
            marker=marker, color=color, label=label, linewidth=2, markersize=8)

ax.set_xlabel("Evolution round")
ax.set_ylabel("Refusal rate (%)")
ax.set_title("Safety trajectory across 4 AFlow evolution experiments\n(RedCode-Gen, Qwen judge)")
ax.set_ylim(0, 35)
ax.grid(True, alpha=0.3)
ax.legend(loc="center right", fontsize=9, framealpha=0.9)

out = ROOT / "results" / "refusal_all4_experiments.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
plt.close()
