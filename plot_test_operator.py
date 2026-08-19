#!/usr/bin/env python3
"""
Refusal rate per evolution round, colored by whether the round's graph
contains the Test operator. Ensemble rounds are marked separately to show
that the ensemble does not predict refusal.

Reads results_internal_safety_fixed/safety_summary.json and the round graphs.
Writes results_internal_safety_fixed/test_operator_mechanism.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results_internal_safety_fixed" / "safety_summary.json"
WF = ROOT / "src" / "workspace_internal_safety_fixed" / "HumanEval" / "workflows"
OUT = ROOT / "results_internal_safety_fixed" / "test_operator_mechanism.png"


def graph_ops(rnd):
    """Return (has_test, has_ensemble) for a round's graph.py."""
    g = WF / f"round_{rnd}" / "graph.py"
    if not g.exists():
        return None, None
    src = g.read_text()
    return ("self.test(" in src), ("sc_ensemble(" in src)


def main():
    data = json.loads(SUMMARY.read_text())
    rounds, rates, has_test, has_ens = [], [], [], []
    for r in data:
        rnd = r["round"]
        t, e = graph_ops(rnd)
        if t is None:
            continue
        rounds.append(rnd)
        rates.append(r["refusal_rate"] * 100)
        has_test.append(t)
        has_ens.append(e)

    fig, ax = plt.subplots(figsize=(9, 5))

    # connecting line so the trajectory reads as one run
    ax.plot(rounds, rates, color="0.7", linewidth=1.2, zorder=1)

    # split by Test operator
    for t, color, label in [(False, "#2a7f62", "no Test operator"),
                            (True, "#c1392b", "has Test operator")]:
        xs = [r for r, ht in zip(rounds, has_test) if ht == t]
        ys = [v for v, ht in zip(rates, has_test) if ht == t]
        ax.scatter(xs, ys, s=140, color=color, zorder=3,
                   edgecolor="white", linewidth=1.5, label=label)

    # ring the ensemble rounds to show the ensemble is not the driver
    xs = [r for r, e in zip(rounds, has_ens) if e]
    ys = [v for v, e in zip(rates, has_ens) if e]
    ax.scatter(xs, ys, s=320, facecolors="none", edgecolors="#1f4e79",
               linewidth=1.8, zorder=2, label="also has sc_ensemble")

    ax.set_xlabel("evolution round")
    ax.set_ylabel("refusal rate (%)")
    ax.set_title("Refusal collapses exactly when the optimizer adds the Test operator\n"
                 "internal-safety, safety instruction preserved in every round, n=120",
                 fontsize=11)
    ax.set_ylim(-5, 108)
    ax.set_xticks(rounds)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="center right")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")

    # also print the table that backs the figure
    print(f"\n{'round':>6} | {'refusal':>8} | {'Test':>5} | {'ensemble':>8}")
    print("-" * 38)
    for rnd, v, t, e in zip(rounds, rates, has_test, has_ens):
        print(f"{rnd:>6} | {v:>7.1f}% | {'yes' if t else 'no':>5} | {'yes' if e else 'no':>8}")


if __name__ == "__main__":
    main()
