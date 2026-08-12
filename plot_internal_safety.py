#!/usr/bin/env python3
"""
Plot the internal-safety experiment's refusal rate trajectory.
Shows the collapse from 100% to ~22% when the optimizer introduces
sc_ensemble at round 5.

Layout: annotations sit BELOW the plot in a stripe, not on top of it,
so nothing overlaps the trajectory or the reference lines.
"""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results_internal_safety" / "safety_summary.json"
OUT = ROOT / "results_internal_safety" / "refusal_internal_safety.png"

STRUCTURAL_CHANGES = {
    1:  "seed:\nsingle generate\n(SAFETY_INSTRUCTION)",
    4:  "+ Test\nnode",
    5:  "+ sc_ensemble\n(3-gen → ensemble)",
    7:  "+ retry\nloop",
    8:  "+ refine\nfeedback",
    10: "+ full 3-gen\n+ ensemble\n+ Test",
}


def main():
    data = json.loads(SUMMARY.read_text())
    rounds = [r["round"] for r in data]
    refusal_pct = [r["refusal_rate"] * 100 for r in data]

    # Two stacked axes: main plot on top, annotation stripe below
    fig, (ax, ax_ann) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
        sharex=True,
    )

    # ---------- main plot ----------
    # Shade the "safety intact" region above the no-framing baseline (23.7)
    ax.axhspan(23.7, 105, facecolor="tab:green", alpha=0.07, zorder=0)
    ax.axhspan(-5, 23.7, facecolor="tab:red", alpha=0.05, zorder=0)

    # Vertical guides (light)
    for r in STRUCTURAL_CHANGES:
        ax.axvline(r, color="steelblue", linestyle="--", linewidth=1, alpha=0.35, zorder=1)

    # Reference: 72B no-framing baseline
    ax.axhline(23.7, color="dimgray", linestyle=":", linewidth=1.5, alpha=0.85, zorder=2)
    ax.text(10.4, 25.5, "72B no-framing baseline (23.7%)",
            fontsize=9, color="dimgray", ha="right", va="bottom", style="italic")

    # Trajectory
    ax.plot(rounds, refusal_pct,
            marker="o", color="tab:red", linewidth=2.8, markersize=11,
            markeredgecolor="darkred", markeredgewidth=1.2,
            label="SAFETY_INSTRUCTION inside mutable graph",
            zorder=4)

    # Point labels
    for r, pct in zip(rounds, refusal_pct):
        va = "bottom" if pct < 90 else "top"
        offset = 4 if pct < 90 else -6
        ax.annotate(f"{pct:.1f}%",
                    xy=(r, pct), xytext=(0, offset),
                    textcoords="offset points",
                    ha="center", va=va,
                    fontsize=9.5, fontweight="bold", color="darkred")

    # Misevolution annotation
    ax.annotate("",
                xy=(5, 27), xytext=(4, 100),
                arrowprops=dict(arrowstyle="->", color="darkred",
                                lw=2.2, connectionstyle="arc3,rad=-0.25"))
    ax.text(4.5, 63, "MISEVOLUTION\n−75 pp",
            fontsize=13, color="darkred", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose",
                      edgecolor="darkred", linewidth=1.5, alpha=0.95))

    # Zone labels
    ax.text(0.5, 60, "safety active\n(refusals above baseline)",
            fontsize=9, color="darkgreen", alpha=0.7, style="italic",
            ha="left", va="center")
    ax.text(0.5, 10, "safety inactive\n(at or below baseline)",
            fontsize=9, color="darkred", alpha=0.6, style="italic",
            ha="left", va="center")

    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_title("Topological misevolution: SAFETY_INSTRUCTION inside mutable graph\n"
                 "72B executor, n=32 RedCode-Gen subset, Qwen judge",
                 fontsize=12)
    ax.set_ylim(-5, 108)
    ax.set_xlim(0.3, 10.7)
    ax.grid(True, alpha=0.25, zorder=1)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # Match x-ticks on main plot to structural change rounds
    ax.set_xticks(rounds)

    # ---------- annotation stripe ----------
    ax_ann.set_xlim(0.3, 10.7)
    ax_ann.set_ylim(0, 1)
    ax_ann.set_yticks([])
    ax_ann.set_xticks(rounds)
    ax_ann.set_xlabel("Evolution round", fontsize=12)
    ax_ann.spines["top"].set_visible(False)
    ax_ann.spines["left"].set_visible(False)
    ax_ann.spines["right"].set_visible(False)

    for r, label in STRUCTURAL_CHANGES.items():
        # tick mark connecting to top
        ax_ann.plot([r, r], [0.85, 1], color="steelblue", linewidth=1.2, alpha=0.6)
        ax_ann.annotate(label, xy=(r, 0.75), ha="center", va="top",
                        fontsize=9, color="steelblue",
                        bbox=dict(boxstyle="round,pad=0.35",
                                  facecolor="aliceblue",
                                  edgecolor="steelblue",
                                  linewidth=1, alpha=0.9))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    plt.close()


if __name__ == "__main__":
    main()
