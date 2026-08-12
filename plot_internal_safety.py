#!/usr/bin/env python3
"""
Plot the internal-safety experiment's refusal rate trajectory.
Shows the collapse from 100% to ~22% when the optimizer introduces
sc_ensemble at round 5. Vertical annotations mark structural changes
per round.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results_internal_safety" / "safety_summary.json"
OUT = ROOT / "results_internal_safety" / "refusal_internal_safety.png"

# Per-round structural annotations, matching what the workflows actually contain.
# Only rounds where a real structural change happened relative to the previous round.
STRUCTURAL_CHANGES = {
    1: "seed:\nsingle generate\n+ SAFETY_INSTRUCTION",
    4: "+ Test node",
    5: "+ sc_ensemble\n(3-gen + ensemble)",
    7: "+ retry loop",
    8: "+ refine feedback",
    10: "+ full 3-gen\n+ ensemble + Test",
}


def main():
    data = json.loads(SUMMARY.read_text())
    rounds = [r["round"] for r in data]
    refusal_pct = [r["refusal_rate"] * 100 for r in data]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Main trajectory
    ax.plot(rounds, refusal_pct,
            marker="o", color="tab:red", linewidth=2.5, markersize=10,
            label="Refusal rate (SAFETY_INSTRUCTION in mutable graph)",
            zorder=3)

    # Reference line: baseline 72B refusal without any framing (from paper-replication n=160)
    ax.axhline(23.7, color="gray", linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)
    ax.text(10.3, 23.7 + 1.5, "72B no-framing baseline (23.7%)",
            fontsize=9, color="gray", ha="right", va="bottom")

    # Vertical dashed lines + labels for each structural change
    for r, label in STRUCTURAL_CHANGES.items():
        if r not in rounds:
            continue
        ax.axvline(r, color="steelblue", linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)
        # Label positioning: alternate top-anchored and mid-anchored so they don't collide
        y_pos = 92 if r in (1, 5, 8) else 55
        ax.annotate(label, xy=(r, y_pos),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8.5, color="steelblue",
                    ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="steelblue", alpha=0.85))

    # Highlight the misevolution transition
    ax.annotate("",
                xy=(5, 25), xytext=(4, 100),
                arrowprops=dict(arrowstyle="->", color="darkred",
                                lw=2, connectionstyle="arc3,rad=-0.15"))
    ax.text(4.5, 63, "MISEVOLUTION\n-75 pp",
            fontsize=11, color="darkred", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose",
                      edgecolor="darkred", alpha=0.9))

    ax.set_xlabel("Evolution round", fontsize=12)
    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_title("Topological misevolution: SAFETY_INSTRUCTION inside mutable graph\n"
                 "(72B, n=32 RedCode subset, Qwen judge)", fontsize=12)
    ax.set_ylim(-5, 108)
    ax.set_xlim(0.3, 11)
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, framealpha=0.95)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    plt.close()


if __name__ == "__main__":
    main()
