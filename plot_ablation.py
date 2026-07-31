#!/usr/bin/env python3
"""Plot the two ablation variants side by side, with a summary markdown table."""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
AB = ROOT / "results" / "ablation"


def load(variant):
    cap = {r["round"]: r["score"] for r in json.loads((AB/variant/"capability.json").read_text())
           if r.get("score") is not None}
    safe = json.loads((AB/variant/"safety.json").read_text())
    return cap, safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="latest")
    args = ap.parse_args()

    cc_cap, cc_safe = load("cc")
    ag_cap, ag_safe = load("ag")

    md = ["# Seed ablation summary\n",
          "| Variant | Seed r1 (validate) | Best validate | Seed r1 refusal | Final refusal (best r) | Seed r1 mean harm | Final mean harm |",
          "|:--------|-------------------:|--------------:|----------------:|----------------------:|-------------------:|----------------:|"]
    for label, cap, safe in [("CustomCodeGenerate (cc)", cc_cap, cc_safe),
                              ("AnswerGenerate (ag)",     ag_cap, ag_safe)]:
        seed = next((r for r in safe if r["round"] == 1), None)
        best = max((r for r in safe), key=lambda r: cap.get(r["round"], 0))
        md.append(f"| {label} | {cap.get(1,0):.3f} | {max(cap.values()):.3f} | "
                  f"{seed['refusal_rate']:.3f} | {best['refusal_rate']:.3f} | "
                  f"{seed['mean_score']:.2f} | {best['mean_score']:.2f} |")
    (ROOT/"results"/f"ablation_summary_{args.run_id}.md").write_text("\n".join(md)+"\n")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, (label, cap, safe, col) in zip(
            [a1, a2],
            [("CustomCodeGenerate seed", cc_cap, cc_safe, "tab:blue"),
             ("AnswerGenerate seed",     ag_cap, ag_safe, "tab:purple")]):
            rs = sorted(r["round"] for r in safe)
            ax.plot(rs, [cap.get(r, 0) for r in rs], "o-", color=col, label="validate")
            ax.plot(rs, [next(x for x in safe if x["round"] == r)["refusal_rate"] for r in rs],
                    "s--", color="tab:green", label="refusal rate")
            ax.plot(rs, [next(x for x in safe if x["round"] == r)["mean_score"]/10 for r in rs],
                    "^--", color="tab:red", label="mean harm /10")
            ax.set_title(label); ax.set_xlabel("round"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
        a1.set_ylabel("score (0-1)"); a1.legend(fontsize=8, loc="center right")
        plt.suptitle("Seed ablation: does the AnswerGenerate seed reproduce the paper's misevolution?")
        plt.tight_layout()
        plt.savefig(ROOT/"results"/f"ablation_{args.run_id}.png", dpi=120)
        print(f"[plot_ablation] wrote results/ablation_{args.run_id}.png and summary md")
    except ImportError:
        print("[plot_ablation] matplotlib missing; wrote table only")


if __name__ == "__main__":
    main()
