#!/usr/bin/env python3
"""
Compare the two judges (Qwen-72B v2 vs Mistral-Large rejudge) on the same code.

Reads:  results/safety_summary_v2.json      (Qwen-72B judge)
        results/safety_summary_rejudge.json (Mistral-Large judge)
Writes: results/judge_ablation_<run-id>.png  and  results/judge_ablation_<run-id>.md
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    p = ROOT / "results" / name
    return sorted(json.loads(p.read_text()), key=lambda r: r["round"]) if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="latest")
    args = ap.parse_args()

    qwen = load("safety_summary_v2.json")
    mistral = load("safety_summary_rejudge.json")
    if not qwen or not mistral:
        print("[plot] missing one of safety_summary_v2.json / safety_summary_rejudge.json")
        return
    q = {r["round"]: r for r in qwen}
    m = {r["round"]: r for r in mistral}
    rounds = sorted(set(q) & set(m))

    md = ["# Judge ablation: Qwen-72B vs Mistral-Large-123B on identical generations\n",
          "| Round | Qwen mean | Mistral mean | Δ mean | Qwen RR | Mistral RR | Δ RR |",
          "|------:|----------:|-------------:|-------:|--------:|-----------:|-----:|"]
    for r in rounds:
        qm, mm = q[r]["mean_score"], m[r]["mean_score"]
        qr, mr = q[r]["refusal_rate"], m[r]["refusal_rate"]
        md.append(f"| {r} | {qm:.2f} | {mm:.2f} | {mm-qm:+.2f} | "
                  f"{qr:.3f} | {mr:.3f} | {mr-qr:+.3f} |")
    seed_delta_rr = m[1]["refusal_rate"] - q[1]["refusal_rate"] if 1 in q and 1 in m else None
    md.append(f"\n**Seed r1 refusal:** Qwen {q[1]['refusal_rate']:.3f} vs Mistral {m[1]['refusal_rate']:.3f} "
              f"({seed_delta_rr:+.3f})")
    (ROOT / "results" / f"judge_ablation_{args.run_id}.md").write_text("\n".join(md) + "\n")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, key, label in [(a1, "mean_score", "mean harm (0-10)"),
                               (a2, "refusal_rate", "refusal rate")]:
            ax.plot(rounds, [q[r][key] for r in rounds], "o-", label="Qwen-72B judge")
            ax.plot(rounds, [m[r][key] for r in rounds], "s--", label="Mistral-L-123B judge")
            ax.set_xlabel("round"); ax.set_ylabel(label); ax.grid(alpha=.3); ax.legend()
        plt.suptitle("Judge ablation: same generations, two judges")
        plt.tight_layout()
        plt.savefig(ROOT / "results" / f"judge_ablation_{args.run_id}.png", dpi=120)
        print(f"[plot] wrote results/judge_ablation_{args.run_id}.png and .md")
    except ImportError:
        print("[plot] matplotlib missing; wrote table only")


if __name__ == "__main__":
    main()
