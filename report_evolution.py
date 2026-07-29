#!/usr/bin/env python3
"""
Readable summary of an AFlow evolution run.

Reads the per-round outputs the optimizer writes under
    src/workspace/HumanEval/workflows/
and produces, in plain language:
  * a per-round table: score, change vs parent, whether it was kept, and the
    workflow structure (how many generate nodes, ensemble?, test?, loop?),
  * the plain-English modification the optimizer logged each round,
  * the key structural events (first time an ensemble / test node appears),
  * the best-scoring lineage from the seed to the top round.

Writes a markdown report and (if matplotlib is available) a score-vs-round plot.

Run from the repo root:
    python report_evolution.py
    python report_evolution.py --workflows src/workspace/HumanEval/workflows --out results
"""
import argparse
import json
import re
from pathlib import Path


def load_results(wf: Path):
    p = wf / "results.json"
    if not p.exists():
        return {}
    return {r["round"]: r for r in json.loads(p.read_text())}


def load_experience(round_dir: Path):
    p = round_dir / "experience.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def structure(round_dir: Path):
    """Human-readable fingerprint of a round's workflow graph."""
    g = round_dir / "graph.py"
    src = g.read_text() if g.exists() else ""
    n_gen = len(re.findall(r"self\.custom_code_generate\(", src)) or \
        len(re.findall(r"custom_code_generate", src))
    has_ens = bool(re.search(r"sc_ensemble|ScEnsemble", src))
    has_test = bool(re.search(r"self\.test\(|\bTest\(", src))
    has_review = bool(re.search(r"self\.review|Review\(", src))
    has_loop = bool(re.search(r"for .* in range\(", src))
    parts = []
    if n_gen:
        parts.append(f"{n_gen}x generate")
    if has_ens:
        parts.append("ensemble")
    if has_test:
        parts.append("test")
    if has_review:
        parts.append("review")
    if has_loop:
        parts.append("loop")
    return parts or ["(single generate)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflows", default="src/workspace/HumanEval/workflows")
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-id", default="latest", help="Slurm job id, to keep per-run reports")
    args = ap.parse_args()

    wf = Path(args.workflows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = load_results(wf)
    round_dirs = sorted(
        (d for d in wf.glob("round_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[1]),
    )

    rows = []
    for d in round_dirs:
        n = int(d.name.split("_")[1])
        exp = load_experience(d)
        score = results.get(n, {}).get("score")
        rows.append({
            "round": n,
            "score": score,
            "parent": exp.get("father node") if exp else None,
            "before": exp.get("before") if exp else None,
            "after": exp.get("after") if exp else score,
            "succeed": exp.get("succeed") if exp else None,
            "modification": exp.get("modification") if exp else "(seed workflow)",
            "structure": structure(d),
        })

    # ---- build markdown ----
    md = ["# Evolution run summary\n"]
    scored = [r for r in rows if r["score"] is not None]
    if scored:
        best = max(scored, key=lambda r: r["score"])
        md.append(f"- Rounds recorded: **{len(rows)}**")
        md.append(f"- Best score: **{best['score']:.3f}** at round **{best['round']}**")
        md.append(f"- Seed (round 1) score: "
                  f"**{results.get(1, {}).get('score', float('nan')):.3f}**\n")

    # structural first-appearances
    def first_with(tag):
        for r in rows:
            if any(tag in s for s in r["structure"]):
                return r["round"]
        return None
    events = []
    for tag, label in [("ensemble", "ensemble node"), ("test", "test node"),
                       ("loop", "retry/loop")]:
        rd = first_with(tag)
        if rd:
            events.append(f"- **{label}** first appears at round **{rd}**")
    if events:
        md.append("## Key structural events\n" + "\n".join(events) + "\n")

    # per-round table
    md.append("## Per-round detail\n")
    md.append("| Round | Parent | Score | Δ vs parent | Kept? | Structure | What changed |")
    md.append("|------:|-------:|------:|:-----------:|:-----:|:----------|:-------------|")
    for r in rows:
        sc = f"{r['score']:.3f}" if r["score"] is not None else "—"
        if r["before"] is not None and r["after"] is not None:
            delta = f"{r['after'] - r['before']:+.3f}"
        else:
            delta = "—"
        kept = {True: "yes", False: "no", None: "—"}[r["succeed"]]
        struct = ", ".join(r["structure"])
        mod = (r["modification"] or "").replace("|", "\\|")
        md.append(f"| {r['round']} | {r['parent'] if r['parent'] is not None else '—'} "
                  f"| {sc} | {delta} | {kept} | {struct} | {mod} |")

    report = out / f"evolution_report_{args.run_id}.md"
    report.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[report_evolution] wrote {report}")
    update_index(out, args.run_id, rows, scored)

    # ---- optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = sorted(((r["round"], r["score"]) for r in scored), key=lambda x: x[0])
        xs, ys = zip(*pts)
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o")
        for tag, color in [("ensemble", "tab:orange"), ("test", "tab:red")]:
            rd = first_with(tag)
            if rd:
                plt.axvline(rd, ls="--", color=color, alpha=.7, label=f"{tag} appears (r{rd})")
        plt.xlabel("round"); plt.ylabel("validate pass rate")
        plt.title("HumanEval score across evolution rounds")
        plt.ylim(0, 1); plt.grid(alpha=.3); plt.legend()
        fig = out / f"evolution_scores_{args.run_id}.png"
        plt.tight_layout(); plt.savefig(fig, dpi=120)
        print(f"[report_evolution] wrote {fig}")
    except ImportError:
        print("[report_evolution] matplotlib not installed; skipped plot")



def update_index(out, run_id, rows, scored):
    idx = out / "INDEX.md"
    best = max(scored, key=lambda r: r["score"]) if scored else None
    b = f"{best['score']:.3f} (r{best['round']})" if best else "-"
    line = (f"| {run_id} | {len(rows)} | {b} | "
            f"[report](evolution_report_{run_id}.md) \u00b7 "
            f"[plot](evolution_scores_{run_id}.png) |\n")
    if not idx.exists():
        idx.write_text("# Evolution runs index\n\n"
                       "| Run ID | Rounds | Best score | Files |\n"
                       "|:-------|-------:|:-----------|:------|\n")
    with idx.open("a") as f:
        f.write(line)


if __name__ == "__main__":
    main()
