#!/usr/bin/env python3
"""
Run one ablation variant: evolve a seed workflow, then safety-sweep it.

Args:
  --variant {cc,ag}    cc = CustomCodeGenerate seed (baseline)
                       ag = AnswerGenerate seed (paper-style Answer Generator)
  --rounds 10          keep the sweep cheap; 10 is enough to hit the plateau

Uses a separate workspace per variant (HumanEval / HumanEval_ag) so the two
runs never collide, and writes per-variant results/ablation/<variant>/ output.

Run from repo root (vLLM up):
    python run_ablation_variant.py --variant cc --rounds 10
    python run_ablation_variant.py --variant ag --rounds 10
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC)); os.chdir(SRC)

from scripts.optimizer import Optimizer                                  # noqa: E402
from scripts.async_llm import LLMsConfig                                  # noqa: E402

OPERATORS = ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test", "AnswerGenerate"]

VARIANTS = {
    "cc": {"dataset": "HumanEval",    "label": "CustomCodeGenerate seed"},
    "ag": {"dataset": "HumanEval_ag", "label": "AnswerGenerate seed"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    cfg = VARIANTS[args.variant]
    print(f"[ablation] variant={args.variant} ({cfg['label']}) rounds={args.rounds}")

    models = LLMsConfig.default()
    opt = Optimizer(
        dataset=cfg["dataset"], question_type="code",
        opt_llm_config=models.get("optimizer"),
        exec_llm_config=models.get("executor"),
        operators=OPERATORS, sample=4, check_convergence=False,
        optimized_path="workspace", initial_round=1, max_rounds=args.rounds,
        validation_rounds=1,
    )
    opt.optimize("Graph")

    # collect scores from THIS variant's workspace
    wf = SRC / "workspace" / cfg["dataset"] / "workflows"
    results = json.loads((wf / "results.json").read_text())

    out = ROOT / "results" / "ablation" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "capability.json").write_text(json.dumps(results, indent=2))
    print(f"[ablation] variant={args.variant} wrote {out/'capability.json'} "
          f"({len(results)} rounds)")


if __name__ == "__main__":
    main()
