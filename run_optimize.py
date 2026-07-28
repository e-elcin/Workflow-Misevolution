#!/usr/bin/env python3
"""
Entrypoint: run the AFlow workflow evolution on HumanEval for the
Misevolution reproduction.

This mirrors AFlow's own run.py, with three deliberate, documented changes
for this experiment (see comments inline):
  1. No dataset download  -> we build our own committed HumanEval split.
  2. check_convergence=False -> we want ALL max_rounds to run (the per-round
     trajectory is the point), not an early stop.
  3. validation_rounds=1   -> matches AFlow's run.py default (single pass);
     chosen for fidelity + speed. See project notes.

Model roles come from src/config/config2.yaml (keys "executor" / "optimizer"),
both pointing at the local vLLM server.

Run from the repo root:
    python run_optimize.py                 # auto-resumes from the last finished round
    python run_optimize.py --initial-round 1   # force a fresh start from the seed
"""
import argparse
import os
import sys
from pathlib import Path

# --- path bootstrap: everything resolves relative to src/ ---------------------
# The engine uses relative paths for config ("config/config2.yaml"), data
# ("data/datasets/...") and the workspace ("workspace/..."), and imports the
# packages `scripts`, `benchmarks`, `workspace`. So we put src/ on sys.path and
# make it the working directory, regardless of where this script is launched.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.chdir(SRC)

from scripts.optimizer import Optimizer          # noqa: E402
from scripts.async_llm import LLMsConfig          # noqa: E402

# AFlow's HumanEval operator pool. The seed (round_1) uses only
# CustomCodeGenerate; ScEnsemble is available here so the optimizer *can*
# introduce it — that emergence is exactly what we want to observe.
OPERATORS = ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
DATASET = "HumanEval"
QUESTION_TYPE = "code"


def detect_last_round(optimized_path: str, dataset: str) -> int:
    """Highest round N that has a finished graph.py, for auto-resume."""
    wf = Path(optimized_path) / dataset / "workflows"
    rounds = [
        int(p.parent.name.split("_")[1])
        for p in wf.glob("round_*/graph.py")
        if p.parent.name.startswith("round_")
    ]
    return max(rounds) if rounds else 1


def parse_args():
    p = argparse.ArgumentParser(description="AFlow workflow evolution (HumanEval, Misevolution reproduction)")
    p.add_argument("--optimized-path", default="workspace", help="Where round_N dirs live (relative to src/)")
    p.add_argument("--max-rounds", type=int, default=20, help="Target number of evolution rounds")
    p.add_argument("--validation-rounds", type=int, default=1, help="Scoring passes per round (AFlow run.py default = 1)")
    p.add_argument("--sample", type=int, default=4, help="Candidate rounds considered when picking a parent")
    p.add_argument("--exec-model", default="executor", help="Config key for the executor role")
    p.add_argument("--opt-model", default="optimizer", help="Config key for the optimizer role")
    p.add_argument("--initial-round", type=int, default=None, help="Round to start from; default = auto-resume")
    p.add_argument("--check-convergence", action="store_true", help="Enable early stop (OFF by default on purpose)")
    return p.parse_args()


def main():
    args = parse_args()

    models = LLMsConfig.default()
    opt_llm_config = models.get(args.opt_model)
    exec_llm_config = models.get(args.exec_model)
    for name, cfg in [(args.opt_model, opt_llm_config), (args.exec_model, exec_llm_config)]:
        if cfg is None:
            raise ValueError(f"Model role '{name}' not found under 'models:' in config/config2.yaml")

    initial_round = args.initial_round
    if initial_round is None:
        initial_round = detect_last_round(args.optimized_path, DATASET)
    print(f"[run_optimize] starting at round {initial_round}, target {args.max_rounds}, "
          f"validation_rounds={args.validation_rounds}, check_convergence={args.check_convergence}")

    optimizer = Optimizer(
        dataset=DATASET,
        question_type=QUESTION_TYPE,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=OPERATORS,
        sample=args.sample,
        check_convergence=args.check_convergence,   # False unless flag passed
        optimized_path=args.optimized_path,
        initial_round=initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
    )
    optimizer.optimize("Graph")


if __name__ == "__main__":
    main()