#!/usr/bin/env python3
"""
Test-set evaluation for the safety-seed evolution run.

Same logic as run_test.py, but with paths pointing at:
    src/workspace_safety_seed/HumanEval/workflows/   (evolved rounds)
    results_safety_seed/                              (test outputs)

Kept as a separate file so run_test.py stays untouched for the baseline.

Run from repo root, after run_optimize.py finishes, while vLLM is still up
and H2_FRAMING is still exported:
    python run_test_safety_seed.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.chdir(SRC)

from scripts.optimizer import Optimizer          # noqa: E402
from scripts.async_llm import LLMsConfig          # noqa: E402

OPERATORS = ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
WORKSPACE = "workspace_safety_seed"
WORKFLOWS = Path(f"{WORKSPACE}/HumanEval/workflows")
RESULTS_DIR_NAME = "results_safety_seed"


def recorded_rounds():
    results = json.loads((WORKFLOWS / "results.json").read_text())
    return [(r["round"], r["score"]) for r in results if r.get("score") is not None]


async def score_round_on_test(opt, round_n):
    directory = opt.graph_utils.create_round_directory(f"{opt.root_path}/workflows", round_n)
    opt.graph = opt.graph_utils.load_graph(round_n, f"{opt.root_path}/workflows")
    score, avg_cost, total_cost = await opt.evaluation_utils.evaluate_graph_test(
        opt, directory, is_test=True)
    return score


def main():
    rounds = recorded_rounds()
    if not rounds:
        print("[run_test_safety_seed] no scored rounds in results.json; nothing to do")
        return

    models = LLMsConfig.default()
    opt = Optimizer(
        dataset="HumanEval", question_type="code",
        opt_llm_config=models.get("optimizer"),
        exec_llm_config=models.get("executor"),
        operators=OPERATORS, sample=4, check_convergence=False,
        optimized_path=WORKSPACE, initial_round=1, max_rounds=1,
        validation_rounds=1,
    )

    combined = []
    for rnd, val in rounds:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        test_score = loop.run_until_complete(score_round_on_test(opt, rnd))
        combined.append({"round": rnd, "validate_score": val, "test_score": test_score})
        print(f"[run_test_safety_seed] round {rnd:>2}: validate {val:.3f} | test {test_score:.3f}")

    out_dir = ROOT / RESULTS_DIR_NAME
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results_test.json").write_text(json.dumps(combined, indent=2))

    val_best = max(combined, key=lambda r: r["validate_score"])
    test_best = max(combined, key=lambda r: r["test_score"])
    summary = (
        f"validate-best (our pick): round {val_best['round']} "
        f"(validate {val_best['validate_score']:.3f}, test {val_best['test_score']:.3f})\n"
        f"test-best (paper pick):   round {test_best['round']} "
        f"(validate {test_best['validate_score']:.3f}, test {test_best['test_score']:.3f})\n"
    )
    print("[run_test_safety_seed]\n" + summary)
    (out_dir / "test_result.txt").write_text(summary)
    print(f"[run_test_safety_seed] wrote {out_dir/'results_test.json'} and {out_dir/'test_result.txt'}")


if __name__ == "__main__":
    main()
