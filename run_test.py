#!/usr/bin/env python3
"""Final test-set evaluation (paper's 'best workflow on HumanEval test set')."""
import asyncio, os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC)); os.chdir(SRC)
from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig
OPERATORS = ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
WORKFLOWS = Path("workspace/HumanEval/workflows")
def best_round():
    results = json.loads((WORKFLOWS / "results.json").read_text())
    scored = [r for r in results if r.get("score") is not None]
    best = max(scored, key=lambda r: r["score"])
    return best["round"], best["score"]
async def score_on_test(opt, round_n):
    directory = opt.graph_utils.create_round_directory(f"{opt.root_path}/workflows", round_n)
    opt.graph = opt.graph_utils.load_graph(round_n, f"{opt.root_path}/workflows")
    score, avg_cost, total_cost = await opt.evaluation_utils.evaluate_graph_test(opt, directory, is_test=True)
    return score, total_cost
def main():
    rnd, val_score = best_round()
    print(f"[run_test] best round on validate = {rnd} (validate score {val_score:.3f})")
    models = LLMsConfig.default()
    opt = Optimizer(dataset="HumanEval", question_type="code",
        opt_llm_config=models.get("optimizer"), exec_llm_config=models.get("executor"),
        operators=OPERATORS, sample=4, check_convergence=False,
        optimized_path="workspace", initial_round=rnd, max_rounds=1, validation_rounds=1)
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    test_score, cost = loop.run_until_complete(score_on_test(opt, rnd))
    print(f"[run_test] round {rnd} TEST-set score (44 held-out) = {test_score:.3f}")
    out = ROOT / "results" / "test_result.txt"; out.parent.mkdir(exist_ok=True)
    out.write_text(f"best_round={rnd}\nvalidate_score={val_score:.4f}\ntest_score={test_score:.4f}\ncost={cost}\n")
    print(f"[run_test] wrote {out}")
if __name__ == "__main__":
    main()
