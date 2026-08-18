#!/usr/bin/env python3
"""
Collect repeated workflow runs for MATU uncertainty analysis.
For each RedCode prompt, run the workflow N times at higher temperature and
save request + response text per run. No judging. Output: run_0.json ... run_{N-1}.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "RedCode" / "evaluation"))
os.chdir(SRC)

from scripts.async_llm import LLMsConfig                                # noqa: E402
from scripts.optimizer_utils.graph_utils import GraphUtils              # noqa: E402
from RedCode_Gen.prompts import CODE_GENERATION_PROMPT                  # noqa: E402
from RedCode_Gen.utils import (read_prompt,                             # noqa: E402
                               extract_entry_point_from_signature)


async def run_once(wf, problem, entry_point):
    try:
        out = await wf(problem=problem, entry_point=entry_point)
        return out[0] if isinstance(out, tuple) else out
    except Exception:
        sol = await wf.custom_code_generate(
            problem=problem, entry_point=entry_point, instruction="")
        return sol["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--subset", default="results/redcode_subset.json")
    ap.add_argument("--workspace", default="workspace_internal_safety/HumanEval")
    ap.add_argument("--out-dir", default="results_matu/internal_safety")
    ap.add_argument("--config", default=None,
                    help="explicit config yaml (absolute path recommended)")
    args = ap.parse_args()

    subset_path = ROOT / args.subset
    dataset = SRC / "RedCode" / "dataset" / "RedCode-Gen"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = json.loads(subset_path.read_text())

    if args.config:
        import yaml
        cfg_data = yaml.safe_load(Path(args.config).read_text())
        if "models" in cfg_data:
            cfg_data = cfg_data["models"]
        models = LLMsConfig(cfg_data)
    else:
        models = LLMsConfig.default()
    exec_cfg = models.get("executor")
    print(f"executor temperature = {exec_cfg.temperature}")

    wf_cls = GraphUtils(args.workspace).load_graph(
        args.round, f"{args.workspace}/workflows")

    runs = [dict() for _ in range(args.n_runs)]
    for i, item in enumerate(subset, 1):
        pid = f"{item['family']}__{item['file'].replace('.py','')}"
        path = dataset / item["family"] / item["file"]
        sig = read_prompt(str(path))
        problem = CODE_GENERATION_PROMPT + sig
        entry_point = extract_entry_point_from_signature(sig)
        for k in range(args.n_runs):
            wf = wf_cls(name=f"round_{args.round}_run_{k}",
                        llm_config=exec_cfg, dataset="HumanEval")
            try:
                resp = asyncio.run(run_once(wf, problem, entry_point))
            except Exception as e:
                resp = f"[RUN ERROR: {e}]"
            runs[k][pid] = {"request": problem, "response": resp}
            print(f"  [{i}/{len(subset)}] {pid} run {k+1}/{args.n_runs}")

    for k in range(args.n_runs):
        (out_dir / f"run_{k}.json").write_text(json.dumps(runs[k], indent=2))
    print(f"\nwrote {args.n_runs} run files to {out_dir}")


if __name__ == "__main__":
    main()
