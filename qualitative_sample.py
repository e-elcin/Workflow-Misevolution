#!/usr/bin/env python3
"""
Qualitative evaluation: seed workflow vs aflow_agent workflow.

Runs one RedCode-Gen prompt through both workflows using the exact same
operators, LLM endpoint, and prompt plumbing the safety sweep uses, then
writes the raw text of every generation to a file.

Seed workflow      = 1x custom_code_generate            (mirrors round_1/graph.py)
AFlow_agent workflow = 3x custom_code_generate + sc_ensemble (mirrors round_2/graph.py)

Run from repo root on the cluster with the vLLM executor already served:
    conda activate /dl_scratch2/ege/envs/misevolution_agent_safety
    python qualitative_sample.py
"""
import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: identical to run_safety_round_paper_replication.py so that the
# `scripts.*` and `workspace_paper_replication.*` packages resolve and the
# operators behave byte-for-byte like the sweep. os.chdir(SRC) is required
# because the template operator module and config loader resolve relative to
# src being the top of the package namespace.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # computed BEFORE chdir
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "RedCode" / "evaluation"))
os.chdir(SRC)

from scripts.async_llm import LLMsConfig, create_llm_instance          # noqa: E402
# Import the operator module the workflow graphs actually use, so the H2/
# instruction handling and the SC_ENSEMBLE prompt are exactly what the sweep ran.
import workspace_paper_replication.HumanEval.workflows.template.operator as operator  # noqa: E402
from RedCode_Gen.prompts import CODE_GENERATION_PROMPT                  # noqa: E402
from RedCode_Gen.utils import (read_prompt,                            # noqa: E402
                               extract_entry_point_from_signature)

# ---------------------------------------------------------------------------
# Which prompt to inspect. Families available: adware, ddos, others,
# ransomware, rootkit, spyware, trojan, virus (20 files each).
# ---------------------------------------------------------------------------
SAMPLE_FAMILY = "adware"
SAMPLE_FILE = "adware_1_email.py"

DATASET = ROOT / "src" / "RedCode" / "dataset" / "RedCode-Gen"
OUT_FILE = ROOT / "results" / "qualitative_sample_output.txt"


def load_sample_prompt():
    path = DATASET / SAMPLE_FAMILY / SAMPLE_FILE
    if not path.exists():
        print(f"Error: could not find {path}")
        sys.exit(1)
    return read_prompt(str(path))


async def main():
    signature_text = load_sample_prompt()
    problem = CODE_GENERATION_PROMPT + signature_text
    entry_point = extract_entry_point_from_signature(signature_text)

    # -----------------------------------------------------------------------
    # Build ONE executor LLM the correct way: from config2.yaml via LLMsConfig,
    # not from a hand-built dict. This is the endpoint the sweep hits (local
    # vLLM). Both workflows share it, so the only difference between them is
    # the workflow structure, which is the whole point of the comparison.
    # -----------------------------------------------------------------------
    print("Loading executor config and building LLM...")
    models = LLMsConfig.default()
    exec_cfg = models.get("executor")
    llm = create_llm_instance(exec_cfg)

    custom_code_generate = operator.CustomCodeGenerate(llm)
    sc_ensemble = operator.ScEnsemble(llm)

    print("\n" + "=" * 60)
    print(f"PROMPT ({SAMPLE_FAMILY}/{SAMPLE_FILE}):")
    print("=" * 60)
    print(signature_text.strip())

    # -----------------------------------------------------------------------
    # 1. SEED WORKFLOW: single custom_code_generate (round_1/graph.py)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SEED WORKFLOW (1x custom_code_generate)")
    print("=" * 60)
    try:
        s = await custom_code_generate(problem=problem, entry_point=entry_point,
                                       instruction="")
        seed_response = s["response"]
        print("-> seed generation done")
    except Exception as e:
        seed_response = f"ERROR GENERATING SEED: {e}"
        print(seed_response)

    # -----------------------------------------------------------------------
    # 2. AFLOW_AGENT WORKFLOW: 3x generate + sc_ensemble (round_2/graph.py)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AFLOW_AGENT WORKFLOW (3x custom_code_generate + sc_ensemble)")
    print("=" * 60)
    generations = []
    for i in range(3):
        print(f"--- candidate {i + 1} of 3 ---")
        try:
            s = await custom_code_generate(problem=problem,
                                           entry_point=entry_point,
                                           instruction="")
            generations.append(s["response"])
        except Exception as e:
            generations.append(f"ERROR: {e}")
            print(f"error on candidate {i + 1}: {e}")

    print("--- sc_ensemble selection ---")
    try:
        best = await sc_ensemble(solutions=generations, problem=problem)
        final_aflow_response = best["response"]
        print("-> ensemble selection done")
    except Exception as e:
        final_aflow_response = f"ERROR ENSEMBLE: {e}"
        print(final_aflow_response)

    # sc_ensemble returns one of the three candidates verbatim (no synthesis),
    # so report which one it picked when we can find an exact match.
    picked_idx = None
    for idx, g in enumerate(generations):
        if g == final_aflow_response:
            picked_idx = idx + 1
            break

    # -----------------------------------------------------------------------
    # 3. WRITE OUTPUT
    # -----------------------------------------------------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write(f"=== {SAMPLE_FAMILY}/{SAMPLE_FILE} ===\n\n")
        f.write("PROMPT:\n")
        f.write(signature_text.strip() + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("SEED WORKFLOW RESPONSE (1x generate):\n")
        f.write("=" * 60 + "\n")
        f.write(seed_response + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("AFLOW_AGENT CANDIDATES (3x generate):\n")
        f.write("=" * 60 + "\n")
        for i, g in enumerate(generations):
            f.write(f"--- candidate {i + 1} of 3 ---\n{g}\n\n")

        f.write("=" * 60 + "\n")
        if picked_idx is not None:
            f.write(f"AFLOW_AGENT FINAL SELECTION (sc_ensemble picked candidate {picked_idx}):\n")
        else:
            f.write("AFLOW_AGENT FINAL SELECTION (sc_ensemble):\n")
        f.write("=" * 60 + "\n")
        f.write(final_aflow_response + "\n")

    print(f"\nSaved qualitative output to {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
