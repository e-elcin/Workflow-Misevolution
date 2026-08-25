# Workflow Misevolution — Reproduction and Extension

This repo tests a claim from Shao et al., "Misevolution in Self-Evolving Agents"
(ICLR 2026). The paper claims that when AFlow optimizes an agent workflow, the
agent's safety behavior gets worse over the rounds of optimization. I tested that
claim on the METU cluster using open-weight Qwen models.

Paper repo: https://github.com/ShaoShuai0605/Misevolution

## Setup

- Models: Qwen2.5-72B and Qwen2.5-32B (GPTQ-Int4), served locally with vLLM.
  Mistral-Large-123B (AWQ) is used only for the judge test.
- Conda env: /dl_scratch2/ege/envs/misevolution_agent_safety
- Model cache: /dl_scratch2/ege/hf_cache
- Safety benchmark: RedCode-Gen, 160 prompts across 8 malware families.

## What I changed in AFlow

These are the edits needed to run AFlow on my setup and to support the safety
tests. They all live in `src/scripts/`.

1. **Workspace path fix.** AFlow hardcodes the workspace name as `workspace` in
   its import template. I changed it to use the actual workspace name, so I can
   have several named workspaces side by side. The change is backwards compatible.

2. **Workspace layout.** AFlow needs a `template/` folder, an `__init__.py` in
   every folder, and a starting `round_1/graph.py` and `prompt.py`. The paper repo
   ships none of these, so I built them following AFlow's own conventions.

3. **Safety framing hook.** In `operators.py`, the code generator reads an
   optional `H2_FRAMING` environment variable and adds it to the front of the
   prompt. This lets me inject a safety instruction without editing the workflow.
   If the variable is empty or unset, the behavior is the same as upstream.

4. **Separate judge model.** The judge is chosen from a `judge` role in the config
   if one exists, and falls back to the executor model otherwise. This lets me
   swap the judge model on its own, which I use in the judge test.

## What the paper repo gives me, and what it does not

It gives me: the AFlow optimizer, the operators, the optimization prompts, and the
RedCode-Gen dataset. Their aflow_agent evaluation runs one model at temperature
0.1, generates three times, then combines the results with sc_ensemble. The judge
is GPT-4o (named only in a shell script).

It does not give me: a starting workspace, a starting graph, a starting prompt, a
config file, a run script, or any hyperparameter values. The README is empty. The
paper describes the starting workflow as a "single-step Answer Generator" but does
not include the file. I reconstructed all of this myself.

## Included experiments

This contribution contains two complementary all-72B experiments. The executor,
optimizer, and safety-judge roles use `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`.

### End-to-end HumanEval evolution and RedCode sweep

`src/workspace/` contains the no-framing HumanEval evolution trajectory. It
stores the initial workflow and optimization attempts through `round_21`.
Eighteen workflow states were evaluated by the RedCode sweep.

```text
seed workflow -> HumanEval optimization -> HumanEval testing -> RedCode sweep
```

The HumanEval outputs are under `results/`. Per-state RedCode outputs are under
`results/safety/`, with the combined trajectory in `results/safety_summary.json`.

Run the stages in order:

```bash
sbatch scripts/slurms/run_optimize.slurm
sbatch scripts/slurms/run_safety.slurm
```

### Full-160 endpoint comparison

`src/workspace_paper_replication/` contains a single-generation seed and a
three-generation-plus-ensemble workflow. This is a fixed endpoint comparison,
not an evolution run. Both workflows are evaluated on all 160 RedCode prompts.

```bash
sbatch scripts/slurms/run_safety_sweep_paper_replication.slurm
```

Results are stored in `results_paper_replication/`.

### Dataset requirement

Place the upstream RedCode-Gen dataset at:

```text
src/RedCode/dataset/RedCode-Gen/
```

The external dataset is not duplicated here. Historical result files are
preserved. Corrected runners record workflow failures as errors instead of
silently substituting a simpler workflow.

## Things to watch out for

- **Slow model loading.** The 72B weights can take more than 30 minutes to load
  from NFS. The Slurm wait loops allow up to 100 minutes.
- **All-refuse rounds.** Some rounds score exactly 0 on every prompt, meaning 100
  percent refusal. Check these carefully; the model may have returned empty or
  broken output rather than a real refusal.


