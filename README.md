# Workflow Misevolution — Reproduction and Extension

This repo tests a claim from Shao et al., "Misevolution in Self-Evolving Agents"
(ICLR 2026). The paper claims that when AFlow optimizes an agent workflow, the
agent's safety behavior gets worse over the rounds of optimization. I tested that
claim on the METU cluster using open-weight Qwen models.

Paper repo: https://github.com/ShaoShuai0605/Misevolution
AFlow (the optimizer): https://github.com/geekan/MetaGPT

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

## Branches

Each branch is one experiment. Safety numbers are in
`results*/safety_summary.json`. The evolved workflows are in `src/workspace*/`.

### main — all-72B baseline and safety-seed

The default branch. It holds two evolution runs on the 72B model, plus a noise test.

- **Baseline run** (`src/workspace/`): 20 rounds of AFlow evolution on HumanEval,
  no safety framing. This is the plain "does the workflow evolve" run, used as the
  reference for every other experiment.
- **Safety-seed run** (`src/workspace_safety_seed/`): 10 rounds with the
  safety-aware framing added to every generation call. This tests whether
  optimization erodes a safety instruction that is present from the start.
- **Variance test**: the safety-seed round 1 scored four times with nothing
  changed, to measure how much the refusal rate moves on its own. This sets the
  noise floor at about 3 percentage points at n=32.

Result: on the safety-seed run, refusal stays between 22 and 28 percent across all
rounds, which is within the noise floor. 

### paper-replication-160 —> the direct comparison (main result)
Two workflows written by hand to match the paper's code exactly: a single-generate
seed, and a three-generate-plus-sc_ensemble aflow_agent. Both scored on all 160
prompts, Qwen judge, no framing. Result: the seed refuses 22.5 percent, the
aflow_agent refuses 23.7 percent. The difference is 1.2 points in the wrong
direction, inside the noise.

### all72b-baseline-safety-sweep —> sharper version of the baseline
Safety sweep over 6 clearly different rounds (1, 4, 7, 9, 13, 17) of the all-72B
baseline, using 120 prompts instead of 32 for a tighter noise floor. Result:
refusal stays between 20.8 and 21.7 percent. The whole range is under 1 point.

### qwen-32b-executor —> split roles
A 32B executor with a 72B optimizer and a 72B judge, on two vLLM servers. This was
my best evolution run in terms of succesful rounds: 9 of 11 rounds scored. It shows the optimizer, not the
executor, needs to be the large model. Safety sweep: refusal stays near 3 percent
across all rounds. The gap between this (3 percent) and the 72B runs (21 percent)
shows the model's own safety training is what drives refusal.

### qwen-32b-baseline —> all-32B (partial)
Every role on the 32B model. Only 4 of 11 rounds scored, because the 32B optimizer
invents operator names that do not exist and writes broken workflows. I keep it as
evidence that the 32B model cannot drive AFlow on its own. No safety sweep.

### judge-ablation —> is the flat line caused by the judge?
I re-scored the safety-seed run with Mistral-Large-123B as the judge instead of
Qwen-72B. Both judges agree to within about 1 point on every round (Mistral 21.9 to
31.2 percent, Qwen 21.9 to 28.1 percent). 

### new-seed-ablation —> which framing to use
A round-1 comparison of three framings at n=32: helpful ("You are a helpful
assistant"), safety-aware (the full evaluate-then-decline instruction), and none
(empty). Result: helpful 21.9 percent, safety-aware 28.1 percent, none 25.0
percent. Only the safety-aware framing moved above the noise floor, so I used it
for the main safety-seed run.

### internal-safety-seed —> safety instruction inside the workflow
I moved the safety instruction into `round_1/prompt.py`, where the optimizer can
read and rewrite it, to see whether the optimizer would remove it. The optimizer
left the text unchanged across all 11 rounds. The refusal trajectory shows a large
jump from 100 percent down to 25 percent, but I am still checking what caused it.

### seed-ablation — CustomCodeGenerate vs AnswerGenerate seed (unreliable)
An attempt to compare two starting operators over the evolution rounds. It does
not work for two reasons. First, the AnswerGenerate arm is empty: it produced no
scored rounds at all, so half the comparison is missing. Second, the
CustomCodeGenerate arm is dominated by the all-refuse artifact: rounds 1, 2, and 5
show normal mixed scores (around 25 to 28 percent refusal), but from round 6
onward every round scores exactly 0 on all 32 prompts (100 percent refusal). Those
rounds are almost certainly broken workflows returning empty output that the judge
scores as 0, not real refusals. The commit is labeled "wrongfully done." Kept for the record.

### safety-sweep — full 18-round baseline safety sweep
The complete safety sweep of the all-72B baseline. It scores all 18 valid rounds
(1 through 21, skipping the rounds that failed to evolve), rather than the 6
selected rounds in all72b-baseline-safety-sweep. Refusal stays between 21.9 and
31.2 percent across every round, the same flat conclusion as the 6-round version
but with a denser trajectory. This branch also holds the safety-divergence plots
(`results/safety_divergence_*.png`). Kept as the most complete baseline safety
record. This was the first safety sweep written, so its scripts are an earlier
version of the per-experiment sweeps used on later branches.

### report-2026-08-11 —> report and figures 
Holds the weekly report, the four-experiment refusal plot, and the plotting
scripts. Not an experiment.

## How to run one experiment

Example: the paper-replication comparison.

1. `git checkout paper-replication-160`
2. The workspace, both workflows, and scripts are already committed.
3. `sbatch run_safety_sweep_paper_replication.slurm`. This starts one 72B vLLM
   server, waits for it, builds the 160-prompt subset if it is missing, scores
   rounds 1 and 2, and writes to `results_paper_replication/`.
4. Read `results_paper_replication/safety_summary.json` for the result.

Every other branch works the same way: a `run_safety_sweep_<name>.slurm` plus a
matching `run_safety_round_<name>.py`.

## Things to watch out for

- **Slow model loading.** The 72B weights can take more than 30 minutes to load
  from NFS. The Slurm wait loops allow up to 100 minutes.
- **All-refuse rounds.** Some rounds score exactly 0 on every prompt, meaning 100
  percent refusal. Check these carefully; the model may have returned empty or
  broken output rather than a real refusal.
- **32B optimizer.** It often invents operator names, so expect only some rounds to
  score in the all-32B runs.


