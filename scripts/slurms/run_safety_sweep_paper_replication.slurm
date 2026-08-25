#!/bin/bash
#SBATCH --job-name=paper_repl_seed_vs_ensemble
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/uncertainty_misevolution/workflow_misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/uncertainty_misevolution/workflow_misevolution/logs/%j.err

# Paper-matching replication of the Misevolution seed-vs-aflow_agent
# comparison, on the FULL 160-prompt RedCode-Gen dataset.
#
# Two hand-written workflows in workspace_paper_replication/:
#   round_1 = single custom_code_generate call (seed baseline)
#   round_2 = 3-generate + sc_ensemble (paper's aflow_agent_evaluate_model)
#
# Roles:
#   executor -> 72B (port 8000)
#   judge    -> 72B (port 8000)  [shared endpoint, main-branch config]
#
# H2_FRAMING intentionally NOT set. Matches paper's no-framing setup.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/uncertainty_misevolution/workflow_misevolution
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port $PORT \
    --quantization gptq \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > $LOGDIR/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on $PORT..."
UP=0
for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        UP=1; echo "vLLM up on $PORT after ${i} attempts"; break
    fi
    sleep 10
done
if [ $UP -eq 0 ]; then
    echo "ABORT: vLLM startup timed out after 100 minutes"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

cd $REPO

unset H2_FRAMING
echo "H2_FRAMING intentionally UNSET (matches paper's no-framing setup)"

# Build the full-160 subset if not already present
[ -f results/redcode_subset_full160.json ] || python build_redcode_full160.py

# Sanity: show config so we can confirm which endpoints the sweep is hitting
grep -E "^  [a-z]+:|base_url:|model:" src/config/config2.yaml | head -20

python run_safety_sweep_paper_replication.py \
    2>&1 | tee $LOGDIR/paper_repl_sweep_${SLURM_JOB_ID}.log

kill $VLLM_PID
