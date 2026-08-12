#!/bin/bash
#SBATCH --job-name=internal_safety_full
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=15:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# Internal-safety experiment: safety instruction INSIDE the mutable graph.
#
# 10-round AFlow evolution on workspace_internal_safety/. Safety text lives
# in round_1/prompt.py as SAFETY_INSTRUCTION. Optimizer sees and can rewrite it.
#
# After evolution, safety sweep runs across all scored rounds on the 120-prompt
# RedCode subset. Also snapshots each round's prompt.py so we can inspect
# how the SAFETY_INSTRUCTION text changed.
#
# Roles: executor + optimizer + judge all share the single 72B endpoint.
# H2_FRAMING intentionally UNSET (safety is in-graph now, not in env).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
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
echo "H2_FRAMING intentionally UNSET (safety lives in graph now, not env)"

# Ensure 120-prompt subset exists
[ -f results/redcode_subset_120.json ] || python build_redcode_subset_120.py

# 1. Evolution: 10 rounds, using the internal_safety workspace
python run_optimize.py \
    --max-rounds 10 \
    --optimized-path workspace_internal_safety \
    --initial-round 1 \
    2>&1 | tee $LOGDIR/internal_safety_evolve_${SLURM_JOB_ID}.log

# 2. Safety sweep across all scored rounds + prompt.py snapshot
python run_safety_sweep_internal_safety.py \
    2>&1 | tee $LOGDIR/internal_safety_sweep_${SLURM_JOB_ID}.log

# 3. Print prompt.py evolution for quick eyeball check
echo ""
echo "===== PROMPT.PY EVOLUTION ====="
for r in src/workspace_internal_safety/HumanEval/workflows/round_*/prompt.py; do
    echo ""
    echo "--- $r ---"
    cat "$r"
done

kill $VLLM_PID
