#!/bin/bash
#SBATCH --job-name=qwen32b_baseline
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# 10-round AFlow evolution on HumanEval, Qwen2.5-32B-Instruct-GPTQ-Int4.
# No safety framing (H2_FRAMING unset). This establishes the 32B baseline
# capability curve to compare against the qwen-32b-safety-seed run.
#
# Writes to a separate workspace (workspace_qwen32b_baseline) and results
# dir (results_qwen32b_baseline/) so nothing existing is touched. Config
# on this branch has both executor and optimizer roles set to the 32B
# model in src/config/config2.yaml.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
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
for i in $(seq 1 180); do
    curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1 && { echo "vLLM up on $PORT"; break; }
    sleep 10
done

cd $REPO

# H2_FRAMING intentionally NOT set. This is the baseline (no framing) at 32B.
unset H2_FRAMING

# 1. evolve 10 rounds
python run_optimize.py \
    --max-rounds 10 \
    --optimized-path workspace_qwen32b_baseline \
    --initial-round 1 \
    2>&1 | tee $LOGDIR/qwen32b_baseline_evolve_${SLURM_JOB_ID}.log

# 2. re-score every round on the held-out HumanEval test set
python run_test_qwen32b.py \
    2>&1 | tee $LOGDIR/qwen32b_baseline_test_${SLURM_JOB_ID}.log

# 3. markdown report
python report_evolution.py \
    --workflows src/workspace_qwen32b_baseline/HumanEval/workflows \
    --out results_qwen32b_baseline \
    --run-id ${SLURM_JOB_ID} \
    2>&1 | tee $LOGDIR/qwen32b_baseline_report_${SLURM_JOB_ID}.log

kill $VLLM_PID
