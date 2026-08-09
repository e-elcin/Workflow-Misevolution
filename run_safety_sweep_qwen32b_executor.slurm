#!/bin/bash
#SBATCH --job-name=qwen32b_exec_safety_sweep
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# Safety sweep on the qwen-32b-executor (no framing) run.
# Reads workflows from src/workspace_qwen32b_executor/HumanEval/workflows/,
# scores each recorded round on the RedCode-Gen subset, writes per-round
# metrics to results_qwen32b_executor/safety/ and a combined summary to
# results_qwen32b_executor/safety_summary.json.
#
# Roles:
#   executor  -> 32B (port 8001)  — runs the evolved workflows
#   judge     -> 72B (port 8000)  — scores the resulting solutions
#
# H2_FRAMING is intentionally NOT set. This is the no-framing safety
# reference trajectory at the split-role setup.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

# ---- launch 72B on GPU 0, port 8000 (judge) ----
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4" \
    --served-model-name "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4" \
    --port 8000 \
    --quantization gptq \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    > $LOGDIR/vllm72_${SLURM_JOB_ID}.log 2>&1 &
VLLM72_PID=$!

# ---- launch 32B on GPU 1, port 8001 (executor) ----
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4" \
    --served-model-name "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4" \
    --port 8001 \
    --quantization gptq \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    > $LOGDIR/vllm32_${SLURM_JOB_ID}.log 2>&1 &
VLLM32_PID=$!

echo "Waiting for both vLLM servers..."
UP72=0; UP32=0
for i in $(seq 1 600); do
    if [ $UP72 -eq 0 ] && curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        UP72=1; echo "72B up on 8000"
    fi
    if [ $UP32 -eq 0 ] && curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        UP32=1; echo "32B up on 8001"
    fi
    if [ $UP72 -eq 1 ] && [ $UP32 -eq 1 ]; then break; fi
    sleep 10
done

if [ $UP72 -eq 0 ] || [ $UP32 -eq 0 ]; then
    echo "ABORT: vLLM startup timed out (72B up=$UP72, 32B up=$UP32)"
    kill $VLLM72_PID $VLLM32_PID 2>/dev/null
    exit 1
fi

cd $REPO

# Explicitly ensure H2_FRAMING is NOT set — this is the no-framing baseline sweep
unset H2_FRAMING
echo "H2_FRAMING intentionally UNSET (no-framing sweep)"

# shared RedCode subset (should already exist from earlier sweeps)
[ -f results/redcode_subset.json ] || python build_redcode_subset.py

python run_safety_sweep_qwen32b_executor.py \
    2>&1 | tee $LOGDIR/qwen32b_exec_safety_sweep_${SLURM_JOB_ID}.log

kill $VLLM72_PID $VLLM32_PID
