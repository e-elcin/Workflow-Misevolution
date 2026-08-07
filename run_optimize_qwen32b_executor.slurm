#!/bin/bash
#SBATCH --job-name=qwen32b_exec_baseline
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# 10-round AFlow evolution with split model roles:
#   executor  -> Qwen2.5-32B-GPTQ-Int4 on GPU 1, port 8001
#   optimizer -> Qwen2.5-72B-GPTQ-Int4 on GPU 0, port 8000
#   judge     -> Qwen2.5-72B-GPTQ-Int4 on GPU 0, port 8000  (n/a for baseline)
#
# Isolates executor scale as the variable. Optimizer stays strong so
# workflow rewrites are as good as they were in the all-72B runs.
#
# NO safety framing (H2_FRAMING unset). Baseline capability curve at the
# split-role setup, before we run the safety-seed follow-up.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

# ---- launch 72B on GPU 0, port 8000 (optimizer + judge) ----
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
for i in $(seq 1 240); do
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
unset H2_FRAMING

# 1. evolve 10 rounds
python run_optimize.py \
    --max-rounds 10 \
    --optimized-path workspace_qwen32b_executor \
    --initial-round 1 \
    2>&1 | tee $LOGDIR/qwen32b_exec_evolve_${SLURM_JOB_ID}.log

# 2. re-score every round on the held-out HumanEval test set
python run_test_qwen32b_executor.py \
    2>&1 | tee $LOGDIR/qwen32b_exec_test_${SLURM_JOB_ID}.log

# 3. report
python report_evolution.py \
    --workflows src/workspace_qwen32b_executor/HumanEval/workflows \
    --out results_qwen32b_executor \
    --run-id ${SLURM_JOB_ID} \
    2>&1 | tee $LOGDIR/qwen32b_exec_report_${SLURM_JOB_ID}.log

kill $VLLM72_PID $VLLM32_PID
