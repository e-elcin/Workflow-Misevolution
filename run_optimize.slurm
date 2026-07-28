#!/bin/bash
#SBATCH --job-name=workflow_evolve
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out   # <<EDIT>> to your clone path
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err    # <<EDIT>> same

# --- CUDA exports from runner_basic.slurm (same as your other jobs) ---
# export PATH=...
# export LD_LIBRARY_PATH=...

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# --- new repo (clone of e-elcin/Workflow-Misevolution) ---
REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution     # <<EDIT>> to wherever you clone it
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

# --- ONE model server. No judge: HumanEval scores by code execution, not an LLM.
#     Executor (temp 0.1) and optimizer (temp 0.7) share this endpoint. ---
MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
PORT=8000                          # must match base_url in src/config/config2.yaml

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

# --- fallback if GPTQ + TP=2 errors at startup: comment the block above and use one GPU ---
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL" --served-model-name "$MODEL" --port $PORT \
#     --quantization gptq --max-model-len 32768 --gpu-memory-utilization 0.90 \
#     > $LOGDIR/vllm_${SLURM_JOB_ID}.log 2>&1 &
# VLLM_PID=$!

# --- wait until the endpoint is live (max ~30 min) ---
echo "Waiting for vLLM on $PORT..."
for i in $(seq 1 180); do
    curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1 && { echo "vLLM up on $PORT"; break; }
    sleep 10
done

# --- run the 20-round evolution: validation_rounds=1, convergence off, auto-resume ---
cd $REPO
python run_optimize.py 2>&1 | tee $LOGDIR/evolve_${SLURM_JOB_ID}.log

# --- shut the server down ---
kill $VLLM_PID