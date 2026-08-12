#!/bin/bash
#SBATCH --job-name=qualitative_sample
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# Runs qualitative_sample.py on one RedCode-Gen prompt through the seed and
# aflow_agent workflows. Same vLLM startup and endpoints as the paper-repl sweep.
# H2_FRAMING intentionally UNSET (matches the paper's no-framing setup).

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
echo "H2_FRAMING intentionally UNSET (matches paper's no-framing setup)"

python qualitative_sample.py 2>&1 | tee $LOGDIR/qualitative_${SLURM_JOB_ID}.log

kill $VLLM_PID
