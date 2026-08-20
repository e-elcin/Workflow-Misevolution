#!/bin/bash
#SBATCH --job-name=matu_clean
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

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
    --model "$MODEL" --served-model-name "$MODEL" --port $PORT \
    --quantization gptq --tensor-parallel-size 2 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    > $LOGDIR/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM..."
UP=0
for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then UP=1; echo "vLLM up after ${i}"; break; fi
    sleep 10
done
if [ $UP -eq 0 ]; then echo "ABORT: vLLM timeout"; kill $VLLM_PID; exit 1; fi

cd $REPO
unset H2_FRAMING

for ROUND in 1 4 5; do
  echo "########## CLEAN COLLECT round $ROUND (no fallback) ##########"
  python run_matu_collection.py \
      --round $ROUND --n-runs 10 \
      --subset results/redcode_subset_120.json \
      --workspace workspace_internal_safety_fixed/HumanEval \
      --config $REPO/src/config/config_matu.yaml \
      --out-dir results_matu/round${ROUND}_n120_clean \
      2>&1 | tee $LOGDIR/matu_clean_r${ROUND}_${SLURM_JOB_ID}.log
done

kill $VLLM_PID
echo "CLEAN COLLECTION DONE for rounds 1,4,5"
