#!/bin/bash
#SBATCH --job-name=rejudge
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
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

MODEL="/dl_scratch2/ege/hf_cache/hub/Mistral-Large-Instruct-2407-AWQ"
PORT=8000
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" --port $PORT \
    --quantization awq --tensor-parallel-size 2 \
    --max-model-len 8192 --gpu-memory-utilization 0.95 \
    --enforce-eager \
    > $LOGDIR/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
    curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1 && { echo "vLLM up"; break; }
    sleep 10
done

cd $REPO
python run_rejudge.py --judge-role rejudge 2>&1 | tee $LOGDIR/rejudge_${SLURM_JOB_ID}.log
python plot_judge_ablation.py --run-id ${SLURM_JOB_ID} 2>&1 | tee $LOGDIR/judgeplot_${SLURM_JOB_ID}.log

kill $VLLM_PID
