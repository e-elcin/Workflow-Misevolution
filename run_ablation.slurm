#!/bin/bash
#SBATCH --job-name=seed_ablation
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety
export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
LOGDIR=$REPO/logs
mkdir -p $LOGDIR $REPO/results/ablation

MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
PORT=8000
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" --port $PORT \
    --quantization gptq --tensor-parallel-size 2 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    > $LOGDIR/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
    curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1 && { echo "vLLM up"; break; }
    sleep 10
done

cd $REPO

# --- variant CC: CustomCodeGenerate seed (our baseline) ---
python run_ablation_variant.py --variant cc --rounds 10 2>&1 | tee $LOGDIR/ablation_cc_evolve_${SLURM_JOB_ID}.log
python run_ablation_safety.py  --variant cc              2>&1 | tee $LOGDIR/ablation_cc_safety_${SLURM_JOB_ID}.log

# --- variant AG: AnswerGenerate seed (paper-style baseline) ---
python run_ablation_variant.py --variant ag --rounds 10 2>&1 | tee $LOGDIR/ablation_ag_evolve_${SLURM_JOB_ID}.log
python run_ablation_safety.py  --variant ag              2>&1 | tee $LOGDIR/ablation_ag_safety_${SLURM_JOB_ID}.log

# --- comparison plot + summary table ---
python plot_ablation.py --run-id ${SLURM_JOB_ID} 2>&1 | tee $LOGDIR/ablation_plot_${SLURM_JOB_ID}.log

kill $VLLM_PID
