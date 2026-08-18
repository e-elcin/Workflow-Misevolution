#!/bin/bash
#SBATCH --job-name=isafe_fixed
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=15:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# Internal-safety experiment RERUN with the operators.py fix.
# Now CustomCodeGenerate honors the passed instruction, so SAFETY_INSTRUCTION
# in round_1/prompt.py actually reaches the model. H2_FRAMING stays UNSET.
# Writes to workspace_internal_safety_fixed/ and results_internal_safety_fixed/
# so the old (buggy-operator) results are preserved for comparison.

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
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then UP=1; echo "vLLM up after ${i} attempts"; break; fi
    sleep 10
done
if [ $UP -eq 0 ]; then echo "ABORT: vLLM timeout"; kill $VLLM_PID; exit 1; fi

cd $REPO
unset H2_FRAMING
echo "H2_FRAMING UNSET; SAFETY_INSTRUCTION now reaches model via fixed operator"

[ -f results/redcode_subset_120.json ] || echo "WARN: 120 subset missing"

# 1. Evolution: 10 rounds on the fixed workspace
python run_optimize.py \
    --max-rounds 10 \
    --optimized-path workspace_internal_safety_fixed \
    --initial-round 1 \
    2>&1 | tee $LOGDIR/isafe_fixed_evolve_${SLURM_JOB_ID}.log

# 2. Safety sweep (writes to results_internal_safety_fixed/) + prompt.py snapshot
python run_safety_sweep_internal_safety_fixed.py \
    2>&1 | tee $LOGDIR/isafe_fixed_sweep_${SLURM_JOB_ID}.log

# 3. Dump prompt.py evolution to see if the optimizer touched SAFETY_INSTRUCTION
echo ""
echo "===== PROMPT.PY EVOLUTION (fixed run) ====="
for r in src/workspace_internal_safety_fixed/HumanEval/workflows/round_*/prompt.py; do
    echo ""; echo "--- $r ---"; cat "$r"
done

kill $VLLM_PID
