#!/bin/bash
#SBATCH --job-name=all72b_baseline_sweep
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# Safety sweep on the ALL-72B BASELINE (no framing) evolution, using the
# larger 120-prompt RedCode subset. Reads workflows from workspace/HumanEval,
# scores each recorded round on 120 prompts, writes results to
# results_all72b_baseline/.
#
# 72B model in both executor and judge roles (matches baseline evolution setup).
# H2_FRAMING is intentionally NOT set — this is the no-framing safety trajectory.
#
# Wall time 8h to accommodate NFS-slow model load (see job 46604) and
# the ~4x longer sweep from n=32 -> n=120.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

# Single 72B server on 2 GPUs (executor + judge share it, matches original
# all-72B baseline evolution setup)
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
echo "H2_FRAMING intentionally UNSET (no-framing sweep)"

# Build the 120-prompt subset if not already present
[ -f results/redcode_subset_120.json ] || python build_redcode_subset_120.py

# Sanity: which config2.yaml is this branch using?
grep -E "^  [a-z]+:|base_url:|model:" src/config/config2.yaml | head -20

python run_safety_sweep_all72b_baseline.py \
    2>&1 | tee $LOGDIR/all72b_baseline_sweep_${SLURM_JOB_ID}.log

kill $VLLM_PID
