#!/bin/bash
#SBATCH --job-name=matu_base
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

echo "########## SMOKE TEST: 4 prompts x 2 runs ##########"
rm -rf results_matu/smoke_baseline
python run_matu_collection.py \
    --round 1 --n-runs 2 \
    --subset results/redcode_subset.json \
    --workspace workspace/HumanEval \
    --config $REPO/src/config/config_matu.yaml \
    --out-dir results_matu/smoke_baseline \
    2>&1 | tail -20

echo "########## SMOKE CHECK ##########"
python3 - << 'PYEOF'
import json, glob, sys
files = sorted(glob.glob('results_matu/smoke_baseline/run_*.json'))
if len(files) < 2:
    print("SMOKE FAIL: expected 2 run files, got", len(files)); sys.exit(1)
runs = [json.load(open(f)) for f in files]
pids = list(runs[0].keys())
err = sum(1 for p in pids for r in runs if r[p]['response'].startswith('[RUN ERROR'))
empty = sum(1 for p in pids for r in runs if len(r[p]['response'].strip()) < 20)
total = len(pids) * len(runs)
# how many prompts produced different text across the 2 runs (temp 0.7 working?)
varied = sum(1 for p in pids if runs[0][p]['response'][:200] != runs[1][p]['response'][:200])
print(f"prompts={len(pids)} total_runs={total} errors={err} empty={empty} varied_prompts={varied}/{len(pids)}")
if err > 0:
    print("SMOKE FAIL: workflow errored"); sys.exit(1)
if empty > total * 0.2:
    print("SMOKE FAIL: too many empty responses"); sys.exit(1)
if varied == 0:
    print("SMOKE FAIL: no variation across runs, temperature not applied"); sys.exit(1)
print("SMOKE PASS")
PYEOF

if [ $? -ne 0 ]; then
    echo "ABORTING before full collection: smoke test failed"
    kill $VLLM_PID
    exit 1
fi

echo "########## FULL COLLECTION: 120 prompts x 10 runs ##########"
python run_matu_collection.py \
    --round 1 --n-runs 10 \
    --subset results/redcode_subset_120.json \
    --workspace workspace/HumanEval \
    --config $REPO/src/config/config_matu.yaml \
    --out-dir results_matu/baseline_r1_n120 \
    2>&1 | tee $LOGDIR/matu_baseline_${SLURM_JOB_ID}.log

kill $VLLM_PID
echo "BASELINE MATU COLLECTION DONE"
