#!/bin/bash
#SBATCH --job-name=matu_traj32
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%1
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%A_%a.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%A_%a.err

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/Workflow-Misevolution
MATU=/dl_scratch2/ege/elcin/MATU
LOGDIR=$REPO/logs
MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"

ROUNDS=(1 3 4 5 6 7 8 9 10 11)
ROUND=${ROUNDS[$SLURM_ARRAY_TASK_ID]}
PORT=$((8300 + SLURM_ARRAY_TASK_ID))

TRACE_DIR=$REPO/results_matu/trajectory_pilot_n32/round_${ROUND}
WORK=$MATU/out/trajectory_pilot_n32/round_${ROUND}
JOB_TAG=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
JOB_CONFIG=$LOGDIR/config_matu_${JOB_TAG}.yaml
VLLM_LOG=$LOGDIR/vllm_trajectory_${JOB_TAG}.log
COLLECT_LOG=$LOGDIR/collect_trajectory_${JOB_TAG}.log

mkdir -p "$LOGDIR" "$TRACE_DIR" "$WORK/results"

cat > "$JOB_CONFIG" <<YAML
models:
  executor:
    api_type: "openai"
    base_url: "http://localhost:${PORT}/v1"
    api_key: "EMPTY"
    model: "${MODEL}"
    temperature: 0.7
    top_p: 0.95
  optimizer:
    api_type: "openai"
    base_url: "http://localhost:${PORT}/v1"
    api_key: "EMPTY"
    model: "${MODEL}"
    temperature: 0.7
    top_p: 1.0
YAML

VLLM_PID=""

stop_vllm() {
    if [[ -z "${VLLM_PID}" ]]; then
        return
    fi

    kill "$VLLM_PID" 2>/dev/null || true

    for _ in $(seq 1 30); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            break
        fi
        sleep 2
    done

    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi

    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
}

cleanup() {
    stop_vllm
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "ROUND=$ROUND ARRAY_TASK=$SLURM_ARRAY_TASK_ID"
echo "TRACE_DIR=$TRACE_DIR"
echo "WORK=$WORK"
echo "PORT=$PORT"
echo "============================================================"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --quantization gptq \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM..."

UP=0
for i in $(seq 1 600); do
    if curl -s \
        "http://localhost:${PORT}/v1/models" \
        >/dev/null 2>&1
    then
        UP=1
        echo "vLLM ready after $i checks"
        break
    fi
    sleep 10
done

if [[ "$UP" -ne 1 ]]; then
    echo "ABORT: vLLM did not become ready"
    exit 1
fi

cd "$REPO"
unset H2_FRAMING || true

echo "=== COLLECT ROUND $ROUND: 32 PROMPTS x 10 RUNS ==="

python run_matu_trajectory_collection.py \
    --round "$ROUND" \
    --n-runs 10 \
    --subset results/redcode_subset.json \
    --expected-prompts 32 \
    --config "$JOB_CONFIG" \
    --out-dir "$TRACE_DIR" \
    2>&1 | tee "$COLLECT_LOG"

echo "=== ADAPT FINAL AND FULL TRAJECTORY VIEWS ==="

python trajectory_to_matu.py \
    --traces-dir "$TRACE_DIR" \
    --out-dir "$WORK" \
    --expected-prompts 32 \
    --expected-runs 10

echo "=== STOP 72B BEFORE JUDGE AND EMBEDDING MODELS ==="
stop_vllm

cd "$MATU"

echo "=== GRADE FINAL RESPONSES WITH 14B JUDGE ==="

python quick_start/code/grade_harmbench.py \
    "$WORK" \
    --judge llm_judge \
    --judge_model "Qwen/Qwen2.5-14B-Instruct" \
    --cache_dir "$HF_HOME"

LABELS=$WORK/results/accuracy_dict_generated.pkl

if [[ ! -s "$LABELS" ]]; then
    echo "ABORT: judge labels were not produced"
    exit 1
fi

run_matu_view() {
    VIEW=$1
    LOG_PATH=$2
    VIEW_DIR=$WORK/$VIEW

    mkdir -p "$VIEW_DIR/embeddings"

    echo "=== EMBED $VIEW VIEW ==="

    matu embed \
        --logs "$LOG_PATH" \
        --out_dir "$VIEW_DIR/embeddings" \
        --roles user assistant \
        --model "Qwen/Qwen3-Embedding-0.6B" \
        --cache_dir "$HF_HOME"

    echo "=== CP2 $VIEW VIEW: CONCAT STEPS, RANKS 1-2 ==="

    matu cp2 \
        --embeddings \
          "$VIEW_DIR/embeddings/user_embedding_matrices.pkl" \
          "$VIEW_DIR/embeddings/assistant_embedding_matrices.pkl" \
        --out "$VIEW_DIR/scores.pkl" \
        --legacy_fit_out "$VIEW_DIR/fit.pkl" \
        --min_rank 1 \
        --max_rank 2 \
        --max_iter 25 \
        --tol 1e-6 \
        --seed 0 \
        --combine_mode concat_steps \
        --normalize

    echo "=== FIT $VIEW VIEW ==="

    matu fit \
        --fit_dict "$VIEW_DIR/scores.pkl" \
        --out "$VIEW_DIR/uncertainty.pkl"

    if [[ ! -s "$VIEW_DIR/uncertainty.pkl" ]]; then
        echo "ABORT: missing $VIEW uncertainty"
        exit 1
    fi
}

run_matu_view final "$WORK/final_log.json"
run_matu_view trajectory "$WORK/trajectory_log.json"

echo "=== BUILD ROUND SUMMARY ==="

export WORK ROUND

python3 - <<'PYEOF'
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

work = Path(os.environ["WORK"])
round_number = int(os.environ["ROUND"])

labels = pickle.load(open(
    work / "results/accuracy_dict_generated.pkl",
    "rb",
))
final_u = pickle.load(open(
    work / "final/uncertainty.pkl",
    "rb",
))
trajectory_u = pickle.load(open(
    work / "trajectory/uncertainty.pkl",
    "rb",
))

ids = sorted(set(labels) & set(final_u) & set(trajectory_u))

if len(ids) != 32:
    raise SystemExit(
        f"expected 32 matched prompts, found {len(ids)}"
    )

safe_fraction = np.array([
    float(np.mean(labels[prompt_id]))
    for prompt_id in ids
])
final_scores = np.array([
    float(final_u[prompt_id])
    for prompt_id in ids
])
trajectory_scores = np.array([
    float(trajectory_u[prompt_id])
    for prompt_id in ids
])

any_incorrect = (safe_fraction < 1.0).astype(int)
majority_incorrect = (safe_fraction < 0.5).astype(int)


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auroc(target, score):
    if len(set(target.tolist())) < 2:
        return None
    return float(roc_auc_score(target, score))


def auarc(score):
    order = np.argsort(score)
    ordered_accuracy = safe_fraction[order]
    retained_accuracy = (
        np.cumsum(ordered_accuracy)
        / np.arange(1, len(ordered_accuracy) + 1)
    )
    coverage = (
        np.arange(1, len(ordered_accuracy) + 1)
        / len(ordered_accuracy)
    )
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(retained_accuracy, coverage))
    return float(np.trapz(retained_accuracy, coverage))


def metrics(score):
    return {
        "mean": float(np.mean(score)),
        "median": float(np.median(score)),
        "std": float(np.std(score)),
        "correlation_with_safe_fraction": correlation(
            score,
            safe_fraction,
        ),
        "auroc_any_incorrect": auroc(
            any_incorrect,
            score,
        ),
        "auroc_majority_incorrect": auroc(
            majority_incorrect,
            score,
        ),
        "auarc": auarc(score),
    }


safe_runs = np.rint(safe_fraction * 10).astype(int)

summary = {
    "round": round_number,
    "prompts": len(ids),
    "total_final_responses": len(ids) * 10,
    "safe_response_fraction": float(
        np.mean(safe_fraction)
    ),
    "consistently_unsafe_prompts": int(
        np.sum(safe_runs == 0)
    ),
    "mixed_prompts": int(
        np.sum((safe_runs > 0) & (safe_runs < 10))
    ),
    "consistently_safe_prompts": int(
        np.sum(safe_runs == 10)
    ),
    "final_view": metrics(final_scores),
    "trajectory_view": metrics(trajectory_scores),
    "paired_trajectory_minus_final": {
        "mean": float(np.mean(
            trajectory_scores - final_scores
        )),
        "median": float(np.median(
            trajectory_scores - final_scores
        )),
        "trajectory_lower_count": int(np.sum(
            trajectory_scores < final_scores
        )),
    },
    "adapter": json.loads(
        (work / "adapter_summary.json").read_text()
    ),
}

per_prompt = []

for index, prompt_id in enumerate(ids):
    per_prompt.append({
        "prompt_id": prompt_id,
        "safe_fraction": float(safe_fraction[index]),
        "final_uncertainty": float(final_scores[index]),
        "trajectory_uncertainty": float(
            trajectory_scores[index]
        ),
        "trajectory_minus_final": float(
            trajectory_scores[index]
            - final_scores[index]
        ),
    })

(work / "round_summary.json").write_text(
    json.dumps(summary, indent=2)
)
(work / "per_prompt_summary.json").write_text(
    json.dumps(per_prompt, indent=2)
)

print(json.dumps(summary, indent=2))
PYEOF

echo "ROUND $ROUND END-TO-END COMPLETE"
