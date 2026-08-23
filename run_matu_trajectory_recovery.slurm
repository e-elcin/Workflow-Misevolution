#!/usr/bin/env bash
#SBATCH --job-name=matu_recover
#SBATCH --partition=debug
#SBATCH --nodelist=adep2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%1
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/matu_recovery_%A_%a.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/matu_recovery_%A_%a.err

set -Eeuo pipefail

WF="/dl_scratch2/ege/elcin/Workflow-Misevolution"
MATU="/dl_scratch2/ege/elcin/MATU"
HF_HOME="/dl_scratch2/ege/hf_cache"

ROUNDS=(1 3 4 5 6 7 8 9 10 11)
ROUND="${ROUNDS[$SLURM_ARRAY_TASK_ID]}"
WORK="$MATU/out/trajectory_pilot_n32/round_${ROUND}"

export HF_HOME
export TOKENIZERS_PARALLELISM=false
export WORK ROUND

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

echo "=== RECOVERY CONFIGURATION ==="
echo "host: $(hostname)"
echo "round: $ROUND"
echo "work: $WORK"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "MATU commit: $(git -C "$MATU" rev-parse HEAD)"

python3 - <<'PYGPU'
import torch

count = torch.cuda.device_count()
print("visible CUDA devices:", count)

if count != 1:
    raise SystemExit(
        f"expected exactly one visible GPU, found {count}"
    )

print("embedding GPU:", torch.cuda.get_device_name(0))
PYGPU

LABELS="$WORK/results/accuracy_dict_generated.pkl"

for required in \
    "$WORK/final_log.json" \
    "$WORK/trajectory_log.json" \
    "$LABELS"
do
    if [[ ! -s "$required" ]]; then
        echo "ABORT: missing required saved artifact: $required"
        exit 1
    fi
done

run_matu_view() {
    local view="$1"
    local log_path="$2"
    local view_dir="$WORK/$view"

    mkdir -p "$view_dir/embeddings"

    echo "=== EMBED $view VIEW ==="

    matu embed \
        --logs "$log_path" \
        --out_dir "$view_dir/embeddings" \
        --roles user assistant \
        --model "Qwen/Qwen3-Embedding-0.6B" \
        --cache_dir "$HF_HOME"

    for role in user assistant
    do
        file="$view_dir/embeddings/${role}_embedding_matrices.pkl"
        if [[ ! -s "$file" ]]; then
            echo "ABORT: missing embedding file: $file"
            exit 1
        fi
    done

    echo "=== CP2 $view VIEW: CONCAT STEPS, RANKS 1-2 ==="

    matu cp2 \
        --embeddings \
          "$view_dir/embeddings/user_embedding_matrices.pkl" \
          "$view_dir/embeddings/assistant_embedding_matrices.pkl" \
        --out "$view_dir/scores.pkl" \
        --legacy_fit_out "$view_dir/fit.pkl" \
        --min_rank 1 \
        --max_rank 2 \
        --max_iter 25 \
        --tol 1e-6 \
        --seed 0 \
        --combine_mode concat_steps \
        --normalize

    echo "=== FIT $view VIEW ==="

    matu fit \
        --fit_dict "$view_dir/scores.pkl" \
        --out "$view_dir/uncertainty.pkl"

    if [[ ! -s "$view_dir/uncertainty.pkl" ]]; then
        echo "ABORT: missing $view uncertainty"
        exit 1
    fi
}

cd "$MATU"

run_matu_view final "$WORK/final_log.json"
run_matu_view trajectory "$WORK/trajectory_log.json"

echo "=== BUILD ROUND SUMMARY ==="

SUMMARY_PY="$WF/logs/matu_summary_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.py"

sed -n '220,376p' \
    "$WF/run_matu_trajectory_pilot.slurm" \
    > "$SUMMARY_PY"

if ! grep -q 'round_summary.json' "$SUMMARY_PY"; then
    echo "ABORT: summary program extraction failed"
    exit 1
fi

python3 "$SUMMARY_PY"

for required in \
    "$WORK/final/uncertainty.pkl" \
    "$WORK/trajectory/uncertainty.pkl" \
    "$WORK/round_summary.json" \
    "$WORK/per_prompt_summary.json"
do
    if [[ ! -s "$required" ]]; then
        echo "ABORT: missing recovery result: $required"
        exit 1
    fi
done

echo "ROUND $ROUND RECOVERY COMPLETE"
