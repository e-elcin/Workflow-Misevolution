#!/bin/bash
#SBATCH --job-name=matu_full
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety
export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

WM=/dl_scratch2/ege/elcin/Workflow-Misevolution
MATU=/dl_scratch2/ege/elcin/MATU
GRADER=/dl_scratch2/ege/ege/repos/MATU/quick_start/code/grade_harmbench.py
LOGDIR=$WM/logs
mkdir -p $LOGDIR
MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
PORT=8000

# ---- start vLLM for the WORKFLOW generation (72B) ----
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" --port $PORT \
    --quantization gptq --tensor-parallel-size 2 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    > $LOGDIR/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM..."
UP=0
for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then UP=1; break; fi
    sleep 10
done
if [ $UP -eq 0 ]; then echo "ABORT: vLLM timeout"; kill $VLLM_PID; exit 1; fi
echo "vLLM up"

cd $WM
unset H2_FRAMING

# Rounds to analyze: 1 (all-refuse), 4 (all-refuse+ensemble), 5 (collapsed)
# On the 120-prompt subset for tighter estimates.
for ROUND in 1 4 5; do
  echo ""
  echo "########## COLLECTING round $ROUND (10 runs, temp 0.7, n=120) ##########"
  python run_matu_collection.py \
      --round $ROUND \
      --n-runs 10 \
      --subset results/redcode_subset_120.json \
      --workspace workspace_internal_safety/HumanEval \
      --config $WM/src/config/config_matu.yaml \
      --out-dir results_matu/round${ROUND}_n120 \
      2>&1 | tee $LOGDIR/matu_collect_r${ROUND}_${SLURM_JOB_ID}.log
done

# vLLM (72B workflow model) no longer needed; free the GPUs for embedding/judging
kill $VLLM_PID
sleep 5

# ---- MATU analysis per round (embed on CPU/GPU, grade with 14B judge) ----
cd $MATU
for ROUND in 1 4 5; do
  echo ""
  echo "########## MATU PIPELINE round $ROUND ##########"
  RUNS=$WM/results_matu/round${ROUND}_n120
  WORK=$MATU/out/round${ROUND}_n120
  mkdir -p $WORK/results

  # 1. adapter: AFlow runs -> MATU log
  python aflow_to_matu.py --runs-dir $RUNS --out $WORK/log.json

  # 2. embed
  python -m matu.embed_logs \
    --logs $WORK/log.json --out_dir $WORK/embeddings \
    --roles user assistant --model "Qwen/Qwen3-Embedding-0.6B"

  # 3. cp2 decomposition
  matu cp2 \
    --embeddings $WORK/embeddings/user_embedding_matrices.pkl $WORK/embeddings/assistant_embedding_matrices.pkl \
    --out $WORK/scores.pkl --legacy_fit_out $WORK/fit.pkl

  # 4. fit -> scalar uncertainty
  matu fit --fit_dict $WORK/scores.pkl --out $WORK/uncertainty.pkl

  # 5. grade the 10 runs per prompt (14B judge)
  cp $WORK/log.json $WORK/conversation_logs_harmbench.json
  python $GRADER $WORK \
    --judge llm_judge --judge_model "Qwen/Qwen2.5-14B-Instruct" \
    --cache_dir $HF_HOME

  # 6. eval both directions
  echo "--- round $ROUND STRICT ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty $WORK/uncertainty.pkl \
    --labels $WORK/results/accuracy_dict_generated.pkl \
    --score_mode raw --error_rule any_incorrect
  echo "--- round $ROUND RELAXED ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty $WORK/uncertainty.pkl \
    --labels $WORK/results/accuracy_dict_generated.pkl \
    --score_mode raw --error_rule majority_incorrect

  # 7. correlation + both-polarity AUROC (the real read)
  python3 -c "
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
unc = pickle.load(open('$WORK/uncertainty.pkl','rb'))
acc = pickle.load(open('$WORK/results/accuracy_dict_generated.pkl','rb'))
ids = [k for k in unc if k in acc]
u = np.array([unc[k] for k in ids])
frac_unsafe = np.array([1-np.mean(acc[k]) for k in ids])
maj = (frac_unsafe>0.5).astype(int)
print('round $ROUND:', len(ids), 'prompts')
if len(set(maj))>1:
    print('  AUROC unc->unsafe:', round(roc_auc_score(maj,u),4))
    print('  AUROC unc->safe:  ', round(roc_auc_score(1-maj,u),4))
print('  corr(unc,frac_unsafe):', round(np.corrcoef(u,frac_unsafe)[0,1],3))
print('  mean uncertainty:', round(float(np.mean(u)),3))
"
done

echo ""
echo "MATU FULL PIPELINE DONE for rounds 1,4,5"
