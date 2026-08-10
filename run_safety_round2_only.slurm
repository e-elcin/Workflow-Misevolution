#!/bin/bash
#SBATCH --job-name=paper_repl_round2_only
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/Workflow-Misevolution/logs/%j.err

# Rerun ONLY round 2 (paper's aflow_agent) after fixing the prompt.py filename bug.
# Round 1 results already saved in results_paper_replication/safety/round_1.json.
# This job scores round 2 and re-generates safety_summary.json with both rounds.

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
    echo "ABORT: vLLM startup timed out"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

cd $REPO

unset H2_FRAMING
echo "H2_FRAMING intentionally UNSET"

# Score round 2 only
python run_safety_round_paper_replication.py --round 2 \
    2>&1 | tee $LOGDIR/paper_repl_round2_${SLURM_JOB_ID}.log

# Now rebuild safety_summary.json by combining the two per-round jsons
python << 'PY'
import json
from pathlib import Path
ROOT = Path("/dl_scratch2/ege/elcin/Workflow-Misevolution")
safety = ROOT / "results_paper_replication" / "safety"
summary = []
for rnd in [1, 2]:
    f = safety / f"round_{rnd}.json"
    if f.exists():
        m = json.loads(f.read_text())
        summary.append({k: m[k] for k in
                        ("round", "mean_score", "refusal_rate", "asr_5", "asr_8",
                         "n_valid", "n_error")})
out = ROOT / "results_paper_replication" / "safety_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"wrote {out} ({len(summary)} rounds)")
if len(summary) == 2:
    s, e = summary[0], summary[1]
    print(f"\n=== paper's seed-vs-aflow_agent delta (n={s['n_valid']} prompts) ===")
    print(f"  Seed         : mean={s['mean_score']:.2f}  RR={s['refusal_rate']:.3f}  ASR@5={s['asr_5']:.3f}  ASR@8={s['asr_8']:.3f}")
    print(f"  AFlow-agent  : mean={e['mean_score']:.2f}  RR={e['refusal_rate']:.3f}  ASR@5={e['asr_5']:.3f}  ASR@8={e['asr_8']:.3f}")
    print(f"  Delta        : mean={e['mean_score']-s['mean_score']:+.2f}  "
          f"RR={e['refusal_rate']-s['refusal_rate']:+.3f}  "
          f"ASR@5={e['asr_5']-s['asr_5']:+.3f}  "
          f"ASR@8={e['asr_8']-s['asr_8']:+.3f}")
PY

kill $VLLM_PID
