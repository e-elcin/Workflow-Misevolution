"""
Build a small, reproducible HumanEval practice/exam split for the
Misevolution workflow-evolution experiment.

Pulls the canonical 164-problem HumanEval set from the public GitHub
mirror, deterministically samples 66 of 164, then splits those into:
  - validate : small "practice" set the optimizer scores itself on each round
  - test     : held-out "exam" set, scored only at the very end

Run once from the repo root:
    python src/data/build_humaneval_data.py
Writes:
    src/data/datasets/humaneval_validate.jsonl   (22 problems)
    src/data/datasets/humaneval_test.jsonl       (44 problems)
"""
import gzip
import json
import random
import urllib.request
from pathlib import Path

SEED = 42
N_TOTAL = 66
N_VALIDATE = 22          # remaining 44 go to test
SRC = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
OUT_DIR = Path(__file__).resolve().parent / "datasets"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(SRC).read()
    problems = [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()]
    assert len(problems) == 164, f"expected 164 problems, got {len(problems)}"

    rng = random.Random(SEED)
    rng.shuffle(problems)
    chosen = problems[:N_TOTAL]
    splits = {"validate": chosen[:N_VALIDATE], "test": chosen[N_VALIDATE:]}

    for name, rows in splits.items():
        path = OUT_DIR / f"humaneval_{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows):3d} problems -> {path}")


if __name__ == "__main__":
    main()