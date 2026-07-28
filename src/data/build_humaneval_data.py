"""
Build a small, reproducible HumanEval practice/exam split for the
Misevolution workflow-evolution experiment.

Writes THREE files:
  - humaneval_validate.jsonl : practice set the optimizer scores each round
  - humaneval_test.jsonl      : held-out exam set, scored only at the very end
  - humaneval_public_test.jsonl : a SMALL visible subset of asserts per problem,
      used only by AFlow's Test operator for in-workflow self-correction. Kept
      to a few asserts on purpose: the full hidden test would let any Test node
      oracle the reward. Final scoring uses the full hidden test above, so there
      is no leakage into the score.

Run once from the repo root:
    python src/data/build_humaneval_data.py
"""
import gzip
import json
import random
import urllib.request
from pathlib import Path

SEED = 42
N_TOTAL = 66
N_VALIDATE = 22            # remaining 44 -> test
N_PUBLIC_ASSERTS = 2       # visible asserts per problem for the Test operator
SRC = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
OUT_DIR = Path(__file__).resolve().parent / "datasets"


def public_asserts(test_src: str, k: int):
    """First k assert lines that reference `candidate`, as a visible subset."""
    lines = [ln.strip() for ln in test_src.splitlines()]
    picked = [ln for ln in lines if ln.startswith("assert candidate")] \
        or [ln for ln in lines if ln.startswith("assert")]
    return picked[:k]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(SRC).read()
    problems = [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()]
    assert len(problems) == 164, f"expected 164 problems, got {len(problems)}"

    rng = random.Random(SEED)
    rng.shuffle(problems)
    chosen = problems[:N_TOTAL]

    for name, rows in [("validate", chosen[:N_VALIDATE]), ("test", chosen[N_VALIDATE:])]:
        path = OUT_DIR / f"humaneval_{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows):3d} problems -> {path}")

    pub_path = OUT_DIR / "humaneval_public_test.jsonl"
    empty = 0
    with pub_path.open("w") as f:
        for r in chosen:
            pa = public_asserts(r["test"], N_PUBLIC_ASSERTS)
            if not pa:
                empty += 1
            f.write(json.dumps({"task_id": r["task_id"],
                                "entry_point": r["entry_point"],
                                "test": pa}) + "\n")
    print(f"wrote {len(chosen):3d} public-test rows -> {pub_path} ({empty} with no extractable assert)")


if __name__ == "__main__":
    main()