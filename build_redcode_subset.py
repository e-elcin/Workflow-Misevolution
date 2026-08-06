#!/usr/bin/env python3
"""
Build a seeded, stratified RedCode-Gen subset: the SAME k files from each of
the 8 malware families, so every evolved round is scored on an identical,
family-balanced prompt set (any RR/ASR change is attributable to the workflow
evolving, not to prompt mix).

Run from repo root:
    python build_redcode_subset.py
Writes:
    results/redcode_subset.json   (list of {family, file, path})
"""
import json
import random
from pathlib import Path

SEED = 42
K_PER_FAMILY = 4
DATASET = Path("src/RedCode/dataset/RedCode-Gen")
OUT = Path("results/redcode_subset.json")


def main():
    rng = random.Random(SEED)
    families = sorted(d.name for d in DATASET.iterdir() if d.is_dir())
    chosen = []
    for fam in families:
        files = sorted(p.name for p in (DATASET / fam).glob("*.py"))
        if len(files) < K_PER_FAMILY:
            raise ValueError(f"family {fam} has only {len(files)} files (< k={K_PER_FAMILY})")
        picks = sorted(rng.sample(files, K_PER_FAMILY))
        for f in picks:
            chosen.append({"family": fam, "file": f})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chosen, indent=2))
    print(f"[subset] {len(families)} families x {K_PER_FAMILY} = {len(chosen)} prompts -> {OUT}")
    for fam in families:
        print(f"  {fam}: {[c['file'] for c in chosen if c['family']==fam]}")


if __name__ == "__main__":
    main()
