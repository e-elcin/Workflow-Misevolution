#!/usr/bin/env python3
"""
Build the FULL RedCode-Gen subset: every prompt in every family, no cap.
The paper's aflow_agent evaluation iterates all files, so this matches
their sample of ~160 prompts across 8 families.

Writes to results/redcode_subset_full160.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "src" / "RedCode" / "dataset" / "RedCode-Gen"
OUT = Path("results/redcode_subset_full160.json")


def main():
    families = sorted(p.name for p in DATASET.iterdir() if p.is_dir())
    chosen = []
    per_family = {}
    for fam in families:
        files = sorted(f.name for f in (DATASET / fam).iterdir()
                       if f.is_file() and f.suffix == ".py")
        per_family[fam] = len(files)
        for f in files:
            chosen.append({"family": fam, "file": f})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chosen, indent=2))
    print(f"[full160] {len(families)} families -> {len(chosen)} prompts -> {OUT}")
    for fam, n in per_family.items():
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
