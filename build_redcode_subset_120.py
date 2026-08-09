#!/usr/bin/env python3
"""
Build a larger RedCode-Gen subset for higher-resolution safety sweeps.

Selects K_PER_FAMILY=15 prompts per malware family (8 families × 15 = 120
prompts total). Writes to results/redcode_subset_120.json without touching
the original results/redcode_subset.json, so earlier safety runs stay
comparable to their subset.

Deterministic: sorts by filename within each family, takes the first K.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "src" / "RedCode" / "dataset" / "RedCode-Gen"
OUT = Path("results/redcode_subset_120.json")
K_PER_FAMILY = 15


def main():
    families = sorted(p.name for p in DATASET.iterdir() if p.is_dir())
    chosen = []
    for fam in families:
        files = sorted(f.name for f in (DATASET / fam).iterdir()
                       if f.is_file() and f.suffix == ".py")
        picked = files[:K_PER_FAMILY]
        if len(picked) < K_PER_FAMILY:
            print(f"  [warn] {fam}: only {len(picked)} prompts available "
                  f"(wanted {K_PER_FAMILY})")
        for f in picked:
            chosen.append({"family": fam, "file": f})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chosen, indent=2))
    print(f"[subset-120] {len(families)} families x {K_PER_FAMILY} target = "
          f"{len(chosen)} prompts -> {OUT}")
    for fam in families:
        n = sum(1 for c in chosen if c["family"] == fam)
        picked_files = [c["file"] for c in chosen if c["family"] == fam]
        print(f"  {fam} ({n}): {picked_files[:3]}...")


if __name__ == "__main__":
    main()
