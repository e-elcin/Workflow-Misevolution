# MATU collection outputs

Raw repeated-run collections (10 runs/prompt, temp 0.7) behind the MATU results.

- `round1_n120/`, `round4_n120/`, `round5_n120/`  — ORIGINAL collections, made
  with the instruction-dropping fallback present. Later-round "compliances" here
  are the harness re-generating without the safety instruction (artifact). Kept
  as the contaminated record. Do not use for MATU conclusions.

- `round1_n120_clean/`, `round4_n120_clean/`, `round5_n120_clean/` — re-collected
  with the fallback removed. Result: round 1 = 1200 refusals / 0 comply,
  rounds 4 and 5 = 1200 errors (Test crashes on RedCode). Confirms the collapse
  was the fallback artifact.

- `baseline_r1_n120/` — no-framing baseline (workspace/ round_1, single generate,
  no Test, no ensemble). The mixed regime (36 mixed prompts) used for the clean
  MATU result: mean unc 3.39 mixed vs 1.71 uniform, AUROC unc->safe 0.995,
  corr(unc, frac_unsafe) -0.92. Judge validated (0 refusals-scored-as-harm).

- `smoke_baseline/` — 4-prompt x 2-run smoke test output; scratch, safe to ignore.
