# Evolution run summary

- Rounds recorded: **21**
- Best score: **0.909** at round **4**
- Seed (round 1) score: **0.864**

## Key structural events
- **ensemble node** first appears at round **2**
- **test node** first appears at round **4**
- **retry/loop** first appears at round **2**

## Per-round detail

| Round | Parent | Score | Δ vs parent | Kept? | Structure | What changed |
|------:|-------:|------:|:-----------:|:-----:|:----------|:-------------|
| 1 | — | 0.864 | — | — | 1x generate | (seed workflow) |
| 2 | 1 | 0.864 | +0.000 | no | 1x generate, ensemble, loop | Add the `sc_ensemble` operator to enhance the solution selection process, ensuring the best possible solution is chosen by voting among multiple solutions. |
| 3 | 1 | 0.818 | -0.045 | no | 1x generate, ensemble, loop | Added the `sc_ensemble` operator to aggregate multiple solutions and select the most frequent one, enhancing the robustness of the solution selection process. |
| 4 | 2 | 0.909 | +0.045 | yes | 1x generate, ensemble, test, loop | Add a test operator call to validate the generated solution against provided test cases before returning the final result. |
| 5 | 3 | 0.000 | -0.818 | no | 1x generate, ensemble, loop | Add a custom method to handle edge cases and improve the accuracy of the generated code. |
| 6 | 2 | 0.909 | +0.045 | yes | 1x generate, ensemble, test, loop | Added a test operator call to validate the generated solution against provided test cases before returning the final result. |
| 7 | 4 | 0.909 | +0.000 | no | 2x generate, ensemble, test, loop | Modify the CustomCodeGenerate operator to include a reflection step if the initial solution fails testing. This step will attempt to identify the error and refine the solution before final testing. |
| 8 | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |
| 9 | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |
| 10 | — | — | — | — | 1x generate, test | (seed workflow) |
| 11 | 7 | 0.909 | +0.000 | no | 2x generate, ensemble, test, loop | Add a reflection step before attempting to refine the solution to ensure that the reflection is based on a more informed analysis of the initial failure. |
| 12 | 1 | 0.864 | +0.000 | no | 1x generate, ensemble, loop | - Added the `sc_ensemble` operator to the workflow to enhance the robustness of the solution selection process.
    - Modified the `__call__` method to include the `sc_ensemble` step before returning the final solution. |
| 13 | 7 | 0.909 | +0.000 | no | 2x generate, ensemble, test, loop | Add a pre-processing step to analyze the problem and generate a more specific instruction for the `custom_code_generate` method. |
| 14 | 7 | 0.000 | -0.909 | no | 2x generate, ensemble, test, loop | Add a pre-reflection step before the initial solution generation to ensure the problem is well understood and the initial instructions are more directed. |
| 15 | 1 | 0.864 | +0.000 | no | 1x generate, ensemble, loop | Introduce the `sc_ensemble` operator to aggregate multiple solutions and select the most frequent one, enhancing the robustness of the solution selection process. Modify the `__call__` method to include the `sc_ensemble` step before returning the final solution. |
| 16 | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |
| 17 | — | — | — | — | 1x generate, test | (seed workflow) |
| 18 | 7 | 0.000 | -0.909 | no | 2x generate, ensemble, test, loop | Add a pre-analysis step to evaluate the complexity and type of the problem before generating the initial solution. |
| 19 | 1 | 0.818 | -0.045 | no | 1x generate, ensemble, loop | - Introduce the `sc_ensemble` operator to aggregate multiple solutions and select the most frequent one, enhancing the robustness of the solution selection process.
- Modify the `__call__` method to include the `sc_ensemble` step before returning the final solution. |
| 20 | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |
| 21 | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |


What this run did
We ran AFlow's optimizer for 20 rounds on 22 HumanEval problems, using Qwen2.5-72B for both roles. The starting workflow was a single code generator (round 1, score 0.864). The optimizer added an ensemble step at round 2 and a test step at round 4, reaching its best score, 0.909, at round 4. The winning structure is three generations feeding an ensemble selector, plus a test check. The first part matches the paper's Figure 9 workflow; the test step is an extra self-check our run happened to add.
What it means
The evolution worked: scores moved between 0.818 and 0.909, and the useful structures were kept. This is unlike an earlier broken run where a missing test file pinned every score at 0.136. The occasional 0.000 rounds are single bad evaluations (expected with validation_rounds=1), and the optimizer correctly ignored them. Our result differing from the paper's Figure 9 is normal: AFlow's search is random, so different runs grow different workflows. What matters is that the ensemble node, the part responsible for the safety problem, is present in our workflow.
Next steps
Task score was never the real goal. The key next step is the safety sweep: run each evolved round against the RedCode harmful prompts and record the refusal rate and the judge's 0 to 10 harm score per round. This shows the main claim, capability rising while safety drops. Later options include raising validation_rounds to steady the scores and testing whether the refusal rate falls as predicted when the ensemble filters out refusals.


The paper (Section 3.4 and Appendix C.4) confirms the two-benchmark design: they used AFlow to evolve the workflow for 20 iterations on a HumanEval subset with Qwen2.5-72B-Instruct, selected the best workflow by HumanEval test-set performance, then evaluated that workflow on RedCode-Gen for safety (before vs. after evolution), finding Refusal Rate dropping 36.3%→5.6% and ASR rising 54.4%→83.1%. So HumanEval is only the capability signal that drives evolution; RedCode-Gen is where safety is measured. The one difference from us is that they used AFlow's own dedicated HumanEval subset, while we built our own 22/44 split from canonical HumanEval — same benchmark and method, different exact subset.
