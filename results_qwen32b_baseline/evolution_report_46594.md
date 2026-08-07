# Evolution run summary

- Rounds recorded: **11**
- Best score: **0.909** at round **2**
- Seed (round 1) score: **0.864**
- Validate-best (our pick): round **2** (validate 0.909)
- Test-best (paper pick): round **1** (test 0.818)

## Key structural events
- **ensemble node** first appears at round **3**
- **test node** first appears at round **2**
- **retry/loop** first appears at round **3**

## Per-round detail

| Round | Parent | Validate | Test | Δ vs parent | Kept? | Structure | What changed |
|------:|-------:|--------:|-----:|:-----------:|:-----:|:----------|:-------------|
| 1 | — | 0.864 | 0.818 | — | — | 1x generate | (seed workflow) |
| 2 | 1 | 0.909 | 0.818 | +0.045 | yes | 1x generate, test | Added the Test operator to verify the correctness of the generated code solution. |
| 3 | — | — | — | — | — | 1x generate, ensemble, loop | (seed workflow) |
| 4 | — | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |
| 5 | — | — | — | — | — | 1x generate, ensemble, loop | (seed workflow) |
| 6 | — | — | — | — | — | 1x generate, ensemble, test, loop | (seed workflow) |
| 7 | 2 | 0.864 | 0.795 | -0.045 | no | 2x generate, ensemble, test, loop | Added a ScEnsemble operator to enhance the selection of the best solution before testing. |
| 8 | — | — | — | — | — | 2x generate, ensemble, test, loop | (seed workflow) |
| 9 | — | — | — | — | — | 1x generate, ensemble | (seed workflow) |
| 10 | 2 | 0.864 | 0.818 | -0.045 | no | 1x generate, test, loop | Added a loop to retry generating a solution if the initial solution fails the tests. Introduced a condition to limit the number of retries to avoid infinite loops. |
| 11 | — | — | — | — | — | 2x generate, ensemble, test, loop | (seed workflow) |
