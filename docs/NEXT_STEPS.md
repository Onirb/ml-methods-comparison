# Implementation plan

## Completed checkpoint

- Repository and original notebook audited.
- Modern and legacy Python environments specified.
- Original notebook reproduced from top to bottom without modifying it.
- Methodological limitations documented.
- Deterministic synthetic dataset generators implemented.
- Fold-local preprocessing, shared repeated stratified splits and three metrics
  implemented.
- Fast tests added for contracts, determinism and expected radial behavior.

## Next sequence

1. Run and freeze the controlled multi-seed baseline. **Completed.**
2. Review whether every dataset actually expresses its stated hypothesis.
   **Completed.**
3. Add a correct empirical bias–variance experiment using independently fitted
   models and an independent evaluation population. **Completed:** the Brier
   score experiment also uses the analytic posterior of the data generator, so
   noise, squared bias and variance are separately identifiable.
4. Add pruning and ensemble experiments with model selection separated from
   final evaluation. **Completed.**
5. Create a short narrative notebook that reads the generated tables rather
   than recomputing hidden state. **Completed.**
6. Rewrite the README around the scientific question, results and limitations.
   **Completed.**
7. Validate tests and result generation from a clean checkout before publishing.

An interactive explorer may be considered later. API, database, Docker and
deployment are deliberately outside this cycle because they do not strengthen
the central scientific demonstration.
