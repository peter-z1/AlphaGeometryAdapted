# Benchmarks

This directory keeps problem formalizations, portable runners, diagrams,
proofs, and compact result summaries together.

- `imo_ag_30/`: canonical 30-problem AlphaGeometry suite.
- `official_alphageometry_canary/`: runner and saved canaries for the official
  JAX/Meliad checkpoint (downloaded separately).
- `results/reconstruction_v4/`: immutable results from the selected educational
  model, including initial and larger-search budgets.

Results depend on model, numerical seed, beam size, search depth, and timeout.
The JSON summaries retain those settings. A direct DD+AR proof is not counted
as a genuine auxiliary-model solve.
