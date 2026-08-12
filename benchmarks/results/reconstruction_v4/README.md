# Reconstruction-v4 saved results

These are the immutable outputs copied from the selected v4 checkpoint before
the development scratch directories were cleaned.

## Summary

- **JGEX-33:** 20 records were marked solved: 1 direct DD+AR solve, 18 genuine
  nonempty auxiliary-construction solves, and 1 no-op artifact that must not be
  credited to the LM. Nine were unsolved and four timed out.
- **IMO-AG-30:** the initial run produced 14 direct DD+AR and 2 genuine
  auxiliary solves. A larger search found 2 additional genuine solves, for a
  merged result of 18/30 under the tested budgets.

Directories ending in `_initial` and `_large_search` preserve the distinct
budgets rather than overwriting one another. Every available proof and summary
is included. These are empirical search results, not claims of equivalence to
the original DeepMind model.
