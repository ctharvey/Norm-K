# Applying ‖K‖ to a statement pair

The first K prototype studied four manually specified transformations.
That does not yet say how to score a pair A and B encountered during memory
conflict review. This document specifies a restricted, falsifiable bridge.

## Definition

Let A and B be two statements and T a *single controlled transformation*
which can be applied consistently to both. A quadrilateral is:

1. A
2. B
3. T(A)
4. T(B)

Define the counterfactual interaction residual:

\[
K(A,B;T)=E(A)+E(T(B))-E(T(A))-E(B)
\]

If E is additive for the A→B displacement and the effect of T is independent
of that displacement, K is approximately zero. Deviations are conditional
geometry signals. **Nothing in this identity establishes logical contradiction.**

The cheap frozen-vector hypothesis is that K-derived features could improve
distinguishing *same-fact incompatibility* from compatible changes.

## Supported transformations

- `negation`: explicit inversion of a small, tested grammar. It must preserve
  subject, object, and clause scope. The present implementation returns None
  for unsupported forms rather than hallucinating a counterfactual.
- `subject_swap`: replace the primary subject with the same unrelated name
  on both sides, only if both sentences contain that subject.

Coverage and failure reasons must be reported by class. A missing K score is
**not** assigned zero and is never treated as semantic evidence.

## Evaluation contract

- Precompute the exact same eligible examples for both competing models.
- Baseline features: cosine, distance, Hoyer sparsity, and simple vector stats.
- Augmented model: those exact baseline features **plus** raw/relative K.
- Fit on training only and report held-out average precision and recall at a
  fixed review budget.
- Hold out subjects, templates, and entire relation families separately.
- Purge overlapping *exact statements* between splits, prioritizing test.
- Bootstrap paired score differences by subject when there are enough
  independent subject clusters; otherwise mark the interval unavailable.
- Test whether any win survives new relation families. Favor a reliable
  negative result over overfitting a promising example.

## What still needs research

- Syntactic rewriting must be externally validated for broader grammar.
- The practical cost of 2 additional embeddings per pair (after caching)
  versus an NLI cross-encoder may erase any arithmetic savings.
- All results so far use a lexical hash control, not a trained encoder.
- Meaning-preserving transformations must be checked independently; semantic
  transformations can be nonlinear and context dependent.
- Temporal truth, attribute cardinality, source trust, and provenance still
  require the structured belief system proposed in Menhir #511.

See [offline-ablation.md](../reports/offline-ablation.md) for the first
engineering-only run, including a held-out relation failure case.
