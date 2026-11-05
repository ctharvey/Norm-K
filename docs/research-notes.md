# ‖K‖ — mathematical notes, visual experiments, and prior art

**Research notes, November 5, 2026.** These are hypotheses and pointers, not validated experimental results. For build order and exit gates, see [research-plan.md](research-plan.md).

## 1. Why ordinary cosine can fail

Text embeddings encode semantic similarity, not logical consistency by design. The statements "Alice owns a dog" and "Alice does not own a dog" can share most of their semantic context. Conversely, "Alice owns a dog" and "Alice works in Texas" may be less similar while perfectly compatible.

A similarity threshold is therefore an *imperfect candidate-selection heuristic*, not a contradiction detector.

In Menhir, [PR #392](https://github.com/Archolith/menhir/pull/392) explored name-vector cosine (a proposed 0.40 floor) with a small hand-authored fixture. The fixture is insufficient to prove no embedding-derived signal exists; it does show a strong reason to test alternatives. "Seattle apartment" versus "Denver house" and "Seattle office" versus "Denver warehouse" demonstrate why cardinality/attribute meaning cannot be inferred from similar raw name cosines alone.

Distinguish three experimental targets:

- **Semantic overlap** (cosine-friendly).
- **Logical opposition/constraint satisfaction** (needs pairwise context).
- **Memory evolution** (needs valid time, provenance, and interpretation).

## 2. Vector operations to test

Assume E(s) ∈ ℝᵈ is a frozen encoder.

First-order probes:

\[
\Delta(s,t)=E(t)-E(s)
\]
\[
c(s,t)=\frac{E(s)\cdot E(t)}{\|E(s)\|\|E(t)\|}
\]

For a controlled operation T, compare the direction of Δ_T(s)=E(T(s))−E(s) across many unrelated sentences. If an operation is represented by a roughly stable direction, within-class directional cosine should exceed matched randomized controls *on held-out material*.

Potential operations:
- Paraphrase / synonym substitution.
- Antonym substitution.
- Explicit negation and double negation.
- Entity swap.
- Numeric or categorical value swap.
- Quantifier change ("all" ↔ "some").
- Time/scope change ("now" ↔ "previously", or specified intervals).

### The K residual

Let N be an explicit negation and P a semantics-preserving-substitution-of-subject operation (i.e. preserving sentence structure, not its exact factual truth). Define the four-corner interaction:

\[
K_{N,P}(s)=E(s)+E(N(P(s)))-E(N(s))-E(P(s))
\]
\[
k(s)=\|K_{N,P}(s)\|_2
\]

This is analogous to a **second finite difference** or parallelogram closure residual: it tests whether applying one operation changes the embedding effect of another. It is not the logical truth value of either statement.

Repeat for operation pairs N/V, P/V, N/T, V/T. A useful signal would show that K-derived features distinguish labels beyond simpler first-order features and that the association transfers to unseen relation types.

Control for:
- Encoder anisotropy and vector normalization.
- Length and lexical overlap.
- Word-order sensitivity.
- Transformation noncommutativity (N(P(s)) may differ semantically from P(N(s)) for quantifiers or scope).
- Small-denominator artifacts in normalized K.
- Leakage from the literal tokens used to generate classes.

### A lightweight learned geometry

Candidate features: cosine, ‖Δ‖, Hoyer sparsity(Δ), u⊙v, |u−v|, K-derived features, and optional attribute/scope metadata.

Learn a regularized mapping z=WΔ, with dimensions 2, 4, 8, 16, 32, 64, 128. Investigate logistic/ridge probes, LDA, and metric learning. PCA is useful to *visualize* directions; it is not an intrinsically supervised contradiction classifier.

Compare:
- Frozen vector features versus text-token heuristics.
- K features added versus withheld (feature ablation).
- Name vectors versus complete-sentence vectors.
- Global learned transformation versus per-relation transformations.
- Geometry only versus structure-aware constraints.

A frozen embedding may omit critical information or leave it linearly inaccessible. If the signal is absent, a stronger encoder, joint model, or explicit beliefs may be necessary; changing the math cannot reconstruct facts that were never encoded.

## 3. Visualizations worth building

### A. Pairwise scatter (first priority)

- X: cosine similarity.
- Y: a second independent score, such as Hoyer sparsity, residual-derived score, or calibrated projection.
- Each dot: a labeled statement pair.
- Colors: contradiction, equivalent, compatible, temporal update, unrelated.
- Hover: exact statements, metadata, source, and label rationale.

This exposes overlapping classes and suspicious outliers directly.

### B. Difference-vector arrow plot

PCA-project Δ vectors onto two axes and show arrow origins, endpoints, and transformation categories. Include explained variance. Two-dimensional proximity is only a visualization, not proof of linguistic structure.

### C. Dimensionality line graph

- X: projection dimension (2→128).
- Y: out-of-distribution contradiction recall at equal review budgets.
- Draw confidence intervals and lexical/cosine/NLI baselines.
- Watch whether the curve plateaus or fails to transfer.

This is a proper *line graph* because dimensionality is ordered.

### D. Four-point residual distribution

For each ordered operation pair (N/P, N/V, etc.), compare ‖K‖ distributions for held-out categories, alongside randomized-control and shuffled-label distributions.

### E. Temporal truth/update timeline

Plot valid time and source evidence for a single subject-attribute slot with A→B→A changes. This reveals the distinction between a contradictory assertion and a legitimate later state.

## 4. Literature map (selected primary sources)

These are relevant adjacent areas—not a claim that no one has explored vector arithmetic for NLI.

| Work | Date | Why it matters / limitation |
| --- | --- | --- |
| [Gated Residual Discrepancy for Efficient Bi-Encoder Factuality Verification](https://aclanthology.org/2026.emnlp-main.84/) | EMNLP 2026 | Dual-view vector discrepancy plus gating for factuality; strong neighboring baseline, not exactly a Menhir contradiction filter. |
| [Computation Over Geometry: Meaning Identity Is Computed, Not Shipped in the Embeddings](https://arxiv.org/abs/2609.28290) | Sep 2026 | Important caution that separate frozen encodings can fail to retain enough recoverable meaning-identity information. |
| [Semantically Rich, Metrically Affect-Blind](https://arxiv.org/abs/2610.32122) | Oct 2026 | Demonstrates that information recoverable by a linear probe need not be captured well by cosine. It studies sentiment, *not* contradiction. |
| [Typed Relation Path Geometry in Sentence Embedding Spaces](https://aclanthology.org/2026.emnlp-main.2530/) | EMNLP 2026 | Directed lexical-relation paths and compositional geometry. |
| [Meaning Contracts: Local Semantic Geometry in LLMs](https://arxiv.org/abs/2610.30715) | Oct 2026 | Local/nonlinear semantic geometry; concerns LLM representations, not directly frozen sentence encoders. |
| [Conceptual Orthospaces—An Embedding Framework Accounting for Negation Operators](https://link.springer.com/article/10.1007/s13218-026-00912-4) | Jun 2026 | Geometric formalization of negation operations. |
| [E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval](https://aclanthology.org/2026.groundlm-1.28/) | Oct 2026 | Subtractive penalty for excluded concepts in retrieval. |
| [eTracer: Polarity-Aware Evidence Grounding](https://aclanthology.org/2026.emnlp-main.314/) | EMNLP 2026 | Signed support/contradict/irrelevant scoring; a practical alternative to purely geometric methods. |
| [Geometry First: Shaping Sentence Embeddings with Controlled Singular Value Regularization](https://aclanthology.org/2026.emnlp-main.682/) | EMNLP 2026 | Embedding-space geometry and singular-value control. |
| [SparseCL: Contradiction Retrieval via Contrastive Learning with Sparsity](https://proceedings.mlr.press/v267/xu25s.html) | ICML 2025 | Contradiction retrieval using cosine/Hoyer-related sparsity; **trains** embeddings, unlike our frozen-encoder first experiment. |
| [Investigating Semantic Subspaces of Transformer Sentence Embeddings through Linear Structural Probing](https://aclanthology.org/2023.blackboxnlp-1.11/) | 2023 | Clear prior work on lower-dimensional sentence embedding probes including NLI. |
| [Counter-fitting Word Vectors to Linguistic Constraints](https://aclanthology.org/N16-1018/) | 2016 | Push/pull word-vector geometry for antonyms and synonyms. |
| [InferSent](https://aclanthology.org/D17-1070/) | 2017 | NLI models using pairwise embedding features such as |u−v| and u⊙v. |
| [SemAntoNeg](https://aclanthology.org/2022.blackboxnlp-1.20/) | 2022 | Antonym/negation tests; not a complete general contradiction benchmark. |
| [MemConflict](https://arxiv.org/abs/2605.20926) | May 2026 | Dynamic/static/conditional memory conflict framing. |
| [STALE](https://arxiv.org/abs/2605.06527) | May 2026 | Implicit temporal invalidation, important to distinguishing updates from contradictions. |
| [TSHMem: Temporal-Semantic Hierarchical Memory Graph](https://aclanthology.org/2026.emnlp-main.1034/) | EMNLP 2026 | Temporal-semantic memory graph architecture. |

### What might still be novel?

Not "using cosine for contradictions" or "linear projections for NLI"—both have adjacent prior art.

A narrower, falsifiable contribution could be:

> For **frozen production embedding vectors**, test whether **composed semantic transformations and their four-point residuals** provide compact, transferable signals that improve **contradiction triage in a temporally scoped memory system**, compared with lexical, cosine, and NLI baselines at the same review cost.

The potential novelty depends on the experimental result *and* a more exhaustive literature review. Results that show no useful residual information are still informative.

## 5. Menhir applicability

[Menhir #511](https://github.com/Archolith/menhir/issues/511) already proposes a single append-only :TypedAssertion log and additional belief-shaped View projections. It specifically highlights updates vs contradictions, single-value cardinality, attribute-name drift, evidence tiers, producer limitations, and surfacing.

‖K‖ should not duplicate the graph's belief log or independently declare truth. It can investigate candidate alignment/incompatibility scores; structured projections handle exclusivity, time, and provenance. Any integration should start as read-only/shadow diagnostics with replay parity.

## 6. Interpretation limits

- "Alice is not happy" can be near "Alice is happy" in embedding space even though the explicit propositions conflict for a fixed time/context.
- "Open" and "closed" are not always factual contradictions; a door may be open at one time and closed later.
- "Lives in Austin" vs "owns a house in Denver" is compatible because they describe different predicates.
- Similarity/discrepancy geometry is not a substitute for evidence reliability.
- A 2D projection can hide high-dimensional effects; charts are diagnostics, not standalone validation.
- Do not present synthetic or illustrative graph values as measured experimental performance.

## 7. First research decisions still open

1. What primary target should be optimized: candidate-recall at fixed budget, ranking quality, or calibrated per-pair classification?
2. How many *real-world, human-audited* cases can be assembled without using sensitive user data?
3. Which embedding models can be tested without changing the Menhir baseline?
4. Is K meaningful after sentence length and lexical confounds are controlled?
5. How much additional value does geometry provide once the subject/attribute/time constraints are available?

These decisions should be answered through documented runs, not intuition.
