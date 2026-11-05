# ‖K‖: phased implementation and testing plan

**Version:** 0.1 — research protocol, November 5, 2026  
**Expected effort:** approximately 30 focused engineering days (4–6 calendar weeks)  
**Initial gate:** two weeks through Phase 3  
**Upstream context:** [Menhir #511](https://github.com/Archolith/menhir/issues/511); [Menhir #392](https://github.com/Archolith/menhir/pull/392)

## 1. Problem definition

We seek a cheap, well-calibrated detector of contradictions *between statements about the same real-world fact*. Similarity, negation, incompatibility, and temporal change are not equivalent.

Decompose the decision into three questions:

1. **Fact identity:** Do A and B share subject, attribute, temporal scope, and relevant qualifiers?
2. **Value compatibility:** Under the attribute's cardinality/constraints, can their values both be true?
3. **Memory status:** Is the apparent disagreement a legitimate update, correction, uncertainty, or unresolved conflict?

Embedding geometry is a *candidate signal*, not a substitute for timestamps, evidence, and belief semantics.

### Pre-register hypotheses

- **H0 (null):** On held-out data, frozen-vector arithmetic adds no useful contradiction detection beyond lexical heuristics, simple cosine, and an NLI baseline.
- **H1 (direction):** Controlled semantic operations produce statistically aligned vector differences across unseen content.
- **H2 (interaction):** Four-point residual magnitude ‖K‖ contains information not supplied by cosine, vector difference, or lexical cues.
- **H3 (projection):** A low-dimensional linear/regularized projection recovers generalizable contradiction-related features.
- **H4 (hybrid):** Structured fact identity and temporal scope combined with geometry improve useful recall at a fixed review budget.

We should attempt to falsify these. A negative result is acceptable.

## 2. Suggested project layout

Proposed, not yet implemented:

~~~text
norm_k/
  __init__.py
  embeddings.py          # encoder registry, cache, batch control
  transforms.py          # controlled synonym, antonym, negation, entity/value swap
  geometry.py            # cosine, delta, Hoyer, K, normalized K
  evaluation.py          # classification, bootstrap CI, calibration, review budgets
  probes.py              # regularized linear and low-rank models
  belief_features.py     # typed subject/attribute/scope comparison
data/
  schemas/
    pair.schema.json
  generated/
    README.md            # generator and seed, not secrets / private data
  splits/
    manifest.json
experiments/
  phase2_baselines.py
  phase3_interactions.py
  phase4_projections.py
  phase5_beliefs.py
reports/
  README.md
tests/
  test_geometry.py
  test_transform_invariants.py
  test_split_integrity.py
  test_reproducibility.py
docs/
  research-plan.md
  research-notes.md
~~~

Python + NumPy + scikit-learn + Matplotlib/Plotly is an appropriate first implementation. Use Parquet/JSONL for local output. All generated data and model outputs must be traceable to a run configuration and seed.

## 3. Dataset and labeling contract

Target **3,000–5,000 controlled pairs**, with independent held-out sets.

Label each pair with:

- Unique stable ID and source/generator.
- Text A and B; versions with short entity names and full statements.
- Pair class: contradiction, equivalent, compatible, temporal update, unrelated, or genuinely ambiguous.
- Binary annotations: same subject, same attribute, same effective time, incompatible value.
- Attribute cardinality (single-valued / multi-valued / unknown).
- Valid-time range and evidence/provenance where relevant.
- Transformation(s): negation, synonym, antonym, value change, entity swap, temporal qualifier, quantifier, etc.
- Template family, subject family, relation family, vocabulary group.
- Human-review status and annotation rationale.

**Do not infer labels solely from string templates.** For example, "lives in Seattle" and "lives in Denver" may be different points in time, and "owns a dog" does not rule out "owns a cat."

Balance the dataset for hard negatives, such as near-identical sentences that are compatible, and lexically distant sentences that entail the same claim.

### Split design

- Exploratory/train split for fitting projections.
- Validation split for thresholds and model selection.
- Sealed test split for final score.
- Separate out-of-distribution holdouts by **template**, **subject**, **lexical vocabulary**, and **relation type**.
- Separate real-worded, human-verified cases (not synthetic template-only).
- Keep the Menhir #392 42-pair fixture and SemAntoNeg as **evaluation-only reference sets**; evaluate their limitations rather than presenting them as conclusive ground truth.

Record embedding model, dimensions, provider/version, normalization, preprocessing, tokenization, API settings, and cache hashes. Never combine incompatible vector spaces.

## Phase 1 — Reproducible laboratory (days 1–4)

### Implementation

- [ ] Establish package, dependency lock, tests, CI.
- [ ] Implement frozen text-embedding-3-small adapter with batch limits, retry, local cache, token/call counters.
- [ ] Add optional local embedding model as an explicitly separate experiment arm.
- [ ] Define JSONL schema and deterministic dataset generator.
- [ ] Produce first 3,000–5,000 pairs across all categories, including confounders.
- [ ] Audit labels: review all complex counterexamples and a stratified sample of simpler pairs.
- [ ] Freeze evaluation splits and store manifest with checksums.
- [ ] Implement command-line execution and manifest recording.

### Tests

- Cache round-trip, invalid vector dimension, empty text, duplicate records, reproducible seeds.
- Schema checks for categories, time scopes, cardinality, and rationale.
- Split-integrity tests ensuring held-out families are truly disjoint.
- Verify ordered input maps to stable output IDs even with retries.

### Exit criteria

A new engineer can run the same command and obtain identical dataset memberships and the same metrics from the same cached vectors. Any labeling disputes are either resolved or explicitly marked ambiguous and excluded from binary scoring.

## Phase 2 — Baseline and exploratory visualization (days 5–8)

### Implement

For embeddings u=E(A), v=E(B):

- Cosine similarity.
- Euclidean distance and squared distance.
- Difference Δ=v−u and its norm.
- Elementwise product u⊙v.
- Hoyer sparsity of Δ.
- Token-level rule baseline (including symmetric differences, explicit negation markers, numeric mismatches).
- Strong baseline: local NLI cross-encoder; optional LLM judge sample.
- Negative control: shuffled labels and random feature projection.

Evaluate names-only versus full-sentence inputs separately. The vectors stored by Menhir's existing conflict path are names; full-sentence successes do not automatically transfer.

### Plots

1. **Scatter:** cosine on X; an independent feature (sparsity or a calibrated score) on Y; label as color.
2. **PCA arrows:** project difference vectors onto 2D; show direction clusters by transformation, with explained variance.
3. **Distribution:** overlays of cosine, Δ norms, and sparsity by category, with quantiles.
4. **Confusion/pr-curve:** false positives and negatives for fixed review budgets.

Do not interpret a visually appealing PCA/UMAP cluster as a predictive result.

### Exit criteria

Record per-class recall, precision, PR-AUC, calibration, latency, and **recall at fixed candidate-review budgets** (e.g. 5%, 10%, 20%; adjust with actual Menhir volumes). Identify the largest unresolved failure groups.

## Phase 3 — Core ‖K‖ hypothesis test (days 9–14)

### Define transformations

Let E(s) map text to a frozen d-dimensional vector. N(s) is a controlled negation and P(s) is a controlled subject substitution. Also define V(s) as a value change and T(s) as a temporal qualification.

First differences:

\[
\Delta_N(s)=E(N(s))-E(s), \qquad
\Delta_P(s)=E(P(s))-E(s)
\]

Four-point interaction:

\[
K_{N,P}(s)=E(s)+E(N(P(s)))-E(N(s))-E(P(s))
\]

Scalar measurement:

\[
k_{N,P}(s)=\|K_{N,P}(s)\|_2
\]

Scale-relative variant (exploratory, ε chosen before evaluation):

\[
k_{\mathrm{rel}}(s)=
\frac{\|K_{N,P}(s)\|_2}
{\|\Delta_N(s)\|_2+\|\Delta_P(s)\|_2+\varepsilon}
\]

Interpretation: If operations are approximately additive and independent, K is near zero. Nonzero K indicates interaction, which *may or may not* relate to logical incompatibility.

### Experiments

- [ ] Test N/P, N/V, P/V, N/T, and V/T pairs; inspect their distinct distributions.
- [ ] Test transformation-vector directional alignment across unseen subjects and attributes.
- [ ] Compare synonyms, antonyms, value changes, paraphrases, negated antonyms, double negation, and quantified statements.
- [ ] Compare raw K, relative K, cosine, and Δ features.
- [ ] Counterbalance transformation order; test whether P(N(s)) and N(P(s)) are actually semantically equivalent for each generated case.
- [ ] Exclude ill-formed or meaning-changing transformations with independent validity checks.
- [ ] Run permutation tests and paired bootstrap confidence intervals.
- [ ] Measure discriminative contribution on the *sealed* evaluation sets only after feature selection is frozen.

### Unit and metamorphic tests

- Identical vectors give zero difference.
- Vector addition/subtraction and norm calculations agree with NumPy references.
- For truly additive synthetic encoders, K=0 to numerical tolerance.
- Altering encoder output scale changes raw K but not normalized comparisons in the expected way.
- Double negation preserves meaning only where linguistic scope really permits it.
- Reordering pair A/B should not change a symmetric contradiction label, even if directional features change.

### Exit criteria

At least one K-derived feature gives a reproducible out-of-sample gain over matched baselines, with confidence interval and realistic effect size, or conclude there is no evidence to warrant K-specific modeling. Do not relabel K magnitude as a contradiction probability.

## Phase 4 — Low-rank and learned geometry (days 15–20)

### Candidate models

1. Regularized logistic regression on cosine, Δ norm, Hoyer, and lexical controls.
2. Same model **plus** K and normalized K features (strict ablation).
3. Ridge/logistic probe over Δ.
4. Projections z=WΔ with ranks 2, 4, 8, 16, 32, 64, 128.
5. Supervised LDA or metric learning, explicitly marked as trained.
6. Only if necessary: a tiny nonlinear model with regularization.

Also test standard pair feature combinations [u,v,|u−v|,u⊙v], with sufficient regularization to avoid fitting thousands of parameters to a small dataset.

### Tests

- Same splits and review budgets as Phase 2.
- Hold out entire relations, not merely rows. Test e.g. number/ownership relations after training on polarity and locations.
- Run ablations for K, lexical features, negation tokens, and structured metadata.
- Plot recall versus projection dimension and precision–recall tradeoffs.
- Compare confidence intervals via paired bootstrap.
- Check model calibration and errors by negative category.
- Check sensitivity to model seed and normalization.

### Exit criteria

Prefer the smallest model with stable out-of-distribution gains and acceptable latency. If a word-difference baseline matches the classifier, the result is not geometric generalization.

## Phase 5 — Belief-aware evaluation (days 21–25)

### Independent hypothesis

P(contradiction | A,B) can be *modeled*, not necessarily exactly factored, as:

\[
P(\mathrm{same\ fact}\mid A,B)\cdot
P(\mathrm{incompatible}\mid\mathrm{same\ fact},A,B)
\]

Then a separate stage resolves temporal change versus live conflict using evidence. Never imply this is a guaranteed calibrated probabilistic identity without assumptions.

### Menhir-related read-only work

- [ ] On a non-production/test graph, count same-slot equal-valid-time ties.
- [ ] Count A→B→A flip-backs, latest-value changes, and explicit correction semantics.
- [ ] Catalog name drift (home/residence/home_city) and subject aliases.
- [ ] Estimate which attributes are exclusive or multi-valued, recording unknown when unsupported.
- [ ] Investigate project facts not produced by the user-only scalar writer.
- [ ] Compare embeddings-only, belief-only, and combined features on the exact same examples.

### Prototype constraints

A candidate ViewKind is a **pure projection** of Menhir's existing append-only :TypedAssertion evidence. It does not become a competing source of truth. It preserves competing values, valid times, labels, and provenance; it suppresses nothing.

### Exit criteria

Report whether structured constraints fix errors geometry cannot, and whether geometry contributes after those constraints are included. Demonstrate live/replay equivalence in tests before discussing production wiring.

## Phase 6 — Final benchmark and integration decision (days 26–30)

### Compare systems

| System | Question answered |
| --- | --- |
| Existing #392 cosine floor | What does Menhir already catch/drop? |
| Lexical and numeric rules | Can simple rules match geometry? |
| Cosine + vector features | What does raw frozen geometry recover? |
| K / projected geometry | Does the new hypothesis add measurable signal? |
| Local NLI | What can a purpose-built sentence-pair model do? |
| LLM judge | What is the strongest available reference, at what cost? |
| Structured belief constraints | What is recoverable without geometry? |
| Hybrid | Does the combination deliver meaningful practical gains? |

### Primary metrics

- Contradiction recall at a **fixed review budget** (equal number of candidates forwarded for review).
- Per-category false-negative rate, notably implicit contradiction and legitimate updates.
- Precision, PR-AUC, calibration, and paired confidence intervals.
- End-to-end candidate throughput, CPU/GPU memory and latency, and judge/embedding call counts.
- False-positive review burden.
- Performance with the local LLM server active on its GPU.

### Decision gates

**Ship a shadow experiment only if:**
- Gains survive sealed out-of-distribution sets and error audits.
- The feature has positive marginal contribution in ablations.
- The tradeoff against latency/calls is favorable.
- The output can be calibrated or treated as a non-authoritative ranking.

**For Menhir integration:** start default-off, log-only shadow predictions, compare without changing persistence/review decisions. A successful result may replace or augment the *candidate filter*, but must not silently change authority decisions or write a second belief store. If a filter is allowed only to drop candidates, validate its false-negative cost before enablement.

**Stop if:** label-shuffled results, token-difference controls, or OOD failures erase the gain. Publish the negative result rather than force a production feature.

## 4. Milestones and reproducibility artifacts

| Milestone | Days | Artifact |
| --- | --- | --- |
| M1 | 1–4 | Audited dataset, schema, encoder cache, tests |
| M2 | 5–8 | Baseline report, similarity scatter, PCA arrows |
| M3 | 9–14 | K notebooks/scripts, residual distributions, statistical test |
| M4 | 15–20 | Low-rank probes, ablations, dimension/performance graph |
| M5 | 21–25 | Belief-aware error taxonomy, read-only prototype |
| M6 | 26–30 | Final scorecard, cost report, written go/no-go decision |

All reports should include source revision, run command, random seed, exact splits, embedding model identity, hardware, and generated output hashes.

## 5. First pull request scope

A small, reviewable PR after this planning commit should contain **only Phase 1 foundation**:

- Package scaffolding and dependencies.
- Typed example schema.
- Initial deterministic sample generator.
- Embedding cache interface with a fake encoder for tests.
- Geometry primitives plus analytical unit tests.
- One reproducible CLI.
- CI that runs without an external API key.

Do **not** make Menhir production changes or train an opaque large model in the first PR.

## 6. Research boundaries

- Never upload real user memories to a public dataset.
- Keep synthetic data and private Menhir evaluation data separate.
- Do not equate "antonym" with "contradiction" outside context.
- Do not infer logical or temporal facts that are absent from the text or metadata.
- Do not claim novelty before a complete prior-art review.
- The brand/name ‖K‖ (Norm K) remains provisional pending availability checks.
