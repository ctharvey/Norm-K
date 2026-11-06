# ‖K‖ / Norm K — research rationale and engineering handoff

**As of November 5, 2026**  
**Repository:** [ctharvey/Norm-K](https://github.com/ctharvey/Norm-K)  
**Originating system:** [Archolith/menhir](https://github.com/Archolith/menhir)  
**Related Menhir work:** [#511 — belief-shaped contradiction detection](https://github.com/Archolith/menhir/issues/511), [#392 — conflict similarity floor](https://github.com/Archolith/menhir/pull/392)

**Handoff status:** Offline research infrastructure is implemented and tested.
We have **not** demonstrated that the proposed geometric features detect
real-world contradictions. All numerical performance experiments so far use a
deterministic lexical hashing control and synthetic, unaudited statements.

## 1. Executive summary

‖K‖ (spoken **Norm K**) is a separate research project asking:

> Can a cheap, mathematically grounded function of **frozen text embeddings**
> distinguish logically incompatible facts from paraphrases, related facts,
> unrelated assertions, and changes over time—especially when paired with
> explicit subject/attribute/time information?

Our motivating idea was that embeddings are vectors, not opaque labels.
We can add and subtract them, measure direction and distance, compare the
effects of individual lexical operations, and use differential/statistical
tools to test whether contradiction-related information survives in a form
that simple cosine similarity misses.

The **specific experimental hook** is a four-point interaction residual K,
whose norm ‖K‖ measures how one controlled transformation changes the
effect of another. This is a proposed feature—not a discovered contradiction
equation, not a truth score, and not yet an independently established novel
algorithm.

The project is deliberately **independent of Menhir production**. A positive
result might become a fast candidate filter or uncertainty trigger before an
NLI/LLM judge. The Menhir belief/View work remains responsible for temporal
scope, exclusive attributes, evidence provenance, and authoritative memory
decisions. Both paths can produce useful results independently.

## 2. How we arrived here

### 2.1 Initial question: why not use the actual embedding numbers?

The original discussion proposed examining embedding mathematics at smaller
granularity: individual words, synonyms, antonyms, negation, and replacement
of a statement's value. If vectors encode semantic meaning, perhaps a change
such as **open → closed**, **owns → does not own**, or **Austin → Denver** leaves
a detectable geometric signature. The user specifically asked about adding,
subtracting, calculus/statistics, line plots, and dot plots across terms.

That is a reasonable **hypothesis**, but it does not follow that antonym
directions are universal or that high-dimensional spaces directly implement
logical truth. The experiment must determine what information is available
and what remains missing.

### 2.2 Menhir exposed the limits of cosine-only approaches

[Menhir #392](https://github.com/Archolith/menhir/pull/392) explored a similarity
floor for conflict review. The associated small investigation used
`text-embedding-3-small` vectors for **short names**, not whole statements.
[Issue #511](https://github.com/Archolith/menhir/issues/511) reports that
ordinary name cosine thresholds failed to flag most contradictory pairs in
roughly 40 hand-written cases, while semantically related but compatible
claims could have overlapping similarity scores.

Example distinction from #511:

- An exclusive **home/residence slot** may permit only one primary value at a
  given valid time.
- Two separate **office/warehouse** attributes or multiple owned objects can
  simultaneously hold different values without contradiction.

The original pairs included “Seattle apartment / Denver house” versus
“Seattle office / Denver warehouse.” Their similarity alone does not express
the relevant attribute/cardinality assumptions. A high cosine can also occur
for a proposition and its negation because they share almost all content.

**Important correction to the shorthand in #511:** these early name-vector
experiments establish limitations of the *tested cosine setup*, not that no
frozen embedding transformation can recover contradiction signals. Norm K
tests that more precise claim rather than assuming either answer.

### 2.3 We separated logical incompatibility from memory evolution

Consider:

1. “Alice lives in Austin.”
2. “Alice moved to Denver.”

These may describe a legitimate transition. A memory system must distinguish
*valid at different times* from *conflicting values for the same fact and
time*. Unstated time, cardinality, and provenance cannot reliably be recovered
by manipulating text vectors. Therefore, the work needs two lanes:

1. **Numerical lane:** investigate cheap geometric features of statement pairs.
2. **Belief lane:** represent the subject, attribute, value, effective time,
   cardinality, source, and correction semantics explicitly.

An approximate working decomposition is:

\[
P(\text{contradiction}\mid A,B) \approx
P(\text{same fact}\mid A,B)\;\times\;
P(\text{incompatible values}\mid\text{same fact},A,B)
\]

This is a **modeling heuristic**, not a calibrated probability identity unless
its assumptions hold. Temporal supersession is a separate decision.

### 2.4 Why isolate the experiment as ‖K‖?

We wanted an exploration that can yield an honest negative result without
complicating Menhir's fold, storage, or production conflict path. A small
standalone Python package supports deterministic examples, cached vectors,
controlled transformations, plots, model ablations, and repeatable evaluation.

The name **‖K‖** comes directly from the norm of the residual. “Norm K” is a
provisional project/research title; the user also raised it as a possible
future company/product identity if the work produces something substantive.
Trademark/domain clearance has **not** been performed.

## 3. The mathematical research program

### 3.1 First-order vector arithmetic

Let E(s)∈ℝᵈ be a **frozen** text embedding. For statement pair A,B:

\[
\Delta(A,B)=E(B)-E(A)
\]

Initial measurements are cosine similarity, Euclidean distance, ‖Δ‖, mean
absolute coordinate change, coordinate-wise product, and Hoyer sparsity of
Δ. A basic token/number/negation-difference baseline is included because a
complex geometric method must beat cheap lexical heuristics on unseen data.

For a controlled semantic operation T, compare:

\[
\Delta_T(s)=E(T(s))-E(s)
\]

Do negation, synonym replacement, antonym substitution, subject changes,
value changes, and temporal qualification produce directions that align
across **new subjects and relations**? If not, a universal linear direction
is unlikely to generalize.

### 3.2 Four-point finite difference: the original K

Let N negate a statement and P replace its subject in a controlled way:

\[
K_{N,P}(s)=E(s)+E(N(P(s)))-E(N(s))-E(P(s))
\]

\[
\|K_{N,P}(s)\|_2
\]

This is analogous to a discrete **second-order interaction** or a
parallelogram-closure residual. When the two transformations' embedding
effects are additive and independent, K tends toward zero. A large norm
indicates context interaction, **not necessarily contradiction**.

We also test a scale-relative variant:

\[
k_{rel} = \frac{\|K\|_2}
{\|E(N(s))-E(s)\|_2+\|E(P(s))-E(s)\|_2+\epsilon}
\]

Normalization, anisotropy, lexical overlap, operation order, quantifier
scope, and grammatical validity must all be controlled.

### 3.3 The practical bridge: a K feature for a pair A/B

The original square had an engineering gap: real candidate pairs arrive as
A and B, not as a hand-constructed four-corner experiment. We added a
**restricted, explicit counterfactual T** that is applied to both statements:

\[
K(A,B;T)=E(A)+E(T(B))-E(T(A))-E(B)
\]

The norm measures whether the A→B displacement changes under T. The present
code supports a limited negation grammar and consistent subject replacement.
Unsupported examples are **skipped**, never silently scored as zero. The
counterfactual grammar is syntactic and needs semantic validation before
use on unrestricted prose. See [pair-level-k.md](pair-level-k.md).

### 3.4 Low-rank and statistical questions

A frozen vector might encode useful distinctions in directions that a
standard cosine ignores. Probe small dimensions:

\[
z=W\,\Delta(A,B)
\]

Test regularized linear/logistic classifiers, PCA(|Δ|) projections, and
low-rank or learned metrics. Compare ranks 2, 4, 8, 16, 32, 64, 128. Always
compare **with versus without K features** on the exact same eligible pairs.

Scoring should focus on recall at a fixed contradiction-review budget,
per-category missed contradictions, PR-AUC, precision, calibration, latency,
and calls avoided. Use held-out subjects, vocabulary, sentence templates,
and **entire relation families**; report uncertainty and failure clusters.

### 3.5 Graphs were chosen as diagnostic instruments

- **Dot/scatter:** cosine on X, second feature or learned score on Y; one
  labeled pair per dot, including outlier inspection.
- **Difference-vector arrows:** PCA projection of E(B)−E(A), colored by
  transformation, with explained variance displayed.
- **Line graph:** projection dimension or review budget (an *ordered* X) versus
  held-out recall with confidence intervals.
- **K residual distributions:** operation-by-operation histograms and null
  controls to see interaction effects and confounding.
- **Temporal timeline:** subject/attribute values over valid time and source
  evidence to distinguish updates from simultaneous disagreement.

Two-dimensional plots can hide high-dimensional effects. They are useful for
investigation but do not establish predictive performance by themselves.

## 4. Relevant prior work and why it did not settle our question

This is a selected reading map, not a completed novelty search:

| Primary work | Relevance and caution |
| --- | --- |
| [SparseCL (ICML 2025)](https://proceedings.mlr.press/v267/xu25s.html) | Contradiction retrieval with sparsity-related features; it **trains embeddings**, unlike our frozen-vector starting point. |
| [InferSent (2017)](https://aclanthology.org/D17-1070/) | Pairwise embedding differences/products and NLI training have longstanding precedent. |
| [Linear Structural Probing (2023)](https://aclanthology.org/2023.blackboxnlp-1.11/) | Semantic subspaces and NLI signals have been probed before; low-rank projections alone are not novel. |
| [Gated Residual Discrepancy (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.84/) | Efficient vector-discrepancy factuality checks suggest a practical adjacent direction, but are not our memory conflict task. |
| [Semantically Rich, Metrically Affect-Blind (Oct 2026)](https://arxiv.org/abs/2610.32122) | Linear recoverability can exceed cosine separability; the paper studies **sentiment**, not fact contradiction. |
| [Computation Over Geometry (Sep 2026)](https://arxiv.org/abs/2609.28290) | Critical skeptical counterweight: independently frozen embeddings can discard or obscure meaning identity; a joint model may be needed. |
| [Typed Relation Path Geometry (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.2530/) | Directed relation paths and composition in sentence embedding spaces. |
| [E-SENS (2026)](https://aclanthology.org/2026.groundlm-1.28/) | Subtractive negative-constraint retrieval; adjacent to excluding incompatible concepts. |
| [eTracer (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.314/) | Signed support/contradiction evidence grounding is a stronger purpose-built comparison. |
| [MemConflict (May 2026)](https://arxiv.org/abs/2605.20926) and [STALE (May 2026)](https://arxiv.org/abs/2605.06527) | Dynamic/static/conditional memory conflicts and implicit temporal invalidation show why factual updates are not simple antonyms. |

We are **not** claiming that vector arithmetic, antipodal directions, or
low-rank NLI features have never been explored. The narrower possible
contribution is a reproducible study of **composed transformations on frozen
production vectors**, their transfer across unseen relations, and their
utility for **advisory contradiction triage** with explicit belief scope.
Only a positive, replicated result plus a deeper literature review could
support a novelty claim.

## 5. Relationship to Menhir, including what not to change

[Menhir #511](https://github.com/Archolith/menhir/issues/511) proposes the
longer-term belief-shaped solution:

- Keep **`:TypedAssertion` as the single append-only evidence log**.
- Build a **pure ViewKind projection**, beside `scalar_state` and
  `scalar_history`, preserving competing values, valid times, and labels.
- Keep live and replay fold behavior identical.
- Begin **advisory-only**: do not silently suppress evidence or alter authority.
- Investigate exclusive versus multi-value attributes, free-text attribute
  drift (`home`/`residence`), producer limitations for project facts, and how
  explicit corrections, restatements, source links, and evidence tiers interact.
- Distinguish same-time ambiguous values from valid chronological changes and
  A→B→A flip-backs.

Norm K does **not** create another belief database and does **not** decide
what Menhir should remember. If a numeric signal works, it can help rank/drop
candidates for review or flag uncertainty. Any integration must be behind a
default-off/shadow flag, preserving the current route when no result is
available. Compare against #392 and the actual NLI/LLM reviewer before ship.

## 6. What is already built in Norm-K

### 6.1 Core laboratory

- Python package with NumPy and optional scikit-learn/Matplotlib.
- Synthetic generator: **3,000 pairs**, 600 in each of contradiction,
  equivalent, compatible, temporal update, and unrelated classes.
- Four transformation-square families; seeded control data; cached embeddings.
- Frozen-vector `HashingEncoder` **control** (256 dimensions; unigram/bigram
  lexical hash, **not semantic**) and an opt-in, capped
  `text-embedding-3-small` adapter using an existing configured API key.
- `--max-new=0` prevents surprise live embedding calls by default.
- Cosine/distance/difference/Hoyer/K functions, baseline review-budget
  metrics, linear/logistic probes, PCA(|Δ|) rank tests, SVG plots.

### 6.2 Dataset integrity and fair testing

- Subject-, template-, and relation-family holdout modes.
- Symmetric duplicate-pair detection; fixes for reversed duplicates and
  secondary-subject leakage across subject splits.
- Fingerprinted audit and **exact-statement purging** between train,
  validation, and test, always preserving the held-out test set first.
- Matched-cohort pair-level K ablation; restricted negation/subject-swap
  counterfactual coverage and class breakdowns.
- Descriptive bootstraps and paired subject-cluster AP intervals. The code
  refuses intervals from too few independent subject clusters.

### 6.3 Blind human review and external data intake

- Deterministically balanced review queues using **opaque IDs** (not labels
  embedded in IDs).
- Localhost-only (127.0.0.1) browser UI with origin/host checks and a
  per-process token on review writes.
- Hidden reference/model labels; reviewer confidence and notes.
- Append-only JSONL review history with latest-view corrections.
- Two-reviewer unanimous consensus, disagreements/ambiguity preserved, and
  pairwise Cohen's kappa / observed agreement.
- Offline CSV/TSV/JSONL importer with explicit label mapping, source file
  SHA-256, source/license provenance placeholders, and raw-label retention.
- Queue/annotation/external source data and generated reports are git-ignored.

### 6.4 Verification

**50 automated tests passed** as of this handoff, including local HTTP API
tests and an end-to-end offline CLI workflow. A Python wheel was built.
Tests do not validate semantic correctness, UI layout in a user's browser,
or performance on real embeddings.

The historical engineering-only run notes are in:

- [First offline smoke](../reports/offline-smoke.md)
- [Pair-level K ablation](../reports/offline-ablation.md)
- [Labeling/import smoke](../reports/offline-labeling-smoke.md)

## 7. What the offline numbers actually revealed

On the **synthetic lexical hash control only**, K-feature ablations showed:

| Held-out dimension | Negation K Δ average precision | Subject-swap K Δ average precision |
| --- | ---: | ---: |
| New subjects | +0.1464 | +0.0062 |
| New sentence templates | 0.0000 | 0.0000 |
| New relation families | **−0.0324** | **−0.0060** |

These values are **not evidence that K understands contradiction**. Indeed,
the result is a useful **warning**: a seemingly promising feature on a
subject-held-out synthetic benchmark failed to transfer to unseen relation
types. Our project should be designed to uncover exactly this failure before
calling anything a discovery.

The pair-level counterfactual coverage in that control was about **75.4%**
for restricted negation and **80%** for subject replacement; other examples
are unsupported, not false or scored as zero. The same-cohort ablation matters
because a score that is simply missing for one category can itself leak a
label. See the [ablation report](../reports/offline-ablation.md).

## 8. The phased plan from here

This roadmap supersedes *estimated timing*, not the earlier hypotheses.
Go/no-go gates matter more than hitting a calendar date. The detailed
engineering basis is in [research-plan.md](research-plan.md), and unresolved
dependencies live in [BLOCKERS.md](../BLOCKERS.md).

### Phase A — Establish a credible labeled evaluation set (next priority)

**Can partly proceed offline.**

- Read [annotation-protocol.md](annotation-protocol.md).
- Run a queue of at least 20 varied examples per class, independently
  double-reviewed; include same-subject compatible pairs, explicit temporal
  transitions, negated antonyms, numbers, scope/quantifiers, and ambiguous
  cardinality.
- Obtain a properly licensed external set (SemAntoNeg and memory-conflict
  sources are candidates), record paper/source/license/version, and **do not
  map NLI entailment to equivalence or neutral to compatible automatically**.
- Adjudicate disputes; track kappa, class counts, and per-source label rules.
- Seal train/validation/test manifests by template, subject, relation, and
  source. Keep naturalistic text separate from synthetic template families.

**Exit:** An independently reviewed, provenance-preserving evaluation
cohort with unresolved cases identified—not just generated ground truth.

### Phase B — Real frozen-embedding baseline (requires configured API/budget)

- Use `text-embedding-3-small` first **because Menhir uses it**; do not silently
  assume an alternative vector space transfers.
- Run **short names and complete sentences separately**.
- Cap and cache API usage; measure embedding cost/latency and unique misses.
- Compare cosine thresholds, distance, Δ norms, Hoyer sparsity, lexical
  difference rules, shuffled labels, and an NLI baseline.
- Report class-specific false negatives and recall at fixed review budgets.

**Exit:** A frozen, reproducible baseline and an honest estimate of how much
room for improvement remains.

### Phase C — Four-point counterfactual geometry and null controls

- Validate T(A)/T(B) meaning, scope, and grammar independently.
- Expand controlled transformation families (synonym, antonym, subject,
  value, negation, time), but skip invalid counterfactuals.
- Measure directional alignment and raw/relative K distributions.
- Add randomized four-corner null controls, length/overlap controls,
  counterbalanced transformation order, and grouped uncertainty estimates.
- Check whether the effect survives held-out **relations**, not just familiar
  syntactic templates.
- Compare K against a baseline holding every pairwise score K is built from,
  plus lexical features. Exclude mirror pairs, where T swaps A and B and K
  reduces to 2(A−B). See the
  [frozen protocol](preregistration-matched-v1.md).

**Exit:** A K feature shows repeatable information beyond that baseline, **or**
the negative result is recorded and K-specific development stops.

### Phase D — Learned projections and marginal contribution

- Fit regularized low-rank WΔ / pairwise embedding features only on train.
- Test dimensions 2–128 and compare a global transformation with per-relation
  ones; try nonlinear models only after simple baselines.
- Run explicit **with/without K** ablations on the same eligible rows, using
  the Phase C baseline. Plain with/without-K comparisons can credit K for what
  the two extra embedded sentences contribute.
- Verify transfer to unseen subject/template/relation/source families, with
  grouped confidence intervals and multiple-comparison discipline.
- Inspect false positives and misses, not just one aggregate metric.

**Exit:** Generalizable performance with a clear marginal gain, not a shortcut
learned from “not” or generated templates. If not, stop or change encoder.

### Phase E — Belief-aware branch on a safe Menhir graph

- Use a sanitized non-production `:TypedAssertion` snapshot; do **not** modify
  live memory state during initial inspection.
- Count same-time ties, A→B→A flips, corrections, plain updates, and
  attribute-name drift; inspect cardinality assumptions.
- Prototype a pure competing-beliefs ViewKind over the existing evidence log.
- Compare structure-only, geometry-only, and combined candidate triage on
  **the same** pairs, preserving source/valid-time metadata.
- Test deterministic live/replay parity and zero authority-path mutations.

**Exit:** Evidence of what geometry contributes after structural facts are
available. A useful belief projection can advance even if K fails.

### Phase F — End-to-end shadow evaluation and decision

- Benchmark against the shipped/current #392 candidate filter and local NLI
  and/or LLM judge, measuring actual candidate counts per scan.
- Under concurrent GPU/LLM load, measure cost per avoided review, CPU/GPU
  latency, false negative rate, precision, PR-AUC, and calibration.
- Use the same fixed review budget across methods; inspect difficult
  conditional and temporal examples.
- Run **default-off, log-only shadow mode** before any candidate suppression.
- Create a separate Menhir PR only if real held-out and shadow measurements
  justify it. The result must not rewrite `:TypedAssertion`, create a second
  belief store, or silently decide truth.

**Exit:** Ship a carefully scoped advisory optimization, publish a qualified
research finding, or publish an informative negative result.

## 9. Immediate implementation backlog (for the next engineer)

In priority order:

1. **Benchmark curation:** use the current local review CLI to prepare a
   balanced cohort, import an appropriately licensed external file, audit
   label semantics, and record a fixed manifest. Do not claim actual human
   labels have been collected until independent reviewers complete it.
2. **Counterfactual validity harness:** add explicit grammar/scope validity
   flags and randomized controls, with tests that detect bad paraphrases and
   unintended operation changes. See `norm_k/counterfactuals.py` and
   `norm_k/ablation.py`.
3. **Visualization/inspection:** add individual-example hover inspection and
   error drill-down, including residual arrows and reasons a pair was skipped.
4. **Scientific scorecard:** freeze baselines, per-class support, review budgets,
   significance criteria, model/seed/split version, and cost accounting.
5. **Real encoder run:** once a configured embedding credential and spending
   cap exist, run a small capped name-vs-sentence pilot before embedding all
   3,000+ pairs and counterfactuals.
6. **NLI comparator:** install/evaluate a suitable local pairwise classifier
   with an explicit model/license pin; benchmark under shared GPU load.
7. **Menhir test-graph analysis:** only with a safe export or test graph,
   proceed with read-only belief slot analysis and replay tests.
8. **Hosted CI and packaging:** local tests and a wheel build already work;
   enable hosted runner jobs only when repo permissions and runner costs are
   acceptable.

## 10. How to reproduce current work

The GitHub repository is private, so the engineer needs normal authorized
access. On their own machine:

~~~bash
git clone https://github.com/ctharvey/Norm-K.git
cd Norm-K
python -m pip install -e '.[research]'
python -m unittest discover -s tests -q

python -m norm_k.cli generate --per-label 600 --seed 42
python -m norm_k.cli audit --pairs data/generated/pairs.jsonl
python -m norm_k.cli evaluate --pairs data/generated/pairs.jsonl --encoder hash
python -m norm_k.cli k-probe --count 100 --encoder hash
python -m norm_k.cli k-ablate --pairs data/generated/pairs.jsonl --encoder hash --operation both

# Human review is separate; no synthetic source labels are displayed:
python -m norm_k.cli prepare-review --per-label 20
python -m norm_k.cli annotate
# After two real reviewers have worked:
python -m norm_k.cli export-reviews
~~~

The review server binds to `127.0.0.1:8765` **on the machine running it**;
it has not been deployed. Review queues, annotations, imported corpora,
embedding cache, and generated outputs are git-ignored. If a review queue
already exists, the CLI refuses to overwrite it: use a new output path.

**Do not put actual personal memories into a public benchmark.** Never commit
API keys or assume that an imported corpus's declared license or label
definitions have been independently verified.

## 11. Decision rules and honest interpretations

Possible outcomes:

| Outcome | Interpretation and action |
| --- | --- |
| K transfers to unseen relations and improves recall at equal review cost | Candidate mathematical contribution; replicate, audit, and shadow-test. |
| K helps only familiar subjects/templates but harms relation holdouts | Likely template/context shortcut; do not integrate. This pattern appeared in the *hashing control*. |
| Simple lexical rules equal learned geometry | Prefer simpler baseline; investigate better data before complexity. |
| Frozen geometry cannot recover the distinction | Try joint NLI/specialized encoder or explicit structured beliefs; publish the limit. |
| Belief structure solves temporal/cardinality errors while K adds nothing | Advance belief View work; stop K-specific production plans. |
| K flags uncertainty more reliably than direct contradiction | Potential triage/router value, subject to cost and false-negative checks. |

**What would count as success?** A preregistered, repeatable gain on real
human-audited **out-of-relation and out-of-source** examples at the **same
review budget**, with uncertainty bounds, acceptable false-negative behavior,
and a favorable end-to-end cost/latency tradeoff. No single accuracy number
from synthetic data is enough.

**What remains unknown?** Whether the information survives Menhir's frozen
short-name embeddings, whether any T is sufficiently well-defined for varied
prose, how much K adds beyond lexical and NLI features, and whether creating
counterfactual embeddings is actually cheaper than asking the reviewer.

## 12. Short handoff prompt to give another work thread

> Continue the ‖K‖ / Norm-K research project at
> https://github.com/ctharvey/Norm-K. Start with `docs/handoff.md`,
> `docs/research-plan.md`, `docs/pair-level-k.md`,
> `docs/annotation-protocol.md`, and `BLOCKERS.md`.
> The repository already has an offline Python harness, 50 passing tests,
> synthetic label data, K math, pair-level counterfactual ablations,
> leak-safe split checks, a local blind review UI, and benchmark import.
> **No real-embedding contradiction result has been demonstrated.**
> First prioritize independent, well-scoped benchmark labels and controlled
> null comparisons. Keep production Menhir unchanged and all integrations
> advisory/shadow-only until held-out tests and a cost comparison justify
> a separate implementation PR. Report negative results honestly and keep
> `BLOCKERS.md` up to date.

---

**Key principle:** We did not pick this lane because the equations were
guaranteed to work. We picked it because the failure of cosine alone left a
testable gap: *is relation-sensitive information hidden in frozen vector
geometry, and if so can it be extracted cheaply enough to matter?* The
repository now exists to answer that question rigorously—including the
possibility that the answer is **no**.
