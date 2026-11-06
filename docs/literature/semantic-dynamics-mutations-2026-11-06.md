# Exploratory function mutations: semantic opposition, dynamics, and NLI

**Date:** 2026-11-06  
**Status:** EXPLORATORY / UNFROZEN. This document is not a preregistration and does not revise the frozen v2 or v4 experiment outcomes.  
**Execution:** `scripts/exploratory/mutation_sanity.py` performs a reproducible math-only test using synthetic vectors, **not** real model embeddings or contradiction labels.

## Update after v5

This note was written before v5's result. Three things have changed:

- **Edit size:** [v5](../../reports/flip-matched-v5.md) ruled it out as the explanation, so Stage 1's size-matched flip is done.
- **Real-encoder scores now exist.** [The mutation run](../../reports/exploratory-mutations-v5.md) scored these variants on v5's saved vectors, as exploratory work on seen pairs. None beat flip.
- **The 50.59% figure** follows from the model: with one shared negation axis, `cos(da,db)` can only be +1 or -1. On real encoders no unrelated pair reached 0.99.

The rest of the note stands as written. Its retrieval figures match the [benchmark's README](https://github.com/nstandif/contradiction-bench), which also says BM25, dense and hybrid retrievers did no better than TF-IDF at K=20. They remain unreproduced here.

## Why explore this?

The v4 negation flip was useful on a narrow NLI subset: [report](../../reports/flip-placebo-v4.md), [frozen protocol](../../docs/preregistration-flip-placebo-v4.md). Its major limitations are an edit-strength confound, strong single-statement/wrong-partner signal, MultiNLI exposure and ~3% grammar eligibility in the earlier public run. NLI alone scored **0.744 AP on WANLI**, versus **0.382** for the OpenAI flip, in [v2](../../reports/public-fair-k-v2.md). The question is whether a simpler, more selective score can flag a strong contradiction **without mistaking semantic distance, temporal change or merely differing values for opposition**.

The word *momentum* suggests two very different axes: (a) an ordered **semantic edit path**; (b) actual **changes to beliefs over time**. Neither exists in a single unordered pair of sentences. We cannot obtain a meaningful physical velocity/acceleration simply by subtracting static embeddings; time or an explicitly defined semantic parameter is essential.

## Algebraic score mutations from four cached vectors

For normalized `a=E(A), b=E(B), na=E(N(A)), nb=E(N(B))`, define `da=na-a`, `db=nb-b`, directional gains `gA=da·b`, `gB=a·db`, and edit magnitudes `mA=1-a·na`, `mB=1-b·nb`.

| Variant | Score | Question / caveat |
| --- | --- | --- |
| Original flip | `(gA + gB)/2` | Existing post-v2 formula; positive if negating tends to move toward the partner; single-statement artifacts remain. |
| **Two-direction agreement** | `min(gA,gB)` | Require both counterfactual directions to favor opposition. Might reduce one-sided annotation shortcuts; could lose recall. |
| **Magnitude-normalized flip** | `(gA/max(mA,ε)+gB/max(mB,ε))/2` | A secant-like sensitivity to negation per unit self-edit distance. Small denominators create severe noise; freeze `ε` from unlabeled pilot. Related to user's logarithmic/asinh compression idea, but controls displacement directly. |
| **Opposition of negation responses** | `-cos(da,db)` | Negative alignment might reveal opposite polarity, but is a *normalized part of the already-tested K residual*, not a new mechanism. It can be pathological. |
| **Pairwise / individual decomposition** | `F = 0.5 (da-db)·(b-a) - 0.5(mA+mB)` | **Exact identity**, no new information. Separates relational variation from per-sentence negation susceptibility to probe the wrong-partner artifact. |
| **Signed compression** | `0.5 [asinh(gA/τ)+asinh(gB/τ)]` | Can change rankings because compression precedes aggregation; not a license to tune τ on v4 data. Previously deferred by v4 protocol. |

**K relation:** `K = db-da`, hence `||K||² = ||da||² + ||db||² - 2 da·db`. Anti-alignment is essentially a normalized version of this difference; it should not be passed off as a fresh discovery. v2's K gain did not replicate.

### Mutation sanity check (already executed)

The accompanying executable uses a 32-dimensional, **synthetic global-axis reflection** as a perfect involutive "negation": `N(x)=x-2(x·u)u`, plus 10,000 unrelated unit vectors and a fixed anchor (seed 7). The algebraic K and flip-decomposition identities held to numerical tolerance. However, **50.59% of unrelated vectors received nearly maximal `-cos(da,db) = +1`**, showing that perfectly consistent negation directions can still create false, maximal opposition under this model. This is a counterexample to a universal anti-alignment decision rule; it is **not** a measured false-positive rate of any embedding encoder.

**Decision:** deprioritize anti-alignment and higher-order K variants. Test the two-direction minimum and normalized directional gain only on a **new sealed evaluation set**, with existing F and actual NLI as fixed comparators. The v4 committed JSON contains aggregated AP metrics rather than per-pair vectors; it does not permit honest real-encoder AP estimates for these mutations. Reusing cached vectors locally is mathematically sufficient but still exploratory on previously viewed pairs.

## Mutation ideas beyond embedding geometry

### 1. Pair retrieval may dominate classifier quality — highest operational priority

A recent, *independent third-party benchmark* of 19 NLI checkpoints on the [ContraDoc](https://aclanthology.org/2024.naacl-long.362/) human-annotated document benchmark reports that its top-20 TF-IDF candidate pairs contain the gold contradiction in only **59.5%** of documents, and that **12%** of gold pairs share no content words. Its reported best NLI pair AUROC is **0.870** (given a gold pair), whereas its best whole-document NLI pipeline is **0.666**. Source: [contradiction-bench](https://github.com/nstandif/contradiction-bench). These **repository-reported**, not independently reproduced results have different data/metrics from Norm-K, but identify a serious failure mode: **a better NLI classifier cannot score pairs retrieval never proposes**.

For Menhir, measure *candidate recall* at the same review budget before adding a new scorer. Try alternative candidate generation: subject/attribute slot joins, corrections on the same topic, explicit source referents, lexical/FTS union with vector candidate retrieval, plus baseline #392 vetoes. No model should declare a stored fact false simply because it disagrees with another value.

### 2. Structured exclusivity gates — simplest detector for a narrow, valuable subclass

If two assertions address the **same resolved subject**, a **single-valued attribute**, **overlapping valid time**, **same scope**, and **different mutually exclusive values**, generate a high-priority review candidate **without** an NLI call. Examples: simultaneously claimed distinct *current primary residences*. Counterexamples: two houses owned concurrently; *birth city* versus *residence*; residence after a move. This requires reliable normalization/cardinality metadata and provenance, not just comparing raw strings. Never overwrite beliefs based on this gate; it is advisory.

### 3. Bidirectional NLI and strong opposition confidence

A standard NLI model produces logits for contradiction (`c`), entailment (`e`), and neutral (`n`). It can be evaluated in both orders: `(A,B)` and `(B,A)`. Alternatives worth comparing **without training a new model**:

- `pC(A,B)` (must remain the first baseline), `max`, `mean` and `min` of `pC(A,B),pC(B,A)`.
- `c-e` to emphasize opposition versus equivalent/supported claims, *with neutral false-positive risk tracked*. `c - logsumexp(e,n)` is mathematically monotone in `pC` and **cannot change ranking on a single ordered pair**; do not report it as an improvement.
- For *audited, scope-correct negation*, check whether high `pC(A,B)` agrees with high `pE(A,not B)`. This is a metamorphic consistency diagnostic, not necessarily a stronger operational classifier and needs additional inference passes.

Two-direction agreement might help designate a **high-confidence review queue**, but should never silently reject a contradictory pair that only one orientation recognizes. Benchmark all these separately against the existing judge and no-signal fail-open route.

### 4. Counterfactual gradient / 'acceleration' across a semantic trajectory

Choose *meaningfully ordered*, audited statement edits `S0 → S1 → S2`, for example the same speaker/target shifting from strong support to neutral to strong opposition. Let `q(Si)` be a **separately calibrated, target-conditioned stance score**, not an arbitrary embedding coordinate. First/second finite differences are `v1=q(S1)-q(S0)`, `v2=q(S2)-q(S1)`, `a=v2-v1`. On nonuniform steps these have no meaningful magnitude until the step spacing is defined. Embedding displacement `E(S2)-2E(S1)+E(S0)` can measure an encoder's curvature under the chosen edits, but is sensitive to wording, scale and path: it does **not** directly measure logical contradiction.

**Can an NLI model be taught this?** Yes as a separate **multitask/ordinal stance objective**, with examples explicitly labeled on a scale like `strongly oppose / oppose / neutral / support / strongly support`, *same topic, subject and valid time*, using a ranking loss plus optional smoothness/monotonicity constraints. Train and test with unseen topics, paraphrases, modalities and negation scopes. This is not ordinary three-way NLI anymore, and strong opposition of **views** is not identical to truth-conditional **contradiction**. **Do not build this until error analysis shows NLI misses graded stances the application needs.** A zero-shot NLI + polarity/stance classifier baseline should come first.

### 5. Actual temporal velocity belongs in belief history, not in a pairwise NLI score

With observations `(slot, valid_time, value, source)`, `q(t)` can describe a particular person's stance or a numerical fact over time. Time-based `dq/dt` and `d²q/dt²` can measure **changes in reported stance**, but require at least 2/3 observations, a valid ordinal/numeric quantity, reliable dates and appropriate handling of irregular spacing. Abrupt reversal can be a **review priority**; sequential updates are not simultaneous contradictions. If the attribute is categorical, treat it as a transition graph rather than pretending Austin→Denver is a physical vector velocity.

### 6. Atom-level NLI / causal mutations

Split a compound statement into audited atomic assertions, align by subject+attribute+scope, then NLI the matched subclaims rather than all words. The published [Atomic Inference for NLI with Generated Facts as Atoms](https://aclanthology.org/2024.emnlp-main.569/) trains a related atomic approach, so decomposition is **not** new. Only consider this after documenting full-sentence failures on multi-clause Menhir-like cases.

## Research gates and minimal next run

**Stage 0 (no fine-tuning):** benchmark real Menhir-shaped paired claims, full-statement NLI (`pC`, two-order mean/min/max and `c-e`), cheap same-slot/time/cardinality rules, and end-to-end retrieval candidate recall. Keep the 42-pair fixture regression-only and construct independent holdout. Measure AP, contradiction recall at fixed judge budget, false negatives, review cost, confidence intervals, latency with GPU contention, coverage, and whether existing #392 candidates are lost.

**Stage 1 (only if Stage 0 misses a real subclass):** test edit-magnitude-matched flips, `min(gA,gB)`, displacement-normalized F, and paraphrase/quantifier stress cases. **Freeze** formula selection, edit-validity audit, model revision and holdouts before embedding new data; keep exploratory scores clearly labeled.

**Stage 2 (only if there is a specific use for graded opinions or longitudinal stance):** build stance-intensity ordinal annotations and estimate trend over real *timestamped* claims; compare against a simple stance extractor and belief View. A smoothness or acceleration-inspired auxiliary loss is an option, not a presumed improvement.

**Stop condition:** if NLI + slot/time rules + stronger candidate generation meets the operational target, stop expanding K variants. If the numeric changes fail to improve OOD review-budget recall or introduce artifacts, pin negative results and do not ship them to Menhir.

## Additional primary sources and prior art

| Source | What it establishes or challenges |
| --- | --- |
| [Sentence Analogies: Linguistic Regularities in Sentence Embeddings (COLING 2020)](https://aclanthology.org/2020.coling-main.300/) | Relational movement in sentence embeddings is prior art, inconsistent across encoders. |
| [NatLog: MacCartney, Natural Language Inference dissertation](https://nlp.stanford.edu/~manning/dissertations/MacCartney-Bill-nli-diss.pdf) | Local edit entailment relations and monotonicity are established symbolic methods. |
| [Supporting Context Monotonicity Abstractions in Neural NLI Models (2021)](https://aclanthology.org/2021.naloma-1.6/) | NLI can train with supplemental linguistic logic constraints. |
| [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList (2020)](https://aclanthology.org/2020.acl-main.442/) | Directional/invariance metamorphic tests for controlled semantic mutations. |
| [Atomic Inference for NLI with Generated Facts as Atoms (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.569/) | Atomic premise fact decomposition and NLI are not novel. |
| [ContraDoc: Understanding Self-Contradictions in Documents (NAACL 2024)](https://aclanthology.org/2024.naacl-long.362/) | Human-annotated long-document contradiction benchmark. |
| [Third-party ContraDoc NLI benchmark](https://github.com/nstandif/contradiction-bench) | Unverified external repository-reported candidate-retrieval and model bottleneck observations. |
| [Probing Language Models for Understanding of Temporal Expressions (2021)](https://aclanthology.org/2021.blackboxnlp-1.31/) | NLI models have limited temporal-order/duration robustness. |
| [FACTTRACK: Time-Aware World State Tracking (2025)](https://aclanthology.org/2025.naacl-long.144/) | Temporal context can disambiguate seeming contradictions; do not equate textual opposition with simultaneous conflict. |
| [Stance Detection: A Practical Guide to Classifying Political Beliefs in Text (2024)](https://www.cambridge.org/core/journals/political-science-research-and-methods/article/stance-detection-a-practical-guide-to-classifying-political-beliefs-in-text/E227E746BD7D9751526DA0EC2C378787) | Stance intensity is a distinct open extension beyond categorical NLI. |
| [Counterfactuals for enhanced NLI, MS thesis (2025)](https://corescholar.libraries.wright.edu/etd_all/2985/) | Counterfactual-augmented NLI training is established; do not claim to invent training with semantic edit trajectories. |