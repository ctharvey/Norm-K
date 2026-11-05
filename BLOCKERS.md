# ‖K‖ — blockers and deferred work

**Updated:** November 5, 2026. Keep this file current as work is unblocked.

The repository's offline scaffold works. These items are deliberately kept
separate from the build plan so no unperformed research looks completed.

## Blocked by missing access, evidence, or external decisions

### 1. Real frozen text embeddings — **BLOCKED: API credentials not present**

- The executor did not have `OPENAI_API_KEY` configured.
- No paid embedding API calls were made. The opt-in API adapter and SQLite cache
  are implemented, with `--max-new=0` as the safe default.
- To unblock: configure the API key in the environment (never commit it), confirm
  permitted API usage/budget and network access, then run capped calls with
  `--encoder openai --max-new <explicit_count>`.
- Re-run both entity/name-only and complete-statement embeddings. Scores from the
  offline hash-control encoder **do not transfer** to text-embedding-3-small.

### 2. Audited labels and realistic holdouts — **BLOCKED: validated data needed**

- The 3,000-pair dataset is synthetically templated and explicitly unaudited.
  No accuracy, generalization, or novelty claims are supportable from it.
- To unblock: independent human audit, separate naturally worded examples,
  contested-label adjudication, and sealed held-out sets by relation, sentence
  template, subject, and lexical vocabulary.
- The small Menhir #392 fixture and SemAntoNeg should be used as evaluation
  references, never as the sole benchmark or as training data.
- Do not publish genuine user memories or personally identifying test content.
- **Tooling is now available:** an offline blind-review UI, append-only reviewer
  log, two-reviewer consensus, pairwise kappa, and a local CSV/TSV/JSONL importer.
  Actual independent human labeling and source/license checks have not happened.

### 3. Menhir belief graph and real conflict distribution — **BLOCKED: safe test data required**

- No Menhir private database or consented non-production assertion snapshot was
  available in this workspace.
- To unblock: supply a sanitized test graph or an export of synthetic `:TypedAssertion`
  records plus schema/valid-time semantics. Then measure same-time ties, A→B→A
  flips, attribute aliases, and ordinary updates.
- No production Menhir changes were made, and no assertions were rewritten.

### 4. Operational baseline and review budgets — **BLOCKED: measurements needed**

- The actual #392 filter and NLI/LLM judge's end-to-end cost, latency, and
  candidate throughput have not been measured here.
- To unblock: run a representative replay from the Menhir conflict pipeline,
  with permitted data and model access. Record review budget and missed conflicts.

### 5. Trademark, naming, and publication — **UNVERIFIED**

- `‖K‖` / `Norm K` is a working project title. Domain/trademark availability has
  not been checked and must not be inferred from creating a GitHub repository.
- No publication or company registration work has been authorized.

## Feasible next, but not completed in this pass

- [ ] Pre-register a full held-out evaluation protocol and independent label audit.
- [ ] Add tokenizer/subword and name-only runs with real embeddings.
- [ ] Add a trained local NLI cross-encoder and measure GPU contention with LLM server.
- [ ] Add randomized-interaction controls; K has descriptive bootstrap intervals
      and matched pair-level ablations, but no validated causal counterfactual check.
- [ ] Add PCA arrow plots and cross-encoder comparisons (dimension-vs-recall
      graph is implemented for the offline control).
- [ ] Implement **vocabulary-held-out**, paraphrase-held-out, and external-domain
      splits (subject/template/relation smoke splits plus exact-text purging work).
- [ ] Validate the restricted counterfactual grammar with independent human
      judgments and test more transformation families. Current coverage is
      incomplete, especially for purchase/negation and different-subject pairs.
- [ ] Upgrade paired K ablations to preregistered tests using real embeddings,
      realistic grouped holdouts, and multiple-comparison controls.
- [ ] Run the local annotation UI on the reviewer's own computer; the interface
      has been API-tested but not visually checked in a real browser here.
- [ ] Obtain an appropriately licensed external benchmark file and verify its
      original label semantics before any canonical mapping. The importer is
      implemented, but no external dataset has been ingested in this workspace.
- [ ] Commission independent reviewers and adjudicate ambiguous or disputed
      cases; no human-reviewed benchmark labels have been collected.
- [ ] Build read-only typed-belief comparisons and pure ViewKind prototype after
      safe Menhir data access is arranged.
- [ ] Add hosted CI once workflow permissions and intended runner use are confirmed;
      unit tests currently run locally with no external credentials.

## What *is* complete

- Python project scaffold, pinned package identity, installation metadata.
- Frozen-embedding provider interface, deterministic lexical smoke encoder,
  opt-in capped OpenAI adapter, persistent SQLite cache.
- Deterministic five-class synthetic pair generation (3,000 examples) and four
  transformation-square families (100 probe cases).
- Cosine/difference/distance/Hoyer and K/relative-K numerical primitives.
- Baseline report, linear-probe smoke comparison, review-budget scoring.
- SVG pair scatter and K distribution, generated locally.
- Subject-, template-, and relation-held-out linear baselines, PCA(|Δ|) low-rank logistic
  probes, dimension-vs-recall SVG, and bootstrap mean intervals for K groups.
- Pair-level counterfactual K feature extraction (negation/subject swap),
  matched-cohort with-vs-without-K ablations, subject-cluster bootstrap AP
  uncertainty, and a pair-level scatter chart.
- Dataset fingerprint audit, symmetric duplicate detection, subject-safe
  generation, mechanical split leakage checks, and exact-statement purging.
- Blind stratified review queues, 127.0.0.1-only review UI, append-only
  corrections, two-reviewer consensus, Cohen's kappa, and CSV/TSV/JSONL import
  with explicit mapping and provenance manifest.
- 50 passing unittest cases, including localhost API security and annotation
  workflow tests.

See [the offline run record](reports/offline-smoke.md) for the exact commands and
their limits.

See [the pair-level K ablation](reports/offline-ablation.md) for a concrete
example of why a favorable simple split is not sufficient evidence of transfer.

See [the labeling protocol](docs/annotation-protocol.md) for instructions,
label definitions, and scientific safeguards.
