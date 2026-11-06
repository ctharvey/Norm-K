# Norm-K: status and blockers

**Updated:** November 6, 2026. Keep this file current.

## Where the research stands

| Test | Result | Report |
| --- | --- | --- |
| v1, K on 144 hand-written pairs | Negation PASS, subject swap STOP | [real-fair-k-v1.md](reports/real-fair-k-v1.md) |
| v2, K on public pairs | INCONCLUSIVE; v1's PASS didn't replicate | [public-fair-k-v2.md](reports/public-fair-k-v2.md) |
| Exploratory, v2's transformed-sentence gain | Found the untrained flip score | [exploratory-flip-signal-v2.md](reports/exploratory-flip-signal-v2.md) |
| v4, flip score vs neutral edits, fresh pairs, two encoders | REPLICATES, NEGATION-SPECIFIC | [flip-placebo-v4.md](reports/flip-placebo-v4.md) |
| v5, flip score vs size-matched edits, two encoders | SURVIVES-MATCHING; cheap gate fails | [flip-matched-v5.md](reports/flip-matched-v5.md) |

The flip score, not K, is the current focus. On unit vectors K's size is a function of
pairwise scores the fair baseline already holds, so K can add little beyond repackaging.
v5 rules out edit size as the explanation, but not negation's direction or cues in single
statements, so negation's meaning still isn't shown to be the cause. Part of the gain doesn't
need the real partner. Raw flip can't be a cheap gate, since it finds far fewer contradictions
than cosine in a mostly unrelated pool.

## Blocked on outside input

**1. Human-reviewed labels and edits.** No person has reviewed any label set or edit. Label
and edit checks are LLM votes, which are a filter and not validation. v5's audit was a weak
one: two of three reviewers approved everything, and no known-bad item tested them. Public-data labels
are human, but MultiNLI train has one annotator per pair.

**2. Realistic text.** Everything so far uses short `is` or `was` statements. Multi-fact
sentences need a dataset that doesn't exist here yet, and claim-level alignment for it
would need building and auditing.

**3. Novelty.** No claim of a new method is supported.
[The prior-art audit](docs/literature/prior-art-flip-2026-11-06.md) found the general idea heavily
preempted, and found no exact prior publication of this formula, which isn't proof of novelty.
- Read here: Schumann and Marx Gomez 2023. Its edits create training data for a fine-tuned BERT
  classifier, and it doesn't score pairs with an untrained edit response.
- Inspected by that audit, not read here: Cui, Zheng and Wang 2020, which uses plain two-vector
  cosine thresholds after encoder training, with no edits at inference. Many other entries are
  abstract-only. Zhu 2018 and Faltings 2025 should be read in full before any novelty wording.

**4. Name and publication.** `Norm K` is a working title. Domain and trademark availability
are unchecked, and no publication or company work has been authorized.

**5. Menhir data, if a Menhir-shaped test is ever wanted.** This is research that might
benefit Menhir, not a Menhir feature. No sanitized assertion export is available, and no
production Menhir data or code has been touched.

## Next research, in order

1. One out-of-domain test on NaN-NLI, as v5's frozen plan sets out. Needs its own protocol.
2. Owner's call: a two-stage gate, with cosine finding related pairs and flip ranking them for
   NLI. v5 only hints at this, after seeing results, so it needs a new rule and fresh data.
3. Split the flip into its two directions, from saved vectors.
4. Trim [the v3 draft](docs/preregistration-decomposition-v3-draft.md) and decide what's still
   worth running: dilution, context ladder, word removal, deletion-response features.
5. Later: whether incompatibility can be told from mere difference, which needs a new labeled
   set, and edits beyond `is` and `was`.

**Parked** by the prior-art audit: `asinh` compression, per-token and chunk-level flip, and learned
projections.

## Smaller open items

- [ ] Tokenizer/subword and name-only embedding runs.
- [ ] Vocabulary-held-out and paraphrase-held-out splits.
- [ ] PCA arrow plots.
- [ ] Check the local annotation UI in a real browser.
- [ ] Hosted CI, once workflow permissions are confirmed. Tests run locally.

## What exists

- Encoders: offline hash control, capped opt-in OpenAI adapter with SQLite cache and measured
  token counts, and a pinned local `all-mpnet-base-v2` script.
- Data: 3,000-pair synthetic set (unaudited), 144-pair matched set, pinned public selections by
  ID, and audit votes.
- Tools: geometry and K functions, fair K tests, public and flip tests, leakage checks, bootstrap
  and shuffle nulls, blind review UI, consensus and kappa, benchmark importer.
- Method: frozen protocols, hash controls, and [multi-model review](docs/multi-model-review.md).
- Tests: 99 passing unittest cases.

See [the labeling protocol](docs/annotation-protocol.md) for label definitions.
