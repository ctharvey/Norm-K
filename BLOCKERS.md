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
| Exploratory, score mutations on v5's saved vectors | No mutation beats flip | [exploratory-mutations-v5.md](reports/exploratory-mutations-v5.md) |
| v6, flip on NaN-NLI | Stopped at bridge check; not run | [draft](docs/preregistration-nan-nli-v6-draft.md) |
| v7, `negated_cosine` gate before NLI, two encoders | GATE-PASS on both | [negated-gate-v7.md](reports/negated-gate-v7.md) |

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

1. One out-of-domain test on NaN-NLI, as v5's frozen plan sets out.
   [The v6 draft](docs/preregistration-nan-nli-v6-draft.md) stopped at its bridge check, before any
   NaN-NLI statement was embedded. v5's edit fits only 32 of 258 pairs, and the wrapper edit moved
   statements 2 to 3.5 times further than its control, failing the size gate. Owner's call.
2. [v7](reports/negated-gate-v7.md) passed: `negated_cosine` found more contradictions than cosine at
   the same NLI budget, but [most of that gain](reports/exploratory-cheap-polarity.md) is on pairs
   where exactly one statement has a negation word. Its frozen next step is realistic text, which needs a dataset that doesn't exist here.
3. Trim [the v3 draft](docs/preregistration-decomposition-v3-draft.md) and decide what's still
   worth running: dilution, context ladder, word removal, deletion-response features.
4. Later: whether incompatibility can be told from mere difference, which needs a new labeled
   set, and edits beyond `is` and `was`.
5. Owner's call: candidate retrieval as its own question. The
   [third-party ContraDoc benchmark](https://github.com/nstandif/contradiction-bench) reports its
   top 20 TF-IDF pairs miss the contradiction in 40.5% of documents, and that BM25, dense and hybrid
   retrievers did no better at that size. Not reproduced here.

## Still to test on this line

Unticked items haven't been run. All would be exploratory unless a frozen rule says otherwise.

**Could change how current results read**
- [x] Negation-mismatch baseline: `negated_cosine` beats the flag, but 78% to 87% of its v7 margin is on
  pairs where exactly one statement has a negation word. See
  [the report](reports/exploratory-cheap-polarity.md).
- [ ] A real lexical baseline, such as BM25 or word overlap. Hash alone reached 55% of contradictions
  on v7's data.
- [ ] A hypothesis-only classifier, to size the single-statement cues that wrong-partner hints at.

**Cheap, from saved vectors**
- [ ] Split the flip into its two directions.
- [ ] Remove the shared negation direction from `da` and `db`, then rescore `-cos(da, db)`. Negation
  moves mostly align, so the opposition idea hasn't had a fair test yet.
- [ ] Rescore `-cos(da, db)` within cosine bands, to separate it from plain similarity.

**Cheap ways to find contradictions, by comparing positive and negative versions**
- [x] Average negation vector: stable across sets, but below flip as a score. As a gate it recovers
  only a fifth to two fifths of the real edit's gain, and is below the plain flag.
- [x] Search with the negated version: a lead on mpnet's fresh set only, where the real negated query
  reached more with fewer candidates. No gain on OpenAI, and WANLI's retrieval is already near full.
- [x] Polarity score: little beyond spotting negation words, and below cosine as a gate.
- [ ] Small combined score: logistic model on cosine, both negated cosines and the mismatch flag, on
  held-out splits. Shows how much signal exists, and whether flip adds to the simple flag.
- [ ] Tiny NLI model as the gate, ahead of the full model. It's the competitor to beat.
- [ ] Antonym edits in place of `not`, for contradictions with no negation word. This is now the main
  open question: 62% of v7's contradictions have none, and `negated_cosine` barely helps them.

**New design needed**
- [ ] v6 retry: a pool of affirming wrappers matched for distance, as in v5. The wrapper edit kept
  strong signal without grammar rules, so this could remove the `is`/`was` limit.
- [ ] Harder audit items: subtle scope errors, not just unchanged text and broken grammar.
- [ ] Gate value: what a gate saves when retrieval is already about five pairs per statement.
- [ ] More encoders, including larger ones. "General" rests on two so far. OpenRouter lists about 40
  embedding models, and direct use of its key needs the owner's approval.

**Tooling**
- [x] GPU parity: every local-model rerun kept its verdicts. Force 32-bit on local NLI, since the model
  loads as 16-bit by default. CPU and GPU then match to about 0.00001, so a GPU is optional. CPU runs
  66 pairs a second. See [the v7 report](reports/negated-gate-v7.md).
- [ ] Which precision v2's original NLI scores used.

**Parked** by the prior-art audit: `asinh` compression, per-token and chunk-level flip, and learned
projections. **Parked** by [the mutation run](reports/exploratory-mutations-v5.md): normalized flip
and `-cos(da, db)` lost to flip on seen data, and `min(gA, gB)` stays a report-only row. **Parked**
until a real use needs them: acceleration scores and stance-trained NLI. One pair has no time axis,
and this repo has no timestamped claims.

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
- Tests: 123 passing unittest cases.

See [the labeling protocol](docs/annotation-protocol.md) for label definitions.
