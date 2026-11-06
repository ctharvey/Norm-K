# Norm-K: status and blockers

**Updated:** November 6, 2026. Keep this file current.

## Where the research stands

| Test | Result | Report |
| --- | --- | --- |
| v1, K on 144 hand-written pairs | Negation PASS, subject swap STOP | [real-fair-k-v1.md](reports/real-fair-k-v1.md) |
| v2, K on public pairs | INCONCLUSIVE; v1's PASS didn't replicate | [public-fair-k-v2.md](reports/public-fair-k-v2.md) |
| Exploratory, v2's transformed-sentence gain | Found the untrained flip score | [exploratory-flip-signal-v2.md](reports/exploratory-flip-signal-v2.md) |
| v4, flip score vs neutral edits, fresh pairs, two encoders | REPLICATES, NEGATION-SPECIFIC | [flip-placebo-v4.md](reports/flip-placebo-v4.md) |
| v5, flip score vs size-matched edits | **Not run; blocked below** | [draft protocol](docs/preregistration-flip-matched-v5.md) |

The flip score, not K, is the current focus. On unit vectors K's size is a function of
pairwise scores the fair baseline already holds, so K can add little beyond repackaging.
v4 doesn't show that negation's meaning is the cause, because negation was also the
largest edit.

## Blocked now

**1. v5 edit audit: OpenCode's free usage limit.** All three reviewer models fail with
"Free usage limit reached", and no reviewer output exists. The owner must choose how to
proceed. The protocol can't freeze without an audit result, and no test pair has gone
through a real encoder. Pairs, calibration, code and the hash control are ready.

## Blocked on outside input

**2. Human-reviewed labels and edits.** No person has reviewed any label set or edit. Label
and edit checks are LLM votes, which are a filter and not validation. Public-data labels
are human, but MultiNLI train has one annotator per pair.

**3. Realistic text.** Everything so far uses short `is` or `was` statements. Multi-fact
sentences need a dataset that doesn't exist here yet, and claim-level alignment for it
would need building and auditing.

**4. Novelty.** No claim of a new method is supported yet.
- Read: Schumann and Marx Gomez 2023. Its edits create training data for a fine-tuned BERT
  classifier, and it doesn't score pairs with an untrained edit response.
- Unread: Cui, Zheng and Wang 2020, cited there for assigning NLI labels from cosine ranges
  on a trained encoder. The other sources in the exploratory report need reading before
  they're cited, and a broader search is still needed.

**5. Name and publication.** `Norm K` is a working title. Domain and trademark availability
are unchecked, and no publication or company work has been authorized.

**6. Menhir data, if a Menhir-shaped test is ever wanted.** This is research that might
benefit Menhir, not a Menhir feature. No sanitized assertion export is available, and no
production Menhir data or code has been touched.

## Next research, in order

1. Run v5 once the audit is resolved.
2. Split the flip into its two directions, from saved vectors.
3. Compress the flip with `asinh` before averaging, as deferred in the v4 protocol.
4. Trim [the v3 draft](docs/preregistration-decomposition-v3-draft.md) and decide what's still
   worth running: dilution, context ladder, word removal, deletion-response features.
5. Later: whether incompatibility can be told from mere difference, which needs a new labeled
   set, and edits beyond `is` and `was`.

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
- Tests: 92 passing unittest cases.

See [the labeling protocol](docs/annotation-protocol.md) for label definitions.
