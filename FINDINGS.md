# Norm-K findings: can embedding geometry find contradictions?

**Short answer: not demonstrated as a general-purpose contradiction detector.** With frozen text embeddings, negating a statement produced a repeatable ranking signal in restricted tests, but most of the measured retrieval gain concentrated on pairs with an explicit negation mismatch. Generalization to other sentence structures and real conflicts remains untested.

This is a small independent project by ctharvey, carried out with AI assistance. **The research phase closed on November 6, 2026**, with experiments paused and the source/results preserved. It isn't peer reviewed. It's shared so others can skip the same work or check it. No novelty is claimed.

## What was tested

The first idea was a four-point residual called K. It passed on 144 hand-written pairs, then
[failed to replicate on public data](reports/public-fair-k-v2.md). On unit vectors it also adds little
beyond pairwise similarities.

That work turned up a simpler score. Embed `A` and `B`, add `not` to each, and compare:

```text
negated_cosine = 0.5 * [cos(N(A), B) + cos(A, N(B))]
flip           = negated_cosine - cos(A, B)
```

Later rounds fixed their pass rules in a protocol before scoring. They used `text-embedding-3-small`
and `all-mpnet-base-v2`, on MultiNLI and WANLI pairs where `is` or `was` could be edited.

## What held up

- **Flip beats cosine at separating contradictions.** On 1,500 fresh pairs it gained +0.296 AP on
  OpenAI and +0.224 on mpnet, and it beat neutral edits ([v4](reports/flip-placebo-v4.md)).
- **Edit size doesn't explain it.** Negation beat size-matched edits by +0.327 AP on OpenAI and +0.166
  on mpnet ([v5](reports/flip-matched-v5.md)).
- **`negated_cosine` works as a ranker before NLI.** Among each statement's top 5 neighbors, sending
  the top 10% to NLI confirmed 0.647 of contradictions against cosine's 0.515 on OpenAI, and 0.595
  against 0.437 on mpnet ([v7](reports/negated-gate-v7.md)).

## What didn't

- **It's mostly about negation words.** 78% to 87% of v7's margin came from pairs where exactly one
  statement had a negation word. The 62% of contradictions with none gained only +0.023 and +0.041
  ([split](reports/exploratory-cheap-polarity.md)).
- **Flip alone is a poor filter.** It found about half of what cosine found, because it ignores
  relatedness. Part of its signal also survives shuffling partners, so it isn't all about the pair.
- **Shortcuts didn't replace real edits.** An average negation vector, a one-number polarity score and
  [other score variants](reports/exploratory-mutations-v5.md) all did worse.
- **Hashed words did about as well as cosine.** They found 0.550 of contradictions on v7's data, so
  that data is easy in ways that flatter simple methods.
- **NLI raises many false alarms.** It flagged 20.7% of labeled neutral pairs, and thousands of
  unlabeled candidates. Twelve read by eye were mostly unrelated pairs. Nobody has reviewed them
  properly.
- **One planned test never ran.** The [NaN-NLI protocol](docs/preregistration-nan-nli-v6-draft.md)
  stopped at its size check, so nothing here shows the result holds outside `is` and `was` statements.

## Two things worth copying

**Split results by negation word.** A contradiction score can pass a fair test and still be mostly a
negation detector. Report the gain separately for pairs where neither, one or both statements have a
negation word. It takes minutes, and it changed how this project read its own best result.

**Check what precision your local model loaded at.**
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` stores 16-bit weights, and transformers 5.22.1 loaded
them as 16-bit with no warning. On CPU that ran about 7 pairs a second, against 66 at 32-bit on a
1,000-pair sample. It also moved scores by up to 0.011 across 36,636 pairs. No v7 result changed
here, but small score gaps between CPU and GPU came from this and not from the hardware. At 32-bit they
agreed to about 0.00001. Calling `model.float()` fixes it.

## How this fits prior work

Embeddings' trouble with negation is well known. See
[HEROS](https://arxiv.org/abs/2306.05083) on sentence encoders and
[this work on negation awareness](https://arxiv.org/abs/2504.00584). A 2026 preprint,
[Negation is Not Semantic](https://arxiv.org/abs/2603.17580), reports dense retrieval failing at
contradiction detection, and uses lexical retrieval, negation filtering and NLI instead. Our result
fits that pattern. The [prior-art review](docs/literature/prior-art-flip-2026-11-06.md) found no exact
earlier publication of the flip formula, but that isn't proof of novelty, and some sources were read
only as abstracts.

## Limits

- Two encoders, and short sentences from two related datasets. WANLI was built from MultiNLI seeds, and
  mpnet's training data included MultiNLI.
- Only `is` and `was` statements could be edited. Edits were checked by LLM reviewers, not people.
- The negation split and the variants are exploratory, run on data already seen.
- A negation word means a word on a fixed list, so `unable` or `refused` count as none.
- Nothing was tested on realistic documents or long text.

## Closure and downstream disposition

**Decision: stop further Norm-K experimentation for now; do not integrate the flip or negated-cosine gate into Menhir.** The frozen v7 gate passed on selected WANLI text, but the post-hoc polarity split, restricted grammar, missing human audit, and absent realistic Menhir benchmark do not justify deployment. There are no additional Norm-K experiments scheduled.

The related [Menhir belief-shaped contradiction issue #511](https://github.com/Archolith/menhir/issues/511) remains a separate, open engineering question. Its typed assertions, valid-time reasoning, and potential small local NLI baseline do not depend on this geometry research. Reopen Norm-K only if representative human-reviewed conflicts, meaningful review volume, or a new intervention that helps non-negation contradictions changes the cost/benefit case. See [BLOCKERS.md](BLOCKERS.md).

## Reproduce it

Everything runs on a CPU. See [BLOCKERS.md](BLOCKERS.md) for all results and untested ideas, and the
`docs/preregistration-*` files for each protocol and its commands. Public datasets aren't included,
and pairs are stored as IDs. OpenAI runs need your own key.
