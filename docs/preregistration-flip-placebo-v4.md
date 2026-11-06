# Flip score placebo test v4: frozen protocol

**Frozen:** November 6, 2026, before any real-embedding score on the fresh set. The commit
that renames this file from its draft is the freeze point. Changes after a real result
must be reported as deviations.

## Question

The [exploratory v2 analysis](../reports/exploratory-flip-signal-v2.md) found that an
untrained flip score roughly matched trained models. Two things are unknown:

1. Does it hold on pairs it wasn't defined on?
2. Does it come from negation, or from editing a sentence in any way?

## Score

For an edit E:

    F_E(A, B) = [ (cos(E(A), B) - cos(A, B)) + (cos(A, E(B)) - cos(A, B)) ] / 2

Same formula, same pairs, same evaluation for every edit. Nothing is trained.

| Edit | Example | Role |
| --- | --- | --- |
| `negation` | `is` to `is not`, `is not` to `is` | The frozen v2 rule |
| `in_fact` | `is` to `is in fact`, on the verb negation would edit | Neutral edit, same spot |
| `note` | `Note: ` added at the start | Neutral edit, built separately, different spot |
| `possibly` | `is` to `is possibly`, same spot | Changes meaning without negating; reported only |

## Data

**Primary, fresh:** 1,500 pairs from MultiNLI's train split, in the same pinned zip as v2,
listed in [flip-v4-pair-ids.tsv](../data/benchmarks/flip-v4-pair-ids.tsv). They pass v2's
eligibility rules, share no statement with v2's sets, and use each premise and each
statement once, because train reuses premises. IDs that occur more than once in train
are excluded, because pairs are stored by ID. Pairs are ordered by a seeded hash of their
ID. Selection uses no labels. Train has one label per pair, so labels are noisier than
dev's, and it has five genres.

**Secondary:** v2's 582 MultiNLI and 725 WANLI pairs, where the flip score was found.

## Encoders

`text-embedding-3-small` carries the verdicts. `all-mpnet-base-v2`, revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`, mean pooled and unit length, is run the same
way. Its card lists MultiNLI in its training data, so its fresh-set result may be
contaminated and WANLI is the cleaner read for it. The offline hash encoder is the control.

## Primary measures

Fresh set, `text-embedding-3-small`, average precision (AP), contradiction vs not.

- **Replication:** F_negation minus the better of cosine and reversed cosine. Verdicts:
  REPLICATES, FAILS, UNCLEAR.
- **Negation vs placebo:** F_negation minus the better of F_in_fact and F_note. Verdicts:
  NEGATION-SPECIFIC, EDIT-GENERAL, UNCLEAR.

The positive verdict needs all of: delta at least +0.03, 95% interval's lower bound above
0, delta above the 97.5th percentile of a null, and delta above the hash encoder's delta.
The negative verdict needs the interval's upper bound below +0.03. Anything else is UNCLEAR.

Intervals use a 2,000-resample pair-level bootstrap, seed 42, since five genres are too few
clusters. The null swaps each pair's two percentile-ranked scores at random, 200 times.
Picking the better comparator after seeing it favors the comparator, which makes a
positive verdict harder.

A result is called general only if `all-mpnet-base-v2` gets the same verdict by the same
rule. Everything else is descriptive and can't change a verdict.

## Reported whatever happens

- Every edit's AP on all three sets, for both encoders and the hash control.
- How far each edit moves a statement, as mean cosine distance. Negation may simply move
  it further.
- Pair shuffling: AP when each B gets another pair's A in its genre, five shuffles.
- Pairs with no negation word on either side, and each genre.
- `possibly`, and negation's flip minus the best neutral edit's flip.

## Edit audit

**Rule, committed before any reviewer sees a statement.** Using the
[multi-model review](multi-model-review.md) method, the same three free models each judge
100 fresh-set statements with their `in_fact` edit, sampled with seed 2029, blind, twice,
in different orders, in empty directories. A model votes invalid if it says invalid in
either run. A statement is invalid if at least two of three models vote invalid. The bar
is 90 valid of 100. The any-dissent count is also reported. Under 90, stop and take it to
the owner. `note` isn't audited, since it adds a label without touching the sentence.
Fresh-set negation edits use v2's rule and aren't re-audited.

## Completed before freezing

**Edit audit: passed, 99 of 100.** The rule was committed at de7e9d1, before any reviewer
saw a statement. All 18 runs returned complete output with no tool use. One statement failed
on all three models, and one more drew a single dissent, so counting any dissent as invalid
gives 98. No person read these statements. Results are in
[flip-v4-edit-audit.tsv](../data/benchmarks/flip-v4-edit-audit.tsv).

**Selection fix:** train repeats 6,988 pair IDs, and 74 selected IDs could have resolved to
different pairs. Those IDs were excluded at 91b020f, before any reviewer or score saw the
selection.

**Hash dry run.** Condition 4's values, frozen in
[flip-v4-hash-control.json](../data/benchmarks/flip-v4-hash-control.json):

| Set | Replication | Negation vs placebo |
| --- | ---: | ---: |
| Fresh | +0.0640 | +0.0415 |
| v2 MultiNLI | +0.0710 | +0.0576 |
| v2 WANLI | -0.0321 | +0.0702 |

**The hash control itself clears the other conditions** on the fresh set, unlike in v2. A
word-only encoder can reach F_negation AP 0.400 against neutral edits' 0.358, probably
through the word `not`. So condition 4 is what makes this rule meaningful: a real result
must beat these deltas. No rule was changed after seeing this.

**Seen before freezing:** the fresh set's size, contradiction count (491 of 1,500), and all
hash scores. No real-encoder score on fresh or secondary sets with these neutral edits.

**Key:** the owner said embedding needs no further approval. Menhir's OpenAI key is loaded
into the run's process only. About 340,000 words, roughly $0.01.

## Run

Once per encoder, each refusing to replace its result:

~~~bash
python scripts/v4_flip_placebo.py run multinli.zip wanli_test.jsonl data/benchmarks/flip-v4-pair-ids.tsv openai outputs/flip-v4/openai.json data/benchmarks/flip-v4-hash-control.json --max-new 27000
python scripts/v4_flip_placebo.py run multinli.zip wanli_test.jsonl data/benchmarks/flip-v4-pair-ids.tsv mpnet.npz outputs/flip-v4/mpnet.json data/benchmarks/flip-v4-hash-control.json
~~~

## Known limits

- Short `is` or `was` statements only, and the fresh set is still MultiNLI.
- The author has seen v2 results, but no fresh-set label or score before the hash dry run.
- Two neutral edits can't cover every kind of edit.
- An EDIT-GENERAL result wouldn't say why editing helps.

## Deferred

**Compression of the flip,** as the next mathematical test, so this test changes one thing:
`[asinh(delta_A / tau) + asinh(delta_B / tau)] / 2`. Transforming each direction before
averaging can change rankings, unlike a log of the final score. Tau must be fixed without
using test results.
