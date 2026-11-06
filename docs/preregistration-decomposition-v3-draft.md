# Decomposition test v3: draft protocol

**Status:** draft, not frozen. Nothing here has been run. The commit that renames this
file from its draft is the freeze point, as in v2.

## Question

Where does contradiction information sit in a sentence embedding, and how does it
change as a sentence is padded, stripped down, or has words removed?

This is a measurement study. It asks what the embedding carries, not whether any
method is useful for a product. A clean negative result counts as a finding.

| Experiment | What it asks |
| --- | --- |
| 1. Dilution | Does the signal fade as unrelated shared sentences are added? |
| 2. Context ladder | Does the signal get clearer or weaker as context is removed? |
| 3. Word removal | Which words does the signal depend on, and does a trained NLI model depend on the same ones? |
| 4. Compression | Does shrinking large differences reveal a spread-out signal that linear features miss? |

## Data

The v2 pairs, unchanged: 582 MultiNLI pairs (primary, 10 genres) and 725 WANLI pairs
(replication), listed in [public-v2-pair-ids.tsv](../data/benchmarks/public-v2-pair-ids.tsv)
and pinned by the SHA-256 hashes in [the v2 protocol](preregistration-public-v2.md).
Labels stay contradiction vs not.

Reusing them keeps results comparable to v2. It also means these are short, simple
`is` or `was` statements, and that the author has seen v2's overall results on them.

## Readouts

The main encoder is text-embedding-3-small. Every experiment also runs on a second, local
encoder: `sentence-transformers/all-mpnet-base-v2` (Apache-2.0, 768 dimensions, revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`), with vectors scaled to unit length. Its card
lists SNLI and MultiNLI among its training data, 277,230 AllNLI pairs out of many sources,
so it has seen this kind of pair. WANLI isn't listed. PCA size and the rules stay the
same, and tau is set separately for each encoder.

Each condition is embedded and scored the same way:

| Name | What it is | Role |
| --- | --- | --- |
| `vector_probe` | Logistic regression (C=0.1) on 64 PCA components of `[abs(a-b), a*b]`; scaling and PCA fit on training folds only | Main readout |
| `base` | v2's five pair scores, same regression | Reference |
| `lexical` | v2's word-overlap features | Shows what plain words alone give |
| `nli` | Same local model and revision as v2, unchanged, contradiction probability | Reference for a model trained on this task |
| `hash` | `vector_probe` on the offline hash encoder | Negative control |

`vector_probe` is new. It reads the whole difference between vectors, not a few summary
scores, because the question is what the embedding holds. It has never been run on real
embeddings, so its accuracy is unknown.

Evaluation matches v2: leave one genre out, pooled average precision (AP), 2,000
bootstrap resamples of genres, seed 42. A probe is trained separately for each
condition, so results show what is recoverable at that condition. WANLI uses a model
trained on the primary set at the same condition, with a pair-level bootstrap.

## Experiment 1: dilution

Add k = 0, 1, 2 or 4 filler sentences, the **same ones to both statements**, so that
most of the text agrees and one part differs.

**Filler rules, none using labels:** drawn from other pairs' statements in the same genre
(all of WANLI counts as one group), sharing no content word with either statement. The
real statement goes in a seeded random position, the same on both sides. Three filler
draws per level, with AP averaged across draws.

**Filler audit, before freezing:** three free models from the
[multi-model review](multi-model-review.md) each judge 100 sampled k=1 pairs, twice, blind:
does the added sentence change whether the two texts contradict? A pair fails if at
least two of three models say yes in either run. The bar is 90 of 100, and the
any-dissent count is reported too. Under 90, stop and take it to the owner rather than
adjusting the rule.

**Primary measure:** D = AP(k=0) minus AP(k=4), for `vector_probe`.

- **DILUTES:** D is at least +0.03, its interval's lower bound is above 0, it beats the
  null's 97.5th percentile, and it beats the hash control's D.
- **STABLE:** the interval's upper bound is below +0.03.
- **UNCLEAR:** anything else.

The null swaps each pair's k=0 and k=4 scores at random, 200 times, within genre.
Every readout is reported at every k, including whether `nli` fades too.

## Experiment 2: context ladder

Each rung is built by a fixed word rule, so no model and no label is involved.

| Rung | Text kept | Rough match to the original idea |
| --- | --- | --- |
| L0 | Full statement | Full context |
| L1 | Content words, keeping `not`, `never`, `no`, `n't` | Attribute and value |
| L2 | Content words, negation words removed | Same, testing how much negation carries |
| L3 | Only words not found in the other statement, negation kept | Value alone |

These are proxies. Public NLI pairs have no labeled subject, attribute or value, so the
exact three-form ladder needs a different dataset.

**Empty sides:** a side with nothing left gets a zero vector and an `empty_side` flag.
L3 leaves a side empty for 18 A and 26 B statements in the primary set (2 pairs both
sides), and 38 A and 189 B in WANLI (15 both). These are draft counts, and WANLI's L3
result must be read with that in mind. The count split by label is reported.

**Primary measures:** L1 minus L0 and L3 minus L0, for `vector_probe`, each called:

- **CLEARER:** delta at least +0.03, lower bound above 0, above the null, above hash.
- **WEAKER:** the mirror image at -0.03.
- **NO CLEAR CHANGE:** anything else.

L3 can win for a dull reason: annotators' favorite contradiction words, such as `never`.
So L2 is compared with L1, and L3 is reported beside `lexical` and v2's statement-B-only
model. An L3 gain that doesn't beat those is about word choice, not geometry.

## Experiment 3: word removal

**Pass A, one word at a time:** delete each word from each statement, re-embed, and
rescore with the probe that never saw that pair's genre. A word's importance is the
change in logit score. `nli` gets the same treatment on the primary set, with
probabilities clipped before the logit, because many sit near 0 or 1.

**Pass B, over and over:** delete a random 10%, 25% or 50% of words from both statements,
with five draws per rate, and plot AP against the rate for each readout.

**Primary measures, MultiNLI:**

- **Concentration:** the share of a pair's absolute importance landing on words unique
  to that side, minus those words' share of the pair's words. CONCENTRATED if the mean
  excess is at least +0.05, its interval's lower bound is above 0, and it beats hash.
- **Agreement with NLI:** per-pair Spearman correlation between probe and NLI importance,
  against a null that shuffles words within pairs 200 times. SHARED if the mean is at
  least 0.2, its lower bound is above 0, and it beats the null's 97.5th percentile.

**Reported whatever happens:** both measures split by label; the share of importance on
negation words; and a check that deleting the top 25% of words by importance lowers the
probe's score more than deleting a random 25%. That check only confirms the importance
scores are meaningful. It is partly circular, so it isn't a finding.

**Labels aren't preserved here.** Deleting `not` can turn a contradiction into a
non-contradiction. Pass A and B measure what the score depends on, not whether the
score is right after deletion. Deleted sentences are also often ungrammatical.

**Weak probe warning:** if `vector_probe` at k=0 doesn't beat `lexical` with an interval
above zero, Experiment 3 explains a weak model and must be reported that way.

## Experiment 4: compression

Apply a nonlinear function to each dimension of `a-b` **before** anything is combined.
Applied to one final score it can't matter: a log keeps the order of examples, so AP
doesn't change. It also can't change a random forest fed that feature, because trees
only use order within a feature. So this test uses the logistic `vector_probe`.

| Arm | Difference features | Compare with |
| --- | --- | --- |
| `abs_linear` | `abs(a-b)`, the main `vector_probe` | |
| `abs_log` | `log1p(abs(a-b)/tau)` | `abs_linear` |
| `signed_linear` | `a-b` | |
| `signed_asinh` | `asinh((a-b)/tau)` | `signed_linear` |

Each arm keeps `a*b` unchanged. Compressed arms are compared with the linear arm of the
same sign handling, so a gain can't just come from keeping the sign.

**Setting tau:** the median `abs(a-b)` across all dimensions, from training folds only.
Scale matters. In a label-blind check of the cached v2 vectors, the median per-dimension
difference was 0.012 on MultiNLI and 0.010 on WANLI. At tau = 0.1, `asinh` stayed within
5% of linear for 99% of values, so it would do almost nothing. At tau near the median,
that fell to about 32%. Half and double tau are reported as sensitivity checks.

**Primary measures, at k=0:** `abs_log` minus `abs_linear`, and `signed_asinh` minus
`signed_linear`, each called CLEARER, WEAKER or NO CLEAR CHANGE under Experiment 2's
rules. Here CLEARER means compression helped.

**Reported whatever happens:** a training-free score, the sum over dimensions of
`log1p(abs(a-b)/tau)`, beside the same sum without the log and Euclidean distance; and
each arm at k=4, to see whether compression slows dilution.

**Prior evidence against the dimension-level version:** the same check found no dominant
dimensions. Across a set, the top 10 of 1,536 dimensions held 1.3% of the squared
difference, and the largest dimension's spread was 1.7 times the median. The probe also
standardizes each dimension first. So a null result here wouldn't rule out compression
across aligned facts, which is a different test.

No new embeddings are needed.

## Reading the results together

Only the seven named primary measures carry a verdict, all on text-embedding-3-small.
Everything else, including WANLI and the second encoder, is descriptive and can't rescue
or overturn one. Each measure's WANLI value and second-encoder result are reported beside
it, and a verdict is described as general only if the second encoder gets the same verdict
under the same rule. There is no overall PASS or STOP.

## Order of work

1. Owner reviews this draft.
2. Write the code and tests, with deletion, filler and ladder rules tested directly.
3. Run the filler audit.
4. Run the full protocol on the hash encoder, and time `nli` on the deletion set. Write
   the hash values here.
5. Freeze by renaming this file.
6. Run once and report whatever it shows. The owner has said embedding needs no further
   approval.

## Cost

Sizes are from label-blind counts on the pinned data. The primary set has 903 unique
statements, averaging 11.1 words, and WANLI has 1,450, averaging 12.3.

| Item | Texts | Tokens, rough |
| --- | ---: | ---: |
| Dilution | 23,500 | 1.2M |
| Ladder | 5,200 | 0.05M |
| One-word deletions | 31,100 | 0.5M |
| Random deletions | 39,200 | 0.5M |
| **Total** | **about 99,000** | **about 2.3M** |

At $0.02 per million tokens that's about $0.05. The real run gets a manual spend ceiling
of $0.25. The second encoder and `nli` run locally for free: about 99,000 texts and about
35,000 pairs, with run time unknown until step 4.

## Known limits

- **This measures the problem, not a cure.** Filler is identical on both sides, so
  removing it is trivial, which means Experiment 1 can't test whether chunking rescues
  the signal. That needs naturally multi-fact sentences.
- Filler is added sentences, not extra clauses within one sentence, which may dilute
  differently.
- The ladder is a word-level proxy for attribute and value.
- Two encoders can show a result isn't unique to one model, not that it holds for all
  embeddings. The second one also trained on NLI pairs, so agreement could partly reflect
  that training. All findings apply to short `is` or `was` statements.
- Probes differ across conditions, so differences mix what's in the embedding with how
  easily 582 pairs can train a probe on it.
- Ten genres is few clusters. The +0.03, +0.05 and 0.2 thresholds are judgment calls,
  set before seeing results.

## Deferred

- Whether incompatibility is separable from mere difference (Austin/Denver vs
  Austin/Texas) on held-out attributes. This needs a new labeled set.
- Whether v2's transformed-sentence gain came from how contradictions were written.
- The exact full-context, attribute-value, value-only ladder, and atomic-fact extraction.
- Compression applied per aligned fact and then combined. This needs multi-fact data.
