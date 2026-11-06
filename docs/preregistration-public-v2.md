# Fair K test v2: public data, frozen protocol

**Frozen:** November 5, 2026, before any real-embedding run on this data. The commit
that renames this file from its draft is the freeze point. Changes after a real
result must be reported as deviations, not folded in.

## Question

On human-labeled pairs nobody on this project wrote, does K still add anything once
the baseline has the same information, including a flexible model and a trained
contradiction model?

v1 can't answer this. Claude wrote its 144 pairs, and an exploratory random forest
had little use for K there (+0.015, interval crossing zero).

## Data

| Role | Dataset | Why | License |
| --- | --- | --- | --- |
| Primary | MultiNLI 1.0, dev matched + mismatched | Human-written pairs, 5 human labels each, 10 genres to hold out | Paper: may be modified and redistributed; mostly OANC terms, fiction under mixed permissive licenses |
| Replication | WANLI test | Different source, no overlap with the NLI baseline's listed training data | CC BY 4.0 |

Rejected: **ANLI** (CC BY-NC, and in the NLI baseline's training data); **SNLI**
(image captions, not statements, and its download host was unreachable, so it was
never measured).

Pinned files, SHA-256:

- `multinli_1.0.zip`: `049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822`
- WANLI `test.jsonl`: `4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd`

Dataset files stay out of the repo. Commit only the selection script and selected
pair IDs.

**Target:** contradiction vs not, from each dataset's gold label. Rows with no gold
label are dropped. NLI's "contradiction" means the hypothesis can't be true given
the premise, which is looser than this project's same-fact definition.

## Eligibility

`norm_k/public_v2.py` applies these rules, none of which use labels. Selected IDs are in
[public-v2-pair-ids.tsv](../data/benchmarks/public-v2-pair-ids.tsv):

1. The frozen negation rule changes both statements.
2. The pair isn't a mirror pair.
3. Each statement has at most 25 words and one sentence.
4. Each edit lands on a main-clause `is` or `was`: no `own`, `live`, `work` or
   `reside` words, no subordinate-clause word or comma before the edit, and no
   "there is".

| Set | Usable | Rules 1-3 | All rules | Contradictions |
| --- | ---: | ---: | ---: | ---: |
| MultiNLI matched | 9,815 | 819 | 316 | 104 |
| MultiNLI mismatched | 9,832 | 801 | 266 | 88 |
| WANLI test | 5,000 | 1,522 | 725 | 139 |

**Why rule 4 exists:** by the author's reading of a label-free sample, about 7 of 30
rules 1-3 edits were broken or misleading, such as "People sacrifice their does not
live for farmers". After rule 4, nearly all of a 45-statement sample looked valid.
Both are early reads, not an audit.

**Selection limit:** only about 3% of MultiNLI qualifies, all of it simple `is` or
`was` statements. Any result applies to those sentences, not to NLI generally.

Primary set: 582 pairs, 192 contradictions, 33% positive, 10 genres. Genre sizes
are uneven; telephone has only 12 pairs.

## Models

All use text-embedding-3-small and the frozen K features, with negation only.
Subject swap needs a known subject and already got STOP on v1, so it's dropped.

| Name | Features |
| --- | --- |
| `full`, `full_k` | v1 logistic regression (C=0.1), without and with K |
| `rf_full`, `rf_full_k` | Same features in a random forest, 500 trees, `min_samples_leaf=3` |
| `rf_base_lexical` | Random forest without any transformed-sentence scores |
| `nli` | Contradiction probability from `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT), unchanged |
| `nli_full`, `nli_full_k` | `full` plus `nli`, without and with K |
| `hypothesis_only` | Word features from statement B alone, to measure known NLI annotation shortcuts |

Evaluation matches v1: leave one genre out, pooled scores, average precision (AP),
2,000 bootstrap resamples of the 10 genres, 200 within-genre K shuffles, seed 42.

## Decision rule

A comparison **clears the bar** when all four hold, as in v1:

1. The AP delta is at least +0.03.
2. The interval's lower bound is above 0.
3. The delta is above the shuffled-K null's 97.5th percentile.
4. The delta is above the hash control's delta on this set, measured before the
   real run and written here at the freeze.

- **PASS:** `full_k` over `full` and `rf_full_k` over `rf_full` both clear the bar.
- **STOP** K-specific work: the interval's upper bound is below +0.03 for either
  comparison.
- **INCONCLUSIVE:** anything else.

A PASS would show that K adds signal on these sentences. It would not justify
Menhir integration unless `nli_full_k` over `nli_full` also clears the bar.

## Named secondary tests

These are reported whatever the primary result is, and can't rescue it.

- **Transformed sentences without K:** `rf_full` over `rf_base_lexical`, using the
  same bar. If this clears and K doesn't, the idea to keep is the transformed
  sentences, not K.
- **Practical value:** `nli_full_k` over `nli_full`, and `nli` alone against every
  embedding model.
- **WANLI replication:** train on all 582 primary pairs, then score WANLI's 725
  eligible pairs with a pair-level bootstrap. Report the delta for each comparison.
- **Robustness:** pairs with at least 4 of 5 labels agreeing; each genre left out
  of the summary in turn; mismatched genres alone, since the NLI model trained on
  the five matched genres.

## Completed before freezing

**Edit audit: passed, 94 of 100.** The rule below was committed at 19a2a53, before
any reviewer saw a statement. The three free models from the
[label review](multi-model-review.md) each judged 100 statements from the primary
set (seed 2028), twice, in a different order each time, seeing only the original,
the edit and the question. A model voted invalid if it said invalid in either run.
A statement counted as invalid if at least two of three models voted invalid. The
bar was 90. No person checked these statements, by the owner's choice. Twelve more
statements drew one dissent, so counting any dissent as invalid would give 82; the
committed rule did not do that. Votes are in
[public-v2-edit-audit.tsv](../data/benchmarks/public-v2-edit-audit.tsv).

The 100-item batches and then a 400-second limit timed out on DeepSeek before any
complete review existed. Batches became three chunks per run with a 900-second
limit. The statements, orderings and rule did not change.

**Hash dry run: STOP, so the rule is usable.** Condition 4's values, frozen in
[public-v2-hash-control.json](../data/benchmarks/public-v2-hash-control.json):

| Comparison | Hash delta |
| --- | ---: |
| `k_over_full` | +0.0026 |
| `k_over_rf` | -0.0003 |
| `k_over_nli_full` | +0.0012 |
| `corners_over_rf_base_lexical` | +0.0248 |

**Seen before freezing:** the dry run shows every number that doesn't depend on the
encoder. The NLI model alone reached AP 0.980 on the primary set and 0.744 on
WANLI, against 0.33 and 0.19 floors. Hash `nli_full` was only 0.916, below `nli` alone. So
`k_over_nli_full` measures gains over a blended model that is weaker than the NLI
score itself, and the primary set leaves almost no practical room. No rule was
changed after seeing this. The practical case for K must also be judged against
`nli` alone.

**Key:** the owner approved Menhir's OpenAI key for the single real run.

**NLI scores:** model revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, stored in
[public-v2-nli-scores.tsv](../data/benchmarks/public-v2-nli-scores.tsv).

## Cost

The set has 4,704 unique statements, at roughly 20 tokens each: about 94,000 tokens.
At $0.02 per million tokens that's about $0.002. The NLI model runs locally for free.

## Run

~~~bash
python -m norm_k.cli k-public --encoder openai --max-new 4800   --mnli-zip multinli.zip --wanli wanli_test.jsonl   --nli-scores data/benchmarks/public-v2-nli-scores.tsv   --hash-control data/benchmarks/public-v2-hash-control.json   --output outputs/k-public-real
~~~

Run it once. The command refuses to replace an existing result.

## Known limits

- Ten genres is still few clusters, and genre sizes are uneven.
- MultiNLI is public and old, so the encoder or NLI model may have seen it.
- The NLI model trained on MultiNLI's train split, including the five matched
  genres. That favors the baseline, which is acceptable for a harder test.
- The author has seen label and genre counts, unlabeled statement samples, and the
  hash dry run, including the NLI model's scores, but no real-embedding or real-K result.
- Public NLI pairs don't look like Menhir statements. Even a clean PASS would need a
  Menhir-shaped test later.
