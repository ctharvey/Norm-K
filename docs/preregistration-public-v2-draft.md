# Fair K test v2: public data (DRAFT, NOT FROZEN)

**Status:** draft for the project owner's approval. No embeddings exist for this
data and no K score has been computed on it. The freeze happens only when the
items under "Before freezing" are done and the owner approves.

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

`scripts/v2_eligibility_count.py` applies these rules, none of which use labels:

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

## Before freezing

1. **Audit the edits.** Check 100 seeded, label-free transformed statements. At
   least 90 must be valid negations of the main claim. If not, tighten rule 4 without
   labels and repeat on a fresh sample. Owner to choose who checks.
2. **Build the code:** a loader for the pinned files, random forest and NLI
   models, the WANLI transfer run, and tests for each.
3. **Dry-run with the hash encoder** and write its delta into the rule. If it clears
   the bar, the rule is invalid and must change before any real run.
4. **Approve the key.** The real run needs an embedding key. v1's approval covered
   that run only.

## Cost

At most 5,228 embeddings (1,307 pairs x 4 statements) at roughly 20 tokens each is
about 105,000 tokens. At $0.02 per million tokens, that's about $0.002. The NLI
model runs locally for free.

## Known limits

- Ten genres is still few clusters, and genre sizes are uneven.
- MultiNLI is public and old, so the encoder or NLI model may have seen it.
- The NLI model trained on MultiNLI's train split, including the five matched
  genres. That favors the baseline, which is acceptable for a harder test.
- The author has seen label and genre counts for this data, and unlabeled samples of
  its statements, but no embeddings or K scores.
- Public NLI pairs don't look like Menhir statements. Even a clean PASS would need a
  Menhir-shaped test later.
