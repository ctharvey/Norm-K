# Fair K test: frozen protocol, matched set v1

**Frozen:** November 5, 2026, before any trained-embedding run on this set.
The commit that adds this file is the freeze point. Changes after a real result
must be reported as deviations, not folded in.

## Question

Once a simple model already has every similarity score K is built from, plus
word-overlap features, does adding K help find contradictions in relation
families it has never seen?

## Fixed choices

- **Data:** `data/benchmarks/matched-2x2-v1.tsv`, 144 hand-written pairs: 12
  relation families x 4 cells x 3 pairs. The cells cross high/low word overlap
  with contradiction/other, so positives are 50% of the set.
- **Labels:** assigned by the author, Claude. They are provisional and have not
  been independently reviewed. Any finding is conditional on them.
- **Encoder:** `text-embedding-3-small`, 1536 dimensions, complete statements.
- **K:** K(A,B;T) from [pair-level-k.md](pair-level-k.md), using `k_norm` and
  `k_relative`. `k_relative` depends on A/B order; the file's order is fixed.
- **Primary transformation:** negation. Subject swap is secondary only.
- **Evaluation:** leave-one-family-out logistic regression (C=0.1), pooled
  out-of-family scores, average precision (AP). Seed 42.
- **Mirror pairs:** excluded automatically. The set contains none.

## Models

| Name | Features |
| --- | --- |
| `lexical` | Word-overlap, negation and number features |
| `base` | Cosine, distance, Hoyer sparsity, simple vector statistics |
| `base_corners` | `base` plus the other pairwise cosines and displacement norms |
| `full` | `base_corners` plus `lexical` |
| `full_k` | `full` plus K features |

**Primary measure:** AP(`full_k`) minus AP(`full`), with a 95% bootstrap interval
resampling the 12 families (2,000 resamples). The shuffled-K null uses 200
within-family shuffles.

## Decision rule

**PASS** requires all four:

1. The primary delta is at least +0.03.
2. The interval's lower bound is above 0.
3. The delta is above the shuffled-K null's 97.5th percentile.
4. The delta is above the hash control's delta on this set, **+0.0224**.

**STOP** K-specific work if the interval's upper bound is below +0.03.

**INCONCLUSIVE** otherwise. That means a larger reviewed set is needed. It is
not a partial win.

`k-fair` applies conditions 1-3 and STOP. Check condition 4 by hand.

A PASS would justify replication on independently reviewed data. It would not
show that K detects contradictions, and it would not justify Menhir integration.

## Secondary, reported but not part of the decision

- AP(`base_corners`) minus AP(`base`): do the extra embedded sentences help?
  This is the counterfactual-probing idea, separate from K.
- AP(`lexical`) compared with every embedding model.
- Mean `k_norm` by cell. Exploratory prediction: higher for contradictions than
  for other pairs in both overlap levels. No mechanism yet supports this, so it
  is a guess to check, not a theory being tested.
- Subject swap under the same measures.

## What was done before freezing

Everything here used the non-semantic hash encoder or no embeddings at all.

- **Lexical balance:** mean word overlap is 0.640 vs 0.624 in the high-overlap
  cells and 0.145 vs 0.117 in the low-overlap cells. Single lexical features
  reach AUC 0.45 to 0.58 overall, and up to 0.625 within an overlap level. The
  combined `lexical` model reaches AP 0.662 against a 0.50 floor, so the set is
  not free of surface cues.
- **Rule change after a dry run:** the first rule used `base_corners` as the
  baseline, with no null or hash condition. The hash encoder PASSED it on
  negation (+0.0487), so the rule was invalid. The baseline became `full`, and
  conditions 3 and 4 were added. The hash encoder now gets INCONCLUSIVE on
  negation and STOP on subject swap.
- **No pairs were edited after the dry run.** Editing pairs until the control
  fails would tune the set to the control. A likely cause remains in the data:
  in many contradictions the changed word sits next to the verb that negation
  edits, which bigram features can detect. A real encoder may use this too.

## Known limits

- 12 families is a small number of clusters, so intervals will be wide and the
  likeliest outcome is INCONCLUSIVE.
- One author wrote the pairs and chose the labels.
- All pairs fit a narrow grammar, so real extracted statements may behave
  differently, and name-only vectors can't be transformed at all.
- 50% prevalence is not the real rate among Menhir candidates.
- No NLI baseline is included. One is required before any practical claim, but
  it does not gate this K decision.

## Label review method

Added after the first real run and before any reviewer labels existed. This
is an **LLM review, not a human review**. Its aim is to find wrong or unclear
author labels.

- **Reviewers:** three free OpenCode models from separate families:
  `opencode/deepseek-v4.1-flash-free`, `opencode/glm-5.3-flash-free`,
  `opencode/nemotron-3.5-super-vl-free`. The study's author model, Claude, is not
  a reviewer.
- **Blinding:** each model sees only the label definitions from
  [annotation-protocol.md](annotation-protocol.md), opaque IDs, and the two
  statements, in shuffled batches of 36. It does not see author labels, cells,
  families, K, or the hypothesis.
- **Isolation:** runs start from an empty directory. Any run that uses a tool is
  rejected and that model's review is void.
- **Reasoning effort:** `low` for all three. At the default setting, a DeepSeek
  batch spent its full 32,000-token output limit on reasoning and returned no
  labels, so this was set before any reviewer labels existed.
- **Retries:** one retry, only for malformed or incomplete output.
- **Flagging:** a pair is flagged if any reviewer differs from the author on
  contradiction vs not contradiction, or answers ambiguous. Other label
  differences are reported but don't affect the binary target.
- **Settling:** unflagged labels stand as provisional. The project owner decides
  every flagged pair. Neither the author nor any model settles them.
- **Rerun:** after settling, the frozen test is rerun once from cached embeddings
  with no other changes. Pairs ruled ambiguous are dropped. The result is reported
  beside the original, whatever it is.
- **Later baselines:** these three models can't serve as LLM-judge baselines on
  this set, because they took part in labeling it.

## Run

~~~bash
python -m norm_k.cli k-fair --encoder openai --max-new 800 --output outputs/k-fair-real
~~~

Run it once. The cap is a safety limit; the set needs 771 new embeddings.

**Correction before any result:** this file first said `--max-new 600`. The
cap stopped that attempt before any API call, because 771 embeddings are needed
across both transformations. Only the cap changed.
