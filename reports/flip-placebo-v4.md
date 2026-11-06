# Flip score placebo test v4: result

**Run date:** November 6, 2026
**Protocol:** [preregistration-flip-placebo-v4.md](../docs/preregistration-flip-placebo-v4.md), frozen at 5a35137
**Data:** 1,500 fresh MultiNLI train pairs, 491 contradictions; v2's 582 and 725 pairs as secondary
**Raw results:** [text-embedding-3-small](flip-placebo-v4-openai.json), [all-mpnet-base-v2](flip-placebo-v4-mpnet.json)

## Frozen result: REPLICATES and NEGATION-SPECIFIC

| Fresh set | Delta AP | 95% interval | Null 97.5% | Hash | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Flip over best cosine, `text-embedding-3-small` | +0.296 | +0.248 to +0.336 | +0.050 | +0.064 | **REPLICATES** |
| Negation over best neutral edit, same encoder | +0.288 | +0.244 to +0.329 | +0.054 | +0.042 | **NEGATION-SPECIFIC** |
| Flip over best cosine, `all-mpnet-base-v2` | +0.224 | +0.182 to +0.260 | +0.050 | +0.064 | REPLICATES |
| Negation over best neutral edit, same encoder | +0.117 | +0.079 to +0.152 | +0.036 | +0.042 | NEGATION-SPECIFIC |

Both encoders get the same verdicts, so the frozen rule calls this general, meaning not
unique to one model. The second encoder listed MultiNLI in its training data, so its
fresh-set result may be contaminated.

## Average precision by edit

| Score | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| Base rate | 0.327 | 0.327 | 0.192 | 0.192 |
| Best cosine | 0.421 | 0.541 | 0.263 | 0.244 |
| `in_fact` flip | 0.411 | 0.458 | 0.237 | 0.230 |
| `note` flip | 0.428 | 0.648 | 0.210 | 0.326 |
| `possibly` flip | 0.436 | 0.627 | 0.231 | 0.302 |
| **`negation` flip** | **0.716** | **0.765** | **0.382** | **0.377** |

The flip didn't shrink on fresh pairs: 0.716 against 0.709 on the pairs where it was found.
Every fresh genre stayed between 0.696 and 0.845 on OpenAI, and 0.753 and 0.798 on mpnet.

On secondary sets, all verdicts match except one: mpnet's negation-vs-placebo on WANLI is
UNCLEAR (+0.051, interval -0.008 to +0.118). WANLI is the set it wasn't listed as trained on.

## What it means

- **The flip is a real, replicated signal on these sentences.** It needs no training and
  uses four embeddings and three cosines.
- **On OpenAI, nearly all of it needs negation.** Neutral edits score about what cosine does.
- **On mpnet, a large part doesn't.** `note` alone reaches 0.648, above that encoder's best
  cosine. Editing in general carries signal there, and negation adds a smaller margin.
- **No negation word needed.** On the 1,190 fresh pairs with no negation word on either
  side, the OpenAI flip reached 0.675 (floor 0.279), and mpnet 0.691, against `note` at
  0.356 and 0.583.

## What weakens it

- **Negation is the biggest edit.** It moved a statement 0.100 in cosine distance on OpenAI,
  against 0.026 to 0.046 for the others, and 0.209 against 0.040 to 0.050 on mpnet. So
  NEGATION-SPECIFIC means better than these edits, not proven to come from meaning. A large
  edit that doesn't negate wasn't tested.
- **Some of it isn't about the pair.** With wrong partners, the negation flip still reached
  0.440 on OpenAI and 0.453 on mpnet, against a 0.327 floor. Neutral edits fell to about
  the floor (0.338 and 0.366), so one statement's reaction to negation carries signal alone.
- **The hash control clears every other condition.** Word overlap alone gives part of this,
  though the real encoders beat it by a wide margin on the fresh set.
- **Narrow sentences.** All are short `is` or `was` statements, the fresh set is still
  MultiNLI, and train labels come from one annotator.
- **Performance falls on WANLI,** 0.38 against 0.72, and a trained NLI model stays far ahead:
  0.744 on WANLI in v2. The NLI model wasn't scored on the fresh set, since it trained on it.
- **Audit:** 99 of 100 by LLM vote, no person read the statements.

## Status

- **Tested:** the frozen rule, once per encoder, on 1,500 fresh pairs.
- **Shown:** on these sentences and encoders, the negation flip beats cosine and two neutral
  edits, and isn't explained by negation words.
- **Not shown:** that it's caused by negation's meaning, that it works on other sentence types,
  or that it is new. Schumann and Marx Gomez 2023 is still unread.
- **Assumptions:** the two neutral edits are fair placebos, and `is` or `was` edits are valid
  at v2's audited rate on the fresh set, which wasn't re-audited.

Spend is estimated at about $0.01 from word counts, not measured.
