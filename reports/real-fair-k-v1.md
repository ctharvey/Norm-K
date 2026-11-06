# Fair K test v1: first real-embedding result

**Run date:** November 5, 2026
**Protocol:** [preregistration-matched-v1.md](../docs/preregistration-matched-v1.md), frozen at ce2b1e7, cap corrected at 17777b1 before any result
**Encoder:** text-embedding-3-small, 1536 dimensions, 771 embeddings, one run
**Data:** 144 hand-written pairs, 12 relation families, labels assigned by Claude and **not independently reviewed**
**Later label review:** [label-review-v1.md](label-review-v1.md) changed no labels, and the rerun matched this result exactly. That review was by LLMs, plus owner rulings on four pairs.

## Frozen result

| Transformation | Delta AP, K over `full` | 95% interval | Shuffled-K 97.5% | Hash control | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Negation (primary) | **+0.0598** | +0.0157 to +0.0902 | +0.0272 | +0.0224 | **PASS** |
| Subject swap (secondary) | +0.0079 | -0.0029 to +0.0259 | +0.0280 | -0.0224 | STOP |

All four PASS conditions were met for negation. Under the protocol, a PASS justifies
replication on independently reviewed data. It does not show that K detects
contradictions, and it does not justify Menhir integration.

Average precision (floor 0.50):

| Model | Negation | Subject swap |
| --- | ---: | ---: |
| `lexical` | 0.662 | 0.662 |
| `base` | 0.571 | 0.571 |
| `base_corners` | 0.660 | 0.573 |
| `full` | 0.712 | 0.677 |
| `full_k` | 0.772 | 0.685 |

Registered secondary results for negation:

- `base_corners` over `base`: +0.089, interval +0.023 to +0.136.
- Mean `k_norm`: 0.337 vs 0.207 at high overlap and 0.460 vs 0.421 at low overlap,
  contradiction vs other. The exploratory prediction held in direction at both levels.

## Exploratory checks

**Everything below was chosen after seeing the result.** These checks used cached
vectors with no new API calls, on the same 144 pairs. They can't change the frozen
decision, and they need replication. The script is not in this repository.

**The gain is not new information.** On unit vectors ||K||^2 is an exact linear
combination of the six pairwise cosines. Giving the logistic baseline all six left
K's gain at +0.058 (interval +0.016 to +0.087), so K is acting as a useful
pre-built combination that the regularized model did not find by itself.

**A flexible model does not need K.** Random forests, leave-one-family-out, mean of
3 seeds:

| Features | AP |
| --- | ---: |
| Lexical only | 0.640 |
| Base only | 0.739 |
| Base + lexical (no transformed sentences) | 0.807 |
| Six cosines only | 0.850 |
| Base + lexical + corners + both remaining cosines | 0.871 |
| The same + K | 0.887 |

- Adding K to the last row: +0.015, interval -0.012 to +0.042.
- Adding the transformed-sentence scores to base + lexical: +0.064, interval +0.015 to +0.125.

This points to the negated sentences carrying the useful information, with K as one
way to combine it.

**The gain is fragile.**

- It sits in the low-overlap pairs (+0.074). High-overlap pairs got worse (-0.020).
- Dropping the ownership family cuts the delta to +0.029, below the 0.03 bar. The
  other 11 families each leave it between +0.056 and +0.065.
- Controlling for the distance between the changed word and the negated verb cuts it
  to +0.042, and its interval then crosses zero (-0.004 to +0.070). That distance
  correlates with `k_norm` at -0.59, and this is the shortcut the protocol flagged.

**K may track opposition, not contradiction.** In high-overlap pairs, mean `k_norm`
was 0.337 for contradictions, 0.281 for temporal updates and 0.179 for compatible
pairs. The largest upward rank moves included temporal updates such as closed/open
at different times. This fits the view that time and cardinality need the belief
lane. Only 10 temporal-update pairs are in the high-overlap cell.

## Limits

- One author wrote the pairs and assigned the labels.
- 12 families, one run, one encoder, 50% prevalence.
- All pairs fit a narrow grammar, and name-only vectors can't be transformed.
- No NLI baseline, so this says nothing about practical value.

## Next

1. Have independent reviewers label this set, and report any change in the decision.
2. Build a new v2 set with the edit-position shortcut balanced, written by someone else
   if possible, with more families.
3. Preregister both K and the transformed-sentence scores with a flexible model, so the
   exploratory finding gets a real test.
4. Add an NLI baseline on the same pairs.

## Reproduce

~~~bash
python -m norm_k.cli k-fair --encoder openai --max-new 800 --output outputs/k-fair-real
~~~

The cache and outputs are git-ignored.
