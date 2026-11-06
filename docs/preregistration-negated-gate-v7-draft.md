# Negated-cosine gate v7: draft protocol

**Status:** draft, not frozen. Owner's decision whether to run it. It doesn't depend on v6.

## Question

Given a fixed number of NLI calls, does ranking candidates by `negated_cosine` find more
contradictions than ranking them by cosine?

## Why this design

v5's gate failed, partly because its 5% budget left almost no room, and its planted pool made
cosine look better than it should. [The exploratory run](../reports/exploratory-mutations-v5.md)
tried cosine then flip, and found it depended on a cutoff that happened to match the planted pairs.
Inside each statement's nearest neighbors, `negated_cosine` beat or tied cosine at every setting,
and flip mostly didn't. So this rule tests `negated_cosine`, retrieves instead of planting, and measures
what the NLI model finds, since that's what a gate is for.

## Data

WANLI train, CC BY 4.0, which no run here has used. Pairs eligible under v5's rule, sharing no
statement with v2, v4, v5 or WANLI test, then up to 5,000 chosen label-blind with seed 2031. Larger
than v5's sets because those were too easy, with the true partner nearly always a top neighbor.
WANLI was built from MultiNLI seeds, so this isn't independent of that corpus.

## Pipeline

1. **Retrieve.** For each A, take the 5 Bs with the highest cosine, over every B in the set. If a
   retrieved pair matches another real pair's two statements, drop it. Its label could differ.
2. **Rank** candidates by cosine or by `negated_cosine`, which is
   `0.5*[cos(N(A),B) + cos(A,N(B))]`, using v5's negation edit.
3. **Pass** the top 10% of candidates to the NLI model. Both rankings get the same number of calls.
4. **Confirm:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, pinned, with `pC(A,B)` at least 0.5.

The share of 10% and k of 5 were picked on seen v5 data, because that left room: cosine reached at
most 0.540 there, against ceilings of 0.822 to 1.000.

## Primary measure

`text-embedding-3-small`. **Found** is the share of all labeled contradictions in the set that the
NLI model confirms after the gate. **Margin** is `negated_cosine`'s found minus cosine's. The
**ceiling** is found when the NLI model reads every candidate.

- **Room check, applied first:** if the ceiling minus cosine's found is below +0.10, the verdict is
  UNCLEAR, so a rule can't fail for lack of room, as v5's did.
- **GATE-PASS:** margin at least +0.05, interval's lower bound above 0, above the hash margin.
- **GATE-FAIL:** the interval's upper bound is below +0.05.
- **UNCLEAR:** anything else.

The interval resamples contradictions 2,000 times with cutoffs fixed, as in v5, so it ignores
cutoff uncertainty. The result is called general only if `all-mpnet-base-v2` gets the same verdict.

## Reported whatever happens

- Share of contradictions reachable after retrieval, at k of 5, 10 and 20.
- Recall before NLI at 5%, 10% and 20%, for cosine, `negated_cosine`, flip and a random ranking.
- Found for flip. It isn't a verdict, since seen data favored it only at the tightest budget.
- NLI calls and measured time with and without the gate, and embedding calls added by `N(A)`.
- Unlabeled retrieved pairs the NLI model calls contradictions, as a count. They count as negatives
  above, though some may be real, so that may understate precision.

## What each result leads to

- **GATE-PASS on both encoders:** a cheap ranker has shown value on this kind of text only. Next
  would be text a realistic pipeline sees, which needs a dataset that doesn't exist here.
- **GATE-FAIL:** the gate idea stops, including chunking and `asinh`.
- **Anything else:** back to the owner.

## Known limits

- Only `is` and `was` statements can be edited, which is a coverage limit for any real use. If
  v6's bridge shows the wrapper works, a later test could use it here.
- Candidates are other pairs from the same set, not true same-topic distractors.
- The NLI model is a fixed choice, and the gate's value may differ with another.
- Nothing here covers time, or whether two statements are about the same subject.
- No person reviews labels or edits. WANLI labels are human.

## Before freezing

Select pairs with labels hidden. Run the hash control, and an edit audit with known-bad items as in
v6. Pin the model revision. Estimated OpenAI cost is a few cents. NLI runs locally.
