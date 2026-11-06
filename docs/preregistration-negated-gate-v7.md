# Negated-cosine gate v7: frozen protocol

**Status:** frozen. The commit that renamed this file from its draft is the freeze point. Changes
after a real result must be reported as deviations. The owner approved running it. It doesn't depend
on v6, which stopped at its bridge check.

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

## Details set before any v7 score

Committed before any reviewer saw a statement and before any encoder read a v7 pair.

- **Data:** `train.jsonl` from `alisawuffles/WANLI` at `61c95318fd71c55b6ba355d76253254615f387ec`, SHA-256
  `85058cf017a911e89242dc29fa0a4ddaad3664cb923dc0a82145fdda14b694e5`. IDs are in
  [gate-v7-pair-ids.tsv](../data/benchmarks/gate-v7-pair-ids.tsv), chosen with labels unread.
- **Duplicates:** selection keeps one pair per pair of statements, by lowest ID, so a retrieved pair
  can't clash with another real pair's label. Retrieval runs once per distinct A.
- **NLI:** revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, as in v2, premise as A. The real
  encoders read every candidate, so ceilings are measured. Hash reads only what its two rankings pass.
- **Hash margin:** hash encoder, same pipeline, committed before any real encoder reads a v7 pair.

**Edit audit.** 100 selected statements with v5's negation edit, seed 2031, plus 10 known-bad items
built from seen v5 statements: 5 left unchanged and 5 with `not` placed before the verb, both invalid
under the prompt. They're mixed in blind. Same three models and route as v5, twice, in different
orders, in fresh sessions with tools off. A model votes invalid if it says so in either run. A model
counts only if it flags at least 8 of the 10. If fewer than two count, stop and go to the owner.
Votes from models that don't count are dropped. A real item is invalid if at least two counted models
say so. The bar is 90 of 100; under it, stop and go to the owner. The any-dissent count is reported.
The prompt is v4's negation prompt. If a model can't complete, stop. Don't swap models.

Estimated OpenAI cost is under a cent for the 19,936 texts. NLI runs locally.

## Completed before freezing

**Edit audit: passed, 94 of 100.** All three models caught all 10 known-bad items, so all three
count. 12 real items drew at least one dissent. Votes are in
[gate-v7-edit-audit.tsv](../data/benchmarks/gate-v7-edit-audit.tsv). All 18 batches ended complete
with no tool use.

Two things went differently from a clean run. The driver crashed on HTTP timeouts with 18 sessions
open, and the six sessions it lost had already answered, so those replies were collected. Four
batches then came back incomplete, GLM's one short and Nemotron's three with prose or a stall, and
each used one retry in a fresh session, as in v5, with no hand repair. First attempts weren't used.
The retry wasn't written into this rule in advance, and the driver's wait was lengthened for it.

**Hash dry run,** frozen in [gate-v7-hash-control.json](../data/benchmarks/gate-v7-hash-control.json):
margin **+0.0276**, with found at 0.577 for `negated_cosine` and 0.550 for cosine. Hash reaches that
much with no semantics, so word overlap alone does a lot on this data.

**Seen before freezing:** the set's size and contradiction count (906 of 5,000), the hash scores, and
NLI's scores on the 2,825 pairs hash passed on. No real-encoder score on any v7 pair.

## Run

Once per encoder, each refusing to replace its result. NLI scores for both encoders go in one file,
since a pair's score doesn't depend on which encoder retrieved it.

~~~bash
python scripts/v7_negated_gate.py candidates train.jsonl data/benchmarks/gate-v7-pair-ids.tsv openai openai-pairs.tsv --max-new 21000
python scripts/v7_negated_gate.py candidates train.jsonl data/benchmarks/gate-v7-pair-ids.tsv mpnet.npz mpnet-pairs.tsv
python scripts/v7_nli_scores.py all-pairs.tsv all-nli.tsv
python scripts/v7_negated_gate.py run train.jsonl data/benchmarks/gate-v7-pair-ids.tsv openai all-nli.tsv openai.json data/benchmarks/gate-v7-hash-control.json --max-new 21000
python scripts/v7_negated_gate.py run train.jsonl data/benchmarks/gate-v7-pair-ids.tsv mpnet.npz all-nli.tsv mpnet.json data/benchmarks/gate-v7-hash-control.json
~~~
