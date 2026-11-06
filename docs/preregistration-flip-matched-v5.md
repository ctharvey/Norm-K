# Flip score size-matched test v5: frozen protocol

**Status:** frozen. The commit that renamed this file from its draft is the freeze point.
Changes after a real result must be reported as deviations.

## Question

[v4](../reports/flip-placebo-v4.md) found negation's flip beat two neutral edits, but
negation moved statements 2 to 4 times further. Does negation still win against a
non-negating edit that moves each statement the same distance?

## Design

Same flip score as v4. The comparison is `matched`: for each statement, the edit from a
fixed pool whose cosine distance from the statement is closest to negation's distance on
that same statement, for that encoder. Matching uses only the statement's own vectors,
never a label or its partner.

| Pool edit | Form |
| --- | --- |
| `in_fact` | `is in fact` |
| `as_it_happens` | `is as it happens` |
| `note` | `Note: ...` |
| `plainly` | `To put it plainly: ...` |
| `for_consideration` | `Here is a statement for your consideration: ...` |
| `read_carefully` | `... Please read this statement carefully.` |
| `record` | `... This sentence is part of a longer written record.` |
| `record_office` | `... This sentence is part of a longer written record kept in an office.` |
| `copied_line` | `Copied line: ... End of line.` |
| `wrapper` | `The following line was copied without changes from a document: ... End of the copied line.` |

`tense`, swapping `is` and `was`, changes meaning without negating and is reported only.

**Calibration, no labels, no test pairs:** distances on 200 statements already used in v4.
The first eight edits left a gap on `all-mpnet-base-v2`, so `record_office` and `copied_line`
were added. Final gaps between matched and negation's distance:

| Encoder | Negation mean | Matched mean | Mean gap | Median absolute gap |
| --- | ---: | ---: | ---: | ---: |
| `text-embedding-3-small` | 0.105 | 0.105 | +0.0006 | 0.0077 |
| `all-mpnet-base-v2` | 0.226 | 0.220 | -0.0054 | 0.0164 |

## Data

**Primary, fresh:** 1,500 more MultiNLI train pairs, in
[flip-v5-pair-ids.tsv](../data/benchmarks/flip-v5-pair-ids.tsv), selected by v4's label-blind
rule with seed 2030, sharing no statement with v2's or v4's sets.
**Replication:** v2's 725 WANLI pairs.

## Primary measure

Fresh set, `text-embedding-3-small`: F_negation AP minus F_matched AP.

- **Gate:** absolute mean gap at most 0.01 and median absolute gap at most 0.02 on the
  test statements. If the gate fails, the verdict is UNCLEAR.
- **SURVIVES-MATCHING:** delta at least +0.03, interval's lower bound above 0, above the
  null's 97.5th percentile, above the hash encoder's delta.
- **SIZE-EXPLAINED:** the interval's upper bound is below +0.03.
- **UNCLEAR:** anything else.

Intervals and null match v4. The result is called general only if `all-mpnet-base-v2`
gets the same verdict by the same rule. Everything else is descriptive.

## Reported whatever happens

Every edit's AP and mean distance, match quality and which edits were chosen, `tense`, the
largest fixed edit, pairs with no negation word, and WANLI. Tokens are measured from the
API's usage reports.

Added after the [prior-art audit](literature/prior-art-flip-2026-11-06.md), before any test
pair reached a real encoder. These are reported only and don't change the primary measure:

- **Wrong partner:** AP for negation and matched with each B given another pair's A in its
  genre, and their delta. Matching cancels most of the edit-size term in
  `F = 0.5*(d_a - d_b)·(b - a) - 0.5*(m_a + m_b)`, so the delta tests direction. It doesn't
  rule out cues in single statements, which is what this checks.
- **Statement only:** AP from negation's edit size alone, for both statements and for B alone,
  in both directions.
- **Matched at the verb:** the delta on pairs where the matched edit is `in_fact` or
  `as_it_happens`, on both statements and on either. This set may be small.

## Cheap gate

A separate question from the primary one: could a cheap score decide which pairs go on to an
NLI model? A gate needs ranking, not a cause, so this has its own rule and doesn't depend on
the verdict above. The baseline is plain cosine, using whichever direction keeps more.

**Pool:** every real pair in the set, plus random pairs of one pair's A with another's B, until
contradictions are 1% of the pool. Random pairs stand in for unrelated negatives. A few may
truly conflict. The 0.1% pool is reported only.

**Scores:** `flip`, the v5 negation flip, and `negated_cosine`, which is
`0.5*[cos(N(A),B) + cos(A,N(B))]`, or flip plus cosine. Each gets its own verdict. The hash
run suggested raw flip can fail here by design, because subtracting cosine discards
relatedness, so `negated_cosine` was declared before any real encoder ran. Two scores are
tested, so a pass on only one is weaker evidence.

**Margin:** recall minus cosine's recall with the top 5% of the 1% pool passed on, fresh set,
`text-embedding-3-small`. The interval resamples contradictions 2,000 times with cutoffs held
fixed, so it ignores cutoff uncertainty.

- **GATE-PASS:** margin at least +0.05, lower bound above 0, above the hash encoder's margin,
  and a positive margin on WANLI.
- **GATE-FAIL:** the interval's upper bound is below +0.05.
- **UNCLEAR:** anything else.

Called general only if `all-mpnet-base-v2` gets the same verdict. Recall at 10%, 20% and 50%
of each undiluted set, and at 1%, 2%, 5% and 10% of both pools, is reported for flip, matched,
negated cosine and cosine. No NLI model runs here, so a pass shows ranking only, not time saved
or end-to-end recall.

## What each result leads to

Set before any real result, adapting the prior-art audit's stop rule.

- **SURVIVES-MATCHING:** one out-of-domain test on NaN-NLI.
- **SIZE-EXPLAINED:** mechanism claims stop. The gate result is still read on its own.
- **GATE-PASS on either score:** one end-to-end test with a local NLI model, against cosine
  followed by NLI.
- **SIZE-EXPLAINED and GATE-FAIL on both scores:** pin the work with a short negative report
  and stop, including `asinh`, extra encoders and chunking.
- **Anything else:** back to the owner.

## Edit audit

**Rule, committed before any reviewer sees a statement.** As in v4: the same three free
models, blind, twice, in different orders, in empty directories. 100 fresh-set statements,
seed 2030, each with one pool edit assigned at random. A model votes invalid if it says
invalid in either run. A statement is invalid if at least two of three vote invalid. The
bar is 90 of 100, and the any-dissent count is reported. Under 90, stop and take it to the
owner. The prompt counts added framing as valid if the main claim is unchanged.

**Route change, committed before any reviewer sees a statement.** OpenCode's free route hit
its usage limit before any reviewer returned output, and the same models on NVIDIA returned
403, so the audit runs through the OpenCode harness on OpenRouter: `z-ai/glm-5.3-flash`,
`deepseek/deepseek-v4.1-flash` and `nvidia/nemotron-3.5-super-vl:free`. The first two are paid,
under a $0.25 cap. The vote rule, bar, statements and batches don't change. This differs from
v4's route:

- A different host, so the weights or settings may not be identical to the free route's.
- Sessions use `noMcp` with auto-approve off, in an empty folder. Batches go in as direct
  input, since the harness's task delivery needs a file read.
- The coding-agent prompt explored the workspace when tested, so sessions use
  [a reviewer agent](../data/benchmarks/flip-v5-reviewer-agent.md) with all tools off. A canary
  confirmed it had no read tool.
- The workspace `AGENTS.md` still loads into context, as general instructions with no labels
  or test data, so reviewers aren't as isolated as under `--pure`. Extra text and reasoning
  around the JSON lines are ignored. Only objects with a known ID and label count, and the last
  one per ID is the vote.
- A fresh session runs each batch, so no run sees another's answers.

If a model can't complete on this route, stop and take it to the owner. Don't swap models.

## Completed before freezing

**Edit audit: passed, 100 of 100.** The rule was committed at 2925ea2 and the route at 3e3a4ca,
before any reviewer saw a statement. All 18 runs ended complete with no tool use, and measured
OpenCode spend rose about $0.01. One Nemotron batch came back with a garbled line and used
its one retry. Votes are in [flip-v5-edit-audit.tsv](../data/benchmarks/flip-v5-edit-audit.tsv).
No person read these statements.

**This audit is a weak filter.** GLM and Nemotron called all 100 valid in both runs, so the
two-of-three rule had no way to fail here, and no known-bad item tested whether they would
catch a broken edit. DeepSeek dissented on 7, all `as_it_happens`, and changed 7 votes between
runs. Counting any dissent as invalid gives 93. Those 7 are the edit the matched-at-verb check
relies on, so read that check with care.

**Hash dry run,** frozen in [flip-v5-hash-control.json](../data/benchmarks/flip-v5-hash-control.json):

| Set | Negation vs matched | Mean gap | Median absolute gap |
| --- | ---: | ---: | ---: |
| Fresh | +0.0484 | +0.0276 | 0.0299 |
| WANLI | +0.0691 | +0.0250 | 0.0272 |

Hash fails the size gate on both sets, so its own verdict is UNCLEAR. Its deltas are still the
values a real result must beat.

| Gate margin, fresh set | Hash |
| --- | ---: |
| `flip` | -0.5797 |
| `negated_cosine` | +0.0062 |

**Seen before freezing:** the fresh set's size and contradiction count (483 of 1,500), WANLI's
(139 of 725), and all hash scores. No real-encoder score on any test pair.

**Key:** the owner said embedding needs no further approval. Menhir's OpenAI key is loaded
into the run's process only. About 971,000 words.

## Run

Once per encoder, each refusing to replace its result:

~~~bash
python scripts/v5_flip_matched.py run multinli.zip wanli_test.jsonl data/benchmarks/flip-v5-pair-ids.tsv openai outputs/flip-v5/openai.json data/benchmarks/flip-v5-hash-control.json --max-new 60000
python scripts/v5_flip_matched.py run multinli.zip wanli_test.jsonl data/benchmarks/flip-v5-pair-ids.tsv mpnet.npz outputs/flip-v5/mpnet.json data/benchmarks/flip-v5-hash-control.json
~~~

## Known limits

- Equal distance isn't equal direction. A pool edit may move a statement a different way
  than negation, and that difference is the point, but it can't be fully separated from
  negation's meaning.
- Long additions change text length, while negation doesn't.
- Same sentence limits as v4: short `is` or `was` statements, and MultiNLI.
- Pool edits were tuned for size on used statements, not for accuracy.
- The gate pool's random pairs are an approximation of a real, mostly unrelated store.
