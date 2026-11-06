# Flip score size-matched test v5: draft protocol

**Status:** draft. The commit that renames this file from its draft is the freeze point.
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

## Edit audit

**Rule, committed before any reviewer sees a statement.** As in v4: the same three free
models, blind, twice, in different orders, in empty directories. 100 fresh-set statements,
seed 2030, each with one pool edit assigned at random. A model votes invalid if it says
invalid in either run. A statement is invalid if at least two of three vote invalid. The
bar is 90 of 100, and the any-dissent count is reported. Under 90, stop and take it to the
owner. The prompt counts added framing as valid if the main claim is unchanged.

## Order of work

1. Commit this draft and the code.
2. Run the edit audit and hash dry run, and write the results here.
3. Freeze by renaming this file.
4. Run once per encoder, and report whatever it shows.

## Known limits

- Equal distance isn't equal direction. A pool edit may move a statement a different way
  than negation, and that difference is the point, but it can't be fully separated from
  negation's meaning.
- Long additions change text length, while negation doesn't.
- Same sentence limits as v4: short `is` or `was` statements, and MultiNLI.
- Pool edits were tuned for size on used statements, not for accuracy.
