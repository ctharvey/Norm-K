# Label review v1: blind LLM review of the matched set

**Run date:** November 5, 2026
**Method:** "Label review method" in [preregistration-matched-v1.md](../docs/preregistration-matched-v1.md), committed at a1279e6 and 2ee1a21 before any reviewer labels existed
**Labels:** [matched-2x2-v1-label-review.tsv](../data/benchmarks/matched-2x2-v1-label-review.tsv)

This is an **LLM review, not a human review**. Agreement here means three other
models read the pairs the same way as the author. It doesn't show the labels are
right, and models can share the same blind spots.

**Status:** four flagged pairs are waiting for the project owner. The frozen test
has not been rerun.

## Result

All three reviews are valid: 144 labels each, and no tool use in any of the 12
final runs.

| Reviewer | Agrees with author on contradiction vs not | Exact label match | Answered ambiguous |
| --- | ---: | ---: | ---: |
| `deepseek-v4.1-flash-free` | 143 / 144 | 129 / 144 | 0 |
| `glm-5.3-flash-free` | 142 / 144 | 124 / 144 | 0 |
| `nemotron-3.5-super-vl-free` | 143 / 144 | 137 / 144 | 0 |

Reviewers agree with each other on contradiction vs not for 141 or 142 of 144
pairs. All three match the author's exact label on 115 pairs. The other exact-label
differences that don't change contradiction vs not are reported only and don't
affect the binary target.

## Flagged pairs

Four pairs are flagged, each by one reviewer. The owner decides these.

| Row | Family, cell | Statement A | Statement B | Author | Dissent |
| ---: | --- | --- | --- | --- | --- |
| 17 | employment, HN | Hugo is a part-time employee of Acme. | Hugo is a part-time employee of Globex. | compatible | GLM: contradiction |
| 57 | ownership, LC | Dev owns no vehicles of any type. | Dev is the registered keeper of a motorcycle and a van. | contradiction | DeepSeek: compatible |
| 66 | location, HN | At 3 PM Friday, Gwen was near the Paris office. | At 3 PM Friday, Gwen was near the Paris station. | compatible | Nemotron: contradiction |
| 85 | event, HC | Reza's wedding is on June 4, 2027. | Reza's wedding is on July 9, 2027. | contradiction | GLM: temporal_update |

## The one retry

GLM's first run of batch 1 wrote `ambiguus` for row 42, which is not a valid label.
The batch was rerun once, as the protocol allows, and the rerun replaced it. The
first output was kept outside the repo in the session scratchpad.

The rerun changed 5 of that batch's 36 labels, so GLM is not stable between runs
on these pairs. Two of the changes affect flagging:

| Row | Statement A | Statement B | Author | GLM first run | GLM rerun |
| ---: | --- | --- | --- | --- | --- |
| 42 | Uri is the tallest of the two twins. | Uri is the youngest of the three cousins. | compatible | `ambiguus` | compatible |
| 92 | Vik's exam is tomorrow morning. | Vik was done with that exam three weeks ago and it is over. | contradiction | temporal_update | contradiction |

Neither pair is flagged under the rule, which uses the rerun. Had the first run
been valid, both would have been. Only batch 1 was run twice, so other batches
may hide the same instability.
