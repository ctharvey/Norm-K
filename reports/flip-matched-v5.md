# Flip score size-matched test v5: result

**Run date:** November 6, 2026
**Protocol:** [preregistration-flip-matched-v5.md](../docs/preregistration-flip-matched-v5.md), frozen at b2cec22
**Data:** 1,500 fresh MultiNLI train pairs, 483 contradictions; v2's 725 WANLI pairs, 139 contradictions
**Raw results:** [text-embedding-3-small](flip-matched-v5-openai.json), [all-mpnet-base-v2](flip-matched-v5-mpnet.json)

## Frozen result: SURVIVES-MATCHING, and the cheap gate fails

| Negation over matched edit | Delta AP | 95% interval | Null 97.5% | Hash | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **Fresh, `text-embedding-3-small` (primary)** | **+0.327** | +0.286 to +0.366 | +0.054 | +0.048 | **SURVIVES-MATCHING** |
| Fresh, `all-mpnet-base-v2` | +0.166 | +0.127 to +0.204 | +0.044 | +0.048 | SURVIVES-MATCHING |
| WANLI, `text-embedding-3-small` | +0.162 | +0.098 to +0.235 | +0.057 | +0.069 | SURVIVES-MATCHING |
| WANLI, `all-mpnet-base-v2` | +0.081 | +0.005 to +0.155 | +0.067 | +0.069 | SURVIVES-MATCHING |

Both encoders get the same verdict, so the frozen rule calls this general, meaning not unique
to one model. The mpnet WANLI row clears its bars narrowly. mpnet listed MultiNLI in its
training data, so its fresh-set row may be contaminated.

The size gate passed everywhere. The largest mean gap was -0.0063 and the largest median
absolute gap 0.0197, against limits of 0.01 and 0.02.

| Cheap gate, fresh set | Recall at 5% | Cosine recall | Margin | Verdict |
| --- | ---: | ---: | ---: | --- |
| `flip`, `text-embedding-3-small` | 0.159 | 0.988 | -0.828 | **GATE-FAIL** |
| `negated_cosine`, same encoder | 0.994 | 0.988 | +0.006 | **GATE-FAIL** |
| `flip`, `all-mpnet-base-v2` | 0.259 | 0.934 | -0.675 | GATE-FAIL |
| `negated_cosine`, same encoder | 0.977 | 0.934 | +0.043 | UNCLEAR |

## Average precision

| Score | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| Base rate | 0.322 | 0.322 | 0.192 | 0.192 |
| `tense` flip | 0.326 | 0.361 | 0.204 | 0.202 |
| `matched` flip | 0.369 | 0.581 | 0.220 | 0.296 |
| Best single pool edit | 0.400 | 0.689 | 0.237 | 0.326 |
| **`negation` flip** | **0.696** | **0.748** | **0.382** | **0.377** |
| Negation over best single edit | +0.296 | +0.059 | +0.145 | +0.051 |

The best single edit is picked after seeing labels, so it flatters that comparator, and this
row has no interval.

## Checks added after the prior-art audit

| Check | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| Wrong partner, negation AP | 0.426 | 0.442 | 0.222 | 0.205 |
| Wrong partner, matched AP | 0.322 | 0.349 | 0.191 | 0.187 |
| Wrong-partner delta | +0.104 | +0.093 | +0.031 | +0.019 |
| Real delta minus wrong-partner delta | +0.223 | +0.073 | +0.131 | +0.062 |
| Best statement-only AP (B's edit size, reversed) | 0.392 | 0.344 | 0.321 | 0.256 |
| No negation words, negation AP | 0.660 | 0.684 | 0.330 | 0.265 |
| No negation words, matched AP | 0.327 | 0.514 | 0.188 | 0.255 |
| Matched at verb on either statement, negation AP | 0.744 | 0.749 | 0.449 | 0.350 |
| Same pairs, matched AP | 0.423 | 0.513 | 0.252 | 0.404 |

Base rates without negation words are 0.281 on fresh and 0.166 on WANLI. Verb-position sets
have 434, 279, 162 and 123 pairs. Pairs with the verb edit on both statements are too few to
read, at 14 to 56.

## What it means

- **Size doesn't explain the flip.** That was the main doubt v4 left. Moving a statement the
  same distance with a non-negating edit gives far less, on both encoders and both sets.
- **The strength depends heavily on the encoder.** On OpenAI negation clears every single edit
  by at least +0.145. On mpnet the best single edit comes within about +0.06, and the matched
  edit isn't the strongest control there. mpnet's generic edits already carry much of the
  signal, as v4's `note` result suggested.
- **Part of the gain is in single statements, not the pair.** With the wrong partner, negation
  still beats matched by +0.104 on OpenAI and +0.093 on mpnet. So about a third of OpenAI's
  fresh delta and over half of mpnet's doesn't need the real partner. Annotation cues of the
  kind Gururangan 2018 describes are one possible source. The subtraction in the table is
  descriptive and has no interval.
- **It isn't only the word `not` in the sentence, on OpenAI.** Without negation words the gap
  holds at 0.660 against 0.327. On mpnet's WANLI pairs it nearly vanishes, at 0.265 against
  0.255.
- **This still doesn't show negation's meaning is the cause.** `tense` also changes meaning and
  does almost nothing, but it moves statements far less, so that isn't a matched comparison.
  Equal distance isn't equal direction.

## The cheap gate

**Raw flip can't be a gate.** It subtracts cosine, so it loses the relatedness that separates
real pairs from random ones, and it finds 14% to 26% of contradictions where cosine finds over
93%. The hash run suggested this before any real encoder ran.

**The `negated_cosine` verdict is partly a flaw in the rule.** Real pairs are about 3% of the
1% pool, and random pairs are easy to reject, so a 5% budget lets cosine keep 98.8% on the
primary row. The most any score could gain there is +0.012, so +0.05 was out of reach. That
should have been caught before freezing. The verdicts stand, and nothing here is a pass.

Two descriptive readings, neither a verdict, and both found after seeing results:

| Recall | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| 1% pool, 1% budget: `negated_cosine` | 0.275 | 0.315 | 0.374 | 0.381 |
| 1% pool, 1% budget: cosine | 0.197 | 0.104 | 0.252 | 0.216 |
| Undiluted, top 10%: `flip` | 0.269 | 0.277 | 0.237 | 0.237 |
| Undiluted, top 10%: better cosine direction | 0.130 | 0.197 | 0.137 | 0.137 |

So a tighter budget may leave room, and flip may be better at ranking pairs already known to
be related. That points to a different design: cosine finds related pairs, then flip ranks them
for NLI. It needs its own rule on fresh data, with a harder pool than random pairs. No NLI model
ran here, so none of this shows time saved or end-to-end recall.

## Edit audit

Passed 100 of 100, but as a weak filter. GLM and Nemotron called everything valid in both runs,
so the two-of-three rule had no way to fail, and no known-bad item tested them. DeepSeek
dissented on 7, all `as_it_happens`, the edit most used in the verb check. It ran on OpenRouter
through the harness because OpenCode's free route hit its limit, with a tool-free reviewer agent
and the workspace `AGENTS.md` still in context. One Nemotron batch used its one retry.

## Cost

1,088,558 OpenAI tokens, measured from API usage, about $0.02 at `text-embedding-3-small`'s list
price. The audit raised OpenCode's rounded spend total by about $0.01.

## Limits

- Short `is` or `was` statements, and the fresh set is still MultiNLI.
- WANLI was already used in v2 and v4, so it isn't new data.
- Pool edits were tuned for size, and long additions change text length where negation doesn't.
- No person reviewed any edit, and dedicated NLI models remain far stronger on WANLI.

## Next

By the frozen plan, SURVIVES-MATCHING leads to one out-of-domain test on NaN-NLI. GATE-PASS
didn't happen on either score, so the end-to-end NLI test doesn't follow. The two-stage design
above would be new work for the owner to approve. No novelty is claimed; see the
[prior-art audit](../docs/literature/prior-art-flip-2026-11-06.md).
