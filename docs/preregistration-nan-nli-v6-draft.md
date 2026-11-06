# Flip score on NaN-NLI v6: draft protocol

**Status:** stopped at the bridge, not frozen. No NaN-NLI statement has reached an encoder or an NLI model.

## Bridge result: failed

The bridge's rule needed the delta, the lower bound and the size gate. The delta passed and the size
gate didn't, so the rule below sends this back to the owner. The hash control and audit weren't run.

| Seen data | Negation AP | Control AP | Delta | 95% interval | Negation moved | Control moved | Size gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Fresh, `text-embedding-3-small`** | 0.571 | 0.343 | **+0.228** | +0.195 to +0.261 | 0.153 | 0.075 | **Fail** |
| Fresh, `all-mpnet-base-v2` | 0.720 | 0.467 | +0.253 | +0.214 to +0.288 | 0.210 | 0.059 | Fail |
| WANLI, `text-embedding-3-small` | 0.301 | 0.187 | +0.114 | +0.073 to +0.168 | 0.167 | 0.077 | Fail |
| WANLI, `all-mpnet-base-v2` | 0.328 | 0.208 | +0.120 | +0.061 to +0.189 | 0.216 | 0.059 | Fail |

The wrapper does carry signal, but a one-word control doesn't hold edit size equal, which is the
confound v5 removed. These deltas can't be read as negation beating a size-matched edit.
Raw results: [OpenAI](../reports/nan-nli-v6-bridge-openai.json),
[mpnet](../reports/nan-nli-v6-bridge-mpnet.json). 178,063 OpenAI tokens, under a cent.

## Question

[v5](../reports/flip-matched-v5.md) found negation's flip beat a size-matched edit, on short `is`
or `was` statements from MultiNLI and WANLI. Does a negation flip still beat its control on a
hand-built negation test set from outside those corpora?

## Why v5's edit can't be used

NaN-NLI has 258 pairs. v5's `is`/`was` rule covers only 32 of them, with 10 contradictions and 5
distinct premises, which can't support a verdict. So v6 tests a different edit, and that has to be
said in any comparison with v5.

| Role | Form |
| --- | --- |
| Negation | `It is not the case that S` |
| Control | `It is the case that S` |

`S` stays verbatim, including its capital and full stop, in both. The two edits differ by one
word and sit in the same place, so this control is tighter than v5's matched edit. The wrapper
applies to every statement and always negates the whole clause, so the scope errors the prior-art
audit warned about can't come from the edit. It's also a blunter edit than a change at the verb,
and it may carry less signal.

## Data

`data/nan.csv` from [joey234/nan-nli](https://github.com/joey234/nan-nli) at commit
`1f84029a53940b675b62ab2b60980d4cae26eb78`, SHA-256
`533cd0a169662834a5bcbb6c064472e52b549424f43164cd7eabc087bfd6e89c`. All 258 rows, with premise as
A and hypothesis as B: 117 contradiction, 97 entailment and 44 neutral, so the base rate is 0.453.
There are 48 distinct premises, and the largest shares 16 pairs.

The GitHub repo lists no license. The [Hub copy](https://huggingface.co/datasets/joey234/nan-nli)
is CC BY-SA 4.0. Only row numbers and the hash are committed, never the text.

## Before freezing

**1. Bridge, on seen data only.** The wrapper hasn't been tested. Run it on v5's fresh set and
WANLI, both encoders. It passes if the fresh-set `text-embedding-3-small` delta over the control is
at least +0.03 with its interval's lower bound above 0, and the size gate below holds. If it fails,
stop and go back to the owner: a NaN-NLI result then couldn't separate a weak edit from a failed
transfer.

**2. Hash control** on NaN-NLI, as in v5. Its delta becomes a bar.

**3. Edit audit with known-bad items.** v5's audit couldn't fail, so this one is built to. 100
statements with the wrapper, seed 2032, plus 10 broken wrappers built from seen v5 statements and
hidden among them. Same three models, blind, twice, and two of three to call an item invalid. A
model counts only if it flags at least 8 of the 10. If fewer than two models count, stop and go to
the owner. The bar is 90 of the 100 real items. Verbless premises may read badly inside a wrapper,
so this is where they'd show up. A person reading a sample would be stronger, and is still blocker 1.

## Primary measure

`text-embedding-3-small`, all 258 pairs: F_negation AP minus F_control AP, using v5's flip formula.

- **Size gate:** as in v5, absolute mean gap at most 0.01 and median absolute gap at most 0.02
  between each edit's distance from its statement. If it fails, the verdict is UNCLEAR.
- **TRANSFERS:** delta at least +0.03, interval's lower bound above 0, above the null's 97.5th
  percentile, above the hash delta.
- **NO-TRANSFER:** the interval's upper bound is below +0.03.
- **UNCLEAR:** anything else.

The interval resamples whole premises, 2,000 times, because pairs sharing a premise aren't
independent. The null shuffles labels within each premise. The result is called general only if
`all-mpnet-base-v2` gets the same verdict.

**What this can show.** At this size the interval is about 0.13 to 0.16 wide before clustering, and
clustering widens it. So NO-TRANSFER needs an observed delta of about -0.05 or lower, and a small
positive delta will be UNCLEAR. This test can find a clear transfer or a clear reversal, but it
can't show the effect is absent.

## Reported whatever happens

- Cosine AP in both directions, and F_negation against each of entailment and neutral.
- **NLI comparator:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` at a pinned revision, local,
  in both orders: `pC(A,B)`, `pC(B,A)`, their mean, minimum and maximum, and `c - e` in logits. The
  first is the baseline. `c - logsumexp(e, n)` is left out, since it can't change a ranking.
- AP by construction for the four with at least 20 pairs (`not + quantifier`, `not in coordination`,
  `not + focus particle`, `not + degree expression`), with the rest pooled. Too small for verdicts.
- `min(gA, gB)`, from [the exploratory run](../reports/exploratory-mutations-v5.md).
- Wrong partner, with each B given the A of a pair in the same construction with a different premise.
- v5's own `is`/`was` edit on the 32 pairs it covers, as counts and AP only.

## What each result leads to

- **TRANSFERS on both encoders:** the flip isn't tied to `is`/`was` text or these corpora. Back to
  the owner, with no claim about cause.
- **NO-TRANSFER:** claims stay limited to `is`/`was` MultiNLI-style text.
- **UNCLEAR, or encoders differ:** report it and stop. No rerun, and no change of edit.

## Seen before freezing

Row, label, premise and construction counts, plus construction subtype names, which include a few
sample phrases. Eligibility under v5's rule, by label. Interval widths from
[seen v5 data](../reports/exploratory-mutations-v5.md). No statement was printed, embedded or scored.

## Known limits

- It tests a new edit, so a pass or fail doesn't transfer directly to v5's edit.
- 48 premises is a small sample, and 249 of the 258 rows have a sub-clausal premise.
- Hypotheses are minimal edits of premises, so cosine will be high throughout. That makes the test
  hard, and unlike any real pool.
- Hand-built by linguists, public since 2022. Whether either encoder saw it is unknown.
- The dataset's hypothesis annotations are blank on 86 rows, so they aren't used.

## Cost

About 900 texts per encoder on NaN-NLI, plus about 8,900 new short texts for the bridge on seen
statements. That's under a cent for OpenAI in all. The audit should be about $0.01 on v5's route.
