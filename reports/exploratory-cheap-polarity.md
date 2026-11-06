# Exploratory: negation-mismatch baseline and cheap polarity checks

**Run date:** November 6, 2026
**Status:** exploratory. Nothing here is a verdict, and v7's frozen GATE-PASS stands. All data has been
seen before, no rule was frozen, and the best mismatch weight was picked on the data it's scored on.
**Raw results:** [cheap-polarity](exploratory-cheap-polarity.json), [no-negation split](exploratory-cheap-polarity-clean.json)
**Scripts:** [cheap_polarity.py](../scripts/exploratory/cheap_polarity.py), [cheap_polarity_clean.py](../scripts/exploratory/cheap_polarity_clean.py)

Saved vectors and saved NLI scores only, with no embedding or paid calls. The script reproduced flip's
AP of 0.696 and v7's found figures. Each negation vector `v` came from a different set from the one it
was scored on.

## Main finding: v7's gain is mostly about negation words

`negated_cosine` beats a simple negation flag, but most of its margin comes from pairs where exactly
one statement has a negation word. Contradictions with no negation word barely gain.

| v7 contradictions, found after NLI | Count | Cosine | `negated_cosine` | Margin | 95% interval | Share of total margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **OpenAI** | | | | | | |
| Exactly one has a negation word | 282 | 0.333 | 0.699 | +0.365 | +0.309 to +0.422 | 87% |
| Neither has one | 562 | 0.582 | 0.605 | +0.023 | 0.000 to +0.044 | 11% |
| Both have one | 62 | 0.742 | 0.790 | +0.048 | 0.000 to +0.113 | 3% |
| **mpnet** | | | | | | |
| Exactly one has a negation word | 282 | 0.284 | 0.681 | +0.397 | +0.340 to +0.457 | 78% |
| Neither has one | 562 | 0.491 | 0.532 | +0.041 | +0.005 to +0.077 | 16% |
| Both have one | 62 | 0.645 | 0.774 | +0.129 | +0.048 to +0.226 | 6% |

So 62% of contradictions, the ones with no negation word, get little from this ranker. That limits
what v7 can be said to show: it's a better way to catch negation-worded contradictions among related
pairs, not evidence of a general contradiction signal.

## Against the mismatch flag

| Found after NLI on v7 | OpenAI | mpnet |
| --- | ---: | ---: |
| Cosine | 0.515 | 0.437 |
| Mismatch first, then cosine | 0.276 | 0.246 |
| Cosine + flag, best weight (0.2 on both) | 0.586 | 0.488 |
| **`negated_cosine`** | **0.647** | **0.595** |
| `negated_cosine` + flag, best weight | 0.658 | 0.599 |

- **`negated_cosine` still beats the flag:** by +0.061 (+0.041 to +0.082) on OpenAI and +0.107 (+0.078
  to +0.137) on mpnet, even with the weight picked in hindsight. The flag gains on exactly-one pairs
  (0.734 against 0.699 on OpenAI, though not on mpnet) and pays for it on the rest (0.502 against
  0.605 with no negation word).
- **The flag adds nothing to `negated_cosine`:** +0.011 (-0.002 to +0.024) and +0.004 (-0.007 to
  +0.015). `negated_cosine` already carries that information.
- **It does favor mismatched pairs:** they are 15% of OpenAI's candidates and 21% of what
  `negated_cosine` passes, against 9% for cosine. On mpnet it's 19%, 24% and 10%.

## The three cheap ideas

**1. Average negation vector.** `v` is stable, with 0.992 and 0.984 cosine between estimates from two
different sets. It doesn't replace real edits, though. As a pair score it's clearly below flip on all
three sets for both encoders, for example 0.519 against 0.696 AP on OpenAI's fresh set.

As a gate it needs no negated embeddings at all, and found 0.566 (OpenAI) and 0.471 (mpnet). That's
+0.051 (+0.034 to +0.068) and +0.034 (+0.019 to +0.049) over cosine, roughly a third and a fifth of
the real edit's gain, and close to hash's own +0.028. It's also below the flag.

**2. Search with the negated version.** Mixed, and only matters where retrieval is hard. WANLI and v7
are near their ceiling with cosine alone, so only the fresh set says anything.

| Fresh set, contradictions reached | OpenAI | Candidates | mpnet | Candidates |
| --- | ---: | ---: | ---: | ---: |
| Cosine top 5 | 0.940 | | 0.822 | |
| Cosine top 10 | 0.969 | 15,000 | 0.874 | 15,000 |
| Cosine 5 + real negated query 5 | 0.959 | 10,215 | 0.913 | 11,460 |
| Cosine 5 + `E(s)+v` query 5 | 0.944 | 9,731 | 0.876 | 9,401 |

On mpnet, the real negated query reached more than cosine's top 10 with fewer candidates. On OpenAI
it didn't. This compared unequal candidate counts on 1,500 pairs, so it's a lead on one encoder only.

**3. Polarity score.** It works on the fresh set (0.562 AP on OpenAI and 0.608 on mpnet, against a
0.322 floor), but much of that is a property of one statement: shuffled partners still get 0.419 and
0.458. Using pairs with no negation word on v7, it scores 0.174 and 0.155 against a 0.147 floor, so
there's little left beyond spotting negation. As a gate it found 0.312 and 0.278, below cosine.

## What this changes

- v7's result stands as frozen, but should be read with the split above.
- The cheapest real option is still `negated_cosine`, at about twice the embedding cost of cosine.
  None of the three ideas replaces it.
- The open question is now the 62% with no negation word. Negation edits may not be the right tool
  for those. The queued antonym idea is the direct test.

## Limits

- Negation word means the repo's fixed word list, so `unable`, `fails to` and `refused` count as none.
- Groups follow the words in the original statements, and 62 pairs is a small group.
- WANLI's contradictions lean on word edits, so these shares may not hold on other text.
- Found counts need NLI to confirm a pair, and NLI itself may treat negation words differently.
- No novelty is claimed.
