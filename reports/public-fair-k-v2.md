# Fair K test v2: public-data result

**Run date:** November 5, 2026
**Protocol:** [preregistration-public-v2.md](../docs/preregistration-public-v2.md), frozen at c2d2a8b
**Encoder:** text-embedding-3-small, 1536 dimensions, 4,704 statements, one run
**Data:** 582 MultiNLI pairs, 192 contradictions, 10 genres; replication on 725 WANLI pairs, 139 contradictions
**Raw result:** [public-fair-k-v2.json](public-fair-k-v2.json)

## Frozen result: INCONCLUSIVE

| Comparison | Delta AP | 95% interval | Shuffled 97.5% | Hash | Clears bar |
| --- | ---: | ---: | ---: | ---: | --- |
| K over `full` (primary) | +0.0120 | -0.0093 to +0.0320 | +0.0067 | +0.0026 | No |
| K over `rf_full` (primary) | +0.0135 | -0.0149 to +0.0494 | +0.0121 | -0.0003 | No |
| K over `nli_full` (practical) | +0.0087 | -0.0006 to +0.0212 | +0.0136 | +0.0012 | No |
| Transformed sentences over `rf_base_lexical` | **+0.2344** | +0.1898 to +0.2767 | +0.0334 | +0.0248 | **Yes** |

K didn't reach the +0.03 minimum on either primary comparison, so there is no PASS.
STOP needed an interval entirely below +0.03. Both upper bounds are above it, the
first only barely, so the frozen rule says INCONCLUSIVE.

**K's v1 PASS did not replicate.** Its gain fell from +0.0598 on the author's pairs to
+0.0120 here, and it was about zero on WANLI.

## Baselines

| Model | AP, MultiNLI | AP, WANLI |
| --- | ---: | ---: |
| Base rate (floor) | 0.330 | 0.192 |
| Word features from statement B alone | 0.389 | not run |
| `lexical` | 0.421 | not run |
| `base` | 0.440 | not run |
| `rf_base_lexical` | 0.464 | 0.214 |
| `rf_full` | 0.697 | 0.330 |
| `rf_full_k` | 0.710 | 0.331 |
| `nli_full_k` | 0.942 | 0.689 |
| `nli` alone | **0.980** | **0.744** |

The NLI model has a home advantage on MultiNLI, because it trained on that corpus's
train split. WANLI is outside its listed training data, and it still leads every
embedding model by a wide margin. Adding embedding features to it scored below `nli`
alone on both sets.

## Replication and robustness

These are reported as frozen, and can't rescue the primary result.

| Comparison | WANLI delta | WANLI interval | 4 of 5 labels agree | Mismatched genres |
| --- | ---: | ---: | ---: | ---: |
| K over `full` | +0.0012 | -0.0164 to +0.0207 | +0.0109 | +0.0275 |
| K over `rf_full` | +0.0009 | -0.0358 to +0.0325 | -0.0040 | +0.0412 |
| K over `nli_full` | -0.0147 | -0.0414 to +0.0086 | +0.0062 | -0.0012 |
| Transformed sentences | **+0.1200** | +0.0754 to +0.1744 | +0.2421 | +0.2034 |

K's delta changes sign across these slices, so none of them is a reliable sign of an
effect. The transformed-sentence gain stayed between +0.224 and +0.248 with any one
genre removed.

## What this supports

- **K, the formula:** no useful gain shown. A gain of +0.03 isn't formally ruled out,
  but the estimate is about +0.01 and replication is about zero.
- **Transformed sentences:** comparing negated statements with the originals adds real
  signal over plain embedding and word features, on two datasets, and beat the hash
  control. This is the secondary test in the frozen protocol.
- **Practical value:** none shown. A small free local NLI model beats every embedding
  approach here, and nothing in this test justifies Menhir integration.

**Unverified explanation:** a contradiction written as the negation of its premise
would make negated B look like A, which `cos(A, T(B))` can detect. If so, part of the
gain may come from how NLI contradictions were written. This run didn't test that.

## Limits

- Only about 3% of MultiNLI qualifies: short `is` or `was` statements with a clean edit.
- Ten genres is few clusters, and genre sizes are uneven.
- The edit audit passed 94 of 100 by LLM vote, with no person reading the statements.
  Counting any single dissent as invalid would give 82.
- NLI's "contradiction" is looser than this project's same-fact definition.
- Public NLI pairs don't look like Menhir statements.
- Spend is estimated at about $0.002 from token counts, not measured.
