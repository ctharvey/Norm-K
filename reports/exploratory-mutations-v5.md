# Score mutations and gate design on v5's saved vectors

**Run date:** November 6, 2026
**Status:** exploratory. Every pair here was already seen in v5, so nothing below is a verdict.
**Raw results:** [mutations](exploratory-mutations-v5.json), [gate design](exploratory-gate-design-v5.json)
**Scripts:** [v5_mutation_scores.py](../scripts/exploratory/v5_mutation_scores.py),
[v5_gate_design_checks.py](../scripts/exploratory/v5_gate_design_checks.py)

These scripts read saved vectors and made no API calls. They reproduce v5's frozen flip AP,
matched AP and wrong-partner AP.

## No mutation beats the flip

| AP with negation | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| **`flip`** | **0.696** | **0.748** | 0.382 | **0.377** |
| `min(gA, gB)` | 0.670 | 0.732 | **0.388** | 0.375 |
| Normalized by edit size | 0.666 | 0.688 | 0.340 | 0.330 |
| `-cos(da, db)` | 0.599 | 0.634 | 0.341 | 0.335 |
| Pairwise term of the identity | 0.633 | 0.689 | 0.287 | 0.305 |

| Paired difference from `flip`, 95% interval | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| `min(gA, gB)` | -0.026 (-0.041, -0.011) | -0.016 (-0.027, -0.005) | +0.006 (-0.011, +0.022) | -0.002 (-0.019, +0.016) |
| Normalized | -0.030 (-0.052, -0.009) | -0.059 (-0.086, -0.032) | -0.042 (-0.081, -0.003) | -0.047 (-0.093, -0.002) |
| `-cos(da, db)` | -0.098 (-0.120, -0.075) | -0.114 (-0.140, -0.087) | -0.041 (-0.082, -0.000) | -0.042 (-0.086, +0.001) |

| Check | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| Delta over matched: `flip` | +0.327 | +0.166 | +0.162 | +0.081 |
| Delta over matched: `min(gA, gB)` | +0.305 | +0.212 | +0.152 | +0.097 |
| Delta over matched: `-cos(da, db)` | +0.239 | +0.241 | +0.127 | +0.136 |
| Wrong-partner delta: `flip` | +0.104 | +0.093 | +0.031 | +0.019 |
| Wrong-partner delta: `min(gA, gB)` | +0.084 | +0.079 | +0.014 | +0.013 |
| Wrong-partner delta: `-cos(da, db)` | +0.089 | +0.077 | +0.020 | +0.014 |

- **`min(gA, gB)` is the one worth keeping as a report-only row.** It ties flip on WANLI, trails
  slightly on fresh pairs, and has a smaller wrong-partner delta on all four rows, as the
  mutation note hoped. Those wrong-partner figures have no intervals.
- **Normalizing costs AP on all four rows,** and its wrong-partner delta doesn't improve.
- **`-cos(da, db)` loses on ranking,** though on mpnet it separates negation from the matched
  edit more than flip does. Its matched AP falls further than its negation AP.

## Contradictions aren't opposed, they fail to align

The note's synthetic check found 50.59% of unrelated vectors near maximal opposition. Every
negation there moves along one shared axis, so `cos(da, db)` can only be +1 or -1 and that figure
is a coin flip by construction. Real encoders look different:

| `-cos(da, db)` | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| Mean, contradictions | -0.303 | -0.092 | -0.397 | -0.319 |
| Mean, other real pairs | -0.612 | -0.495 | -0.613 | -0.527 |
| Mean, every unrelated cross pair | -0.364 | -0.167 | -0.302 | -0.143 |
| Above 0.5, contradictions | 9.1% | 11.4% | 18.0% | 15.1% |
| Above 0.5, other real pairs | 1.2% | 0.9% | 3.4% | 2.0% |
| Above 0.5, unrelated | 2.6% | 1.2% | 3.1% | 1.2% |
| Above 0.99, any group | 0 | 0 | 0 | 0 |

Means are negative in every group, so negation responses mostly point the same way. Contradictions
sit near unrelated pairs. The score mainly separates pairs that respond alike from those that
don't, which isn't the opposite-directions picture the hypothesis describes. Similar sentences may
simply respond alike, and that wasn't separated here.

## Gate design: `negated_cosine`, not cosine then flip

v5 hinted that cosine could find related pairs and flip could rank them. On v5's planted pool,
that hint depends on the cutoff:

| Recall at 1% budget, planted 1% pool | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| `negated_cosine`, one stage | 0.275 | 0.317 | 0.374 | 0.381 |
| Cosine top 5%, then flip | 0.474 | 0.540 | 0.417 | 0.345 |
| Cosine top 10%, then flip | 0.306 | 0.451 | 0.230 | 0.245 |
| Cosine top 20%, then flip | 0.209 | 0.371 | 0.165 | 0.209 |

Real pairs are 3% to 5% of that pool, so a 5% cut mostly isolates the planted pairs. Taking each
statement's five nearest partners by cosine removes the planted cut:

| Recall, each A's top 5 by cosine | Fresh, OpenAI | Fresh, mpnet | WANLI, OpenAI | WANLI, mpnet |
| --- | ---: | ---: | ---: | ---: |
| Reachable after top 5 | 0.940 | 0.822 | 1.000 | 0.986 |
| Pass 10%: cosine | 0.348 | 0.232 | 0.540 | 0.432 |
| Pass 10%: `negated_cosine` | **0.443** | **0.427** | **0.676** | **0.655** |
| Pass 10%: `flip` | 0.366 | 0.406 | 0.252 | 0.252 |
| Pass 5%: cosine | 0.135 | 0.066 | 0.338 | 0.252 |
| Pass 5%: `negated_cosine` | 0.217 | 0.238 | **0.439** | **0.432** |
| Pass 5%: `flip` | **0.251** | **0.325** | 0.194 | 0.223 |

`negated_cosine` beat cosine at every setting tried, with k of 5, 10 and 20 and shares of 5%, 10%
and 20%, apart from ties at or near full recall on WANLI. Flip wins only at the tightest budget on fresh pairs. This design also leaves room: at
the setting above, cosine reaches at most 0.540, against ceilings of 0.822 to 1.000. These sets are
easy, since the true partner is nearly always a top neighbor, so a fresh test should use a larger
set. [The draft](../docs/preregistration-negated-gate-v7-draft.md) does.

## Interval width at NaN-NLI's size

Drawing 258 pairs at 45% contradictions from v5's sets gives mean 95% interval widths of 0.13 to
0.16 for the delta, before any clustering by premise. See
[the v6 draft](../docs/preregistration-nan-nli-v6-draft.md) for what that means for its verdicts.
