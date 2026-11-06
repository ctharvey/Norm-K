# Negated-cosine gate v7: result

**Run date:** November 6, 2026
**Protocol:** [preregistration-negated-gate-v7.md](../docs/preregistration-negated-gate-v7.md), frozen at db51208
**Data:** 5,000 unused WANLI train pairs, 906 contradictions
**Raw results:** [text-embedding-3-small](negated-gate-v7-openai.json), [all-mpnet-base-v2](negated-gate-v7-mpnet.json)

## Frozen result: GATE-PASS on both encoders

| Found after NLI, same number of calls | Cosine | `negated_cosine` | Margin | 95% interval | Hash margin | Room | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **`text-embedding-3-small` (primary)** | 0.515 | **0.647** | **+0.131** | +0.107 to +0.155 | +0.028 | 0.309 | **GATE-PASS** |
| `all-mpnet-base-v2` | 0.437 | 0.595 | +0.158 | +0.127 to +0.189 | +0.028 | 0.364 | GATE-PASS |

Both encoders get the same verdict, so the frozen rule calls this general, meaning not unique to one
model. Each ranking passed 2,499 of 24,990 candidates to NLI.

| Found after NLI | `text-embedding-3-small` | `all-mpnet-base-v2` |
| --- | ---: | ---: |
| Ceiling, NLI reads every candidate | 0.825 | 0.801 |
| `negated_cosine` | **0.647** | **0.595** |
| Cosine | 0.515 | 0.437 |
| Flip | 0.270 | 0.265 |
| Random | 0.086 | 0.083 |

| Recall before NLI | Pass 5% | Pass 10% | Pass 20% |
| --- | ---: | ---: | ---: |
| `negated_cosine`, OpenAI | 0.498 | 0.728 | 0.917 |
| Cosine, OpenAI | 0.337 | 0.592 | 0.876 |
| Flip, OpenAI | 0.223 | 0.289 | 0.375 |
| `negated_cosine`, mpnet | 0.471 | 0.673 | 0.838 |
| Cosine, mpnet | 0.289 | 0.515 | 0.764 |
| Flip, mpnet | 0.222 | 0.284 | 0.380 |

Contradictions reachable after retrieval at k of 5, 10 and 20: 0.982, 0.990 and 0.994 on OpenAI, and
0.956, 0.977 and 0.982 on mpnet.

## What it means

- **At a fixed NLI budget, `negated_cosine` ranks contradictions better than cosine** on fresh data,
  and this rule was built to be able to fail. It reads about a quarter more contradictions on
  OpenAI and over a third more on mpnet.
- **Flip isn't the useful score here.** Its found rate is about half of cosine's, as the exploratory
  run suggested. Relatedness has to stay in the score.
- **A gate still costs recall.** On OpenAI it reads a tenth of the candidates and finds 0.647 of
  contradictions, against 0.825 with no gate.

## What it doesn't show

- **No simple baseline was in the rule.** `negated_cosine` may partly reward `not` in the edited
  statement matching `not` in a partner. Hash's +0.028 suggests a small effect, but a plain negation
  mismatch check hasn't been run, so the margin can't yet be credited to meaning.
- **Word overlap does a lot here.** Hash found 0.550 with cosine and 0.577 with `negated_cosine`,
  above mpnet's cosine and OpenAI's cosine. On this data, embeddings' cosine ranking did no better than
  hashed words.
- **Retrieval was easy.** The true partner is nearly always a top neighbor, even with 5,000 pairs.
- **NLI's flags aren't all contradictions.** It called 3,756 unlabeled OpenAI candidates and 2,685
  mpnet candidates contradictions. These count as negatives above. An unsystematic read of 12 found
  mostly pairs about different things, such as two unrelated newspapers, so this looks more like
  false alarms than missed labels. It also flagged 20.7% of labeled neutral pairs. A person hasn't
  reviewed any of these.
- **The gate may not be needed.** The GPU check below read all 36,294 pairs in 15 seconds. At this
  scale there's little left to save, and a gate pays only with far more pairs or no free GPU.
- Only `is` and `was` statements can be edited, and WANLI was built from MultiNLI seeds.

## Audit and process

Audit passed at 94 of 100, and all three models caught all 10 known-bad items. Those items were crude,
so this doesn't show the reviewers catch subtle scope errors. 12 real items drew a dissent, and the 6
invalid ones stay in the data, as the rule sets no exclusion. OpenRouter spend for the audit wasn't
measured.

A harness overload and four incomplete batches were handled as the protocol records. Separately,
v6 stopped at its bridge check while the owner had approved running v6 then v7. v7 went ahead because
it doesn't depend on v6. That was a judgment call and it was disclosed after the fact.

## GPU parity

The owner asked for past local-model work to be rerun on GPU so runs can move there. Frozen results
stay the record. These are parity checks only. Scripts: [gpu_parity.py](../scripts/gpu_parity.py),
[gpu_parity_compare.py](../scripts/gpu_parity_compare.py). Runs used an RTX 4070 Ti, PyTorch 2.14.1
with CUDA 13.0 and full precision, against CPU PyTorch 2.15.0.

| Rerun | Numbers compared | Largest gap | Verdicts changed |
| --- | ---: | ---: | ---: |
| v4 mpnet | 163 | 0.000005 | 0 |
| v5 mpnet | 266 | 0.000014 | 0 |
| v6 bridge mpnet | 28 | 0.000003 | 0 |
| v7 OpenAI vectors with GPU NLI | 31 | 1 pair in a count | 0 |
| v7 mpnet with GPU NLI | 31 | 2 pairs in a count | 0 |
| v7 hash with GPU NLI | 9 | 0 | 0 |

| NLI scores | Pairs | Largest gap | Mean gap | Crossed 0.5 |
| --- | ---: | ---: | ---: | ---: |
| v2 | 1,307 | 0.0041 | 0.00010 | 0 |
| v7 | 36,294 | 0.0181 | 0.00016 | 6 |

v2's AP-based verdicts weren't recomputed, so its row compares scores only.

mpnet's GPU vectors sit within about 2e-7 cosine distance of the CPU ones, at about 3,000 texts a
second. NLI ran at 700 to 2,400 pairs a second against about 7 on CPU.

**The gap isn't specific to the GPU.** On the 150 worst pairs of a 4,000-pair sample, padding alone on
CPU moved scores by up to 0.0084, the same size as CPU against GPU with no padding at 0.0079. Small
floating-point differences of that size appear whenever batches change. A pre-set 0.001 tolerance
stopped GPU scores from being used in v7's canonical run, and that tolerance was tighter than the
CPU's own repeatability. Canonical v7 ran on CPU only.

**Recommendation: use GPU for new local-model work.** Pin the device, batch size and library versions
in each frozen protocol, don't mix devices within a run, and treat NLI scores as repeatable to about
0.02, not exactly. OpenAI runs don't use it.

## Cost

289,635 OpenAI tokens for v7 and 178,063 for v6's bridge, about $0.01 together at list price.

## Next

By the frozen plan, GATE-PASS on both encoders means a cheap ranker has shown value on this kind of
text only, and the next step would be realistic text, which needs a dataset that doesn't exist here.
Before that, the negation-mismatch baseline should run, since it could explain much of this margin.
No novelty is claimed.
