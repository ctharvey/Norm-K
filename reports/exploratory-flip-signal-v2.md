# Where v2's transformed-sentence gain comes from

**Status:** exploratory, not preregistered. It uses labels on the same pairs that produced
the v2 result, and it can't change that result. Treat every number as a lead to test on
fresh data. No new embeddings were made.
**Script:** [v2_corner_analysis.py](../scripts/exploratory/v2_corner_analysis.py). It
reproduced v2's AP exactly: 0.464 to 0.698 on MultiNLI, and 0.214 to 0.334 on WANLI.

## Main findings

**1. One untrained number roughly matches the trained models.** The flip score asks whether
negating one statement moves it toward the other:
`mean(cos(A, T(B)), cos(B, T(A))) - cos(A, B)`.

| Score | MultiNLI AP | WANLI AP |
| --- | ---: | ---: |
| Base rate (floor) | 0.330 | 0.192 |
| Plain cosine | 0.243 | 0.263 |
| Plain cosine, reversed | 0.448 | 0.164 |
| Flip score, hash encoder | 0.449 | 0.244 |
| **Flip score, real embeddings** | **0.709** | **0.382** |
| `rf_full`, trained | 0.697 | 0.330 |
| `nli` alone | 0.980 | 0.744 |

The flip's AP is above reversed cosine by +0.181 to +0.342 on MultiNLI, and by +0.159 to
+0.290 on WANLI. WANLI's reversed cosine is a weak comparison, and it is +0.119 above
plain cosine there, with no interval computed. Plain cosine's direction changes between
datasets, while the flip's doesn't. Its correlation with cosine is only -0.34.

**2. The gain isn't explained by negation words.** This goes against the explanation in
the v2 report.

| Pairs with no negation word on either side | Pairs | Contradictions | Gain over baseline | Flip AP | Floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| MultiNLI | 466 | 131 | +0.219 | 0.669 | 0.281 |
| WANLI | 566 | 94 | +0.103 | 0.330 | 0.166 |
| MultiNLI, hash encoder | 466 | 131 | +0.025 | 0.304 | 0.281 |
| WANLI, hash encoder | 566 | 94 | -0.015 | 0.147 | 0.166 |

Pairs where the edit removed a written negation are too few to read (26, 18 and 3 on
MultiNLI). The gain also appears at every level of word overlap: +0.204, +0.296 and
+0.178 on MultiNLI, and +0.158, +0.161 and +0.110 on WANLI.

**3. Most of the gain comes from comparing across the pair.** Each row adds features to
the same baseline. They overlap, so they don't sum.

| Added | MultiNLI gain | 95% interval | WANLI gain | Hash, MultiNLI |
| --- | ---: | ---: | ---: | ---: |
| All corners | +0.234 | +0.190 to +0.277 | +0.120 | +0.025 |
| `cos(A,T(B))`, `cos(B,T(A))` | +0.196 | +0.141 to +0.245 | +0.098 | +0.041 |
| `cos(T(A),T(B))` | +0.143 | +0.105 to +0.187 | +0.027 | +0.011 |
| How far the edit moves B | +0.064 | +0.031 to +0.095 | +0.030 | -0.024 |
| How far the edit moves A | +0.024 | +0.001 to +0.049 | +0.012 | +0.003 |

**4. Some of it is a one-statement effect.** Pairing each B with the wrong A, in five
shuffles, still left a gain of +0.035 to +0.078 (mean +0.050, AP 0.369). So about a fifth
of the gain may not need the relation between the statements at all. A contradiction
hypothesis seems to respond to negation differently by itself.

## What this doesn't show

- **That negation is the cause.** There was no placebo edit. If "is really" or "is indeed"
  produced a similar score, the signal would come from editing, not from negation's
  meaning. This is the most important open test.
- **That it generalizes.** All pairs were picked because the edit works, and all are short
  `is` or `was` statements on one encoder.
- **That the flip is a tuned-free result.** It was defined after v2's results were known.
  Its direction was chosen by reasoning, and reversed AP was reported (0.210 and 0.130).
- **That it's useful beside NLI,** which is still far ahead.

## Prior art

Searched by a background agent. Items marked checked were opened directly. The others are
its reports and need reading before being cited.

**Already known:**
- Sentence embeddings are often insensitive to negation:
  [HEROS](https://ar5iv.labs.arxiv.org/html/2306.05083), which also reports that NLI-tuned
  encoders are more sensitive.
- Negation words are a contradiction cue in NLI data:
  [Gururangan et al. 2018](https://aclanthology.org/N18-2017/).
- Four-vector analogy tests on sentences:
  [Zhu and de Melo 2020](https://aclanthology.org/2020.coling-main.300/) (checked exists;
  the negation details are the agent's report).
- Masking words and using embedding change as classifier input:
  [GPS](https://arxiv.org/abs/2508.11667) (checked), for adversarial text detection, not
  sentence pairs.
- Rule-based negation to create contradiction training data:
  [Varshney et al. 2022](https://ar5iv.labs.arxiv.org/html/2110.08438).

**Possibly new, not established:** applying the edit at test time and using the four-vector
similarities as contradiction features on a frozen encoder; and using deletion response as
features for a sentence-pair relation.

**Unresolved:** Schumann and Marx Gomez, "Unsupervised Contradiction Detection using
Sentence Transformations", ICNLP 2023, DOI 10.1109/ICNLP58431.2023.00065. Its
[repository](https://github.com/schumanng/unsupervised-contradiction-detection) (checked)
describes making "a training base" for unsupervised NLI and doesn't mention test-time use,
but nobody on this project has read the paper. Semantic Scholar returned a 429 and was not
retried. No novelty claim should be made until this paper is read.
