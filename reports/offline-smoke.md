# First offline engineering smoke run

**Run date:** November 5, 2026  
**Nature of evidence:** CLI/infrastructure verification only — **NOT a semantic model benchmark**.

## Commands

~~~bash
python -m unittest discover -s tests -v
python -m norm_k.cli generate --per-label 600 --seed 42
python -m norm_k.cli evaluate --pairs data/generated/pairs.jsonl --encoder hash --output outputs/offline_baseline
python -m norm_k.cli k-probe --count 100 --encoder hash --output outputs/offline_k
~~~

## Observed results

- **27 / 27 tests passed**, including an end-to-end test that generates pairs,
  evaluates them, creates SVG charts, and computes K distributions.
- Generated **3,000 pairs**: 600 each for contradiction, equivalence, compatible
  facts, temporal updates, and unrelated facts; deterministic seed 42.
- Subject-held-out split in this run: **2,089 train / 424 validation / 487 test**.
- Template-held-out split: **1,478 train / 748 validation / 774 test**.
- Baseline report and pair scatter successfully generated.
- Trained a regularized tabular lexical/geometry baseline on held-out subjects
  and held-out templates; ran a low-rank PCA(|Δ|) probe at ranks 2–128 and
  produced a dimension-versus-recall SVG. This is a methodological test only.
- Four-point probe completed **100 squares**, 25 each for negation×subject,
  negation×value, subject×value, and time×value.
- Four-point scalar distribution SVG successfully generated.
- Each K operation group now also includes a percentile bootstrap 95% interval
  for its mean (a descriptive interval; it does not account for shared templates).
- Packaging smoke: `python -m pip wheel . --no-deps --no-build-isolation`
  succeeded and generated a `norm_k-0.1.0-py3-none-any.whl` locally.

The actual figures in this report are from the **hashing-control-v1-d256**
encoder. It hashes word unigrams and bigrams, and is *not* a trained semantic
embedding model. Consequently, even seemingly impressive classifier results
would mainly demonstrate recoverable **template and lexical cues**. They must
not be cited as contradiction-detection or semantic geometry performance.

For reference, the observed raw K means (units of hashed normalized vectors)
were approximately:

| Transformation square | Mean ‖K‖ |
| --- | ---: |
| Negation × subject | 0.5714 |
| Negation × value | 0.0875 |
| Subject × value | 0.0192 |
| Time × value | 0.0250 |

These differences are expected to be influenced by literal token positions,
bigrams, and normalization. This is not evidence for a contradiction-sensitive
semantic residual. A valid next study needs frozen *trained* embeddings, human-
audited labels, shuffled controls, and held-out relation families.

## Reproducibility

- Generated data, SQLite embedding cache, and SVG outputs are intentionally
  git-ignored. Recreate them using the commands above.
- The default embedding API miss allowance is zero; the offline run used no
  external models, API calls, personal data, or paid services.
- See [BLOCKERS.md](../BLOCKERS.md) for work not completed in this environment.
