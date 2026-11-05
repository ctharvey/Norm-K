# Pair-level K feature ablation — offline control run

**Run date:** November 5, 2026  
**Encoder:** deterministic unigram/bigram `hashing-control-v1-d256` — **NOT a trained semantic embedding**.  
**Dataset:** 3,000 synthetically labeled, unaudited statement pairs.  
**Scientific status:** engineering test only; **no contradictory-meaning finding**.

## What changed

The original K prototype could measure a four-corner transformation square,
but it could not produce a K feature for an arbitrary candidate pair A/B.

Now, for an explicitly validated *syntactic* transformation T applied to both
statements, compute:

\[
K(A,B;T)=E(A)+E(T(B))-E(T(A))-E(B)
\]

This tests whether the A→B displacement is approximately invariant to T.
It is not inherently a contradiction measurement. The prototype supports:

- **Negation:** controlled inversion of selected finite clauses (`is`, `was`,
  `owns`, `lives`, etc.). Unsupported grammar is skipped.
- **Subject swap:** consistent whole-token replacement of the primary subject
  in both sentences. Different-subject cases are skipped.

Each operation is assessed on its **eligible cohort only**, with identical
rows and splits for the baseline versus the baseline+K model. No arbitrary
zero-imputation/missingness signal is added.

## Verified offline coverage

| Counterfactual operation | Eligible pairs | Share of 3,000 |
| --- | ---: | ---: |
| Negation | 2,261 | 75.4% |
| Subject swap | 2,400 | 80.0% |

### Synthetic hash-control AP change from adding K features

| Operation | Subject-held-out Δ AP | Template-held-out Δ AP | Relation-held-out Δ AP |
| --- | ---: | ---: | ---: |
| Negation | +0.1464 | 0.0000 | **−0.0324** |
| Subject swap | +0.0062 | 0.0000 | **−0.0060** |

All values compare regularized linear probes on the same eligible rows.
The deliberately weak lexical hashing encoder is capable of producing
syntactic structure and template shortcuts. These are **not measurements of
contradiction understanding** and cannot be used as proof of K's predictive
value with real embeddings.

The key engineering lesson: **a positive gain under subject holdout can vanish
or reverse when entire relation families are held out.**

Subject-cluster bootstrap intervals were omitted for the subject-held-out
test because it contained only five independent subject groups. Intervals
on template/relation splits are descriptive and still depend on synthetic
labels. For negation under relation holdout, the approximate descriptive 95%
subject-cluster CI for Δ AP was **[−0.0429, −0.0233]**.

## Reproduce

~~~bash
python -m norm_k.cli generate --per-label 600 --seed 42
python -m norm_k.cli audit --pairs data/generated/pairs.jsonl
python -m norm_k.cli k-ablate --pairs data/generated/pairs.jsonl --encoder hash --operation both
~~~

Inspect generated `outputs/k-ablate/k_ablation.json` and
`outputs/k-ablate/pair_k_*.svg`.

## What this does *not* establish

- That syntactic negation is valid for arbitrary prose, modality, or scope.
- That contradiction can be determined from a K residual or scalar threshold.
- That these results will transfer to text-embedding-3-small.
- That the synthetic labels are free of ambiguity or family/template cues.
- That our two fitted models generalize outside the generated distribution.

Next scientific run: audited data, real frozen sentence embeddings, real
name-only Menhir embeddings, shuffled counterfactual controls, and an NLI
baseline, all on sealed subject/template/relation holdouts.
