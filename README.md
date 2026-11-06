# ‖K‖ — Norm K

**Experimental geometry for contradiction detection in text embeddings.**

**Start here:** [Findings](FINDINGS.md), the short version of what worked and what didn't. Then the [full research rationale, decisions, present evidence, and phased engineering handoff](docs/handoff.md).

‖K‖ (“Norm K”) is a research project exploring whether vector arithmetic, local geometric transformations, and lightweight learned projections can distinguish contradictions from paraphrases, compatible facts, unrelated statements, and ordinary changes over time.

The name comes from the norm of a four-point interaction residual:

\[
K(s) = E(s) + E(N(P(s))) - E(N(s)) - E(P(s))
\]

\[
\|K(s)\|
\]

Here E embeds text, N negates a statement, and P applies a controlled transformation such as changing its subject. The residual measures departure from additive independence of those transformations. **A high ‖K‖ is not itself proof of contradiction.** Testing whether it supplies useful signal is one of the central research questions.

**Latest exploratory work:** [Function mutations, momentum/acceleration hypotheses and retrieval bottlenecks](docs/literature/semantic-dynamics-mutations-2026-11-06.md) (math-only sanity check). [Scored on v5's saved vectors](reports/exploratory-mutations-v5.md), no mutation beat the flip.

## Status

**Paused November 6, 2026.** Negation gives a real, repeatable signal in frozen embeddings, but it isn't a general contradiction detector. See [FINDINGS.md](FINDINGS.md), and [BLOCKERS.md](BLOCKERS.md) for every result and what would reopen this.

The code below is the offline research scaffold. Its smoke tests check the engineering and aren't evidence about meaning.

The first implementation includes a five-class synthetic dataset generator, frozen-embedding geometry functions, four-point K experiments, a capped opt-in API adapter, SQLite cache, review-budget scoring, a small linear-probe control, SVG charts, subject-, template-, and relation-held-out low-rank probes, explicit pair-level counterfactual K ablations, dataset leakage auditing, bootstrap intervals, a blind local annotation UI, provenance-preserving benchmark intake, two-reviewer consensus, and 99 passing tests.

### Run the offline smoke test

~~~bash
python -m pip install -e '.[research]'
python -m unittest discover -s tests -v
python -m norm_k.cli generate --per-label 600 --seed 42
python -m norm_k.cli audit --pairs data/generated/pairs.jsonl
python -m norm_k.cli evaluate --pairs data/generated/pairs.jsonl --encoder hash --output outputs/offline_baseline
python -m norm_k.cli k-probe --count 100 --encoder hash --output outputs/offline_k
python -m norm_k.cli k-ablate --pairs data/generated/pairs.jsonl --encoder hash --operation both
~~~

### Prepare independent human reviews

~~~bash
python -m norm_k.cli prepare-review --pairs data/generated/pairs.jsonl --per-label 20
python -m norm_k.cli annotate --queue data/review/queue.jsonl --annotations data/annotations/reviews.jsonl
# After at least two reviewers have worked:
python -m norm_k.cli export-reviews --queue data/review/queue.jsonl --annotations data/annotations/reviews.jsonl
~~~

The browser UI runs only at `http://127.0.0.1:8765` **on the same computer as the command**. Reference labels are withheld and reviews are append-only. See the [annotation protocol](docs/annotation-protocol.md) for definitions and a safe CSV/TSV/JSONL benchmark import example. Generated review data and imported corpora are git-ignored.

**The hash encoder is a lexical smoke-test control, not a trained semantic embedding.** Its high synthetic classifier scores are not evidence of contradiction-detection quality. Using a real API requires an explicit `--encoder openai --max-new N` invocation and configured credentials; the default cap is zero.

- [What could not be completed (BLOCKERS.md)](BLOCKERS.md)
- [Verified offline run (engineering only)](reports/offline-smoke.md)
- [Pair-level K math and restrictions](docs/pair-level-k.md)
- [Counterfactual ablation results (synthetic control)](reports/offline-ablation.md)
- [Blind annotation protocol and CLI](docs/annotation-protocol.md)
- [Annotation/import engineering smoke test](reports/offline-labeling-smoke.md)

This project is independent of [Menhir](https://github.com/Archolith/menhir) while the hypotheses are tested. Potential downstream integration belongs to [Menhir issue #511](https://github.com/Archolith/menhir/issues/511) (belief-shaped contradiction detection) and is contingent on evidence from reproducible experiments.

## Data and attribution

Public datasets aren't included, and most files store pair IDs. The `*-edit-audit.tsv` files in `data/benchmarks/` quote short statements from:

- [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) (Liu, Swayamdipta, Smith and Choi, 2022), CC BY 4.0.
- [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) (Williams, Nangia and Bowman, 2018), mostly under the OANC license, with other terms for some genres.

Those statements stay under their original terms.

## Objectives

1. Determine whether contradiction-sensitive information can be recovered from **frozen** embedding vectors.
2. Test difference vectors, sparsity, local projection geometry, and four-point interaction residuals.
3. Separate **same fact?**, **incompatible values?**, and **legitimate update?** instead of treating every semantic change as a contradiction.
4. Quantify generalization to unseen subjects, attributes, templates, and relation types.
5. Measure improvement at a fixed downstream review budget against cosine, token heuristics, NLI, and an LLM judge.

## Research phases

| Phase | Research work | Exit result |
| --- | --- | --- |
| 1. Laboratory | Reproducible datasets, embedding cache, frozen encoder, labeling | Audited dataset and test harness |
| 2. Baselines | Cosine, Euclidean, vector differences, sparsity, NLI, plots | Initial benchmark and geometry atlas |
| 3. ‖K‖ | Controlled transformations and second-order finite differences | Statistical test of interaction signal |
| 4. Projections | Regularized probes and low-rank contradiction-sensitive subspaces | Out-of-distribution generalization report |
| 5. Beliefs | Subject/attribute/value/scope matching and temporal updates | Read-only belief-aware prototype |
| 6. Decision | Ablation, cost/latency, shadow evaluation, integration decision | Ship/iterate/stop report |

The first **two-week gate** covers phases 1–3. Proceed to learned models only if the simple baselines leave room and the mathematical signal survives held-out tests.

## Read the specs
- [Source-verified 2026-11-06 prior-art audit and continue/pin decision](docs/literature/prior-art-flip-2026-11-06.md)


- [Phased build and test plan](docs/research-plan.md)
- [Mathematics, visualization experiments, and prior art](docs/research-notes.md)

## Scientific guardrails

- Start with Menhir's existing text-embedding-3-small encoder, frozen. Evaluate entity names **separately** from complete statements.
- Preserve a truly unseen holdout; the [#392 fixture](https://github.com/Archolith/menhir/pull/392) is a small regression/evaluation set, not training data.
- Train/test splits must hold out *templates, subjects, and relation types*, not just random rows.
- Compare each proposed feature to an ablation without that feature.
- Report uncertainty, per-category false negatives, calibration, and recall at a fixed review budget.
- Do not assume frozen embeddings encode unstated time, exclusive cardinality, or factual truth.
- A useful negative result is a valid outcome. No production integration is promised.

## Naming

**‖K‖ / Norm K** is a provisional research title, not a cleared company or product trademark.

---

*Research plan established November 5, 2026. No experimental scores in this repository should be interpreted as measured until a reproducible run is linked.*