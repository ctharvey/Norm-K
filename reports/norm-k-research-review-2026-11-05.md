# Norm-K research review

**Date:** 2026-11-05
**Reviewed:** ctharvey/Norm-K at e721369 (12 commits, 39 files, about 4k lines), all files read
**Method:** no audit type in `.agent/audit/` covers research validity, so this is a direct review. Published numbers were rerun, and new controls were run from a scratch venv (Python 3.12.10, numpy 2.5.4, scikit-learn 1.9.1). The repo was not modified.
**Evidence limit:** every experiment here uses the repo's hash encoder and synthetic data. Nothing here measures K on a trained embedding.

## Verdict

Norm-K is an honest, well-built scaffold that has not yet tested any of its hypotheses. Its caveats are accurate, and it claims no semantic result.

The main problem is experimental design. The K ablation cannot show that K adds anything, because its baseline lacks the information K is computed from. Given that information, K added nothing in any split on the hash control. The dataset also cannot test contradiction detection, because its labels come from 20 templates and its only negated sentences are all contradictions.

A real-embedding run should wait until both are fixed. Run today, it would produce a number that looks like evidence but can't be interpreted.

## Findings

### 1. The ablation does not isolate K (high, applies to any encoder)

On unit vectors, ||K||^2 is an exact function of the six pairwise cosines among A, B, T(A), T(B). The identity held to 1.8e-15 on the negation cohort. text-embedding-3-small also returns unit vectors, so the identity should carry over.

The baseline in `ablation.py` sees only A and B, while the augmented model also draws on T(A) and T(B). Any gain may therefore come from having two more embedded texts, not from the interaction residual that H2 is about.

The fair control gives the baseline those cosines plus the two T displacement norms, then adds K.

| Cohort / split | Baseline AP | + K | Baseline + six cosines | + six cosines + K | K's marginal value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negation / subject | 0.680 | 0.826 | 0.867 | 0.867 | -0.0004 |
| Negation / relation | 0.694 | 0.661 | 0.821 | 0.824 | +0.0030 |
| Subject swap / subject | 0.685 | 0.692 | 0.702 | 0.704 | +0.0020 |
| Subject swap / relation | 0.696 | 0.691 | 0.929 | 0.927 | -0.0015 |

Across 20 regenerated datasets, K's mean marginal value was +0.00002, +0.0015, -0.0035 and -0.0007 in the same row order. The SDs were 0.0018, 0.0026, 0.0167 and 0.0024. Lexical features alone also erase the headline gain: negation / subject goes from 0.9373 to 0.9373 when K is added.

The extra embeddings do carry signal here, but on a lexical encoder that is probably token overlap. This does not rule out K on real embeddings. It shows the current design can't detect the difference.

### 2. Mirror pairs make K degenerate (high, applies to any encoder)

When B is the toggled negation of A, T(A)=B and T(B)=A, so K = 2(A-B). That gives ||K|| = 2||A-B|| and k_relative = 1 for any encoder.

- This covers 124 pairs, all from `contradiction-2`. The largest k_relative on any other row is 0.735, so k_relative = 1 acts as a label flag for that template.
- Removing them cuts the published +0.146 gain to +0.045 (0.732 to 0.778). With the six-cosine control, that remainder becomes -0.003 (0.798 to 0.795).
- In practice, this case needs a string comparison, not an embedding.

### 3. The dataset cannot support the question (high)

- **Few independent units:** 3,000 pairs come from 20 templates, so name and value swaps add little semantic variety.
- **Negation equals the label:** the word "not" appears in 124 pairs, and all 124 are contradictions. There are no negated hard negatives, although the plan asks for hard negatives and an ambiguous class.
- **Template split:** its test positives come from one template. AP is 1.0 with or without K, so the published 0.0000 is a ceiling, not evidence of no effect. The report omits absolute APs, which hides this.
- **Relation split:** training positives come from one template (door open/closed) and test positives from two.
- **Interval is too narrow:** the relation CI resamples 32 subjects, but only two positive templates sit behind it. The repo does call it descriptive, but its width still suggests more precision than exists.
- **Prevalence:** positives are 20% overall and 31% in the relation test set. The real rate among Menhir candidates is unknown, and AP and budget recall both depend on it.
- **Selected cohorts:** eligibility depends on class. Subject swap drops all 600 unrelated pairs, and negation keeps only 150 of them.

### 4. The report's main lesson is not shown by its own example (medium)

`offline-ablation.md` presents the negation result as a gain that reverses on new relations. The general caution is sound, but this example doesn't demonstrate it:

- Adding K columns shuffled across rows already costs -0.023 AP (200 shuffles, 95% range -0.026 to -0.022), against the observed -0.033. So most of the drop doesn't depend on K's real values. I did not establish why adding those columns hurts.
- Mirror rows are in the relation test set but absent from its training set. Without them the change is +0.0035.
- The subject-swap gain of about +0.006 is noise. Across 20 datasets it was positive half the time and negative half the time.
- The negation signs were stable across those 20 datasets (subject always positive, relation always negative), which fits the mirror-row explanation. These runs carry the seed bug in finding 7.

### 5. K has no stated mechanism, and its definition changed (medium)

- **Two constructs:** the README and H2 define K(s) on one statement with operators N and P. `pair-level-k.md` defines K(A,B;T) on a pair. Only the second is ever evaluated against labels, and H2 doesn't say which one it means.
- **No mechanism:** nothing explains why contradiction should break commutation with T, or which direction K should move. Without a prediction, any association can be read as support.
- **Semantics of T:** negating both sides of a contradiction doesn't preserve contradiction. "City is not Austin" and "city is not Denver" are compatible, so the square has no clear logical reading.
- **Nonzero floor:** normalization alone makes K nonzero. A purely additive unigram encoder, normalized, gives mean ||K|| of 0.074 to 0.077 on the negation squares and 0.000 on subject x value. The additivity test uses unnormalized vectors, so it misses this.
- **Bigram artifact:** the published 0.5714 for negation x subject falls to 0.0743 without bigrams. The report warned about bigrams, and this measures the size of the effect.
- **Order dependence:** k_relative changes by up to 0.172 when A and B swap, because its denominator is asymmetric. ||K|| is unaffected.

### 6. The method doesn't yet fit Menhir (medium)

- **Names:** the notes say Menhir's conflict path stores name vectors, and #511's examples are names such as "Seattle apartment". Negation and subject swap aren't defined on those, so pair-level K needs full statements that this path doesn't use.
- **Coverage:** the 12-pattern grammar reaches 75% of templated pairs. Coverage on real extracted statements is unknown, and a production design needs a rule for uncovered pairs.
- **Cost:** two extra embeddings per pair per transformation, which `pair-level-k.md` already notes.
- **Baseline status:** the plan calls the #392 floor "existing", but PR #392 is open and unmerged. Its own text says the fixture pins behaviour rather than measuring accuracy.
- **Priority:** Menhir memory records owner direction on 2026-11-05 that recall comes before Counterfactual Residual Geometry and #511. That memory doesn't name Norm-K, so linking the two is an inference.

### 7. Reproducibility and engineering (mostly low)

**Bug (medium):** `dataset.py` calls `split_subject(candidate)` without the seed, so secondary subjects stay in the same split only at seed 42. Cross-split secondary names: 0 at seed 42, and 369 to 441 per dataset at seeds 100-119. `audit_pairs` checks primary subjects only, so it won't report this. The 20-dataset sweeps above include this leak, and I didn't measure its effect.

**Published table no longer matches the code.** The repo's own CLI on the same dataset gives:

| Value | Published | Rerun |
| --- | ---: | ---: |
| Negation / subject | +0.1464 | +0.1460 |
| Negation / relation | -0.0324 | -0.0333 |
| Negation / relation CI | [-0.0429, -0.0233] | [-0.0435, -0.0241] |
| Subject swap / subject | +0.0062 | +0.0071 |
| Subject swap / relation | -0.0060 | -0.0055 |

The cause isn't established. Library versions are one candidate, because dependencies are unpinned (`>=`), there is no lockfile, and reports don't record versions, despite the plan requiring it. No conclusion changes.

**Confirmed as published:** dataset fingerprint, 3,000 pairs, subject and template split counts, purge counts (0/54/60), coverage (2,261 and 2,400), all four mean ||K|| values, and 50 of 50 tests passing with none skipped.

**Other gaps:**

- `smoke.md` still says 40/40 tests while the README says 50. It may simply predate the later tests.
- The plan's checkboxes are all unticked and its layout says "not yet implemented", so it no longer tracks the repo.
- The annotation protocol mentions a "ChatGPT cloud workspace", left over from another environment.
- The review API returns stored labels for any typed reviewer ID, so independence relies on trust. All reviewers also get the same item order.

## Hypothesis status

| Hypothesis | Status |
| --- | --- |
| H0 null | Untested. No NLI baseline and no trained encoder. |
| H1 direction | Untested. One alignment mean is computed, with no control or held-out design. |
| H2 interaction | Untested on real embeddings. No marginal value on the hash control. |
| H3 projection | Smoke only. PCA is unsupervised, and the supervised variants aren't built. |
| H4 hybrid | Not started. |

The hypotheses are labelled pre-registered, but `BLOCKERS.md` correctly lists pre-registration as not done. There is no frozen protocol, and shuffled-label and random-projection controls are still missing.

## Novelty and literature

The literature map is good. 16 of 17 citations resolved to the described work. The exception is "Conceptual Orthospaces" (Springer, Jun 2026): the link redirected to a login, and the title matches a 2024 Luebeck dissertation. Treat it as unconfirmed, not wrong.

I found no paper testing this exact four-point residual, from a limited search that doesn't establish novelty. Additions the notes need:

- **Frias, arXiv:2608.10216 (add first).** Audits cosine gates on reversals. It reports that lexical overlap corrupted evaluations until a balanced 2x2 design was used, and that an NLI drop-in did no better. This is close to the design Norm-K needs. I read the abstract only.
- **Andreas 2019, arXiv:1902.07181.** Tree Reconstruction Error is a general test of whether vectors are explained by composed primitives, so it is direct prior art for an additivity residual.
- **Negation robustness work:** CANNOT / NegMPNet (arXiv:2307.13989), NevIR, ExcluIR. Useful as evaluation sets and as a stronger encoder arm.

Deng (arXiv:2609.28290), already cited, deserves more weight. It reports separately encoded vectors near chance on meaning identity, which challenges the premise behind a frozen-vector approach. The plan should treat it as a threat to test, not just background.

## What is strong

- Claims stay inside the evidence, including in machine-readable output (`scientific_conclusion_supported: false`).
- `BLOCKERS.md` separates done from not done, and it held up against the code.
- Same-row comparisons, purging, refusal to impute missing K, and refusal to give a CI with too few clusters are all good practice.
- The blind review tool and the importer's refusal to guess NLI label mappings are thoughtful.
- A negative result is treated as a valid outcome, with stop conditions written down.

## Recommended next steps

Order matters. Steps 1-3 need no paid calls.

1. **Fix the ablation.** Make baseline + six cosines (and displacement norms) the primary comparison. Add lexical features, shuffled-K and random-T controls. Report absolute AP and ceilings.
2. **Handle degeneracy.** Exclude or separately report mirror pairs, and pick a symmetric k_relative.
3. **Replace the data design.** Hand-write a few hundred matched pairs in a balanced 2x2 (lexical overlap x contradiction), including negated non-contradictions and non-negated contradictions. Have two real reviewers label them, with templates as the bootstrap unit.
4. **Freeze the protocol.** Commit one primary metric, one K definition, an expected direction, and a stop rule before any real-embedding run.
5. **Run the kill test.** Use text-embedding-3-small plus a local NLI baseline. The current text set is 11,334 strings and about 104k words, likely under a cent. No key is set in this shell, and `BLOCKERS.md` says the same of its environment.
6. **Stop or continue.** If K's lower CI bound over the six-cosine baseline doesn't clear the frozen threshold, drop K. The harness can still serve probes and #511.

Also fix the seed bug, pin dependencies, record versions in reports, and regenerate the published table.

## Review confidence

- **High:** the two algebraic findings (1, 2) and the code and data facts.
- **Medium:** the shuffled-K reading, which is one design and one pass of 200 shuffles.
- **Not assessed:** any behaviour of K on trained embeddings.

The review scripts were kept only in the session scratchpad: `nk_review.py`, `nk_review2.py`.

## Follow-up: docs/handoff.md (fc9e37f)

The update is docs only. `norm_k/`, `tests/` and `reports/` are unchanged from e721369, so the experimental findings above still apply.

**What changed in this review's view:**

- The project's question is broader than K. The handoff calls K a proposed feature and asks whether useful relation information sits in frozen vectors at all. Findings 1, 2 and 5 are about K, not that wider question.
- The two-lane structure means the belief lane continues if K fails, so a K failure doesn't end the project.
- Phase A (labels first) already matches the recommended order. Phase B is a baseline without K, so it doesn't depend on the ablation fix. The earlier line that a real-embedding run should wait applies to the K comparison, not to that baseline, though Phase B still needs a usable labeled set.
- The handoff already lists several open risks: names, whether T is well defined, and whether counterfactual embeddings cost less than a reviewer.

**What did not change:**

- Phases C and D still test K against simple vector features or with/without K. Neither requires the six-cosine baseline, so finding 1 would carry into real-embedding runs.
- Mirror pairs are not addressed.
- Section 7 repeats the published table, treats the relation result as a transfer warning, and shows 0.0000 without noting the ceiling.
- Section 6.2 lists the secondary-subject leak as fixed. That holds at seed 42 only: `dataset.py:37` still calls `split_subject` without the seed.
- Phase F refers to the shipped/current #392 filter, but PR #392 is still open and unmerged.
- The handoff gives no mechanism linking K to contradiction.

**Checked:** the repo is private, as section 10 says.

## Follow-up: first fair real-embedding run

The protocol was frozen at ce2b1e7 and its cap corrected at 17777b1, both before any result. On text-embedding-3-small, negation got PASS (+0.0598, interval +0.016 to +0.090) and subject swap got STOP. Full report: Norm-K `reports/real-fair-k-v1.md`.

**Correction to this review:** it said a fair test would likely show K adding nothing. On this set K cleared the frozen bar, so that expectation was wrong for negation. Finding 1 still stands as a design flaw, and the hash-control numbers in it are unchanged.

Exploratory checks after the result suggest K holds no new information, a flexible model doesn't need it (+0.015, interval crosses zero), and the transformed-sentence scores carry most of the signal. The gain is also fragile: one family gives about half of it, and an edit-position control puts its interval across zero. Labels remain author-assigned and unreviewed.
