# Independent labeling and benchmark intake protocol

**Scope:** Offline, human-controlled benchmark preparation. This protocol does
not make synthetic labels ground truth or replace Menhir's evidence handling.

## 1. What the human reviewer sees

The local reviewer shows only statement A, statement B, a confidence selector,
and a free-text rationale field. It deliberately hides:

- The generator's synthetic label.
- Class names embedded in original synthetic pair IDs (the review queue uses
  opaque IDs and the screen shows only pair position).
- Imported original benchmark labels (even when explicitly mapped).
- Model predictions, cosine scores, K residuals, and confidence.

The original reference labels and source metadata remain in the queue file for
later audit, not the browser response.

## 2. Label definitions

Use **the same subject, fact, effective time, and scope** when determining if
two statements contradict. Do not force every pair into the positive class.

| Label | Rule of thumb | Example |
| --- | --- | --- |
| Contradiction | Both cannot be true at the same time under a mutually exclusive attribute | At noon, door is open / at noon, door is closed |
| Equivalent | Both convey substantially the same proposition | Bought / purchased the same bicycle |
| Compatible | Both can hold at once, even when different | Owns a dog / owns a cat |
| Temporal update | The same fact changes with *explicitly different valid times* | Home city in 2024 / home city in 2026 |
| Unrelated | The pair addresses different subjects or independent facts without a meaningful same-fact comparison | Alice's age / Bob's job |
| Ambiguous | Missing scope, unclear entity identity, contested cardinality, or insufficient context | Lives in Austin / has a house in Denver |

Temporal update is not a euphemism for unexplained conflict. If neither text
contains time information, do not infer a move from the order of presentation.
Similarly, one person can own multiple objects or hold multiple compatible
roles. Note all important assumptions in the rationale field.

## 3. Review process

1. Assemble the review queue from synthetic examples and/or licensed external
   datasets, preserving source and split information.
2. Have two independent reviewers label each candidate without seeing model
   results or each other's labels.
3. Require each reviewer to use a stable reviewer ID so corrections are
   appended to, rather than overwriting, the log.
4. Export a consensus report. Two distinct reviewers must agree, and neither
   may select **ambiguous**, before a pair appears as unanimously resolved.
5. Independently adjudicate disagreements and scope-sensitive examples.
   Preserve the original opinions and adjudication rationale.
6. Seal final benchmark splits before fitting or tuning models. Keep the final
   test set unseen until model and feature choices are frozen.

The export includes pairwise observed agreement and **Cohen's kappa** on the
overlapping reviewer cohorts. Kappa may be undefined for constant labels or
too little overlap; do not silently treat undefined as excellent agreement.

## 4. Run locally

~~~bash
python -m pip install -e '.[research]'
python -m norm_k.cli generate --per-label 600 --seed 42
python -m norm_k.cli prepare-review --pairs data/generated/pairs.jsonl --per-label 20 --output data/review/queue.jsonl
python -m norm_k.cli annotate --queue data/review/queue.jsonl --annotations data/annotations/reviews.jsonl --port 8765
~~~

Open `http://127.0.0.1:8765/` **on the same computer** running the server.
It binds only to loopback, enforces a same-origin token on writes, and is not
deployed or reachable from the ChatGPT cloud workspace by default. Never
expose the server on a public network.

Once multiple reviewers have annotated the same queue:

~~~bash
python -m norm_k.cli export-reviews --queue data/review/queue.jsonl --annotations data/annotations/reviews.jsonl --output outputs/review_consensus.json
~~~

The raw `data/annotations/reviews.jsonl` is **append-only** and git-ignored.
The consensus export reports unresolved IDs and does not rewrite original
source records. If an output queue file already exists, the CLI refuses to
overwrite it. Use a new path for another run.

## 5. Import a local external benchmark

Accepts UTF-8 CSV, TSV, or JSONL. For example, given a local file with
`sentence1`, `sentence2`, and `gold_label` columns:

~~~bash
python -m norm_k.cli import-benchmark \
  --input /path/to/benchmark.tsv \
  --source external-study \
  --a-field sentence1 --b-field sentence2 \
  --label-field gold_label \
  --source-url https://example.org/study \
  --output data/imported/external-study.jsonl
~~~

This creates a review queue and a neighboring `.manifest.json` recording the
file SHA-256, column mapping, provenance, number of rows, and any unmapped raw
labels. The provided source URL and license string are **not independently
verified** by the importer.

Raw labels are **not automatically converted**. In particular:

- NLI *entailment* is directional and does not imply equivalence.
- NLI *neutral* includes numerous kinds of compatible/unrelated/underspecified
  pairs, so it cannot safely map wholesale to our compatible class.
- A benchmark's *contradiction* definition may use different temporal or
  quantifier assumptions from ours.

To map a verified subset, create an explicit JSON file such as:

~~~json
{"contradiction": "contradiction"}
~~~

Then pass `--label-map /path/to/mapping.json`. Unmapped source labels are
preserved in metadata, while the canonical reference label remains unset.

## 6. Privacy and limitations

- Review only synthetic data or properly authorized, privacy-safe inputs.
- Keep raw imported corpora, annotations, queues, and generated results outside
  Git by using the ignored data directories.
- Reviewers should not use a source's label as justification; give a reason
  from the statements' meaning and scope.
- The UI is a research convenience, not a remotely authenticated annotation
  platform. It has no multiuser access management, centralized audit service,
  or protected production deployment.
- A consensus of two reviewers is a necessary quality control, not enough to
  establish a professionally audited gold benchmark without adjudication.

## 7. Exit criterion for first audited cohort

Collect at least 20 independently double-reviewed items per class, deliberately
including hard negatives. Resolve or explicitly exclude disagreements. Record
agreement by category and do not tune models using unresolved labels.

For a publishable benchmark, the sample will need expansion, more natural
language, independent source validation, and a stricter error audit.
