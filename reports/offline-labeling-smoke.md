# Human-labeling and benchmark-import engineering smoke test

**Date:** November 5, 2026  
**Status:** Offline tooling tested. **No real human annotations or external
gold labels collected.**

## Verified commands

~~~bash
python -m unittest discover -s tests -q
python -m norm_k.cli generate --per-label 600 --seed 42
python -m norm_k.cli prepare-review --pairs data/generated/pairs.jsonl --per-label 20
python -m norm_k.cli annotate --queue data/review/queue.jsonl --annotations data/annotations/reviews.jsonl
# After at least two reviewers work through the queue:
python -m norm_k.cli export-reviews --queue data/review/queue.jsonl --annotations data/annotations/reviews.jsonl
~~~

Open `http://127.0.0.1:8765` **on the computer running the server**. It is
intentionally localhost-only. This workspace does not publish the UI to the
user's web browser.

## Engineering checks

- **50 unittest cases passed** across the geometry, splitting, caching,
  baseline, K ablation, import, annotation, and HTTP API modules.
- Prepared balanced blind review queues without exposing reference labels to
  the browser; found and fixed synthetic class labels embedded in original IDs.
- The HTTP API rejects requests with an untrusted Host header and refuses
  review writes without a per-process token. The server binds to loopback.
- Review writes append new immutable JSONL records; a second review by the
  same reviewer supersedes only in the computed latest-review view.
- Consensus requires **two distinct reviewers**. Disagreements and ambiguous
  votes remain unresolved and are not coerced into a training label.
- Pairwise Cohen's kappa and observed agreement are exported; kappa remains
  undefined when only one class occurs or overlap is insufficient.
- CSV, TSV, and JSONL importers retain source provenance, raw label strings,
  and a SHA-256 fingerprint. No NLI entailment/neutral mapping is inferred.
- Queue imports and exports refuse to overwrite existing files by default.
- Annotation, review queue, external data, and generated output directories
  are git-ignored to prevent accidental publishing of reviewer/private data.

## Deliberately *not* done

- No external paper dataset was downloaded or incorporated; data licensing and
  original benchmark semantics require independent verification.
- No human adjudication occurred, and no consensus scores reflect real people.
- The HTML layout was not visually inspected in an interactive end-user
  browser; the localhost API and HTML serving paths were tested.
- No OpenAI embedding API calls, local NLI inference, or production Menhir
  database access occurred.

See [annotation-protocol.md](../docs/annotation-protocol.md) for the labeling
standard and [BLOCKERS.md](../BLOCKERS.md) for outstanding authority and data
dependencies.
