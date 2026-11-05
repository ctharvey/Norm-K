"""CLI for dataset creation, baseline analysis, and four-point experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from .dataset import generate_pairs, generate_quads, split_relation, split_subject, split_template
from .counterfactuals import counterfactual_square
from .embeddings import EmbeddingCache, HashingEncoder, OpenAIEncoder
from .evaluation import (baseline_report, features_for_pairs, plot_dimension_curve,
                         plot_k_distribution, plot_pair_k_scatter, plot_scatter, train_linear_probe)
from .geometry import cosine, difference, k_norm, k_relative
from .probes import low_rank_probe
from .ablation import counterfactual_features, run_k_ablation
from .audit import audit_pairs, purged_indices
from .review import (AnnotationLog, consensus_report, load_items, queue_from_pairs,
                     save_items)
from .review_server import serve_review_ui
from .importers import import_benchmark
from .schema import load_pairs, save_pairs
from .statistics import bootstrap_mean_interval


def _encoder(args: argparse.Namespace):
    if args.encoder == "hash":
        return HashingEncoder(dimensions=args.dimensions or 256)
    return OpenAIEncoder(dimensions=args.dimensions or 1536)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def _require_new_path(path: str | Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Refusing to replace existing file: {path}; choose a new output path")


def _generate(args: argparse.Namespace) -> None:
    pairs = generate_pairs(args.per_label, args.seed)
    save_pairs(args.output, pairs)
    print(f"Saved {len(pairs)} synthetically labeled pairs to {args.output} (NOT audited ground truth)")


def _audit(args: argparse.Namespace) -> None:
    pairs = load_pairs(args.pairs)
    report = audit_pairs(pairs, args.seed)
    _write_json(Path(args.output), report)
    print(f"Dataset audit written to {args.output}: {report['pair_count']} pairs, "
          f"{report['duplicate_text_pairs']} duplicate text pairs")


def _prepare_review(args: argparse.Namespace) -> None:
    _require_new_path(args.output)
    pairs = load_pairs(args.pairs)
    items = queue_from_pairs(pairs, args.per_label, args.seed)
    save_items(args.output, items)
    print(f"Prepared {len(items)} blind review items in {args.output}; source labels are not shown in the UI")


def _import_benchmark(args: argparse.Namespace) -> None:
    _require_new_path(args.output)
    manifest_file = str(Path(args.output).with_suffix(".manifest.json"))
    _require_new_path(manifest_file)
    mapping = None
    if args.label_map:
        mapping = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                for k, v in mapping.items()):
            raise ValueError("--label-map must be a JSON object with string keys and values")
    items, manifest = import_benchmark(
        args.input, source=args.source, a_field=args.a_field, b_field=args.b_field,
        label_field=args.label_field, label_map=mapping, split_field=args.split_field,
        source_url=args.source_url, license_name=args.license, max_rows=args.max_rows,
    )
    save_items(args.output, items)
    _write_json(Path(manifest_file), manifest)
    print(f"Imported {len(items)} benchmark rows into {args.output}; original label mapping was explicit only")


def _annotate(args: argparse.Namespace) -> None:
    items = load_items(args.queue)
    if not items:
        raise ValueError("Review queue is empty")
    serve_review_ui(items, args.annotations, args.port)


def _export_reviews(args: argparse.Namespace) -> None:
    _require_new_path(args.output)
    items = load_items(args.queue)
    log = AnnotationLog(args.annotations, [item.id for item in items])
    report = consensus_report(items, log.read_all(), minimum_reviewers=args.minimum_reviewers)
    _write_json(Path(args.output), report)
    print(f"Consensus export: {len(report['resolved'])} unanimous, "
          f"{len(report['disagreement'])} disputed, {len(report['ambiguous'])} ambiguous; "
          f"saved {args.output}")


def _evaluate(args: argparse.Namespace) -> None:
    pairs = load_pairs(args.pairs)
    encoder = _encoder(args)
    texts = list(dict.fromkeys(text for pair in pairs for text in (pair.text_a, pair.text_b)))
    with EmbeddingCache(args.cache) as cache:
        vectors = dict(zip(texts, cache.embed(encoder, texts, max_new=args.max_new,
                                              batch_size=args.batch_size)))
    feature_rows = features_for_pairs(pairs, vectors)
    splits = [split_subject(pair.subject, args.seed) for pair in pairs]
    template_splits = [split_template(pair.template_family, args.seed) for pair in pairs]
    relation_splits = [split_relation(pair.relation_family, args.seed) for pair in pairs]
    rank_runs, linear_runs, purge_report = {}, {}, {}
    for name, assignments in (("subject", splits), ("template", template_splits),
                              ("relation", relation_splits)):
        indices = purged_indices(pairs, assignments)
        kept_pairs = [pairs[i] for i in indices]
        kept_features = [feature_rows[i] for i in indices]
        kept_assignments = [assignments[i] for i in indices]
        rank_runs[f"{name}-held-out"] = low_rank_probe(kept_pairs, vectors,
                                                         kept_assignments, args.seed)
        linear_runs[name] = train_linear_probe(kept_pairs, kept_features,
                                               kept_assignments, args.seed)
        purge_report[name] = {"kept": len(indices), "dropped": len(pairs) - len(indices),
                              "support": dict(Counter(kept_assignments))}
    report = {
        "experiment": "phase2_baselines", "date": date.today().isoformat(),
        "encoder": encoder.name, "encoder_dimensions": encoder.dimensions,
        "synthetic_labels_not_audited": all(p.generated for p in pairs),
        "scientific_conclusion_supported": False,
        "split_counts": dict(Counter(splits)),
        "template_split_counts": dict(Counter(template_splits)),
        "relation_split_counts": dict(Counter(relation_splits)),
        "purged_split_counts": purge_report,
        "baselines": baseline_report(pairs, feature_rows),
        "subject_held_out_linear_probe": linear_runs["subject"],
        "template_held_out_linear_probe": linear_runs["template"],
        "relation_held_out_linear_probe": linear_runs["relation"],
        "low_rank_probes": rank_runs,
        "seed": args.seed,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "baseline.json", report)
    if not args.no_plots:
        plot_scatter(pairs, feature_rows, str(output / "pair_scatter.svg"))
        plot_dimension_curve(rank_runs, str(output / "dimensions.svg"))
    print(f"Baseline report written to {output / 'baseline.json'} (smoke result, not validation)")


def _k_probe(args: argparse.Namespace) -> None:
    quads = generate_quads(args.count, args.seed)
    encoder = _encoder(args)
    texts = list(dict.fromkeys(text for q in quads for text in
                               (q.base, q.first, q.second, q.both)))
    with EmbeddingCache(args.cache) as cache:
        vectors = dict(zip(texts, cache.embed(encoder, texts, max_new=args.max_new,
                                              batch_size=args.batch_size)))
    values = []
    relative = []
    negation_directions = []
    by_operation: dict[str, list[float]] = {}
    for quad in quads:
        v = [vectors[t] for t in (quad.base, quad.first, quad.second, quad.both)]
        values.append(k_norm(*v))
        relative.append(k_relative(*v))
        by_operation.setdefault(quad.operation_pair, []).append(values[-1])
        if quad.operation_pair.startswith("negation"):
            negation_directions.append(cosine(difference(v[0], v[1]), difference(v[2], v[3])))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "phase3_four_point_smoke", "date": date.today().isoformat(),
        "encoder": encoder.name, "encoder_dimensions": encoder.dimensions,
        "count": len(quads), "seed": args.seed,
        "operation_pairs": {name: {"count": len(group), "k_mean": float(np.mean(group)),
                                   "k_std": float(np.std(group)), "k_median": float(np.median(group)),
                                   "k_mean_bootstrap_ci_95": list(bootstrap_mean_interval(group, seed=args.seed))
                                   if len(group) >= 2 else None}
                            for name, group in by_operation.items()},
        "k_mean": float(np.mean(values)), "k_median": float(np.median(values)),
        "k_std": float(np.std(values)), "k_min": float(np.min(values)),
        "k_max": float(np.max(values)),
        "k_relative_mean": float(np.mean(relative)),
        "negation_direction_alignment_mean": float(np.mean(negation_directions)) if negation_directions else None,
        "scientific_conclusion_supported": False,
        "warning": "Controlled synthetic templates and no external/randomized controls; this is a pipeline smoke test.",
    }
    _write_json(output / "k_probe.json", report)
    if not args.no_plots:
        plot_k_distribution(by_operation, str(output / "k_distribution.svg"))
    print(f"K-probe written to {output / 'k_probe.json'} (smoke result, not validation)")


def _k_ablate(args: argparse.Namespace) -> None:
    all_pairs = load_pairs(args.pairs)
    if args.limit > 0:
        all_pairs = all_pairs[:args.limit]
    operations = ("negation", "subject_swap") if args.operation == "both" else (args.operation,)
    cohorts = {}
    texts = set()
    for op in operations:
        eligible = [(pair, square) for pair in all_pairs
                    if (square := counterfactual_square(pair, op)) is not None]
        cohorts[op] = eligible
        for pair, square in eligible:
            texts.update((pair.text_a, pair.text_b, square.transformed_base, square.transformed_other))
    encoder = _encoder(args)
    ordered_texts = sorted(texts)
    with EmbeddingCache(args.cache) as cache:
        vectors = dict(zip(ordered_texts, cache.embed(encoder, ordered_texts,
                                                      max_new=args.max_new, batch_size=args.batch_size)))
    results = {}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for op, examples in cohorts.items():
        if not examples:
            results[op] = {"skipped": "No supported counterfactuals"}
            continue
        pairs = [pair for pair, _ in examples]
        squares = [square for _, square in examples]
        subject_splits = [split_subject(pair.subject, args.seed) for pair in pairs]
        template_splits = [split_template(pair.template_family, args.seed) for pair in pairs]
        relation_splits = [split_relation(pair.relation_family, args.seed) for pair in pairs]
        results[op] = {
            "coverage": len(examples) / len(all_pairs),
            "eligible": len(examples),
            "subject_held_out": run_k_ablation(pairs, squares, vectors, subject_splits, args.seed),
            "template_held_out": run_k_ablation(pairs, squares, vectors, template_splits, args.seed),
            "relation_held_out": run_k_ablation(pairs, squares, vectors, relation_splits, args.seed),
        }
        if not args.no_plots:
            plot_pair_k_scatter([
                (pair.label, features["cosine"], features["k_norm"])
                for pair, square in examples
                for features in [counterfactual_features(square, vectors)]
            ], str(output / f"pair_k_{op}.svg"))
    report = {"experiment": "pair_level_k_ablation", "date": date.today().isoformat(),
              "encoder": encoder.name, "encoder_dimensions": encoder.dimensions,
              "pair_count": len(all_pairs), "operations": results,
              "synthetic_labels_not_audited": all(p.generated for p in all_pairs),
              "scientific_conclusion_supported": False,
              "warning": "Counterfactual grammar is restricted and merely syntactic; models are trained on synthetic labels."}
    _write_json(output / "k_ablation.json", report)
    print(f"Pair-level K ablation written to {output / 'k_ablation.json'} (research scaffold only)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="norm-k", description="Norm K experimental harness")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate synthetic, NOT human-audited statement pairs")
    generate.add_argument("--per-label", type=int, default=600)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", default="data/generated/pairs.jsonl")
    generate.set_defaults(func=_generate)
    audit = sub.add_parser("audit", help="Audit exact duplicates and split leakage (no model needed)")
    audit.add_argument("--pairs", default="data/generated/pairs.jsonl")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--output", default="outputs/dataset_audit.json")
    audit.set_defaults(func=_audit)
    prepare = sub.add_parser("prepare-review", help="Create a blind, balanced review queue")
    prepare.add_argument("--pairs", default="data/generated/pairs.jsonl")
    prepare.add_argument("--per-label", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--output", default="data/review/queue.jsonl")
    prepare.set_defaults(func=_prepare_review)

    importer = sub.add_parser("import-benchmark", help="Import local CSV/TSV/JSONL with explicit label mapping")
    importer.add_argument("--input", required=True)
    importer.add_argument("--source", required=True)
    importer.add_argument("--a-field", default="text_a")
    importer.add_argument("--b-field", default="text_b")
    importer.add_argument("--label-field")
    importer.add_argument("--label-map", help="Path to explicit raw-label -> Norm-K label JSON map")
    importer.add_argument("--split-field")
    importer.add_argument("--source-url")
    importer.add_argument("--license", default="UNVERIFIED")
    importer.add_argument("--max-rows", type=int, default=100000)
    importer.add_argument("--output", default="data/imported/benchmark_queue.jsonl")
    importer.set_defaults(func=_import_benchmark)

    annotate = sub.add_parser("annotate", help="Start local-only blind review UI at 127.0.0.1")
    annotate.add_argument("--queue", default="data/review/queue.jsonl")
    annotate.add_argument("--annotations", default="data/annotations/reviews.jsonl")
    annotate.add_argument("--port", type=int, default=8765)
    annotate.set_defaults(func=_annotate)

    exporter = sub.add_parser("export-reviews", help="Summarize append-only review consensus")
    exporter.add_argument("--queue", default="data/review/queue.jsonl")
    exporter.add_argument("--annotations", default="data/annotations/reviews.jsonl")
    exporter.add_argument("--minimum-reviewers", type=int, default=2)
    exporter.add_argument("--output", default="outputs/review_consensus.json")
    exporter.set_defaults(func=_export_reviews)
    for name, handler in (("evaluate", _evaluate), ("k-probe", _k_probe),
                          ("k-ablate", _k_ablate)):
        command = sub.add_parser(name)
        command.add_argument("--encoder", choices=("hash", "openai"), default="hash")
        command.add_argument("--dimensions", type=int, default=None)
        command.add_argument("--cache", default=".norm_k_cache/embeddings.sqlite3")
        command.add_argument("--max-new", type=int, default=0,
                             help="Upper bound on new paid API embeddings (default 0)")
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--no-plots", action="store_true")
        command.add_argument("--output", default=f"outputs/{name}")
        command.set_defaults(func=handler)
        if name in ("evaluate", "k-ablate"):
            command.add_argument("--pairs", default="data/generated/pairs.jsonl")
            if name == "k-ablate":
                command.add_argument("--operation", choices=("negation", "subject_swap", "both"),
                                     default="both")
                command.add_argument("--limit", type=int, default=0,
                                     help="Optional cap on input pairs (0 = all)")
        else:
            command.add_argument("--count", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
