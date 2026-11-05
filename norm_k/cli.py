"""CLI for dataset creation, baseline analysis, and four-point experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from .dataset import generate_pairs, generate_quads, split_subject
from .embeddings import EmbeddingCache, HashingEncoder, OpenAIEncoder
from .evaluation import baseline_report, features_for_pairs, plot_k_distribution, plot_scatter, train_linear_probe
from .geometry import cosine, difference, k_norm, k_relative
from .schema import load_pairs, save_pairs


def _encoder(args: argparse.Namespace):
    if args.encoder == "hash":
        return HashingEncoder(dimensions=args.dimensions or 256)
    return OpenAIEncoder(dimensions=args.dimensions or 1536)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def _generate(args: argparse.Namespace) -> None:
    pairs = generate_pairs(args.per_label, args.seed)
    save_pairs(args.output, pairs)
    print(f"Saved {len(pairs)} synthetically labeled pairs to {args.output} (NOT audited ground truth)")


def _evaluate(args: argparse.Namespace) -> None:
    pairs = load_pairs(args.pairs)
    encoder = _encoder(args)
    texts = list(dict.fromkeys(text for pair in pairs for text in (pair.text_a, pair.text_b)))
    with EmbeddingCache(args.cache) as cache:
        vectors = dict(zip(texts, cache.embed(encoder, texts, max_new=args.max_new,
                                              batch_size=args.batch_size)))
    feature_rows = features_for_pairs(pairs, vectors)
    splits = [split_subject(pair.subject, args.seed) for pair in pairs]
    report = {
        "experiment": "phase2_baselines", "date": date.today().isoformat(),
        "encoder": encoder.name, "encoder_dimensions": encoder.dimensions,
        "synthetic_labels_not_audited": all(p.generated for p in pairs),
        "scientific_conclusion_supported": False,
        "split_counts": dict(Counter(splits)),
        "baselines": baseline_report(pairs, feature_rows),
        "subject_held_out_linear_probe": train_linear_probe(pairs, feature_rows, splits, args.seed),
        "seed": args.seed,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "baseline.json", report)
    if not args.no_plots:
        plot_scatter(pairs, feature_rows, str(output / "pair_scatter.svg"))
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
                                   "k_std": float(np.std(group)), "k_median": float(np.median(group))}
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="norm-k", description="Norm K experimental harness")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate synthetic, NOT human-audited statement pairs")
    generate.add_argument("--per-label", type=int, default=600)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", default="data/generated/pairs.jsonl")
    generate.set_defaults(func=_generate)
    for name, handler in (("evaluate", _evaluate), ("k-probe", _k_probe)):
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
        if name == "evaluate":
            command.add_argument("--pairs", default="data/generated/pairs.jsonl")
        else:
            command.add_argument("--count", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
