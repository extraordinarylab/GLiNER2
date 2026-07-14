#!/usr/bin/env python
"""Evaluate a GLiNER2 checkpoint on a normalized Hugging Face NER dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from gliner2 import GLiNER2
from train import load_normalized_dataset


Span = tuple[str, int, int]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def update(self, gold: Counter[Span], predicted: Counter[Span]) -> None:
        self.tp += (gold & predicted).total()
        self.fp += (predicted - gold).total()
        self.fn += (gold - predicted).total()

    def report(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact character-span NER evaluation for GLiNER2 checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/deberta-v3-base/best",
        help="Local GLiNER2 checkpoint directory.",
    )
    parser.add_argument("--dataset", default="mneb/low-ner")
    parser.add_argument(
        "--config",
        dest="config_name",
        help=(
            "Optional Hugging Face dataset configuration name, for example "
            "'simple' or 'complex' for mneb/mit-movie."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--exclude-entities",
        nargs="+",
        default=[],
        metavar="LABEL",
        help=(
            "Entity labels to exclude from both the inference schema and metrics, "
            "for example: --exclude-entities misc event"
        ),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Evaluate only the first N NER examples (useful for a smoke test).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        help="Maximum number of word tokens passed to the model.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Allow one exact span to receive multiple entity labels.",
    )
    parser.add_argument(
        "--allow-text-only-gold",
        action="store_true",
        help=(
            "Resolve legacy string-only gold mentions from their surface text. "
            "Exact start/end gold spans are preferred."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optionally write the aggregate metrics to this JSON file.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def inference_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _validate_span(text: str, label: str, mention: dict[str, Any]) -> Span:
    value = mention.get("text")
    start = mention.get("start")
    end = mention.get("end")
    if (
        not isinstance(value, str)
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or start >= end
        or end > len(text)
    ):
        raise ValueError(f"invalid {label!r} span: {mention!r}")
    if text[start:end] != value:
        raise ValueError(
            f"{label!r} span [{start}:{end}] is {text[start:end]!r}, not {value!r}"
        )
    return label, start, end


def _resolve_text_mention(
    text: str,
    label: str,
    value: str,
    cursors: dict[tuple[str, str], int],
) -> Span:
    key = label, value
    start = text.find(value, cursors.get(key, 0))
    if start < 0:
        raise ValueError(f"cannot find {label!r} gold mention {value!r} in input")
    cursors[key] = start + len(value)
    return label, start, start + len(value)


def gold_spans(
    record: dict[str, Any],
    allow_text_only: bool = False,
    excluded_labels: set[str] | None = None,
) -> Counter[Span]:
    text = record["input"]
    entities = record.get("output", {}).get("entities", {})
    spans: Counter[Span] = Counter()
    cursors: dict[tuple[str, str], int] = {}
    excluded_labels = excluded_labels or set()

    for label, mentions in entities.items():
        if label in excluded_labels:
            continue
        if not isinstance(mentions, list):
            mentions = [mentions]
        for mention in mentions:
            if isinstance(mention, dict):
                span = _validate_span(text, label, mention)
            elif isinstance(mention, str) and allow_text_only:
                span = _resolve_text_mention(text, label, mention, cursors)
            else:
                raise ValueError(
                    f"gold mention for {label!r} has no character offsets: {mention!r}; "
                    "use --allow-text-only-gold only for legacy datasets"
                )
            spans[span] += 1
    return spans


def _entity_mapping(result: dict[str, Any]) -> dict[str, Any]:
    entities = result.get("entities", {})
    if isinstance(entities, list):
        merged = {}
        for item in entities:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return entities if isinstance(entities, dict) else {}


def predicted_spans(
    text: str,
    result: dict[str, Any],
    invalid_counts: Counter[str] | None = None,
) -> Counter[Span]:
    if invalid_counts is None:
        invalid_counts = Counter()
    spans: Counter[Span] = Counter()
    for label, mentions in _entity_mapping(result).items():
        if mentions is None:
            continue
        if not isinstance(mentions, list):
            mentions = [mentions]
        for mention in mentions:
            if not isinstance(mention, dict):
                invalid_counts[label] += 1
                continue
            try:
                span = _validate_span(text, label, mention)
            except ValueError:
                invalid_counts[label] += 1
                continue
            spans[span] += 1
    return spans


def collect_schema(
    records: Iterable[dict[str, Any]],
    excluded_labels: set[str] | None = None,
) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    excluded_labels = excluded_labels or set()
    for record in records:
        output = record.get("output", {})
        for label in output.get("entities", {}):
            if label in excluded_labels:
                continue
            descriptions.setdefault(label, "")
        for label, description in output.get("entity_descriptions", {}).items():
            if label in descriptions and description:
                descriptions[label] = description
    return dict(sorted(descriptions.items()))


def _label_counts(
    gold: Counter[Span], predicted: Counter[Span]
) -> dict[str, tuple[Counter[Span], Counter[Span]]]:
    labels = {span[0] for span in gold} | {span[0] for span in predicted}
    return {
        label: (
            Counter({span: count for span, count in gold.items() if span[0] == label}),
            Counter(
                {span: count for span, count in predicted.items() if span[0] == label}
            ),
        )
        for label in labels
    }


def evaluate_predictions(
    texts: list[str],
    gold: list[Counter[Span]],
    results: list[dict[str, Any]],
) -> tuple[Counts, dict[str, Counts]]:
    if not (len(texts) == len(gold) == len(results)):
        raise ValueError("texts, gold spans, and results must have equal lengths")

    micro = Counts()
    per_label: dict[str, Counts] = defaultdict(Counts)
    for text, expected, result in zip(texts, gold, results):
        invalid_counts: Counter[str] = Counter()
        predicted = predicted_spans(text, result, invalid_counts)
        micro.update(expected, predicted)
        micro.fp += invalid_counts.total()
        for label, (label_gold, label_predicted) in _label_counts(
            expected, predicted
        ).items():
            per_label[label].update(label_gold, label_predicted)
        for label, count in invalid_counts.items():
            per_label[label].fp += count
    return micro, dict(per_label)


def print_report(micro: Counts, per_label: dict[str, Counts]) -> None:
    print("\nOVERALL (exact label + character span)")
    print(
        f"micro precision={micro.precision:.4f} recall={micro.recall:.4f} "
        f"f1={micro.f1:.4f} (TP={micro.tp}, FP={micro.fp}, FN={micro.fn})"
    )
    macro_f1 = (
        sum(counts.f1 for counts in per_label.values()) / len(per_label)
        if per_label
        else 0.0
    )
    print(f"macro f1={macro_f1:.4f} across {len(per_label)} labels")

    print("\nPER LABEL")
    print(f"{'label':30} {'precision':>10} {'recall':>10} {'f1':>10} {'TP':>7} {'FP':>7} {'FN':>7}")
    for label, counts in sorted(per_label.items()):
        print(
            f"{label[:30]:30} {counts.precision:10.4f} {counts.recall:10.4f} "
            f"{counts.f1:10.4f} {counts.tp:7d} {counts.fp:7d} {counts.fn:7d}"
        )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")

    checkpoint = args.checkpoint # Path(args.checkpoint)
    # if not checkpoint.exists():
    #     raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    config_text = f" config {args.config_name!r}" if args.config_name else ""
    print(f"Loading {args.dataset!r}{config_text} split {args.split!r}")
    try:
        records = load_normalized_dataset(
            args.dataset,
            split=args.split,
            config_name=args.config_name,
        )
    except ValueError as error:
        if "Config name is missing" in str(error):
            raise ValueError(
                f"dataset {args.dataset!r} requires --config; "
                "use one of the configuration names listed below\n\n"
                f"{error}"
            ) from error
        raise
    records = [
        record
        for record in records
        if record.get("output", {}).get("entities") is not None
    ]
    if args.max_examples is not None:
        records = records[: args.max_examples]
    if not records:
        raise ValueError("the selected dataset contains no NER examples")

    excluded_labels = set(args.exclude_entities)
    schema_spec = collect_schema(records, excluded_labels=excluded_labels)
    if not schema_spec:
        raise ValueError(
            "no entity labels remain after applying --exclude-entities"
        )

    texts = [record["input"] for record in records]
    expected = [
        gold_spans(
            record,
            allow_text_only=args.allow_text_only_gold,
            excluded_labels=excluded_labels,
        )
        for record in records
    ]

    device = choose_device(args.device)
    print(f"Examples: {len(records)}")
    print(f"Entity labels ({len(schema_spec)}): {', '.join(schema_spec)}")
    if excluded_labels:
        print(f"Excluded entity labels: {', '.join(sorted(excluded_labels))}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print(f"Entity decoding: {'multi-label' if args.multi_label else 'flat'}")

    model = GLiNER2.from_pretrained(str(checkpoint), map_location=device)
    model.eval()
    schema = model.create_schema().entities(
        schema_spec,
        multi_label=args.multi_label,
    )

    results = []
    with torch.inference_mode(), inference_context(device):
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start : start + args.batch_size]
            results.extend(
                model.batch_extract(
                    chunk,
                    schema,
                    batch_size=args.batch_size,
                    threshold=args.threshold,
                    include_confidence=True,
                    include_spans=True,
                    max_len=args.max_len,
                )
            )
            print(f"Evaluated {min(start + len(chunk), len(texts))}/{len(texts)}", end="\r")
    print()

    micro, per_label = evaluate_predictions(texts, expected, results)
    print_report(micro, per_label)

    if args.report_json:
        macro_f1 = (
            sum(counts.f1 for counts in per_label.values()) / len(per_label)
            if per_label
            else 0.0
        )
        report = {
            "dataset": args.dataset,
            "config": args.config_name,
            "split": args.split,
            "checkpoint": str(checkpoint),
            "threshold": args.threshold,
            "examples": len(records),
            "labels": list(schema_spec),
            "excluded_entities": sorted(excluded_labels),
            "micro": micro.report(),
            "macro_f1": macro_f1,
            "per_label": {
                label: counts.report() for label, counts in sorted(per_label.items())
            },
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report: {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
