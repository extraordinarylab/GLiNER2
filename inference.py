#!/usr/bin/env python
"""Run quick inference checks for a fine-tuned GLiNER2 NER checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gliner2 import GLiNER2


@dataclass(frozen=True)
class ExpectedEntity:
    label: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    schema: dict[str, str]
    expected: tuple[ExpectedEntity, ...]


@dataclass(frozen=True)
class CheckResult:
    failures: tuple[str, ...]
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


CASES = [
    Case(
        name="repeated_surface_form",
        text="July told the board that July revenue would be reviewed on July 31.",
        schema={
            "person": "Named people, including full names and nicknames.",
            "date": "Calendar dates, months, years, and relative dates.",
        },
        expected=(
            ExpectedEntity("person", "July", 0, 4),
            ExpectedEntity("date", "July 31", 59, 66),
        ),
    ),
    Case(
        name="general_news",
        text="Apple CEO Tim Cook introduced the iPhone 15 in Cupertino on September 12, 2023.",
        schema={
            "person": "Named people, including full names and nicknames.",
            "organization": "Companies, agencies, institutions, and other named organizations.",
            "location": "Named places such as cities, countries, regions, and facilities.",
            "date": "Calendar dates, months, years, and relative dates.",
            "product": "Named products, models, software, devices, or commercial items.",
        },
        expected=(
            ExpectedEntity("organization", "Apple", 0, 5),
            ExpectedEntity("person", "Tim Cook", 10, 18),
            ExpectedEntity("product", "iPhone 15", 34, 43),
            ExpectedEntity("location", "Cupertino", 47, 56),
            ExpectedEntity("date", "September 12, 2023", 60, 78),
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLiNER2 entity extraction smoke tests against a local checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="finerweb_ner/best",
        help="Path to a saved GLiNER2 checkpoint directory, e.g. finerweb_ner/best.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Entity confidence threshold. Lower this if the model is under-trained.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--case",
        choices=[case.name for case in CASES],
        help="Run only one named test case.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report incorrect predictions without returning a non-zero exit status.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help=(
            "Allow the same exact span to have multiple entity labels. "
            "By default this script uses flat NER and keeps only the "
            "highest-confidence label per exact span."
        ),
    )
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def inference_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_mentions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"text": value}]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        mentions = []
        for item in value:
            if isinstance(item, str):
                mentions.append({"text": item})
            elif isinstance(item, dict):
                mentions.append(item)
            else:
                mentions.append({"text": str(item)})
        return mentions
    return [{"text": str(value)}]


def get_entity_predictions(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    entities = result.get("entities", {})
    if isinstance(entities, list):
        merged = {}
        for item in entities:
            if isinstance(item, dict):
                merged.update(item)
        entities = merged
    if not isinstance(entities, dict):
        return {}
    return {label: normalize_mentions(value) for label, value in entities.items()}


def _entity_key(label: str, text: str, start: int, end: int) -> tuple[Any, ...]:
    return label, start, end, text.casefold()


def check_case(case: Case, predictions: dict[str, list[dict[str, Any]]]) -> CheckResult:
    failures = []
    expected = Counter(
        _entity_key(item.label, item.text, item.start, item.end)
        for item in case.expected
    )
    predicted = Counter()
    prediction_details = {}

    for label, mentions in predictions.items():
        for mention in mentions:
            text = mention.get("text")
            start = mention.get("start")
            end = mention.get("end")
            if (
                not isinstance(text, str)
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or start >= end
                or end > len(case.text)
            ):
                failures.append(
                    f"{case.name}: malformed prediction for '{label}': {mention!r}"
                )
                continue
            if case.text[start:end] != text:
                failures.append(
                    f"{case.name}: span mismatch for '{label}': text[{start}:{end}] "
                    f"is {case.text[start:end]!r}, prediction says {text!r}"
                )
                continue
            key = _entity_key(label, text, start, end)
            predicted[key] += 1
            prediction_details[key] = mention

    matched = expected & predicted
    missing = expected - predicted
    unexpected = predicted - expected

    for key, count in missing.items():
        label, start, end, _ = key
        failures.append(
            f"{case.name}: missing {count}x expected {label!r} entity "
            f"{case.text[start:end]!r} at [{start}:{end}]"
        )
    for key, count in unexpected.items():
        label, start, end, _ = key
        detail = prediction_details[key]
        confidence = detail.get("confidence")
        confidence_text = f", confidence={confidence:.4f}" if isinstance(confidence, (int, float)) else ""
        failures.append(
            f"{case.name}: unexpected {count}x {label!r} entity "
            f"{case.text[start:end]!r} at [{start}:{end}]{confidence_text}"
        )

    malformed_count = sum("malformed prediction" in failure or "span mismatch" in failure for failure in failures)
    return CheckResult(
        failures=tuple(failures),
        true_positives=matched.total(),
        false_positives=unexpected.total() + malformed_count,
        false_negatives=missing.total(),
    )


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    device = choose_device(args.device)
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print(f"Entity decoding: {'multi-label' if args.multi_label else 'flat'}")

    model = GLiNER2.from_pretrained(str(checkpoint), map_location=device)
    model.eval()

    cases = [case for case in CASES if args.case is None or case.name == args.case]

    all_failures = []
    total_tp = total_fp = total_fn = 0
    for case in cases:
        print("\n" + "=" * 80)
        print(f"CASE: {case.name}")
        print(f"TEXT: {case.text}")
        schema = model.create_schema().entities(
            case.schema,
            multi_label=args.multi_label,
        )
        with inference_context(device):
            result = model.extract(
                case.text,
                schema,
                threshold=args.threshold,
                include_confidence=True,
                include_spans=True,
            )
        predictions = get_entity_predictions(result)
        print("PREDICTIONS:")
        print(json.dumps(predictions, indent=2, ensure_ascii=False))

        check = check_case(case, predictions)
        total_tp += check.true_positives
        total_fp += check.false_positives
        total_fn += check.false_negatives
        print(
            f"METRICS: precision={check.precision:.3f} "
            f"recall={check.recall:.3f} f1={check.f1:.3f} "
            f"(TP={check.true_positives}, FP={check.false_positives}, FN={check.false_negatives})"
        )
        if check.failures:
            print("CHECKS: FAIL")
            for failure in check.failures:
                print(f"  - {failure}")
            all_failures.extend(check.failures)
        else:
            print("CHECKS: PASS")

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        f"\nOVERALL: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
        f"(TP={total_tp}, FP={total_fp}, FN={total_fn})"
    )

    if all_failures and not args.warn_only:
        print("\nInference checks failed:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1

    if all_failures:
        print("\nCompleted with failures because --warn-only was requested.")
    else:
        print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
