from collections import Counter

import pytest

from evaluate_ner import Counts, collect_schema, evaluate_predictions, gold_spans
from train import load_normalized_dataset, normalize_record


def _record():
    return {
        "input": "July met Apple in July.",
        "output": {
            "entities": {
                "person": [{"text": "July", "start": 0, "end": 4}],
                "organization": [{"text": "Apple", "start": 9, "end": 14}],
            },
            "entity_descriptions": {"person": "A named person"},
        },
    }


def test_gold_spans_use_character_offsets():
    assert gold_spans(_record()) == Counter(
        {("person", 0, 4): 1, ("organization", 9, 14): 1}
    )


def test_text_only_gold_is_rejected_by_default():
    record = {"input": "July met July.", "output": {"entities": {"person": ["July"]}}}

    with pytest.raises(ValueError, match="no character offsets"):
        gold_spans(record)


def test_normalization_preserves_negative_ner_examples():
    record = {
        "input": "interceded with the rebels and",
        "output": {"entities": {}},
        "schema": [],
    }

    assert normalize_record(record)["output"] == {"entities": {}}


def test_normalized_loader_passes_dataset_config(monkeypatch):
    calls = []

    def fake_load_dataset(name, config_name, split):
        calls.append((name, config_name, split))
        return [_record()]

    monkeypatch.setattr("train.load_dataset", fake_load_dataset)

    records = load_normalized_dataset(
        "mneb/mit-movie",
        split="test",
        config_name="simple",
    )

    assert calls == [("mneb/mit-movie", "simple", "test")]
    assert records == [_record()]


def test_collect_schema_includes_labels_without_descriptions():
    assert collect_schema([_record()]) == {
        "organization": "",
        "person": "A named person",
    }


def test_excluded_entities_are_removed_from_schema_and_gold():
    record = _record()

    assert collect_schema([record], excluded_labels={"organization"}) == {
        "person": "A named person"
    }
    assert gold_spans(record, excluded_labels={"organization"}) == Counter(
        {("person", 0, 4): 1}
    )


def test_exact_micro_and_per_label_metrics():
    text = _record()["input"]
    expected = [gold_spans(_record())]
    results = [
        {
            "entities": {
                "person": [
                    {"text": "July", "start": 0, "end": 4, "confidence": 0.9}
                ],
                "organization": [
                    {"text": "July", "start": 18, "end": 22, "confidence": 0.8}
                ],
            }
        }
    ]

    micro, per_label = evaluate_predictions([text], expected, results)

    assert micro == Counts(tp=1, fp=1, fn=1)
    assert micro.f1 == pytest.approx(0.5)
    assert per_label["person"] == Counts(tp=1, fp=0, fn=0)
    assert per_label["organization"] == Counts(tp=0, fp=1, fn=1)


def test_malformed_prediction_counts_as_false_positive_instead_of_crashing():
    text = "ends with br /"
    expected = [Counter()]
    results = [
        {
            "entities": {
                "location": [
                    {
                        "text": "br /.",
                        "start": 10,
                        "end": len(text) + 1,
                        "confidence": 0.6,
                    }
                ]
            }
        }
    ]

    micro, per_label = evaluate_predictions([text], expected, results)

    assert micro == Counts(tp=0, fp=1, fn=0)
    assert per_label["location"] == Counts(tp=0, fp=1, fn=0)
