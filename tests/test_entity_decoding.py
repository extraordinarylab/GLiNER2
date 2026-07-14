from collections import OrderedDict

import pytest
import torch

from gliner2 import GLiNER2
from gliner2.inference.schema import Schema


def test_flat_ner_keeps_only_highest_scoring_label_for_exact_span():
    model = object.__new__(GLiNER2)
    # [structure, entity type, span width, token start]
    scores = torch.tensor([[[[0.8]], [[0.7]]]])

    result = GLiNER2._extract_entities(
        model,
        ["organization", "location"],
        scores,
        text_len=1,
        text_tokens=["Cupertino"],
        text="Cupertino",
        start_map=[0],
        end_map=[9],
        threshold=0.5,
        metadata={
            "entity_order": ["organization", "location"],
            "entity_metadata": {},
            "entity_multi_label": False,
        },
        include_confidence=True,
        include_spans=True,
    )

    assert result[0]["organization"][0]["text"] == "Cupertino"
    assert result[0]["organization"][0]["confidence"] == pytest.approx(0.8)
    assert result[0]["organization"][0]["start"] == 0
    assert result[0]["organization"][0]["end"] == 9
    assert result[0]["location"] == []


def test_multi_label_ner_preserves_labels_for_same_exact_span():
    model = object.__new__(GLiNER2)
    scores = torch.tensor([[[[0.8]], [[0.7]]]])

    result = GLiNER2._extract_entities(
        model,
        ["organization", "location"],
        scores,
        text_len=1,
        text_tokens=["Cupertino"],
        text="Cupertino",
        start_map=[0],
        end_map=[9],
        threshold=0.5,
        metadata={
            "entity_order": ["organization", "location"],
            "entity_metadata": {},
            "entity_multi_label": True,
        },
        include_confidence=True,
        include_spans=True,
    )

    assert result[0]["organization"][0]["text"] == "Cupertino"
    assert result[0]["location"][0]["text"] == "Cupertino"


def test_entity_schema_multi_label_round_trip():
    schema = Schema().entities(["organization", "location"], multi_label=False)

    restored = Schema.from_dict(schema.to_dict())

    assert restored._entity_multi_label is False


def test_entity_decoding_does_not_use_untrained_count_prediction():
    class DummyModel:
        def count_pred(self, _):
            raise AssertionError("entity decoding must not use count_pred")

        def count_embed(self, fields, count):
            assert count == 1
            return fields.unsqueeze(0)

        def _extract_entities(self, *args, **kwargs):
            return [OrderedDict(person=[])]

    results = {}
    GLiNER2._extract_span_result(
        DummyModel(),
        results=results,
        schema_name="entities",
        task_type="entities",
        embs=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        span_info={"span_rep": torch.tensor([[[1.0, 0.0]]])},
        schema_tokens=["(", "[P]", "entities", "(", "[E]", "person", ")", ")"],
        text_tokens=["July"],
        text_len=1,
        original_text="July",
        start_mapping=[0],
        end_mapping=[4],
        threshold=0.5,
        metadata={},
        cls_fields={},
        include_confidence=True,
        include_spans=True,
    )

    assert results == {"entities": [OrderedDict(person=[])]}
