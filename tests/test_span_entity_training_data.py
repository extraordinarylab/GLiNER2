import pytest

from gliner2.processor import SchemaTransformer
from gliner2.training.data import DataLoaderFactory, InputExample, TrainingDataset, ValidationError


class FakeTokenizer:
    def __init__(self):
        self.vocab = {}

    def add_special_tokens(self, _tokens):
        return 0

    def tokenize(self, token):
        return [token]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._id(tokens)
        return [self._id(token) for token in tokens]

    def _id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


def test_repeated_entity_text_uses_offsets_for_distinct_labels():
    text = "July met July."
    schema = {
        "entities": {
            "person": [{"text": "July", "start": 0, "end": 4}],
            "month": [{"text": "July", "start": 9, "end": 13}],
        }
    }

    processor = SchemaTransformer(tokenizer=FakeTokenizer())
    batch = processor.collate_fn_inference([(text, schema)])

    assert batch.structure_labels[0] == [[1, [[[(0, 0)], [(2, 2)]]]]]


def test_valid_span_based_entities_validate():
    example = InputExample(
        text="Petronas paid 80 million dollars.",
        entities={
            "oil_company": [{"text": "Petronas", "start": 0, "end": 8}],
            "monetary_value": [{"text": "80 million dollars", "start": 14, "end": 32}],
        },
    )

    assert example.validate() == []


def test_existing_string_only_entity_format_still_validates():
    dataset = TrainingDataset([
        InputExample(text="Tim Cook works at Apple.", entities={"person": ["Tim Cook"], "company": ["Apple"]})
    ])

    assert dataset.validate() == {"valid": 1, "invalid": 0, "total": 1, "invalid_indices": [], "errors": []}


@pytest.mark.parametrize(
    "mention, expected",
    [
        ({"text": "Apple", "start": "0", "end": 5}, "non-integer start"),
        ({"text": "Apple", "start": 5, "end": 5}, "start < end"),
        ({"text": "Apple", "start": -1, "end": 5}, "outside text bounds"),
        ({"text": "Apple", "start": 0, "end": 50}, "outside text bounds"),
    ],
)
def test_invalid_span_offsets_raise_clear_errors(mention, expected):
    dataset = TrainingDataset([
        InputExample(text="Apple released iPhone.", entities={"company": [mention]})
    ])

    with pytest.raises(ValidationError, match=expected):
        dataset.validate()


def test_mismatched_span_text_raises_clear_error():
    dataset = TrainingDataset([
        InputExample(
            text="Apple released iPhone.",
            entities={"company": [{"text": "Google", "start": 0, "end": 5}]},
        )
    ])

    with pytest.raises(ValidationError, match="Entity span text mismatch"):
        dataset.validate()


def test_huggingface_schema_list_becomes_entity_descriptions():
    records = DataLoaderFactory.load(
        [
            {
                "input": "Petronas paid 80 million dollars.",
                "output": {
                    "entities": {
                        "oil_company": [{"text": "Petronas", "start": 0, "end": 8}],
                    }
                },
                "schema": [
                    {
                        "label": "oil_company",
                        "description": "Companies involved in oil and gas.",
                    }
                ],
            }
        ],
        shuffle=False,
    )

    assert records[0]["output"]["entity_descriptions"] == {
        "oil_company": "Companies involved in oil and gas."
    }


def test_relation_dict_with_span_fields_normalizes_and_validates():
    record = {
        "input": "Alice founded OpenAI.",
        "output": {
            "entities": {
                "person": [{"text": "Alice", "start": 0, "end": 5}],
                "organization": [{"text": "OpenAI", "start": 14, "end": 20}],
            },
            "relations": {
                "founded": [
                    {
                        "head": {"text": "Alice", "start": 0, "end": 5},
                        "tail": {"text": "OpenAI", "start": 14, "end": 20},
                    }
                ]
            },
        },
        "schema": [],
    }

    records = DataLoaderFactory.load([record], shuffle=False, validate=True)

    assert records[0]["output"]["relations"] == [
        {
            "founded": {
                "head": {"text": "Alice", "start": 0, "end": 5},
                "tail": {"text": "OpenAI", "start": 14, "end": 20},
            }
        }
    ]


def test_processor_handles_relation_dict_with_span_fields_without_validation():
    text = "Alice founded OpenAI."
    schema = {
        "relations": {
            "founded": [
                {
                    "head": {"text": "Alice", "start": 0, "end": 5},
                    "tail": {"text": "OpenAI", "start": 14, "end": 20},
                }
            ]
        }
    }

    processor = SchemaTransformer(tokenizer=FakeTokenizer())
    batch = processor.collate_fn_inference([(text, schema)])

    assert batch.task_types[0] == ["relations"]
    assert batch.structure_labels[0] == [[1, [[[(0, 0)], [(2, 2)]]]]]


def test_repeated_relation_arguments_use_explicit_offsets():
    text = "July met July and thanked July."
    schema = {
        "relations": {
            "thanks": [
                {
                    "head": {"text": "July", "start": 9, "end": 13},
                    "tail": {"text": "July", "start": 26, "end": 30},
                }
            ]
        }
    }

    processor = SchemaTransformer(tokenizer=FakeTokenizer())
    batch = processor.collate_fn_inference([(text, schema)])

    assert batch.structure_labels[0] == [[1, [[[(2, 2)], [(5, 5)]]]]]


def test_span_beyond_max_len_is_marked_unmatched():
    text = "Alice met Bob."
    schema = {
        "entities": {
            "person": [{"text": "Bob", "start": 10, "end": 13}],
        }
    }

    processor = SchemaTransformer(tokenizer=FakeTokenizer())
    batch = processor.collate_fn_train([(text, schema)], max_len=2)

    assert batch.structure_labels[0] == [[1, [[[(-1, -1)]]]]]


def test_text_classification_record_collates():
    text = "WHY THE FUCK IS BAYLESS ISDING"
    schema = {
        "classifications": [
            {
                "task": "sentiment",
                "labels": ["anger", "joy", "neutral"],
                "true_label": ["anger"],
            }
        ]
    }

    records = DataLoaderFactory.load([{"input": text, "output": schema, "schema": []}], shuffle=False, validate=True)
    processor = SchemaTransformer(tokenizer=FakeTokenizer())
    batch = processor.collate_fn_train([(records[0]["input"], records[0]["output"])])

    assert batch.task_types[0] == ["classifications"]
    assert sum(batch.structure_labels[0][0]) == 1


def test_fallback_record_keeps_fast_path_schema_indices():
    text = "This malformed classification should fall back."
    schema = {
        "classifications": [
            {
                "task": "topic",
                "labels": ["sport", "business"],
            }
        ]
    }

    processor = SchemaTransformer(tokenizer=FakeTokenizer())
    batch = processor.collate_fn_train([(text, schema)])

    assert batch.schema_counts == [1]
    assert len(batch.schema_special_indices[0]) == 1
    assert batch.schema_special_indices[0][0]
