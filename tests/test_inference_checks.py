import importlib.util
import sys
from pathlib import Path


INFERENCE_PATH = Path(__file__).resolve().parents[1] / "inference.py"
SPEC = importlib.util.spec_from_file_location("gliner2_inference_script", INFERENCE_PATH)
inference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inference
SPEC.loader.exec_module(inference)

CASES = inference.CASES
Case = inference.Case
ExpectedEntity = inference.ExpectedEntity
check_case = inference.check_case


def _case():
    return Case(
        name="repeated",
        text="July met July on July 31.",
        schema={"person": "People", "date": "Dates"},
        expected=(
            ExpectedEntity("person", "July", 0, 4),
            ExpectedEntity("date", "July 31", 17, 24),
        ),
    )


def test_exact_predictions_pass():
    result = check_case(
        _case(),
        {
            "person": [{"text": "July", "start": 0, "end": 4}],
            "date": [{"text": "July 31", "start": 17, "end": 24}],
        },
    )

    assert result.failures == ()
    assert (result.true_positives, result.false_positives, result.false_negatives) == (2, 0, 0)


def test_unexpected_label_is_false_positive():
    result = check_case(
        _case(),
        {
            "person": [
                {"text": "July", "start": 0, "end": 4},
                {"text": "July", "start": 9, "end": 13},
            ],
            "date": [{"text": "July 31", "start": 17, "end": 24}],
        },
    )

    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert any("unexpected" in failure for failure in result.failures)


def test_missing_expected_span_is_false_negative():
    result = check_case(
        _case(),
        {"person": [{"text": "July", "start": 0, "end": 4}]},
    )

    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert any("missing" in failure for failure in result.failures)


def test_same_text_at_wrong_offset_does_not_match():
    result = check_case(
        _case(),
        {
            "person": [{"text": "July", "start": 9, "end": 13}],
            "date": [{"text": "July 31", "start": 17, "end": 24}],
        },
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_builtin_expected_offsets_match_their_text():
    for case in CASES:
        for expected in case.expected:
            assert case.text[expected.start:expected.end] == expected.text
