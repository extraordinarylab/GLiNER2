from gliner2.processor import SchemaTransformer


def test_collator_does_not_append_punctuation_to_source_text():
    captured = []

    class DummyProcessor:
        def _transform_record(self, record, max_len=None):
            captured.append(record)
            return record

        def _create_fallback_record(self, text, schema):
            raise AssertionError("fallback should not be used")

        def _pad_batch(self, records):
            return records

    result = SchemaTransformer._collate_batch(
        DummyProcessor(),
        [("Paleontologists - Dong Zhiming br /", {"entities": ["location"]})],
    )

    assert captured[0]["text"] == "Paleontologists - Dong Zhiming br /"
    assert result[0]["text"] == "Paleontologists - Dong Zhiming br /"
