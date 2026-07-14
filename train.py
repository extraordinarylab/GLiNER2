from copy import deepcopy
from typing import Any, Dict, Iterable, List

import torch
from datasets import load_dataset
from gliner2 import ExtractorConfig, GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


TRAIN_DATASETS = [
    "mneb/finerweb-english",
    # "mneb/go-emotions",
    # "mneb/bbc-news",
    "mneb/banking77",
    # "mneb/biored",
    "mneb/kbp37",
    # "mneb/gap",
]


def _schema_descriptions(schema: Any) -> Dict[str, str]:
    if not isinstance(schema, list):
        return {}

    descriptions = {}
    for item in schema:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        description = item.get("description")
        if label and description:
            descriptions[label] = description
    return descriptions


def _normalize_entities(entities: Any) -> Dict[str, List[Any]]:
    if not isinstance(entities, dict):
        return {}

    normalized = {}
    for label, mentions in entities.items():
        if mentions is None:
            continue
        if not isinstance(mentions, list):
            mentions = [mentions]
        normalized[label] = deepcopy(mentions)
    return normalized


def _normalize_relation_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(fields)


def _normalize_relations(relations: Any) -> List[Dict[str, Dict[str, Any]]]:
    if not relations:
        return []

    if isinstance(relations, dict):
        normalized = []
        for relation_name, occurrences in relations.items():
            if not isinstance(occurrences, list):
                occurrences = [occurrences]
            for fields in occurrences:
                if isinstance(fields, dict):
                    normalized.append({relation_name: _normalize_relation_fields(fields)})
        return normalized

    if isinstance(relations, list):
        normalized = []
        for item in relations:
            if not isinstance(item, dict):
                continue
            for relation_name, fields in item.items():
                if isinstance(fields, dict):
                    normalized.append({relation_name: _normalize_relation_fields(fields)})
        return normalized

    return []


def _normalize_classifications(classifications: Any) -> List[Dict[str, Any]]:
    if not isinstance(classifications, list):
        return []

    normalized = []
    for classification in classifications:
        if not isinstance(classification, dict):
            continue

        item = deepcopy(classification)
        true_label = item.get("true_label", [])
        if isinstance(true_label, str):
            true_label = [true_label]
        item["true_label"] = list(true_label)

        if len(item["true_label"]) > 1:
            item["multi_label"] = True
        normalized.append(item)
    return normalized


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "input": record.get("input", record.get("text", "")),
        "output": {},
    }

    output = deepcopy(record.get("output", record.get("schema", {})))
    if not isinstance(output, dict):
        output = {}

    entities = _normalize_entities(output.get("entities"))
    # Preserve explicit empty entity mappings so normalization does not erase
    # negative NER records (they are required for false-positive evaluation).
    if "entities" in output:
        normalized["output"]["entities"] = entities

    descriptions = output.get("entity_descriptions") or _schema_descriptions(record.get("schema"))
    if descriptions:
        normalized["output"]["entity_descriptions"] = descriptions

    classifications = _normalize_classifications(output.get("classifications"))
    if classifications:
        normalized["output"]["classifications"] = classifications

    relations = _normalize_relations(output.get("relations"))
    if relations:
        normalized["output"]["relations"] = relations

    for key in ("json_structures", "json_descriptions"):
        if key in output:
            normalized["output"][key] = output[key]

    return normalized


def load_normalized_dataset(
    name: str,
    split: str,
    config_name: str | None = None,
) -> List[Dict[str, Any]]:
    dataset = load_dataset(name, config_name, split=split)
    return [normalize_record(record) for record in dataset]


def load_normalized_datasets(names: Iterable[str], split: str) -> List[Dict[str, Any]]:
    records = []
    for name in names:
        records.extend(load_normalized_dataset(name, split))
    return records


def main() -> None:
    model_name = "microsoft/deberta-v3-base"
    model_config = ExtractorConfig(
        model_name=model_name,
        max_width=8,
        counting_layer="count_lstm_v2",
        token_pooling="first",
    )
    # Some Hugging Face checkpoints are materialized as FP16 by from_pretrained().
    # Keep trainable parameters in FP32; TrainingConfig controls autocast separately.
    model = GLiNER2(model_config).float()
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(f"Expected FP32 trainable parameters, got: {parameter_dtypes}")
    # model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    train_data = load_normalized_datasets(TRAIN_DATASETS, split="train")
    eval_data = load_normalized_dataset("mneb/low-ner", split="test")

    config = TrainingConfig(
        output_dir=f"./outputs/{model_name.split('/')[-1]}",
        num_epochs=1,
        # Global batch size = batch_size * torchrun processes * gradient_accumulation_steps.
        # With torchrun --nproc_per_node=4 this is 8 * 4 * 1 = 32.
        batch_size=8,
        gradient_accumulation_steps=1,
        encoder_lr=1e-5,
        task_lr=2e-5,
        weight_decay=0.01,
        scheduler_type="linear",
        warmup_steps=200,
        # Start full precision when training newly initialized GLiNER2 heads.
        # Once loss is stable, bf16=True can be retried.
        fp16=False,
        bf16=False,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        validate_data=True,
    )
    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=train_data, eval_data=eval_data)


if __name__ == "__main__":
    main()
