import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


class DummyProcessor:
    def change_mode(self, is_training):
        self.is_training = is_training


class TinyResumeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.4)
        self.processor = DummyProcessor()

    def forward(self, batch):
        inputs, targets = batch
        predictions = self.linear(self.dropout(inputs))
        return {"total_loss": ((predictions - targets) ** 2).mean()}

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, checkpoint_dir):
        model = cls()
        state = torch.load(
            Path(checkpoint_dir) / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        return model


class TinyResumeTrainer(GLiNER2Trainer):
    def _setup_device(self):
        self.device = torch.device("cpu")
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.model.to(self.device)
        self.config.fp16 = False
        self.config.bf16 = False

    def _prepare_data(self, data, is_train=True):
        return data

    def _validate_training_setup(self, train_dataset, eval_dataset):
        return None

    def _create_dataloader(self, dataset, batch_size, shuffle, is_training):
        return dataset


def _seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches():
    return [
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.5]])),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[-0.5]])),
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[0.25]])),
        (torch.tensor([[2.0, -1.0]]), torch.tensor([[1.0]])),
    ]


def _config(output_dir, max_steps):
    return TrainingConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        batch_size=1,
        gradient_accumulation_steps=1,
        scheduler_type="constant",
        warmup_steps=0,
        eval_strategy="steps",
        eval_steps=1,
        logging_steps=100,
        save_total_limit=10,
        encoder_lr=1e-2,
        task_lr=1e-2,
        fp16=False,
        bf16=False,
        num_workers=0,
        validate_data=False,
    )


def test_resume_matches_uninterrupted_training(tmp_path):
    _seed_everything()
    uninterrupted = TinyResumeTrainer(
        TinyResumeModel(), _config(tmp_path / "uninterrupted", max_steps=4)
    )
    uninterrupted_result = uninterrupted.train(train_data=_batches())
    uninterrupted_state = {
        key: value.detach().clone()
        for key, value in uninterrupted._unwrap_model().state_dict().items()
    }

    _seed_everything()
    interrupted = TinyResumeTrainer(
        TinyResumeModel(), _config(tmp_path / "resumed", max_steps=1)
    )
    interrupted.train(train_data=_batches())
    checkpoint = tmp_path / "resumed" / "checkpoint-1"

    assert (checkpoint / "trainer_state.pt").exists()

    resumed = TinyResumeTrainer(
        TinyResumeModel(), _config(tmp_path / "resumed", max_steps=4)
    )
    resumed_result = resumed.train(
        train_data=_batches(), resume_from_checkpoint=checkpoint
    )

    assert uninterrupted_result["total_steps"] == 4
    assert resumed_result["total_steps"] == 4
    assert resumed.step_in_epoch == 4
    for key, expected in uninterrupted_state.items():
        torch.testing.assert_close(
            resumed._unwrap_model().state_dict()[key], expected, rtol=0, atol=0
        )


def test_latest_requires_complete_trainer_checkpoint(tmp_path):
    trainer = TinyResumeTrainer(
        TinyResumeModel(), _config(tmp_path / "run", max_steps=1)
    )

    incomplete = tmp_path / "run" / "checkpoint-999"
    incomplete.mkdir()
    complete = tmp_path / "run" / "checkpoint-1"
    complete.mkdir()
    torch.save({}, complete / "trainer_state.pt")

    assert trainer._resolve_checkpoint_path("latest") == complete


def test_close_destroys_only_owned_process_group(monkeypatch):
    trainer = object.__new__(GLiNER2Trainer)
    trainer._owns_process_group = True
    destroyed = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda: destroyed.append(True)
    )

    trainer.close()
    trainer.close()

    assert destroyed == [True]
    assert trainer._owns_process_group is False


def test_train_closes_owned_process_group_when_training_raises(monkeypatch):
    trainer = object.__new__(GLiNER2Trainer)
    closed = []

    def fail(**_kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(trainer, "_train_impl", fail)
    monkeypatch.setattr(trainer, "close", lambda: closed.append(True))

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.train()

    assert closed == [True]
