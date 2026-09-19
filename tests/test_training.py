import json
import sys
import types

import torch

from hermes.data.tokenize import tokenize_split
from hermes.model.model import HermesModel
from hermes.tokenizer.tokenizer import HermesTokenizer
from hermes.training.data import TokenizedSplitDataset, create_dataloader
from hermes.training.trainer import set_deterministic_seed, train_basic, validate


def _make_tokenized_split(tmp_path, name, texts):
    raw_path = tmp_path / f"{name}_raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps({"text": text, "language": "en"}) + "\n" for text in texts),
        encoding="utf-8",
    )
    tokenizer = HermesTokenizer().train(
        texts,
        vocab_size=64,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    output_path = tmp_path / f"{name}.jsonl"
    tokenize_split(
        input_path=raw_path,
        output_path=output_path,
        tokenizer=tokenizer,
        context_length=8,
    )
    return output_path


def test_dataset_reads_only_requested_split(tmp_path):
    train_path = _make_tokenized_split(tmp_path, "train", ["hello world"])
    dataset = TokenizedSplitDataset(train_path)
    assert len(dataset) == 1
    assert set(dataset[0]) == {"input_ids", "attention_mask"}


def test_basic_training_and_validation_losses_are_finite(tmp_path):
    train_path = _make_tokenized_split(tmp_path, "train", ["hello world", "hello again"])
    validation_path = _make_tokenized_split(tmp_path, "validation", ["hello world"])
    train_loader = create_dataloader(train_path, batch_size=1, shuffle=True)
    validation_loader = create_dataloader(validation_path, batch_size=1, shuffle=False)
    model = HermesModel(
        vocab_size=64,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        pad_token_id=0,
    )

    # Clone a parameter before any training to prove weights actually change.
    param_before = next(p for p in model.parameters() if p.requires_grad).detach().clone()

    initial_validation = validate(model, validation_loader, device=torch.device("cpu"))
    history = train_basic(
        model,
        train_loader,
        validation_loader,
        learning_rate=0.001,
        weight_decay=0.0,
        max_steps=2,
        validation_interval=1,
        log_interval=1,
        device=torch.device("cpu"),
        logger=lambda _: None,
    )

    # Confirm the optimizer actually mutated the model weights.
    param_after = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    assert not torch.equal(param_before, param_after), (
        "Model parameters did not change after training — optimizer.step() had no effect."
    )

    assert torch.isfinite(torch.tensor(initial_validation))
    assert len(history["train_loss"]) == 2
    assert len(history["validation_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in history["train_loss"])
    assert all(torch.isfinite(torch.tensor(value)) for value in history["validation_loss"])


def test_training_resume_continues_from_saved_step(tmp_path):
    train_path = _make_tokenized_split(tmp_path, "train", ["hello world", "hello again"])
    validation_path = _make_tokenized_split(tmp_path, "validation", ["hello world"])
    checkpoint_path = tmp_path / "checkpoint.pt"

    def make_loaders():
        return (
            create_dataloader(train_path, batch_size=1, shuffle=False),
            create_dataloader(validation_path, batch_size=1, shuffle=False),
        )

    def make_model():
        return HermesModel(
            vocab_size=64,
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=4,
            pad_token_id=0,
        )

    set_deterministic_seed(7)
    first_model = make_model()
    first_train, first_validation = make_loaders()
    first_history = train_basic(
        first_model,
        first_train,
        first_validation,
        learning_rate=0.001,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=2,
        validation_interval=2,
        log_interval=1,
        device=torch.device("cpu"),
        logger=lambda _: None,
        checkpoint_path=str(checkpoint_path),
    )

    set_deterministic_seed(7)
    resumed_model = make_model()
    resumed_train, resumed_validation = make_loaders()
    resumed_history = train_basic(
        resumed_model,
        resumed_train,
        resumed_validation,
        learning_rate=0.001,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=4,
        validation_interval=2,
        log_interval=1,
        device=torch.device("cpu"),
        logger=lambda _: None,
        checkpoint_path=str(checkpoint_path),
        resume_from=str(checkpoint_path),
    )

    assert checkpoint_path.exists()
    assert len(first_history["train_loss"]) == 2
    assert len(resumed_history["train_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(resumed_history["train_loss"])))
    assert not torch.isclose(
        torch.tensor(first_history["train_loss"][-1]),
        torch.tensor(resumed_history["train_loss"][0]),
    )


def test_wandb_enabled_and_local_logging_paths(tmp_path, monkeypatch):
    events = []

    class FakeRun:
        def log(self, values):
            events.append(("log", values))

        def finish(self):
            events.append(("finish", None))

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: (events.append(("init", kwargs)) or FakeRun())
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger_lines = []
    train_path = _make_tokenized_split(tmp_path, "train", ["hello world"])
    validation_path = _make_tokenized_split(tmp_path, "validation", ["hello world"])
    model = HermesModel(
        vocab_size=64,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        pad_token_id=0,
    )
    train_basic(
        model,
        create_dataloader(train_path, batch_size=1, shuffle=False),
        create_dataloader(validation_path, batch_size=1, shuffle=False),
        learning_rate=0.001,
        weight_decay=0.0,
        max_steps=1,
        validation_interval=1,
        log_interval=1,
        device=torch.device("cpu"),
        logger=logger_lines.append,
        use_wandb=True,
    )

    assert any(event[0] == "init" for event in events)
    assert any(event[0] == "log" for event in events)
    assert any(event[0] == "finish" for event in events)
    assert any("effective_batch_size=1x1=1" in line for line in logger_lines)