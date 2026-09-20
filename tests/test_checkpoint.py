import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from hermes.model.model import HermesModel
from hermes.training.data import EpochSeededRandomSampler
from hermes.training.trainer import (
    set_deterministic_seed,
    train_basic,
)


class IndexedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        size: int,
        *,
        fail_after: int | None = None,
    ) -> None:
        self.size = size
        self.fail_after = fail_after
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        if (
            self.fail_after is not None
            and len(self.accessed) >= self.fail_after
        ):
            raise RuntimeError(
                "intentional test interruption"
            )

        self.accessed.append(index)

        value = index + 2

        return {
            "input_ids": torch.tensor(
                [1, value, value, value],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                4,
                dtype=torch.long,
            ),
        }


def _make_model() -> HermesModel:
    return HermesModel(
        vocab_size=32,
        context_length=4,
        d_model=8,
        n_layers=1,
        n_heads=2,
        pad_token_id=0,
    )


def _make_train_loader(
    dataset: IndexedDataset,
    *,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    sampler = EpochSeededRandomSampler(
        dataset,
        seed=seed,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        generator=loader_generator,
    )


def _train_kwargs(
    checkpoint_path,
    max_steps,
    checkpoint_interval=0,
    logger=None,
):
    return dict(
        learning_rate=0.001,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=max_steps,
        gradient_accumulation_steps=2,
        validation_interval=100,
        log_interval=1,
        device=torch.device("cpu"),
        logger=logger or (lambda _: None),
        checkpoint_path=str(checkpoint_path),
        checkpoint_interval=checkpoint_interval,
    )


def test_resume_reconstructs_exact_production_shuffle_order(
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint.pt"

    validation_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor(
                    [1, 2, 2, 2],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(
                    4,
                    dtype=torch.long,
                ),
            }
        ],
        batch_size=1,
    )

    # ---------------------------------------------------------
    # Continuous reference run.
    # ---------------------------------------------------------
    set_deterministic_seed(11)

    continuous_model = _make_model()
    continuous_dataset = IndexedDataset(size=8)
    continuous_loader = _make_train_loader(
        continuous_dataset,
        seed=123,
    )

    continuous_history = train_basic(
        continuous_model,
        continuous_loader,
        validation_loader,
        **_train_kwargs(
            tmp_path / "continuous.pt",
            max_steps=3,
            checkpoint_interval=2,
        ),
    )

    continuous_parameters = [
        parameter.detach().clone()
        for parameter in continuous_model.parameters()
    ]

    continuous_accessed = list(
        continuous_dataset.accessed
    )

    # Three optimizer steps with gradient accumulation of two
    # consume six micro-batches.
    assert len(continuous_accessed) == 6

    # ---------------------------------------------------------
    # Interrupted run.
    #
    # The failure occurs after the step-2 checkpoint has been
    # written, so exactly four micro-batches have been consumed.
    # ---------------------------------------------------------
    set_deterministic_seed(11)

    interrupted_model = _make_model()
    interrupted_dataset = IndexedDataset(
        size=8,
        fail_after=4,
    )
    interrupted_loader = _make_train_loader(
        interrupted_dataset,
        seed=123,
    )

    with pytest.raises(
        RuntimeError,
        match="intentional test interruption",
    ):
        train_basic(
            interrupted_model,
            interrupted_loader,
            validation_loader,
            **_train_kwargs(
                checkpoint_path,
                max_steps=3,
                checkpoint_interval=2,
            ),
        )

    assert interrupted_dataset.accessed == (
        continuous_accessed[:4]
    )

    # ---------------------------------------------------------
    # Inspect the checkpoint created at step 2.
    # ---------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert {
        "model",
        "optimizer",
        "scheduler",
        "step",
    }.issubset(checkpoint)

    assert {
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
    }.issubset(checkpoint)

    assert "data_state" in checkpoint

    data_state = checkpoint["data_state"]

    assert data_state["sampler_type"] == (
        "epoch_seeded_random"
    )
    assert data_state["sampler_seed"] == 123
    assert data_state["sampler_epoch"] == 0
    assert data_state["batches_consumed_in_epoch"] == 4
    assert data_state["global_micro_batches"] == 4
    assert checkpoint["step"] == 2

    # The checkpoint must contain only compact position metadata,
    # not a materialized permutation or index list.
    assert "permutation" not in checkpoint
    assert "indices" not in checkpoint

    # ---------------------------------------------------------
    # Fresh-process-style resume.
    #
    # Deliberately construct the new loader with a DIFFERENT
    # sampler seed. The checkpoint's saved sampler seed must be
    # authoritative and restore seed=123.
    # ---------------------------------------------------------
    set_deterministic_seed(999)

    resumed_model = _make_model()
    resumed_dataset = IndexedDataset(size=8)
    resumed_loader = _make_train_loader(
        resumed_dataset,
        seed=999,
    )

    resumed_history = train_basic(
        resumed_model,
        resumed_loader,
        validation_loader,
        resume_from=str(checkpoint_path),
        **_train_kwargs(
            checkpoint_path,
            max_steps=3,
        ),
    )

    # Resume must reconstruct the same complete epoch ordering.
    #
    # The first four accesses are the batches that were already
    # consumed before the checkpoint. They must be replayed so
    # the iterator reaches the saved position.
    assert resumed_dataset.accessed[:4] == (
        continuous_accessed[:4]
    )

    # The next two accesses must be exactly the same batches used
    # by optimizer step 3 in the uninterrupted run.
    assert resumed_dataset.accessed[4:6] == (
        continuous_accessed[4:6]
    )

    # No additional training batches should have been consumed.
    assert len(resumed_dataset.accessed) == 6

    assert resumed_history["train_loss"] == pytest.approx(
        [continuous_history["train_loss"][2]],
        rel=1e-6,
        abs=1e-6,
    )

    # Exact state equivalence is the strongest test:
    # same data, same model state, same optimizer state,
    # same scheduler state, and same RNG state should reproduce
    # the same final parameters.
    for expected, actual in zip(
        continuous_parameters,
        resumed_model.parameters(),
    ):
        assert torch.equal(
            expected,
            actual,
        ), (
            "Resumed parameters differ from the continuous "
            "reference run. "
            f"max parameter difference="
            f"{torch.max(torch.abs(expected - actual)).item()}"
        )


def test_sequential_resume_still_works(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"

    validation_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor(
                    [1, 2, 2, 2],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(
                    4,
                    dtype=torch.long,
                ),
            }
        ],
        batch_size=1,
    )

    set_deterministic_seed(11)

    continuous_model = _make_model()
    continuous_dataset = IndexedDataset(size=6)
    loader_generator_1 = torch.Generator()
    loader_generator_1.manual_seed(11)
    continuous_loader = DataLoader(
        continuous_dataset,
        batch_size=1,
        shuffle=False,
        generator=loader_generator_1,
    )

    continuous_history = train_basic(
        continuous_model,
        continuous_loader,
        validation_loader,
        **_train_kwargs(
            tmp_path / "continuous.pt",
            max_steps=3,
        ),
    )

    continuous_parameters = [
        parameter.detach().clone()
        for parameter in continuous_model.parameters()
    ]

    continuous_accessed = list(
        continuous_dataset.accessed
    )

    assert continuous_accessed == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

    set_deterministic_seed(11)

    interrupted_model = _make_model()
    interrupted_dataset = IndexedDataset(
        size=6,
        fail_after=4,
    )
    loader_generator_2 = torch.Generator()
    loader_generator_2.manual_seed(11)
    interrupted_loader = DataLoader(
        interrupted_dataset,
        batch_size=1,
        shuffle=False,
        generator=loader_generator_2,
    )

    with pytest.raises(
        RuntimeError,
        match="intentional test interruption",
    ):
        train_basic(
            interrupted_model,
            interrupted_loader,
            validation_loader,
            **_train_kwargs(
                checkpoint_path,
                max_steps=3,
                checkpoint_interval=2,
            ),
        )

    set_deterministic_seed(999)

    resumed_model = _make_model()
    resumed_dataset = IndexedDataset(size=6)
    loader_generator_3 = torch.Generator()
    loader_generator_3.manual_seed(11)
    resumed_loader = DataLoader(
        resumed_dataset,
        batch_size=1,
        shuffle=False,
        generator=loader_generator_3,
    )

    resumed_history = train_basic(
        resumed_model,
        resumed_loader,
        validation_loader,
        resume_from=str(checkpoint_path),
        **_train_kwargs(
            checkpoint_path,
            max_steps=3,
        ),
    )

    # The first four accesses replay the already-consumed portion
    # of the sequential stream.
    assert resumed_dataset.accessed[:4] == [
        0,
        1,
        2,
        3,
    ]

    # The final two accesses are the actual resumed training
    # batches.
    assert resumed_dataset.accessed[4:6] == [
        4,
        5,
    ]

    assert len(resumed_dataset.accessed) == 6

    assert resumed_history["train_loss"] == pytest.approx(
        [continuous_history["train_loss"][2]],
        rel=1e-6,
        abs=1e-6,
    )

    for expected, actual in zip(
        continuous_parameters,
        resumed_model.parameters(),
    ):
        assert torch.equal(
            expected,
            actual,
        )


def test_checkpoint_contains_complete_training_state(
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint.pt"

    validation_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor(
                    [1, 2, 2, 2],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(
                    4,
                    dtype=torch.long,
                ),
            }
        ],
        batch_size=1,
    )

    set_deterministic_seed(11)

    model = _make_model()
    dataset = IndexedDataset(size=8)
    loader = _make_train_loader(
        dataset,
        seed=123,
    )

    train_basic(
        model,
        loader,
        validation_loader,
        **_train_kwargs(
            checkpoint_path,
            max_steps=2,
            checkpoint_interval=2,
        ),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert {
        "model",
        "optimizer",
        "scheduler",
        "step",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "data_state",
    }.issubset(checkpoint)

    if torch.cuda.is_available():
        assert "cuda_rng_state" in checkpoint

    assert checkpoint["step"] == 2

    assert checkpoint["data_state"][
        "global_micro_batches"
    ] == 4