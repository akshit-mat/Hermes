import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from hermes.model.model import HermesModel
from hermes.training.trainer import set_deterministic_seed, train_basic


class IndexedDataset(Dataset[dict[str, torch.Tensor]]):
	def __init__(self, size: int) -> None:
		self.size = size

	def __len__(self) -> int:
		return self.size

	def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
		value = index + 2
		return {"input_ids": torch.tensor([1, value, value, value]), "attention_mask": torch.ones(4, dtype=torch.long)}


class RecordingLoader:
	batch_size = 1

	def __init__(self, size: int, fail_after: int | None = None) -> None:
		self.dataset = IndexedDataset(size)
		self.fail_after = fail_after
		self.iterators: list[RecordingIterator] = []

	def __iter__(self):
		iterator = RecordingIterator(self.dataset, self.fail_after)
		self.iterators.append(iterator)
		return iterator


class RecordingIterator:
	def __init__(self, dataset: IndexedDataset, fail_after: int | None = None) -> None:
		self.dataset = dataset
		self.fail_after = fail_after
		self.index = 0
		self.values: list[int] = []

	def __next__(self) -> dict[str, torch.Tensor]:
		if self.fail_after is not None and self.index >= self.fail_after:
			raise RuntimeError("intentional test interruption")
		if self.index >= len(self.dataset):
			raise StopIteration
		batch = self.dataset[self.index]
		self.values.append(self.index)
		self.index += 1
		return {key: value.unsqueeze(0) for key, value in batch.items()}


def _make_model() -> HermesModel:
	return HermesModel(
		vocab_size=32,
		context_length=4,
		d_model=8,
		n_layers=1,
		n_heads=2,
		pad_token_id=0,
	)


def _train_kwargs(checkpoint_path, max_steps, checkpoint_interval=0, logger=None):
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


def test_resume_restores_micro_batch_position_and_checkpoint_state(tmp_path):
	checkpoint_path = tmp_path / "checkpoint.pt"
	validation_loader = DataLoader(
		[{"input_ids": torch.tensor([1, 2, 2, 2]), "attention_mask": torch.ones(4, dtype=torch.long)}],
		batch_size=1,
	)

	set_deterministic_seed(11)
	continuous_model = _make_model()
	continuous_loader = RecordingLoader(size=6)
	continuous_step_two = []
	continuous_history = train_basic(
		continuous_model,
		continuous_loader,
		validation_loader,
		**_train_kwargs(
			tmp_path / "continuous.pt",
			max_steps=3,
			checkpoint_interval=2,
			logger=lambda message: continuous_step_two.extend(
				[[parameter.detach().clone() for parameter in continuous_model.parameters()]]
				if message.startswith("step=2 train_loss")
				else []
			),
		),
	)
	continuous_parameters = [parameter.detach().clone() for parameter in continuous_model.parameters()]

	set_deterministic_seed(11)
	interrupted_model = _make_model()
	interrupted_loader = RecordingLoader(size=6, fail_after=4)
	interrupted_step_two = []
	with pytest.raises(RuntimeError, match="intentional test interruption"):
		train_basic(
			interrupted_model,
			interrupted_loader,
			validation_loader,
			**_train_kwargs(
				checkpoint_path,
				max_steps=3,
				checkpoint_interval=2,
				logger=lambda message: interrupted_step_two.extend(
					[[parameter.detach().clone() for parameter in interrupted_model.parameters()]]
					if message.startswith("step=2 train_loss")
					else []
				),
			),
		)
	assert all(
		torch.equal(expected, actual)
		for expected, actual in zip(continuous_step_two[0], interrupted_step_two[0])
	)
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	assert {"model", "optimizer", "scheduler", "step"}.issubset(checkpoint)
	assert {"python_rng_state", "numpy_rng_state", "torch_rng_state"}.issubset(checkpoint)
	if torch.cuda.is_available():
		assert "cuda_rng_state" in checkpoint
	assert checkpoint["step"] == 2

	set_deterministic_seed(999)
	resumed_model = _make_model()
	resumed_loader = RecordingLoader(size=6)
	resumed_history = train_basic(
		resumed_model,
		resumed_loader,
		validation_loader,
		resume_from=str(checkpoint_path),
		**_train_kwargs(checkpoint_path, max_steps=3),
	)

	assert resumed_loader.iterators[0].values[-2:] == [4, 5]
	assert resumed_history["train_loss"] == pytest.approx(
		[continuous_history["train_loss"][2]], rel=1e-6, abs=1e-6
	)
	for expected, actual in zip(continuous_parameters, resumed_model.parameters()):
		assert torch.allclose(
			expected,
			actual,
			rtol=1e-4,
			atol=1e-4,
		), f"max parameter difference={torch.max(torch.abs(expected - actual)).item()}"
