import torch

from hermes.model.model import HermesModel


def make_model() -> HermesModel:
	return HermesModel(
		vocab_size=16000,
		context_length=320,
		d_model=384,
		n_layers=6,
		n_heads=6,
		pad_token_id=0,
	)


def test_full_model_forward_shape_and_finite_loss():
	model = make_model()
	input_ids = torch.randint(4, 16000, (2, 16))
	input_ids[:, -2:] = 0
	logits = model(input_ids)
	loss = model.compute_loss(logits, input_ids)
	assert len(model.blocks) == 6
	assert logits.shape == (2, 16, 16000)
	assert loss.ndim == 0
	assert torch.isfinite(loss)


def test_padding_positions_are_excluded_from_loss():
	model = HermesModel(
		vocab_size=32,
		context_length=8,
		d_model=16,
		n_layers=1,
		n_heads=4,
		pad_token_id=0,
	)
	logits = torch.randn(1, 4, 32)
	labels = torch.tensor([[5, 6, 0, 0]])
	loss = model.compute_loss(logits, labels)
	expected = torch.nn.functional.cross_entropy(
		logits[:, :-1].reshape(-1, 32),
		labels[:, 1:].reshape(-1),
		ignore_index=0,
	)
	assert torch.allclose(loss, expected)


def test_full_model_preserves_causal_mask():
	torch.manual_seed(0)
	model = HermesModel(
		vocab_size=32,
		context_length=8,
		d_model=16,
		n_layers=2,
		n_heads=4,
		pad_token_id=0,
	)
	model.eval()
	original = torch.randint(4, 32, (1, 5))
	changed_future = original.clone()
	changed_future[:, 4] = (changed_future[:, 4] + 7) % 32
	original_logits = model(original)
	changed_logits = model(changed_future)
	assert torch.allclose(original_logits[:, :4], changed_logits[:, :4], atol=1e-6)
