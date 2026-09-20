import torch

from hermes.model.attention import CausalSelfAttention
from hermes.model.embeddings import PositionalEmbedding, TokenEmbedding
from hermes.model.transformer import FeedForward, TransformerBlock


def test_attention_shapes_and_head_dimension():
	attention = CausalSelfAttention(d_model=384, n_heads=6, context_length=320)
	inputs = torch.randn(2, 12, 384)
	queries = attention.split_heads(attention.query(inputs))
	assert attention.head_dim == 64
	assert queries.shape == (2, 6, 12, 64)
	assert attention(inputs).shape == inputs.shape


def test_causal_mask_blocks_future_tokens():
	torch.manual_seed(0)
	attention = CausalSelfAttention(d_model=16, n_heads=4, context_length=8)
	attention.eval()
	original = torch.randn(1, 5, 16)
	changed_future = original.clone()
	changed_future[:, 4, :] += 1000
	original_output = attention(original)
	changed_output = attention(changed_future)
	assert torch.allclose(original_output[:, :4], changed_output[:, :4], atol=1e-6)


def test_head_split_does_not_mix_head_channels():
	attention = CausalSelfAttention(d_model=8, n_heads=2, context_length=4)
	inputs = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
	split = attention.split_heads(inputs)
	assert torch.equal(split[0, 0, 0], inputs[0, 0, :4])
	assert torch.equal(split[0, 1, 0], inputs[0, 0, 4:])
	merged = attention.merge_heads(split)
	assert torch.equal(merged, inputs)


def test_embeddings_and_block_shapes_at_locked_dimensions():
	token_embedding = TokenEmbedding(vocab_size=16000, d_model=384)
	positional_embedding = PositionalEmbedding(context_length=320, d_model=384)
	token_ids = torch.randint(0, 16000, (2, 320))
	embedded = token_embedding(token_ids)
	positioned = positional_embedding(embedded)
	block = TransformerBlock(d_model=384, n_heads=6, context_length=320)
	output = block(positioned)
	assert embedded.shape == (2, 320, 384)
	assert positioned.shape == (2, 320, 384)
	assert output.shape == positioned.shape


def test_feed_forward_shape():
	feed_forward = FeedForward(d_model=384, expansion_ratio=4)
	inputs = torch.randn(2, 10, 384)
	assert feed_forward(inputs).shape == inputs.shape
