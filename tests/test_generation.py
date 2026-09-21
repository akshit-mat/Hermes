import pytest
import torch
import math
from typing import Any
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from hermes.inference.generate import generate
from hermes.model.model import HermesModel

class DummyTokenizer:
    def __init__(self):
        self.special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    def encode(self, text):
        class Encoded:
            ids = [4, 5, 6] if text else []
        return Encoded()
    def token_to_id(self, token):
        return self.special_tokens.get(token, -1)
    def decode(self, ids):
        return " ".join(str(i) for i in ids)

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10, context_length=10):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.fixed_logits = None
        self.call_count = 0

    def forward(self, input_ids):
        self.call_count += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        if self.fixed_logits is not None:
            # Replicate fixed logits across all positions
            logits[:, :, :] = self.fixed_logits.unsqueeze(0).unsqueeze(0)
        else:
            # Default mock behavior: predict 7, 8, 9, EOS
            preds = [7, 8, 9, 3]
            idx = (self.call_count - 1) % len(preds)
            logits[:, -1, preds[idx]] = 10.0
        return logits

def test_generation_runs_without_error():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    out = generate(model, tokenizer, "hello", max_new_tokens=2, device=torch.device("cpu"))
    assert isinstance(out, str)

def test_greedy_is_deterministic():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    out1 = generate(model, tokenizer, "hello", max_new_tokens=3, do_sample=False, device=torch.device("cpu"))
    # Reset call count so it produces same sequence
    model.call_count = 0
    out2 = generate(model, tokenizer, "hello", max_new_tokens=3, do_sample=False, device=torch.device("cpu"))
    assert out1 == out2
    assert out1 == "7 8 9"

def test_max_new_tokens():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    out = generate(model, tokenizer, "hello", max_new_tokens=2, do_sample=False, device=torch.device("cpu"))
    # 2 tokens generated
    assert out == "7 8"

def test_max_new_tokens_zero():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    out = generate(model, tokenizer, "hello", max_new_tokens=0, device=torch.device("cpu"))
    assert out == ""

def test_eos_stops_immediately():
    class EOSModel(DummyModel):
        def forward(self, input_ids):
            self.call_count += 1
            logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
            logits[:, -1, 3] = 10.0 # Force EOS immediately
            return logits

    model = EOSModel()
    tokenizer = DummyTokenizer()
    out = generate(model, tokenizer, "hello", max_new_tokens=10, do_sample=False, device=torch.device("cpu"))
    assert out == ""
    assert model.call_count == 1 # Loop must break immediately

def test_temperature_distribution_changes():
    model = DummyModel(vocab_size=5)
    # Logits: [0.0, 1.0, 2.0, 0.0, 0.0]
    model.fixed_logits = torch.tensor([-100.0, 1.0, 2.0, -100.0, -100.0]) # Only indices 1 and 2 are viable
    tokenizer = DummyTokenizer()

    # We will sample 100 times for T=0.5 and T=2.0, and check distributions
    # Since we can't easily intercept the multinomial in generate() without a mock,
    # we'll measure the empirical frequency of token 2 vs token 1.

    def sample_n_times(t, n):
        counts = {1: 0, 2: 0}
        for _ in range(n):
            out = generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, temperature=t, device=torch.device("cpu"))
            if "1" in out: counts[1] += 1
            if "2" in out: counts[2] += 1
        return counts

    # With T=1.0, probs are proportional to [e^1, e^2] = [2.71, 7.38], so token 2 is ~73% likely
    # With T=0.1, probs are proportional to [e^10, e^20], token 2 is ~99.99% likely
    # With T=10.0, probs are proportional to [e^0.1, e^0.2], token 2 is ~52% likely

    counts_low_t = sample_n_times(0.1, 100)
    counts_high_t = sample_n_times(10.0, 100)

    # Lower temp should sharpen (more skewed toward 2)
    # Higher temp should flatten (closer to 50/50)
    assert counts_low_t[2] > counts_high_t[2]

def test_top_k_restricts_candidates():
    model = DummyModel(vocab_size=5)
    # Logits: [10, 9, 8, 7, 6]
    model.fixed_logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
    tokenizer = DummyTokenizer()

    # With top_k=2, only tokens 0 and 1 can ever be generated
    for _ in range(20):
        out = generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, top_k=2, device=torch.device("cpu"))
        assert out in ["0", "1"]

def test_top_k_1_behaves_like_greedy():
    model = DummyModel(vocab_size=5)
    model.fixed_logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0]) # 0 is always max
    tokenizer = DummyTokenizer()
    for _ in range(20):
        out = generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, top_k=1, device=torch.device("cpu"))
        assert out == "0"

def test_top_p_restricts_candidates():
    model = DummyModel(vocab_size=5)
    # Logits: [100.0, 100.0, 10.0, 0.0, 0.0] -> Probs: [0.5, 0.5, 0, 0, 0]
    model.fixed_logits = torch.tensor([100.0, 100.0, 10.0, 0.0, 0.0])
    tokenizer = DummyTokenizer()

    # If top_p=0.6, it will take the first token (0.5), which is < 0.6, so it must take the second token (0.5) to cross 0.6.
    # Therefore tokens 0 and 1 are both viable.
    # Wait, if we change logits to give probs: [0.6, 0.3, 0.1, 0, 0]
    # log(0.6) = -0.51, log(0.3) = -1.2, log(0.1) = -2.3
    model.fixed_logits = torch.tensor([math.log(0.6), math.log(0.3), math.log(0.1), -100.0, -100.0])

    # top_p = 0.5: Should only retain token 0 (since 0.6 > 0.5)
    for _ in range(20):
        out = generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, top_p=0.5, device=torch.device("cpu"))
        assert out == "0"

    # top_p = 0.85: Should retain tokens 0 and 1 (0.6 + 0.3 = 0.9 >= 0.85)
    # 2 should never appear
    for _ in range(20):
        out = generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, top_p=0.85, device=torch.device("cpu"))
        assert out in ["0", "1"]

def test_causal_behavior():
    # Load actual small dummy HermesModel
    model = HermesModel(vocab_size=10, context_length=20, d_model=16, n_layers=1, n_heads=2, pad_token_id=0)
    model.eval()

    # Prefix
    prefix_ids = torch.tensor([[2, 5, 6]], dtype=torch.long)
    # Full with future tokens
    full_ids = torch.tensor([[2, 5, 6, 7, 8]], dtype=torch.long)

    with torch.inference_mode():
        logits_prefix = model(prefix_ids)
        logits_full = model(full_ids)

    # Logits at positions 0, 1, 2 must be IDENTICAL for prefix and full!
    assert torch.allclose(logits_prefix[0, :3, :], logits_full[0, :3, :], atol=1e-5), "Causality leak detected!"

def test_context_length_sliding_window():
    model = DummyModel(context_length=5)
    tokenizer = DummyTokenizer()
    # Mock tokenizer to return a long sequence
    tokenizer.encode = lambda x: type('obj', (object,), {'ids': [1]*10})()

    out = generate(model, tokenizer, "long", max_new_tokens=2, do_sample=False, device=torch.device("cpu"))
    # Should run without error because it truncates to 5 tokens before passing to model
    assert model.call_count == 2

def test_dynamic_special_tokens():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    # Change EOS to 9
    tokenizer.special_tokens["<eos>"] = 9

    class EOSModel9(DummyModel):
        def forward(self, input_ids):
            self.call_count += 1
            logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
            logits[:, -1, 9] = 10.0 # Emit new EOS
            return logits

    model9 = EOSModel9()
    out = generate(model9, tokenizer, "A", max_new_tokens=5, do_sample=False, device=torch.device("cpu"))
    assert out == "" # Stopped immediately
    assert model9.call_count == 1

def test_invalid_parameters():
    model = DummyModel()
    tokenizer = DummyTokenizer()

    with pytest.raises(ValueError):
        generate(model, tokenizer, "A", max_new_tokens=-1)

    with pytest.raises(ValueError):
        generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, temperature=0.0)

    with pytest.raises(ValueError):
        generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, top_k=-5)

    with pytest.raises(ValueError):
        generate(model, tokenizer, "A", max_new_tokens=1, do_sample=True, top_p=1.5)
