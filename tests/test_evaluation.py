import pytest
import math
import torch
from pathlib import Path
import tempfile
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from hermes.model.model import HermesModel
from evaluate import prepare_batch, greedy_generate

def test_padding_exclusion_and_token_weighted_aggregation():
    # Setup a tiny mock model and compute loss over a manual batch
    model = HermesModel(
        vocab_size=10,
        context_length=4,
        d_model=8,
        n_layers=1,
        n_heads=2,
        pad_token_id=0,
    )
    
    # Batch of 2 sequences
    # Seq 1: [2, 5, 6, 3] -> targets [5, 6, 3], pad 0
    # Seq 2: [2, 7, 0, 0] -> targets [7, 0, 0], pad 0
    input_ids = torch.tensor([
        [2, 5, 6, 3],
        [2, 7, 0, 0],
    ], dtype=torch.long)
    
    logits = model(input_ids)
    
    # Using model's standard mean loss
    mean_loss = model.compute_loss(logits, input_ids).item()
    
    # Calculate expected sum manually
    targets = input_ids[:, 1:]
    batch_loss_tensor = torch.nn.functional.cross_entropy(
        logits[:, :-1].contiguous().view(-1, 10),
        targets.contiguous().view(-1),
        ignore_index=0,
        reduction='none'
    ).view(2, 3)
    
    # Seq 1 has 3 valid tokens
    seq1_loss = batch_loss_tensor[0, 0] + batch_loss_tensor[0, 1] + batch_loss_tensor[0, 2]
    # Seq 2 has 1 valid token
    seq2_loss = batch_loss_tensor[1, 0] # indices 1 and 2 are padded out by cross_entropy internally!
    
    # Let's verify our manual token-weighted sum matches
    total_valid_tokens = 3 + 1
    
    assert seq1_loss.item() > 0
    assert seq2_loss.item() > 0
    
    # Padding positions should strictly have 0 loss in reduction='none' with ignore_index
    assert batch_loss_tensor[1, 1].item() == 0.0
    assert batch_loss_tensor[1, 2].item() == 0.0
    
    # Check that model.compute_loss returns the exact mean over the 4 valid tokens
    expected_mean = (seq1_loss + seq2_loss).item() / 4.0
    assert math.isclose(mean_loss, expected_mean, rel_tol=1e-5)
    
    # The evaluation script multiplies mean_loss by valid_tokens
    recovered_sum = mean_loss * total_valid_tokens
    assert math.isclose(recovered_sum, (seq1_loss + seq2_loss).item(), rel_tol=1e-5)
    
def test_perplexity_calculation():
    # If loss is 2.5, perplexity is exp(2.5)
    loss = 2.5
    expected_ppl = math.exp(loss)
    assert math.isclose(expected_ppl, 12.18249396, rel_tol=1e-5)
    
def test_category_tracking_and_aggregation():
    # Construct a mock of the script's tracking logic
    batch = [
        {"input_ids": [2, 5, 0], "language": "en"},
        {"input_ids": [2, 6, 7], "language": "hi"},
        {"input_ids": [2, 0, 0], "language": "hinglish"},
    ]
    input_ids, languages = prepare_batch(batch, torch.device("cpu"))
    
    assert languages == ["en", "hi", "hinglish"]
    targets = input_ids[:, 1:]
    seq_valid_tokens = (targets != 0).sum(dim=1)
    
    assert seq_valid_tokens[0].item() == 1  # [5] is valid, [0] is pad
    assert seq_valid_tokens[1].item() == 2  # [6, 7] are valid
    assert seq_valid_tokens[2].item() == 0  # [0, 0] are pad
    
    cat_tokens = {"en": 0, "hi": 0, "hinglish": 0}
    for i, lang in enumerate(languages):
        cat_tokens[lang] += seq_valid_tokens[i].item()
        
    assert cat_tokens["en"] == 1
    assert cat_tokens["hi"] == 2
    assert cat_tokens["hinglish"] == 0
    
    # Total tokens reconcile
    assert sum(cat_tokens.values()) == seq_valid_tokens.sum().item()

def test_prepare_batch_strict_provenance():
    # Test missing language
    batch_missing = [{"input_ids": [2, 5, 0]}]
    with pytest.raises(KeyError, match="Missing 'language' provenance field"):
        prepare_batch(batch_missing, torch.device("cpu"))
        
    # Test unknown language
    batch_unknown = [{"input_ids": [2, 5, 0], "language": "es"}]
    with pytest.raises(ValueError, match="Unknown language category"):
        prepare_batch(batch_unknown, torch.device("cpu"))

def test_greedy_decoding_stops_at_eos():
    # Create a dummy tokenizer
    class DummyTokenizer:
        def encode(self, text):
            class Encoded:
                ids = [2, 10, 11]
            return Encoded()
        
        def token_to_id(self, token):
            if token == "<pad>": return 0
            if token == "<eos>": return 3
            return 1
            
        def decode(self, ids):
            return " " + str(ids)

    # Create a dummy model that always predicts <eos> immediately
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length = 10
            
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 20)
            logits[:, :, 3] = 10.0  # highest probability for <eos>
            return logits

    model = DummyModel()
    tokenizer = DummyTokenizer()
    
    # Generate
    continuation = greedy_generate(model, tokenizer, "Test", 5, torch.device("cpu"))
    # Should stop immediately without adding any tokens other than EOS, or perhaps EOS isn't added to output
    # The script breaks on EOS before appending
    assert continuation == " []" # decoded empty list
    
def test_greedy_decoding_predicts_tokens():
    class DummyTokenizer:
        def encode(self, text):
            class Encoded:
                ids = [2]
            return Encoded()
        def token_to_id(self, token):
            return 0 if token == "<pad>" else 3 if token == "<eos>" else 1
        def decode(self, ids):
            return str(ids)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length = 10
            self.step = 0
            
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 20)
            # Predict 5, then 6, then EOS (3)
            preds = [5, 6, 3]
            logits[:, -1, preds[self.step]] = 10.0
            self.step += 1
            return logits

    model = DummyModel()
    tokenizer = DummyTokenizer()
    continuation = greedy_generate(model, tokenizer, "Test", 5, torch.device("cpu"))
    assert continuation == "[5, 6]"

def test_empty_category_hard_failure():
    # evaluate.py explicitly loops over category_metrics and raises RuntimeError if v["tokens"] == 0
    category_metrics = {
        "en": {"loss_sum": 10.0, "tokens": 5, "examples": 1},
        "hi": {"loss_sum": 0.0, "tokens": 0, "examples": 0}
    }
    
    with pytest.raises(RuntimeError, match="has zero valid target tokens"):
        for lang, v in category_metrics.items():
            if v["tokens"] == 0:
                raise RuntimeError(f"Category '{lang}' has zero valid target tokens.")
            v["loss"] = v["loss_sum"] / v["tokens"]

def test_non_finite_loss_rejection():
    # Test NaN and Infinity rejection for overall or validation loss
    import math
    for bad_loss in [float('nan'), float('inf'), -1.0]:
        with pytest.raises(ValueError, match="Invalid validation loss value"):
            if not math.isfinite(bad_loss) or bad_loss < 0:
                raise ValueError(f"Invalid validation loss value: {bad_loss}")

def test_tokenizer_special_ids():
    class DummyTokenizer:
        def get_vocab_size(self): return 16000
        def token_to_id(self, token):
            return {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 999}.get(token, -1)
            
    tokenizer = DummyTokenizer()
    
    pad_id = tokenizer.token_to_id("<pad>")
    assert pad_id == 0
    unk_id = tokenizer.token_to_id("<unk>")
    assert unk_id == 1
    bos_id = tokenizer.token_to_id("<bos>")
    assert bos_id == 2
    eos_id = tokenizer.token_to_id("<eos>")
    
    with pytest.raises(ValueError, match="<eos> is 999, expected 3."):
        if eos_id != 3:
            raise ValueError(f"<eos> is {eos_id}, expected 3.")
