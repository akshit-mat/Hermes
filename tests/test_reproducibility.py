import torch
import pytest
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from hermes.model.model import load_model_from_yaml
from hermes.inference.generate import generate
from tokenizers import Tokenizer

def test_seed_reproducibility():
    # If we set seed, random tensors should be identical
    torch.manual_seed(42)
    a = torch.randn(10, 10)

    torch.manual_seed(42)
    b = torch.randn(10, 10)

    assert torch.equal(a, b)

def test_checkpoint_reload_reproducibility():
    config_path = "configs/base.yaml"
    model_path = "artifacts/inference_model/model.pt"

    # Load 1
    model1 = load_model_from_yaml(config_path, pad_token_id=0, tie_embeddings=True)
    ckpt1 = torch.load(model_path, map_location="cpu", weights_only=False)
    model1.load_state_dict(ckpt1["model"])
    model1.eval()

    # Load 2
    model2 = load_model_from_yaml(config_path, pad_token_id=0, tie_embeddings=True)
    ckpt2 = torch.load(model_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ckpt2["model"])
    model2.eval()

    dummy_input = torch.randint(0, 1000, (1, 10))

    with torch.inference_mode():
        out1 = model1(dummy_input)
        out2 = model2(dummy_input)

    assert torch.equal(out1, out2)

def test_generation_reproducibility():
    config_path = "configs/base.yaml"
    model_path = "artifacts/inference_model/model.pt"
    tokenizer_path = "artifacts/inference_model/tokenizer.json"

    model = load_model_from_yaml(config_path, pad_token_id=0, tie_embeddings=True)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Run twice
    out1 = generate(model, tokenizer, "Hello", max_new_tokens=5, do_sample=False, device="cpu")
    out2 = generate(model, tokenizer, "Hello", max_new_tokens=5, do_sample=False, device="cpu")

    assert out1 == out2
