import torch
import pytest
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from hermes.model.model import load_model_from_yaml

def test_inference_artifact_loads_correctly():
    artifact_path = Path("artifacts/inference_model/model.pt")
    original_path = Path("experiments/baseline/checkpoint.pt")
    config_path = Path("configs/base.yaml")

    # Must exist
    assert artifact_path.exists()

    # Check that training state is stripped
    inference_ckpt = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert "optimizer" not in inference_ckpt
    assert "scheduler" not in inference_ckpt
    assert "data_state" not in inference_ckpt
    assert "python_rng_state" not in inference_ckpt

    assert "model" in inference_ckpt

    # Verify parameters match
    orig_ckpt = torch.load(original_path, map_location="cpu", weights_only=False)

    inf_model = inference_ckpt["model"]
    orig_model = orig_ckpt["model"]

    # Check keys
    assert set(inf_model.keys()) == set(orig_model.keys())

    # Check tensors are identical
    for k in inf_model.keys():
        assert torch.equal(inf_model[k], orig_model[k]), f"Parameter mismatch for {k}"

def test_tokenizer_artifact_exists():
    assert Path("artifacts/inference_model/tokenizer.json").exists()
    assert Path("artifacts/inference_model/config.yaml").exists()
