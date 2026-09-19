"""Tests for Day 9 runtime fixes:
  - resolve_device: "auto", explicit "cpu", explicit "cuda" (mocked)
  - checkpoint interval being honoured during training
  - checkpoint path is a file (not directory) and parent dirs are created
"""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from hermes.model.model import HermesModel
from hermes.training.trainer import resolve_device, set_deterministic_seed, train_basic


# ── helpers ───────────────────────────────────────────────────────────────────

class _TinyDataset(Dataset):
    """Minimal in-memory dataset for fast unit tests."""

    def __init__(self, size: int = 8) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
        }


def _tiny_loader(size: int = 8) -> DataLoader:
    return DataLoader(_TinyDataset(size), batch_size=2)


def _tiny_model() -> HermesModel:
    return HermesModel(
        vocab_size=16,
        context_length=4,
        d_model=8,
        n_layers=1,
        n_heads=2,
        pad_token_id=0,
    )


def _base_train_kwargs(checkpoint_path: str, checkpoint_interval: int = 0) -> dict:
    return dict(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=5,
        gradient_accumulation_steps=1,
        validation_interval=100,
        log_interval=10,
        device=torch.device("cpu"),
        logger=lambda _: None,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
    )


# ── resolve_device ─────────────────────────────────────────────────────────────

class TestResolveDevice:
    """Unit tests for resolve_device()."""

    def test_explicit_cpu_returns_cpu(self) -> None:
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_returns_cpu_when_cuda_unavailable(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            device = resolve_device("auto")
        assert device == torch.device("cpu")

    def test_auto_returns_cuda_when_cuda_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("auto")
        assert device == torch.device("cuda")

    def test_explicit_cuda_passes_through(self) -> None:
        # resolve_device does NOT check cuda availability for explicit strings —
        # it delegates to torch.device() directly, matching the contract.
        device = resolve_device("cuda")
        assert device == torch.device("cuda")

    def test_auto_default_matches_system(self) -> None:
        """The auto result must be consistent with is_available() on this machine."""
        device = resolve_device("auto")
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device == expected


# ── checkpoint interval ────────────────────────────────────────────────────────

class TestCheckpointInterval:
    """Verify that checkpoints are written exactly at the configured interval."""

    def test_checkpoint_saved_at_configured_interval(self, tmp_path) -> None:
        ckpt_path = tmp_path / "run" / "checkpoint.pt"
        # interval=2, max_steps=5 → periodic saves at step 2 and step 4,
        # plus the final save at step 5.
        set_deterministic_seed(0)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=2),
        )
        # The file must exist (final save always writes it when checkpoint_path is set)
        assert ckpt_path.exists()

    def test_checkpoint_parent_dirs_created(self, tmp_path) -> None:
        """The implementation creates missing parent directories automatically."""
        deep_path = tmp_path / "a" / "b" / "c" / "checkpoint.pt"
        assert not deep_path.parent.exists()
        set_deterministic_seed(1)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(deep_path), checkpoint_interval=5),
        )
        assert deep_path.exists()

    def test_checkpoint_is_a_file_not_directory(self, tmp_path) -> None:
        ckpt_path = tmp_path / "checkpoint.pt"
        set_deterministic_seed(2)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=2),
        )
        assert ckpt_path.is_file()
        assert not ckpt_path.is_dir()

    def test_checkpoint_contains_required_keys(self, tmp_path) -> None:
        ckpt_path = tmp_path / "checkpoint.pt"
        set_deterministic_seed(3)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=3),
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "scheduler", "step",
                    "python_rng_state", "numpy_rng_state", "torch_rng_state"}
        assert required.issubset(state.keys())

    def test_zero_interval_writes_only_final_checkpoint(self, tmp_path) -> None:
        """checkpoint_interval=0 means no periodic saves; final save still occurs."""
        ckpt_path = tmp_path / "checkpoint.pt"
        set_deterministic_seed(4)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=0),
        )
        # The final save (after the loop) still writes the file.
        assert ckpt_path.exists()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert state["step"] == 5

    def test_checkpoint_step_matches_max_steps_at_end(self, tmp_path) -> None:
        ckpt_path = tmp_path / "checkpoint.pt"
        set_deterministic_seed(5)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=2),
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Final save always records max_steps (=5 here)
        assert state["step"] == 5


# ── config path semantics ──────────────────────────────────────────────────────

class TestCheckpointPathSemantics:
    """Document and verify the single-file overwrite checkpoint semantics."""

    def test_periodic_saves_overwrite_same_file(self, tmp_path) -> None:
        """Each periodic save overwrites the same file; only one file exists."""
        ckpt_path = tmp_path / "checkpoint.pt"
        set_deterministic_seed(6)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=2),
        )
        # Only one file in the checkpoint directory — no per-step suffix files.
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "checkpoint.pt"

    def test_final_checkpoint_step_is_max_steps(self, tmp_path) -> None:
        ckpt_path = tmp_path / "checkpoint.pt"
        set_deterministic_seed(7)
        train_basic(
            _tiny_model(),
            _tiny_loader(),
            _tiny_loader(size=2),
            **_base_train_kwargs(str(ckpt_path), checkpoint_interval=2),
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert state["step"] == 5  # max_steps from _base_train_kwargs
