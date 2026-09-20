from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    scaler: Any | None = None,
    data_state: dict[str, Any] | None = None,
) -> Path:
    """Save complete model, optimizer, scheduler, RNG, and data state."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if data_state is not None:
        state["data_state"] = data_state

    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
    return_data_state: bool = False,
) -> int | tuple[int, dict[str, Any] | None]:
    """Load training state and optionally return the saved data position.

    The default return value remains the integer training step for backward
    compatibility with existing callers. Callers that need exact resumable
    data position can set ``return_data_state=True``.
    """

    state = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    step = int(state["step"])
    data_state = state.get("data_state")

    if return_data_state:
        return step, data_state

    return step