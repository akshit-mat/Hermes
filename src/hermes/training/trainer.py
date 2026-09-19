from __future__ import annotations

import math
import logging
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hermes.model.model import HermesModel
from hermes.training.checkpoint import load_checkpoint, save_checkpoint
from hermes.training.scheduler import create_warmup_cosine_scheduler


def set_deterministic_seed(seed: int) -> None:
    """Seed all training-related random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device.

    Accepts the special value ``"auto"`` which selects CUDA when available and
    falls back to CPU otherwise.  Any other string is passed directly to
    ``torch.device()`` unchanged so that explicit ``"cpu"`` or ``"cuda"``
    values continue to work as before.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _create_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, RuntimeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def _batch_loss(model: HermesModel, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    logits = model(input_ids)
    return model.compute_loss(logits, input_ids)


def _fast_forward_iterator(
    train_loader: DataLoader[dict[str, torch.Tensor]],
    train_iterator: Any,
    batches_to_skip: int,
) -> Any:
    """Advance a lazy loader without materializing its dataset."""
    for _ in range(batches_to_skip):
        try:
            next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            next(train_iterator)
    return train_iterator


def validate(
    model: HermesModel,
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in validation_loader:
            losses.append(float(_batch_loss(model, batch, device).item()))
    if not losses:
        raise ValueError("Validation split is empty")
    return sum(losses) / len(losses)


def train_basic(
    model: HermesModel,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int = 0,
    max_steps: int,
    gradient_clip_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    validation_interval: int,
    log_interval: int,
    device: torch.device,
    logger: Callable[[str], None] = print,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 0,
    resume_from: str | None = None,
    use_mixed_precision: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "hermes",
    wandb_run_name: str | None = None,
    wandb_config: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = create_warmup_cosine_scheduler(
        optimizer, warmup_steps=warmup_steps, max_steps=max_steps
    )
    model.to(device)
    amp_enabled = bool(
        use_mixed_precision and device.type == "cuda" and torch.cuda.is_available()
    )
    scaler = _create_grad_scaler(amp_enabled)
    if use_mixed_precision and not amp_enabled:
        logger("mixed_precision=disabled (CUDA AMP is unavailable on this device)")
    micro_batch_size = train_loader.batch_size or 1
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    logger(
        f"effective_batch_size={micro_batch_size}x{gradient_accumulation_steps}="
        f"{effective_batch_size}"
    )
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config or {},
                reinit="return_previous",
            )
            logger("wandb=enabled")
        except Exception as error:
            logger(f"wandb=disabled ({error})")
    history = {"train_loss": [], "validation_loss": []}
    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        logger(f"resumed_from={resume_from} step={start_step}")

    restored_torch_rng_state = torch.get_rng_state() if start_step else None
    train_iterator = iter(train_loader)
    if restored_torch_rng_state is not None:
        torch.set_rng_state(restored_torch_rng_state)
    if start_step:
        consumed_micro_batches = start_step * gradient_accumulation_steps
        train_iterator = _fast_forward_iterator(
            train_loader, train_iterator, consumed_micro_batches
        )
        logger(f"resumed_data_position=micro_batch={consumed_micro_batches + 1}")
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, max_steps + 1):
        model.train()
        accumulated_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = _batch_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")
            accumulated_loss += float(loss.detach().item())
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_value = accumulated_loss / gradient_accumulation_steps
        history["train_loss"].append(train_value)

        if step % log_interval == 0:
            learning_rate = scheduler.get_last_lr()[0]
            logger(f"step={step} train_loss={train_value:.6f} lr={learning_rate:.8g}")
            if wandb_run is not None:
                wandb_run.log({"step": step, "train_loss": train_value, "learning_rate": learning_rate})
        if step % validation_interval == 0 or step == max_steps:
            validation_value = validate(model, validation_loader, device=device)
            if not math.isfinite(validation_value):
                raise FloatingPointError(
                    f"Non-finite validation loss at step {step}: {validation_value}"
                )
            history["validation_loss"].append(validation_value)
            logger(f"step={step} validation_loss={validation_value:.6f}")
            if wandb_run is not None:
                wandb_run.log({"step": step, "validation_loss": validation_value})
        if checkpoint_path and checkpoint_interval > 0 and step % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                scaler=scaler,
            )
            logger(f"checkpoint_saved={checkpoint_path} step={step}")
    if wandb_run is not None:
        wandb_run.finish()
    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=max_steps,
            scaler=scaler,
        )
    return history


def create_local_logger(log_path: str | None = None) -> Callable[[str], None]:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logger = logging.getLogger(f"hermes.training.{id(handlers)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger.info