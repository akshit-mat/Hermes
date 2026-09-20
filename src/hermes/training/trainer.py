from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler

from hermes.model.model import HermesModel
from hermes.training.checkpoint import load_checkpoint, save_checkpoint
from hermes.training.data import EpochSeededRandomSampler
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
    falls back to CPU otherwise. Any other string is passed directly to
    ``torch.device()``.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_str)


def _create_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, RuntimeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def _batch_loss(
    model: HermesModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    logits = model(input_ids)
    return model.compute_loss(logits, input_ids)


def _get_sampler(
    train_loader: DataLoader[dict[str, torch.Tensor]] | Any,
) -> Any | None:
    """Return the underlying batch sampler's sampler when available."""
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is None:
        return None

    return getattr(batch_sampler, "sampler", None)


def _skip_batches(
    train_iterator: Any,
    batches_to_skip: int,
) -> Any:
    """Advance an iterator by an exact number of already-consumed batches.

    This function intentionally does not wrap around. Resume logic computes a
    position within the current epoch before calling this helper.
    """
    for _ in range(batches_to_skip):
        next(train_iterator)

    return train_iterator


def _build_data_state(
    train_loader: DataLoader[dict[str, torch.Tensor]] | Any,
    *,
    step: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """Build a compact checkpoint description of the consumed data position."""

    consumed_micro_batches = step * gradient_accumulation_steps
    sampler = _get_sampler(train_loader)

    if isinstance(sampler, EpochSeededRandomSampler):
        batches_per_epoch = len(train_loader)

        if batches_per_epoch <= 0:
            raise ValueError("Training DataLoader must contain at least one batch")

        epoch, batches_consumed_in_epoch = divmod(
            consumed_micro_batches,
            batches_per_epoch,
        )

        return {
            "sampler_type": "epoch_seeded_random",
            "sampler_seed": sampler.seed,
            "sampler_epoch": epoch,
            "batches_consumed_in_epoch": batches_consumed_in_epoch,
            "global_micro_batches": consumed_micro_batches,
        }

    if isinstance(sampler, SequentialSampler):
        batches_per_epoch = len(train_loader)

        if batches_per_epoch <= 0:
            raise ValueError("Training DataLoader must contain at least one batch")

        epoch, batches_consumed_in_epoch = divmod(
            consumed_micro_batches,
            batches_per_epoch,
        )

        return {
            "sampler_type": "sequential",
            "sampler_epoch": epoch,
            "batches_consumed_in_epoch": batches_consumed_in_epoch,
            "global_micro_batches": consumed_micro_batches,
        }

    raise ValueError(
        "Checkpointing/resume requires either EpochSeededRandomSampler or "
        "SequentialSampler. The training DataLoader uses an unsupported sampler."
    )


def _restore_data_position(
    train_loader: DataLoader[dict[str, torch.Tensor]] | Any,
    *,
    start_step: int,
    gradient_accumulation_steps: int,
    data_state: dict[str, Any] | None,
) -> tuple[Any, int]:
    """Reconstruct the exact DataLoader position represented by a checkpoint.

    Returns:
        (iterator, current_epoch)
    """

    sampler = _get_sampler(train_loader)

    if not isinstance(sampler, (EpochSeededRandomSampler, SequentialSampler)):
        raise ValueError(
            "Resumable training requires EpochSeededRandomSampler or "
            "SequentialSampler."
        )

    if len(train_loader) <= 0:
        raise ValueError("Training DataLoader must contain at least one batch")

    expected_global_micro_batches = (
        start_step * gradient_accumulation_steps
    )

    if data_state is None:
        if isinstance(sampler, EpochSeededRandomSampler):
            raise ValueError(
                "Checkpoint does not contain deterministic data-state metadata. "
                "The checkpoint was created before exact shuffled-data resume "
                "was implemented and cannot be resumed safely."
            )

        # Sequential data order is independent of RNG state, so an old
        # checkpoint without data_state can still be resumed exactly.
        _, batches_consumed_in_epoch = divmod(
            expected_global_micro_batches,
            len(train_loader),
        )
        iterator = iter(train_loader)
        _skip_batches(iterator, batches_consumed_in_epoch)
        return iterator, 0

    saved_global_micro_batches = int(
        data_state.get("global_micro_batches", -1)
    )

    if saved_global_micro_batches != expected_global_micro_batches:
        raise ValueError(
            "Checkpoint data position does not match the saved training step: "
            f"checkpoint={saved_global_micro_batches}, "
            f"expected={expected_global_micro_batches}"
        )

    sampler_type = data_state.get("sampler_type")

    if isinstance(sampler, EpochSeededRandomSampler):
        if sampler_type != "epoch_seeded_random":
            raise ValueError(
                "Checkpoint sampler type does not match the current training "
                "DataLoader."
            )

        saved_seed = int(data_state["sampler_seed"])
        saved_epoch = int(data_state["sampler_epoch"])
        batches_consumed_in_epoch = int(
            data_state["batches_consumed_in_epoch"]
        )

        expected_epoch, expected_offset = divmod(
            expected_global_micro_batches,
            len(train_loader),
        )

        if (
            saved_epoch != expected_epoch
            or batches_consumed_in_epoch != expected_offset
        ):
            raise ValueError(
                "Checkpoint data position is inconsistent with the saved "
                "training step."
            )

        # The checkpoint's sampler seed is authoritative. This guarantees that
        # a fresh process reconstructs the exact same ordering even if the
        # caller accidentally created the loader with another seed.
        sampler.seed = saved_seed
        sampler.set_epoch(saved_epoch)

        iterator = iter(train_loader)
        _skip_batches(iterator, batches_consumed_in_epoch)

        return iterator, saved_epoch

    if sampler_type != "sequential":
        raise ValueError(
            "Checkpoint sampler type does not match the current training "
            "DataLoader."
        )

    saved_epoch = int(data_state["sampler_epoch"])
    batches_consumed_in_epoch = int(
        data_state["batches_consumed_in_epoch"]
    )

    expected_epoch, expected_offset = divmod(
        expected_global_micro_batches,
        len(train_loader),
    )

    if (
        saved_epoch != expected_epoch
        or batches_consumed_in_epoch != expected_offset
    ):
        raise ValueError(
            "Checkpoint data position is inconsistent with the saved "
            "training step."
        )

    iterator = iter(train_loader)
    _skip_batches(iterator, batches_consumed_in_epoch)

    return iterator, saved_epoch


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
            losses.append(
                float(_batch_loss(model, batch, device).item())
            )

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
        raise ValueError(
            "gradient_accumulation_steps must be positive"
        )

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )

    model.to(device)

    amp_enabled = bool(
        use_mixed_precision
        and device.type == "cuda"
        and torch.cuda.is_available()
    )

    scaler = _create_grad_scaler(amp_enabled)

    if use_mixed_precision and not amp_enabled:
        logger(
            "mixed_precision=disabled "
            "(CUDA AMP is unavailable on this device)"
        )

    micro_batch_size = train_loader.batch_size or 1
    effective_batch_size = (
        micro_batch_size * gradient_accumulation_steps
    )

    logger(
        f"effective_batch_size={micro_batch_size}"
        f"x{gradient_accumulation_steps}="
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

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    start_step = 0
    data_state: dict[str, Any] | None = None

    if resume_from is not None:
        loaded = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            return_data_state=True,
        )

        start_step, data_state = loaded

        logger(
            f"resumed_from={resume_from} "
            f"step={start_step}"
        )

        if start_step > max_steps:
            raise ValueError(
                f"Checkpoint step {start_step} exceeds requested "
                f"max_steps {max_steps}"
            )

    sampler = _get_sampler(train_loader)

    if isinstance(
        sampler,
        (EpochSeededRandomSampler, SequentialSampler),
    ):
        if len(train_loader) <= 0:
            raise ValueError(
                "Training DataLoader must contain at least one batch"
            )

    if start_step:
        train_iterator, current_epoch = _restore_data_position(
            train_loader,
            start_step=start_step,
            gradient_accumulation_steps=gradient_accumulation_steps,
            data_state=data_state,
        )

        consumed_micro_batches = (
            start_step * gradient_accumulation_steps
        )

        logger(
            "resumed_data_position="
            f"micro_batch={consumed_micro_batches + 1}"
        )

    else:
        current_epoch = 0

        if isinstance(sampler, EpochSeededRandomSampler):
            sampler.set_epoch(0)

        train_iterator = iter(train_loader)

    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        accumulated_loss = 0.0

        for _micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)

            except StopIteration:
                current_epoch += 1

                if isinstance(
                    sampler,
                    EpochSeededRandomSampler,
                ):
                    sampler.set_epoch(current_epoch)

                train_iterator = iter(train_loader)

                try:
                    batch = next(train_iterator)
                except StopIteration as error:
                    raise ValueError(
                        "Training DataLoader is empty."
                    ) from error

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = _batch_loss(
                    model,
                    batch,
                    device,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at step {step}: "
                    f"{loss.item()}"
                )

            accumulated_loss += float(
                loss.detach().item()
            )

            scaled_loss = (
                loss / gradient_accumulation_steps
            )

            scaler.scale(scaled_loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_value = (
            accumulated_loss / gradient_accumulation_steps
        )

        history["train_loss"].append(train_value)

        if step % log_interval == 0:
            learning_rate = scheduler.get_last_lr()[0]

            logger(
                f"step={step} "
                f"train_loss={train_value:.6f} "
                f"lr={learning_rate:.8g}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "train_loss": train_value,
                        "learning_rate": learning_rate,
                    }
                )

        if step % validation_interval == 0 or step == max_steps:
            validation_value = validate(
                model,
                validation_loader,
                device=device,
            )

            if not math.isfinite(validation_value):
                raise FloatingPointError(
                    f"Non-finite validation loss at step {step}: "
                    f"{validation_value}"
                )

            history["validation_loss"].append(
                validation_value
            )

            logger(
                f"step={step} "
                f"validation_loss={validation_value:.6f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "validation_loss": validation_value,
                    }
                )

        if (
            checkpoint_path
            and checkpoint_interval > 0
            and step % checkpoint_interval == 0
        ):
            checkpoint_data_state = _build_data_state(
                train_loader,
                step=step,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                scaler=scaler,
                data_state=checkpoint_data_state,
            )

            logger(
                f"checkpoint_saved={checkpoint_path} "
                f"step={step}"
            )

    if wandb_run is not None:
        wandb_run.finish()

    if checkpoint_path:
        final_data_state = _build_data_state(
            train_loader,
            step=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=max_steps,
            scaler=scaler,
            data_state=final_data_state,
        )

    return history


def create_local_logger(
    log_path: str | None = None,
) -> Callable[[str], None]:
    handlers: list[logging.Handler] = [
        logging.StreamHandler()
    ]

    if log_path:
        Path(log_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handlers.append(
            logging.FileHandler(
                log_path,
                encoding="utf-8",
            )
        )

    logger = logging.getLogger(
        f"hermes.training.{id(handlers)}"
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in handlers:
        handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        logger.addHandler(handler)

    return logger.info