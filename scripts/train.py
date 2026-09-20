from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes.model.model import build_model_from_config
from hermes.tokenizer.tokenizer import HermesTokenizer
from hermes.training.data import create_dataloader
from hermes.training.trainer import (
    create_local_logger,
    resolve_device,
    set_deterministic_seed,
    train_basic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the basic Hermes training loop."
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    with Path(args.config).open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle) or {}

    training_config = config["training"]
    tokenized_config = config["tokenized_data"]

    seed = int(
        training_config.get(
            "seed",
            config.get("data", {}).get("seed", 42),
        )
    )

    set_deterministic_seed(seed)

    tokenizer = HermesTokenizer.load(
        config["tokenizer"]["output_path"]
    )

    pad_token_id = tokenizer.token_to_id("<pad>")

    if pad_token_id is None:
        raise ValueError(
            "Loaded tokenizer does not define <pad>"
        )

    model = build_model_from_config(
        config,
        pad_token_id=pad_token_id,
    )

    logger = create_local_logger(
        training_config.get("log_path")
    )

    logger(f"seed={seed}")

    split_dir = Path(
        tokenized_config["output_dir"]
    )

    train_loader = create_dataloader(
        split_dir / "train.jsonl",
        batch_size=int(
            training_config["batch_size"]
        ),
        shuffle=True,
        seed=seed,
    )

    validation_loader = create_dataloader(
        split_dir / "validation.jsonl",
        batch_size=int(
            training_config["batch_size"]
        ),
        shuffle=False,
    )

    history = train_basic(
        model,
        train_loader,
        validation_loader,
        learning_rate=float(
            training_config["learning_rate"]
        ),
        weight_decay=float(
            training_config["weight_decay"]
        ),
        warmup_steps=int(
            training_config.get("warmup_steps", 0)
        ),
        max_steps=int(
            args.max_steps
            or training_config["max_steps"]
        ),
        gradient_clip_norm=float(
            training_config.get(
                "gradient_clip_norm",
                1.0,
            )
        ),
        gradient_accumulation_steps=int(
            training_config.get(
                "gradient_accumulation_steps",
                1,
            )
        ),
        validation_interval=int(
            training_config.get(
                "validation_interval",
                100,
            )
        ),
        log_interval=int(
            training_config.get(
                "log_interval",
                10,
            )
        ),
        device=resolve_device(
            training_config.get(
                "device",
                "auto",
            )
        ),
        logger=logger,
        checkpoint_path=training_config.get(
            "checkpoint_path"
        ),
        checkpoint_interval=int(
            training_config.get(
                "checkpoint_interval",
                0,
            )
        ),
        resume_from=training_config.get(
            "resume_from"
        ),
        use_mixed_precision=bool(
            training_config.get(
                "use_mixed_precision",
                False,
            )
        ),
        use_wandb=bool(
            config.get(
                "logging",
                {},
            ).get(
                "use_wandb",
                False,
            )
        ),
        wandb_project=str(
            config.get(
                "logging",
                {},
            ).get(
                "project",
                "hermes",
            )
        ),
        wandb_run_name=config.get(
            "logging",
            {},
        ).get("run_name"),
        wandb_config=config,
    )

    print(
        f"completed_steps="
        f"{len(history['train_loss'])}"
    )


if __name__ == "__main__":
    main()