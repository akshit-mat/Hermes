from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_warmup_cosine_scheduler(
	optimizer: Optimizer,
	*,
	warmup_steps: int,
	max_steps: int,
) -> LambdaLR:
	if warmup_steps < 0:
		raise ValueError("warmup_steps must be non-negative")
	if max_steps <= 0:
		raise ValueError("max_steps must be positive")
	if warmup_steps > max_steps:
		raise ValueError("warmup_steps cannot exceed max_steps")

	def learning_rate_multiplier(step: int) -> float:
		if step < warmup_steps:
			return (step + 1) / max(1, warmup_steps)
		if step >= max_steps:
			return 0.0
		decay_steps = max_steps - warmup_steps
		progress = (step - warmup_steps) / max(1, decay_steps)
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	return LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
