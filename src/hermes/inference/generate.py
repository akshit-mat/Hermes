import torch
from typing import Any

def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | str | None = None
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    if do_sample:
        if temperature <= 0.0:
            raise ValueError("temperature must be strictly positive when do_sample=True.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if top_p is not None and (top_p <= 0.0 or top_p > 1.0):
            raise ValueError("top_p must be in range (0, 1].")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer does not have an <eos> token.")

    generated_ids = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Enforce context length via sliding window
            if input_ids.shape[1] > model.context_length:
                input_ids = input_ids[:, -model.context_length:]

            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]

            if not do_sample:
                next_token_id = torch.argmax(next_token_logits).item()
            else:
                # 1. Temperature scaling
                next_token_logits = next_token_logits / temperature

                # 2. Top-K filtering
                if top_k is not None:
                    k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # 3. Top-P (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')

                # 4. Probability normalization & Check for invalid values
                probs = torch.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any() or probs.sum() == 0:
                    raise RuntimeError("Invalid probabilities encountered after filtering.")

                # 5. Token selection
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id == eos_id:
                break

            generated_ids.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=device)], dim=1)

    if not generated_ids:
        return ""

    return tokenizer.decode(generated_ids)
