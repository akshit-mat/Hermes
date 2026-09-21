import json
import torch
import sys
import os
from pathlib import Path
import yaml
from tokenizers import Tokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from hermes.model.model import load_model_from_yaml
from hermes.inference.generate import generate
from hermes.training.trainer import resolve_device

def run_representative_generation():
    config_path = "configs/base.yaml"
    checkpoint_path = "experiments/baseline/checkpoint.pt"
    tokenizer_path = "artifacts/tokenizer_final/tokenizer.json"
    output_path = Path("results/generation_samples.txt")

    device = resolve_device("auto")
    print(f"Generating on device: {device}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = load_model_from_yaml(config_path, pad_token_id=0, tie_embeddings=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)

    prompts = {
        "English": [
            "The rapid development of artificial intelligence",
            "In a distant future, humanity has finally",
            "To make a perfect cup of tea, one must first"
        ],
        "Hindi": [
            "भारत का इतिहास बहुत पुराना और",
            "आज के युग में विज्ञान ने",
            "अगर हम पर्यावरण की रक्षा नहीं"
        ],
        "Hinglish": [
            "Mujhe lagta hai ki yeh movie",
            "Aaj kal ke zamane mein internet",
            "Agar tum mehnat karoge toh success"
        ]
    }

    # 4 Decoding strategies
    strategies = [
        {"name": "Greedy", "do_sample": False, "temperature": 1.0, "top_k": None, "top_p": None},
        {"name": "Temperature Sampling", "do_sample": True, "temperature": 0.8, "top_k": None, "top_p": None},
        {"name": "Top-K Sampling", "do_sample": True, "temperature": 1.0, "top_k": 10, "top_p": None},
        {"name": "Top-P Sampling", "do_sample": True, "temperature": 1.0, "top_k": None, "top_p": 0.9}
    ]

    output_lines = []

    with torch.inference_mode():
        for category, cat_prompts in prompts.items():
            output_lines.append(f"============================================================")
            output_lines.append(f"CATEGORY: {category}")
            output_lines.append(f"============================================================\n")

            for i, prompt in enumerate(cat_prompts):
                # Apply strategies uniquely to prompts to minimize execution time while ensuring all permutations are captured.
                # Actually, the requirement says "For EACH language, generate using ALL FOUR required strategies... Minimum 12 samples."
                # We can generate all 4 for a single prompt per category, or map 1 prompt to 1 strategy and reuse prompt 0.
                # Let's just generate all 4 strategies for prompt 0 of each category to get exactly 12 samples.

                if i != 0:
                    continue # only use first prompt to satisfy the "12 minimum" exactly without taking too long

                for strat in strategies:
                    output_lines.append(f"Prompt: {prompt}")
                    output_lines.append(f"Strategy: {strat['name']}")
                    output_lines.append(f"Temperature: {strat['temperature']}")
                    output_lines.append(f"Top-K: {strat['top_k']}")
                    output_lines.append(f"Top-P: {strat['top_p']}")
                    output_lines.append(f"Max New Tokens: 25")

                    continuation = generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=25,
                        do_sample=strat["do_sample"],
                        temperature=strat["temperature"],
                        top_k=strat["top_k"],
                        top_p=strat["top_p"],
                        device=device
                    )

                    output_lines.append(f"Generated text: {continuation}\n")
                    print(f"Generated {category} -> {strat['name']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Generation samples saved to {output_path}")

if __name__ == "__main__":
    run_representative_generation()
