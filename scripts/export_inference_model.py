import torch
import os
from pathlib import Path
import shutil

def export_model():
    input_path = "experiments/baseline/checkpoint.pt"
    output_dir = Path("artifacts/inference_model")
    output_path = output_dir / "model.pt"

    print(f"Loading {input_path}...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    inference_ckpt = {
        "model": ckpt["model"],
        "step": ckpt["step"]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(inference_ckpt, output_path)

    # Copy tokenizer
    shutil.copy("artifacts/tokenizer_final/tokenizer.json", output_dir / "tokenizer.json")
    # Copy config
    shutil.copy("configs/base.yaml", output_dir / "config.yaml")

    orig_size = os.path.getsize(input_path) / (1024*1024)
    new_size = os.path.getsize(output_path) / (1024*1024)

    print(f"Original size: {orig_size:.2f} MB")
    print(f"New size: {new_size:.2f} MB")
    print(f"Exported to {output_dir}")

if __name__ == "__main__":
    export_model()
