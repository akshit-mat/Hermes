import gradio as gr
import torch
import yaml
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from hermes.model.model import load_model_from_yaml
from hermes.inference.generate import generate
from tokenizers import Tokenizer

# Global model loading
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = Path("artifacts/inference_model/model.pt")
config_path = Path("artifacts/inference_model/config.yaml")
tokenizer_path = Path("artifacts/inference_model/tokenizer.json")

print(f"Loading inference artifact on {device}...")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model = load_model_from_yaml(str(config_path), pad_token_id=0, tie_embeddings=True)
ckpt = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

tokenizer = Tokenizer.from_file(str(tokenizer_path))
print("Model and Tokenizer loaded successfully.")

def generate_text(prompt, temperature, top_k, top_p, max_new_tokens):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")

    if max_new_tokens < 0:
        raise gr.Error("Max new tokens must be non-negative.")

    do_sample = temperature > 0.0 or top_k > 0 or top_p < 1.0

    # We will safely pass top_k and top_p. If top_k is 0, we treat it as None for the engine.
    # Actually, in our Gradio we can just pass them directly, but the API expects positive ints or None.
    k = int(top_k) if top_k > 0 else None
    p = float(top_p) if top_p < 1.0 else None

    if do_sample and temperature <= 0.0:
        raise gr.Error("Temperature must be strictly positive when sampling.")

    try:
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else 1.0,
            top_k=k,
            top_p=p,
            device=device
        )
        return output
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

with gr.Blocks(title="Hermes Phase 1 Baseline Demo") as demo:
    gr.Markdown("# Hermes Phase 1 Baseline")
    gr.Markdown("Minimal generation demo for the Hermes Phase 1 (16.9M parameter) checkpoint. Supports English, Hindi (Devanagari), and Hinglish.")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(lines=5, label="Prompt", placeholder="Enter prompt here...")
            max_new_tokens = gr.Number(value=50, label="Max New Tokens", precision=0)

            with gr.Accordion("Generation Parameters"):
                temperature = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Temperature (0 = Greedy)")
                top_k = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Top-K (0 = Disabled)")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-P Nucleus (1.0 = Disabled)")

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_display = gr.Textbox(lines=10, label="Generated Output")

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, temperature, top_k, top_p, max_new_tokens],
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
