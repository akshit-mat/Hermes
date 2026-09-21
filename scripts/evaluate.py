import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable
import datetime

import torch

from hermes.model.model import load_model_from_yaml
from hermes.training.trainer import resolve_device

def stream_jsonl_batches(path: str | Path, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """Stream records from a JSONL file in batches."""
    batch = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def prepare_batch(batch: List[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    """Convert a batch of JSON records into an input_ids tensor and a list of languages."""
    input_ids = torch.tensor([record["input_ids"] for record in batch], dtype=torch.long, device=device)
    languages = []
    for record in batch:
        if "language" not in record:
            raise KeyError("Missing 'language' provenance field in production data.")
        lang = record["language"]
        if lang not in ["en", "hi", "hinglish"]:
            raise ValueError(f"Unknown language category '{lang}'")
        languages.append(lang)
    return input_ids, languages

def greedy_generate(model: torch.nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    """Perform simple greedy decoding for qualitative generation."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    
    generated_ids = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > model.context_length:
                # Truncate left if we exceed context length
                input_ids = input_ids[:, -model.context_length:]
            
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            if next_token_id == eos_id:
                break
                
            generated_ids.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=device)], dim=1)
            
    return tokenizer.decode(generated_ids)

def run_evaluation():
    config_path = "configs/base.yaml"
    checkpoint_path = "experiments/baseline/checkpoint.pt"
    tokenizer_path = "artifacts/tokenizer_final/tokenizer.json"
    validation_path = "data/tokenized_final/validation.jsonl"
    test_path = "data/tokenized_final/test.jsonl"
    
    device = resolve_device("auto")
    print(f"Evaluating on device: {device}")
    
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    model = load_model_from_yaml(config_path, pad_token_id=0, tie_embeddings=True)
    
    if model.vocab_size != 16000:
        raise ValueError(f"Model vocab size is {model.vocab_size}, expected 16000.")
    if model.context_length != 320:
        raise ValueError(f"Model context length is {model.context_length}, expected 320.")
    if len(model.blocks) != 6:
        raise ValueError(f"Model layers is {len(model.blocks)}, expected 6.")
    if model.blocks[0].attention.n_heads != 6:
        raise ValueError(f"Model heads is {model.blocks[0].attention.n_heads}, expected 6.")
    if config["model"]["d_model"] != 384:
        raise ValueError(f"Model d_model is {config['model']['d_model']}, expected 384.")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["step"] != 10000:
        raise ValueError(f"Checkpoint step is {checkpoint['step']}, expected 10000")
        
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.get_vocab_size() != 16000:
        raise ValueError(f"Tokenizer vocab size is {tokenizer.get_vocab_size()}, expected 16000.")
        
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id != 0: raise ValueError(f"<pad> is {pad_id}, expected 0.")
    unk_id = tokenizer.token_to_id("<unk>")
    if unk_id != 1: raise ValueError(f"<unk> is {unk_id}, expected 1.")
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id != 2: raise ValueError(f"<bos> is {bos_id}, expected 2.")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id != 3: raise ValueError(f"<eos> is {eos_id}, expected 3.")
        
    batch_size = 128
    
    # ---------------------------------------------------------
    # VALIDATION EVALUATION
    # ---------------------------------------------------------
    print("Evaluating validation split...", flush=True)
    val_loss_sum = 0.0
    val_tokens = 0
    val_examples = 0
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(stream_jsonl_batches(validation_path, batch_size)):
            if batch_idx % 100 == 0:
                print(f"  Validation Batch {batch_idx}", flush=True)
            input_ids, _ = prepare_batch(batch, device)
            # Forward pass and loss
            logits = model(input_ids)
            # loss is mean over non-padded tokens
            loss = model.compute_loss(logits, input_ids)
            
            # Count valid tokens (exclude pad token 0 from targets, shifted by 1)
            targets = input_ids[:, 1:]
            valid_tokens = (targets != 0).sum().item()
            
            if valid_tokens > 0:
                val_loss_sum += loss.item() * valid_tokens
                val_tokens += valid_tokens
            val_examples += len(batch)
            
    if val_tokens == 0:
        raise RuntimeError("Empty validation split or zero valid tokens found.")
        
    val_overall_loss = val_loss_sum / val_tokens
    
    if not math.isfinite(val_overall_loss) or val_overall_loss < 0:
        raise ValueError(f"Invalid validation loss value: {val_overall_loss}")
    
    # ---------------------------------------------------------
    # TEST EVALUATION
    # ---------------------------------------------------------
    print("Evaluating test split...", flush=True)
    test_total_loss_sum = 0.0
    test_total_tokens = 0
    test_total_examples = 0
    
    category_metrics = {
        "en": {"loss_sum": 0.0, "tokens": 0, "examples": 0},
        "hi": {"loss_sum": 0.0, "tokens": 0, "examples": 0},
        "hinglish": {"loss_sum": 0.0, "tokens": 0, "examples": 0},
    }
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(stream_jsonl_batches(test_path, batch_size)):
            if batch_idx % 100 == 0:
                print(f"  Test Batch {batch_idx}", flush=True)
            input_ids, languages = prepare_batch(batch, device)
            logits = model(input_ids)
            
            # We must compute token counts per sequence to correctly bin category losses.
            # model.compute_loss computes batch mean. We can compute per-sequence loss using cross_entropy directly.
            targets = input_ids[:, 1:]
            batch_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, model.vocab_size),
                targets.contiguous().view(-1),
                ignore_index=model.pad_token_id,
                reduction='none'
            ).view(input_ids.shape[0], -1)
            
            seq_loss_sum = batch_loss.sum(dim=1)
            seq_valid_tokens = (targets != model.pad_token_id).sum(dim=1)
            
            for i, lang in enumerate(languages):
                sloss = seq_loss_sum[i].item()
                stoks = seq_valid_tokens[i].item()
                
                if stoks > 0:
                    test_total_loss_sum += sloss
                    test_total_tokens += stoks
                    
                    if lang in category_metrics:
                        category_metrics[lang]["loss_sum"] += sloss
                        category_metrics[lang]["tokens"] += stoks
                
                test_total_examples += 1
                if lang in category_metrics:
                    category_metrics[lang]["examples"] += 1
                    
    if test_total_tokens == 0:
        raise RuntimeError("Empty test split or zero valid tokens found.")
    
    test_overall_loss = test_total_loss_sum / test_total_tokens
    
    if not math.isfinite(test_overall_loss) or test_overall_loss < 0:
        raise ValueError(f"Invalid test loss value: {test_overall_loss}")
        
    test_overall_perplexity = math.exp(test_overall_loss)
    
    # Sanity checks
    cat_tokens_sum = sum(v["tokens"] for v in category_metrics.values())
    cat_examples_sum = sum(v["examples"] for v in category_metrics.values())
    
    if cat_tokens_sum != test_total_tokens:
        raise RuntimeError(f"Category token count mismatch! Total: {test_total_tokens}, Categories sum: {cat_tokens_sum}")
    if cat_examples_sum != test_total_examples:
        raise RuntimeError(f"Category example count mismatch! Total: {test_total_examples}, Categories sum: {cat_examples_sum}")
        
    for lang, v in category_metrics.items():
        if v["tokens"] == 0:
            raise RuntimeError(f"Category '{lang}' has zero valid target tokens.")
            
        v["loss"] = v["loss_sum"] / v["tokens"]
        
        if not math.isfinite(v["loss"]) or v["loss"] < 0:
            raise ValueError(f"Invalid category loss for '{lang}': {v['loss']}")
            
        v["perplexity"] = math.exp(v["loss"])
        
    # ---------------------------------------------------------
    # QUALITATIVE GENERATION
    # ---------------------------------------------------------
    print("Running qualitative generations...")
    prompts = {
        "en": [
            "The rapid development of artificial intelligence",
            "In a distant future, humanity has finally",
            "To make a perfect cup of tea, one must first"
        ],
        "hi": [
            "भारत का इतिहास बहुत पुराना और",
            "आज के युग में विज्ञान ने",
            "अगर हम पर्यावरण की रक्षा नहीं"
        ],
        "hinglish": [
            "Mujhe lagta hai ki yeh movie",
            "Aaj kal ke zamane mein internet",
            "Agar tum mehnat karoge toh success"
        ]
    }
    
    generations_text = ""
    qualitative_results = []
    
    for lang, lang_prompts in prompts.items():
        for prompt in lang_prompts:
            continuation = greedy_generate(model, tokenizer, prompt, max_new_tokens=30, device=device)
            generations_text += f"Category: {lang}\nPrompt: {prompt}\nGenerated: {continuation}\n\n"
            qualitative_results.append({
                "category": lang,
                "prompt": prompt,
                "generated": continuation
            })
            
    # Save outputs
    eval_json = {
        "checkpoint": {
            "path": checkpoint_path,
            "step": checkpoint["step"]
        },
        "model": {
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
            "d_model": config["model"]["d_model"],
            "n_layers": len(model.blocks),
            "n_heads": model.blocks[0].attention.n_heads,
            "parameter_count": model.count_parameters(trainable_only=False)
        },
        "tokenizer_path": tokenizer_path,
        "validation_dataset_path": validation_path,
        "test_dataset_path": test_path,
        "evaluation_device": str(device),
        "timestamp": datetime.datetime.now().isoformat(),
        "validation": {
            "loss": val_overall_loss,
            "nonpadding_target_tokens": val_tokens,
            "examples": val_examples
        },
        "test": {
            "loss": test_overall_loss,
            "perplexity": test_overall_perplexity,
            "examples": test_total_examples,
            "nonpadding_target_tokens": test_total_tokens
        },
        "categories": {
            "english": {
                "examples": category_metrics["en"]["examples"],
                "nonpadding_target_tokens": category_metrics["en"]["tokens"],
                "loss": category_metrics["en"]["loss"],
                "perplexity": category_metrics["en"]["perplexity"]
            },
            "hindi": {
                "examples": category_metrics["hi"]["examples"],
                "nonpadding_target_tokens": category_metrics["hi"]["tokens"],
                "loss": category_metrics["hi"]["loss"],
                "perplexity": category_metrics["hi"]["perplexity"]
            },
            "hinglish": {
                "examples": category_metrics["hinglish"]["examples"],
                "nonpadding_target_tokens": category_metrics["hinglish"]["tokens"],
                "loss": category_metrics["hinglish"]["loss"],
                "perplexity": category_metrics["hinglish"]["perplexity"]
            }
        },
        "qualitative_generations": qualitative_results
    }
    
    Path("results/evaluation.json").write_text(json.dumps(eval_json, indent=2), encoding="utf-8")
    Path("results/qualitative_generations.txt").write_text(generations_text, encoding="utf-8")
    
    print("Evaluation complete. Results saved to results/evaluation.json and results/qualitative_generations.txt")

if __name__ == "__main__":
    run_evaluation()
