# Hermes

A small, from-scratch, decoder-only Transformer language model trained on English, Hindi, and Hindi-English code-mixed (Hinglish) text. Hermes is a 14-day ML engineering project: a complete pipeline from raw data through a custom tokenizer, a Transformer implemented from scratch in PyTorch, a configurable training system, evaluation, and interactive generation.

> **Status:** Planning complete. Architecture, data sources, and training infrastructure locked in. Implementation in progress — see [Reproduction](#reproduction) and [Results](#results) for current state.

---

## Features

- Custom BPE tokenizer (shared vocabulary across English, Devanagari Hindi, and Romanized Hinglish)
- Decoder-only Transformer implemented from scratch in PyTorch (no pretrained GPT import)
- Causal self-attention, pre-normalization, residual connections
- Configurable training system (single YAML source of truth)
- Mixed precision + gradient accumulation
- Resumable checkpointing (model, optimizer, scheduler, step)
- Experiment tracking (Weights & Biases, with local logging fallback)
- Held-out evaluation with per-language perplexity breakdown
- Autoregressive generation (temperature / top-k / top-p sampling)
- Gradio demo + Hugging Face deployment

---

## Architecture

```
Dataset
  ↓
Cleaning / Normalization
  ↓
BPE Tokenizer
  ↓
Token IDs
  ↓
Embedding (token + positional)
  ↓
Transformer Blocks × 6
  (LayerNorm → Causal Self-Attention → residual)
  (LayerNorm → MLP → residual)
  ↓
Language Model Head
  ↓
Next-token probabilities
  ↓
Autoregressive Generation
```

**Design decision:** the model is deliberately kept small rather than sized to maximize parameter count. Given the available free-tier GPU hardware (see [Hardware](#hardware)), this size trains in a small number of hours per run, which keeps iteration, debugging, and re-running fast throughout the 14-day build — a smaller model that trains reliably and is fully understood beats a larger one that's slow to iterate on or hard to explain end-to-end.

---

## Model

| Parameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension (d_model) | 384 |
| Context length | 320 |
| Vocabulary size | ~16,000 |
| Parameters (approx.) | ~15–18M |

Context length was set slightly above the spec's original default (256 → 320) to give headroom for English/Hindi Wikipedia sentence lengths, at negligible extra compute cost. It was not pushed further because the Hinglish source (Twitter-derived) is dominated by short sequences, so a large context window buys little.

This is a **compute-informed architecture choice**, distinct from any purely hardware-constrained shrinking (see [Training](#training) → Hardware).

---

## Dataset

Hermes v1 uses three deliberately chosen, reproducible sources rather than a custom scrape — prioritizing reproducibility over dataset naturalism.

| Language | Source | Notes |
|---|---|---|
| English | Hugging Face `wikimedia/wikipedia` (`en`) | Clean, deterministic, easy to slice to a fixed token budget |
| Hindi (Devanagari) | Hugging Face `wikimedia/wikipedia` (`hi`) | Same rationale as English |
| Hinglish (code-mixed) | `l3cube-pune/code-mixed-nlp` (L3Cube-HingCorpus) | Official L3Cube repo and Google Drive corpus artifact; Twitter-derived Hindi-English Roman-script code-mixed corpus |

**Corpus size (final):** 60M tokens each for English, Hindi, and Hinglish — **180M tokens total**. This is a deliberate sampling target, not a natural language distribution: all three sources (Wikipedia English, Wikipedia Hindi, L3Cube-HingCorpus) contain far more raw text than this, so the limiting factor is the training compute budget below, not data availability. 60M tokens per category is small enough to fit inside a single weekly free-GPU quota for a full epoch, with room left for multiple epochs or reruns in the same week (see [Hardware](#hardware)).

**Pipeline (per spec):** acquisition → normalization → cleaning → deduplication → filtering → deterministic train/val/test split → recorded statistics (example counts, approximate token counts, distribution, filtering rules). Validation/test sets are strictly excluded from tokenizer training and model training to avoid leakage.

**Normalization is conservative:** no aggressive transliteration. Devanagari, Latin script, punctuation, and meaningful whitespace are preserved; only obvious formatting issues are normalized.

---

## Tokenizer

- Custom BPE trained with the Hugging Face `tokenizers` library
- Shared vocabulary across all three languages/categories
- Target vocabulary size: ~16,000 tokens
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Supports `train()`, `encode()`, `decode()`, `save()`, `load()`
- Verified with unit tests and manual inspection of real tokenization examples before model training begins

---

## Training

- **Objective:** standard causal language modeling (cross-entropy, next-token prediction, padding ignored in loss)
- **Optimizer:** AdamW
- **Schedule:** linear warmup → cosine decay
- **Stability:** gradient clipping, mixed precision, gradient accumulation as needed
- **Reproducibility:** fixed seeds (Python, NumPy, PyTorch); deterministic behavior where practical
- **Checkpointing:** model, optimizer state, scheduler state, and step are all saved — training is fully resumable
- **Tracking:** Weights & Biases when available, with a local-logging fallback so correctness never depends on W&B

### Hardware

Training runs on **free-tier cloud GPUs**:

| Platform | Role | Specs |
|---|---|---|
| **Kaggle Notebooks** | Primary | 30 GPU hours/week (free tier), single P100 (16GB) or dual T4 (32GB combined), 12-hour max session length, quota resets weekly |
| **Google Colab (free)** | Secondary / overflow | ~15–30 GPU hours/week (dynamically allocated, less predictable), T4 or P100, 12-hour session cap with a shorter idle-disconnect window |

Kaggle is primary because its weekly allocation is a guaranteed floor rather than a dynamically throttled pool, which matters more than raw GPU class at this model size. Both platforms cap individual sessions, so the training loop checkpoints frequently (model, optimizer, scheduler, step) and resumes cleanly across sessions and across platforms — a single Kaggle weekly quota (30 hours) is enough for multiple full passes over the training corpus at this model and dataset size (see [Dataset](#dataset)).

**Hardware-constrained vs. architecture decisions:** if compute pressure requires cuts, the order of reduction is: (1) training steps, (2) batch size, (3) sequence length, (4) model width/depth — architecture is changed last, and any such change will be recorded here explicitly, separate from the fast-iteration rationale in [Model](#model).

---

## Evaluation

Held-out evaluation reports validation loss, test loss, and test perplexity — computed **separately for English, Hindi, and Hinglish** to establish a baseline profile of the model's behavior across languages. Qualitative generation samples are included alongside the quantitative results.

---

## Results

*Pending — to be filled in after training completes (train/validation loss curves, test perplexity by language, final parameter count, hardware/time used).*

---

## Examples

*Pending — representative generation samples per language/category will be added here after the generation system (Day 11) is complete.*

---

## Reproduction

```bash
python scripts/prepare_data.py
python scripts/train_tokenizer.py
python scripts/train.py --config configs/base.yaml
python scripts/evaluate.py --checkpoint <path>
python scripts/generate.py --checkpoint <path>
```

All major hyperparameters live in `configs/base.yaml` — nothing training-relevant is hard-coded in source.

---

## Limitations

- Small model (~15–18M parameters) — not competitive with production LLMs, by design
- Limited dataset (180M tokens total, 60M per language/category) relative to modern pretraining scale — a deliberate, compute-bounded choice, not a data-availability constraint
- Three languages/categories only (English, Hindi, Hinglish)
- Generation quality reflects the model's scale and training budget
- Trained entirely on free-tier GPU compute, which bounds both dataset size and training duration

---

## Explicitly Out of Scope

To keep this an engineering deliverable rather than a research study or product: no authentication, databases, user accounts, conversation memory, RAG, vector databases, agents, tool calling, web search integration, complex frontend, mobile app, multiple model architectures, RLHF, DPO, LoRA experiments, unrelated fine-tuning, or unnecessary microservices/APIs.