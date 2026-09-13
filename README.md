# 🗣️ Hermes — Multilingual Code-Mixed Conversational LM

A small language model, **pretrained from scratch** and **aligned with DPO**, that understands and generates natural code-mixed text across **Hindi, Punjabi, Tamil, Kannada, Bengali, and English** — the way people actually talk online in India.

**Hermes** is built by [Your Name] and [Friend's Name] as an end-to-end LLM engineering project: tokenizer → transformer → pretraining → preference alignment → deployment. Nothing here wraps an existing API — the tokenizer, model, training loop, and alignment step are all built and trained by us.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Why This Project](#why-this-project)
- [Tech Stack](#tech-stack)
- [Repo Structure](#repo-structure)
- [Data Sources](#data-sources)
- [Tokenizer Strategy](#tokenizer-strategy)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Compute: Which GPUs and Why](#compute-which-gpus-and-why)
- [DPO Alignment](#dpo-alignment)
- [Evaluation Methodology](#evaluation-methodology)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Team Split](#team-split)
- [Roadmap & Timeline (6 Weeks)](#roadmap--timeline-6-weeks)
- [Results](#results)
- [Demo](#demo)
- [Topics This Project Demonstrates](#topics-this-project-demonstrates)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Large commercial LLMs are trained mostly on monolingual, "clean" text and struggle with **code-mixing** — the way multilingual speakers naturally blend languages mid-sentence (e.g., "yaar this movie is mast, dekh lena free time mein"). This is a small fraction of what big models see, so they handle it poorly.

This project builds a **small transformer LM from scratch**, trains it on real code-mixed conversational text across five Indian languages + English, and then uses **DPO (Direct Preference Optimization)** — a lightweight, modern alternative to full RLHF — to align its behavior toward a specific, chosen goal.

---

## Why This Project

- Multi-script, multi-language tokenization (Devanagari, Gurmukhi, Tamil, Kannada, Bengali scripts + Latin) is a genuinely hard, underexplored problem — most student and even research projects stick to one language
- Code-switching NLP is an active academic research area (see: **LinCE** and **GLUECoS** benchmarks)
- Building from scratch + DPO alignment demonstrates the full LLM pipeline (pretraining → alignment), not just prompting an existing model — this is rare among student portfolios where ~95% of "LLM projects" are just API calls wrapped in LangChain
- Authentic story: built by two multilingual Indian students, for multilingual Indian text

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| DL Framework | PyTorch | Model implementation & training |
| Architecture base | [nanoGPT](https://github.com/karpathy/nanoGPT) | Decoder-only transformer, modified for our vocab/config |
| Tokenizer | Hugging Face `tokenizers` | Custom BPE tokenizer trained on multilingual multi-script corpus |
| Alignment | Hugging Face `trl` (`DPOTrainer`) | Preference-based fine-tuning without a separate reward model or PPO loop |
| Data handling | Hugging Face `datasets`, `pandas` | Loading, cleaning, splitting data |
| Experiment tracking | [Weights & Biases](https://wandb.ai) | Loss curves, run comparisons, per-language metrics |
| Compute | Kaggle Notebooks (primary), Google Colab (secondary) | Free GPU training (T4/P100, 16GB) |
| Demo | Gradio | Side-by-side base vs. aligned model chat comparison |
| Hosting | Hugging Face Hub + Spaces | Free model & demo hosting |
| Version control | GitHub | Shared repo, both collaborators |

---

## Repo Structure

```
hermes/
├── data/
│   ├── raw/                    # scraped/downloaded raw text per language
│   ├── cleaned/                 # cleaned, deduplicated corpus
│   └── preference_pairs/        # chosen/rejected pairs for DPO
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.json           # saved trained tokenizer
├── model/
│   ├── config.py                 # model architecture config
│   ├── model.py                   # transformer implementation (nanoGPT-based)
│   └── checkpoints/              # saved model weights
├── training/
│   ├── pretrain.py
│   ├── dpo_train.py
│   ├── build_preference_data.py
│   └── utils.py
├── eval/
│   ├── perplexity_eval.py
│   ├── winrate_eval.py
│   └── eval_prompts.json
├── demo/
│   └── app.py                     # Gradio app
├── notebooks/
│   └── (Kaggle/Colab notebooks used for training runs)
├── requirements.txt
└── README.md
```

---

## Data Sources

| Language | Primary Source | Supplementary Source |
|---|---|---|
| Hindi | AI4Bharat IndicCorp v2, LinCE/GLUECoS (real code-mixed) | Twitter/Reddit scraping (romanized Hinglish) |
| Punjabi | AI4Bharat IndicCorp v2 | Scraped text (r/punjab-style sources) |
| Tamil | AI4Bharat IndicCorp v2 | YouTube comments, r/chennai-style sources |
| Kannada | AI4Bharat IndicCorp v2 | YouTube comments, r/bangalore-style sources |
| Bengali | AI4Bharat IndicCorp v2 | YouTube comments, r/kolkata-style sources |

- **AI4Bharat** (ai4bharat.iitm.ac.in): primary monolingual corpus source across all five languages. Their **Samanantar** parallel corpus is a fallback option for synthesizing code-mixed data via translation-blending if authentic code-mixed data proves insufficient.
- **Target corpus size**: ~30–50 million tokens total, balanced as evenly as possible across all five language pairs (roughly 6–10M tokens per language pair).
- **Script preference**: romanized (Latin script) code-mixed text is prioritized first — it's far more available online and represents a more novel modeling challenge. Native-script mixing is a stretch goal, not a requirement for v1.
- **YouTube Data API** (free tier) is the practical route for scraping regional-language comment sections, which are naturally code-mixed.

---

## Tokenizer Strategy

This is the most technically distinctive part of the project — document decisions here carefully.

- **Method**: Byte-Pair Encoding (BPE), trained from scratch using Hugging Face's `tokenizers` library on the full combined multilingual corpus (not per-language tokenizers, though see experiment below).
- **Vocab size**: 24,000–40,000 tokens. This is larger than a typical single-language tiny model (~16k) because six scripts (5 Indic + Latin) need enough representation each to avoid excessive fragmentation.
- **Required experiment**: Train and compare (a) one shared multilingual tokenizer vs. (b) separate per-language tokenizers merged together. Measure **fragmentation rate** (average tokens per word) per language. Expected finding: English/Latin script will fragment less than Tamil/Kannada/Bengali — document this asymmetry, it's a well-known real issue in multilingual NLP and shows genuine understanding if you can explain *why* it happens (data imbalance + script complexity) and what it implies for model performance per language.

---

## Model Architecture

| Parameter | Value |
|---|---|
| Total parameters | ~25–40M |
| Transformer layers | 8 decoder blocks |
| Embedding dimension | 384–512 |
| Attention heads | 6–8 |
| Context length | 256–512 tokens |
| Tokenizer vocab size | 24,000–40,000 |
| Positional encoding | Learned or rotary (decide & document choice) |

Rationale: conversational code-mixed text is naturally short, so long context isn't a priority — better to spend the parameter/compute budget on vocabulary coverage across scripts.

---

## Training Details

**Pretraining objective**: standard causal language modeling (next-token prediction).

**Optimizer & schedule**:
- AdamW optimizer
- Cosine learning rate decay with a linear warmup period
- Gradient clipping (to prevent instability from rare/high-loss multilingual batches)
- Mixed precision training (fp16/bf16 via `torch.cuda.amp`) — necessary to fit training comfortably within 16GB GPU memory on Kaggle/Colab

**Logging**: every run logged to a **shared W&B project** — both collaborators log to the same project so all experiments (tokenizer variants, architecture tweaks, hyperparameter choices) are visible and comparable in one place, not scattered across two accounts.

**Per-language tracking**: validation loss and perplexity should be tracked **per language**, not just as one aggregate number — this is what lets you catch and discuss imbalance issues (e.g., if Punjabi/Bengali underperform due to less data).

---

## Compute: Which GPUs and Why

Neither laptop is suited to real training: the Mac (Intel i9) has no CUDA-capable GPU, and the RTX 4050's 6GB VRAM is too tight for a multilingual vocab at reasonable batch size.

- **Primary: Kaggle Notebooks** — free tier gives **30 GPU-hours/week per account**, with P100 (16GB) or T4 x2 (16GB each) access, no credit card required. With **two separate accounts** (one per person), you effectively get **60 GPU-hours/week combined** — split experiments across accounts and sync results via GitHub + shared W&B.
- **Secondary: Google Colab (free tier)** — free T4 (16GB), used as overflow/backup when Kaggle hours run out, or for quick debugging runs. Colab's free sessions can disconnect unpredictably, so it's not the primary choice.
- **Optional fallback: Colab Pro (~$10/month)** — worth it only if you hit a hard compute wall; gives longer, more reliable sessions and occasional A100 access.
- **Local laptops are used only for**: data cleaning/preprocessing (CPU-bound, fine on both machines), tokenizer training (CPU-based, fast even on modest hardware), tiny-scale code sanity checks (e.g., 2-layer model on a few KB of text just to confirm the code runs — never real training), and building the Gradio demo app.

---

## DPO Alignment

**Step 1 — Lock in the alignment goal.** Pick exactly one, and record the decision here once chosen:

> **Chosen alignment goal:** _______________________ (fill in once decided)

Options considered:
1. **Natural code-switching** — prefer coherent, realistic mixing patterns over awkward/garbled mixing
2. **Respectful tone** — prefer polite phrasing over slang/aggressive text
3. **Balanced language use** — prefer responses that don't default to 90%+ English when a user writes primarily in a regional language

**Step 2 — Build preference pairs.**
1. Generate multiple candidate responses from the pretrained base model for a set of prompts
2. For each prompt, label one response "chosen" and one "rejected" according to the alignment goal — do this manually for at least a few hundred pairs (quality over quantity; manual labeling is a legitimate, explainable methodology worth defending in an interview)
3. Optionally scale up using a larger existing LLM as an automatic judge, but always keep a manually verified subset for trustworthiness

**Step 3 — Train.** Use Hugging Face `trl`'s `DPOTrainer` on the pretrained checkpoint with the preference pairs. DPO is used specifically because it optimizes directly on preference pairs without needing a separate reward model or a PPO loop — much cheaper computationally than full RLHF, which matters given the compute budget here.

---

## Evaluation Methodology

| Metric | What it measures | How |
|---|---|---|
| Perplexity (overall + per-language) | Base language modeling quality | Standard perplexity on held-out validation set, before and after DPO |
| Win-rate | Whether DPO-aligned outputs are actually preferred over base outputs | On a held-out eval prompt set, compare base vs. aligned generations — judged either manually or via LLM-as-judge (decide and document which) |
| Qualitative samples | Real demo material | Side-by-side generations across all 5 languages for the same prompts, base vs. aligned |

---

## Setup & Installation

```bash
git clone https://github.com/<your-username>/hermes.git
cd hermes
pip install -r requirements.txt
```

`requirements.txt` should include (pin exact versions once finalized):
```
torch
transformers
trl
tokenizers
datasets
wandb
gradio
pandas
```

---

## How to Run

```bash
# 1. Train tokenizer
python tokenizer/train_tokenizer.py --data data/cleaned/ --vocab_size 32000

# 2. Pretrain model (run on Kaggle/Colab GPU, not locally)
python training/pretrain.py --config model/config.py

# 3. Build preference pairs
python training/build_preference_data.py

# 4. Run DPO alignment
python training/dpo_train.py --base_checkpoint model/checkpoints/pretrained.pt

# 5. Evaluate
python eval/perplexity_eval.py
python eval/winrate_eval.py

# 6. Launch demo
python demo/app.py
```

---

## Team Split

| Task | Person A (RTX 4050, Windows) | Person B (Mac i9, Intel) |
|---|---|---|
| Data collection | Hindi, Punjabi, Tamil | Kannada, Bengali + cleaning pipeline |
| Tokenizer | Train & tune vocab size | Analyze per-language fragmentation rate |
| Model code | Architecture (nanoGPT modifications) | Training loop + W&B logging integration |
| Pretraining | Runs on own Kaggle account | Parallel runs on own Kaggle account |
| Preference data | Generate candidate outputs, label pairs | Cross-check labels, build held-out eval prompt set |
| DPO | Run DPO fine-tuning | Run before/after evaluation |
| Deployment | Build Gradio demo | Push model to HF Hub, deploy Spaces, polish README |

---

## Roadmap & Timeline (6 Weeks)

Check items off as you complete them. If you're past a week's target date and items are still unchecked, that's your "we're behind" signal.

### Week 1 — Data
- [ ] Hindi corpus collected & cleaned
- [ ] Punjabi corpus collected & cleaned
- [ ] Tamil corpus collected & cleaned
- [ ] Kannada corpus collected & cleaned
- [ ] Bengali corpus collected & cleaned
- [ ] Combined corpus deduplicated, train/val split made
- [ ] Alignment goal for DPO locked in and recorded above

### Week 2 — Tokenizer, Architecture, Tiny-Scale Test
- [ ] BPE tokenizer trained on combined corpus
- [ ] Vocab size experiments run & compared (per-language fragmentation analysis done)
- [ ] Model architecture implemented (nanoGPT-based, config finalized)
- [ ] Tiny-scale local run confirms code works end-to-end (no real training yet)

### Week 3 — Pretraining
- [ ] Full pretraining run(s) completed on Kaggle/Colab
- [ ] Training/val loss logged in shared W&B project
- [ ] Per-language perplexity measured on base model
- [ ] Qualitative base-model samples reviewed for coherence issues

### Week 4 — Preference Data
- [ ] Candidate responses generated from base model for eval/preference prompts
- [ ] At least a few hundred preference pairs labeled
- [ ] Held-out evaluation prompt set finalized (separate from training preference pairs)

### Week 5 — DPO & Evaluation
- [ ] DPO training completed using `trl`'s `DPOTrainer`
- [ ] Before/after perplexity comparison done
- [ ] Win-rate evaluation completed (method documented: manual or LLM-as-judge)
- [ ] Qualitative side-by-side samples collected across all 5 languages

### Week 6 — Deployment & Polish
- [ ] Gradio demo built (base vs. aligned side-by-side)
- [ ] Model pushed to Hugging Face Hub
- [ ] Demo deployed on Hugging Face Spaces
- [ ] Results section below filled in with final numbers
- [ ] README fully finalized with screenshots and live demo link

---

## Results

*(Fill in once available)*

| Metric | Base Model | DPO-Aligned Model |
|---|---|---|
| Perplexity (overall) | — | — |
| Perplexity (Hindi) | — | — |
| Perplexity (Punjabi) | — | — |
| Perplexity (Tamil) | — | — |
| Perplexity (Kannada) | — | — |
| Perplexity (Bengali) | — | — |
| Win-rate (aligned preferred) | — | — |

---

## Demo

🔗 Live demo: *(add Hugging Face Spaces link once deployed)*
🔗 Model on Hugging Face Hub: *(add link once uploaded)*

---

## Topics This Project Demonstrates

Useful to keep here for quick reference before interviews:

- Multilingual & multi-script tokenization (BPE, vocabulary design tradeoffs, fragmentation analysis)
- Transformer architecture from first principles (self-attention, positional encoding, causal masking)
- Pretraining dynamics (AdamW, cosine LR scheduling, gradient clipping, mixed precision)
- Code-switching / code-mixed NLP (an active academic research area — LinCE, GLUECoS)
- Preference modeling & DPO (modern, compute-efficient alignment technique, RLHF-adjacent)
- Per-language evaluation and bias/imbalance analysis in multilingual models
- Experiment tracking and reproducibility (shared W&B project)
- Model deployment and hosting (Hugging Face Hub + Spaces, Gradio)
- Parallel experimentation across a two-person team on shared free-tier cloud compute

---

## License

*(Decide: MIT is the standard permissive choice for a portfolio project — add a LICENSE file if so.)*

---

## Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for open Indic language datasets (IndicCorp v2, Samanantar)
- [LinCE](https://ritual.uh.edu/lince/) and [GLUECoS](https://microsoft.github.io/GLUECoS/) benchmarks for code-switched data/evaluation references
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy as an architectural base
- Hugging Face `trl`, `transformers`, `tokenizers`, and `datasets` libraries