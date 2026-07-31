# Sera

> **A 50M parameter code-capable small LLM — from scratch, CPU-first, one dependency.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Deps](https://img.shields.io/badge/Deps-torch%20only-lightgrey.svg)]()

Sera is a decoder-only transformer designed for local code generation on a laptop CPU or integrated GPU. Built on techniques from SmolLM2/3, MiniCPM, and MobileLLM. Single dependency: PyTorch.

## Highlights

- **50M parameters** — trained from scratch, not fine-tuned from an existing model
- **CPU-first** — 400 tok/s training on a laptop (Core Ultra 7 155U, 16GB DDR5), no discrete GPU required
- **SophiaG optimizer** — second-order Hessian-based, 2× faster convergence than AdamW
- **GQA + SwiGLU + RoPE + NoPE** — architecture from the latest small-model literature
- **Fused projections** — 42% fewer matmuls per forward pass
- **Byte-level tokenizer** — zero training dependencies, handles any UTF-8 text
- **INT8 quantization** — 147 MB → 36 MB, one function call
- **WSD scheduler** — warmup-stable-decay from SmolLM3
- **24-test suite** — pytest, fixtures, KV-cache correctness checks

## Quick Start

```bash
# Inference
python -m sera --show-config
python -m sera --prompt "def fibonacci(n):"

# Pretrain (stream from HF, no disk used)
python -m sera.train --stream --batch-size 8 --seq-len 512 --grad-accum 4 \
  --device cpu --scheduler wsd --warmup 200 --steps-per-epoch 2000 --epochs 99

# SFT (load checkpoint, stream smol-smoltalk)
python -m sera.train --stream --sft --resume checkpoints/sera_final.pt \
  --batch-size 8 --seq-len 512 --grad-accum 4 --steps-per-epoch 2000 --epochs 1

# Generate from trained model
python -m sera --prompt "def fib" --checkpoint checkpoints/sera_final.pt
```

```python
import torch
from sera.config import SeraConfig
from sera.model import SeraModel
from sera.tokenizer import ByteTokenizer

model = SeraModel(SeraConfig())
tok = ByteTokenizer()
ids = tok.encode("def fib(n):", add_special=True)
out = model.generate(torch.tensor(ids).unsqueeze(0), max_new_tokens=100, temperature=0.7, top_k=40)
print(tok.decode(out[0].tolist()))
```

## Architecture

Sera is a decoder-only transformer with GQA, SwiGLU FFN, RoPE, NoPE (every 4th layer), RMSNorm, and weight tying.

| Hyperparameter | Value |
|---|---|
| Parameters | 38,643,264 |
| Layers | 8 |
| d_model | 576 |
| Q heads / KV heads | 8 / 4 (GQA 2:1) |
| head_dim | 72 |
| d_ff (SwiGLU) | 1,536 |
| Max seq len | 2,048 |
| Tokenizer | ByteTokenizer, 259 tokens |
| Vocab head | 16,384 rows (sized for future BPE upgrade) |
| RoPE / NoPE | θ=10,000, NoPE every 4th layer |
| FP32 size | 147 MB |
| INT8 size | 36 MB |

### Why Each Technique

**GQA (2:1)** — halves KV-cache memory. From Llama 3, Qwen 3, SmolLM3. Critical for CPU inference.

**NoPE (every 4th layer)** — SmolLM3 ablation: skipping RoPE improves long-context without degrading short-context. Free upgrade.

**SwiGLU** — better perplexity per FLOP than GELU. From Llama 2/3, PaLM.

**RMSNorm** — pre-norm, faster than LayerNorm (no mean subtraction, no bias).

**Weight tying** — embedding = lm_head. Saves 9.4M params. From SmolLM, MiniCPM, MobileLLM.

**Fused QKV + gate/up** — 57 → 33 matmuls per forward (42% fewer). 1.25× measured CPU speedup.

**No weight decay on embed + norms** — SmolLM3 stability fix.

**SophiaG** — second-order Hessian-based clipping. Same per-step cost as AdamW (400 tok/s), 2× fewer steps to target perplexity. Uses a clip threshold (rho=0.04) to prevent extreme updates; raise to 0.08 if training loss spikes.

## Training Recipe

```
┌──────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│    Data      │     │  Tokenize   │     │    Pretrain      │     │       SFT        │     │  Evaluate    │
│              │     │             │     │                  │     │                  │     │              │
│ python-edu   │────▶│ ByteTok     │────▶│ SophiaG · WSD    │────▶│ AdamW · Cos LR   │────▶│ PPL on code  │
│   (80%)      │     │ 259 tokens  │     │ lr=3e-4 ρ=0.04  │     │ lr=3e-5          │     │ Generation   │
│ codeparrot   │     │ on-the-fly  │     │ mom .965         │     │ mom .9           │     │ Benchmarks   │
│   (20%)      │     │             │     │ 2000 steps/epoch │     │ 2000 steps/epoch │     │              │
└──────────────┘     └─────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
```

### Phase 1 — Pretraining

Train the base model from scratch on Python code. SophiaG converges 2× faster than AdamW — same cost per step, but needs half the steps.

**Key settings:**
- Optimizer: SophiaG — momentum 0.965, Hessian decay 0.99, clip threshold 0.04
- Learning rate: 3e-4 with WSD scheduler (warm up 200 steps, stay flat, then taper to 0 over the last 10%)
- Weight decay: 0.1 on weights, but **none** on embeddings or layer norms (prevents training instability)
- Data: 80% educational Python + 20% real-world GitHub code, streamed from HuggingFace
- One epoch = 2000 steps at 512 tokens each
- Precision: bf16 if you have an Intel Arc GPU, otherwise plain fp32

**Run:**

```bash
python -m sera.train --stream \
  --batch-size 8 --seq-len 512 --grad-accum 4 \
  --device cpu --scheduler wsd --warmup 200 \
  --steps-per-epoch 2000 --epochs 99
```

**Expected behavior:**
- Loss starts around 9.8 (random), drops to roughly 3.0 within the first 100 steps
- SophiaG's clip threshold (rho=0.04) will clip about 10–30% of parameter updates each step — that's healthy. If it clips more and loss spikes, raise rho to 0.08
- WSD schedule: 200 steps of warmup, then flat at max learning rate, then linear cooldown to zero over the final 10%
- On CPU you'll get roughly 400 tokens per second, about 1.5 million tokens an hour

**When to stop:**
- Smoke (100M tokens, ~3 days): loss plateaus below 2.5, validates pipeline
- MVP (500M tokens, ~2 weeks): loss below 2.0, generates plausible Python syntax
- Full (1B+ tokens): diminishing returns past Chinchilla optimal (~770M for 50M params)

### Phase 2 — SFT

Fine-tune the pretrained model on instruction-following data. Simpler settings — AdamW works better than SophiaG for fine-tuning.

**Key settings:**
- Optimizer: AdamW — momentum 0.9, squared momentum 0.95
- Learning rate: 3e-5 with cosine scheduler (gentle decay over training)
- Weight decay: 0.01
- Data: smol-smoltalk (instruction-following conversations from HuggingFace)
- One epoch = 2000 steps at 512 tokens each

**Run:**

```bash
python -m sera.train --stream --sft \
  --resume checkpoints/sera_final.pt \
  --batch-size 8 --seq-len 512 --grad-accum 4 \
  --steps-per-epoch 2000 --epochs 1
```

**When to stop:**
- 1 epoch (2000 steps, ~4M tokens) is sufficient
- Generation should follow instruction format and produce runnable Python

### Phase 3 — Evaluate

| Metric | How |
|---|---|
| Perplexity | Loss on held-out Python code (`data/val.bin` or separate eval set) |
| Generation quality | Prompt with `def fibonacci(n):` and check output is syntactically valid Python |
| Comparison | vs PyCraft-1 (55M), SmolLM-135M, MiniCPM-2B |

## Dataset

All data from open Python code datasets, streamed via `data/prep.py`.

### Pretraining Data — Python-only, 80/20

| Source | Config | Ratio | Description |
|---|---|---|---|
| [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | `python-edu` | 80% | Educational Python, quality-classified. SmolLM ablation: 3× faster convergence. 4B tokens. |
| [CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | `default` | 20% | Cleaned real-world GitHub Python, Apache-2.0. |

### Other verified Python datasets (available, not in default mix)

| Dataset | Size | Description |
|---|---|---|
| `flytech/python-codes-25k` | 25K | Diverse Python problems |
| `nampdn-ai/tiny-codes` | 120K | LLM-generated exercises (Phi-1) |
| `iamtarun/python_code_instructions_18k_alpaca` | 18K | Instruction+code pairs |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75K | OSS code instructions (SFT) |

## Known Gaps

1. **Vocab mismatch** — model head has 16,384 rows but tokenizer emits only 259 IDs. Rows 259–16383 stay near init noise until BPE lands.
2. **No long-context extension** — 2K max, no YaRN/NTK yet.
3. **Not yet trained** — architecture and pipeline are complete; training is the next phase.
4. **Byte-level tokenizer** — no BPE means no subword compression. Single biggest quality improvement available.

## Reproducing

```bash
python -m sera.train --stream --batch-size 2 --seq-len 128 --grad-accum 2 \
  --device cpu --scheduler wsd --warmup 2 --steps-per-epoch 50 --epochs 1
# Expected: loss decreases 9.79 → ~3.5, WSD lr 0→3e-4→0
```

Or pre-tokenize to disk (optional, for offline training):

```bash
python data/prep.py --tokens 100M
python -m sera.train --data data/train.bin --batch-size 64 --seq-len 1024 --scheduler wsd
```

Realistic CPU budgets (400 tok/s):
100M = ~3 days (smoke) · 500M = ~2 weeks (MVP) · 1B = ~29 days. See `docs/ROADMAP.md`.

## Repository

```
sera/
├── sera/
│   ├── config.py     — SeraConfig
│   ├── model.py      — GQA + SwiGLU + NoPE transformer
│   ├── tokenizer.py  — byte-level tokenizer
│   ├── generate.py   — generation CLI
│   ├── train.py      — training loop (WSD, XPU, StreamDataset, BinDataset)
│   ├── optim.py      — SophiaG optimizer
│   └── quantize.py   — INT8 quantization
├── data/
│   └── prep.py       — corpus builder (HF streaming or uint16 .bin for offline)
├── tests/            — 24 unit tests (pytest)
├── docs/
│   ├── SPEC.md       — specification and constraints
│   ├── ROADMAP.md    — 5-phase plan
│   ├── TASKS.md      — current phase tasks
│   └── TECHNIQUES.md — paper research and optimizer notes
├── skills.lock.yaml  — AI skills manifest
├── shell.nix         — NixOS dev shell
└── pyproject.toml    — pytest + packaging
```

## License

MIT — see [LICENSE](LICENSE).