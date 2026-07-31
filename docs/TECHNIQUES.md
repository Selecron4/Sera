# Papers & Techniques for Sera

Papers that informed the Sera architecture, training recipe, and optimizer choices.

## Architecture

### MobileLLM (2024) — arxiv:2402.14905
**Key finding**: Deep-and-thin architectures + GQA + embedding sharing outperform wider shallow ones at sub-billion scale. Our GQA (8/4), NoPE, and embedding tying directly follow this blueprint.

### SmolLM2 (2025) — arxiv:2502.02737
**Key finding**: Data quality + 3-stage training stages (code/web/math ratios evolving). We adopt their NoPE (every 4th layer), weight decay exclusion on embeddings, and WSD scheduler.

### SmolLM3 (2025) — blog post
**Key finding**: Three-stage pretraining with code upsampling from 12%→24% across phases. Our 80/20 code/codeparrot target goes further — pure Python code for a code-first model.

### PyCraft-1 (2026)
**Key finding**: 55M Python code model trained from scratch on a laptop GPU (4GB). Proves small code models are feasible on consumer hardware. Our training recipe design is heavily inspired by its reproducibility and documentation.

### MiniCPM (2024)
**Key finding**: 2.4B-parameter model (non-embedding) beats 7B+ models on reasoning/code. INT4 quant runs on phones. Shows single-GPU fine-tuning is practical at this scale.

### Phi-1 (2023) — arxiv:2306.11644
**Key finding**: Textbook-quality synthetic data beats raw web scrapes for small models. Motivates our use of SmolLM-Corpus (educational Python) + CodeParrot Clean (real-world GitHub) over raw GitHub dumps.

## Optimizers

### Sophia (2023) — arxiv:2305.14342
**Status**: Default optimizer for Sera pretraining.

Second-order optimizer using diagonal Hessian estimate via Gauss-Newton-Bartlett. Update clips `momentum / (ρ·hessian + ε)` to `[-1, 1]`. Achieves 2× speedup over AdamW for LLM pretraining on GPT-2 (125M-1.5B). Our implementation uses gradient-based Hessian (hessian = EMA of gradient²), avoiding the extra forward pass from the paper for simplicity on CPU.

Hyperparameters for Sera-38M:
- `lr=3e-4`, `ρ=0.04`, `β=(0.965, 0.99)`, `k=5` (Hessian update interval)
- Increase ρ to 0.08 if loss spikes during training.
- For larger models: follow the paper's table (reproduced in the section below).

Hyperparameters from the paper for comparable model sizes:
| Model | AdamW lr | Sophia lr | Sophia ρ | Sophia wd |
|-------|----------|-----------|----------|-----------|
| 125M | 6e-4 | 6e-4 | 0.05 | 0.2 |
| 355M | 3e-4 | 7e-4 | 0.08 | 0.2 |
| 770M | 2e-4 | 3e-4 | 0.05 | 0.2 |

### AdamW (2017)
Used for SFT fine-tuning. Simpler, predictable behavior for instruction tuning.
