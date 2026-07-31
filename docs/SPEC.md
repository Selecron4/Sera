# Sera Spec

## Problem

Open-weight small LLM for code generation that runs on a laptop CPU or integrated GPU.
No discrete GPU, no cloud dependency, minimal footprint.

## Requirements

- Inference on CPU (Intel Core Ultra 7 155U, 16GB DDR5)
- Training on Intel Arc iGPU (Xe-LPG, 8 cores via IPEX XPU) or CPU
- FP32 model < 150 MB, INT8 < 40 MB
- Single hard dependency: `torch >= 2.0`
- Optional: `intel-extension-for-pytorch` for Arc iGPU acceleration
- 2048 token context window
- Generates syntactically plausible code
- Top-k, top-p, temperature sampling
- KV-cached incremental decoding

## Constraints

- **Architecture**: 38.6M param decoder-only transformer
  - 8 layers, 576 dim, 8 heads / 4 KV heads (GQA)
  - SwiGLU FFN (1536 intermediate), RoPE + NoPE every 4th layer, RMSNorm
  - Weight tying (embedding = lm_head), no bias in linears
  - Fused QKV and gate+up projections (42% fewer matmuls)
- **Tokenizer**: byte-level, 259 tokens (256 bytes + BOS/EOS/PAD)
  - No training deps; character-level encoding is a quality ceiling. Upgrade to BPE when throughput matters more than setup simplicity.
- **Vocab head**: `SeraConfig.vocab_size = 16384` (model embedding rows). Sized for a future BPE upgrade; only 259 rows are currently used by `ByteTokenizer`. The remaining ~9.4M params in the embedding (and tied LM head) stay near init noise. See **Known gaps** below.
- **Training graph — pretraining**: SophiaG (β₁=0.965, β₂=0.99, ρ=0.04), WSD scheduler, grad clipping, gradient accumulation
  - No weight decay on embeddings or RMSNorm (SmolLM3 stability fix)
  - bf16 autocast on XPU/CUDA only; fp32 on CPU (CPU bf16 backward is 20x slower)
- **Training graph — SFT**: AdamW (β₁=0.9, β₂=0.95), cosine LR, 1 epoch on ~50M instruction tokens
  - Lower weight decay (0.01), fp32 only
- **Quantization**: PyTorch dynamic INT8 (`torch.quantization.quantize_dynamic`)
- **Data pipeline**: streaming from HF (`--stream`) — no disk download. `StreamDataset` interleaves python-edu (80%) + codeparrot-clean (20%) for pretraining, smol-smoltalk for SFT. Optional `data/prep.py` can pre-tokenize to `data/train.bin` for offline training.
- **SFT data**: `HuggingFaceTB/smol-smoltalk` (instruction following), same byte-level tokenizer

## Data mixture

### Pretraining — Python-only, 80/20

| Source | Config | Ratio | Tokens (500M target) |
|--------|--------|-------|---------------------|
| SmolLM-Corpus | `python-edu` | 80% | 400M |
| CodeParrot Clean | `default` | 20% | 100M |

Streamed from HF via `--stream` flag (no disk). Optional: `data/prep.py --tokens 500M` pre-tokenizes to `data/train.bin` as `uint16` ID stream.

### SFT

| Source | Focus | Tokens |
|--------|-------|--------|
| `smol-smoltalk` (HuggingFaceTB) | General instruction following | ~50M |

## Non-goals

- Distributed / multi-node training
- Structured output (JSON mode, grammars)
- Streaming / server-mode inference
- Multi-language beyond English and code (UTF-8 supported)
- Custom CUDA kernels
- Pre-trained weight distribution (user trains or fine-tunes from checkpoint)

## Out of scope

- BPE / SentencePiece tokenizer (see Known gaps above)
- Model serving infrastructure (API server, batching)
- Fine-tuning adapters (LoRA, QLoRA)
- Mid-training reasoning stage (100B+ tokens needed; post-MVP)

## Known gaps

1. **Vocab mismatch** — `SeraConfig.vocab_size=16384` reserves a future-BPE shape, but the byte tokenizer emits only IDs 0–258. ~9.4M params in rows 259–16383 stay near init. Effect at inference: `torch.multinomial` can sample an out-of-range ID occasionally; `ByteTokenizer.decode` drops IDs ≥ 256, so they vanish in output text. Workaround: bump temperature or constrain output to `< 259` until BPE lands.
2. **No long-context extension** — `max_seq_len=2048`, no YaRN/NTK yet. SmolLM3's free long-context win via increased RoPE theta is post-MVP.
3. **CPU only has 400 tok/s peak** — `torch.compile` was 0.4× slower on this model size; we fuse projections instead. Static quantization / FlashAttention paths for XPU unexplored.
