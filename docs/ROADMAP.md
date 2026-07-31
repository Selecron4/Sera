# Sera Roadmap

## Phase 1 — Core (DONE)

Architecture, inference, and basic tests.

- [x] Decoder-only transformer with GQA + SwiGLU + NoPE
- [x] Byte-level tokenizer
- [x] Generation loop with KV cache, top-k/p, temperature
- [x] Training script (WSD, SophiaG, XPU, BinDataset)
- [x] INT8 dynamic quantization
- [x] 24-unit test suite (pytest)
- [x] NixOS dev shell (shell.nix)

## Phase 2 — Scaffolding (DONE)

Project infrastructure, documentation, skills.

- [x] .gitignore, docs/, skills/ directory
- [x] docs/SPEC.md, ROADMAP.md, TASKS.md, TECHNIQUES.md
- [x] skills.lock.yaml reconciled
- [x] Fused QKV + gate/up projections (42% fewer matmuls)
- [x] NoPE every 4th layer (SmolLM3)
- [x] WSD scheduler + SophiaG optimizer
- [x] Weight decay groups (no decay on embed + norms)
- [x] Streaming BinDataset, data/prep.py corpus builder
- [x] StreamDataset (HF streaming, no disk download)
- [x] SFT mode (--sft, --resume, AdamW+Cosine)

## Phase 3 — Pretraining (CURRENT)

Train the base model from scratch.

- [ ] Train: `python -m sera.train --stream --batch-size 8 --seq-len 512 --grad-accum 4 --device cpu --scheduler wsd --steps-per-epoch 2000 --epochs 99`
- [ ] Monitor loss + Sophia ρ clipping rate (target: win_rate 0.1–0.5)
- [ ] Export checkpoints to `checkpoints/`

Realistic CPU budgets (400 tok/s):
| Tokens | CPU time | XPU (~10×) |
|--------|----------|-------------|
| 100M | ~3 days | ~7 hr |
| 500M | ~2 weeks | ~1.5 days |
| 1B | ~29 days | ~3 days |

## Phase 4 — SFT (planned)

Supervised fine-tuning on code instruction datasets.

- [ ] Train: `python -m sera.train --stream --sft --resume checkpoints/sera_final.pt --batch-size 8 --steps-per-epoch 2000 --epochs 1`
- [ ] Evaluate instruction following and code completion quality
- [ ] Compare base vs SFT model on held-out code

## Phase 5 — Evaluate & Optimize

Measure and improve.

- [ ] Perplexity on held-out Python corpus
- [ ] Generation quality (prompt completion accuracy)
- [ ] Compare vs PyCraft-1, SmolLM-135M, MiniCPM-2B
- [ ] Further: int8 static quantization, half-precision on XPU
- [ ] Benchmark: aim for 30+ tok/s on 155U
