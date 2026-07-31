# Tasks — Phase 3: Training Data

- [ ] Smoke: `python data/prep.py --tokens 100M` → ~3 days on CPU, ~7 hr on XPU
  - Falls back to local repo text if `datasets` not installed
  - Outputs `data/train.bin` and `data/val.bin` as `uint16` ID streams
- [ ] Verify output quality:
  - `wc -c data/train.bin` should equal `2 × <token count>`
  - Sample decode: `python -c "import numpy as np; from sera.tokenizer import ByteTokenizer; t=ByteTokenizer(); ids=list(np.memmap('data/train.bin',dtype=np.uint16,mode='r')[:256]); print(t.decode(ids)[:200])"`
- [ ] Optionally scale: `--tokens 500M` (~2 wks CPU / ~1.2 day XPU), or `--tokens 1B` (~29 days CPU / ~3 days XPU). See `docs/ROADMAP.md` for full budget table.
- [ ] Write eval script: loss/perplexity on `data/val.bin` after each checkpoint
- [ ] Commit `data/prep.py` and lockfile; `.gitignore` already covers `*.bin`
