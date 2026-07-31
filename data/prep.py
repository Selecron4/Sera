"""Prepare the Sera training corpus — pre-tokenized binary stream.

Adapted from SmolLM1/2/3 recipe and MiniCPM-style efficient packing.
Writes data/train.bin and data/val.bin with uint16 token IDs.

Usage:
    python data/prep.py --tokens 500M --code-ratio 0.8 --val-frac 0.005
    python data/prep.py --tokens tiny        # 100K tokens for smoke test
"""

import argparse
import os
import sys
from pathlib import Path

# support being run directly as a script without installing sera
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sera.tokenizer import ByteTokenizer

# lazy import HF datasets — fail with clear message only when needed
DATASETS = {
    "code": ("HuggingFaceTB/smollm-corpus", "python-edu", "train", "text"),
    "code2": ("codeparrot/codeparrot-clean", "default", "train", "content"),
}


def stream(split_ratio: dict, args):
    try:
        from datasets import load_dataset  # noqa
    except ImportError:
        print("warning: 'datasets' not installed — using local fallback corpus")
        return
    for name, (repo, config, split, col) in DATASETS.items():
        ratio = split_ratio.get(name, 0)
        if ratio <= 0:
            continue
        budget = int(args.tokens * ratio)
        print(f"[{name}] {repo} → {budget:,} tokens (~{budget // 4:,} bytes target)")
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
            n_bytes = 0
            for ex in ds:
                if n_bytes >= budget // args.bytes_per_tok:
                    break
                text = ex.get(col, "") or ""
                yield text
                n_bytes += len(text)
        except Exception as e:
            print(f"  warn: {repo} unavailable ({e}); skipping")
            continue


def encode_stream(tokenizer, texts, val_frac: float):
    """Tokenize each text, append to train/val ids by val_frac split."""
    train_ids, val_ids = [], []
    # rotate val every ~1/val_frac docs (avoid clustering)
    n = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special=False)
        # per-doc flush; simple uniform val rotation, not random
        if (n % max(1, int(1 / max(val_frac, 1e-6)))) == 0:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)
        n += 1
    return train_ids, val_ids


def pack(ids: list, out_path: str, dtype=np.uint16):
    arr = np.array(ids, dtype=dtype)
    arr.tofile(out_path)
    return len(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=str, default="tiny", help="500M, 100M, tiny — or absolute int")
    parser.add_argument("--code-ratio", type=float, default=0.8)
    parser.add_argument("--code2-ratio", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.005)
    parser.add_argument("--bytes-per-tok", type=int, default=4)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    # "tiny"/"100M" shortcuts evaluate to ints
    named = {"tiny": 100_000, "10M": 10_000_000, "100M": 100_000_000, "500M": 500_000_000, "1B": 1_000_000_000}
    if args.tokens in named:
        args.tokens = named[args.tokens]
    else:
        args.tokens = int(args.tokens)

    os.makedirs(args.out, exist_ok=True)
    tokenizer = ByteTokenizer()

    split_ratio = {"code": args.code_ratio, "code2": args.code2_ratio}
    texts = list(stream(split_ratio, args))

    if not texts:
        print("No data downloaded — writing a tiny placeholder corpus from AGENTS.md / repo files for smoke test")
        repo_root = Path(__file__).resolve().parent.parent
        texts = []
        for f in ["sera/model.py", "sera/train.py", "README.md", "docs/SPEC.md", "docs/ROADMAP.md"]:
            p = repo_root / f
            if p.exists():
                texts.append(p.read_text(encoding="utf-8"))
        # replicate to reach at least 100K tokens
        joined = "\n\n".join(texts)
        while len(joined) < args.tokens * args.bytes_per_tok:
            joined += joined

    print(f"Tokenizing {len(texts)} docs...")
    train_ids, val_ids = encode_stream(tokenizer, texts, args.val_frac)
    if not val_ids:
        val_ids = train_ids[:len(train_ids) // 200]

    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")
    n_train = pack(train_ids, train_path)
    n_val = pack(val_ids, val_path)
    print(f"train.bin: {n_train:,} tokens ({n_train * 2 // 1024 // 1024} MB) → {train_path}")
    print(f"val.bin:   {n_val:,} tokens ({n_val * 2 // 1024 // 1024} MB) → {val_path}")


if __name__ == "__main__":
    main()