import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import SeraConfig
from .model import SeraModel
from .tokenizer import ByteTokenizer
from .optim import SophiaG


def _extract_text(example):
    """Extract text from HF examples — handles plain text and messages formats."""
    if isinstance(example, str):
        return example
    text = example.get("text", "")
    if text:
        return text
    # smol-smoltalk has 'messages' — flatten conversation
    messages = example.get("messages", [])
    if messages:
        parts = []
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            if content:
                parts.append(content)
        return "\n".join(parts)
    return ""


class StreamDataset(torch.utils.data.IterableDataset):
    """Streaming dataset from HF — no disk download. Tokenizes on-the-fly."""

    # dataset spec per training phase
    PRETRAIN = [
        ("HuggingFaceTB/smollm-corpus", "python-edu", 0.8),   # educational Python (SmolLM)
        ("codeparrot/codeparrot-clean", "default", 0.2),       # real-world GitHub Python
    ]
    SFT = [
        ("HuggingFaceTB/smol-smoltalk", "default", 1.0),
    ]

    def __init__(self, tokenizer: ByteTokenizer, seq_len: int, spec: str = "pretrain", steps_per_epoch: int = 2000):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.spec = spec

        from datasets import load_dataset
        sources = self.PRETRAIN if spec == "pretrain" else self.SFT
        self.streams = []
        self.weights = []
        for repo, config, weight in sources:
            ds = load_dataset(repo, config, split="train", streaming=True)
            self.streams.append(ds)
            self.weights.append(weight)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        import random, sys

        streams = [iter(ds) for ds in self.streams]
        buffer = []
        count = 0
        n_streams = len(streams)

        print(f"[stream:{self.spec}] epoch start ({self.steps_per_epoch} steps)", file=sys.stderr, flush=True)
        while count < self.steps_per_epoch:
            idx = random.choices(range(n_streams), weights=self.weights)[0]
            try:
                example = next(streams[idx])
            except StopIteration:
                continue

            ids = self.tokenizer.encode(_extract_text(example), add_special=False)
            if count == 0:
                print(f"[stream] first example: {len(ids)} bytes", file=sys.stderr, flush=True)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                y = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]
                yield x, y
                count += 1
                if count % 100 == 0:
                    print(f"[stream] {count}/{self.steps_per_epoch}", file=sys.stderr, flush=True)
                if count >= self.steps_per_epoch:
                    return


class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer: ByteTokenizer, seq_len: int):
        data = Path(path).read_text(encoding="utf-8")
        self._build(data, tokenizer, seq_len)

    @classmethod
    def from_text(cls, text: str, tokenizer: ByteTokenizer, seq_len: int):
        self = cls.__new__(cls)
        self._build(text, tokenizer, seq_len)
        return self

    def _build(self, data: str, tokenizer: ByteTokenizer, seq_len: int):
        ids = tokenizer.encode(data, add_special=False)
        self.seq_len = seq_len
        # single big tensor, OOM guard by seq_len * batch_size
        n = (len(ids) - 1) // seq_len
        self.X = torch.tensor([ids[i * seq_len:(i + 1) * seq_len] for i in range(n)], dtype=torch.long)
        self.Y = torch.tensor([ids[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(n)], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class BinDataset(Dataset):
    """Memory-mapped pre-tokenized stream of uint16 IDs (Karpathy/nanochat style)."""

    def __init__(self, path: str, seq_len: int):
        import numpy as np
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        import numpy as np
        i = idx * self.seq_len
        x = torch.from_numpy(np.array(self.data[i:i + self.seq_len])).long()
        y = torch.from_numpy(np.array(self.data[i + 1:i + 1 + self.seq_len])).long()
        return x, y


def wsd_lr_lambda(step: int, total_steps: int, warmup: int, decay_frac: float = 0.1):
    """Warmup-Stable-Decay (SmolLM3 style). Linear warmup, flat plateau, linear -> 0."""
    if step < warmup:
        return step / max(1, warmup)
    decay_start = int(total_steps * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    decay_steps = max(1, total_steps - decay_start)
    return max(0.0, 1.0 - (step - decay_start) / decay_steps)


def _autocast(device: str):
    """bf16 autocast context — XPU/Arc and CUDA only. CPU backward in bf16 is ~20x slower."""
    if device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex  # noqa
            return torch.autocast("xpu", dtype=torch.bfloat16)
        except ImportError:
            return nullcontext()
    if device == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    # CPU fp32 is faster than CPU bf16 — no autocast on CPU
    return nullcontext()


def train(
    config: SeraConfig,
    data_path: str,
    output_dir: str = "checkpoints",
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 3e-4,
    epochs: int = 1,
    log_every: int = 50,
    save_every: int = 500,
    device: str = "cpu",
    scheduler: str = "wsd",  # "wsd" (SmolLM3) or "cosine"
    warmup: int = 200,
    decay_frac: float = 0.1,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    use_bin: bool = False,
    use_stream: bool = False,
    mode: str = "pretrain",
    steps_per_epoch: int = 2000,
    resume: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = ByteTokenizer()
    model = SeraModel(config).to(device)

    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {resume}")

    if use_stream:
        dataset = StreamDataset(tokenizer, config.max_seq_len, spec=mode, steps_per_epoch=steps_per_epoch)
        loader = DataLoader(dataset, batch_size=batch_size)
    elif data_path and (use_bin or data_path.endswith(".bin")):
        dataset = BinDataset(data_path, config.max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif data_path:
        dataset = TextDataset(data_path, tokenizer, config.max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        raise ValueError("Need --data or --stream")

    # Weight decay groups maintained for SmolLM3 stability fix (no decay on embed + norms)
    param_groups = model.optimizer_param_groups(weight_decay)

    if mode == "sft":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        sched_type = "cosine"
    else:
        optimizer = SophiaG(param_groups, lr=lr, betas=(0.965, 0.99), rho=0.04, weight_decay=0)
        sched_type = scheduler

    steps_per_epoch_val = max(1, len(loader) // grad_accum)
    total_steps = epochs * steps_per_epoch_val
    if sched_type == "wsd":
        sched_fn = lambda s: wsd_lr_lambda(s, total_steps, warmup, decay_frac)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_fn)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    step = 0
    _t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            with _autocast(device):
                logits, _ = model(X)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), Y.view(-1))
            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_every == 0:
                    tok_s = batch_size * grad_accum * config.max_seq_len / (time.time() - _t0)
                    print(f"E{epoch} S{step} loss={loss.item():.4f} lr={lr_scheduler.get_last_lr()[0]:.2e} tok/s={tok_s:.0f}")
                _t0 = time.time()

                if step % save_every == 0:
                    path = os.path.join(output_dir, f"sera_{step}.pt")
                    torch.save(model.state_dict(), path)
                    print(f"  saved {path}")

    path = os.path.join(output_dir, "sera_final.pt")
    torch.save(model.state_dict(), path)
    print(f"Done. Final model: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Sera LLM")
    parser.add_argument("--data", type=str, default=None, help="Path to text file OR pre-tokenized .bin (not needed with --stream)")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--scheduler", type=str, default="wsd", choices=["wsd", "cosine"])
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--decay-frac", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    parser.add_argument("--stream", action="store_true", help="Stream from HF (no disk download)")
    parser.add_argument("--sft", action="store_true", help="SFT mode: instruction data, AdamW, cosine LR")
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for SFT or resume")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    config = SeraConfig(max_seq_len=args.seq_len)

    # Device selection: XPU (Arc) > CUDA > CPU
    if args.device == "auto":
        try:
            import intel_extension_for_pytorch as ipex  # noqa
            if ipex.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Training Sera-{config.total_params // 1_000_000}M on {device} ({args.scheduler} sched)")

    mode = "sft" if args.sft else "pretrain"
    lr = args.lr if not args.sft else 3e-5
    wd = 0.1 if not args.sft else 0.01

    train(
        config, args.data, args.output,
        args.batch_size, args.grad_accum, lr, args.epochs,
        device=device, scheduler=args.scheduler,
        warmup=args.warmup, decay_frac=args.decay_frac,
        weight_decay=wd,
        log_every=args.log_every, save_every=args.save_every,
        use_bin=args.data.endswith(".bin") if args.data else False,
        use_stream=args.stream,
        mode=mode,
        steps_per_epoch=args.steps_per_epoch,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()