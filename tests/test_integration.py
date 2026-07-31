"""Integration test: train on tiny data, verify loss decreases."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print("SKIP: torch not installed")
    sys.exit(0)

from sera.config import SeraConfig
from sera.model import SeraModel
from sera.tokenizer import ByteTokenizer
from sera.generate import generate
from sera.train import TextDataset
from torch.utils.data import DataLoader


def test_train_tiny():
    config = SeraConfig(max_seq_len=64)
    model = SeraModel(config)
    tok = ByteTokenizer()

    train_data = "def add(a, b): return a + b\ndef sub(a, b): return a - b\ndef mul(a, b): return a * b\n"

    before = generate(model, tok, "def add(", max_new_tokens=32, temperature=1.0, top_k=40)

    dataset = TextDataset.from_text(train_data, tok, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(5):
        for X, Y in loader:
            logits, _ = model(X)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())

    after = generate(model, tok, "def add(", max_new_tokens=32, temperature=1.0, top_k=40)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"
    print(f"  loss: {losses[0]:.2f} -> {losses[-1]:.2f}")
    print(f"  before: {before[:80]}...")
    print(f"  after:  {after[:80]}...")

    return losses


if __name__ == "__main__":
    print("Integration test: train + generate")
    test_train_tiny()
    print("Done!")
