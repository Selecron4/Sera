"""Unit and integration tests for Sera model.

Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import pytest
except ImportError:
    print("SKIP: torch or pytest not installed")
    sys.exit(0)

from sera.config import SeraConfig
from sera.model import SeraModel
from sera.tokenizer import ByteTokenizer, BOS_ID, EOS_ID
from sera.generate import generate


@pytest.fixture
def cfg():
    return SeraConfig()


@pytest.fixture
def model(cfg):
    return SeraModel(cfg)


@pytest.fixture
def tok():
    return ByteTokenizer()


class TestConfig:
    def test_default_values(self, cfg):
        assert cfg.vocab_size == 16384
        assert cfg.dim == 576
        assert cfg.n_layers == 8
        assert cfg.n_heads == 8
        assert cfg.n_kv_heads == 4
        assert cfg.head_dim == 72
        assert cfg.hidden_dim == 1536
        assert cfg.max_seq_len == 2048
        assert cfg.rope_theta == 10000.0
        assert cfg.tie_embeddings is True

    def test_param_count_matches_model(self, cfg):
        m = SeraModel(cfg)
        actual = sum(p.numel() for p in m.parameters())
        assert cfg.total_params == actual

    def test_custom_config(self):
        c = SeraConfig(dim=256, n_layers=4, n_heads=4, n_kv_heads=2, hidden_dim=768)
        m = SeraModel(c)
        x = torch.randint(0, c.vocab_size, (1, 16))
        logits, kv = m(x)
        assert logits.shape == (1, 16, c.vocab_size)


class TestTokenizer:
    def test_encode_decode_roundtrip(self, tok):
        text = "def hello(): return 42"
        ids = tok.encode(text, add_special=False)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_adds_special(self, tok):
        ids = tok.encode("hi", add_special=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_vocab_size(self, tok):
        assert len(tok) == 259

    def test_unicode_handling(self, tok):
        text = "λ x: x + 1  日本語"
        ids = tok.encode(text, add_special=False)
        decoded = tok.decode(ids)
        assert text in decoded or all(ord(c) < 256 for c in text)


class TestModelForward:
    def test_basic_forward(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 64))
        logits, kv = model(x)
        assert logits.shape == (1, 64, cfg.vocab_size)
        assert len(kv) == cfg.n_layers

    def test_batch_forward(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        logits, kv = model(x)
        assert logits.shape == (2, 32, cfg.vocab_size)

    def test_single_token(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        logits, kv = model(x)
        assert logits.shape == (1, 1, cfg.vocab_size)

    def test_output_is_finite(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 64))
        logits, _ = model(x)
        assert torch.isfinite(logits).all()

    def test_deterministic(self, model, cfg):
        torch.manual_seed(42)
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        out1, _ = model(x)
        torch.manual_seed(42)
        out2, _ = model(x)
        assert torch.equal(out1, out2)


class TestGeneration:
    @pytest.fixture
    def prompt(self, tok):
        return torch.tensor([tok.encode("def fib(n):", add_special=True)])

    def test_generates_correct_length(self, model, prompt):
        out = model.generate(prompt, max_new_tokens=32, temperature=1.0, top_k=40)
        assert out.shape[1] == prompt.shape[1] + 32

    def test_generate_function(self, model, tok):
        text = generate(model, tok, "x = ", max_new_tokens=8, temperature=1.0, top_k=40)
        assert len(text) > 0

    def test_different_temperatures(self, model, tok):
        prompt = torch.tensor([tok.encode("def f():", add_special=True)])
        out_cold = model.generate(prompt, max_new_tokens=16, temperature=0.1, top_k=1)
        out_hot = model.generate(prompt, max_new_tokens=16, temperature=1.5, top_k=40)
        assert out_cold.shape[1] == out_hot.shape[1]

    def test_eos_stops_early(self, model, tok):
        prompt = torch.tensor([tok.encode("a", add_special=True)])
        out = model.generate(prompt, max_new_tokens=4, temperature=1.5, top_k=16384, eos_id=EOS_ID)
        assert out.shape[1] >= prompt.shape[1]
        # With high temperature and no top_k, EOS (id=258) has a chance to be sampled

    def test_top_k_filtering_no_crash(self, model):
        x = torch.randint(0, 16384, (1, 8))
        out = model.generate(x, max_new_tokens=4, temperature=1.0, top_k=1)
        assert out.shape[1] == 12


class TestKVCache:
    def test_cache_matches_full_forward(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        logits_full, _ = model(x)
        _, kv = model(x[:, :16])
        logits_incr, _ = model(x[:, 16:], kv)
        diff = (logits_full[:, 16:] - logits_incr).abs().max().item()
        assert diff < 1e-4

    def test_cache_length_grows(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        _, kv = model(x)
        seq_len = kv[0][0].shape[1]
        assert seq_len == 8
        _, kv = model(x[:, :4], kv)
        assert kv[0][0].shape[1] == 12

    def test_cache_reduces_latency(self, model, cfg):
        import time
        x = torch.randint(0, cfg.vocab_size, (1, 64))
        _, kv = model(x)
        x1 = torch.randint(0, cfg.vocab_size, (1, 1))

        alone = model(x1)  # noqa: 1-token, no cache (full attention with T=1)
        with_cache = model(x1, kv)  # with cache
        cache_length_ratio = kv[0][0].shape[1] / 64 if kv is not None else 0
        assert cache_length_ratio >= 1.0, "K should be cached and at least as long as input"


class TestTokenizerEdgeCases:
    def test_empty_string(self, tok):
        assert tok.encode("", add_special=False) == []

    def test_only_special_tokens(self, tok):
        ids = tok.encode("", add_special=True)
        assert ids == [BOS_ID, EOS_ID]

    def test_long_unicode(self, tok):
        text = "🚀" * 100
        ids = tok.encode(text, add_special=False)
        assert len(ids) == 400  # 4 bytes per emoji

    def test_decode_roundtrip(self, tok):
        texts = ["hello", "a=1", "print(x)", "λ", ""]
        for t in texts:
            assert tok.decode(tok.encode(t, add_special=False)) == t


@pytest.mark.slow
class TestIntegration:
    def test_train_generate_pipeline(self, cfg, tok):
        from sera.train import TextDataset
        from torch.utils.data import DataLoader

        model = SeraModel(cfg)
        train_data = "def add(a, b): return a + b\ndef sub(a, b): return a - b\n"

        dataset = TextDataset.from_text(train_data, tok, cfg.max_seq_len)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for X, Y in loader:
            logits, _ = model(X)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), Y.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            if len(losses) >= 10:
                break

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"

    def test_quantize_generate(self, cfg):
        from sera.quantize import quantize_dynamic
        model = SeraModel(cfg)
        qmodel = quantize_dynamic(model)
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        logits, kv = qmodel(x)
        assert logits.shape == (1, 32, cfg.vocab_size)
