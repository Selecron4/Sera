from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SeraConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope(seq_len: int, head_dim: int, theta: float = 10000.0, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, config: SeraConfig, use_rope: bool = True):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.use_rope = use_rope  # NoPE: some layers skip RoPE entirely

        qkv_dim = config.n_heads * config.head_dim + 2 * config.n_kv_heads * config.head_dim
        self.qkv_proj = nn.Linear(config.dim, qkv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q_end = self.n_heads * self.head_dim
        k_end = q_end + self.n_kv_heads * self.head_dim
        qkv = self.qkv_proj(x)
        Q = qkv[:, :, :q_end].view(B, T, self.n_heads, self.head_dim)
        K = qkv[:, :, q_end:k_end].view(B, T, self.n_kv_heads, self.head_dim)
        V = qkv[:, :, k_end:].view(B, T, self.n_kv_heads, self.head_dim)

        cache_len = kv_cache[0].shape[1] if kv_cache is not None else 0
        if self.use_rope:
            Q = apply_rotary(Q, cos[cache_len:cache_len + T], sin[cache_len:cache_len + T])
            K = apply_rotary(K, cos[cache_len:cache_len + T], sin[cache_len:cache_len + T])

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            K = torch.cat([k_cache, K], dim=1)
            V = torch.cat([v_cache, V], dim=1)

        n_repeat = self.n_heads // self.n_kv_heads
        K = K.repeat_interleave(n_repeat, dim=2)
        V = V.repeat_interleave(n_repeat, dim=2)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if mask is None and kv_cache is not None:
            kv_len = K.shape[2]
            q_pos = torch.arange(T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            allow = (k_pos <= (kv_len - T) + q_pos)
            mask = torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=(mask is None and kv_cache is None))
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(out), (K[:, ::n_repeat].transpose(1, 2), V[:, ::n_repeat].transpose(1, 2))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: SeraConfig, layer_idx: int = 0):
        super().__init__()
        # SmolLM3 NoPE: every nth layer skips RoPE entirely
        use_rope = (config.nope_every <= 0) or (layer_idx % config.nope_every != 0)
        self.attention = Attention(config, use_rope=use_rope)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)
        self.input_norm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv_cache = self.attention(self.input_norm(x), cos, sin, mask, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.post_attention_norm(x))
        return x, kv_cache


class SeraModel(nn.Module):
    def __init__(self, config: SeraConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        cos, sin = precompute_rope(config.max_seq_len, config.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._norm_paths = set()
        for name, mod in self.named_modules():
            if isinstance(mod, RMSNorm):
                for pname in mod._parameters:
                    self._norm_paths.add(f"{name}.{pname}")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def optimizer_param_groups(self, weight_decay: float = 0.1):
        """SmolLM3: no weight decay on embeddings or RMSNorm weights (stability fix)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "embed" in name or "lm_head" in name or name in self._norm_paths:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _create_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        T = input_ids.shape[1]
        x = self.embed(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
            mask = self._create_causal_mask(T, input_ids.device)
        else:
            mask = None

        new_kv_caches = []
        for layer, cache in zip(self.layers, kv_caches):
            x, new_cache = layer(x, self.rope_cos, self.rope_sin, mask, cache)
            new_kv_caches.append(new_cache)

        x = self.norm(x)
        if self.config.tie_embeddings:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)

        return logits, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        kv_caches = None

        for _ in range(max_new_tokens):
            if kv_caches is None:
                logits, kv_caches = self(generated, kv_caches)
                logits = logits[:, -1, :]
            else:
                logits, kv_caches = self(generated[:, -1:], kv_caches)
                logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                remove_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_mask)
                logits[remove_mask] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated
