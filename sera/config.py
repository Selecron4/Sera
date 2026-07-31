from dataclasses import dataclass


@dataclass
class SeraConfig:
    """Model configuration for Sera."""
    vocab_size: int = 16384
    dim: int = 576
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 72
    hidden_dim: int = 1536
    max_seq_len: int = 2048
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    nope_every: int = 4  # SmolLM3: skip RoPE on every nth layer (NoPE) for free long-context win
    weight_decay_off_embed_norm: bool = True  # SmolLM3 stability fix

    @property
    def total_params(self) -> int:
        vocab = self.vocab_size * self.dim
        per_layer = (
            self.dim * self.n_heads * self.head_dim  # Q
            + self.dim * self.n_kv_heads * self.head_dim  # K
            + self.dim * self.n_kv_heads * self.head_dim  # V
            + self.n_heads * self.head_dim * self.dim  # O
            + 2 * self.dim * self.hidden_dim  # gate, up
            + self.hidden_dim * self.dim  # down
        )
        layers = per_layer * self.n_layers
        layer_norms = 2 * self.dim * self.n_layers
        final_norm = self.dim
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.dim
        return vocab + layers + layer_norms + final_norm + lm_head
