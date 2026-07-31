import torch

from .model import SeraModel
from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate(
    model: SeraModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
) -> str:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special=True)], device=next(model.parameters()).device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(output_ids[0].tolist())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sera - minimal code LLM")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--show-config", action="store_true", help="Print model config and exit")
    args = parser.parse_args()

    from .config import SeraConfig

    config = SeraConfig()
    if args.show_config:
        print(f"Sera-{config.total_params // 1_000_000}M")
        print(f"  dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")
        print(f"  kv_heads={config.n_kv_heads}, head_dim={config.head_dim}")
        print(f"  hidden_dim={config.hidden_dim}, max_seq_len={config.max_seq_len}")
        print(f"  vocab={config.vocab_size}, params={config.total_params:,}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeraModel(config).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    tokenizer = ByteTokenizer()
    output = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(output)


if __name__ == "__main__":
    main()
