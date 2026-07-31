"""
Minimal byte-level tokenizer for code.

Maps each byte to a unique token (0-255) plus special tokens.
No training required, no external dependencies at runtime.
"""


PAD_ID = 256
BOS_ID = 257
EOS_ID = 258


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 259  # 256 bytes + pad/bos/eos
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        bytes_ = bytes(b for b in ids if b < 256)
        return bytes_.decode("utf-8", errors="replace")

    def __len__(self):
        return self.vocab_size
