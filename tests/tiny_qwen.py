"""Build a tiny random Qwen2 checkpoint (safetensors + tokenizer) for residency tests."""
from __future__ import annotations

from pathlib import Path


def torch_backend_usable() -> bool:
    """True when torch, transformers and accelerate can actually run together."""
    try:
        import accelerate  # noqa: F401
        from transformers.utils import is_torch_available
    except ImportError:
        return False
    return bool(is_torch_available())


def make_tiny_qwen(root: Path, layers: int = 8) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = "explain how persistent cognitive state can improve an agent reasoning the of and a to in is".split()
    vocab = {"<unk>": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>")
    config = Qwen2Config(
        vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True,  # like Qwen2.5-1.5B/3B
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(root, safe_serialization=True)
    fast.save_pretrained(root)
    return root
