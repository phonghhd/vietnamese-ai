"""Extension Transformer - Kiến trúc Transformer, GPT, RoPE, GQA, Sparse MoE, PreTrainer."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "MultiHeadAttention": ("vietnamese_ai.transformer.attention", "MultiHeadAttention"),
    "TransformerModel": ("vietnamese_ai.transformer.model", "TransformerModel"),
    "VietnameseTokenizer": ("vietnamese_ai.transformer.tokenizer", "VietnameseTokenizer"),
    "GPTModel": ("vietnamese_ai.transformer.gpt_model", "GPTModel"),
    "RotaryPositionEmbedding": ("vietnamese_ai.transformer.moe_rope", "RotaryPositionEmbedding"),
    "GroupedQueryAttention": ("vietnamese_ai.transformer.moe_rope", "GroupedQueryAttention"),
    "SparseMoELayer": ("vietnamese_ai.transformer.moe_rope", "SparseMoELayer"),
    "PreTrainer": ("vietnamese_ai.transformer.pretrainer", "PreTrainer"),
    "TextDataset": ("vietnamese_ai.transformer.pretrainer", "TextDataset"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
