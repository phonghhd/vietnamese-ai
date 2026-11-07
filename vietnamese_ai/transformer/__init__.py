"""Transformer module - Transformer, GPT, RoPE, GQA, Sparse MoE, Tokenizer, PreTrainer (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

__all__ = [
    "MultiHeadAttention",
    "TransformerModel",
    "VietnameseTokenizer",
    "GPTModel",
    "RotaryPositionEmbedding",
    "GroupedQueryAttention",
    "SparseMoELayer",
    "PreTrainer",
    "TextDataset",
]

__getattr__ = tao_lazy_getattr(
    __name__,
    {
        "MultiHeadAttention": ("vietnamese_ai.transformer.attention", "MultiHeadAttention"),
        "TransformerModel": ("vietnamese_ai.transformer.model", "TransformerModel"),
        "VietnameseTokenizer": ("vietnamese_ai.transformer.tokenizer", "VietnameseTokenizer"),
        "GPTModel": ("vietnamese_ai.transformer.gpt_model", "GPTModel"),
        "RotaryPositionEmbedding": (
            "vietnamese_ai.transformer.moe_rope",
            "RotaryPositionEmbedding",
        ),
        "GroupedQueryAttention": (
            "vietnamese_ai.transformer.moe_rope",
            "GroupedQueryAttention",
        ),
        "SparseMoELayer": ("vietnamese_ai.transformer.moe_rope", "SparseMoELayer"),
        "PreTrainer": ("vietnamese_ai.transformer.pretrainer", "PreTrainer"),
        "TextDataset": ("vietnamese_ai.transformer.pretrainer", "TextDataset"),
    },
    globals(),
)
