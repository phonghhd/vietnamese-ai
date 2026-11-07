"""Serving & Streaming - batch, continuous batching, stream, rate limit, speculative, paged attention."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "MayChuBatch": ("vietnamese_ai.serving.batch_server", "MayChuBatch"),
    "MayChuStream": ("vietnamese_ai.serving.streaming", "MayChuStream"),
    "BoGioiHanTocDo": ("vietnamese_ai.serving.rate_limiter", "BoGioiHanTocDo"),
    "SpeculativeEngine": ("vietnamese_ai.serving.speculative", "SpeculativeEngine"),
    "BlockManager": ("vietnamese_ai.serving.paged_attention", "BlockManager"),
    "PagedAttentionSimulation": ("vietnamese_ai.serving.paged_attention", "PagedAttentionSimulation"),
    "ContinuousBatcher": ("vietnamese_ai.serving.continuous_batching", "ContinuousBatcher"),
    "BoPhanLoLienTuc": ("vietnamese_ai.serving.continuous_batching", "BoPhanLoLienTuc"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
