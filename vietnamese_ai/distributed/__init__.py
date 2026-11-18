"""Distributed module - Multi-GPU và Distributed Training."""

from vietnamese_ai.distributed.distributed import PhanTanHuanLuyen
from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer
from vietnamese_ai.distributed.offload import OffloadEngine
from vietnamese_ai.distributed.ring_attention import RingAttentionNode

# Alias tiếng Việt
DongCoOffload = OffloadEngine
NutRingAttention = RingAttentionNode

__all__ = [
    "MultiGPUTrainer",
    "PhanTanHuanLuyen",
    "OffloadEngine",
    "DongCoOffload",
    "RingAttentionNode",
    "NutRingAttention",
]
