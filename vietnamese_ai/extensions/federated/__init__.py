"""Extension Federated - Học liên kết phân tán & Federated RLHF (Thử nghiệm)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "HocLienKet": ("vietnamese_ai.federated.learning", "HocLienKet"),
    "FederatedRLHF": ("vietnamese_ai.federated.rlhf", "FederatedRLHF"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
