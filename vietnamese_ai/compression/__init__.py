"""Model Compression - nén và tối ưu mô hình."""

from vietnamese_ai.compression.distillation import HocRutGon
from vietnamese_ai.compression.pruning import CatTiaMoHinh

__all__ = ["HocRutGon", "CatTiaMoHinh"]
