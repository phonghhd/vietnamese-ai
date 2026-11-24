"""Extension Robotics & Spatial AI - Động học ngược 3-DOF & Spatial RAG (extra: `robotics`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "SpatialVectorStore": ("vietnamese_ai.rag.spatial_store", "SpatialVectorStore"),
    "giai_dong_hoc_nguoc_3dof": ("vietnamese_ai.agents.spatial_tools", "giai_dong_hoc_nguoc_3dof"),
    "cong_cu_di_chuyen_robot": ("vietnamese_ai.agents.spatial_tools", "cong_cu_di_chuyen_robot"),
    "cong_cu_quet_radar_3d": ("vietnamese_ai.agents.spatial_tools", "cong_cu_quet_radar_3d"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
