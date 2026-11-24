"""Gói Học Máy Cổ Điển Tùy Chọn (`vietnamese_ai.classic_ml`, Lazy PEP 562).

Nhóm các lớp bao bọc học máy cổ điển (`PhanLoai`, `HoiQuy`, `PhanCum`, `MangNron`, `MoHinhTapHop`)
thành một gói mở rộng tùy chọn (`pip install vietnamese-ai[classic-ml]`), giữ tách biệt khỏi lõi
tối thiểu và vẫn duy trì 100% tương thích ngược qua `vietnamese_ai.models`.

Khuyến nghị kiến trúc:
- Với các tác vụ học máy bảng chuyên sâu trong sản xuất, khuyến nghị sử dụng trực tiếp `scikit-learn`,
  `xgboost` hoặc `lightgbm` hoặc cài đặt `pip install vietnamese-ai[classic-ml]`.
- Khi chưa cài `scikit-learn`, các lớp vẫn tự động chuyển về fallback thuần NumPy cho học tập/thử nghiệm nhẹ.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "BaseModel": ("vietnamese_ai.models.base", "BaseModel"),
    "PhanLoai": ("vietnamese_ai.models.classifier", "PhanLoai"),
    "HoiQuy": ("vietnamese_ai.models.regression", "HoiQuy"),
    "PhanCum": ("vietnamese_ai.models.clustering", "PhanCum"),
    "MoHinhTapHop": ("vietnamese_ai.models.ensemble", "MoHinhTapHop"),
    "MangNron": ("vietnamese_ai.models.neural_net", "MangNron"),
    "MangSau": ("vietnamese_ai.models.mang_sau", "MangSau"),
}


def thong_tin_goi_classic_ml() -> Dict[str, Any]:
    """Trả về thông tin trạng thái gói học máy cổ điển và khuyến nghị sử dụng `scikit-learn`."""
    co_sklearn = importlib.util.find_spec("sklearn") is not None
    co_pandas = importlib.util.find_spec("pandas") is not None
    return {
        "ten_goi": "vietnamese_ai.classic_ml",
        "extra_cai_dat": "classic-ml",
        "lenh_cai_dat": 'pip install "vietnamese-ai[classic-ml]"',
        "co_scikit_learn": co_sklearn,
        "co_pandas": co_pandas,
        "khuyen_nghi": (
            "Các lớp PhanLoai/HoiQuy/PhanCum/MangNron là lớp bao bọc tiện lợi cho người học tiếng Việt; "
            "với pipeline ML cổ điển quy mô sản xuất nên cài `vietnamese-ai[classic-ml]` hoặc dùng trực tiếp scikit-learn."
        ),
    }


__all__ = list(_LAZY.keys()) + ["thong_tin_goi_classic_ml"]
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
