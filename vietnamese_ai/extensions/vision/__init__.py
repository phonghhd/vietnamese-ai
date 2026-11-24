"""Extension Vision - Thị giác máy tính, OCR và VLM (yêu cầu extra: `vision` / `torch`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "PhanLoaiHinhAnh": ("vietnamese_ai.vision.image_classifier", "PhanLoaiHinhAnh"),
    "TrichXuatVanBanAnh": ("vietnamese_ai.vision.ocr", "TrichXuatVanBanAnh"),
    "MoHinhThiGiacNgonNgu": ("vietnamese_ai.vision.vlm", "MoHinhThiGiacNgonNgu"),
    "VietnameseVLM": ("vietnamese_ai.vision.vlm", "VietnameseVLM"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
