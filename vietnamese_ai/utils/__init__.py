"""Utils module - Các công cụ tiện ích & Quản lý phụ thuộc siêu nhẹ (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import DependencyManager, QuanLyPhuThuoc, tao_lazy_getattr

__all__ = [
    "Logger",
    "Metrics",
    "Validator",
    "LuuTai",
    "QuanLyPhuThuoc",
    "DependencyManager",
    "tao_lazy_getattr",
    "TrangThaiTinhNang",
    "BAN_DO_MODULAR",
    "gan_nhan_tinh_nang",
    "lay_thong_tin_tinh_nang",
    "liet_ke_tinh_nang",
    "bao_cao_ranh_gioi_he_thong",
    "PHIEN_BAN_CHUAN",
    "DANH_MUC_GOI_DOC_LAP",
    "PhienBanSemVer",
    "SemVerVersion",
    "QuanLyPhatHanhGoi",
    "ReleasePackageManager",
]

__getattr__ = tao_lazy_getattr(
    __name__,
    {
        "Logger": ("vietnamese_ai.utils.logger", "Logger"),
        "Metrics": ("vietnamese_ai.utils.metrics", "Metrics"),
        "Validator": ("vietnamese_ai.utils.validators", "Validator"),
        "LuuTai": ("vietnamese_ai.utils.io_utils", "LuuTai"),
        "TrangThaiTinhNang": ("vietnamese_ai.utils.feature_status", "TrangThaiTinhNang"),
        "BAN_DO_MODULAR": ("vietnamese_ai.utils.feature_status", "BAN_DO_MODULAR"),
        "gan_nhan_tinh_nang": ("vietnamese_ai.utils.feature_status", "gan_nhan_tinh_nang"),
        "lay_thong_tin_tinh_nang": (
            "vietnamese_ai.utils.feature_status",
            "lay_thong_tin_tinh_nang",
        ),
        "liet_ke_tinh_nang": ("vietnamese_ai.utils.feature_status", "liet_ke_tinh_nang"),
        "bao_cao_ranh_gioi_he_thong": (
            "vietnamese_ai.utils.feature_status",
            "bao_cao_ranh_gioi_he_thong",
        ),
        "PHIEN_BAN_CHUAN": ("vietnamese_ai.utils.versioning", "PHIEN_BAN_CHUAN"),
        "DANH_MUC_GOI_DOC_LAP": ("vietnamese_ai.utils.versioning", "DANH_MUC_GOI_DOC_LAP"),
        "PhienBanSemVer": ("vietnamese_ai.utils.versioning", "PhienBanSemVer"),
        "SemVerVersion": ("vietnamese_ai.utils.versioning", "SemVerVersion"),
        "QuanLyPhatHanhGoi": ("vietnamese_ai.utils.versioning", "QuanLyPhatHanhGoi"),
        "ReleasePackageManager": ("vietnamese_ai.utils.versioning", "ReleasePackageManager"),
    },
    globals(),
)
