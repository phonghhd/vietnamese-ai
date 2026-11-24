"""Hệ thống gắn nhãn trạng thái tính năng, ranh giới Core/Extensions và phụ thuộc công cụ ngoài."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


class TrangThaiTinhNang:
    """Các mức trưởng thành và phân loại tính năng chuẩn hóa trong Vietnamese AI Framework."""

    CORE = "core"
    EXTENSION = "extension"
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    SIMULATED = "simulated"
    EXTERNAL_DEPENDENCY = "external_dependency"


BAN_DO_MODULAR: Dict[str, Dict[str, Any]] = {
    "core": {
        "mo_ta": "Lõi tối thiểu chỉ phụ thuộc NumPy và thư viện chuẩn Python",
        "extra": "core",
        "modules": [
            "vietnamese_ai.core",
            "vietnamese_ai.core.edge",
            "vietnamese_ai.core.preprocessing",
            "vietnamese_ai.utils",
            "vietnamese_ai.preprocessing.numerical",
            "vietnamese_ai.preprocessing.feature_engineering",
            "vietnamese_ai.edge.native_inference",
            "vietnamese_ai.edge.native_build",
            "vietnamese_ai.edge.c_bridge",
        ],
    },
    "extensions": {
        "vision": {
            "module_goc": "vietnamese_ai.vision",
            "namespace": "vietnamese_ai.extensions.vision",
            "extra": "vision",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": ["tesseract-ocr (tùy chọn cho OCR thực tế)"],
        },
        "timeseries": {
            "module_goc": "vietnamese_ai.timeseries",
            "namespace": "vietnamese_ai.extensions.timeseries",
            "extra": "ml",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": [],
        },
        "federated": {
            "module_goc": "vietnamese_ai.federated",
            "namespace": "vietnamese_ai.extensions.federated",
            "extra": "ml",
            "trang_thai": TrangThaiTinhNang.EXPERIMENTAL,
            "cong_cu_ngoai": [],
        },
        "fine_tuning": {
            "module_goc": "vietnamese_ai.fine_tuning",
            "namespace": "vietnamese_ai.extensions.fine_tuning",
            "extra": "dl",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": ["CUDA GPU (khuyến nghị cho PyTorch/Unsloth)"],
        },
        "transformer": {
            "module_goc": "vietnamese_ai.transformer",
            "namespace": "vietnamese_ai.extensions.transformer",
            "extra": "dl",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": [],
        },
        "llm": {
            "module_goc": "vietnamese_ai.llm",
            "namespace": "vietnamese_ai.extensions.llm",
            "extra": "llm",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": ["Ollama / vLLM / llama.cpp (tùy chọn khi chạy local LLM)"],
        },
        "nlp": {
            "module_goc": "vietnamese_ai.nlp",
            "namespace": "vietnamese_ai.extensions.nlp",
            "extra": "vietnamese-nlp",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": ["underthesea (tùy chọn), Rust toolchain (tùy chọn build rust_core)"],
        },
        "mobile": {
            "module_goc": "vietnamese_ai.mobile",
            "namespace": "vietnamese_ai.extensions.mobile",
            "extra": "mobile",
            "trang_thai": TrangThaiTinhNang.EXPERIMENTAL,
            "cong_cu_ngoai": [
                "Android SDK/NDK + TensorFlow/LiteRT (cho .tflite native)",
                "Xcode + coremltools trên macOS (cho .mlmodel native)",
                "PyTorch ExecuTorch toolchain (cho .pte native)",
                "Flutter SDK / React Native CLI (cho ứng dụng di động)",
            ],
        },
        "web3": {
            "module_goc": "vietnamese_ai.edge",
            "namespace": "vietnamese_ai.extensions.web3",
            "extra": "web3",
            "trang_thai": TrangThaiTinhNang.SIMULATED,
            "cong_cu_ngoai": [],
        },
        "robotics": {
            "module_goc": "vietnamese_ai.agents.spatial_tools",
            "namespace": "vietnamese_ai.extensions.robotics",
            "extra": "robotics",
            "trang_thai": TrangThaiTinhNang.EXPERIMENTAL,
            "cong_cu_ngoai": ["ROS2 / Driver phần cứng LiDAR & Robot (khi kết nối phần cứng thật)"],
        },
        "extreme": {
            "module_goc": "vietnamese_ai.extreme",
            "namespace": "vietnamese_ai.extensions.extreme",
            "extra": "core",
            "trang_thai": TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
            "cong_cu_ngoai": ["g++ (GCC/Clang C++ compiler có hỗ trợ OpenMP)"],
        },
        "classic_ml": {
            "module_goc": "vietnamese_ai.models",
            "namespace": "vietnamese_ai.classic_ml",
            "extra": "classic-ml",
            "trang_thai": TrangThaiTinhNang.EXTENSION,
            "cong_cu_ngoai": ["scikit-learn (khuyến nghị cho sản xuất; có fallback NumPy nhẹ)"],
        },
        "legal_rag": {
            "module_goc": "vietnamese_ai.rag.legal",
            "namespace": "vietnamese_ai.legal.rag",
            "extra": "legal-rag",
            "trang_thai": TrangThaiTinhNang.STABLE,
            "cong_cu_ngoai": ["pypdf (đọc PDF), tesseract-ocr (tùy chọn cho OCR ảnh scan)"],
        },
        "crypto_zkml": {
            "module_goc": "vietnamese_ai.crypto",
            "namespace": "vietnamese_ai.zkml",
            "extra": "core",
            "trang_thai": TrangThaiTinhNang.EXPERIMENTAL,
            "cong_cu_ngoai": ["Khuyến nghị kiểm toán mật mã độc lập trước khi triển khai trọng yếu"],
        },
    },
}

_SO_DANG_KY_TINH_NANG: Dict[str, Dict[str, Any]] = {}


def gan_nhan_tinh_nang(
    trang_thai: str,
    mo_ta: str = "",
    phu_thuoc_ngoai: Optional[List[str]] = None,
    nhom_extra: Optional[str] = None,
    la_mo_phong: bool = False,
) -> Callable[[T], T]:
    """Gắn metadata minh bạch về trạng thái trưởng thành, chế độ mô phỏng và công cụ ngoài."""

    def _decorator(doi_tuong: T) -> T:
        ten_day_du = (
            f"{getattr(doi_tuong, '__module__', '')}.{getattr(doi_tuong, '__qualname__', str(doi_tuong))}"
        ).lstrip(".")
        thong_tin: Dict[str, Any] = {
            "ten": ten_day_du,
            "trang_thai": trang_thai,
            "mo_ta": mo_ta,
            "phu_thuoc_ngoai": list(phu_thuoc_ngoai or []),
            "nhom_extra": nhom_extra,
            "la_mo_phong": bool(la_mo_phong or trang_thai == TrangThaiTinhNang.SIMULATED),
        }
        setattr(doi_tuong, "__trang_thai_tinh_nang__", thong_tin)
        _SO_DANG_KY_TINH_NANG[ten_day_du] = thong_tin
        return doi_tuong

    return _decorator


def lay_thong_tin_tinh_nang(doi_tuong: Any) -> Optional[Dict[str, Any]]:
    """Trả về thông tin nhãn trạng thái của một class, phương thức hoặc instance."""
    thong_tin = getattr(doi_tuong, "__trang_thai_tinh_nang__", None)
    if thong_tin is None and not isinstance(doi_tuong, type):
        thong_tin = getattr(type(doi_tuong), "__trang_thai_tinh_nang__", None)
    return dict(thong_tin) if isinstance(thong_tin, dict) else None


def liet_ke_tinh_nang(trang_thai: Optional[str] = None) -> List[Dict[str, Any]]:
    """Liệt kê tất cả tính năng đã gắn nhãn trong phiên runtime hiện tại."""
    ket_qua = list(_SO_DANG_KY_TINH_NANG.values())
    if trang_thai is not None:
        ket_qua = [item for item in ket_qua if item.get("trang_thai") == trang_thai]
    return [dict(item) for item in ket_qua]


def bao_cao_ranh_gioi_he_thong() -> Dict[str, Any]:
    """
    Xuất bản đồ minh bạch phân tách rõ 3 tầng trưởng thành của Vietnamese AI Framework
    (phục vụ doanh nghiệp triển khai thực tế và thẩm định khoa học JOSS/SoftwareX):
    1. `core_production`: Lõi sản xuất 100% thực thi thật, zero-dependency (NumPy + stdlib).
    2. `optional_extensions`: Phần mở rộng sản xuất kích hoạt khi cài đúng nhóm extra `pip install 'vietnamese-ai[...]'`.
    3. `experimental_or_simulated`: Phân hệ thử nghiệm, cần công cụ ngoài chuyên sâu hoặc đang ở chế độ mô phỏng.
    """
    core_production = [
        {"ten": "Micro-Core & Lazy Loader PEP 562", "module": "vietnamese_ai.core / vietnamese_ai.utils"},
        {"ten": "Kho Tri Thức Bền Vững `.vaidb` (SQLite WAL + FTS5 + Vector)", "module": "vietnamese_ai.rag.vaidb_store"},
        {"ten": "RAG Pháp Lý 100 Điều Khoản, Hiệu Lực, Đồ Thị Luật & Rà Soát Hợp Đồng `.docx`", "module": "vietnamese_ai.rag.legal"},
        {"ten": "Ngữ Nghĩa Đồng Nghĩa Tiếng Việt & ONNX CPU Embedder/Reranker", "module": "vietnamese_ai.rag.onnx_semantic"},
        {"ten": "Zero-GPU Edge GGUF Hub (`vai pull`) & 3-Tầng Runtime", "module": "vietnamese_ai.edge.model_hub / native_inference"},
        {"ten": "Remote GPU Burst Flywheel (Modal / SSH / Local) + Eval Gate", "module": "vietnamese_ai.fine_tuning.remote_gpu / eval_gate"},
        {"ten": "Tường Lửa AI, Ẩn PII, Phòng Thủ Dữ Liệu & Khử Độc Ảnh CDR", "module": "vietnamese_ai.security"},
        {"ten": "MCP Server Chuẩn `stdio` & `HTTP/SSE` Cho IDE", "module": "vietnamese_ai.agents.mcp_transport"},
        {"ten": "Bộ Kết Nối Kênh Zalo OA & Telegram Bot", "module": "vietnamese_ai.connectors"},
        {"ten": "Web Studio Dashboard & Chat Đa Phiên Workspace SQLite + RBAC", "module": "vietnamese_ai.ui"},
    ]

    optional_extensions: List[Dict[str, Any]] = []
    experimental_or_simulated: List[Dict[str, Any]] = []
    for ten_nhom, info in BAN_DO_MODULAR["extensions"].items():
        tt = str(info["trang_thai"])
        muc = {
            "nhom": ten_nhom,
            "module": info["module_goc"],
            "extra": info["extra"],
            "trang_thai": tt,
            "cong_cu_ngoai": list(info.get("cong_cu_ngoai") or []),
        }
        if tt in (TrangThaiTinhNang.EXPERIMENTAL, TrangThaiTinhNang.SIMULATED, TrangThaiTinhNang.EXTERNAL_DEPENDENCY):
            experimental_or_simulated.append(muc)
        else:
            optional_extensions.append(muc)

    return {
        "tong_quan": {
            "so_phan_he_core_production": len(core_production),
            "so_phan_he_optional_extensions": len(optional_extensions),
            "so_phan_he_experimental_simulated": len(experimental_or_simulated),
        },
        "core_production": core_production,
        "optional_extensions": optional_extensions,
        "experimental_or_simulated": experimental_or_simulated,
    }


__all__ = [
    "TrangThaiTinhNang",
    "BAN_DO_MODULAR",
    "gan_nhan_tinh_nang",
    "lay_thong_tin_tinh_nang",
    "liet_ke_tinh_nang",
    "bao_cao_ranh_gioi_he_thong",
]
