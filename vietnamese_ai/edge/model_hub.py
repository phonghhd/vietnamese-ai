"""QuanLyMoHinhEdge (EdgeModelHub) - Trình Quản Lý Tải, Chọn Lượng Tử Theo RAM & Kiểm Định GGUF Tiếng Việt.

Hỗ trợ trải nghiệm `vai pull` chuẩn công nghiệp cho LLM tiếng Việt chạy trực tiếp trên CPU:
- Danh mục mô hình GGUF tiếng Việt chuẩn (Qwen2.5, Qwen3, PhoGPT-4B, Vistral-7B, SeaLLM-7B, Gemma2-2B).
- Tự động đo RAM khả dụng của hệ thống và đề xuất mức lượng tử tối ưu (`Q3_K_M`, `Q4_K_M`, `Q5_K_M`, `Q8_0`).
- Tải tệp GGUF bằng HTTP Range Resume (tiếp tục khi đứt mạng), kiểm tra tiêu đề `GGUF` và mã băm SHA-256.
- Định dạng mẫu hội thoại chuẩn (`chatml`, `phogpt`, `mistral`, `gemma`) cho suy luận tiếng Việt tốc độ cao.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang
from vietnamese_ai.utils.io_utils import LuuTai
from vietnamese_ai.utils.logger import Logger


@dataclass
class BienTheGGUF:
    """Thông tin một biến thể lượng tử GGUF cụ thể của mô hình."""

    muc_luong_tu: str
    tep_gguf: str
    dung_luong_gb: float
    ram_toi_thieu_gb: float
    sha256: Optional[str] = None
    url_truc_tiep: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển thuần JSON."""
        return asdict(self)


@dataclass
class ThongTinMoHinhEdge:
    """Siêu dữ liệu chuẩn của một mô hình LLM tiếng Việt trong danh mục `QuanLyMoHinhEdge`."""

    ma_mo_hinh: str
    ten_hien_thi: str
    repo_hf: str
    kien_truc: str
    so_ty_tham_so: float
    chat_template: str
    mo_ta: str
    bien_the: Dict[str, BienTheGGUF] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển thông tin mô hình sang dict."""
        return {
            "ma_mo_hinh": self.ma_mo_hinh,
            "ten_hien_thi": self.ten_hien_thi,
            "repo_hf": self.repo_hf,
            "kien_truc": self.kien_truc,
            "so_ty_tham_so": self.so_ty_tham_so,
            "chat_template": self.chat_template,
            "mo_ta": self.mo_ta,
            "bien_the": {k: v.to_dict() for k, v in self.bien_the.items()},
        }


DANH_MUC_MO_HINH_VIET: Dict[str, ThongTinMoHinhEdge] = {
    "qwen2.5-1.5b": ThongTinMoHinhEdge(
        ma_mo_hinh="qwen2.5-1.5b",
        ten_hien_thi="Qwen2.5-1.5B-Instruct (Siêu nhẹ CPU 4GB RAM)",
        repo_hf="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        kien_truc="qwen2",
        so_ty_tham_so=1.5,
        chat_template="chatml",
        mo_ta="Mô hình đa ngữ hỗ trợ tiếng Việt mượt mà, tốc độ sinh rất cao trên laptop 4GB-8GB RAM.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1.12, 2.5),
            "Q5_K_M": BienTheGGUF("Q5_K_M", "qwen2.5-1.5b-instruct-q5_k_m.gguf", 1.29, 3.0),
            "Q8_0": BienTheGGUF("Q8_0", "qwen2.5-1.5b-instruct-q8_0.gguf", 1.89, 4.0),
        },
    ),
    "qwen2.5-3b": ThongTinMoHinhEdge(
        ma_mo_hinh="qwen2.5-3b",
        ten_hien_thi="Qwen2.5-3B-Instruct (Cân bằng Laptop 8GB RAM)",
        repo_hf="Qwen/Qwen2.5-3B-Instruct-GGUF",
        kien_truc="qwen2",
        so_ty_tham_so=3.0,
        chat_template="chatml",
        mo_ta="Điểm cân bằng tốt nhất giữa độ thông minh tiếng Việt và tốc độ CPU cho máy văn phòng 8GB RAM.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "qwen2.5-3b-instruct-q4_k_m.gguf", 2.10, 4.5),
            "Q5_K_M": BienTheGGUF("Q5_K_M", "qwen2.5-3b-instruct-q5_k_m.gguf", 2.44, 5.2),
            "Q8_0": BienTheGGUF("Q8_0", "qwen2.5-3b-instruct-q8_0.gguf", 3.55, 7.0),
        },
    ),
    "qwen2.5-7b": ThongTinMoHinhEdge(
        ma_mo_hinh="qwen2.5-7b",
        ten_hien_thi="Qwen2.5-7B-Instruct (Chuyên sâu Tiếng Việt & Pháp lý)",
        repo_hf="Qwen/Qwen2.5-7B-Instruct-GGUF",
        kien_truc="qwen2",
        so_ty_tham_so=7.6,
        chat_template="chatml",
        mo_ta="Suy luận tiếng Việt, tóm tắt và phân tích văn bản pháp lý/doanh nghiệp chính xác cao (16GB RAM).",
        bien_the={
            "Q3_K_M": BienTheGGUF("Q3_K_M", "qwen2.5-7b-instruct-q3_k_m.gguf", 3.81, 6.5),
            "Q4_K_M": BienTheGGUF("Q4_K_M", "qwen2.5-7b-instruct-q4_k_m.gguf", 4.68, 8.0),
            "Q5_K_M": BienTheGGUF("Q5_K_M", "qwen2.5-7b-instruct-q5_k_m.gguf", 5.44, 9.5),
            "Q8_0": BienTheGGUF("Q8_0", "qwen2.5-7b-instruct-q8_0.gguf", 8.10, 14.0),
        },
    ),
    "qwen3-4b": ThongTinMoHinhEdge(
        ma_mo_hinh="qwen3-4b",
        ten_hien_thi="Qwen3-4B (Thế hệ mới hỗ trợ suy luận từng bước)",
        repo_hf="Qwen/Qwen3-4B-GGUF",
        kien_truc="qwen3",
        so_ty_tham_so=4.0,
        chat_template="chatml",
        mo_ta="Mô hình Qwen3 4 tỷ tham số tối ưu suy luận logic và tiếng Việt trên CPU 8GB-16GB RAM.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "Qwen3-4B-Q4_K_M.gguf", 2.50, 5.0),
            "Q5_K_M": BienTheGGUF("Q5_K_M", "Qwen3-4B-Q5_K_M.gguf", 2.90, 5.8),
            "Q8_0": BienTheGGUF("Q8_0", "Qwen3-4B-Q8_0.gguf", 4.28, 8.0),
        },
    ),
    "phogpt-4b": ThongTinMoHinhEdge(
        ma_mo_hinh="phogpt-4b",
        ten_hien_thi="VinAI PhoGPT-4B-Chat (Thuần Việt chuyên sâu)",
        repo_hf="vinai/PhoGPT-4B-Chat-gguf",
        kien_truc="mpt",
        so_ty_tham_so=3.7,
        chat_template="phogpt",
        mo_ta="Mô hình ngôn ngữ thuần tiếng Việt của VinAI tối ưu văn phong và tri thức Việt Nam.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "PhoGPT-4B-Chat-Q4_K_M.gguf", 2.45, 5.0),
            "Q8_0": BienTheGGUF("Q8_0", "PhoGPT-4B-Chat-Q8_0.gguf", 4.10, 7.5),
        },
    ),
    "vistral-7b": ThongTinMoHinhEdge(
        ma_mo_hinh="vistral-7b",
        ten_hien_thi="Vistral-7B-Chat (Tiếng Việt 7B thực chiến)",
        repo_hf="Viet-Mistral/Vistral-7B-Chat-gguf",
        kien_truc="mistral",
        so_ty_tham_so=7.2,
        chat_template="mistral",
        mo_ta="Mô hình Mistral tinh chỉnh chuyên biệt cho tiếng Việt, rất mạnh về RAG hành chính và viết lách.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "ggml-vistral-7B-chat-q4_1.gguf", 4.40, 8.0),
            "Q8_0": BienTheGGUF("Q8_0", "ggml-vistral-7B-chat-q8.gguf", 7.70, 13.0),
        },
    ),
    "seallm-7b": ThongTinMoHinhEdge(
        ma_mo_hinh="seallm-7b",
        ten_hien_thi="SeaLLMs-v3-7B-Chat (Đông Nam Á & Tiếng Việt)",
        repo_hf="SeaLLMs/SeaLLMs-v3-7B-Chat-GGUF",
        kien_truc="qwen2",
        so_ty_tham_so=7.6,
        chat_template="chatml",
        mo_ta="Mô hình chuyên biệt cho ngôn ngữ Đông Nam Á, phản hồi tiếng Việt tự nhiên và ít ảo giác.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf", 4.68, 8.0),
            "Q8_0": BienTheGGUF("Q8_0", "SeaLLMs-v3-7B-Chat-Q8_0.gguf", 8.10, 14.0),
        },
    ),
    "gemma2-2b": ThongTinMoHinhEdge(
        ma_mo_hinh="gemma2-2b",
        ten_hien_thi="Gemma-2-2B-Instruct (Gọn nhẹ đa năng)",
        repo_hf="bartowski/gemma-2-2b-it-GGUF",
        kien_truc="gemma2",
        so_ty_tham_so=2.6,
        chat_template="gemma",
        mo_ta="Mô hình Gemma 2 gọn nhẹ chạy nhanh trên CPU phổ thông, hỗ trợ tốt hỏi đáp và tóm tắt.",
        bien_the={
            "Q4_K_M": BienTheGGUF("Q4_K_M", "gemma-2-2b-it-Q4_K_M.gguf", 1.71, 3.8),
            "Q8_0": BienTheGGUF("Q8_0", "gemma-2-2b-it-Q8_0.gguf", 2.78, 5.5),
        },
    ),
}

@dataclass
class ThongTinMoHinhONNX:
    """Siêu dữ liệu chuẩn cho mô hình nhúng hoặc Cross-Encoder ONNX INT8 tiếng Việt kèm `tokenizer.json`."""

    ma_mo_hinh: str
    ten_hien_thi: str
    repo_hf: str
    loai: str
    tep_onnx: str
    tep_tokenizer: str
    so_chieu: int
    dung_luong_mb: float
    ram_toi_thieu_gb: float
    mo_ta: str

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển thông tin mô hình ONNX sang từ điển JSON."""
        return asdict(self)


DANH_MUC_MO_HINH_ONNX_VIET: Dict[str, ThongTinMoHinhONNX] = {
    "multilingual-e5-small-onnx": ThongTinMoHinhONNX(
        ma_mo_hinh="multilingual-e5-small-onnx",
        ten_hien_thi="Multilingual-E5-Small INT8 ONNX (384 chiều, siêu nhẹ CPU)",
        repo_hf="Xenova/multilingual-e5-small",
        loai="embedding",
        tep_onnx="onnx/model_quantized.onnx",
        tep_tokenizer="tokenizer.json",
        so_chieu=384,
        dung_luong_mb=118.0,
        ram_toi_thieu_gb=0.5,
        mo_ta="Mô hình nhúng đa ngữ hỗ trợ tiếng Việt tốt, lượng tử hóa INT8 chạy trực tiếp trên CPU qua ONNX Runtime.",
    ),
    "bge-m3-onnx": ThongTinMoHinhONNX(
        ma_mo_hinh="bge-m3-onnx",
        ten_hien_thi="BGE-M3 INT8 ONNX (1024 chiều, truy hồi ngữ nghĩa pháp lý/đa ngữ)",
        repo_hf="Xenova/bge-m3",
        loai="embedding",
        tep_onnx="onnx/model_quantized.onnx",
        tep_tokenizer="tokenizer.json",
        so_chieu=1024,
        dung_luong_mb=570.0,
        ram_toi_thieu_gb=1.5,
        mo_ta="Bộ nhúng BGE-M3 lượng tử INT8 kèm `tokenizer.json` cho truy hồi văn bản dài tiếng Việt.",
    ),
    "bge-reranker-v2-m3-onnx": ThongTinMoHinhONNX(
        ma_mo_hinh="bge-reranker-v2-m3-onnx",
        ten_hien_thi="BGE-Reranker-v2-M3 INT8 ONNX (Cross-Encoder sắp xếp lại tiếng Việt)",
        repo_hf="onnx-community/bge-reranker-v2-m3-ONNX",
        loai="reranker",
        tep_onnx="onnx/model_int8.onnx",
        tep_tokenizer="tokenizer.json",
        so_chieu=1,
        dung_luong_mb=570.0,
        ram_toi_thieu_gb=1.5,
        mo_ta="Cross-Encoder INT8 tiếng Việt/đa ngữ dùng trực tiếp cho `BoSapXepLaiONNXTiengViet`.",
    ),
}

_BI_DANH_LUONG_TU: Dict[str, str] = {
    "q3": "Q3_K_M",
    "q3_k_m": "Q3_K_M",
    "q4": "Q4_K_M",
    "q4_k_m": "Q4_K_M",
    "q4_0": "Q4_K_M",
    "q5": "Q5_K_M",
    "q5_k_m": "Q5_K_M",
    "q8": "Q8_0",
    "q8_0": "Q8_0",
}


def do_ram_he_thong_gb() -> Dict[str, Any]:
    """Đo tổng RAM và RAM khả dụng thực tế của hệ điều hành tính theo GB (không bắt buộc cài `psutil`)."""
    tong_bytes = 0
    kha_dung_bytes = 0

    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("MemTotal:"):
                        tong_bytes = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        kha_dung_bytes = int(line.split()[1]) * 1024
            except Exception:
                pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
            if proc.returncode == 0 and proc.stdout.strip().isdigit():
                tong_bytes = int(proc.stdout.strip())
                kha_dung_bytes = int(tong_bytes * 0.65)
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            windll = getattr(ctypes, "windll", None)
            if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                tong_bytes = int(stat.ullTotalPhys)
                kha_dung_bytes = int(stat.ullAvailPhys)
        except Exception:
            pass

    if tong_bytes == 0 and hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            tong_bytes = pages * page_size
            kha_dung_bytes = int(tong_bytes * 0.65)
        except Exception:
            pass

    if tong_bytes == 0:
        tong_bytes = 8 * 1024**3
        kha_dung_bytes = 5 * 1024**3
    if kha_dung_bytes == 0:
        kha_dung_bytes = int(tong_bytes * 0.65)

    return {
        "tong_ram_gb": round(tong_bytes / (1024**3), 2),
        "ram_kha_dung_gb": round(kha_dung_bytes / (1024**3), 2),
        "so_nhan_cpu": os.cpu_count() or 1,
        "kien_truc": platform.machine(),
    }


def dinh_dang_prompt_chat(
    cau_hoi: str,
    he_thong: str = "Bạn là trợ lý AI tiếng Việt chính xác, ngắn gọn và trung thực.",
    chat_template: str = "chatml",
    ngu_canh: Optional[str] = None,
) -> str:
    """Định dạng chuỗi prompt theo mẫu chuẩn của từng họ LLM (`chatml`, `phogpt`, `mistral`, `gemma`)."""
    noi_dung_nguoi_dung = (
        f"Dựa vào ngữ cảnh sau để trả lời chính xác:\n{ngu_canh}\n\nCâu hỏi: {cau_hoi}"
        if ngu_canh
        else cau_hoi
    )
    tpl = chat_template.strip().lower()
    if tpl == "phogpt":
        return f"### Hệ thống:\n{he_thong}\n\n### Câu hỏi:\n{noi_dung_nguoi_dung}\n\n### Trả lời:\n"
    if tpl == "mistral":
        return f"<s>[INST] <<SYS>>\n{he_thong}\n<</SYS>>\n\n{noi_dung_nguoi_dung} [/INST]"
    if tpl == "gemma":
        return (
            f"<start_of_turn>user\n{he_thong}\n\n{noi_dung_nguoi_dung}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    return (
        f"<|im_start|>system\n{he_thong}<|im_end|>\n"
        f"<|im_start|>user\n{noi_dung_nguoi_dung}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _tinh_sha256_tep(duong_dan: Path) -> str:
    """Tính mã băm SHA-256 theo khối 128KB cho tệp GGUF lớn mà không tốn RAM."""
    h = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        while True:
            buf = f.read(131072)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Trình quản lý tải mô hình GGUF tiếng Việt (`vai pull`) tự chọn mức lượng tử theo RAM máy, "
        "hỗ trợ HTTP Range Resume, xác minh tiêu đề GGUF & SHA-256 và ký số toàn vẹn HMAC."
    ),
    nhom_extra="core",
)
class QuanLyMoHinhEdge:
    """
    Trình Quản Lý Mô Hình Edge Tiếng Việt (`EdgeModelHub`):
    - Phân giải bí danh ngắn gọn (`qwen2.5-3b-q4`, `phogpt-4b:q8`, `vistral-7b`, hoặc `hf://org/repo/file.gguf`).
    - Tự động đo RAM hệ thống để chọn biến thể lượng tử tối ưu (`Q8_0` / `Q5_K_M` / `Q4_K_M` / `Q3_K_M`).
    - Tải tệp GGUF có hỗ trợ tiếp tục khi đứt mạng (HTTP Range `206 Partial Content`), kiểm tra SHA-256 và magic bytes `GGUF`.
    """

    def __init__(
        self,
        thu_muc_luu: Optional[Union[str, Path]] = None,
        danh_muc_bo_sung: Optional[Dict[str, ThongTinMoHinhEdge]] = None,
    ) -> None:
        goc_env = os.environ.get("VAI_MODELS_DIR")
        if thu_muc_luu is not None:
            self.thu_muc_luu = Path(thu_muc_luu)
        elif goc_env:
            self.thu_muc_luu = Path(goc_env)
        else:
            self.thu_muc_luu = Path.home() / ".cache" / "vietnamese-ai" / "models"
        self.thu_muc_luu.mkdir(parents=True, exist_ok=True)
        self._tep_chi_muc = self.thu_muc_luu / "models_index.json"
        self.logger = Logger("EdgeModelHub")
        self.danh_muc: Dict[str, ThongTinMoHinhEdge] = dict(DANH_MUC_MO_HINH_VIET)
        self.danh_muc_onnx: Dict[str, ThongTinMoHinhONNX] = dict(DANH_MUC_MO_HINH_ONNX_VIET)
        if danh_muc_bo_sung:
            self.danh_muc.update(danh_muc_bo_sung)

    def _doc_chi_muc(self) -> Dict[str, Dict[str, Any]]:
        if not self._tep_chi_muc.exists():
            return {}
        try:
            data = json.loads(self._tep_chi_muc.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _luu_chi_muc(self, chi_muc: Dict[str, Dict[str, Any]]) -> None:
        self._tep_chi_muc.write_text(
            json.dumps(chi_muc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def phan_tich_ten_mo_hinh(
        self,
        ten_hoac_tag: str,
        muc_luong_tu: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Phân tách tên gọi như `qwen2.5-3b-q4` hoặc `phogpt-4b:q8` thành `(ma_mo_hinh, muc_luong_tu)`."""
        raw = ten_hoac_tag.strip()
        if raw.startswith(("hf://", "http://", "https://")):
            return raw, muc_luong_tu

        ten_lower = raw.lower()
        quant_chon = muc_luong_tu
        if ":" in ten_lower:
            base, tag = ten_lower.rsplit(":", 1)
            ten_lower = base
            if quant_chon is None or quant_chon == "auto":
                quant_chon = _BI_DANH_LUONG_TU.get(tag, tag.upper())

        if ten_lower not in self.danh_muc:
            for hau_to, q_chuan in _BI_DANH_LUONG_TU.items():
                suf = f"-{hau_to}"
                if ten_lower.endswith(suf) and ten_lower[: -len(suf)] in self.danh_muc:
                    ten_lower = ten_lower[: -len(suf)]
                    if quant_chon is None or quant_chon == "auto":
                        quant_chon = q_chuan
                    break

        if quant_chon and quant_chon.lower() != "auto":
            quant_chon = _BI_DANH_LUONG_TU.get(quant_chon.lower(), quant_chon.upper())
        else:
            quant_chon = None
        return ten_lower, quant_chon

    def chon_muc_luong_tu_theo_ram(
        self,
        ten_mo_hinh: str,
        ram_kha_dung_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Tự động chọn mức lượng tử GGUF (`Q8_0` -> `Q5_K_M` -> `Q4_K_M` -> `Q3_K_M`) khớp với RAM khả dụng."""
        ma_goc, quant_chi_dinh = self.phan_tich_ten_mo_hinh(ten_mo_hinh)
        if ma_goc not in self.danh_muc:
            hop_le = ", ".join(sorted(self.danh_muc.keys()))
            raise ValueError(f"Không tìm thấy mô hình '{ten_mo_hinh}' trong danh mục. Hỗ trợ: {hop_le}")

        info = self.danh_muc[ma_goc]
        ram_info = do_ram_he_thong_gb()
        ram_gb = float(ram_info["ram_kha_dung_gb"]) if ram_kha_dung_gb is None else float(ram_kha_dung_gb)

        if quant_chi_dinh and quant_chi_dinh in info.bien_the:
            bt = info.bien_the[quant_chi_dinh]
            du_ram = ram_gb >= bt.ram_toi_thieu_gb
            return {
                "ma_mo_hinh": ma_goc,
                "muc_luong_tu": bt.muc_luong_tu,
                "bien_the": bt.to_dict(),
                "ram_kha_dung_gb": ram_gb,
                "du_ram_khuyen_nghi": du_ram,
                "ly_do": (
                    f"Sử dụng biến thể chỉ định {bt.muc_luong_tu} (yêu cầu {bt.ram_toi_thieu_gb} GB RAM)."
                    if du_ram
                    else f"Cảnh báo: {bt.muc_luong_tu} cần ~{bt.ram_toi_thieu_gb} GB RAM nhưng máy chỉ còn {ram_gb:.1f} GB khả dụng."
                ),
            }

        thu_tu_uu_tien = ("Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M")
        co_san = [q for q in thu_tu_uu_tien if q in info.bien_the]
        for q in co_san:
            bt = info.bien_the[q]
            if ram_gb >= bt.ram_toi_thieu_gb:
                return {
                    "ma_mo_hinh": ma_goc,
                    "muc_luong_tu": bt.muc_luong_tu,
                    "bien_the": bt.to_dict(),
                    "ram_kha_dung_gb": ram_gb,
                    "du_ram_khuyen_nghi": True,
                    "ly_do": (
                        f"Tự chọn {bt.muc_luong_tu} ({bt.dung_luong_gb} GB) phù hợp với {ram_gb:.1f} GB RAM khả dụng."
                    ),
                }

        bt_nhe_nhat = info.bien_the[co_san[-1]]
        return {
            "ma_mo_hinh": ma_goc,
            "muc_luong_tu": bt_nhe_nhat.muc_luong_tu,
            "bien_the": bt_nhe_nhat.to_dict(),
            "ram_kha_dung_gb": ram_gb,
            "du_ram_khuyen_nghi": False,
            "ly_do": (
                f"RAM khả dụng ({ram_gb:.1f} GB) thấp hơn mức khuyến nghị ({bt_nhe_nhat.ram_toi_thieu_gb} GB); "
                f"chọn bản nhẹ nhất {bt_nhe_nhat.muc_luong_tu}."
            ),
        }

    def goi_y_mo_hinh_theo_may(self, ram_kha_dung_gb: Optional[float] = None) -> List[Dict[str, Any]]:
        """Liệt kê toàn bộ danh mục mô hình tiếng Việt kèm biến thể tối ưu và trạng thái cài đặt trên máy hiện tại."""
        ram_info = do_ram_he_thong_gb()
        ram_gb = float(ram_info["ram_kha_dung_gb"]) if ram_kha_dung_gb is None else float(ram_kha_dung_gb)
        chi_muc = self._doc_chi_muc()
        ket_qua: List[Dict[str, Any]] = []
        for ma, info in self.danh_muc.items():
            de_xuat = self.chon_muc_luong_tu_theo_ram(ma, ram_kha_dung_gb=ram_gb)
            da_tai_keys = [
                k for k, v in chi_muc.items() if v.get("ma_mo_hinh") == ma and Path(v.get("duong_dan", "")).is_file()
            ]
            ket_qua.append(
                {
                    "ma_mo_hinh": ma,
                    "ten_hien_thi": info.ten_hien_thi,
                    "so_ty_tham_so": info.so_ty_tham_so,
                    "kien_truc": info.kien_truc,
                    "chat_template": info.chat_template,
                    "muc_luong_tu_de_xuat": de_xuat["muc_luong_tu"],
                    "dung_luong_gb": de_xuat["bien_the"]["dung_luong_gb"],
                    "phu_hop_may": de_xuat["du_ram_khuyen_nghi"],
                    "da_tai_cuc_bo": len(da_tai_keys) > 0,
                    "cac_ban_da_tai": da_tai_keys,
                    "mo_ta": info.mo_ta,
                }
            )
        return ket_qua

    def _xay_dung_thong_so_tai(
        self,
        ten_hoac_nguon: str,
        muc_luong_tu: str = "auto",
        sha256_mong_doi: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Chuyển đổi bí danh mô hình, đường dẫn `hf://` hoặc URL HTTP(S) thành kế hoạch tải chuẩn hóa."""
        raw = ten_hoac_nguon.strip()
        if raw.startswith("hf://"):
            phan = raw[5:].split("/")
            if len(phan) < 3:
                raise ValueError("Cú pháp hf:// phải theo dạng hf://<org>/<repo>/<duong_dan_tep.gguf>.")
            repo = f"{phan[0]}/{phan[1]}"
            tep = "/".join(phan[2:])
            ten_tep = Path(tep).name
            url = f"https://huggingface.co/{repo}/resolve/main/{tep}"
            return {
                "ma_mo_hinh": Path(ten_tep).stem.lower(),
                "muc_luong_tu": "CUSTOM",
                "ten_tep": ten_tep,
                "url": url,
                "repo_hf": repo,
                "chat_template": "chatml",
                "sha256_mong_doi": sha256_mong_doi,
                "ly_do": f"Tải trực tiếp từ HuggingFace repo {repo}.",
            }

        if raw.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(raw)
            ten_tep = Path(parsed.path).name or "custom_model.gguf"
            if not ten_tep.lower().endswith(".gguf"):
                ten_tep += ".gguf"
            return {
                "ma_mo_hinh": Path(ten_tep).stem.lower(),
                "muc_luong_tu": "CUSTOM",
                "ten_tep": ten_tep,
                "url": raw,
                "repo_hf": None,
                "chat_template": "chatml",
                "sha256_mong_doi": sha256_mong_doi,
                "ly_do": f"Tải trực tiếp từ URL {parsed.netloc}.",
            }

        ma_goc, quant_tu_ten = self.phan_tich_ten_mo_hinh(raw, muc_luong_tu=muc_luong_tu)
        if ma_goc not in self.danh_muc:
            hop_le = ", ".join(sorted(self.danh_muc.keys()))
            raise ValueError(
                f"Mô hình '{ten_hoac_nguon}' chưa có trong danh mục. "
                f"Chọn một trong: {hop_le} hoặc dùng hf://<org>/<repo>/<file.gguf>."
            )

        info = self.danh_muc[ma_goc]
        if quant_tu_ten and quant_tu_ten in info.bien_the:
            bt = info.bien_the[quant_tu_ten]
            ly_do = f"Dùng biến thể {bt.muc_luong_tu} do người dùng chỉ định."
        else:
            chon = self.chon_muc_luong_tu_theo_ram(ma_goc)
            bt = info.bien_the[chon["muc_luong_tu"]]
            ly_do = chon["ly_do"]

        url = bt.url_truc_tiep or f"https://huggingface.co/{info.repo_hf}/resolve/main/{bt.tep_gguf}"
        return {
            "ma_mo_hinh": ma_goc,
            "muc_luong_tu": bt.muc_luong_tu,
            "ten_tep": Path(bt.tep_gguf).name,
            "url": url,
            "repo_hf": info.repo_hf,
            "chat_template": info.chat_template,
            "dung_luong_gb": bt.dung_luong_gb,
            "ram_toi_thieu_gb": bt.ram_toi_thieu_gb,
            "sha256_mong_doi": sha256_mong_doi or bt.sha256,
            "ly_do": ly_do,
        }

    @staticmethod
    def xac_minh_tep_gguf(
        duong_dan: Union[str, Path],
        sha256_mong_doi: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Kiểm tra tiêu đề nhị phân `GGUF` và đối chiếu SHA-256 (nếu có) để chống tệp hỏng hoặc bị thay đổi."""
        path = Path(duong_dan)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp GGUF: {path}")
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Tệp '{path.name}' không đúng chuẩn nhị phân GGUF (magic={magic!r}).")
        sha_thuc = _tinh_sha256_tep(path)
        if sha256_mong_doi and sha_thuc.lower() != sha256_mong_doi.strip().lower():
            raise ValueError(
                f"Kiểm định SHA-256 thất bại cho '{path.name}': mong đợi {sha256_mong_doi}, thực tế {sha_thuc}."
            )
        return {
            "hop_le": True,
            "duong_dan": str(path),
            "kich_thuoc_bytes": path.stat().st_size,
            "sha256": sha_thuc,
        }

    def danh_sach_mo_hinh_onnx(self) -> List[Dict[str, Any]]:
        """Liệt kê danh mục mô hình nhúng & reranker ONNX INT8 tiếng Việt kèm `tokenizer.json`."""
        chi_muc = self._doc_chi_muc()
        ket_qua: List[Dict[str, Any]] = []
        for ma, info in self.danh_muc_onnx.items():
            khoa = f"{ma}:INT8"
            item = chi_muc.get(khoa, {})
            da_tai = bool(
                item
                and Path(item.get("duong_dan", "")).is_file()
                and Path(item.get("duong_dan_tokenizer", "")).is_file()
            )
            ket_qua.append(
                {
                    **info.to_dict(),
                    "khoa_mo_hinh": khoa,
                    "da_tai_cuc_bo": da_tai,
                    "duong_dan_onnx": item.get("duong_dan") if da_tai else None,
                    "duong_dan_tokenizer": item.get("duong_dan_tokenizer") if da_tai else None,
                }
            )
        return ket_qua

    def _tai_tep_don_gian(
        self,
        url: str,
        dich: Path,
        hf_token: Optional[str] = None,
        timeout_giay: float = 60.0,
    ) -> int:
        """Tải một tệp HTTP(S) an toàn về tệp tạm rồi di chuyển nguyên tử sang đường dẫn đích."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Giao thức URL không được phép: {parsed.scheme}")
        headers: Dict[str, str] = {"User-Agent": "vietnamese-ai/31.3.0"}
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token and "huggingface.co" in (parsed.netloc or ""):
            headers["Authorization"] = f"Bearer {token}"
        dich.parent.mkdir(parents=True, exist_ok=True)
        part = dich.with_suffix(dich.suffix + ".part")
        req = urllib.request.Request(url, headers=headers, method="GET")
        tong = 0
        with urllib.request.urlopen(req, timeout=timeout_giay) as resp:
            with open(part, "wb") as out_f:
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    tong += len(chunk)
        shutil.move(str(part), str(dich))
        return tong

    def tai_mo_hinh_onnx(
        self,
        ma_mo_hinh: str,
        chay_that: bool = True,
        tai_lai: bool = False,
        hf_token: Optional[str] = None,
        timeout_giay: float = 60.0,
    ) -> Dict[str, Any]:
        """Tải bộ đôi tệp ONNX INT8 và `tokenizer.json` vào thư mục cục bộ cho `BoNhungONNXTiengViet` / `BoSapXepLaiONNXTiengViet`."""
        ma = ma_mo_hinh.strip().lower()
        if ma not in self.danh_muc_onnx:
            hop_le = ", ".join(sorted(self.danh_muc_onnx.keys()))
            raise ValueError(f"Không tìm thấy preset ONNX '{ma_mo_hinh}'. Hỗ trợ: {hop_le}")
        info = self.danh_muc_onnx[ma]
        thu_muc_onnx = self.thu_muc_luu / "onnx" / ma
        dich_onnx = thu_muc_onnx / Path(info.tep_onnx).name
        dich_tok = thu_muc_onnx / Path(info.tep_tokenizer).name
        url_onnx = f"https://huggingface.co/{info.repo_hf}/resolve/main/{info.tep_onnx}"
        url_tok = f"https://huggingface.co/{info.repo_hf}/resolve/main/{info.tep_tokenizer}"
        khoa_luu = f"{ma}:INT8"

        ke_hoach: Dict[str, Any] = {
            "ma_mo_hinh": ma,
            "loai_mo_hinh": info.loai,
            "muc_luong_tu": "INT8",
            "ten_tep": dich_onnx.name,
            "url": url_onnx,
            "url_tokenizer": url_tok,
            "repo_hf": info.repo_hf,
            "so_chieu": info.so_chieu,
            "dung_luong_mb": info.dung_luong_mb,
            "ly_do": f"Tải preset ONNX INT8 ({info.loai}) kèm tokenizer.json chuẩn từ {info.repo_hf}.",
        }
        if not chay_that:
            return {
                "trang_thai": "xem_truoc_dry_run",
                "khoa_mo_hinh": khoa_luu,
                "duong_dan": str(dich_onnx),
                "duong_dan_onnx": str(dich_onnx),
                "duong_dan_tokenizer": str(dich_tok),
                "da_ton_tai": dich_onnx.is_file() and dich_tok.is_file(),
                **ke_hoach,
            }

        if dich_onnx.is_file() and dich_tok.is_file() and not tai_lai:
            json.loads(dich_tok.read_text(encoding="utf-8"))
            sha_onnx = _tinh_sha256_tep(dich_onnx)
            return {
                "trang_thai": "da_co_san",
                "khoa_mo_hinh": khoa_luu,
                "duong_dan": str(dich_onnx),
                "duong_dan_onnx": str(dich_onnx),
                "duong_dan_tokenizer": str(dich_tok),
                "sha256": sha_onnx,
                "kich_thuoc_bytes": dich_onnx.stat().st_size,
                **ke_hoach,
            }

        self._tai_tep_don_gian(url_tok, dich_tok, hf_token=hf_token, timeout_giay=timeout_giay)
        try:
            json.loads(dich_tok.read_text(encoding="utf-8"))
        except Exception as exc:
            if dich_tok.exists():
                dich_tok.unlink()
            raise ValueError(f"Tệp tokenizer.json tải về không hợp lệ JSON: {exc}") from exc

        size_onnx = self._tai_tep_don_gian(url_onnx, dich_onnx, hf_token=hf_token, timeout_giay=timeout_giay)
        if size_onnx <= 0:
            if dich_onnx.exists():
                dich_onnx.unlink()
            raise ValueError("Tệp mô hình ONNX tải về rỗng.")
        sha_onnx = _tinh_sha256_tep(dich_onnx)
        chu_ky = LuuTai.ky_file(str(dich_onnx))

        chi_muc = self._doc_chi_muc()
        chi_muc[khoa_luu] = {
            "ma_mo_hinh": ma,
            "loai_mo_hinh": info.loai,
            "muc_luong_tu": "INT8",
            "ten_tep": dich_onnx.name,
            "duong_dan": str(dich_onnx),
            "duong_dan_tokenizer": str(dich_tok),
            "sha256": sha_onnx,
            "chu_ky_hmac": chu_ky,
            "kich_thuoc_bytes": size_onnx,
            "url": url_onnx,
            "url_tokenizer": url_tok,
        }
        self._luu_chi_muc(chi_muc)
        return {
            "trang_thai": "da_tai_thanh_cong",
            "khoa_mo_hinh": khoa_luu,
            "duong_dan": str(dich_onnx),
            "duong_dan_onnx": str(dich_onnx),
            "duong_dan_tokenizer": str(dich_tok),
            "sha256": sha_onnx,
            "chu_ky_hmac": chu_ky,
            "kich_thuoc_bytes": size_onnx,
            **ke_hoach,
        }

    def tai_mo_hinh(
        self,
        ten_hoac_nguon: str,
        muc_luong_tu: str = "auto",
        chay_that: bool = True,
        tai_lai: bool = False,
        sha256_mong_doi: Optional[str] = None,
        hf_token: Optional[str] = None,
        timeout_giay: float = 60.0,
        ham_tien_do: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Tải mô hình GGUF hoặc preset ONNX INT8 tiếng Việt về bộ nhớ đệm cục bộ:
        - Tự phát hiện preset ONNX INT8 (tải trọn bộ `.onnx` + `tokenizer.json`).
        - Tự chọn mức lượng tử GGUF khớp RAM khi `muc_luong_tu='auto'`.
        - Hỗ trợ tiếp tục tải khi đứt mạng qua HTTP `Range` (`206 Partial Content`).
        - Kiểm tra magic bytes `GGUF`, đối chiếu SHA-256 và ký toàn vẹn HMAC-SHA256 cục bộ.
        """
        if ten_hoac_nguon.strip().lower() in self.danh_muc_onnx:
            return self.tai_mo_hinh_onnx(
                ten_hoac_nguon,
                chay_that=chay_that,
                tai_lai=tai_lai,
                hf_token=hf_token,
                timeout_giay=timeout_giay,
            )
        ke_hoach = self._xay_dung_thong_so_tai(
            ten_hoac_nguon,
            muc_luong_tu=muc_luong_tu,
            sha256_mong_doi=sha256_mong_doi,
        )
        dich = self.thu_muc_luu / ke_hoach["ten_tep"]
        khoa_luu = f"{ke_hoach['ma_mo_hinh']}:{ke_hoach['muc_luong_tu']}"

        if not chay_that:
            return {
                "trang_thai": "xem_truoc_dry_run",
                "khoa_mo_hinh": khoa_luu,
                "duong_dan": str(dich),
                "da_ton_tai": dich.is_file(),
                **ke_hoach,
            }

        if dich.is_file() and not tai_lai:
            xac_minh = self.xac_minh_tep_gguf(dich, sha256_mong_doi=ke_hoach["sha256_mong_doi"])
            return {
                "trang_thai": "da_co_san",
                "khoa_mo_hinh": khoa_luu,
                "duong_dan": str(dich),
                "sha256": xac_minh["sha256"],
                "kich_thuoc_bytes": xac_minh["kich_thuoc_bytes"],
                "da_tiep_tuc_tai": False,
                **ke_hoach,
            }

        parsed_url = urllib.parse.urlparse(ke_hoach["url"])
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"Giao thức URL không được phép: {parsed_url.scheme}")

        part_file = self.thu_muc_luu / f"{ke_hoach['ten_tep']}.part"
        if tai_lai and part_file.exists():
            part_file.unlink()

        da_co_bytes = part_file.stat().st_size if part_file.is_file() else 0
        headers: Dict[str, str] = {"User-Agent": "vietnamese-ai/31.3.0"}
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token and "huggingface.co" in (parsed_url.netloc or ""):
            headers["Authorization"] = f"Bearer {token}"
        if da_co_bytes > 0:
            headers["Range"] = f"bytes={da_co_bytes}-"

        req = urllib.request.Request(ke_hoach["url"], headers=headers, method="GET")
        da_tiep_tuc = False
        sha_tu_header: Optional[str] = None
        try:
            with urllib.request.urlopen(req, timeout=timeout_giay) as resp:
                status = getattr(resp, "status", 200)
                etag = str(resp.headers.get("X-Linked-Etag") or resp.headers.get("ETag") or "").strip('"')
                if len(etag) == 64 and all(c in "0123456789abcdefABCDEF" for c in etag):
                    sha_tu_header = etag.lower()

                content_len_hdr = resp.headers.get("Content-Length")
                content_len = int(content_len_hdr) if content_len_hdr and content_len_hdr.isdigit() else 0

                if status == 206 and da_co_bytes > 0:
                    mode = "ab"
                    da_tiep_tuc = True
                    tong_bytes = da_co_bytes + content_len if content_len > 0 else 0
                    da_ghi = da_co_bytes
                else:
                    mode = "wb"
                    tong_bytes = content_len
                    da_ghi = 0

                with open(part_file, mode) as out_f:
                    while True:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        da_ghi += len(chunk)
                        if ham_tien_do is not None:
                            ham_tien_do(da_ghi, tong_bytes)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part_file.is_file():
                da_tiep_tuc = True
            else:
                raise RuntimeError(
                    f"Lỗi HTTP {exc.code} khi tải mô hình từ '{ke_hoach['url']}': {exc.reason}"
                ) from exc

        sha_kiem_tra = ke_hoach["sha256_mong_doi"] or sha_tu_header
        try:
            xac_minh = self.xac_minh_tep_gguf(part_file, sha256_mong_doi=sha_kiem_tra)
        except Exception:
            if part_file.exists():
                part_file.unlink()
            raise

        shutil.move(str(part_file), str(dich))
        chu_ky = LuuTai.ky_file(str(dich))

        chi_muc = self._doc_chi_muc()
        chi_muc[khoa_luu] = {
            "ma_mo_hinh": ke_hoach["ma_mo_hinh"],
            "muc_luong_tu": ke_hoach["muc_luong_tu"],
            "ten_tep": ke_hoach["ten_tep"],
            "duong_dan": str(dich),
            "sha256": xac_minh["sha256"],
            "chu_ky_hmac": chu_ky,
            "kich_thuoc_bytes": xac_minh["kich_thuoc_bytes"],
            "chat_template": ke_hoach["chat_template"],
            "url": ke_hoach["url"],
        }
        self._luu_chi_muc(chi_muc)

        return {
            "trang_thai": "da_tai_thanh_cong",
            "khoa_mo_hinh": khoa_luu,
            "duong_dan": str(dich),
            "sha256": xac_minh["sha256"],
            "chu_ky_hmac": chu_ky,
            "kich_thuoc_bytes": xac_minh["kich_thuoc_bytes"],
            "da_tiep_tuc_tai": da_tiep_tuc,
            **ke_hoach,
        }

    def danh_sach_da_tai(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả mô hình GGUF đang có trong bộ nhớ đệm cục bộ."""
        chi_muc = self._doc_chi_muc()
        ds: List[Dict[str, Any]] = []
        for khoa, item in chi_muc.items():
            p = Path(item.get("duong_dan", ""))
            if p.is_file():
                ban_sao = dict(item)
                ban_sao["khoa_mo_hinh"] = khoa
                ban_sao["hmac_hop_le"] = LuuTai.xac_thuc_file(str(p))
                ds.append(ban_sao)
        return ds

    def tim_duong_dan_mo_hinh(self, ten_hoac_duong_dan: Union[str, Path]) -> Optional[Path]:
        """Phân giải đường dẫn tệp GGUF trực tiếp hoặc bí danh đã tải (`qwen2.5-3b-q4`, `phogpt-4b`, ...)."""
        p = Path(ten_hoac_duong_dan)
        if p.is_file():
            return p

        raw = str(ten_hoac_duong_dan).strip()
        chi_muc = self._doc_chi_muc()
        if raw in chi_muc and Path(chi_muc[raw].get("duong_dan", "")).is_file():
            return Path(chi_muc[raw]["duong_dan"])

        ma_goc, quant_chon = self.phan_tich_ten_mo_hinh(raw)
        if quant_chon:
            khoa = f"{ma_goc}:{quant_chon}"
            if khoa in chi_muc and Path(chi_muc[khoa].get("duong_dan", "")).is_file():
                return Path(chi_muc[khoa]["duong_dan"])

        for item in chi_muc.values():
            if item.get("ma_mo_hinh") == ma_goc and Path(item.get("duong_dan", "")).is_file():
                return Path(item["duong_dan"])

        truc_tiep = self.thu_muc_luu / raw
        if truc_tiep.is_file():
            return truc_tiep
        return None

    def lay_chat_template(self, ten_hoac_duong_dan: Union[str, Path]) -> str:
        """Tra cứu mẫu hội thoại (`chatml`, `phogpt`, `mistral`, `gemma`) cho mô hình đã chọn."""
        raw = str(ten_hoac_duong_dan).strip()
        chi_muc = self._doc_chi_muc()
        for item in chi_muc.values():
            if raw in (item.get("duong_dan"), item.get("ten_tep"), item.get("ma_mo_hinh")):
                return str(item.get("chat_template", "chatml"))
        ma_goc, _ = self.phan_tich_ten_mo_hinh(raw)
        if ma_goc in self.danh_muc:
            return self.danh_muc[ma_goc].chat_template
        low = raw.lower()
        if "phogpt" in low:
            return "phogpt"
        if "vistral" in low or "mistral" in low:
            return "mistral"
        if "gemma" in low:
            return "gemma"
        return "chatml"

    def xoa_mo_hinh(self, khoa_hoac_ten: str) -> bool:
        """Xóa một mô hình GGUF đã tải cùng chữ ký `.sig` và mục tương ứng trong chỉ mục cục bộ."""
        chi_muc = self._doc_chi_muc()
        target_key: Optional[str] = None
        if khoa_hoac_ten in chi_muc:
            target_key = khoa_hoac_ten
        else:
            ma_goc, quant_chon = self.phan_tich_ten_mo_hinh(khoa_hoac_ten)
            if quant_chon and f"{ma_goc}:{quant_chon}" in chi_muc:
                target_key = f"{ma_goc}:{quant_chon}"
            else:
                for k, v in chi_muc.items():
                    if v.get("ma_mo_hinh") == ma_goc or v.get("ten_tep") == khoa_hoac_ten:
                        target_key = k
                        break

        if target_key is None:
            return False

        item = chi_muc.pop(target_key)
        p = Path(item.get("duong_dan", ""))
        if p.is_file():
            p.unlink()
        sig = p.with_suffix(p.suffix + ".sig")
        if sig.is_file():
            sig.unlink()
        self._luu_chi_muc(chi_muc)
        return True


EdgeModelHub = QuanLyMoHinhEdge
VietnameseModelPuller = QuanLyMoHinhEdge

__all__ = [
    "BienTheGGUF",
    "ThongTinMoHinhEdge",
    "ThongTinMoHinhONNX",
    "DANH_MUC_MO_HINH_VIET",
    "DANH_MUC_MO_HINH_ONNX_VIET",
    "do_ram_he_thong_gb",
    "dinh_dang_prompt_chat",
    "QuanLyMoHinhEdge",
    "EdgeModelHub",
    "VietnameseModelPuller",
]
