"""BoNhungONNXTiengViet & BoSapXepLaiONNXTiengViet - Dense Embedding & Cross-Encoder Reranker Tiếng Việt ONNX INT8 + Ngữ Nghĩa Đồng Nghĩa Thuần CPU."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang
from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc
from vietnamese_ai.utils.logger import Logger

TU_DIEN_DONG_NGHIA_VIET: Dict[str, Tuple[str, ...]] = {
    "concept_cham_dut_hdld": (
        "sa thải",
        "sa thai",
        "cho thôi việc",
        "cho thoi viec",
        "đuổi việc",
        "duoi viec",
        "nghỉ việc",
        "nghi viec",
        "chấm dứt hợp đồng lao động",
        "cham dut hop dong lao dong",
        "đơn phương chấm dứt hợp đồng",
        "don phuong cham dut hop dong",
        "báo trước nghỉ việc",
        "thôi việc",
    ),
    "concept_lam_them_gio": (
        "tăng ca",
        "tang ca",
        "làm thêm giờ",
        "lam them gio",
        "làm ngoài giờ",
        "lam ngoai gio",
        "overtime",
        "làm thêm",
    ),
    "concept_thoi_gio_lam_viec": (
        "giờ làm việc",
        "gio lam viec",
        "thời giờ làm việc bình thường",
        "thoi gio lam viec binh thuong",
        "giờ hành chính",
        "gio hanh chinh",
        "tuần làm việc",
    ),
    "concept_thu_viec": (
        "thử việc",
        "thu viec",
        "lương thử việc",
        "luong thu viec",
        "thời gian thử việc",
        "thoi gian thu viec",
    ),
    "concept_du_lieu_ca_nhan": (
        "lộ dữ liệu",
        "lo du lieu",
        "rò rỉ dữ liệu",
        "ro ri du lieu",
        "rò rỉ thông tin",
        "vi phạm bảo vệ dữ liệu cá nhân",
        "vi pham bao ve du lieu ca nhan",
        "dữ liệu cá nhân",
        "du lieu ca nhan",
        "dữ liệu nhạy cảm",
        "thông báo bộ công an",
        "72 giờ",
    ),
    "concept_bhxh_luong_huu": (
        "lương hưu",
        "luong huu",
        "về hưu",
        "ve huu",
        "nghỉ hưu",
        "nghi huu",
        "bảo hiểm xã hội",
        "bao hiem xa hoi",
        "bhxh",
        "đóng bảo hiểm",
        "15 năm",
        "rút bảo hiểm một lần",
    ),
    "concept_quyen_su_dung_dat": (
        "sổ đỏ",
        "so do",
        "sổ hồng",
        "so hong",
        "bán đất",
        "ban dat",
        "sang tên đất",
        "sang ten",
        "chuyển nhượng quyền sử dụng đất",
        "chuyen nhuong quyen su dung dat",
        "giấy chứng nhận quyền sử dụng đất",
        "quyền sử dụng đất",
        "đất đai",
    ),
    "concept_thanh_lap_doanh_nghiep": (
        "mở công ty",
        "mo cong ty",
        "lập doanh nghiệp",
        "lap doanh nghiep",
        "thành lập doanh nghiệp",
        "thanh lap doanh nghiep",
        "đăng ký kinh doanh",
        "dang ky kinh doanh",
        "đăng ký doanh nghiệp",
        "giấy phép kinh doanh",
    ),
    "concept_gop_von_co_dong": (
        "góp vốn",
        "gop von",
        "vốn điều lệ",
        "von dieu le",
        "tài sản góp vốn",
        "cổ đông",
        "co dong",
        "công ty tnhh",
        "công ty cổ phần",
    ),
    "concept_thue_gtgt_hoa_don": (
        "thuế vat",
        "thue vat",
        "vat",
        "thuế giá trị gia tăng",
        "thue gia tri gia tang",
        "gtgt",
        "hóa đơn đỏ",
        "hoa don do",
        "xuất hóa đơn",
        "xuat hoa don",
        "hóa đơn điện tử",
        "hoa don dien tu",
    ),
    "concept_lai_suat_vay": (
        "mượn tiền",
        "muon tien",
        "cho vay",
        "lãi suất vay",
        "lai suat vay",
        "lãi cắt cổ",
        "tín dụng đen",
        "20%/năm",
        "hợp đồng vay tài sản",
    ),
    "concept_giao_dich_dien_tu": (
        "chữ ký số",
        "chu ky so",
        "chữ ký điện tử",
        "chu ky dien tu",
        "ký online",
        "hợp đồng điện tử",
        "hop dong dien tu",
        "thông điệp dữ liệu",
        "giao dịch điện tử",
    ),
    "concept_so_huu_tri_tue": (
        "bản quyền",
        "ban quyen",
        "đăng ký thương hiệu",
        "dang ky thuong hieu",
        "nhãn hiệu",
        "nhan hieu",
        "sáng chế",
        "sang che",
        "quyền tác giả",
        "sở hữu trí tuệ",
    ),
    "concept_dau_thau": (
        "đấu thầu",
        "dau thau",
        "mở thầu",
        "nhà thầu",
        "lựa chọn nhà thầu",
        "chỉ định thầu",
        "hồ sơ mời thầu",
    ),
    "concept_nha_o_bds": (
        "mua nhà",
        "mua nha",
        "chung cư",
        "chung cu",
        "bất động sản",
        "bat dong san",
        "nhà ở hình thành trong tương lai",
        "đặt cọc mua nhà",
        "đặt cọc",
    ),
    "concept_xu_phat_hanh_chinh": (
        "phạt hành chính",
        "phat hanh chinh",
        "xử phạt vi phạm hành chính",
        "xu phat vi pham hanh chinh",
        "biên bản vi phạm",
        "thời hiệu xử phạt",
    ),
}

_RE_TRICH_DAN_SO_HIEU = re.compile(r"\b\d{1,4}/\d{4}/[A-Za-zĐđ0-9\-]+\b", re.UNICODE)
_RE_DIEU_LUAT = re.compile(r"\bđiều\s+\d+\b", re.IGNORECASE | re.UNICODE)


def _chuan_khong_dau(van_ban: str) -> str:
    """Bỏ dấu tiếng Việt và viết thường để chuẩn hóa khớp ngữ nghĩa."""
    thay_d = van_ban.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", thay_d)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")).lower()


def _trich_xuat_khai_niem_viet(van_ban: str) -> List[str]:
    """Ánh xạ văn bản tiếng Việt sang các nhãn khái niệm đồng nghĩa chuẩn hóa."""
    if not van_ban:
        return []
    thuong = van_ban.lower()
    khong_dau = _chuan_khong_dau(van_ban)
    ket_qua: List[str] = []
    for ma_khai_niem, danh_sach_cum in TU_DIEN_DONG_NGHIA_VIET.items():
        for cum in danh_sach_cum:
            if cum in thuong or cum in khong_dau:
                ket_qua.append(ma_khai_niem)
                break
    return ket_qua


def _ma_hoa_token_ids_don_gian(
    van_ban_list: List[str],
    do_dai_toi_da: int = 128,
    kich_thuoc_tu_dien: int = 30522,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo `input_ids` và `attention_mask` dạng int64 phục vụ phiên ONNX khi không có tokenizer rời."""
    batch = len(van_ban_list)
    input_ids = np.zeros((batch, do_dai_toi_da), dtype=np.int64)
    attention_mask = np.zeros((batch, do_dai_toi_da), dtype=np.int64)
    for row_idx, text in enumerate(van_ban_list):
        tokens = re.findall(r"[0-9A-Za-zÀ-ỹ_]+", text.lower())
        seq = [101]
        for tok in tokens[: max(1, do_dai_toi_da - 2)]:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            tid = 1000 + (int.from_bytes(digest, "little") % max(100, kich_thuoc_tu_dien - 1005))
            seq.append(tid)
        seq.append(102)
        n = min(len(seq), do_dai_toi_da)
        input_ids[row_idx, :n] = np.asarray(seq[:n], dtype=np.int64)
        attention_mask[row_idx, :n] = 1
    return input_ids, attention_mask


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ nhúng Dense Tiếng Việt ưu tiên ONNX Runtime INT8/FP32 trên CPU (không cần PyTorch) "
        "và có chế độ ngữ nghĩa đồng nghĩa thuần NumPy zero-dependency làm lớp dự phòng."
    ),
    nhom_extra="export",
)
class BoNhungONNXTiengViet:
    """
    Bộ nhúng văn bản tiếng Việt 2 tầng tối ưu CPU:

    - Tầng 1 (`onnx_runtime_cpu`): Chạy mô hình ONNX INT8/FP32 thật (như `bge-m3`, `vietnamese-bi-encoder`,
      `multilingual-e5-small`) trực tiếp qua `onnxruntime.InferenceSession` (CPUExecutionProvider).
    - Tầng 2 (`semantic_synonym_cpu`): Khi chưa cung cấp tệp `.onnx` hoặc chưa cài `onnxruntime`,
      kết hợp `BoTachTuTiengVietSieuToc`, từ đồng nghĩa pháp lý/doanh nghiệp tiếng Việt
      (`TU_DIEN_DONG_NGHIA_VIET`), unaccented n-gram và Signed Multi-Probe Feature Hashing chuẩn hóa L2.
    """

    def __init__(
        self,
        duong_dan_onnx: Optional[Union[str, Path]] = None,
        duong_dan_tokenizer: Optional[Union[str, Path]] = None,
        kich_thuoc: int = 384,
        so_luong_thread: int = 4,
        do_dai_toi_da: int = 256,
    ) -> None:
        self.duong_dan_onnx = Path(duong_dan_onnx) if duong_dan_onnx else None
        self.duong_dan_tokenizer = Path(duong_dan_tokenizer) if duong_dan_tokenizer else None
        self.kich_thuoc = int(kich_thuoc)
        self.so_luong_thread = max(1, int(so_luong_thread))
        self.do_dai_toi_da = int(do_dai_toi_da)

        self.logger = Logger("BoNhungONNXTiengViet")
        self._tokenizer_viet = BoTachTuTiengVietSieuToc()
        self._hf_tokenizer: Optional[Any] = None
        self._ort_session: Optional[Any] = None
        self._che_do = "semantic_synonym_cpu"
        self._cache: Dict[str, np.ndarray] = {}
        self._max_cache = 4096

        if self.duong_dan_onnx is not None and self.duong_dan_onnx.is_file():
            self._khoi_tao_onnx_session()

    def _khoi_tao_onnx_session(self) -> None:
        """Nạp phiên `onnxruntime.InferenceSession` trên CPU nếu có `onnxruntime`."""
        ort = QuanLyPhuThuoc.nhap_tuy_chon("onnxruntime", "export", "nhúng văn bản ONNX INT8 trên CPU")
        if ort is None or self.duong_dan_onnx is None:
            return
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = self.so_luong_thread
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort_session = ort.InferenceSession(
                str(self.duong_dan_onnx),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            outputs = self._ort_session.get_outputs()
            if outputs and outputs[0].shape and isinstance(outputs[0].shape[-1], int):
                self.kich_thuoc = int(outputs[0].shape[-1])
            self._che_do = "onnx_runtime_cpu"

            if self.duong_dan_tokenizer is not None and self.duong_dan_tokenizer.is_file():
                tok_mod = QuanLyPhuThuoc.nhap_tuy_chon("tokenizers", "dl", "tách token ONNX")
                if tok_mod is not None:
                    self._hf_tokenizer = tok_mod.Tokenizer.from_file(str(self.duong_dan_tokenizer))
        except Exception as exc:
            self.logger.warning(f"Không nạp được phiên ONNX ({exc}); chuyển về chế độ semantic_synonym_cpu.")
            self._ort_session = None
            self._che_do = "semantic_synonym_cpu"

    @staticmethod
    def luong_tu_hoa_int8(
        duong_dan_onnx_goc: Union[str, Path],
        duong_dan_int8: Union[str, Path],
    ) -> Dict[str, Any]:
        """Lượng tử hóa động mô hình ONNX FP32 sang INT8 để tăng tốc suy luận CPU."""
        goc = Path(duong_dan_onnx_goc)
        dich = Path(duong_dan_int8)
        if not goc.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp ONNX gốc: {goc}")

        ort_quant = QuanLyPhuThuoc.nhap_tuy_chon(
            "onnxruntime.quantization",
            "export",
            "lượng tử hóa động mô hình ONNX sang INT8",
        )
        if ort_quant is None:
            return {
                "thanh_cong": False,
                "ly_do": "Chưa cài `onnxruntime` (cài bằng: pip install 'vietnamese-ai[export]').",
                "duong_dan_goc": str(goc),
            }

        dich.parent.mkdir(parents=True, exist_ok=True)
        ort_quant.quantize_dynamic(
            model_input=str(goc),
            model_output=str(dich),
            weight_type=ort_quant.QuantType.QInt8,
        )
        size_goc = goc.stat().st_size
        size_moi = dich.stat().st_size
        return {
            "thanh_cong": True,
            "duong_dan_int8": str(dich),
            "kich_thuoc_goc_bytes": size_goc,
            "kich_thuoc_int8_bytes": size_moi,
            "ty_le_nen": round(size_goc / max(size_moi, 1), 2),
        }

    @property
    def che_do(self) -> str:
        """Trả về chế độ nhúng hiện hành (`onnx_runtime_cpu` hoặc `semantic_synonym_cpu`)."""
        return self._che_do

    def _nhung_onnx_batch(self, texts: List[str]) -> np.ndarray:
        """Chạy suy luận lô trên phiên `onnxruntime.InferenceSession` + Mean Pooling + chuẩn hóa L2."""
        assert self._ort_session is not None
        input_names = [inp.name for inp in self._ort_session.get_inputs()]

        if self._hf_tokenizer is not None:
            encodings = self._hf_tokenizer.encode_batch(texts)
            max_len = min(self.do_dai_toi_da, max((len(e.ids) for e in encodings), default=1))
            input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
            for i, enc in enumerate(encodings):
                n = min(len(enc.ids), max_len)
                input_ids[i, :n] = enc.ids[:n]
                attention_mask[i, :n] = enc.attention_mask[:n]
        else:
            input_ids, attention_mask = _ma_hoa_token_ids_don_gian(texts, do_dai_toi_da=self.do_dai_toi_da)

        feed: Dict[str, Any] = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        if not feed and input_names:
            feed[input_names[0]] = input_ids

        raw_out = np.asarray(self._ort_session.run(None, feed)[0], dtype=np.float32)
        if raw_out.ndim == 3:
            mask_exp = attention_mask[:, :, None].astype(np.float32)
            summed = np.sum(raw_out * mask_exp, axis=1)
            counts = np.maximum(np.sum(mask_exp, axis=1), 1e-6)
            pooled = summed / counts
        elif raw_out.ndim == 2:
            pooled = raw_out
        else:
            pooled = raw_out.reshape(len(texts), -1)

        norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-10)
        return (pooled / norms).astype(np.float32)

    def _nhung_ngu_nghia_viet(self, van_ban: str) -> np.ndarray:
        """Nhúng văn bản tiếng Việt mở rộng khái niệm đồng nghĩa + tách từ ghép + không dấu."""
        if van_ban in self._cache:
            return self._cache[van_ban].copy()

        dim = self.kich_thuoc
        vec = np.zeros(dim, dtype=np.float32)

        tokens = self._tokenizer_viet.tach_tu(van_ban, noi_tu_ghep=True)
        khong_dau_tokens = re.findall(r"[0-9a-z_]+", _chuan_khong_dau(van_ban))
        khai_niem = _trich_xuat_khai_niem_viet(van_ban)

        features: Counter[str] = Counter()
        weights: Dict[str, float] = {}

        for tok in tokens:
            t_low = tok.lower()
            if len(t_low) >= 2:
                key = f"w:{t_low}"
                features[key] += 1
                weights[key] = 2.8 if "_" in t_low else 2.2

        for tok_kd in khong_dau_tokens:
            if len(tok_kd) >= 2:
                key = f"kd:{tok_kd}"
                features[key] += 1
                weights[key] = 1.6

        for i in range(len(tokens) - 1):
            key = f"bi:{tokens[i].lower()}_{tokens[i + 1].lower()}"
            features[key] += 1
            weights[key] = 1.9

        for concept in khai_niem:
            key = f"concept:{concept}"
            features[key] += 1
            weights[key] = 5.5

        for sh in _RE_TRICH_DAN_SO_HIEU.findall(van_ban):
            key = f"sh:{sh.upper()}"
            features[key] += 1
            weights[key] = 4.5

        for dieu in _RE_DIEU_LUAT.findall(van_ban):
            key = f"dieu:{_chuan_khong_dau(dieu)}"
            features[key] += 1
            weights[key] = 3.2

        clean_kd = " ".join(khong_dau_tokens)
        if clean_kd:
            padded = f" {clean_kd} "
            for n in (3, 4):
                for i in range(max(0, len(padded) - n + 1)):
                    ng = padded[i : i + n]
                    if not ng.isspace():
                        key = f"c{n}:{ng}"
                        features[key] += 1
                        weights[key] = 0.55

        if not features:
            features[f"raw:{van_ban}"] = 1
            weights[f"raw:{van_ban}"] = 1.0

        for feat, count in features.items():
            tf = 1.0 + math.log(count)
            w = weights.get(feat, 1.0) * tf
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=16).digest()
            h1 = int.from_bytes(digest[0:4], "little")
            h2 = int.from_bytes(digest[4:8], "little")
            h3 = int.from_bytes(digest[8:12], "little")
            s1 = 1.0 if (digest[12] & 1) == 0 else -1.0
            s2 = 1.0 if (digest[13] & 1) == 0 else -1.0
            s3 = 1.0 if (digest[14] & 1) == 0 else -1.0
            vec[h1 % dim] += s1 * w
            vec[h2 % dim] += s2 * (w * 0.7)
            vec[h3 % dim] += s3 * (w * 0.45)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            vec /= norm

        if len(self._cache) >= self._max_cache:
            self._cache.clear()
        self._cache[van_ban] = vec.copy()
        return vec

    def nhung(self, van_ban: Union[str, List[str]]) -> np.ndarray:
        """Nhúng một câu hoặc danh sách câu văn bản tiếng Việt sang vector float32 chuẩn hóa L2."""
        if isinstance(van_ban, str):
            la_don = True
            texts = [van_ban]
        else:
            la_don = False
            texts = [str(t) for t in van_ban]

        if not texts:
            return np.zeros((0, self.kich_thuoc), dtype=np.float32)

        if self._ort_session is not None:
            out = self._nhung_onnx_batch(texts)
        else:
            out = np.vstack([self._nhung_ngu_nghia_viet(t) for t in texts]).astype(np.float32)

        return out[0] if la_don else out

    def nhung_mot(self, van_ban: str) -> np.ndarray:
        """Nhúng một câu văn bản đơn lẻ."""
        return self.nhung(van_ban)

    def __call__(self, van_ban: str) -> np.ndarray:
        """Cho phép truyền trực tiếp vào tham số `ham_embed` của `RAGPipeline`."""
        return self.nhung(van_ban)

    def thong_ke(self) -> Dict[str, Any]:
        """Trả về thông tin cấu hình của bộ nhúng ONNX tiếng Việt."""
        return {
            "che_do": self._che_do,
            "kich_thuoc": self.kich_thuoc,
            "so_khai_niem_dong_nghia": len(TU_DIEN_DONG_NGHIA_VIET),
            "duong_dan_onnx": str(self.duong_dan_onnx) if self.duong_dan_onnx else None,
        }


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ sắp xếp lại Cross-Encoder tiếng Việt hỗ trợ ONNX Runtime INT8 trên CPU và chấm điểm tương tác "
        "chéo câu hỏi–tài liệu (tách từ ghép, đồng nghĩa pháp lý, trích dẫn Điều/Số hiệu)."
    ),
    nhom_extra="export",
)
class BoSapXepLaiONNXTiengViet:
    """
    Cross-Encoder Reranker tối ưu cho tiếng Việt:

    - Tầng 1 (`onnx_cross_encoder`): Chạy cặp `(câu hỏi, đoạn văn)` qua mô hình ONNX Cross-Encoder.
    - Tầng 2 (`vietnamese_cross_interaction`): Chấm điểm tương tác chéo đa đặc trưng (từ ghép tiếng Việt,
      không dấu, cầu nối khái niệm đồng nghĩa, khớp số hiệu văn bản & số Điều) chuẩn hóa sigmoid.
    """

    def __init__(
        self,
        duong_dan_onnx: Optional[Union[str, Path]] = None,
        so_luong_thread: int = 4,
        do_dai_toi_da: int = 256,
    ) -> None:
        self.duong_dan_onnx = Path(duong_dan_onnx) if duong_dan_onnx else None
        self.so_luong_thread = max(1, int(so_luong_thread))
        self.do_dai_toi_da = int(do_dai_toi_da)
        self._tokenizer_viet = BoTachTuTiengVietSieuToc()
        self._ort_session: Optional[Any] = None
        self._che_do = "vietnamese_cross_interaction"

        if self.duong_dan_onnx is not None and self.duong_dan_onnx.is_file():
            ort = QuanLyPhuThuoc.nhap_tuy_chon("onnxruntime", "export", "chạy Cross-Encoder Reranker ONNX")
            if ort is not None:
                try:
                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = self.so_luong_thread
                    self._ort_session = ort.InferenceSession(
                        str(self.duong_dan_onnx),
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self._che_do = "onnx_cross_encoder"
                except Exception:
                    self._ort_session = None

    @property
    def che_do(self) -> str:
        """Chế độ chấm điểm reranker hiện hành."""
        return self._che_do

    def cham_diem(self, cau_hoi: str, tai_lieu: str) -> float:
        """Chấm điểm mức độ liên quan trong đoạn `[0.0, 1.0]` giữa một câu hỏi và một đoạn văn bản."""
        if not cau_hoi or not tai_lieu:
            return 0.0

        if self._ort_session is not None:
            pair_text = f"{cau_hoi} [SEP] {tai_lieu}"
            input_ids, attention_mask = _ma_hoa_token_ids_don_gian([pair_text], do_dai_toi_da=self.do_dai_toi_da)
            input_names = [inp.name for inp in self._ort_session.get_inputs()]
            feed: Dict[str, Any] = {}
            if "input_ids" in input_names:
                feed["input_ids"] = input_ids
            if "attention_mask" in input_names:
                feed["attention_mask"] = attention_mask
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
            if not feed and input_names:
                feed[input_names[0]] = input_ids
            logits = np.asarray(self._ort_session.run(None, feed)[0], dtype=np.float32).ravel()
            logit = float(logits[0]) if logits.size > 0 else 0.0
            return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit)))))

        q_toks = {t.lower() for t in self._tokenizer_viet.tach_tu(cau_hoi, noi_tu_ghep=True) if len(t) >= 2}
        d_toks = {t.lower() for t in self._tokenizer_viet.tach_tu(tai_lieu, noi_tu_ghep=True) if len(t) >= 2}
        overlap_tok = len(q_toks & d_toks) / max(len(q_toks), 1)

        q_kd = set(re.findall(r"[0-9a-z_]{2,}", _chuan_khong_dau(cau_hoi)))
        d_kd = set(re.findall(r"[0-9a-z_]{2,}", _chuan_khong_dau(tai_lieu)))
        overlap_kd = len(q_kd & d_kd) / max(len(q_kd), 1)

        q_concepts = set(_trich_xuat_khai_niem_viet(cau_hoi))
        d_concepts = set(_trich_xuat_khai_niem_viet(tai_lieu))
        concept_score = len(q_concepts & d_concepts) / max(len(q_concepts), 1) if q_concepts else 0.0

        q_sh = {s.upper() for s in _RE_TRICH_DAN_SO_HIEU.findall(cau_hoi)}
        d_sh = {s.upper() for s in _RE_TRICH_DAN_SO_HIEU.findall(tai_lieu)}
        legal_boost = 0.0
        if q_sh and (q_sh & d_sh):
            legal_boost += 0.35
        q_dieu = {_chuan_khong_dau(d) for d in _RE_DIEU_LUAT.findall(cau_hoi)}
        d_dieu = {_chuan_khong_dau(d) for d in _RE_DIEU_LUAT.findall(tai_lieu)}
        if q_dieu and (q_dieu & d_dieu):
            legal_boost += 0.25

        raw = 0.32 * overlap_tok + 0.25 * overlap_kd + 0.48 * concept_score + legal_boost
        return float(round(1.0 / (1.0 + math.exp(-4.0 * (raw - 0.35))), 6))

    def sap_xep_lai(
        self,
        cau_hoi: str,
        ket_qua: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sắp xếp lại danh sách ứng viên truy hồi bằng điểm Cross-Encoder tiếng Việt."""
        if not ket_qua:
            return []

        kq_moi: List[Dict[str, Any]] = []
        for item in ket_qua:
            ban_sao = dict(item)
            noi_dung = str(
                ban_sao.get("noi_dung")
                or ban_sao.get("metadata", {}).get("noi_dung", "")
            )
            diem_ce = self.cham_diem(cau_hoi, noi_dung)
            diem_goc = float(ban_sao.get("diem", 0.0))
            ban_sao["diem_cross_encoder"] = diem_ce
            ban_sao["diem_rerank"] = round(0.65 * diem_ce + 0.35 * max(0.0, diem_goc), 6)
            kq_moi.append(ban_sao)

        kq_moi.sort(key=lambda x: float(x.get("diem_rerank", 0.0)), reverse=True)
        return kq_moi[:top_k]

    def thong_ke(self) -> Dict[str, Any]:
        """Trả về thông tin cấu hình của bộ reranker."""
        return {
            "che_do": self._che_do,
            "duong_dan_onnx": str(self.duong_dan_onnx) if self.duong_dan_onnx else None,
        }


VietnameseONNXEmbedder = BoNhungONNXTiengViet
VietnameseONNXReranker = BoSapXepLaiONNXTiengViet

__all__ = [
    "TU_DIEN_DONG_NGHIA_VIET",
    "BoNhungONNXTiengViet",
    "VietnameseONNXEmbedder",
    "BoSapXepLaiONNXTiengViet",
    "VietnameseONNXReranker",
]
