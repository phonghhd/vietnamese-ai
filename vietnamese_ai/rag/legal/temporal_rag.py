"""BoTruyHoiPhapLyThoiGian (TemporalGraphLegalRAG) - Động Cơ Truy Hồi Pháp Lý Đa Bước Có Điều Kiện Thời Gian.

Kết hợp 5 tầng kiến trúc có thể bật/tắt độc lập phục vụ cả vận hành thực tế trên CPU
lẫn thí nghiệm đối chứng (Ablation Study) chuẩn học thuật:
1. `bm25`: Okapi BM25 trên token từ ghép tiếng Việt (`BoTachTuTiengVietSieuToc`).
2. `hybrid`: BM25 + Biểu diễn ngữ nghĩa vector n-gram/ONNX qua Reciprocal Rank Fusion (RRF).
3. `temporal`: Hybrid + Bóc tách mốc thời gian tự động (`BoPhanTichThoiGianPhapLy`) + Lọc hiệu lực Point-in-Time
   và chuyển hướng văn bản thay thế khi gặp bẫy điều luật hết hiệu lực.
4. `temporal_graph`: Temporal + Lan truyền đồ thị đa bước cấp Điều/Khoản (`DoThiThamChieuPhapLy`).
5. `full`: Temporal-Graph + Xếp hạng lại chéo (ONNX Cross-Encoder / Legal Cross-Interaction)
   + Cổng từ chối trả lời (Abstention Gate) + Kiểm chứng trích dẫn 3 tầng (`BoKiemChungTrichDanPhapLy`).
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
from vietnamese_ai.rag.legal.citation_graph import (
    DoThiThamChieuPhapLy,
    NutDieuKhoanPhapLy,
    QuanHeDieuKhoan,
)
from vietnamese_ai.rag.legal.citation_verifier import BoKiemChungTrichDanPhapLy
from vietnamese_ai.rag.legal.temporal_query import BoPhanTichThoiGianPhapLy
from vietnamese_ai.rag.legal.validity_tracker import (
    BoTheoDoiHieuLucPhapLy,
    HoSoHieuLucVanBan,
    _chuan_hoa_dieu,
    _chuan_hoa_so_hieu,
    _phan_tich_ngay,
)
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

_TU_DUNG_PHAP_LY_VN: Set[str] = {
    "là",
    "và",
    "của",
    "các",
    "cho",
    "với",
    "trong",
    "theo",
    "tại",
    "từ",
    "đến",
    "về",
    "có",
    "được",
    "không",
    "hay",
    "hoặc",
    "như",
    "thế",
    "nào",
    "bao",
    "nhiêu",
    "gì",
    "khi",
    "nếu",
    "thì",
    "phải",
    "cần",
    "quy",
    "định",
    "quy_định",
    "áp",
    "dụng",
    "áp_dụng",
    "hiện",
    "hành",
    "hiện_hành",
    "ngày",
    "tháng",
    "năm",
    "trước",
    "sau",
    "vào",
    "một",
    "những",
    "này",
    "đó",
    "để",
    "bị",
    "bởi",
    "do",
    "tôi",
    "bạn",
    "hãy",
    "biết",
    "hỏi",
    "rằng",
    "còn",
    "chưa",
    "đã",
    "sẽ",
    "vẫn",
    "ra",
    "sao",
}

_TU_DUNG_CO_BAN: Set[str] = {
    "là",
    "và",
    "của",
    "các",
    "cho",
    "với",
    "trong",
    "tại",
    "từ",
    "đến",
    "về",
    "có",
    "được",
    "hay",
    "hoặc",
    "như",
    "thế",
    "nào",
    "bao",
    "nhiêu",
    "gì",
    "khi",
    "nếu",
    "thì",
    "này",
    "đó",
    "để",
    "tôi",
    "bạn",
    "hãy",
    "ra",
    "sao",
    "một",
    "những",
}


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Động cơ truy hồi pháp lý đa bước theo mốc thời gian (Temporal & Graph-Grounded Legal RAG), "
        "hỗ trợ đầy đủ 5 cấu hình Ablation trên CPU và kiểm chứng trích dẫn 3 tầng."
    ),
    nhom_extra="core",
)
class BoTruyHoiPhapLyThoiGian:
    """Động cơ Temporal & Graph-Grounded Legal RAG chạy mượt 100% trên CPU và hỗ trợ cắm ONNX/LLM ngoài."""

    CAC_CHE_DO_HOP_LE = ("bm25", "hybrid", "naive_date_filter", "temporal", "temporal_graph", "full")

    _RE_SO_HIEU = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+)\b", re.UNICODE)
    _RE_TEN_LUAT_NAM = re.compile(
        r"\b((bộ\s+luật|luật|nghị\s+định|thông\s+tư)\s+([^.;?!()]{2,65}?)\s+(?:năm\s+)?((?:19|20)\d{2}))\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_DIEU = re.compile(r"\bđiều\s+(\d+[a-zA-ZđĐ]?)\b", re.IGNORECASE | re.UNICODE)
    _RE_TOKEN_SACH = re.compile(r"[^\w/_-]+", re.UNICODE)
    _SO_CHIEU_HASH = 1536
    _TU_NGAT_TEN_LUAT: Set[str] = {
        "trong",
        "vào",
        "từ",
        "trước",
        "sau",
        "quy",
        "định",
        "yêu",
        "cầu",
        "cần",
        "được",
        "phải",
        "có",
        "là",
        "thì",
        "để",
        "hiện",
        "hành",
        "tháng",
        "ngày",
        "quý",
        "nếu",
        "khi",
        "về",
        "cho",
        "của",
    }
    _MO_RONG_VIET_TAT_LUAT: Dict[str, str] = {
        "bhxh": "bảo hiểm xã hội",
        "gtgt": "giá trị gia tăng",
        "gdđt": "giao dịch điện tử",
        "gddt": "giao dịch điện tử",
        "kdbđs": "kinh doanh bất động sản",
        "bllđ": "lao động",
        "blds": "dân sự",
    }

    def __init__(
        self,
        bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None,
        do_thi: Optional[DoThiThamChieuPhapLy] = None,
        bo_phan_tich_thoi_gian: Optional[BoPhanTichThoiGianPhapLy] = None,
        bo_kiem_chung: Optional[BoKiemChungTrichDanPhapLy] = None,
        ham_nhung: Optional[Callable[[str], np.ndarray]] = None,
        bo_xep_hang_cheo: Optional[Any] = None,
        nguong_tu_choi: float = 0.22,
    ) -> None:
        self.bo_theo_doi = bo_theo_doi or BoTheoDoiHieuLucPhapLy()
        self.do_thi = do_thi or DoThiThamChieuPhapLy(bo_theo_doi=self.bo_theo_doi)
        self.bo_phan_tich_thoi_gian = bo_phan_tich_thoi_gian or BoPhanTichThoiGianPhapLy()
        self.bo_kiem_chung = bo_kiem_chung or BoKiemChungTrichDanPhapLy(
            bo_theo_doi=self.bo_theo_doi,
            bo_phan_tich_thoi_gian=self.bo_phan_tich_thoi_gian,
        )
        self.ham_nhung = ham_nhung
        self.bo_xep_hang_cheo = bo_xep_hang_cheo
        self.nguong_tu_choi = float(nguong_tu_choi)
        self._tokenizer = BoTachTuTiengVietSieuToc()

        self._tai_lieu: List[NutDieuKhoanPhapLy] = []
        self._id_den_chi_so: Dict[str, int] = {}
        self._so_hieu_trong_kho: Set[str] = set()
        self._tokens_kho: List[List[str]] = []
        self._tf_kho: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._do_dai_tb: float = 1.0
        self._ma_tran_dense: Optional[np.ndarray] = None
        self._can_xay_chi_muc: bool = False

    def _tach_token_phap_ly(self, van_ban: str, loai_bo_tu_dung: bool = False) -> List[str]:
        """Tách token tiếng Việt bảo toàn số hiệu văn bản và cụm Điều luật."""
        if not van_ban:
            return []
        text = self._tokenizer.chuan_hoa_dau_thanh(van_ban)
        sh_list = [_chuan_hoa_so_hieu(m).lower() for m in self._RE_SO_HIEU.findall(text)]
        dieu_list = [f"dieu_{m.lower()}" for m in self._RE_DIEU.findall(text)]

        text_khong_sh = self._RE_SO_HIEU.sub(" ", text)
        da_tach = self._tokenizer.tach_tu(text_khong_sh.lower(), noi_tu_ghep=True)
        tokens: List[str] = []
        tu_dung = _TU_DUNG_PHAP_LY_VN if loai_bo_tu_dung else _TU_DUNG_CO_BAN
        for tok in da_tach:
            sach = self._RE_TOKEN_SACH.sub("", tok).strip("_")
            if not sach or len(sach) < 2 and not sach.isdigit():
                continue
            if sach in tu_dung:
                continue
            tokens.append(sach)
            if "_" in sach:
                for sub in sach.split("_"):
                    if sub and (len(sub) >= 2 or sub.isdigit()) and sub not in tu_dung:
                        tokens.append(sub)
        tokens.extend(sh_list)
        tokens.extend(dieu_list)
        return tokens

    def _vector_dac_trung_cpu(self, van_ban: str, tokens: List[str]) -> np.ndarray:
        """Tạo vector biểu diễn ngữ nghĩa bề mặt (word unigram/bigram + char 3-4-gram) chuẩn hóa L2 chạy trên CPU."""
        if self.ham_nhung is not None:
            vec = np.asarray(self.ham_nhung(van_ban), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 1e-12 else vec

        vec = np.zeros(self._SO_CHIEU_HASH, dtype=np.float32)
        dim = self._SO_CHIEU_HASH

        def _cap_nhat(dac_trung: str, trong_so: float) -> None:
            digest = hashlib.blake2b(dac_trung.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            idx = val % dim
            dau = 1.0 if ((val >> 16) & 1) == 0 else -1.0
            vec[idx] += dau * trong_so

        for i, tok in enumerate(tokens):
            w = self._idf.get(tok, 1.5)
            _cap_nhat(f"w:{tok}", w)
            if i + 1 < len(tokens):
                _cap_nhat(f"b:{tok}|{tokens[i + 1]}", 1.35 * w)

        text_low = re.sub(r"\s+", " ", self._tokenizer.chuan_hoa_dau_thanh(van_ban).lower()).strip()
        if len(text_low) >= 4:
            for n in (3, 4):
                for j in range(0, len(text_low) - n + 1):
                    gram = text_low[j : j + n]
                    if gram.strip():
                        _cap_nhat(f"c{n}:{gram}", 0.35)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec

    def dang_ky_van_ban(self, ho_so: HoSoHieuLucVanBan) -> None:
        """Đăng ký hồ sơ hiệu lực văn bản vào cả bộ theo dõi hiệu lực và đồ thị tham chiếu."""
        self.do_thi.them_van_ban(ho_so)

    def them_dieu_khoan(self, nut: NutDieuKhoanPhapLy) -> None:
        """Thêm một nút Điều/Khoản có phiên bản thời gian vào kho truy hồi và đồ thị."""
        self.do_thi.them_nut_dieu_khoan(nut)
        nut_chuan = self.do_thi.lay_nut_dieu_khoan(nut.ma_nut)
        if nut_chuan is None:
            return
        if nut_chuan.ma_nut in self._id_den_chi_so:
            idx = self._id_den_chi_so[nut_chuan.ma_nut]
            self._tai_lieu[idx] = nut_chuan
        else:
            self._id_den_chi_so[nut_chuan.ma_nut] = len(self._tai_lieu)
            self._tai_lieu.append(nut_chuan)
        if nut_chuan.so_hieu:
            self._so_hieu_trong_kho.add(_chuan_hoa_so_hieu(nut_chuan.so_hieu))
        self._can_xay_chi_muc = True

    def them_quan_he_dieu_khoan(self, canh: QuanHeDieuKhoan) -> bool:
        """Thêm một cạnh quan hệ cấp Điều/Khoản có khoảng hiệu lực vào đồ thị."""
        return self.do_thi.them_quan_he_dieu_khoan(canh)

    def nap_tu_danh_sach(
        self,
        danh_sach_dieu_khoan: Sequence[Dict[str, Any]],
        danh_sach_canh: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> int:
        """Nạp hàng loạt điều khoản (JSONL/dict) và cạnh quan hệ cấp Điều/Khoản rồi xây chỉ mục."""
        da_nap = 0
        for item in danh_sach_dieu_khoan:
            ma = str(item.get("id") or item.get("ma_nut") or f"dk_{da_nap}")
            sh = str(item.get("so_hieu") or "")
            dieu = str(item.get("dieu") or "")
            nut = NutDieuKhoanPhapLy(
                ma_nut=ma,
                so_hieu=sh,
                dieu=dieu,
                noi_dung=str(item.get("noi_dung") or ""),
                ten_van_ban=str(item.get("ten_van_ban") or item.get("van_ban") or ""),
                khoan=str(item["khoan"]) if item.get("khoan") else None,
                tieu_de=str(item.get("tieu_de") or ""),
                linh_vuc=str(item.get("linh_vuc") or ""),
                hieu_luc_tu=str(item["hieu_luc_tu"]) if item.get("hieu_luc_tu") else None,
                het_hieu_luc_tu=str(item["het_hieu_luc_tu"]) if item.get("het_hieu_luc_tu") else None,
                sua_doi_boi=str(item["sua_doi_boi"]) if item.get("sua_doi_boi") else None,
                trang_thai=str(item.get("trang_thai") or "hieu_luc"),
            )
            self.them_dieu_khoan(nut)
            da_nap += 1

        if danh_sach_canh:
            for e in danh_sach_canh:
                self.them_quan_he_dieu_khoan(
                    QuanHeDieuKhoan(
                        nguon_nut=str(e.get("nguon_nut") or e.get("nguon") or ""),
                        dich_nut=str(e.get("dich_nut") or e.get("dich") or ""),
                        loai_quan_he=str(e.get("loai_quan_he") or "vien_dan_cheo"),
                        hieu_luc_tu=str(e["hieu_luc_tu"]) if e.get("hieu_luc_tu") else None,
                        het_hieu_luc_tu=str(e["het_hieu_luc_tu"]) if e.get("het_hieu_luc_tu") else None,
                        trong_so=float(e.get("trong_so", 1.0)),
                        mo_ta=str(e.get("mo_ta") or ""),
                    )
                )

        self.xay_dung_chi_muc()
        return da_nap

    @staticmethod
    def _van_ban_chi_muc(nut: NutDieuKhoanPhapLy) -> str:
        """Tạo chuỗi đầy đủ của nút Điều/Khoản phục vụ lập chỉ mục BM25 và vector."""
        phan = [
            nut.ten_van_ban,
            nut.so_hieu,
            nut.dieu,
            nut.khoan or "",
            nut.tieu_de,
            nut.noi_dung,
        ]
        return " ".join(p for p in phan if p).strip()

    def xay_dung_chi_muc(self) -> None:
        """Xây dựng chỉ mục Okapi BM25 và ma trận vector ngữ nghĩa chuẩn hóa L2."""
        n = len(self._tai_lieu)
        self._tokens_kho = []
        self._tf_kho = []
        self._df = {}

        tong_len = 0
        for nut in self._tai_lieu:
            toks = self._tach_token_phap_ly(self._van_ban_chi_muc(nut), loai_bo_tu_dung=False)
            self._tokens_kho.append(toks)
            tong_len += len(toks)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf_kho.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1

        self._do_dai_tb = (tong_len / n) if n > 0 else 1.0
        self._idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self._df.items()
        }

        if n > 0:
            vecs = [
                self._vector_dac_trung_cpu(self._van_ban_chi_muc(nut), self._tokens_kho[i])
                for i, nut in enumerate(self._tai_lieu)
            ]
            self._ma_tran_dense = np.vstack(vecs).astype(np.float32)
        else:
            self._ma_tran_dense = None
        self._can_xay_chi_muc = False

    def _tinh_diem_bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Tính điểm Okapi BM25 cho toàn bộ kho điều khoản."""
        n = len(self._tai_lieu)
        scores = np.zeros(n, dtype=np.float64)
        if not query_tokens or n == 0:
            return scores
        for i in range(n):
            tf_doc = self._tf_kho[i]
            dl = len(self._tokens_kho[i])
            if dl == 0:
                continue
            denom_base = k1 * (1.0 - b + b * (dl / self._do_dai_tb))
            s = 0.0
            for tok in query_tokens:
                tf = tf_doc.get(tok, 0)
                if tf > 0:
                    idf = self._idf.get(tok, 0.0)
                    s += idf * (tf * (k1 + 1.0)) / (tf + denom_base)
            scores[i] = s
        return scores

    def _tinh_diem_dense(self, cau_hoi: str, query_tokens: List[str]) -> np.ndarray:
        """Tính độ tương đồng cosine trên không gian biểu diễn ngữ nghĩa CPU/ONNX."""
        n = len(self._tai_lieu)
        if n == 0 or self._ma_tran_dense is None:
            return np.zeros(n, dtype=np.float64)
        q_vec = self._vector_dac_trung_cpu(cau_hoi, query_tokens)
        if q_vec.shape[0] != self._ma_tran_dense.shape[1]:
            return np.zeros(n, dtype=np.float64)
        sims = self._ma_tran_dense @ q_vec.astype(np.float32)
        return np.clip(sims.astype(np.float64), 0.0, 1.0)

    def _cham_diem_reranker_ngoai(self, cau_hoi: str, tai_lieu: str) -> Optional[float]:
        """Gọi thống nhất giao diện của bộ xếp hạng lại ngoài (`cham_diem`, `cham_diem_lo`, `sap_xep_lai`, `tinh_diem_cap`)."""
        rr = self.bo_xep_hang_cheo
        if rr is None:
            return None
        try:
            if hasattr(rr, "cham_diem"):
                return float(rr.cham_diem(cau_hoi, tai_lieu))
            if hasattr(rr, "cham_diem_lo"):
                scores = rr.cham_diem_lo(cau_hoi, [tai_lieu])
                if scores:
                    return float(scores[0])
            if hasattr(rr, "sap_xep_lai"):
                kq_rr = rr.sap_xep_lai(cau_hoi, [{"noi_dung": tai_lieu, "diem": 0.5}], top_k=1)
                if kq_rr and isinstance(kq_rr[0], dict):
                    val = kq_rr[0].get("diem_rerank", kq_rr[0].get("diem"))
                    if val is not None:
                        return float(val)
            if hasattr(rr, "tinh_diem_cap"):
                return float(rr.tinh_diem_cap(cau_hoi, tai_lieu))
        except Exception:
            return None
        return None

    def _diem_tuong_tac_cheo(
        self,
        cau_hoi: str,
        q_tokens_noi_dung: List[str],
        nut: NutDieuKhoanPhapLy,
        idx_nut: int,
        sh_trong_hoi: List[str],
        sh_thay_the: Set[str],
        dieu_trong_hoi: Set[str],
        diem_co_so: float,
        diem_do_thi: float,
        diem_reranker_san: Optional[float] = None,
    ) -> float:
        """Chấm điểm tương tác chéo giữa câu hỏi và điều khoản (ONNX Cross-Encoder hoặc tương tác cấu trúc)."""
        if diem_reranker_san is not None:
            return 0.65 * float(diem_reranker_san) + 0.20 * diem_co_so + 0.15 * diem_do_thi
        if self.bo_xep_hang_cheo is not None:
            diem_nn = self._cham_diem_reranker_ngoai(cau_hoi, self._van_ban_chi_muc(nut))
            if diem_nn is not None:
                return 0.65 * diem_nn + 0.20 * diem_co_so + 0.15 * diem_do_thi

        doc_tf = self._tf_kho[idx_nut]
        doc_tokens = self._tokens_kho[idx_nut]
        doc_set = set(doc_tokens)

        tong_idf = sum(self._idf.get(t, 1.0) for t in set(q_tokens_noi_dung)) or 1.0
        khop_idf = sum(self._idf.get(t, 1.0) for t in set(q_tokens_noi_dung) if t in doc_set)
        do_phu_idf = khop_idf / tong_idf

        q_bigrams = {
            (q_tokens_noi_dung[i], q_tokens_noi_dung[i + 1])
            for i in range(len(q_tokens_noi_dung) - 1)
        }
        if q_bigrams and len(doc_tokens) >= 2:
            d_bigrams = {(doc_tokens[i], doc_tokens[i + 1]) for i in range(len(doc_tokens) - 1)}
            do_khop_cum = len(q_bigrams & d_bigrams) / len(q_bigrams)
        else:
            do_khop_cum = 0.0

        bonus_cau_truc = 0.0
        if nut.so_hieu in sh_trong_hoi:
            bonus_cau_truc += 0.22
        if nut.so_hieu in sh_thay_the:
            bonus_cau_truc += 0.25
        if nut.dieu and nut.dieu in dieu_trong_hoi:
            bonus_cau_truc += 0.18
        if nut.tieu_de:
            td_toks = set(self._tach_token_phap_ly(nut.tieu_de, loai_bo_tu_dung=True))
            if td_toks and q_tokens_noi_dung:
                bonus_cau_truc += 0.12 * (len(td_toks & set(q_tokens_noi_dung)) / max(1, len(td_toks)))

        do_day_tf = min(1.0, sum(min(3, doc_tf.get(t, 0)) for t in set(q_tokens_noi_dung)) / max(1, 2 * len(set(q_tokens_noi_dung))))

        diem_cheo = (
            0.35 * do_phu_idf
            + 0.25 * do_khop_cum
            + 0.15 * do_day_tf
            + 0.25 * bonus_cau_truc
        )
        return float(diem_co_so) + 0.18 * diem_cheo + 0.05 * diem_do_thi

    @classmethod
    def _chuan_hoa_tap_tu_ten_luat(cls, ten: str) -> Set[str]:
        """Chuẩn hóa tập từ khóa lõi của tên văn bản luật để đối khớp không phụ thuộc dấu phẩy hay viết tắt."""
        sach = re.sub(r"\([^)]*\)", " ", ten.lower())
        sach = re.split(r"\b(?:hướng\s+dẫn|sửa\s+đổi|về\s+việc|quy\s+định)\b", sach, maxsplit=1)[0]
        sach = re.sub(r"\b(?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|năm)\b", " ", sach)
        sach = re.sub(r"\b(?:19|20)\d{2}\b", " ", sach)
        tu_list = re.findall(r"[0-9a-zà-ỹđ]+", sach)
        kq: Set[str] = set()
        for tu in tu_list:
            if tu in cls._MO_RONG_VIET_TAT_LUAT:
                kq.update(cls._MO_RONG_VIET_TAT_LUAT[tu].split())
            elif len(tu) >= 2:
                kq.add(tu)
        return kq

    def _truy_tim_ten_luat_trong_cau_hoi(self, cau_hoi: str) -> Tuple[List[str], List[str]]:
        """Phân giải tên đạo luật kèm năm (không cần ghi số hiệu) sang số hiệu đã đăng ký hoặc đánh dấu ngoài kho."""
        van_ban_sach = self._RE_SO_HIEU.sub(" ", cau_hoi)
        ung_vien_ten: List[Tuple[str, str, Set[str], str]] = []
        for m in self._RE_TEN_LUAT_NAM.finditer(van_ban_sach):
            ten_day_du = m.group(1).strip()
            loai_vb = re.sub(r"\s+", " ", m.group(2).strip().lower())
            phan_giua = m.group(3).strip()
            nam_str = m.group(4).strip()
            tu_giua = re.findall(r"[0-9a-zà-ỹđ]+", phan_giua.lower())
            if not tu_giua or any(t in self._TU_NGAT_TEN_LUAT for t in tu_giua):
                continue
            tap_tu = self._chuan_hoa_tap_tu_ten_luat(phan_giua)
            if not tap_tu:
                continue
            ung_vien_ten.append((ten_day_du, loai_vb, tap_tu, nam_str))

        if not ung_vien_ten:
            return [], []

        kho_luat: List[Tuple[str, str, Set[str], str]] = []
        da_co_sh: Set[str] = set()
        for hs in self.bo_theo_doi.danh_sach_van_ban():
            sh = _chuan_hoa_so_hieu(str(hs.get("so_hieu", "")))
            ten_vb = str(hs.get("ten_van_ban", ""))
            if not sh or not ten_vb:
                continue
            m_nam = re.search(r"/((?:19|20)\d{2})/", sh) or re.search(r"\b((?:19|20)\d{2})\b", ten_vb)
            nam_vb = m_nam.group(1) if m_nam else ""
            low_ten = ten_vb.lower()
            loai_vb = (
                "bộ luật"
                if low_ten.startswith("bộ luật")
                else ("nghị định" if low_ten.startswith("nghị định") else ("thông tư" if low_ten.startswith("thông tư") else "luật"))
            )
            kho_luat.append((sh, loai_vb, self._chuan_hoa_tap_tu_ten_luat(ten_vb), nam_vb))
            da_co_sh.add(sh)

        for nut in self._tai_lieu:
            sh = _chuan_hoa_so_hieu(nut.so_hieu)
            if not sh or sh in da_co_sh or not nut.ten_van_ban:
                continue
            m_nam = re.search(r"/((?:19|20)\d{2})/", sh) or re.search(r"\b((?:19|20)\d{2})\b", nut.ten_van_ban)
            nam_vb = m_nam.group(1) if m_nam else ""
            low_ten = nut.ten_van_ban.lower()
            loai_vb = (
                "bộ luật"
                if low_ten.startswith("bộ luật")
                else ("nghị định" if low_ten.startswith("nghị định") else ("thông tư" if low_ten.startswith("thông tư") else "luật"))
            )
            kho_luat.append((sh, loai_vb, self._chuan_hoa_tap_tu_ten_luat(nut.ten_van_ban), nam_vb))
            da_co_sh.add(sh)

        sh_tim_thay: List[str] = []
        ten_khong_ton_tai: List[str] = []
        for ten_day_du, loai_q, tap_tu_q, nam_q in ung_vien_ten:
            best_sh: Optional[str] = None
            best_score = 0.0
            for sh, loai_k, tap_tu_k, nam_k in kho_luat:
                if nam_k and nam_k != nam_q:
                    continue
                if loai_q != loai_k and not ({loai_q, loai_k} == {"luật", "bộ luật"}):
                    continue
                giao = len(tap_tu_q & tap_tu_k)
                if giao == 0:
                    continue
                hop = len(tap_tu_q | tap_tu_k)
                jaccard = giao / float(hop) if hop > 0 else 0.0
                phu_q = giao / float(len(tap_tu_q))
                diem = max(jaccard, 0.85 * phu_q if giao >= 2 else 0.0)
                if diem > best_score:
                    best_score = diem
                    best_sh = sh
            if best_sh is not None and best_score >= 0.65:
                if best_sh not in sh_tim_thay:
                    sh_tim_thay.append(best_sh)
            else:
                ten_khong_ton_tai.append(ten_day_du)
        return sh_tim_thay, ten_khong_ton_tai

    def truy_hoi(
        self,
        cau_hoi: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        top_k: int = 5,
        che_do: str = "full",
        so_buoc_do_thi: int = 2,
        ngay_mac_dinh: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Truy hồi danh sách điều khoản pháp lý theo một trong 6 cấu hình (`bm25`, `hybrid`, `naive_date_filter`, `temporal`, `temporal_graph`, `full`)."""
        if che_do not in self.CAC_CHE_DO_HOP_LE:
            raise ValueError(f"Chế độ truy hồi không hợp lệ: {che_do}. Hỗ trợ: {', '.join(self.CAC_CHE_DO_HOP_LE)}")
        if self._can_xay_chi_muc:
            self.xay_dung_chi_muc()

        t0 = time.perf_counter()
        kq_tg = self.bo_phan_tich_thoi_gian.phan_tich(cau_hoi, ngay_mac_dinh=ngay_mac_dinh)
        moc = _phan_tich_ngay(ngay_tra_cuu) if ngay_tra_cuu is not None else _phan_tich_ngay(kq_tg.ngay_tra_cuu)
        moc_iso = moc.isoformat()

        bat_hybrid = che_do in ("hybrid", "naive_date_filter", "temporal", "temporal_graph", "full")
        bat_naive_date = che_do == "naive_date_filter"
        bat_temporal = che_do in ("temporal", "temporal_graph", "full")
        bat_graph = che_do in ("temporal_graph", "full")
        bat_rerank = che_do == "full"
        bat_tu_choi = che_do == "full"

        n = len(self._tai_lieu)
        if n == 0:
            return {
                "cau_hoi": cau_hoi,
                "che_do": che_do,
                "ngay_tra_cuu": moc_iso,
                "phan_tich_thoi_gian": kq_tg.to_dict(),
                "tu_choi_tra_loi": True,
                "ly_do_tu_choi": "Kho tri thức pháp lý đang trống.",
                "canh_bao_thoi_gian": [],
                "van_ban_thay_the_goi_y": [],
                "ket_qua": [],
                "thoi_gian_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        q_tokens = self._tach_token_phap_ly(cau_hoi, loai_bo_tu_dung=False)
        q_tokens_noi_dung = self._tach_token_phap_ly(cau_hoi, loai_bo_tu_dung=True)
        if not q_tokens_noi_dung:
            q_tokens_noi_dung = list(q_tokens)

        sh_trong_hoi = [_chuan_hoa_so_hieu(m) for m in self._RE_SO_HIEU.findall(cau_hoi)]
        sh_theo_ten, ten_luat_khong_ton_tai = self._truy_tim_ten_luat_trong_cau_hoi(cau_hoi)
        for sh_ten in sh_theo_ten:
            if sh_ten not in sh_trong_hoi:
                sh_trong_hoi.append(sh_ten)
        dieu_trong_hoi = {_chuan_hoa_dieu(m) for m in self._RE_DIEU.findall(cau_hoi)}

        canh_bao_tg: List[str] = []
        sh_het_hl_trong_hoi: Set[str] = set()
        sh_thay_the: Set[str] = set()
        sh_khong_ton_tai: List[str] = list(ten_luat_khong_ton_tai)

        for sh in sh_trong_hoi:
            hl = self.bo_theo_doi.kiem_tra_hieu_luc(sh, ngay_tra_cuu=moc)
            if hl.get("tim_thay"):
                if hl.get("con_hieu_luc") is False:
                    sh_het_hl_trong_hoi.add(sh)
                    if hl.get("canh_bao"):
                        canh_bao_tg.append(str(hl["canh_bao"]))
                    kq_dt = self.do_thi.truy_van_van_ban_lien_quan(sh, ngay_tra_cuu=moc, chi_lay_con_hieu_luc=True)
                    for tt_item in kq_dt.get("van_ban_thay_the_hien_hanh", []):
                        sh_thay_the.add(str(tt_item["so_hieu"]))
                    for sh_tt in hl.get("bi_thay_the_boi", []):
                        hl_tt = self.bo_theo_doi.kiem_tra_hieu_luc(sh_tt, ngay_tra_cuu=moc)
                        if hl_tt.get("con_hieu_luc") is True:
                            sh_thay_the.add(str(sh_tt))
            elif sh not in self._so_hieu_trong_kho:
                sh_khong_ton_tai.append(sh)

        if sh_thay_the:
            canh_bao_tg.append(
                f"Đã tự động chuyển hướng sang văn bản thay thế đang còn hiệu lực tại {moc_iso}: "
                f"{', '.join(sorted(sh_thay_the))}."
            )

        q_tokens_mo_rong = list(q_tokens)
        if bat_temporal and sh_thay_the:
            for sh_tt in sorted(sh_thay_the):
                q_tokens_mo_rong.append(sh_tt.lower())

        bm25_scores = self._tinh_diem_bm25(q_tokens_mo_rong)
        if bat_hybrid:
            dense_scores = self._tinh_diem_dense(cau_hoi, q_tokens_mo_rong)
        else:
            dense_scores = np.zeros(n, dtype=np.float64)

        order_bm25 = np.argsort(-bm25_scores, kind="mergesort")
        rank_bm25 = np.empty(n, dtype=np.int64)
        rank_bm25[order_bm25] = np.arange(1, n + 1)

        if bat_hybrid:
            order_dense = np.argsort(-dense_scores, kind="mergesort")
            rank_dense = np.empty(n, dtype=np.int64)
            rank_dense[order_dense] = np.arange(1, n + 1)
            rrf_k = 60.0
            base_scores = (1.0 / (rrf_k + rank_bm25)) + (1.0 / (rrf_k + rank_dense))
            max_bm25 = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0
            max_dense = float(np.max(dense_scores)) if np.max(dense_scores) > 0 else 1.0
            norm_ret = 0.55 * (bm25_scores / max_bm25) + 0.45 * (dense_scores / max_dense)
            base_scores = 0.55 * (base_scores / max(1e-9, float(np.max(base_scores)))) + 0.45 * norm_ret
        else:
            max_bm25 = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0
            base_scores = bm25_scores / max_bm25

        hop_le_mask = np.ones(n, dtype=bool)
        if bat_temporal:
            for i, nut in enumerate(self._tai_lieu):
                hop_le_mask[i] = self.do_thi.dieu_khoan_con_hieu_luc(nut, ngay_tra_cuu=moc)
                if hop_le_mask[i] and nut.so_hieu in sh_thay_the:
                    base_scores[i] = float(base_scores[i]) + 0.35
        elif bat_naive_date:
            for i, nut in enumerate(self._tai_lieu):
                if nut.hieu_luc_tu:
                    hop_le_mask[i] = _phan_tich_ngay(nut.hieu_luc_tu) <= moc
                else:
                    hop_le_mask[i] = True

        ung_vien_idx = [i for i in range(n) if hop_le_mask[i]]
        if not ung_vien_idx and not bat_temporal:
            ung_vien_idx = list(range(n))

        ung_vien_idx.sort(key=lambda i: (-float(base_scores[i]), self._tai_lieu[i].ma_nut))

        diem_do_thi_map: Dict[str, float] = {}
        thong_tin_do_thi_map: Dict[str, Dict[str, Any]] = {}
        top1_base = float(base_scores[ung_vien_idx[0]]) if ung_vien_idx else 1.0
        top1_id = self._tai_lieu[ung_vien_idx[0]].ma_nut if ung_vien_idx else ""
        if bat_graph and ung_vien_idx:
            so_hat_giong = min(2, len(ung_vien_idx))
            hat_giong: List[str] = []
            diem_hat_giong: Dict[str, float] = {}
            for i in ung_vien_idx[:so_hat_giong]:
                if float(base_scores[i]) <= 0.0:
                    continue
                ma_n = self._tai_lieu[i].ma_nut
                hat_giong.append(ma_n)
                diem_hat_giong[ma_n] = float(base_scores[i])

            if sh_het_hl_trong_hoi:
                for i, nut in enumerate(self._tai_lieu):
                    if nut.so_hieu in sh_het_hl_trong_hoi and float(bm25_scores[i]) > 0.0:
                        if nut.ma_nut not in diem_hat_giong:
                            hat_giong.append(nut.ma_nut)
                            diem_hat_giong[nut.ma_nut] = max(0.5, float(bm25_scores[i]) / max(1.0, float(np.max(bm25_scores))))

            mo_rong = self.do_thi.mo_rong_da_buoc_theo_thoi_gian(
                ma_nut_khoi_dau=hat_giong,
                ngay_tra_cuu=moc,
                so_buoc=so_buoc_do_thi,
                chi_lay_con_hieu_luc=bat_temporal,
                diem_khoi_dau=diem_hat_giong,
            )
            max_g = max((float(x["diem_do_thi"]) for x in mo_rong), default=1.0) or 1.0
            for item_g in mo_rong:
                ma_g = str(item_g["ma_nut"])
                diem_do_thi_map[ma_g] = float(item_g["diem_do_thi"]) / max_g
                thong_tin_do_thi_map[ma_g] = item_g

        diem_rerank_lo: Dict[int, float] = {}
        if bat_rerank and self.bo_xep_hang_cheo is not None and ung_vien_idx and hasattr(self.bo_xep_hang_cheo, "cham_diem_lo"):
            try:
                docs_lo = [self._van_ban_chi_muc(self._tai_lieu[i]) for i in ung_vien_idx]
                scores_lo = self.bo_xep_hang_cheo.cham_diem_lo(cau_hoi, docs_lo)
                if len(scores_lo) == len(ung_vien_idx):
                    diem_rerank_lo = {idx_i: float(sc) for idx_i, sc in zip(ung_vien_idx, scores_lo)}
            except Exception:
                diem_rerank_lo = {}

        diem_cuoi: Dict[int, float] = {}
        for i in ung_vien_idx:
            nut = self._tai_lieu[i]
            g_score = diem_do_thi_map.get(nut.ma_nut, 0.0)
            ppr_restart = 0.38 * float(base_scores[i]) if nut.ma_nut == top1_id else 0.0
            boost_do_thi = min(0.32 * max(0.25, top1_base), 0.32 * g_score) if bat_graph else 0.0
            diem_do_thi_gop = float(base_scores[i]) + ppr_restart + boost_do_thi
            if bat_rerank:
                score_i = self._diem_tuong_tac_cheo(
                    cau_hoi=cau_hoi,
                    q_tokens_noi_dung=q_tokens_noi_dung,
                    nut=nut,
                    idx_nut=i,
                    sh_trong_hoi=sh_trong_hoi,
                    sh_thay_the=sh_thay_the,
                    dieu_trong_hoi=dieu_trong_hoi,
                    diem_co_so=diem_do_thi_gop,
                    diem_do_thi=g_score,
                    diem_reranker_san=diem_rerank_lo.get(i),
                )
            elif bat_graph:
                score_i = diem_do_thi_gop
            else:
                score_i = float(base_scores[i])
            diem_cuoi[i] = score_i

        da_xep_hang = sorted(
            ung_vien_idx,
            key=lambda i: (-diem_cuoi[i], self._tai_lieu[i].ma_nut),
        )

        tu_choi = False
        ly_do_tu_choi = ""
        if bat_tu_choi:
            if sh_khong_ton_tai and not sh_thay_the:
                tu_choi = True
                ly_do_tu_choi = (
                    f"Văn bản pháp luật được yêu cầu ({', '.join(sh_khong_ton_tai)}) không tồn tại "
                    f"trong kho tri thức pháp lý hiện có."
                )
            elif not da_xep_hang:
                tu_choi = True
                ly_do_tu_choi = f"Không tìm thấy điều khoản nào còn hiệu lực tại ngày {moc_iso}."
            else:
                top_i = da_xep_hang[0]
                top_set = set(self._tokens_kho[top_i])
                tu_ghep_q = {t for t in q_tokens_noi_dung if "_" in t}
                am_con_tu_ghep = {sub for tg in tu_ghep_q for sub in tg.split("_")}
                q_doc_lap = {t for t in q_tokens_noi_dung if t not in am_con_tu_ghep or t in tu_ghep_q}
                if not q_doc_lap:
                    q_doc_lap = set(q_tokens_noi_dung)
                tong_idf_q = sum(self._idf.get(t, 1.0) for t in q_doc_lap) or 1.0
                khop_doc_lap = {
                    t
                    for t in q_doc_lap
                    if t in top_set or ("_" in t and any(sub in top_set for sub in t.split("_") if len(sub) >= 2))
                }
                khop_idf_q = sum(
                    self._idf.get(t, 1.0) * (1.0 if t in top_set else 0.65)
                    for t in khop_doc_lap
                )
                do_phu_top1 = khop_idf_q / tong_idf_q
                dense_top1 = float(dense_scores[top_i])
                if not sh_thay_the and dense_top1 < 0.28 and (do_phu_top1 < self.nguong_tu_choi or len(khop_doc_lap) < 2):
                    tu_choi = True
                    ly_do_tu_choi = (
                        "Câu hỏi nằm ngoài phạm vi kho văn bản quy phạm pháp luật (độ phủ từ khóa và ngữ nghĩa dưới ngưỡng an toàn)."
                    )

        ket_qua_list: List[Dict[str, Any]] = []
        if not tu_choi:
            for rank_pos, idx_i in enumerate(da_xep_hang[: max(1, int(top_k))], start=1):
                nut = self._tai_lieu[idx_i]
                con_hl = self.do_thi.dieu_khoan_con_hieu_luc(nut, ngay_tra_cuu=moc)
                g_info = thong_tin_do_thi_map.get(nut.ma_nut)
                ket_qua_list.append(
                    {
                        "hang": rank_pos,
                        "id": nut.ma_nut,
                        "so_hieu": nut.so_hieu,
                        "ten_van_ban": nut.ten_van_ban,
                        "dieu": nut.dieu,
                        "khoan": nut.khoan,
                        "tieu_de": nut.tieu_de,
                        "linh_vuc": nut.linh_vuc,
                        "noi_dung": nut.noi_dung,
                        "hieu_luc_tu": nut.hieu_luc_tu,
                        "het_hieu_luc_tu": nut.het_hieu_luc_tu,
                        "con_hieu_luc": con_hl,
                        "diem": round(float(diem_cuoi[idx_i]), 6),
                        "diem_bm25": round(float(bm25_scores[idx_i]), 6),
                        "diem_dense": round(float(dense_scores[idx_i]), 6),
                        "diem_do_thi": round(float(diem_do_thi_map.get(nut.ma_nut, 0.0)), 6),
                        "tu_do_thi_da_buoc": g_info is not None,
                        "duong_dan_do_thi": g_info["duong_dan"] if g_info is not None else [],
                        "metadata": nut.to_dict(),
                    }
                )

        return {
            "cau_hoi": cau_hoi,
            "che_do": che_do,
            "ngay_tra_cuu": moc_iso,
            "phan_tich_thoi_gian": kq_tg.to_dict(),
            "tu_choi_tra_loi": tu_choi,
            "ly_do_tu_choi": ly_do_tu_choi,
            "canh_bao_thoi_gian": canh_bao_tg,
            "van_ban_het_hieu_luc_trong_hoi": sorted(sh_het_hl_trong_hoi),
            "van_ban_thay_the_goi_y": sorted(sh_thay_the),
            "ket_qua": ket_qua_list,
            "thoi_gian_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def tra_loi_phap_ly(
        self,
        cau_hoi: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        top_k: int = 5,
        che_do: str = "full",
        ham_sinh_llm: Optional[Callable[[str], str]] = None,
        ngay_mac_dinh: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Thực thi đầy đủ chuỗi Truy hồi Thời gian-Đồ thị → Sinh câu trả lời (hoặc từ chối) → Kiểm chứng trích dẫn 3 tầng."""
        kq_truy = self.truy_hoi(
            cau_hoi=cau_hoi,
            ngay_tra_cuu=ngay_tra_cuu,
            top_k=top_k,
            che_do=che_do,
            ngay_mac_dinh=ngay_mac_dinh,
        )
        moc_iso = str(kq_truy["ngay_tra_cuu"])
        nguon = list(kq_truy["ket_qua"])

        if kq_truy["tu_choi_tra_loi"] or not nguon:
            tra_loi = (
                f"Từ chối trả lời do thiếu căn cứ pháp lý hợp lệ tại ngày {moc_iso}: "
                f"{kq_truy.get('ly_do_tu_choi') or 'Không tìm thấy điều khoản phù hợp trong kho tri thức.'}"
            )
            kc = self.bo_kiem_chung.kiem_chung(
                cau_tra_loi=tra_loi,
                nguon_truy_hoi=[],
                ngay_tra_cuu=moc_iso,
                cau_hoi=cau_hoi,
            )
            return {
                **kq_truy,
                "tra_loi": tra_loi,
                "kiem_chung_trich_dan": kc.to_dict(),
            }

        if ham_sinh_llm is not None:
            ngu_canh_lines = [
                f"- [{item['so_hieu']} - {item['dieu']} ({item['ten_van_ban']})]: {item['noi_dung']}"
                for item in nguon[:3]
            ]
            canh_bao_txt = "\n".join(kq_truy.get("canh_bao_thoi_gian", []))
            prompt = (
                f"Bạn là trợ lý pháp lý Việt Nam. Chỉ trả lời dựa trên các điều khoản còn hiệu lực tại ngày {moc_iso} dưới đây, "
                f"trích rõ số hiệu văn bản và Điều luật, không bịa đặt số liệu.\n"
                f"{f'Cảnh báo hiệu lực: {canh_bao_txt}' if canh_bao_txt else ''}\n"
                f"Căn cứ pháp lý:\n" + "\n".join(ngu_canh_lines) + f"\n\nCâu hỏi: {cau_hoi}\nTrả lời:"
            )
            tra_loi = str(ham_sinh_llm(prompt)).strip()
        else:
            cac_doan: List[str] = []
            if kq_truy.get("canh_bao_thoi_gian"):
                cac_doan.append(" ".join(kq_truy["canh_bao_thoi_gian"]))
            chinh = nguon[0]
            cac_doan.append(
                f"Tại ngày {moc_iso}, căn cứ {chinh['dieu']} ({chinh['so_hieu']} - {chinh['ten_van_ban']}): "
                f"{chinh['noi_dung']}"
            )
            if len(nguon) > 1 and (nguon[1].get("tu_do_thi_da_buoc") or nguon[1]["so_hieu"] != chinh["so_hieu"]):
                phu = nguon[1]
                cac_doan.append(
                    f"Văn bản hướng dẫn/liên quan tại {phu['dieu']} ({phu['so_hieu']} - {phu['ten_van_ban']}): "
                    f"{phu['noi_dung']}"
                )
            tra_loi = "\n".join(cac_doan)

        kc = self.bo_kiem_chung.kiem_chung(
            cau_tra_loi=tra_loi,
            nguon_truy_hoi=nguon,
            ngay_tra_cuu=moc_iso,
            cau_hoi=cau_hoi,
        )
        return {
            **kq_truy,
            "tra_loi": tra_loi,
            "kiem_chung_trich_dan": kc.to_dict(),
        }


def tao_dong_co_phap_ly_thoi_gian_mac_dinh(
    gop_kho_phap_ly_mo: bool = True,
) -> BoTruyHoiPhapLyThoiGian:
    """Khởi tạo nhanh `BoTruyHoiPhapLyThoiGian` với đầy đủ 60 điều khoản đa phiên bản, 38 cạnh đồ thị và 100 điều luật mở."""
    from vietnamese_ai.rag.legal.open_corpus import (
        tai_bo_phap_ly_mo,
        tai_bo_phap_ly_thoi_gian,
        tai_canh_do_thi_phap_ly_thoi_gian,
    )

    dieu_khoan = list(tai_bo_phap_ly_thoi_gian())
    da_co = {str(x.get("id")) for x in dieu_khoan}
    if gop_kho_phap_ly_mo:
        for item in tai_bo_phap_ly_mo():
            ma = str(item.get("id", ""))
            if not ma or ma in da_co:
                ma = f"mo_{ma}"
            da_co.add(ma)
            ban_sao = dict(item)
            ban_sao["id"] = ma
            dieu_khoan.append(ban_sao)
    engine = BoTruyHoiPhapLyThoiGian()
    engine.nap_tu_danh_sach(dieu_khoan, tai_canh_do_thi_phap_ly_thoi_gian())
    return engine


TemporalGraphLegalRAG = BoTruyHoiPhapLyThoiGian

__all__ = [
    "BoTruyHoiPhapLyThoiGian",
    "TemporalGraphLegalRAG",
    "tao_dong_co_phap_ly_thoi_gian_mac_dinh",
]
