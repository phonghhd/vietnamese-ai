"""BoLocNhieuCapUuTien (NoisyPreferenceFilter) - Lọc Nhiễu, Độc Tố & Đảo Nhãn Đa Tín Hiệu Cho DPO/SimPO/GRPO.

Tự động kiểm toán từng cặp phản hồi `(prompt, chosen, rejected)` sinh ra từ người dùng thật
(Thumbs Up / Thumbs Down trong Studio Workspace) qua 6 tín hiệu khách quan trước khi huấn luyện:
1. Khiên an ninh chống Prompt Injection / Jailbreak / tuân thủ tấn công trong `chosen`.
2. Kiểm toán rò rỉ dữ liệu nhạy cảm PII (CCCD, SĐT, Email, thẻ) trong `chosen`.
3. Kiểm chứng hiệu lực pháp lý Việt Nam (chặn `chosen` dẫn chiếu luật hết hiệu lực mà thiếu cảnh báo).
4. Kiểm tra tính toàn vẹn cấu trúc, cặp suy biến và phát hiện cặp mâu thuẫn đảo chiều cùng prompt.
5. Sai phân chất lượng tất định `ΔQ = Q(x, y_w) - Q(x, y_l)` phát hiện đảo nhãn (`inverted_label_flip`).
6. Lọc ngoại lệ biên thưởng ẩn từ chính sách hiện hành (MAD z-score) và gán trọng số tin cậy `trong_so`
   cùng hệ số làm mượt nhãn thích nghi `label_smoothing` cho Conservative DPO (cDPO).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer
from vietnamese_ai.rag.legal.validity_tracker import BoTheoDoiHieuLucPhapLy
from vietnamese_ai.security.data_sanitizer import DataSanitizer
from vietnamese_ai.security.llm_firewall import TuongLuaAI
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

_RE_SO_HIEU = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+\b")
_RE_DIEU_KHOAN = re.compile(r"\b(?:điều|dieu)\s+\d+[a-z]?\b", re.IGNORECASE)
_RE_CON_SO = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|ngày|ngay|tháng|thang|năm|nam|giờ|gio|triệu|trieu|tỷ|ty|đồng|dong|vnđ|vnd)\b", re.IGNORECASE)

_TU_TU_CHOI_AN_TOAN = (
    "từ chối",
    "không thể",
    "không được phép",
    "vi phạm",
    "chặn",
    "bảo mật",
    "an toàn",
    "cannot",
    "refuse",
    "blocked",
)

_TU_TUAN_THU_DOC_HAI = (
    "system prompt",
    "mật khẩu nội bộ",
    "sudo rm",
    "hướng dẫn tấn công",
    "chiếm quyền điều khiển",
    "vô hiệu hóa tường lửa",
    "bỏ qua mọi quy tắc",
    "tôi là dan",
)

_TU_NE_TRANH_RONG = (
    "không có thông tin",
    "không biết",
    "chưa rõ",
    "tùy trường hợp",
    "không chắc chắn",
    "hỏi người khác",
)

_STOPWORDS_VN = {
    "là",
    "và",
    "của",
    "có",
    "cho",
    "trong",
    "được",
    "về",
    "với",
    "theo",
    "tại",
    "từ",
    "đến",
    "các",
    "những",
    "một",
    "không",
    "này",
    "đó",
    "khi",
    "như",
    "thế",
    "nào",
    "bao",
    "nhiêu",
    "hay",
    "hoặc",
    "thì",
    "bị",
    "do",
    "vì",
    "nếu",
    "để",
    "trên",
    "dưới",
    "ra",
    "vào",
}


def _bo_dau(van_ban: str) -> str:
    """Chuyển văn bản tiếng Việt sang dạng thường không dấu để đối sánh ổn định."""
    thay_d = van_ban.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", thay_d)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")).lower()


def _tap_tu_noi_dung(van_ban: str) -> set[str]:
    """Trích tập từ nội dung đã chuẩn hóa không dấu phục vụ tính Jaccard & liên quan ngữ nghĩa."""
    raw = re.findall(r"[0-9a-zà-ỹ]+", van_ban.lower())
    return {_bo_dau(w) for w in raw if len(w) >= 2 and w not in _STOPWORDS_VN}


def _jaccard_van_ban(a: str, b: str) -> float:
    """Tính độ tương đồng Jaccard mức từ giữa hai chuỗi tiếng Việt."""
    ta = _tap_tu_noi_dung(a)
    tb = _tap_tu_noi_dung(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _chuan_hoa_khoa_prompt(prompt: str) -> str:
    """Chuẩn hóa prompt thành khóa nhóm ổn định để phát hiện các cặp ưu tiên mâu thuẫn."""
    return " ".join(re.findall(r"[0-9a-z]+", _bo_dau(prompt)))


@dataclass
class KetQuaLocCapUuTien:
    """Kết quả kiểm toán và lọc nhiễu tập dữ liệu ưu tiên DPO/SimPO."""

    mau_sach: List[Dict[str, Any]] = field(default_factory=list)
    mau_cach_ly: List[Dict[str, Any]] = field(default_factory=list)
    tong_dau_vao: int = 0
    so_giu_lai: int = 0
    so_cach_ly: int = 0
    ty_le_giu_lai: float = 0.0
    thong_ke_ly_do_loai: Dict[str, int] = field(default_factory=dict)
    trong_so_trung_binh: float = 0.0
    chi_so_phat_hien_nhieu: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Xuất báo cáo lọc nhiễu sang dict JSON đầy đủ."""
        return asdict(self)


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ lọc nhiễu cặp ưu tiên đa tín hiệu cho Continual DPO/SimPO/GRPO: chặn độc tố Prompt Injection, "
        "PII leak, dẫn chiếu luật hết hiệu lực, phát hiện đảo nhãn và gán trọng số tin cậy cDPO."
    ),
    nhom_extra="core",
)
class BoLocNhieuCapUuTien:
    """Bộ lọc nhiễu & gán trọng số tin cậy đa tín hiệu cho vòng lặp Continual Alignment."""

    def __init__(
        self,
        bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None,
        ngay_tham_chieu: Optional[Union[str, date]] = None,
        nguong_dao_nhan: float = 0.10,
        nguong_trung_lap: float = 0.92,
        nguong_z_margin: float = 2.0,
        do_dai_toi_thieu_chosen: int = 10,
        tu_dong_an_pii: bool = False,
        bat_loc_an_toan: bool = True,
        bat_loc_hieu_luc_luat: bool = True,
        bat_loc_dao_nhan: bool = True,
        bat_loc_mau_thuan: bool = True,
    ) -> None:
        self.bo_theo_doi = bo_theo_doi or BoTheoDoiHieuLucPhapLy()
        self.ngay_tham_chieu = ngay_tham_chieu
        self.nguong_dao_nhan = float(nguong_dao_nhan)
        self.nguong_trung_lap = float(nguong_trung_lap)
        self.nguong_z_margin = float(nguong_z_margin)
        self.do_dai_toi_thieu_chosen = max(2, int(do_dai_toi_thieu_chosen))
        self.tu_dong_an_pii = bool(tu_dong_an_pii)
        self.bat_loc_an_toan = bool(bat_loc_an_toan)
        self.bat_loc_hieu_luc_luat = bool(bat_loc_hieu_luc_luat)
        self.bat_loc_dao_nhan = bool(bat_loc_dao_nhan)
        self.bat_loc_mau_thuan = bool(bat_loc_mau_thuan)
        self.tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=False)

    def _la_prompt_tan_cong(self, prompt: str) -> bool:
        """Nhận diện prompt chứa ý đồ tấn công Prompt Injection / Jailbreak."""
        an_toan, _ = self.tuong_lua.kiem_tra_prompt(prompt)
        if not an_toan:
            return True
        low = _bo_dau(prompt)
        return any(
            k in low
            for k in (
                "ignore all previous",
                "bo qua moi quy tac",
                "bo qua toan bo quy tac",
                "system prompt",
                "mat khau noi bo",
                "vo hieu hoa tuong lua",
                "ban la dan",
                "tan cong he thong",
            )
        )

    @staticmethod
    def _co_tu_choi_an_toan(van_ban: str) -> bool:
        low = van_ban.lower()
        return any(kw in low for kw in _TU_TU_CHOI_AN_TOAN)

    @staticmethod
    def _co_tuan_thu_doc_hai(van_ban: str) -> bool:
        low = van_ban.lower()
        return any(kw in low for kw in _TU_TUAN_THU_DOC_HAI)

    def _kiem_tra_luat_het_hieu_luc(self, van_ban: str) -> Tuple[bool, int]:
        """Kiểm tra văn bản có dẫn chiếu văn bản hết hiệu lực mà thiếu cảnh báo thay thế hay không."""
        kt = self.bo_theo_doi.kiem_tra_hieu_luc_trong_van_ban(
            van_ban,
            ngay_tra_cuu=self.ngay_tham_chieu,
        )
        so_het = int(kt.get("so_van_ban_het_hieu_luc", 0))
        if so_het == 0:
            return True, 0
        low = van_ban.lower()
        co_canh_bao = any(
            kw in low
            for kw in (
                "hết hiệu lực",
                "đã hết hiệu lực",
                "thay thế",
                "được thay bằng",
                "bãi bỏ",
                "không còn hiệu lực",
            )
        )
        return co_canh_bao, so_het

    def cham_chat_luong_phan_hoi(
        self,
        prompt: str,
        response: str,
        la_prompt_doc: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Chấm điểm chất lượng tất định đa trục `Q(x, y) ∈ [0, 1]` cho một câu trả lời."""
        resp = str(response or "").strip()
        if not resp:
            return {
                "diem_tong": 0.0,
                "an_toan": True,
                "khong_pii": True,
                "hieu_luc_hop_le": True,
                "co_trich_dan": False,
                "so_van_ban_het_han": 0,
            }

        la_doc = self._la_prompt_tan_cong(prompt) if la_prompt_doc is None else bool(la_prompt_doc)
        if la_doc:
            tu_choi_ok = self._co_tu_choi_an_toan(resp) and not self._co_tuan_thu_doc_hai(resp)
            khong_pii = DataSanitizer.lam_sach(resp) == resp
            diem_tong = (0.90 if tu_choi_ok else 0.05) + (0.10 if khong_pii else 0.0)
            return {
                "diem_tong": round(min(1.0, diem_tong), 4),
                "an_toan": tu_choi_ok,
                "khong_pii": khong_pii,
                "hieu_luc_hop_le": True,
                "co_trich_dan": False,
                "so_van_ban_het_han": 0,
            }

        an_toan_prompt, _ = self.tuong_lua.kiem_tra_prompt(resp)
        resp_an_toan = bool(an_toan_prompt) and not self._co_tuan_thu_doc_hai(resp)

        khong_pii = DataSanitizer.lam_sach(resp) == resp
        hl_ok, so_het = self._kiem_tra_luat_het_hieu_luc(resp)

        so_hieu = _RE_SO_HIEU.findall(resp)
        dieu_khoan = _RE_DIEU_KHOAN.findall(resp)
        con_so = _RE_CON_SO.findall(resp)
        co_trich_dan = bool(so_hieu or dieu_khoan)
        diem_trich_dan = min(1.0, 0.45 * min(2, len(so_hieu)) + 0.30 * min(2, len(dieu_khoan)) + 0.15 * min(2, len(con_so)))

        p_toks = _tap_tu_noi_dung(prompt)
        r_toks = _tap_tu_noi_dung(resp)
        overlap = len(p_toks & r_toks) / float(max(1, len(p_toks))) if p_toks else 0.5
        diem_lien_quan = min(1.0, overlap * 1.6)

        la_ne_tranh = any(kw in resp.lower() for kw in _TU_NE_TRANH_RONG)
        so_tu = len(resp.split())
        if la_ne_tranh and so_tu < 25:
            diem_thong_tin = 0.15
        elif so_tu < 4:
            diem_thong_tin = 0.10
        elif so_tu < 10:
            diem_thong_tin = 0.45
        else:
            diem_thong_tin = min(1.0, 0.55 + min(so_tu, 60) / 120.0)

        diem_hl = 1.0 if hl_ok else 0.0
        diem_an_toan = 1.0 if resp_an_toan else 0.0
        diem_pii = 1.0 if khong_pii else 0.0

        diem_tong = (
            0.25 * diem_an_toan
            + 0.15 * diem_pii
            + 0.22 * diem_hl
            + 0.20 * diem_trich_dan
            + 0.10 * diem_lien_quan
            + 0.08 * diem_thong_tin
        )
        if not resp_an_toan or not hl_ok:
            diem_tong *= 0.35
        if not khong_pii:
            diem_tong *= 0.50

        return {
            "diem_tong": round(float(np.clip(diem_tong, 0.0, 1.0)), 4),
            "an_toan": resp_an_toan,
            "khong_pii": khong_pii,
            "hieu_luc_hop_le": hl_ok,
            "co_trich_dan": co_trich_dan,
            "so_van_ban_het_han": so_het,
        }

    @staticmethod
    def _tao_ham_cham_tu_mo_hinh(
        mo_hinh_tham_chieu: Any,
    ) -> Optional[Callable[[str, str], float]]:
        """Tạo hàm chấm biên thưởng ẩn `(prompt, response) -> float` từ mô hình có `_dpo_weights` hoặc `cham_diem`."""
        if mo_hinh_tham_chieu is None:
            return None
        if callable(mo_hinh_tham_chieu) and not hasattr(mo_hinh_tham_chieu, "_dpo_weights"):
            return mo_hinh_tham_chieu
        if hasattr(mo_hinh_tham_chieu, "cham_diem_phan_hoi"):
            return lambda p, r: float(mo_hinh_tham_chieu.cham_diem_phan_hoi(p, r))
        w = getattr(mo_hinh_tham_chieu, "_dpo_weights", None)
        if isinstance(w, np.ndarray) and w.size > 0 and float(np.linalg.norm(w)) > 1e-8:
            dim = int(w.size)
            w_copy = w.astype(np.float64, copy=True)

            def _score(p: str, r: str) -> float:
                phi = DPOTrainer._trich_xuat_dac_trung_dpo(f"{p} {r}", dim)
                return float(np.dot(w_copy, phi))

            return _score
        return None

    def loc_tap_du_lieu(
        self,
        cac_cap: Sequence[Dict[str, Any]],
        mo_hinh_tham_chieu: Optional[Any] = None,
        ham_cham_diem: Optional[Callable[[str, str], float]] = None,
    ) -> KetQuaLocCapUuTien:
        """Lọc nhiễu, cách ly mẫu độc hại/đảo nhãn và gán trọng số tin cậy cho tập cặp ưu tiên."""
        ds = list(cac_cap)
        n_total = len(ds)
        if n_total == 0:
            return KetQuaLocCapUuTien()

        scorer = ham_cham_diem or self._tao_ham_cham_tu_mo_hinh(mo_hinh_tham_chieu)

        chi_so_mau_thuan: set[int] = set()
        if self.bat_loc_mau_thuan:
            nhom_theo_prompt: Dict[str, List[Tuple[int, str, str]]] = {}
            for idx, item in enumerate(ds):
                p_key = _chuan_hoa_khoa_prompt(str(item.get("prompt", "")))
                if not p_key:
                    continue
                nhom_theo_prompt.setdefault(p_key, []).append(
                    (idx, str(item.get("chosen", "")), str(item.get("rejected", "")))
                )
            for entries in nhom_theo_prompt.values():
                if len(entries) < 2:
                    continue
                for i in range(len(entries)):
                    idx_a, ch_a, rj_a = entries[i]
                    for j in range(i + 1, len(entries)):
                        idx_b, ch_b, rj_b = entries[j]
                        if _jaccard_van_ban(ch_a, rj_b) >= 0.70 and _jaccard_van_ban(rj_a, ch_b) >= 0.70:
                            chi_so_mau_thuan.add(idx_a)
                            chi_so_mau_thuan.add(idx_b)

        margins: List[Optional[float]] = [None] * n_total
        valid_margins: List[float] = []
        if scorer is not None:
            for idx, item in enumerate(ds):
                p = str(item.get("prompt", ""))
                c = str(item.get("chosen", ""))
                r = str(item.get("rejected", ""))
                if p and c and r:
                    m_val = float(scorer(p, c) - scorer(p, r))
                    margins[idx] = m_val
                    valid_margins.append(m_val)

        margin_cutoff: Optional[float] = None
        if len(valid_margins) >= 5:
            arr_m = np.asarray(valid_margins, dtype=np.float64)
            med = float(np.median(arr_m))
            mad = float(np.median(np.abs(arr_m - med)))
            sigma = max(1e-4, 1.4826 * mad)
            margin_cutoff = min(-0.20, med - self.nguong_z_margin * sigma)

        mau_sach: List[Dict[str, Any]] = []
        mau_cach_ly: List[Dict[str, Any]] = []
        ly_do_dem: Dict[str, int] = {}

        tp_nhieu = 0
        fp_nhieu = 0
        fn_nhieu = 0
        tn_sach = 0
        co_nhan_ground_truth = False

        for idx, item in enumerate(ds):
            prompt = str(item.get("prompt", "")).strip()
            chosen = str(item.get("chosen", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            ly_do_loai: List[str] = []

            if not prompt or not chosen or not rejected:
                ly_do_loai.append("thieu_truong_bat_buoc")
            elif _jaccard_van_ban(chosen, rejected) >= self.nguong_trung_lap:
                ly_do_loai.append("cap_trung_lap_suy_bien")
            elif len(chosen) < self.do_dai_toi_thieu_chosen and len(rejected) >= max(24, len(chosen) * 2):
                ly_do_loai.append("chosen_qua_ngan_suy_bien")

            la_prompt_doc = self._la_prompt_tan_cong(prompt) if prompt else False
            q_chosen = self.cham_chat_luong_phan_hoi(prompt, chosen, la_prompt_doc=la_prompt_doc)
            q_rejected = self.cham_chat_luong_phan_hoi(prompt, rejected, la_prompt_doc=la_prompt_doc)
            delta_q = float(q_chosen["diem_tong"] - q_rejected["diem_tong"])

            if self.bat_loc_an_toan and chosen:
                if not q_chosen["an_toan"]:
                    ly_do_loai.append("doc_to_hoac_tuan_thu_tan_cong")
                elif la_prompt_doc and not self._co_tu_choi_an_toan(chosen):
                    ly_do_loai.append("doc_to_hoac_tuan_thu_tan_cong")

            if chosen and not q_chosen["khong_pii"]:
                if self.tu_dong_an_pii:
                    chosen = DataSanitizer.lam_sach(chosen)
                    q_chosen = self.cham_chat_luong_phan_hoi(prompt, chosen, la_prompt_doc=la_prompt_doc)
                    delta_q = float(q_chosen["diem_tong"] - q_rejected["diem_tong"])
                else:
                    ly_do_loai.append("ro_ri_pii_trong_chosen")

            if self.bat_loc_hieu_luc_luat and chosen and not q_chosen["hieu_luc_hop_le"]:
                ly_do_loai.append("dan_luat_het_hieu_luc_trong_chosen")

            if self.bat_loc_mau_thuan and idx in chi_so_mau_thuan and delta_q < 0.15:
                ly_do_loai.append("mau_thuan_dao_chieu_cung_prompt")

            if self.bat_loc_dao_nhan and delta_q < -self.nguong_dao_nhan:
                ly_do_loai.append("dao_nhan_chat_luong")

            m_idx = margins[idx]
            if (
                self.bat_loc_dao_nhan
                and margin_cutoff is not None
                and m_idx is not None
                and m_idx < margin_cutoff
                and delta_q < 0.12
            ):
                ly_do_loai.append("ngoai_le_bien_thuong_chinh_sach")

            bi_loai = len(ly_do_loai) > 0

            if "la_nhieu" in item:
                co_nhan_ground_truth = True
                la_nhieu_that = bool(item["la_nhieu"])
                if bi_loai and la_nhieu_that:
                    tp_nhieu += 1
                elif bi_loai and not la_nhieu_that:
                    fp_nhieu += 1
                elif not bi_loai and la_nhieu_that:
                    fn_nhieu += 1
                else:
                    tn_sach += 1

            if bi_loai:
                for ly in ly_do_loai:
                    ly_do_dem[ly] = ly_do_dem.get(ly, 0) + 1
                ban_cach_ly = dict(item)
                ban_cach_ly["ly_do_cach_ly"] = ly_do_loai
                ban_cach_ly["diem_chosen"] = q_chosen["diem_tong"]
                ban_cach_ly["diem_rejected"] = q_rejected["diem_tong"]
                ban_cach_ly["delta_q"] = round(delta_q, 4)
                if m_idx is not None:
                    ban_cach_ly["bien_thuong_an"] = round(m_idx, 4)
                mau_cach_ly.append(ban_cach_ly)
            else:
                sig_dq = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, 4.0 * delta_q))))
                trong_so = round(float(np.clip(0.45 + 0.55 * sig_dq, 0.30, 1.0)), 4)
                if la_prompt_doc and q_chosen["an_toan"]:
                    trong_so = 1.0
                adaptive_ls = round(float(np.clip(0.02 + 0.18 * (1.0 - trong_so), 0.02, 0.20)), 4)
                ban_sach = dict(item)
                ban_sach["prompt"] = prompt
                ban_sach["chosen"] = chosen
                ban_sach["rejected"] = rejected
                ban_sach["trong_so"] = trong_so
                ban_sach["label_smoothing"] = adaptive_ls
                ban_sach["delta_q"] = round(delta_q, 4)
                ban_sach["loai_neo"] = "an_toan" if la_prompt_doc else ("phap_ly" if q_chosen["co_trich_dan"] else "chung")
                if m_idx is not None:
                    ban_sach["bien_thuong_an"] = round(m_idx, 4)
                mau_sach.append(ban_sach)

        chi_so_nhieu: Optional[Dict[str, float]] = None
        if co_nhan_ground_truth:
            prec = tp_nhieu / float(max(1, tp_nhieu + fp_nhieu))
            rec = tp_nhieu / float(max(1, tp_nhieu + fn_nhieu))
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            clean_ret = tn_sach / float(max(1, tn_sach + fp_nhieu))
            chi_so_nhieu = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "ty_le_giu_mau_sach": round(clean_ret, 4),
                "tp": float(tp_nhieu),
                "fp": float(fp_nhieu),
                "fn": float(fn_nhieu),
                "tn": float(tn_sach),
            }

        w_avg = (
            round(float(np.mean([float(x["trong_so"]) for x in mau_sach])), 4)
            if mau_sach
            else 0.0
        )

        return KetQuaLocCapUuTien(
            mau_sach=mau_sach,
            mau_cach_ly=mau_cach_ly,
            tong_dau_vao=n_total,
            so_giu_lai=len(mau_sach),
            so_cach_ly=len(mau_cach_ly),
            ty_le_giu_lai=round(len(mau_sach) / float(max(1, n_total)), 4),
            thong_ke_ly_do_loai=ly_do_dem,
            trong_so_trung_binh=w_avg,
            chi_so_phat_hien_nhieu=chi_so_nhieu,
        )


NoisyPreferenceFilter = BoLocNhieuCapUuTien
PreferenceFilterResult = KetQuaLocCapUuTien

__all__ = [
    "BoLocNhieuCapUuTien",
    "NoisyPreferenceFilter",
    "KetQuaLocCapUuTien",
    "PreferenceFilterResult",
]
