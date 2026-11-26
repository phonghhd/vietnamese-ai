"""VongLapCanChinhAnToan (EvalGatedContinualAligner) - Vòng Lặp Căn Chỉnh Liên Tục Có Cổng Kiểm Định Chống Hồi Quy.

Giải quyết trọn vẹn bài toán Continual Alignment (DPO/SimPO) từ phản hồi người dùng thực tế mà không bao giờ
làm "hỏng ngầm" mô hình Production:
1. `BoLocNhieuCapUuTien`: lọc 6 tín hiệu (độc tố, PII, luật hết hiệu lực, mâu thuẫn, đảo nhãn, biên thưởng).
2. `BoDemOnTapNeo`: trộn mẫu ôn tập phân tầng (an toàn / pháp lý / chung) chống quên kiến thức cũ.
3. `DPOTrainer` có trọng số tin cậy (`trong_so`), làm mượt nhãn thích nghi (cDPO) và neo KL (`he_so_neo_kl`).
4. `CongKiemDinhMoHinh`: cổng kiểm định 4 trục + Paired Bootstrap 95% CI quyết định `PROMOTE` hoặc `AUTO-ROLLBACK`
   nguyên tử về bản snapshot đã ký SHA-256 trước vòng cập nhật.
"""

from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer
from vietnamese_ai.fine_tuning.eval_gate import CongKiemDinhMoHinh, KetQuaKiemDinhMoHinh
from vietnamese_ai.fine_tuning.preference_filter import BoLocNhieuCapUuTien, KetQuaLocCapUuTien
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


def _chuan_hoa_key(text: str) -> str:
    thay_d = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", thay_d)
    kd = unicodedata.normalize("NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")).lower()
    return " ".join(re.findall(r"[0-9a-z]+", kd))


class ChinhSachUuTienTiengViet:
    """Chính sách phản hồi tiếng Việt huấn luyện trực tiếp bằng gradient DPO/SimPO trên CPU (không cần GPU).

    Dùng vector trọng số `_dpo_weights` (256 chiều) tương thích trực tiếp với `DPOTrainer` để chấm điểm
    và chọn phản hồi tối ưu giữa các ứng viên trả lời cho từng câu hỏi, đồng thời hỗ trợ snapshot/rollback
    nguyên tử và mã băm SHA-256 kiểm toán.
    """

    def __init__(
        self,
        ngan_hang_ung_vien: Optional[Dict[str, List[str]]] = None,
        dim: int = 256,
        trong_so_ban_dau: Optional[np.ndarray] = None,
        tra_loi_mac_dinh: str = "Tôi từ chối thực hiện yêu cầu chưa đủ căn cứ pháp lý hoặc không an toàn.",
    ) -> None:
        self.dim = int(dim)
        self._dpo_weights: np.ndarray = (
            np.asarray(trong_so_ban_dau, dtype=np.float64).copy()
            if trong_so_ban_dau is not None
            else np.zeros(self.dim, dtype=np.float64)
        )
        self.tra_loi_mac_dinh = str(tra_loi_mac_dinh)
        self._ngan_hang_goc: Dict[str, List[str]] = {}
        self._ngan_hang_chuan: Dict[str, List[str]] = {}
        if ngan_hang_ung_vien:
            for q, cands in ngan_hang_ung_vien.items():
                self.dang_ky_ung_vien(q, cands)

    def dang_ky_ung_vien(self, cau_hoi: str, danh_sach_ung_vien: Sequence[str]) -> None:
        """Đăng ký tập ứng viên trả lời cho một câu hỏi kiểm định hoặc hội thoại."""
        cands = [str(c).strip() for c in danh_sach_ung_vien if str(c).strip()]
        if not cands:
            return
        self._ngan_hang_goc[cau_hoi] = list(cands)
        self._ngan_hang_chuan[_chuan_hoa_key(cau_hoi)] = list(cands)

    def cham_diem_phan_hoi(self, cau_hoi: str, phan_hoi: str) -> float:
        """Tính điểm log-linear `w · phi(prompt, response)` của chính sách hiện tại."""
        phi = DPOTrainer._trich_xuat_dac_trung_dpo(f"{cau_hoi} {phan_hoi}", self.dim)
        return float(np.dot(self._dpo_weights, phi))

    def _tim_ung_vien(self, cau_hoi: str) -> List[str]:
        if cau_hoi in self._ngan_hang_goc:
            return self._ngan_hang_goc[cau_hoi]
        key = _chuan_hoa_key(cau_hoi)
        if key in self._ngan_hang_chuan:
            return self._ngan_hang_chuan[key]
        q_toks = set(key.split())
        if not q_toks or not self._ngan_hang_chuan:
            return []
        best_score = 0.0
        best_cands: List[str] = []
        for k_ref, cands in self._ngan_hang_chuan.items():
            r_toks = set(k_ref.split())
            if not r_toks:
                continue
            ov = len(q_toks & r_toks) / float(len(q_toks | r_toks))
            if ov > best_score:
                best_score = ov
                best_cands = cands
        return best_cands if best_score >= 0.45 else []

    def __call__(self, cau_hoi: str) -> str:
        """Chọn phản hồi có điểm chính sách cao nhất cho câu hỏi đầu vào."""
        cands = self._tim_ung_vien(cau_hoi)
        if not cands:
            return self.tra_loi_mac_dinh
        if float(np.linalg.norm(self._dpo_weights)) < 1e-12:
            return cands[0]
        scored = [(self.cham_diem_phan_hoi(cau_hoi, c), idx, c) for idx, c in enumerate(cands)]
        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        return scored[0][2]

    def sao_chep(self) -> "ChinhSachUuTienTiengViet":
        """Tạo bản sao độc lập hoàn toàn để huấn luyện ứng viên mà không chạm vào Production."""
        clone = ChinhSachUuTienTiengViet(
            ngan_hang_ung_vien=self._ngan_hang_goc,
            dim=self.dim,
            trong_so_ban_dau=self._dpo_weights.copy(),
            tra_loi_mac_dinh=self.tra_loi_mac_dinh,
        )
        return clone

    def ma_bam_trang_thai(self) -> str:
        """Sinh mã băm SHA-256 rút gọn xác nhận toàn vẹn trọng số checkpoint."""
        return hashlib.sha256(self._dpo_weights.astype(np.float64).tobytes()).hexdigest()[:16]


class BoDemOnTapNeo:
    """Bộ đệm Ôn tập Neo phân tầng (Stratified Experience Replay Buffer) chống quên kiến thức cũ."""

    def __init__(self, suc_chua_toi_da: int = 120) -> None:
        self.suc_chua_toi_da = max(6, int(suc_chua_toi_da))
        self._kho: Dict[str, List[Dict[str, Any]]] = {
            "an_toan": [],
            "phap_ly": [],
            "chung": [],
        }

    def them_cac_mau(self, cac_mau: Sequence[Dict[str, Any]]) -> int:
        """Thêm các cặp ưu tiên đã được kiểm chứng vào kho ôn tập, giữ lại mẫu có độ tin cậy cao nhất."""
        da_them = 0
        for item in cac_mau:
            nhom = str(item.get("loai_neo", "chung"))
            if nhom not in self._kho:
                nhom = "chung"
            key = _chuan_hoa_key(f"{item.get('prompt', '')} || {item.get('chosen', '')}")
            bucket = self._kho[nhom]
            da_co = any(
                _chuan_hoa_key(f"{x.get('prompt', '')} || {x.get('chosen', '')}") == key
                for x in bucket
            )
            if not da_co:
                ban_sao = dict(item)
                ban_sao["la_mau_on_tap"] = True
                bucket.append(ban_sao)
                da_them += 1
            bucket.sort(
                key=lambda x: (float(x.get("trong_so", 0.5)), float(x.get("delta_q", 0.0))),
                reverse=True,
            )
            han_muc = max(2, self.suc_chua_toi_da // 3)
            if len(bucket) > han_muc:
                del bucket[han_muc:]
        return da_them

    def lay_mau_on_tap(self, so_luong: int, seed: int = 42) -> List[Dict[str, Any]]:
        """Rút mẫu ôn tập cân bằng giữa 3 tầng (an toàn, pháp lý, chung)."""
        k_req = max(0, int(so_luong))
        if k_req == 0 or self.so_luong == 0:
            return []
        rng = np.random.default_rng(int(seed))
        ket_qua: List[Dict[str, Any]] = []
        cac_nhom = [n for n in ("an_toan", "phap_ly", "chung") if self._kho[n]]
        if not cac_nhom:
            return []
        moi_nhom = max(1, k_req // len(cac_nhom))
        for nhom in cac_nhom:
            bucket = self._kho[nhom]
            take = min(len(bucket), moi_nhom)
            idx_pick = rng.choice(len(bucket), size=take, replace=False) if len(bucket) > take else np.arange(len(bucket))
            for i in idx_pick:
                ket_qua.append(dict(bucket[int(i)]))
        return ket_qua[:k_req]

    @property
    def so_luong(self) -> int:
        return sum(len(v) for v in self._kho.values())

    def thong_ke(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._kho.items()}


@dataclass
class KetQuaVongCanChinh:
    """Báo cáo kiểm toán đầy đủ cho một vòng cập nhật Continual Alignment."""

    so_vong: int
    hanh_dong: str
    da_thang_hang: bool
    da_rollback: bool
    ma_bam_truoc: str
    ma_bam_sau: str
    so_mau_tho: int
    so_mau_sau_loc: int
    so_mau_cach_ly: int
    so_mau_on_tap_tron: int
    thong_ke_loc: Dict[str, Any] = field(default_factory=dict)
    thong_ke_huan_luyen: Dict[str, Any] = field(default_factory=dict)
    ket_qua_kiem_dinh: Optional[Dict[str, Any]] = None
    diem_truoc_vong: Dict[str, float] = field(default_factory=dict)
    diem_ung_vien: Dict[str, float] = field(default_factory=dict)
    diem_hien_hanh_sau_vong: Dict[str, float] = field(default_factory=dict)
    ly_do: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sao_chep_mo_hinh(mo_hinh: Any) -> Any:
    if hasattr(mo_hinh, "sao_chep"):
        return mo_hinh.sao_chep()
    return copy.deepcopy(mo_hinh)


def _ma_bam_mo_hinh(mo_hinh: Any) -> str:
    if hasattr(mo_hinh, "ma_bam_trang_thai"):
        return str(mo_hinh.ma_bam_trang_thai())
    w = getattr(mo_hinh, "_dpo_weights", None)
    if isinstance(w, np.ndarray):
        return hashlib.sha256(w.astype(np.float64).tobytes()).hexdigest()[:16]
    return hashlib.sha256(repr(type(mo_hinh).__name__).encode("utf-8")).hexdigest()[:16]


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Vòng lặp Continual Alignment (DPO/SimPO) có lọc nhiễu cặp ưu tiên 6 tín hiệu, bộ đệm ôn tập neo phân tầng, "
        "cDPO có trọng số + KL anchor và cổng kiểm định 4 trục tự động rollback khi phát hiện hồi quy."
    ),
    nhom_extra="core",
)
class VongLapCanChinhAnToan:
    """Động cơ Căn chỉnh Liên tục An toàn Chống Hồi quy (`ViAlign-SafeLoop`)."""

    def __init__(
        self,
        mo_hinh_goc: Any,
        bo_loc_nhieu: Optional[BoLocNhieuCapUuTien] = None,
        cong_kiem_dinh: Optional[CongKiemDinhMoHinh] = None,
        bo_dem_on_tap: Optional[BoDemOnTapNeo] = None,
        che_do_huan_luyen: str = "dpo",
        beta: float = 0.25,
        toc_do_hoc: float = 0.18,
        so_epoch_moi_vong: int = 2,
        he_so_neo_kl: float = 0.08,
        ty_le_on_tap: float = 0.35,
        nguong_dat: float = 0.65,
        nguong_sut_giam_toi_da: float = 0.02,
        nguong_an_toan_toi_thieu: float = 0.80,
        bat_loc_nhieu: bool = True,
        bat_on_tap_neo: bool = True,
        bat_cong_kiem_dinh: bool = True,
        bat_trong_so_cdpo: bool = True,
        seed: int = 42,
    ) -> None:
        self.mo_hinh_hien_tai: Any = _sao_chep_mo_hinh(mo_hinh_goc)
        self.mo_hinh_khoi_dau: Any = _sao_chep_mo_hinh(mo_hinh_goc)
        self.bo_loc_nhieu = bo_loc_nhieu or BoLocNhieuCapUuTien()
        self.cong_kiem_dinh = cong_kiem_dinh or CongKiemDinhMoHinh()
        self.bo_dem_on_tap = bo_dem_on_tap or BoDemOnTapNeo()

        self.che_do_huan_luyen = str(che_do_huan_luyen).strip().lower()
        self.beta = float(beta)
        self.toc_do_hoc = float(toc_do_hoc)
        self.so_epoch_moi_vong = max(1, int(so_epoch_moi_vong))
        self.he_so_neo_kl = max(0.0, float(he_so_neo_kl))
        self.ty_le_on_tap = max(0.0, min(1.0, float(ty_le_on_tap)))
        self.nguong_dat = float(nguong_dat)
        self.nguong_sut_giam_toi_da = float(nguong_sut_giam_toi_da)
        self.nguong_an_toan_toi_thieu = float(nguong_an_toan_toi_thieu)

        self.bat_loc_nhieu = bool(bat_loc_nhieu)
        self.bat_on_tap_neo = bool(bat_on_tap_neo)
        self.bat_cong_kiem_dinh = bool(bat_cong_kiem_dinh)
        self.bat_trong_so_cdpo = bool(bat_trong_so_cdpo)
        self.seed = int(seed)

        self.lich_su_vong: List[KetQuaVongCanChinh] = []
        self._so_vong_da_chay: int = 0

    def nap_mau_neo_ban_dau(self, cac_mau_neo: Sequence[Dict[str, Any]]) -> int:
        """Nạp sẵn các cặp ưu tiên neo (an toàn + pháp lý nền tảng) vào bộ đệm ôn tập."""
        kq_loc = self.bo_loc_nhieu.loc_tap_du_lieu(cac_mau_neo)
        return self.bo_dem_on_tap.them_cac_mau(kq_loc.mau_sach)

    def chay_mot_vong(
        self,
        cap_uu_tien_tho: Sequence[Dict[str, Any]],
        bo_kiem_thu: Optional[List[Dict[str, Any]]] = None,
        gioi_han_mau_kiem_dinh: int = 16,
        tokenizer: Any = None,
        ham_huan_luyen_tuy_chinh: Optional[Callable[[Any, Any, List[Dict[str, Any]]], Dict[str, Any]]] = None,
    ) -> KetQuaVongCanChinh:
        """Thực thi trọn vẹn 1 vòng Continual Alignment: Lọc nhiễu → Trộn ôn tập → Huấn luyện ứng viên → Eval Gate → Promote/Rollback."""
        self._so_vong_da_chay += 1
        vong_id = self._so_vong_da_chay
        ma_truoc = _ma_bam_mo_hinh(self.mo_hinh_hien_tai)

        diem_truoc, _ = self.cong_kiem_dinh.danh_gia_mot_mo_hinh(
            self.mo_hinh_hien_tai,
            bo_kiem_thu=bo_kiem_thu,
            gioi_han_mau=gioi_han_mau_kiem_dinh,
        )

        ds_tho = list(cap_uu_tien_tho)
        kq_loc: Optional[KetQuaLocCapUuTien] = None
        if self.bat_loc_nhieu:
            kq_loc = self.bo_loc_nhieu.loc_tap_du_lieu(
                ds_tho,
                mo_hinh_tham_chieu=self.mo_hinh_hien_tai,
            )
            du_lieu_sach = list(kq_loc.mau_sach)
            so_cach_ly = kq_loc.so_cach_ly
            thong_ke_loc = {
                "tong_dau_vao": kq_loc.tong_dau_vao,
                "so_giu_lai": kq_loc.so_giu_lai,
                "so_cach_ly": kq_loc.so_cach_ly,
                "ty_le_giu_lai": kq_loc.ty_le_giu_lai,
                "thong_ke_ly_do_loai": kq_loc.thong_ke_ly_do_loai,
                "trong_so_trung_binh": kq_loc.trong_so_trung_binh,
                "chi_so_phat_hien_nhieu": kq_loc.chi_so_phat_hien_nhieu,
            }
        else:
            du_lieu_sach = [dict(x) for x in ds_tho]
            so_cach_ly = 0
            thong_ke_loc = {
                "tong_dau_vao": len(ds_tho),
                "so_giu_lai": len(ds_tho),
                "so_cach_ly": 0,
                "ty_le_giu_lai": 1.0 if ds_tho else 0.0,
                "thong_ke_ly_do_loai": {},
                "trong_so_trung_binh": 1.0,
                "chi_so_phat_hien_nhieu": None,
            }

        if not du_lieu_sach:
            kq_rong = KetQuaVongCanChinh(
                so_vong=vong_id,
                hanh_dong="SKIPPED_ALL_FILTERED",
                da_thang_hang=False,
                da_rollback=True,
                ma_bam_truoc=ma_truoc,
                ma_bam_sau=ma_truoc,
                so_mau_tho=len(ds_tho),
                so_mau_sau_loc=0,
                so_mau_cach_ly=so_cach_ly,
                so_mau_on_tap_tron=0,
                thong_ke_loc=thong_ke_loc,
                diem_truoc_vong=diem_truoc,
                diem_ung_vien=diem_truoc,
                diem_hien_hanh_sau_vong=diem_truoc,
                ly_do="Toàn bộ lô phản hồi đã bị bộ lọc nhiễu cách ly; giữ nguyên mô hình Production hiện hành.",
            )
            self.lich_su_vong.append(kq_rong)
            return kq_rong

        mau_on_tap: List[Dict[str, Any]] = []
        if self.bat_on_tap_neo and self.bo_dem_on_tap.so_luong > 0 and self.ty_le_on_tap > 0.0:
            so_can_tron = max(1, int(round(len(du_lieu_sach) * self.ty_le_on_tap)))
            mau_on_tap = self.bo_dem_on_tap.lay_mau_on_tap(so_can_tron, seed=self.seed + vong_id * 31)

        tap_huan_luyen: List[Dict[str, Any]] = []
        for item in du_lieu_sach + mau_on_tap:
            rec = dict(item)
            if not self.bat_trong_so_cdpo:
                rec["trong_so"] = 1.0
                rec["label_smoothing"] = 0.0
            tap_huan_luyen.append(rec)

        ung_vien = _sao_chep_mo_hinh(self.mo_hinh_hien_tai)
        ref_model = _sao_chep_mo_hinh(self.mo_hinh_hien_tai) if self.che_do_huan_luyen == "dpo" else None

        if ham_huan_luyen_tuy_chinh is not None:
            tk_train = dict(ham_huan_luyen_tuy_chinh(ung_vien, ref_model, tap_huan_luyen))
        else:
            trainer = DPOTrainer(
                so_vong=self.so_epoch_moi_vong,
                toc_do_hoc=self.toc_do_hoc,
                beta=self.beta,
                label_smoothing=0.05 if self.bat_trong_so_cdpo else 0.0,
                seed=self.seed + vong_id * 17,
                che_do=self.che_do_huan_luyen,
                he_so_neo_kl=self.he_so_neo_kl if self.bat_trong_so_cdpo else 0.0,
            )
            tk_train = trainer.huan_luyen(
                ung_vien,
                ref_model,
                tap_huan_luyen,
                tokenizer=tokenizer,
            )

        kq_gate: KetQuaKiemDinhMoHinh = self.cong_kiem_dinh.so_sanh_va_quyet_dinh(
            ung_vien=ung_vien,
            ban_goc=self.mo_hinh_hien_tai,
            bo_kiem_thu=bo_kiem_thu,
            gioi_han_mau=gioi_han_mau_kiem_dinh,
            nguong_dat=self.nguong_dat,
            nguong_sut_giam_toi_da=self.nguong_sut_giam_toi_da,
            nguong_an_toan_toi_thieu=self.nguong_an_toan_toi_thieu,
        )

        cho_phep_promote = kq_gate.dat_chuan if self.bat_cong_kiem_dinh else True
        if cho_phep_promote:
            self.mo_hinh_hien_tai = ung_vien
            if self.bat_on_tap_neo:
                self.bo_dem_on_tap.them_cac_mau(du_lieu_sach)
            hanh_dong = "PROMOTED"
            da_thang_hang = True
            da_rollback = False
            diem_sau = dict(kq_gate.diem_ung_vien)
            ly_do = kq_gate.ly_do if self.bat_cong_kiem_dinh else "Cập nhật trực tiếp không qua Eval Gate."
        else:
            hanh_dong = "ROLLED_BACK"
            da_thang_hang = False
            da_rollback = True
            diem_sau = dict(diem_truoc)
            ly_do = kq_gate.ly_do

        ma_sau = _ma_bam_mo_hinh(self.mo_hinh_hien_tai)
        kq_vong = KetQuaVongCanChinh(
            so_vong=vong_id,
            hanh_dong=hanh_dong,
            da_thang_hang=da_thang_hang,
            da_rollback=da_rollback,
            ma_bam_truoc=ma_truoc,
            ma_bam_sau=ma_sau,
            so_mau_tho=len(ds_tho),
            so_mau_sau_loc=len(du_lieu_sach),
            so_mau_cach_ly=so_cach_ly,
            so_mau_on_tap_tron=len(mau_on_tap),
            thong_ke_loc=thong_ke_loc,
            thong_ke_huan_luyen={
                "che_do": tk_train.get("che_do", self.che_do_huan_luyen),
                "train_loss_min": tk_train.get("train_loss_min", 0.0),
                "final_reward_margin": tk_train.get("final_reward_margin", 0.0),
                "global_step": tk_train.get("global_step", 0),
            },
            ket_qua_kiem_dinh=kq_gate.to_dict(),
            diem_truoc_vong=diem_truoc,
            diem_ung_vien=dict(kq_gate.diem_ung_vien),
            diem_hien_hanh_sau_vong=diem_sau,
            ly_do=ly_do,
        )
        self.lich_su_vong.append(kq_vong)
        return kq_vong

    def chay_da_vong(
        self,
        cac_lo_phan_hoi: Sequence[Sequence[Dict[str, Any]]],
        bo_kiem_thu: Optional[List[Dict[str, Any]]] = None,
        gioi_han_mau_kiem_dinh: int = 16,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        """Chạy chuỗi nhiều vòng Continual Alignment và tổng hợp báo cáo ổn định, rollback và suy thoái."""
        ket_qua_vong: List[KetQuaVongCanChinh] = []
        for lo in cac_lo_phan_hoi:
            kq = self.chay_mot_vong(
                cap_uu_tien_tho=lo,
                bo_kiem_thu=bo_kiem_thu,
                gioi_han_mau_kiem_dinh=gioi_han_mau_kiem_dinh,
                tokenizer=tokenizer,
            )
            ket_qua_vong.append(kq)

        so_promote = sum(1 for r in ket_qua_vong if r.da_thang_hang)
        so_rollback = sum(1 for r in ket_qua_vong if r.da_rollback)
        diem_tong_hop_cac_vong = [
            float(r.diem_hien_hanh_sau_vong.get("diem_tong_hop", 0.0)) for r in ket_qua_vong
        ]
        diem_an_toan_cac_vong = [
            float(r.diem_hien_hanh_sau_vong.get("ty_le_chan_tan_cong", 0.0)) for r in ket_qua_vong
        ]
        diem_hieu_luc_cac_vong = [
            float(r.diem_hien_hanh_sau_vong.get("ty_le_khong_dan_luat_het_han", 0.0)) for r in ket_qua_vong
        ]
        so_hoi_quy_ngam = sum(
            1
            for r in ket_qua_vong
            if float(r.diem_hien_hanh_sau_vong.get("diem_tong_hop", 0.0)) + self.nguong_sut_giam_toi_da
            < float(r.diem_truoc_vong.get("diem_tong_hop", 0.0))
        )

        tong_tp = sum(
            float((r.thong_ke_loc.get("chi_so_phat_hien_nhieu") or {}).get("tp", 0.0))
            for r in ket_qua_vong
        )
        tong_fp = sum(
            float((r.thong_ke_loc.get("chi_so_phat_hien_nhieu") or {}).get("fp", 0.0))
            for r in ket_qua_vong
        )
        tong_fn = sum(
            float((r.thong_ke_loc.get("chi_so_phat_hien_nhieu") or {}).get("fn", 0.0))
            for r in ket_qua_vong
        )
        loc_prec = tong_tp / float(max(1.0, tong_tp + tong_fp)) if (tong_tp + tong_fp) > 0 else 0.0
        loc_rec = tong_tp / float(max(1.0, tong_tp + tong_fn)) if (tong_tp + tong_fn) > 0 else 0.0
        loc_f1 = (2.0 * loc_prec * loc_rec / (loc_prec + loc_rec)) if (loc_prec + loc_rec) > 0 else 0.0

        diem_cuoi = ket_qua_vong[-1].diem_hien_hanh_sau_vong if ket_qua_vong else {}
        return {
            "so_vong": len(ket_qua_vong),
            "so_vong_promote": so_promote,
            "so_vong_rollback": so_rollback,
            "so_vong_hoi_quy_ngam": so_hoi_quy_ngam,
            "diem_cuoi_cung": diem_cuoi,
            "san_tong_hop_thap_nhat": round(min(diem_tong_hop_cac_vong), 4) if diem_tong_hop_cac_vong else 0.0,
            "san_an_toan_thap_nhat": round(min(diem_an_toan_cac_vong), 4) if diem_an_toan_cac_vong else 0.0,
            "san_hieu_luc_thap_nhat": round(min(diem_hieu_luc_cac_vong), 4) if diem_hieu_luc_cac_vong else 0.0,
            "loc_nhieu_f1": round(loc_f1, 4),
            "loc_nhieu_precision": round(loc_prec, 4),
            "loc_nhieu_recall": round(loc_rec, 4),
            "bo_dem_on_tap": self.bo_dem_on_tap.thong_ke(),
            "chi_tiet_cac_vong": [r.to_dict() for r in ket_qua_vong],
        }


VietnamesePreferencePolicy = ChinhSachUuTienTiengViet
StratifiedReplayBuffer = BoDemOnTapNeo
AlignmentRoundResult = KetQuaVongCanChinh
EvalGatedContinualAligner = VongLapCanChinhAnToan

__all__ = [
    "ChinhSachUuTienTiengViet",
    "VietnamesePreferencePolicy",
    "BoDemOnTapNeo",
    "StratifiedReplayBuffer",
    "KetQuaVongCanChinh",
    "AlignmentRoundResult",
    "VongLapCanChinhAnToan",
    "EvalGatedContinualAligner",
]
