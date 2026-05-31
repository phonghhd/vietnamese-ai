"""TimKiemKienTruc - Neural Architecture Search (NAS)."""

import itertools
import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.core.cross_validation import KiemDinhCheo
from vietnamese_ai.models.classifier import PhanLoai
from vietnamese_ai.models.neural_net import MangNron
from vietnamese_ai.models.regression import HoiQuy
from vietnamese_ai.utils.logger import Logger
from vietnamese_ai.utils.validators import Validator


class TimKiemKienTruc:
    """
    Neural Architecture Search (NAS) - Tìm kiếm kiến trúc mạng tối ưu.

    Tính năng:
    - Tìm kiếm kiến trúc MLP (số lớp, số nơ-ron, hàm kích hoạt)
    - Random Search trên không gian kiến trúc
    - Grid Search trên không gian kiến trúc nhỏ
    - Early stopping: Dừng sớm nếu kiến trúc kém
    - Multi-objective: Accuracy vs model complexity
    - So sánh với traditional ML models

    Sử dụng:
        >>> tkt = TimKiemKienTruc(so_fold=3, so_lan_thu=10)
        >>> ket_qua = tkt.tim_kiem_ngau_nhien(X, y)
        >>> print(ket_qua['kien_truc_tot_nhat'])
    """

    CAC_HAM_KICH_HOAT = ["relu", "sigmoid", "tanh"]

    PHAM_VI_MAC_DINH = {
        "so_lop_an": [1, 2, 3],
        "so_neron_lop": [8, 16, 32, 64],
        "ham_kich_hoat": ["relu", "sigmoid", "tanh"],
        "so_vong": [50, 100, 200],
        "toc_do_hoc": [0.001, 0.01, 0.1],
    }

    def __init__(
        self,
        so_fold: int = 3,
        seed: int = 42,
        diem_toi_thieu: float = 0.5,
        toi_da_hoa_do_phuc_tap: bool = False,
    ):
        if so_fold < 2:
            raise ValueError("so_fold phải >= 2")

        self.so_fold = so_fold
        self.seed = seed
        self.diem_toi_thieu = diem_toi_thieu
        self.toi_da_hoa_do_phuc_tap = toi_da_hoa_do_phuc_tap
        self.logger = Logger("TimKiemKienTruc")

        self._ket_qua: List[Dict[str, Any]] = []
        self._kien_truc_tot_nhat: Optional[Dict] = None
        self._diem_tot_nhat: float = -np.inf
        self._da_tim_kiem = False

    def _tao_kien_truc_mlp(self, tham_so: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo thông số kiến trúc MLP từ tham số."""
        so_lop = tham_so.get("so_lop_an", 1)
        so_neron = tham_so.get("so_neron_lop", 16)
        ham_kich_hoat = tham_so.get("ham_kich_hoat", "relu")

        if isinstance(so_neron, int):
            lop_an = [so_neron] * so_lop
        elif isinstance(so_neron, list):
            lop_an = so_neron[:so_lop]
            while len(lop_an) < so_lop:
                lop_an.append(lop_an[-1])
        else:
            lop_an = [16] * so_lop

        return {
            "lop_an": lop_an,
            "ham_kich_hoat": ham_kich_hoat,
            "so_vong": tham_so.get("so_vong", 100),
            "toc_do_hoc": tham_so.get("toc_do_hoc", 0.01),
        }

    def _tinh_do_phuc_tap(self, kien_truc: Dict[str, Any]) -> int:
        """Tính độ phức tạp của kiến trúc (tổng số nơ-ron)."""
        lop_an = kien_truc.get("lop_an", [])
        return sum(lop_an)

    def _danh_gia_kien_truc(
        self,
        kien_truc: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        la_phan_loai: bool,
    ) -> Dict[str, Any]:
        """Đánh giá một kiến trúc bằng cross-validation."""
        try:
            mo_hinh = MangNron(
                lop_an=kien_truc["lop_an"],
                ham_kich_hoat=kien_truc["ham_kich_hoat"],
                so_vong=kien_truc["so_vong"],
                toc_do_hoc=kien_truc["toc_do_hoc"],
            )

            bat_dau = time.time()
            kdc = KiemDinhCheo(so_fold=self.so_fold, seed=self.seed)
            chi_so = "do_chinh_xac" if la_phan_loai else "mse"
            ket_qua_cv = kdc.chay(mo_hinh, X, y, chi_so=chi_so)
            thoi_gian = time.time() - bat_dau

            diem = ket_qua_cv["diem_trung_binh"]
            do_phuc_tap = self._tinh_do_phuc_tap(kien_truc)

            return {
                "kien_truc": kien_truc,
                "diem": float(diem),
                "do_lech_chuan": float(ket_qua_cv["do_lech_chuan"]),
                "do_phuc_tap": do_phuc_tap,
                "thoi_gian": round(thoi_gian, 2),
                "trang_thai": "thanh_cong",
            }

        except Exception as e:
            return {
                "kien_truc": kien_truc,
                "diem": None,
                "do_phuc_tap": self._tinh_do_phuc_tap(kien_truc),
                "trang_thai": "that_bai",
                "loi": str(e),
            }

    def _so_sanh_tot_nhat(self, diem: float, do_phuc_tap: int) -> bool:
        """So sánh với kết quả tốt nhất hiện tại."""
        if self.toi_da_hoa_do_phuc_tap:
            if diem >= self._diem_tot_nhat - 0.01:
                if diem > self._diem_tot_nhat:
                    return True
                if self._kien_truc_tot_nhat is not None:
                    phuc_tap_hien_tai = self._tinh_do_phuc_tap(self._kien_truc_tot_nhat)
                    return do_phuc_tap < phuc_tap_hien_tai
                return True
            return False
        return diem > self._diem_tot_nhat

    def tim_kiem_ngau_nhien(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pham_vi: Optional[Dict[str, Any]] = None,
        so_lan: int = 10,
    ) -> Dict[str, Any]:
        """
        Random Search trên không gian kiến trúc.

        Args:
            X: Dữ liệu đầu vào
            y: Nhãn
            pham_vi: Phạm vi tham số (mặc định: PHAM_VI_MAC_DINH)
            so_lan: Số lần thử

        Returns:
            Dict chứa kết quả: kien_truc_tot_nhat, tat_ca_ket_qua
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if pham_vi is None:
            pham_vi = self.PHAM_VI_MAC_DINH.copy()

        la_phan_loai = Validator.kiem_tra_nhiem_vu(y) == "phan_loai"
        self.logger.info("=" * 60)
        self.logger.info(f"NAS RANDOM SEARCH ({so_lan} lần thử)")
        self.logger.info("=" * 60)

        rng = np.random.RandomState(self.seed)
        self._ket_qua = []
        self._kien_truc_tot_nhat = None
        self._diem_tot_nhat = -np.inf
        bat_dau = time.time()

        for lan in range(so_lan):
            tham_so = {}
            for key, gia_tri in pham_vi.items():
                if isinstance(gia_tri, list):
                    idx = rng.randint(0, len(gia_tri))
                    tham_so[key] = gia_tri[idx]
                elif isinstance(gia_tri, tuple) and len(gia_tri) == 2:
                    tham_so[key] = rng.uniform(gia_tri[0], gia_tri[1])
                else:
                    tham_so[key] = gia_tri

            kien_truc = self._tao_kien_truc_mlp(tham_so)
            self.logger.info(
                f"\nLần {lan + 1}/{so_lan}: "
                f"layers={kien_truc['lop_an']}, "
                f"activation={kien_truc['ham_kich_hoat']}"
            )

            ket_qua = self._danh_gia_kien_truc(kien_truc, X, y, la_phan_loai)
            self._ket_qua.append(ket_qua)

            if ket_qua["trang_thai"] == "thanh_cong":
                diem = ket_qua["diem"]
                if diem < self.diem_toi_thieu:
                    self.logger.info(f"  Điểm {diem:.4f} < ngưỡng {self.diem_toi_thieu}, bỏ qua")
                    continue

                if self._so_sanh_tot_nhat(diem, ket_qua["do_phuc_tap"]):
                    self._diem_tot_nhat = diem
                    self._kien_truc_tot_nhat = kien_truc

                self.logger.info(
                    f"  Điểm: {diem:.4f} (+/- {ket_qua['do_lech_chuan']:.4f}), "
                    f"phức tạp: {ket_qua['do_phuc_tap']}"
                )
            else:
                self.logger.warning(f"  Lỗi: {ket_qua.get('loi', 'unknown')}")

        tong_thoi_gian = time.time() - bat_dau
        self._da_tim_kiem = True

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(
            f"KẾT QUẢ NAS: kien_truc={self._kien_truc_tot_nhat}, diem={self._diem_tot_nhat:.4f}"
        )
        self.logger.info(f"{'=' * 60}")

        return self._tao_bao_cao(tong_thoi_gian)

    def tim_kiem_luoi(
        self,
        X: np.ndarray,
        y: np.ndarray,
        luoi_tham_so: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        """
        Grid Search trên không gian kiến trúc nhỏ.

        Args:
            X: Dữ liệu đầu vào
            y: Nhãn
            luoi_tham_so: Lưới tham số

        Returns:
            Dict chứa kết quả
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if luoi_tham_so is None:
            luoi_tham_so = {
                "so_lop_an": [1, 2],
                "so_neron_lop": [16, 32],
                "ham_kich_hoat": ["relu", "sigmoid"],
            }

        la_phan_loai = Validator.kiem_tra_nhiem_vu(y) == "phan_loai"
        self.logger.info("=" * 60)
        self.logger.info("NAS GRID SEARCH")
        self.logger.info("=" * 60)

        keys = list(luoi_tham_so.keys())
        values = list(luoi_tham_so.values())
        cac_to_hop = list(itertools.product(*values))

        self.logger.info(f"Tổng số kiến trúc: {len(cac_to_hop)}")

        self._ket_qua = []
        self._kien_truc_tot_nhat = None
        self._diem_tot_nhat = -np.inf
        bat_dau = time.time()

        for i, to_hop in enumerate(cac_to_hop):
            tham_so = dict(zip(keys, to_hop))
            kien_truc = self._tao_kien_truc_mlp(tham_so)

            self.logger.info(
                f"\n[{i + 1}/{len(cac_to_hop)}] "
                f"layers={kien_truc['lop_an']}, "
                f"activation={kien_truc['ham_kich_hoat']}"
            )

            ket_qua = self._danh_gia_kien_truc(kien_truc, X, y, la_phan_loai)
            self._ket_qua.append(ket_qua)

            if ket_qua["trang_thai"] == "thanh_cong":
                diem = ket_qua["diem"]
                if self._so_sanh_tot_nhat(diem, ket_qua["do_phuc_tap"]):
                    self._diem_tot_nhat = diem
                    self._kien_truc_tot_nhat = kien_truc

                self.logger.info(f"  Điểm: {diem:.4f}")

        tong_thoi_gian = time.time() - bat_dau
        self._da_tim_kiem = True

        return self._tao_bao_cao(tong_thoi_gian)

    def so_sanh_voi_ml_truyen_thong(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        So sánh NAS result với các thuật toán ML truyền thống.

        Args:
            X: Dữ liệu
            y: Nhãn

        Returns:
            Dict chứa kết quả so sánh
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        la_phan_loai = Validator.kiem_tra_nhiem_vu(y) == "phan_loai"

        if la_phan_loai:
            cac_thuat_toan = ["logistic", "knn", "rung_ngau_nhien", "gradient_boosting"]
            lop_mo_hinh = PhanLoai
            chi_so = "do_chinh_xac"
        else:
            cac_thuat_toan = ["tuyen_tinh", "ridge", "rung_ngau_nhien"]
            lop_mo_hinh = HoiQuy
            chi_so = "mse"

        ket_qua_truyen_thong = []
        for tt in cac_thuat_toan:
            try:
                mo_hinh = lop_mo_hinh(thuat_toan=tt)
                kdc = KiemDinhCheo(so_fold=self.so_fold, seed=self.seed)
                kq = kdc.chay(mo_hinh, X, y, chi_so=chi_so)
                ket_qua_truyen_thong.append(
                    {
                        "thuat_toan": tt,
                        "diem": float(kq["diem_trung_binh"]),
                        "do_lech_chuan": float(kq["do_lech_chuan"]),
                    }
                )
            except Exception as e:
                ket_qua_truyen_thong.append(
                    {
                        "thuat_toan": tt,
                        "diem": None,
                        "loi": str(e),
                    }
                )

        return {
            "nas": {
                "kien_truc": self._kien_truc_tot_nhat,
                "diem": float(self._diem_tot_nhat) if self._da_tim_kiem else None,
            },
            "truyen_thong": ket_qua_truyen_thong,
        }

    def _tao_bao_cao(self, tong_thoi_gian: float) -> Dict[str, Any]:
        """Tạo báo cáo kết quả NAS."""
        thanh_cong = [kq for kq in self._ket_qua if kq["trang_thai"] == "thanh_cong"]
        that_bai = [kq for kq in self._ket_qua if kq["trang_thai"] == "that_bai"]

        return {
            "kien_truc_tot_nhat": self._kien_truc_tot_nhat,
            "diem_tot_nhat": float(self._diem_tot_nhat),
            "so_lan_thu": len(self._ket_qua),
            "thanh_cong": len(thanh_cong),
            "that_bai": len(that_bai),
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "tat_ca_ket_qua": self._ket_qua,
        }

    def bao_cao(self) -> str:
        """Tạo báo cáo dạng text."""
        if not self._ket_qua:
            return "Chưa có kết quả. Gọi tim_kiem_ngau_nhien() hoặc tim_kiem_luoi() trước."

        lines = ["=== BÁO CÁO NEURAL ARCHITECTURE SEARCH ===\n"]

        if self._kien_truc_tot_nhat:
            kt = self._kien_truc_tot_nhat
            lines.append("Kiến trúc tốt nhất:")
            lines.append(f"  Lớp ẩn: {kt['lop_an']}")
            lines.append(f"  Hàm kích hoạt: {kt['ham_kich_hoat']}")
            lines.append(f"  Số vòng: {kt['so_vong']}")
            lines.append(f"  Tốc độ học: {kt['toc_do_hoc']}")
            lines.append(f"  Điểm: {self._diem_tot_nhat:.4f}")
            lines.append(f"  Độ phức tạp: {self._tinh_do_phuc_tap(kt)}")
            lines.append("")

        lines.append(
            f"{'STT':<5} {'Lớp ẩn':<20} {'Kích hoạt':<12} {'Điểm':<12} {'Phức tạp':<10} {'Trạng thái'}"
        )
        lines.append("-" * 75)

        for i, kq in enumerate(
            sorted(
                [k for k in self._ket_qua if k["trang_thai"] == "thanh_cong"],
                key=lambda x: x["diem"],
                reverse=True,
            )
        ):
            kt = kq["kien_truc"]
            lines.append(
                f"{i + 1:<5} "
                f"{str(kt['lop_an']):<20} "
                f"{kt['ham_kich_hoat']:<12} "
                f"{kq['diem']:<12.4f} "
                f"{kq['do_phuc_tap']:<10} "
                f"{kq['trang_thai']}"
            )

        return "\n".join(lines)
