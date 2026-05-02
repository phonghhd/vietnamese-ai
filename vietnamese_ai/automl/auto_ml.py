"""AutoML - Tự động chọn mô hình và tham số tốt nhất."""

from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.core.cross_validation import KiemDinhCheo
from vietnamese_ai.models.classifier import PhanLoai
from vietnamese_ai.models.regression import HoiQuy
from vietnamese_ai.preprocessing.numerical import XuLySo
from vietnamese_ai.utils.logger import Logger
from vietnamese_ai.utils.validators import Validator


class AutoML:
    """
    AutoML - Tự động hóa toàn bộ quy trình học máy.

    Tự động:
    1. Phát hiện nhiệm vụ (phân loại / hồi quy)
    2. Tiền xử lý dữ liệu
    3. Thử nhiều thuật toán
    4. Chọn mô hình tốt nhất

    Sử dụng:
        >>> auto = AutoML()
        >>> ket_qua = auto.fit(X_train, y_train)
        >>> print(ket_qua['mo_hinh_tot_nhat'])
        >>> du_doan = auto.predict(X_test)
    """

    CAC_THUAT_TOAN_PHAN_LOAI = [
        "logistic", "knn", "rung_ngau_nhien", "gradient_boosting", "naive_bayes"
    ]

    CAC_THUAT_TOAN_HOI_QUY = [
        "tuyen_tinh", "ridge", "lasso", "rung_ngau_nhien", "gradient_boosting"
    ]

    def __init__(
        self,
        so_fold: int = 5,
        chuan_hoa: bool = True,
        chi_so_phan_loai: str = "do_chinh_xac",
        chi_so_hoi_quy: str = "mse",
    ):
        self.so_fold = so_fold
        self.chuan_hoa = chuan_hoa
        self.chi_so_phan_loai = chi_so_phan_loai
        self.chi_so_hoi_quy = chi_so_hoi_quy
        self.logger = Logger("AutoML")

        self._nhiem_vu: Optional[str] = None
        self._mo_hinh_tot_nhat: Any = None
        self._xu_ly_so: Optional[XuLySo] = None
        self._ket_qua: List[Dict] = []
        self._da_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Chạy AutoML trên dữ liệu.

        Args:
            X: Dữ liệu đầu vào
            y: Nhãn

        Returns:
            Dict chứa: mo_hinh_tot_nhat, tat_ca_ket_qua, nhiem_vu
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.logger.info("=" * 60)
        self.logger.info("BẮT ĐẦU AUTOML")
        self.logger.info("=" * 60)

        # 1. Phát hiện nhiệm vụ
        self._nhiem_vu = Validator.kiem_tra_nhiem_vu(y)
        self.logger.info(f"Nhiệm vụ: {self._nhiem_vu}")

        # 2. Tiền xử lý
        if self.chuan_hoa:
            self._xu_ly_so = XuLySo()
            X = self._xu_ly_so.chuan_hoa_zscore(X)
            self.logger.info("Đã chuẩn hóa dữ liệu (Z-Score)")

        # 3. Thử tất cả thuật算法
        if self._nhiem_vu == "phan_loai":
            cac_thuat_toan = self.CAC_THUAT_TOAN_PHAN_LOAI
            lop_mo_hinh = PhanLoai
            chi_so = self.chi_so_phan_loai
            tot_nhat_cao = True
        else:
            cac_thuat_toan = self.CAC_THUAT_TOAN_HOI_QUY
            lop_mo_hinh = HoiQuy
            chi_so = self.chi_so_hoi_quy
            tot_nhat_cao = False

        self._ket_qua = []
        diem_tot_nhat = -np.inf if tot_nhat_cao else np.inf

        for tt in cac_thuat_toan:
            self.logger.info(f"\nThử thuật toán: {tt}")
            try:
                mo_hinh = lop_mo_hinh(thuat_toan=tt)
                kdc = KiemDinhCheo(so_fold=self.so_fold)
                ket_qua_cv = kdc.chay(mo_hinh, X, y, chi_so=chi_so)
                diem = ket_qua_cv["diem_trung_binh"]

                ban_ghi = {
                    "thuat_toan": tt,
                    "diem": diem,
                    "do_lech_chuan": ket_qua_cv["do_lech_chuan"],
                    "trang_thai": "thanh_cong",
                }
                self._ket_qua.append(ban_ghi)

                la_tot_nhat = (tot_nhat_cao and diem > diem_tot_nhat) or (
                    not tot_nhat_cao and diem < diem_tot_nhat
                )
                if la_tot_nhat:
                    diem_tot_nhat = diem
                    self._mo_hinh_tot_nhat = lop_mo_hinh(thuat_toan=tt)

                self.logger.info(f"  {tt}: {chi_so}={diem:.4f} (+/- {ket_qua_cv['do_lech_chuan']:.4f})")

            except Exception as e:
                self.logger.warning(f"  {tt}: LỖI - {e}")
                self._ket_qua.append({
                    "thuat_toan": tt,
                    "diem": None,
                    "trang_thai": "that_bai",
                    "loi": str(e),
                })

        # 4. Huấn luyện mô hình tốt nhất trên toàn bộ dữ liệu
        if self._mo_hinh_tot_nhat is not None:
            self._mo_hinh_tot_nhat.huan_luyen(X, y)
            self._da_fit = True
            ten_tot_nhat = [
                r["thuat_toan"] for r in self._ket_qua
                if r["diem"] == diem_tot_nhat
            ]
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"MÔ HÌNH TỐT NHẤT: {ten_tot_nhat[0]} ({chi_so}={diem_tot_nhat:.4f})")
            self.logger.info(f"{'='*60}")

        return {
            "mo_hinh_tot_nhat": self._mo_hinh_tot_nhat,
            "thuat_toan_tot_nhat": ten_tot_nhat[0] if ten_tot_nhat else None,
            "diem_tot_nhat": float(diem_tot_nhat),
            "nhiem_vu": self._nhiem_vu,
            "tat_ca_ket_qua": self._ket_qua,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán với mô hình tốt nhất."""
        if not self._da_fit:
            raise RuntimeError("Chưa fit. Gọi fit() trước.")
        X = np.asarray(X, dtype=float)
        if self._xu_ly_so is not None:
            X = self._xu_ly_so.transform(X)
        return self._mo_hinh_tot_nhat.du_doan(X)

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        """Đánh giá mô hình tốt nhất."""
        if not self._da_fit:
            raise RuntimeError("Chưa fit.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self._xu_ly_so is not None:
            X = self._xu_ly_so.transform(X)
        return self._mo_hinh_tot_nhat.danh_gia(X, y)

    def bao_cao(self) -> str:
        """Tạo báo cáo so sánh các thuật toán."""
        if not self._ket_qua:
            return "Chưa có kết quả. Gọi fit() trước."

        lines = ["=== BÁO CÁO AUTOML ===\n"]
        lines.append(f"Nhiệm vụ: {self._nhiem_vu}\n")
        lines.append(f"{'Thuật toán':<25} {'Điểm':<12} {'Độ lệch chuẩn':<15} {'Trạng thái'}")
        lines.append("-" * 70)

        for r in sorted(self._ket_qua, key=lambda x: x.get("diem") or 0, reverse=True):
            diem_str = f"{r['diem']:.4f}" if r['diem'] else "N/A"
            lech_str = f"{r.get('do_lech_chuan', 0):.4f}" if r['diem'] else "N/A"
            lines.append(f"{r['thuat_toan']:<25} {diem_str:<12} {lech_str:<15} {r['trang_thai']}")

        return "\n".join(lines)
