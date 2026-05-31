"""TimKiemThamSo - Hyperparameter tuning."""

import itertools
from typing import Any, Dict, List

import numpy as np

from vietnamese_ai.core.cross_validation import KiemDinhCheo
from vietnamese_ai.utils.logger import Logger


class TimKiemThamSo:
    """
    Tìm kiếm tham số tối ưu cho mô hình.

    Hỗ trợ:
    - GridSearch: Tìm kiếm toàn bộ tổ hợp tham số
    - RandomSearch: Tìm kiếm ngẫu nhiên

    Sử dụng:
        >>> # GridSearch
        >>> ts = TimKiemThamSo()
        >>> ket_qua = ts.tim_kiem_luoi(
        ...     lop_mo_hinh=PhanLoai,
        ...     luoi_tham_so={'thuat_toan': ['logistic', 'knn'], 'C': [0.1, 1, 10]},
        ...     X=X_train, y=y_train
        ... )
        >>> print(ket_qua['tham_so_tot_nhat'])

        >>> # RandomSearch
        >>> ket_qua = ts.tim_kiem_ngau_nhien(
        ...     lop_mo_hinh=PhanLoai,
        ...     pham_vi_tham_so={'C': (0.01, 100), 'max_iter': (50, 500)},
        ...     X=X_train, y=y_train,
        ...     so_lan=20
        ... )
    """

    def __init__(self, so_fold: int = 5, seed: int = 42):
        self.so_fold = so_fold
        self.seed = seed
        self.logger = Logger("TimKiemThamSo")

    def tim_kiem_luoi(
        self,
        lop_mo_hinh: type,
        luoi_tham_so: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        chi_so: str = "do_chinh_xac",
        tot_nhat_cao: bool = True,
    ) -> Dict[str, Any]:
        """
        Tìm kiếm toàn bộ tổ hợp tham số (Grid Search).

        Args:
            lop_mo_hinh: Class mô hình (PhanLoai, HoiQuy, ...)
            luoi_tham_so: Dict ánh xạ tên_tham_so -> danh_sach_gia_tri
            X: Dữ liệu huấn luyện
            y: Nhãn
            chi_so: Chỉ số đánh giá
            tot_nhat_cao: True nếu điểm cao hơn = tốt hơn

        Returns:
            Dict chứa tham_so_tot_nhat, diem_tot_nhat, lich_su
        """
        X, y = np.asarray(X), np.asarray(y)

        # Tạo tất cả tổ hợp
        ten_tham_so = list(luoi_tham_so.keys())
        gia_tri_tham_so = list(luoi_tham_so.values())
        toan_bo_to_hop = list(itertools.product(*gia_tri_tham_so))

        self.logger.info(f"GridSearch: {len(toan_bo_to_hop)} tổ hợp tham số")

        lich_su = []
        diem_tot_nhat = -np.inf if tot_nhat_cao else np.inf
        tham_so_tot_nhat = None

        for i, to_hop in enumerate(toan_bo_to_hop):
            tham_so = dict(zip(ten_tham_so, to_hop))
            self.logger.info(f"  [{i + 1}/{len(toan_bo_to_hop)}] {tham_so}")

            try:
                mo_hinh = lop_mo_hinh(**tham_so)
                kdc = KiemDinhCheo(so_fold=self.so_fold, seed=self.seed)
                ket_qua_cv = kdc.chay(mo_hinh, X, y, chi_so=chi_so)
                diem = ket_qua_cv["diem_trung_binh"]

                ban_ghi = {
                    "tham_so": tham_so,
                    "diem": diem,
                    "do_lech_chuan": ket_qua_cv["do_lech_chuan"],
                }
                lich_su.append(ban_ghi)

                la_tot_nhat = (tot_nhat_cao and diem > diem_tot_nhat) or (
                    not tot_nhat_cao and diem < diem_tot_nhat
                )
                if la_tot_nhat:
                    diem_tot_nhat = diem
                    tham_so_tot_nhat = tham_so

            except Exception as e:
                self.logger.warning(f"  Lỗi với {tham_so}: {e}")
                lich_su.append({"tham_so": tham_so, "diem": None, "loi": str(e)})

        self.logger.info(
            f"Kết quả: tham số tốt nhất = {tham_so_tot_nhat}, điểm = {diem_tot_nhat:.4f}"
        )

        return {
            "tham_so_tot_nhat": tham_so_tot_nhat,
            "diem_tot_nhat": float(diem_tot_nhat),
            "chi_so": chi_so,
            "lich_su": lich_su,
            "so_to_hop": len(toan_bo_to_hop),
        }

    def tim_kiem_ngau_nhien(
        self,
        lop_mo_hinh: type,
        pham_vi_tham_so: Dict[str, tuple],
        X: np.ndarray,
        y: np.ndarray,
        so_lan: int = 20,
        chi_so: str = "do_chinh_xac",
        tot_nhat_cao: bool = True,
    ) -> Dict[str, Any]:
        """
        Tìm kiếm ngẫu nhiên tham số (Random Search).

        Args:
            lop_mo_hinh: Class mô hình
            pham_vi_tham_so: Dict ánh xạ tên_tham_so -> (min, max)
            X: Dữ liệu huấn luyện
            y: Nhãn
            so_lan: Số lần thử ngẫu nhiên
            chi_so: Chỉ số đánh giá
            tot_nhat_cao: True nếu điểm cao hơn = tốt hơn

        Returns:
            Dict chứa tham_so_tot_nhat, diem_tot_nhat, lich_su
        """
        X, y = np.asarray(X), np.asarray(y)
        np.random.seed(self.seed)

        self.logger.info(f"RandomSearch: {so_lan} lần thử ngẫu nhiên")

        lich_su = []
        diem_tot_nhat = -np.inf if tot_nhat_cao else np.inf
        tham_so_tot_nhat = None

        for i in range(so_lan):
            tham_so = {}
            for ten, (min_val, max_val) in pham_vi_tham_so.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    tham_so[ten] = np.random.randint(min_val, max_val + 1)
                else:
                    tham_so[ten] = np.random.uniform(min_val, max_val)

            self.logger.info(f"  [{i + 1}/{so_lan}] {tham_so}")

            try:
                mo_hinh = lop_mo_hinh(**tham_so)
                kdc = KiemDinhCheo(so_fold=self.so_fold, seed=self.seed)
                ket_qua_cv = kdc.chay(mo_hinh, X, y, chi_so=chi_so)
                diem = ket_qua_cv["diem_trung_binh"]

                ban_ghi = {
                    "tham_so": tham_so,
                    "diem": diem,
                    "do_lech_chuan": ket_qua_cv["do_lech_chuan"],
                }
                lich_su.append(ban_ghi)

                la_tot_nhat = (tot_nhat_cao and diem > diem_tot_nhat) or (
                    not tot_nhat_cao and diem < diem_tot_nhat
                )
                if la_tot_nhat:
                    diem_tot_nhat = diem
                    tham_so_tot_nhat = tham_so

            except Exception as e:
                self.logger.warning(f"  Lỗi với {tham_so}: {e}")
                lich_su.append({"tham_so": tham_so, "diem": None, "loi": str(e)})

        self.logger.info(
            f"Kết quả: tham số tốt nhất = {tham_so_tot_nhat}, điểm = {diem_tot_nhat:.4f}"
        )

        return {
            "tham_so_tot_nhat": tham_so_tot_nhat,
            "diem_tot_nhat": float(diem_tot_nhat),
            "chi_so": chi_so,
            "lich_su": lich_su,
            "so_lan": so_lan,
        }
