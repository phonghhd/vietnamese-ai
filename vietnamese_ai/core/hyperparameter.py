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

    def tim_kiem_bayesian(
        self,
        lop_mo_hinh: type,
        pham_vi_tham_so: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        so_lan: int = 15,
        chi_so: str = "do_chinh_xac",
        tot_nhat_cao: bool = True,
        gamma: float = 0.25,
        so_ung_vien_ei: int = 24,
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa siêu tham số bằng thuật toán Bayesian TPE (Tree-structured Parzen Estimator).

        Mô hình hóa hai phân phối mật độ hạt nhân (Parzen Window KDE):
        - $l(x)$: Phân phối của nhóm tham số tốt nhất (top `gamma` quantile).
        - $g(x)$: Phân phối của nhóm tham số còn lại.
        Ở mỗi bước, chọn ứng viên cực đại hóa tỷ số Expected Improvement $l(x) / g(x)$.
        """
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)

        self.logger.info(f"Bayesian TPE Search: {so_lan} vòng tối ưu")

        lich_su: List[Dict[str, Any]] = []
        diem_tot_nhat = -np.inf if tot_nhat_cao else np.inf
        tham_so_tot_nhat = None
        so_khoi_dong = max(3, so_lan // 3)

        def _lay_mau_ngau_nhien() -> Dict[str, Any]:
            mau = {}
            for ten, spec in pham_vi_tham_so.items():
                if isinstance(spec, list):
                    mau[ten] = spec[int(rng.integers(0, len(spec)))]
                elif isinstance(spec, tuple) and len(spec) == 2:
                    lo, hi = spec
                    if isinstance(lo, int) and isinstance(hi, int):
                        mau[ten] = int(rng.integers(lo, hi + 1))
                    else:
                        mau[ten] = float(rng.uniform(float(lo), float(hi)))
            return mau

        def _chon_ung_vien_tpe() -> Dict[str, Any]:
            hop_le = [r for r in lich_su if r.get("diem") is not None]
            if len(hop_le) < so_khoi_dong:
                return _lay_mau_ngau_nhien()

            da_sap_xep = sorted(
                hop_le,
                key=lambda r: float(r["diem"]),
                reverse=tot_nhat_cao,
            )
            n_good = max(1, int(np.ceil(gamma * len(da_sap_xep))))
            nhom_tot = da_sap_xep[:n_good]
            nhom_xau = da_sap_xep[n_good:] or da_sap_xep

            ung_vien_tot_nhat = None
            ty_so_ei_max = -np.inf

            for _ in range(so_ung_vien_ei):
                ung_vien: Dict[str, Any] = {}
                log_ei = 0.0
                for ten, spec in pham_vi_tham_so.items():
                    if isinstance(spec, list):
                        # Categorical Parzen smoothing (Laplace smoothed counts)
                        counts_good = {v: 1.0 for v in spec}
                        counts_bad = {v: 1.0 for v in spec}
                        for r in nhom_tot:
                            v = r["tham_so"].get(ten)
                            if v in counts_good:
                                counts_good[v] += 1.0
                        for r in nhom_xau:
                            v = r["tham_so"].get(ten)
                            if v in counts_bad:
                                counts_bad[v] += 1.0
                        probs_good = np.array([counts_good[v] for v in spec], dtype=float)
                        probs_good /= probs_good.sum()
                        probs_bad = np.array([counts_bad[v] for v in spec], dtype=float)
                        probs_bad /= probs_bad.sum()

                        idx = int(rng.choice(len(spec), p=probs_good))
                        ung_vien[ten] = spec[idx]
                        log_ei += float(np.log((probs_good[idx] + 1e-9) / (probs_bad[idx] + 1e-9)))
                    elif isinstance(spec, tuple) and len(spec) == 2:
                        lo, hi = spec
                        vals_good = np.array([float(r["tham_so"][ten]) for r in nhom_tot])
                        vals_bad = np.array([float(r["tham_so"][ten]) for r in nhom_xau])
                        span = max(1e-6, float(hi) - float(lo))
                        sigma_g = max(span * 0.05, float(np.std(vals_good)) if len(vals_good) > 1 else span * 0.2)
                        sigma_b = max(span * 0.05, float(np.std(vals_bad)) if len(vals_bad) > 1 else span * 0.3)

                        mu = float(vals_good[int(rng.integers(0, len(vals_good)))])
                        raw_val = float(np.clip(rng.normal(mu, sigma_g), float(lo), float(hi)))
                        if isinstance(lo, int) and isinstance(hi, int):
                            val_cast: Any = int(round(raw_val))
                        else:
                            val_cast = raw_val
                        ung_vien[ten] = val_cast

                        # Tính mật độ KDE l(x) và g(x)
                        l_x = float(np.mean(np.exp(-0.5 * (( float(val_cast) - vals_good) / sigma_g) ** 2)) / sigma_g)
                        g_x = float(np.mean(np.exp(-0.5 * ((float(val_cast) - vals_bad) / sigma_b) ** 2)) / sigma_b)
                        log_ei += float(np.log((l_x + 1e-12) / (g_x + 1e-12)))

                if log_ei > ty_so_ei_max:
                    ty_so_ei_max = log_ei
                    ung_vien_tot_nhat = ung_vien

            return ung_vien_tot_nhat or _lay_mau_ngau_nhien()

        for i in range(so_lan):
            tham_so = _chon_ung_vien_tpe()
            self.logger.info(f"  [TPE {i + 1}/{so_lan}] {tham_so}")

            try:
                mo_hinh = lop_mo_hinh(**tham_so)
                kdc = KiemDinhCheo(so_fold=self.so_fold, seed=self.seed)
                ket_qua_cv = kdc.chay(mo_hinh, X, y, chi_so=chi_so)
                diem = ket_qua_cv["diem_trung_binh"]

                lich_su.append(
                    {
                        "tham_so": tham_so,
                        "diem": diem,
                        "do_lech_chuan": ket_qua_cv["do_lech_chuan"],
                    }
                )
                la_tot_nhat = (tot_nhat_cao and diem > diem_tot_nhat) or (
                    not tot_nhat_cao and diem < diem_tot_nhat
                )
                if la_tot_nhat:
                    diem_tot_nhat = diem
                    tham_so_tot_nhat = tham_so
            except Exception as e:
                self.logger.warning(f"  Lỗi TPE với {tham_so}: {e}")
                lich_su.append({"tham_so": tham_so, "diem": None, "loi": str(e)})

        return {
            "tham_so_tot_nhat": tham_so_tot_nhat,
            "diem_tot_nhat": float(diem_tot_nhat),
            "chi_so": chi_so,
            "lich_su": lich_su,
            "so_lan": so_lan,
            "thuat_toan_toi_uu": "bayesian_tpe",
        }
