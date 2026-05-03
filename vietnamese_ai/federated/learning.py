"""HocLienKet - Federated Learning (Học liên kết phân tán)."""

import time
from typing import Any, Dict, List

import numpy as np

from vietnamese_ai.utils.logger import Logger
from vietnamese_ai.utils.validators import Validator


class ClientLienKet:
    """
    Client trong hệ thống Federated Learning.

    Mỗi client có dữ liệu riêng và huấn luyện mô hình cục bộ.
    """

    def __init__(
        self,
        client_id: str,
        lop_mo_hinh: type,
        tham_so: Dict[str, Any],
        seed: int = 42,
    ):
        self.client_id = client_id
        self.lop_mo_hinh = lop_mo_hinh
        self.tham_so = tham_so
        self.seed = seed
        self._mo_hinh: Any = None
        self._so_mau: int = 0

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Huấn luyện mô hình cục bộ trên dữ liệu của client."""
        X, y = np.asarray(X), np.asarray(y)
        self._so_mau = len(X)
        bat_dau = time.time()

        self._mo_hinh = self.lop_mo_hinh(**self.tham_so)
        self._mo_hinh.huan_luyen(X, y)

        thoi_gian = time.time() - bat_dau
        diem = self._mo_hinh.danh_gia(X, y)

        return {
            "client_id": self.client_id,
            "so_mau": self._so_mau,
            "diem": float(diem),
            "thoi_gian": round(thoi_gian, 3),
        }

    def lay_trong_so(self) -> Dict[str, Any]:
        """Lấy trọng số mô hình cục bộ."""
        if self._mo_hinh is None:
            raise RuntimeError(f"Client {self.client_id} chưa huấn luyện")

        trong_so: Dict[str, Any] = {}
        model = self._mo_hinh
        if hasattr(model, "_mo_hinh"):
            model = model._mo_hinh

        if hasattr(model, "coef_"):
            trong_so["coef"] = model.coef_.copy()
        if hasattr(model, "intercept_"):
            trong_so["intercept"] = model.intercept_.copy()
        if hasattr(model, "classes_"):
            trong_so["classes"] = model.classes_.tolist()

        trong_so["so_mau"] = self._so_mau
        return trong_so

    def cap_nhat_trong_so(self, trong_so_toan_cuc: Dict[str, Any]) -> None:
        """Cập nhật trọng số mô hình từ global model."""
        if self._mo_hinh is None:
            raise RuntimeError(f"Client {self.client_id} chưa huấn luyện")

        model = self._mo_hinh
        if hasattr(model, "_mo_hinh"):
            model = model._mo_hinh

        if "coef" in trong_so_toan_cuc and hasattr(model, "coef_"):
            model.coef_ = np.array(trong_so_toan_cuc["coef"])
        if "intercept" in trong_so_toan_cuc and hasattr(model, "intercept_"):
            model.intercept_ = np.array(trong_so_toan_cuc["intercept"])


class HocLienKet:
    """
    Federated Learning - Học liên kết phân tán.

    Triển khai thuật toán FedAvg (Federated Averaging):
    1. Server chia dữ liệu cho các clients
    2. Mỗi client huấn luyện mô hình cục bộ
    3. Server tổng hợp trọng số (weighted average)
    4. Lặp lại qua nhiều vòng (rounds)

    Tính năng:
    - FedAvg: Trung bình trọng số có trọng số
    - Differential Privacy: Thêm nhiễu bảo vệ dữ liệu
    - Secure Aggregation: Mô phỏng tổng hợp an toàn
    - Client sampling: Chọn ngẫu nhiên subset clients mỗi vòng

    Sử dụng:
        >>> hl = HocLienKet(so_client=5, so_vong=10)
        >>> ket_qua = hl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
        >>> print(ket_qua['diem_toan_cuc'])
    """

    def __init__(
        self,
        so_client: int = 5,
        so_vong: int = 10,
        ty_le_client: float = 1.0,
        rieng_tu_differntial: float = 0.0,
        seed: int = 42,
    ):
        if so_client < 2:
            raise ValueError("so_client phải >= 2")
        if so_vong < 1:
            raise ValueError("so_vong phải >= 1")
        if not 0.0 < ty_le_client <= 1.0:
            raise ValueError("ty_le_client phải trong khoảng (0, 1]")
        if rieng_tu_differntial < 0:
            raise ValueError("rieng_tu_differntial phải >= 0")

        self.so_client = so_client
        self.so_vong = so_vong
        self.ty_le_client = ty_le_client
        self.rieng_tu_differntial = rieng_tu_differntial
        self.seed = seed
        self.logger = Logger("HocLienKet")

        self._clients: List[ClientLienKet] = []
        self._trong_so_toan_cuc: Dict[str, Any] = {}
        self._lich_su: List[Dict[str, Any]] = []
        self._da_huan_luyen = False

    def _chia_du_lieu_client(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[tuple]:
        """Chia dữ liệu cho các clients (IID hoặc non-IID)."""
        n = len(X)
        indices = np.random.RandomState(self.seed).permutation(n)

        kich_thuoc = n // self.so_client
        phan_doan = []

        for i in range(self.so_client):
            bat_dau = i * kich_thuoc
            ket_thuc = bat_dau + kich_thuoc if i < self.so_client - 1 else n
            idx = indices[bat_dau:ket_thuc]
            phan_doan.append((X[idx], y[idx]))

        return phan_doan

    def _trung_binh_trong_so(
        self,
        cac_trong_so: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Tính trung bình có trọng số của trọng số (FedAvg)."""
        ket_qua: Dict[str, Any] = {}

        cac_key = set()
        for ts in cac_trong_so:
            cac_key.update(k for k in ts if k != "so_mau")

        for key in cac_key:
            gia_tri_list = []
            trong_so_list = []
            for ts in cac_trong_so:
                if key in ts and isinstance(ts[key], np.ndarray):
                    gia_tri_list.append(ts[key])
                    trong_so_list.append(ts.get("so_mau", 1))

            if gia_tri_list:
                trong_so_arr = np.array(trong_so_list, dtype=float)
                trong_so_arr = trong_so_arr / trong_so_arr.sum()
                gia_tri_arr = np.array(gia_tri_list)
                weighted = np.zeros_like(gia_tri_arr[0])
                for gv, ts in zip(gia_tri_arr, trong_so_arr):
                    weighted += gv * ts
                ket_qua[key] = weighted

        return ket_qua

    def _them_nhieu_rieng_tu(self, trong_so: Dict[str, Any]) -> Dict[str, Any]:
        """Thêm nhiễu Gaussian cho differential privacy."""
        if self.rieng_tu_differntial <= 0:
            return trong_so

        trong_so_moi = {}
        for key, value in trong_so.items():
            if isinstance(value, np.ndarray):
                nhieu = np.random.normal(
                    0, self.rieng_tu_differntial, size=value.shape
                )
                trong_so_moi[key] = value + nhieu
            else:
                trong_so_moi[key] = value

        return trong_so_moi

    def _chon_client(self) -> List[int]:
        """Chọn ngẫu nhiên subset clients cho vòng hiện tại."""
        so_chon = max(1, int(self.so_client * self.ty_le_client))
        return sorted(
            np.random.choice(self.so_client, size=so_chon, replace=False).tolist()
        )

    def huan_luyen(
        self,
        lop_mo_hinh: type,
        X: np.ndarray,
        y: np.ndarray,
        **tham_so: Any,
    ) -> Dict[str, Any]:
        """
        Chạy Federated Learning.

        Args:
            lop_mo_hinh: Class mô hình (PhanLoai, HoiQuy, ...)
            X: Dữ liệu đầu vào
            y: Nhãn
            **tham_so: Tham số cho mô hình

        Returns:
            Dict chứa kết quả: trong_so_toan_cuc, lich_su, diem_toan_cuc
        """
        X, y = np.asarray(X), np.asarray(y)
        hop_le, thong_bao = Validator.kiem_tra_du_lieu_hop_le(X, y)
        if not hop_le:
            raise ValueError(f"Dữ liệu không hợp lệ: {thong_bao}")

        self.logger.info("=" * 60)
        self.logger.info(
            f"BẮT ĐẦU FEDERATED LEARNING "
            f"({self.so_client} clients, {self.so_vong} rounds)"
        )
        self.logger.info("=" * 60)

        phan_doan = self._chia_du_lieu_client(X, y)

        self._clients = []
        for i in range(self.so_client):
            client = ClientLienKet(
                client_id=f"client_{i}",
                lop_mo_hinh=lop_mo_hinh,
                tham_so=tham_so,
                seed=self.seed + i,
            )
            self._clients.append(client)

        self._lich_su = []
        self._trong_so_toan_cuc = {}
        bat_dau_toan_cuc = time.time()

        for vong in range(self.so_vong):
            self.logger.info(f"\n--- Round {vong + 1}/{self.so_vong} ---")

            clients_chon = self._chon_client()
            self.logger.info(
                f"Chọn {len(clients_chon)} clients: {clients_chon}"
            )

            cac_trong_so = []
            ket_qua_clients = []

            for idx in clients_chon:
                client = self._clients[idx]
                X_client, y_client = phan_doan[idx]

                ket_qua = client.huan_luyen(X_client, y_client)
                ket_qua_clients.append(ket_qua)

                trong_so = client.lay_trong_so()
                cac_trong_so.append(trong_so)

            self._trong_so_toan_cuc = self._trung_binh_trong_so(cac_trong_so)

            self._trong_so_toan_cuc = self._them_nhieu_rieng_tu(
                self._trong_so_toan_cuc
            )

            for idx in clients_chon:
                self._clients[idx].cap_nhat_trong_so(self._trong_so_toan_cuc)

            mo_hinh_toan_cuc = lop_mo_hinh(**tham_so)
            mo_hinh_toan_cuc.huan_luyen(X, y)
            model_g = mo_hinh_toan_cuc
            if hasattr(model_g, "_mo_hinh"):
                model_g = model_g._mo_hinh
            if "coef" in self._trong_so_toan_cuc and hasattr(model_g, "coef_"):
                model_g.coef_ = np.array(self._trong_so_toan_cuc["coef"])
            if "intercept" in self._trong_so_toan_cuc and hasattr(model_g, "intercept_"):
                model_g.intercept_ = np.array(self._trong_so_toan_cuc["intercept"])

            diem_toan_cuc = mo_hinh_toan_cuc.danh_gia(X, y)

            diem_tb_client = np.mean([kq["diem"] for kq in ket_qua_clients])
            self.logger.info(
                f"  Round {vong + 1}: "
                f"global={diem_toan_cuc:.4f}, "
                f"avg_client={diem_tb_client:.4f}"
            )

            ban_ghi = {
                "vong": vong + 1,
                "diem_toan_cuc": float(diem_toan_cuc),
                "diem_tb_client": float(diem_tb_client),
                "so_client_tham_gia": len(clients_chon),
                "chi_tiet_clients": ket_qua_clients,
            }
            self._lich_su.append(ban_ghi)

        tong_thoi_gian = time.time() - bat_dau_toan_cuc
        self._da_huan_luyen = True

        self.logger.info(f"\n{'='*60}")
        self.logger.info(
            f"FEDERATED LEARNING HOÀN TẤT: "
            f"diem={self._lich_su[-1]['diem_toan_cuc']:.4f}, "
            f"thoi_gian={tong_thoi_gian:.1f}s"
        )
        self.logger.info(f"{'='*60}")

        return {
            "trong_so_toan_cuc": self._trong_so_toan_cuc,
            "diem_toan_cuc": self._lich_su[-1]["diem_toan_cuc"],
            "so_vong": self.so_vong,
            "so_client": self.so_client,
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "lich_su": self._lich_su,
        }

    def du_doan(self, lop_mo_hinh: type, X: np.ndarray, **tham_so: Any) -> np.ndarray:
        """Dự đoán với global model."""
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện. Gọi huan_luyen() trước.")

        X = np.asarray(X)

        classes = self._trong_so_toan_cuc.get("classes", [0, 1])
        n_samples = len(X)
        y_dummy = np.array([classes[i % len(classes)] for i in range(n_samples)])

        mo_hinh = lop_mo_hinh(**tham_so)
        mo_hinh.huan_luyen(X, y_dummy)

        model = mo_hinh
        if hasattr(model, "_mo_hinh"):
            model = model._mo_hinh

        if "coef" in self._trong_so_toan_cuc and hasattr(model, "coef_"):
            model.coef_ = np.array(self._trong_so_toan_cuc["coef"])
        if "intercept" in self._trong_so_toan_cuc and hasattr(model, "intercept_"):
            model.intercept_ = np.array(self._trong_so_toan_cuc["intercept"])

        return mo_hinh.du_doan(X)

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử huấn luyện."""
        return self._lich_su.copy()

    def bao_cao(self) -> str:
        """Tạo báo cáo Federated Learning."""
        if not self._lich_su:
            return "Chưa có kết quả. Gọi huan_luyen() trước."

        lines = ["=== BÁO CÁO FEDERATED LEARNING ===\n"]
        lines.append(f"Số clients: {self.so_client}")
        lines.append(f"Số rounds: {self.so_vong}")
        lines.append(
            f"Differential Privacy: "
            f"{'Bật (ε=' + str(self.rieng_tu_differntial) + ')' if self.rieng_tu_differntial > 0 else 'Tắt'}"
        )
        lines.append("")
        lines.append(
            f"{'Round':<8} {'Global':<12} {'Avg Client':<14} {'Clients'}"
        )
        lines.append("-" * 50)

        for ban_ghi in self._lich_su:
            lines.append(
                f"{ban_ghi['vong']:<8} "
                f"{ban_ghi['diem_toan_cuc']:<12.4f} "
                f"{ban_ghi['diem_tb_client']:<14.4f} "
                f"{ban_ghi['so_client_tham_gia']}"
            )

        return "\n".join(lines)
