"""FederatedRLHF - Học Tăng cường Phân tán có Ký số HMAC-SHA256 & Phòng vệ Byzantine."""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.edge.p2p_network import TokenLedger
from vietnamese_ai.federated.learning import HocLienKet


class FederatedRLHF(HocLienKet):
    """
    Học Tăng Cường Phân Tán (Federated RLHF / Federated DPO) có bảo vệ toàn vẹn:
    - Xác thực chữ ký HMAC-SHA256 cho bản cập nhật `delta_lora` từ các Edge Node.
    - Chống tấn công đầu độc trọng số (Byzantine Gradient Poisoning) bằng kiểm tra `NaN`/`Inf`,
      giới hạn chuẩn L2 (`nguong_chuan_toi_da`) và gộp bền vững (`fedavg`, `median`, `trimmed_mean`).
    - Kết nối trực tiếp vào `LoRAAdapter` hoặc `GPTModel` thông qua `ap_dung_vao_lora()`.
    """

    def __init__(
        self,
        so_client: int,
        so_vong: int = 1,
        token_ledger: Optional[TokenLedger] = None,
        khoa_bi_mat_mang: Optional[str] = None,
        yeu_cau_chu_ky: bool = False,
        nguong_chuan_toi_da: float = 50.0,
        phuong_phap_gop: str = "fedavg",
    ):
        super().__init__(so_client=so_client, so_vong=so_vong)
        self.token_ledger = token_ledger or TokenLedger()
        self.khoa_bi_mat_mang = khoa_bi_mat_mang or "vai_federated_default_secret"
        self.yeu_cau_chu_ky = yeu_cau_chu_ky
        self.nguong_chuan_toi_da = max(1e-3, float(nguong_chuan_toi_da))
        self.phuong_phap_gop = phuong_phap_gop.lower()
        self.logger = logging.getLogger("FederatedRLHF")
        self.danh_sach_delta_lora: List[Any] = []
        self.so_goi_bi_chan: int = 0

    @staticmethod
    def _tinh_chuan_delta(delta: Any) -> float:
        """Tính chuẩn L2 của `delta_lora` (hỗ trợ số thực, `np.ndarray` và `Dict[str, np.ndarray]`)."""
        if isinstance(delta, dict):
            tong_binh_phuong = 0.0
            for v in delta.values():
                arr = np.asarray(v, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    return float("inf")
                tong_binh_phuong += float(np.sum(arr * arr))
            return float(np.sqrt(tong_binh_phuong))
        arr = np.asarray(delta, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return float("inf")
        return float(np.linalg.norm(arr))

    def tao_chu_ky_dpo(self, user_id: str, delta_lora: Any, loss: float = 0.0) -> str:
        """Tạo chữ ký HMAC-SHA256 cho gói cập nhật Local DPO của Edge Node."""
        if isinstance(delta_lora, dict):
            serial = {k: np.asarray(v, dtype=np.float64).round(8).tolist() for k, v in sorted(delta_lora.items())}
        else:
            serial = np.asarray(delta_lora, dtype=np.float64).round(8).tolist()
        payload = json.dumps({"u": user_id, "d": serial, "l": round(float(loss), 6)}, sort_keys=True)
        return hmac.new(
            self.khoa_bi_mat_mang.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def nhan_phan_hoi_dpo(self, user_id: str, data_dpo: Dict[str, Any]) -> bool:
        """
        Nhận và thẩm định gói cập nhật Local DPO từ Edge Node.
        Trả về `True` nếu gói hợp lệ và được thưởng EVON Token; ném `PermissionError` / `ValueError` nếu bị giả mạo hoặc đầu độc.
        """
        if "delta_lora" not in data_dpo:
            raise ValueError("Thiếu trường 'delta_lora' trong gói phản hồi DPO")

        delta = data_dpo["delta_lora"]
        loss = float(data_dpo.get("loss", 0.0))
        chu_ky = data_dpo.get("chu_ky")

        if self.yeu_cau_chu_ky or chu_ky is not None:
            if not chu_ky or not isinstance(chu_ky, str):
                self.so_goi_bi_chan += 1
                raise PermissionError(f"Từ chối gói DPO từ '{user_id}': Thiếu chữ ký HMAC-SHA256")
            ky_vong = self.tao_chu_ky_dpo(user_id, delta, loss)
            if not hmac.compare_digest(chu_ky, ky_vong):
                self.so_goi_bi_chan += 1
                raise PermissionError(f"Từ chối gói DPO từ '{user_id}': Chữ ký HMAC không hợp lệ!")

        chuan_l2 = self._tinh_chuan_delta(delta)
        if not np.isfinite(chuan_l2) or chuan_l2 > self.nguong_chuan_toi_da:
            self.so_goi_bi_chan += 1
            raise ValueError(
                f"Bảo mật Byzantine: Chặn gói DPO bất thường từ '{user_id}' "
                f"(L2 norm={chuan_l2:.2f} > ngưỡng {self.nguong_chuan_toi_da})"
            )

        self.logger.info(f"Đã nhận trọng số Local DPO hợp lệ từ {user_id} (Loss: {loss:.4f})")
        self.danh_sach_delta_lora.append(delta)
        self.token_ledger.thuong_token_danh_gia(user_id, chat_luong_danh_gia=1.0)
        return True

    def _gop_mang(self, danh_sach_arr: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(danh_sach_arr, axis=0)
        if self.phuong_phap_gop == "median":
            return np.median(stacked, axis=0)
        if self.phuong_phap_gop == "trimmed_mean" and len(danh_sach_arr) >= 3:
            sap_xep = np.sort(stacked, axis=0)
            cat = max(1, len(danh_sach_arr) // 5)
            if len(danh_sach_arr) - 2 * cat >= 1:
                return np.mean(sap_xep[cat:-cat], axis=0)
        return np.mean(stacked, axis=0)

    def tong_hop_lora(self) -> Any:
        """
        Tổng hợp các bản cập nhật LoRA bằng `FedAvg`, `Coordinate-wise Median` hoặc `Trimmed Mean`.
        """
        if not self.danh_sach_delta_lora:
            self.logger.warning("Không có trọng số LoRA nào để tổng hợp.")
            return None

        tong_so = len(self.danh_sach_delta_lora)
        mau_dau = self.danh_sach_delta_lora[0]

        if isinstance(mau_dau, dict):
            global_lora: Dict[str, np.ndarray] = {}
            for key in mau_dau:
                cac_mang = [np.asarray(d[key], dtype=np.float64) for d in self.danh_sach_delta_lora if key in d]
                global_lora[key] = self._gop_mang(cac_mang)
        elif isinstance(mau_dau, (int, float, np.number)):
            arr_vo_huong = self._gop_mang([np.asarray(d, dtype=np.float64) for d in self.danh_sach_delta_lora])
            global_lora = float(arr_vo_huong)  # type: ignore[assignment]
        else:
            global_lora = self._gop_mang([np.asarray(d, dtype=np.float64) for d in self.danh_sach_delta_lora])  # type: ignore[assignment]

        self.logger.info(f"Đã tổng hợp thành công {tong_so} bản cập nhật LoRA an toàn.")
        self.danh_sach_delta_lora = []
        return global_lora

    def ap_dung_vao_lora(self, muc_tieu: Any, he_so: float = 1.0) -> Dict[str, Any]:
        """Cập nhật trực tiếp trọng số tổng hợp vào `LoRAAdapter` hoặc `GPTModel`."""
        ban_gop = self.tong_hop_lora()
        if ban_gop is None:
            return {"da_cap_nhat": False, "so_lop": 0}

        so_lop = 0
        if hasattr(muc_tieu, "_layers") and isinstance(ban_gop, dict):
            for ten_lop, layer in muc_tieu._layers.items():
                if f"{ten_lop}.B" in ban_gop:
                    layer.B = layer.B + he_so * np.asarray(ban_gop[f"{ten_lop}.B"])
                    so_lop += 1
                elif ten_lop in ban_gop and np.asarray(ban_gop[ten_lop]).shape == layer.B.shape:
                    layer.B = layer.B + he_so * np.asarray(ban_gop[ten_lop])
                    so_lop += 1
        return {"da_cap_nhat": so_lop > 0, "so_lop": so_lop, "ban_gop": ban_gop}
