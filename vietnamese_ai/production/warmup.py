"""LamNongModel - Model warm-up cho production."""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class LamNongModel:
    """
    Model warm-up - pre-load và warm-up models trước khi nhận traffic.

    Hỗ trợ:
    - Pre-load models vào memory
    - Warm-up inference (chạy dummy predictions)
    - Lazy loading với warm-up on first request
    - Model pool management
    - Background refresh

    Sử dụng:
        >>> warmup = LamNongModel()
        >>> warmup.dang_ky_model("classifier", model, du_lieu_mau=X_sample)
        >>> warmup.lam_nong_tat_ca()
        >>> model = warmup.lay_model("classifier")  # Đã sẵn sàng
    """

    def __init__(
        self,
        tu_dong_lam_nong: bool = True,
        thoi_gian_refresh: float = 3600.0,
        so_lan_warmup: int = 3,
    ):
        self.tu_dong_lam_nong = tu_dong_lam_nong
        self.thoi_gian_refresh = thoi_gian_refresh
        self.so_lan_warmup = so_lan_warmup
        self.logger = Logger("LamNongModel")

        self._models: Dict[str, Any] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._warmup_data: Dict[str, Any] = {}
        self._trang_thai: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._refresh_thread: Optional[threading.Thread] = None
        self._dang_chay = False

    def dang_ky_model(
        self,
        ten: str,
        model: Any,
        du_lieu_mau: Optional[Any] = None,
        ham_du_doan: Optional[Callable] = None,
        tu_dong_refresh: bool = False,
    ) -> None:
        """
        Đăng ký model cho warm-up.

        Args:
            ten: Tên model
            model: Model object
            du_lieu_mau: Dữ liệu mẫu để warm-up
            ham_du_doan: Custom predict function
            tu_dong_refresh: Tự động refresh model
        """
        with self._lock:
            self._models[ten] = model
            self._configs[ten] = {
                "ham_du_doan": ham_du_doan,
                "tu_dong_refresh": tu_dong_refresh,
            }
            self._warmup_data[ten] = du_lieu_mau
            self._trang_thai[ten] = {
                "da_warmup": False,
                "lan_warmup_cuoi": 0.0,
                "so_lan_warmup": 0,
                "thoi_gian_warmup": 0.0,
            }

    def lam_nong(self, ten: str) -> Dict[str, Any]:
        """
        Warm-up một model.

        Args:
            ten: Tên model

        Returns:
            {ten, thoi_gian_ms, so_lan, trang_thai}
        """
        with self._lock:
            if ten not in self._models:
                raise ValueError(f"Model '{ten}' chưa được đăng ký")

        model = self._models[ten]
        config = self._configs[ten]
        du_lieu = self._warmup_data.get(ten)

        bat_dau = time.time()
        loi = None

        for i in range(self.so_lan_warmup):
            try:
                if du_lieu is not None:
                    if config["ham_du_doan"]:
                        config["ham_du_doan"](du_lieu)
                    elif hasattr(model, "du_doan"):
                        model.du_doan(du_lieu)
                    elif hasattr(model, "predict"):
                        model.predict(du_lieu)
                    elif callable(model):
                        model(du_lieu)
            except Exception as e:
                loi = str(e)
                if i == 0:
                    self.logger.warning(f"Warm-up '{ten}' lần {i + 1} lỗi: {e}")

        thoi_gian = (time.time() - bat_dau) * 1000

        with self._lock:
            self._trang_thai[ten] = {
                "da_warmup": True,
                "lan_warmup_cuoi": time.time(),
                "so_lan_warmup": self.so_lan_warmup,
                "thoi_gian_warmup": round(thoi_gian, 2),
                "loi": loi,
            }

        self.logger.info(f"Warm-up '{ten}' hoàn tất: {thoi_gian:.1f}ms, {self.so_lan_warmup} lần")

        return {
            "ten": ten,
            "thoi_gian_ms": round(thoi_gian, 2),
            "so_lan": self.so_lan_warmup,
            "trang_thai": "ok" if loi is None else "loi",
            "loi": loi,
        }

    def lam_nong_tat_ca(self) -> Dict[str, Dict[str, Any]]:
        """Warm-up tất cả models đã đăng ký."""
        ket_qua = {}
        for ten in list(self._models.keys()):
            try:
                ket_qua[ten] = self.lam_nong(ten)
            except Exception as e:
                ket_qua[ten] = {"ten": ten, "trang_thai": "loi", "loi": str(e)}
        return ket_qua

    def lay_model(self, ten: str) -> Any:
        """
        Lấy model (tự warm-up nếu chưa).

        Args:
            ten: Tên model

        Returns:
            Model object
        """
        with self._lock:
            if ten not in self._models:
                raise ValueError(f"Model '{ten}' chưa được đăng ký")

            trang_thai = self._trang_thai.get(ten, {})
            if self.tu_dong_lam_nong and not trang_thai.get("da_warmup"):
                # Warm-up ngay bây giờ
                self._lock.release()
                try:
                    self.lam_nong(ten)
                finally:
                    self._lock.acquire()

            return self._models[ten]

    def bat_dau_auto_refresh(self) -> None:
        """Bắt đầu auto-refresh background thread."""
        if self._dang_chay:
            return

        self._dang_chay = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
        )
        self._refresh_thread.start()

    def dung_auto_refresh(self) -> None:
        """Dừng auto-refresh."""
        self._dang_chay = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5.0)

    def _refresh_loop(self) -> None:
        """Background refresh loop."""
        while self._dang_chay:
            time.sleep(self.thoi_gian_refresh)
            if not self._dang_chay:
                break

            for ten, config in self._configs.items():
                if config.get("tu_dong_refresh"):
                    try:
                        self.lam_nong(ten)
                    except Exception as e:
                        self.logger.warning(f"Refresh '{ten}' lỗi: {e}")

    def xoa_model(self, ten: str) -> bool:
        """Xóa model."""
        with self._lock:
            if ten in self._models:
                del self._models[ten]
                del self._configs[ten]
                del self._trang_thai[ten]
                self._warmup_data.pop(ten, None)
                return True
            return False

    def trang_thai_model(self, ten: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái warm-up của model."""
        with self._lock:
            return self._trang_thai.get(ten, {}).copy()

    def danh_sach_models(self) -> List[str]:
        """Danh sách models đã đăng ký."""
        with self._lock:
            return list(self._models.keys())

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê warm-up manager."""
        with self._lock:
            da_warmup = sum(1 for t in self._trang_thai.values() if t.get("da_warmup"))
            tong_time = sum(t.get("thoi_gian_warmup", 0) for t in self._trang_thai.values())

            return {
                "so_models": len(self._models),
                "da_warmup": da_warmup,
                "chua_warmup": len(self._models) - da_warmup,
                "tong_thoi_gian_warmup_ms": tong_time,
                "tu_dong_lam_nong": self.tu_dong_lam_nong,
                "auto_refresh": self._dang_chay,
            }

    def __repr__(self) -> str:
        return (
            f"LamNongModel(so_models={len(self._models)}, "
            f"da_warmup={sum(1 for t in self._trang_thai.values() if t.get('da_warmup'))})"
        )
