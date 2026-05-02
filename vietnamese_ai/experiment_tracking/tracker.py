"""TheoDoiThiNghiem - Theo dõi và quản lý thí nghiệm học máy."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class TheoDoiThiNghiem:
    """
    Theo dõi thí nghiệm học máy (tương thích MLflow API).

    Hỗ trợ:
    - Lưu thông số (params), chỉ số (metrics), mô hình (artifacts)
    - So sánh các thí nghiệm
    - Tích hợp MLflow (nếu có)

    Sử dụng:
        >>> td = TheoDoiThiNghiem("thu_nghiem_1")
        >>> td.bat_dau("logistic_lr0.01")
        >>> td.log_param("thuat_toan", "logistic")
        >>> td.log_param("toc_do_hoc", 0.01)
        >>> td.log_metric("accuracy", 0.95)
        >>> td.log_metric("f1", 0.93)
        >>> td.ket_thuc()

        >>> # Xem lịch sử
        >>> td.bao_cao()
    """

    def __init__(
        self,
        ten_thu_nghiem: str = "default",
        luu_tai: str = "experiments",
        su_dung_mlflow: bool = False,
    ):
        self.ten_thu_nghiem = ten_thu_nghiem
        self.luu_tai = Path(luu_tai) / ten_thu_nghiem
        self.luu_tai.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("TheoDoi")

        self._chay_hien_tai: Optional[Dict] = None
        self._lich_su: List[Dict] = []

        # MLflow integration
        self._mlflow = None
        if su_dung_mlflow:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_experiment(ten_thu_nghiem)
                self.logger.info("Đã kết nối MLflow")
            except ImportError:
                self.logger.warning("MLflow chưa cài. pip install mlflow")

    def bat_dau(self, ten_chay: Optional[str] = None) -> str:
        """
        Bắt đầu một thí nghiệm mới.

        Args:
            ten_chay: Tên của lần chạy (tự động nếu None)

        Returns:
            ID của lần chạy
        """
        chay_id = ten_chay or f"chay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._chay_hien_tai = {
            "id": chay_id,
            "thoi_gian_bat_dau": time.time(),
            "params": {},
            "metrics": [],
            "artifacts": [],
            "trang_thai": "dang_chay",
        }

        if self._mlflow:
            self._mlflow.start_run(run_name=chay_id)

        self.logger.info(f"Bắt đầu thí nghiệm: {chay_id}")
        return chay_id

    def log_param(self, key: str, value: Any) -> None:
        """Ghi một tham số."""
        if self._chay_hien_tai is None:
            raise RuntimeError("Chưa bắt đầu thí nghiệm. Gọi bat_dau() trước.")

        self._chay_hien_tai["params"][key] = value

        if self._mlflow:
            self._mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Ghi nhiều tham số."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Ghi một chỉ số.

        Args:
            key: Tên chỉ số
            value: Giá trị
            step: Bước (tùy chọn, cho loss theo epoch)
        """
        if self._chay_hien_tai is None:
            raise RuntimeError("Chưa bắt đầu thí nghiệm.")

        ban_ghi = {"key": key, "value": value, "step": step, "thoi_gian": time.time()}
        self._chay_hien_tai["metrics"].append(ban_ghi)

        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Ghi nhiều chỉ số."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_model(self, mo_hinh: Any, ten: str = "model") -> None:
        """Lưu mô hình."""
        if self._chay_hien_tai is None:
            raise RuntimeError("Chưa bắt đầu thí nghiệm.")

        duong_dan = self.luu_tai / self._chay_hien_tai["id"] / f"{ten}.pkl"
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        mo_hinh.luu(str(duong_dan))

        self._chay_hien_tai["artifacts"].append(str(duong_dan))
        self.logger.info(f"Đã lưu mô hình: {duong_dan}")

    def ket_thuc(self) -> Dict:
        """Kết thúc thí nghiệm và lưu kết quả."""
        if self._chay_hien_tai is None:
            raise RuntimeError("Chưa bắt đầu thí nghiệm.")

        self._chay_hien_tai["thoi_gian_ket_thuc"] = time.time()
        self._chay_hien_tai["thoi_gian_chay"] = (
            self._chay_hien_tai["thoi_gian_ket_thuc"]
            - self._chay_hien_tai["thoi_gian_bat_dau"]
        )
        self._chay_hien_tai["trang_thai"] = "hoan_tat"

        # Lưu ra JSON
        duong_dan = self.luu_tai / self._chay_hien_tai["id"] / "thong_tin.json"
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "w", encoding="utf-8") as f:
            json.dump(self._chay_hien_tai, f, ensure_ascii=False, indent=2, default=str)

        self._lich_su.append(self._chay_hien_tai)

        if self._mlflow:
            self._mlflow.end_run()

        ket_qua = self._chay_hien_tai.copy()
        self._chay_hien_tai = None

        self.logger.info(f"Kết thúc thí nghiệm: {ket_qua['id']}")
        return ket_qua

    def lay_lich_su(self) -> List[Dict]:
        """Trả về lịch sử tất cả thí nghiệm."""
        if not self._lich_su:
            # Đọc từ thư mục
            for chay_dir in sorted(self.luu_tai.iterdir()):
                if chay_dir.is_dir():
                    thong_tin = chay_dir / "thong_tin.json"
                    if thong_tin.exists():
                        with open(thong_tin, "r", encoding="utf-8") as f:
                            self._lich_su.append(json.load(f))
        return self._lich_su

    def bao_cao(self) -> str:
        """Tạo báo cáo so sánh các thí nghiệm."""
        lich_su = self.lay_lich_su()
        if not lich_su:
            return "Chưa có thí nghiệm nào."

        lines = [f"=== BÁO CÁO THÍ NGHIỆM: {self.ten_thu_nghiem} ===\n"]

        for chay in lich_su:
            lines.append(f"--- {chay['id']} ---")
            lines.append(f"  Trạng thái: {chay['trang_thai']}")
            lines.append(f"  Thời gian: {chay.get('thoi_gian_chay', 0):.2f}s")

            if chay.get("params"):
                lines.append("  Params:")
                for k, v in chay["params"].items():
                    lines.append(f"    {k}: {v}")

            if chay.get("metrics"):
                lines.append("  Metrics:")
                seen = set()
                for m in chay["metrics"]:
                    if m["key"] not in seen:
                        lines.append(f"    {m['key']}: {m['value']:.4f}")
                        seen.add(m["key"])

            lines.append("")

        return "\n".join(lines)

    def so_sanh(self) -> Dict[str, List]:
        """So sánh metrics giữa các thí nghiệm."""
        lich_su = self.lay_lich_su()
        so_sanh = {}

        for chay in lich_su:
            for m in chay.get("metrics", []):
                key = m["key"]
                if key not in so_sanh:
                    so_sanh[key] = []
                so_sanh[key].append({
                    "chay": chay["id"],
                    "gia_tri": m["value"],
                })

        return so_sanh
