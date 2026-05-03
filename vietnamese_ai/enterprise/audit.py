"""NhatKyHoatDong - Audit log cho hệ thống."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class NhatKyHoatDong:
    """
    Nhật ký hoạt động (Audit Log).

    Ghi lại mọi hoạt động quan trọng trong hệ thống:
    - Huấn luyện mô hình
    - Dự đoán
    - Triển khai
    - Quản lý người dùng
    - Truy cập API

    Sử dụng:
        >>> nhat_ky = NhatKyHoatDong()
        >>> nhat_ky.ghi("train", "admin", "Huấn luyện PhanLoai", {"accuracy": 0.95})
        >>> nhat_ky.tim_kiem(nguoi_dung="admin")
        >>> nhat_ky.bao_cao()
    """

    def __init__(self, duong_dan: str = "audit_log"):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("NhatKyHoatDong")
        self._log_file = self.duong_dan / "audit.jsonl"

    def ghi(
        self,
        hanh_dong: str,
        nguoi_dung: str = "system",
        mo_ta: str = "",
        du_lieu: Optional[Dict] = None,
        muc_do: str = "info",
    ) -> Dict:
        """
        Ghi một hoạt động.

        Args:
            hanh_dong: Loại hành động (train, predict, deploy, login, ...)
            nguoi_dung: Người thực hiện
            mo_ta: Mô tả chi tiết
            du_lieu: Dữ liệu đính kèm
            muc_do: Mức độ (info, warning, error)

        Returns:
            Bản ghi đã ghi
        """
        ban_ghi = {
            "thoi_gian": datetime.now().isoformat(),
            "timestamp": time.time(),
            "hanh_dong": hanh_dong,
            "nguoi_dung": nguoi_dung,
            "mo_ta": mo_ta,
            "du_lieu": du_lieu or {},
            "muc_do": muc_do,
        }

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(ban_ghi, ensure_ascii=False) + "\n")

        return ban_ghi

    def tim_kiem(
        self,
        nguoi_dung: Optional[str] = None,
        hanh_dong: Optional[str] = None,
        tu_ngay: Optional[str] = None,
        den_ngay: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Tìm kiếm nhật ký.

        Args:
            nguoi_dung: Lọc theo người dùng
            hanh_dong: Lọc theo hành động
            tu_ngay: Từ ngày (ISO format)
            den_ngay: Đến ngày
            limit: Số kết quả tối đa
        """
        if not self._log_file.exists():
            return []

        ket_qua = []
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ban_ghi = json.loads(line)

                    if nguoi_dung and ban_ghi.get("nguoi_dung") != nguoi_dung:
                        continue
                    if hanh_dong and ban_ghi.get("hanh_dong") != hanh_dong:
                        continue
                    if tu_ngay and ban_ghi.get("thoi_gian", "") < tu_ngay:
                        continue
                    if den_ngay and ban_ghi.get("thoi_gian", "") > den_ngay:
                        continue

                    ket_qua.append(ban_ghi)

                    if len(ket_qua) >= limit:
                        break
                except json.JSONDecodeError:
                    continue

        return ket_qua

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê tổng quan nhật ký."""
        tat_ca = self.tim_kiem(limit=10000)

        dem_hanh_dong = {}
        dem_nguoi_dung = {}
        dem_muc_do = {"info": 0, "warning": 0, "error": 0}

        for bg in tat_ca:
            ha = bg.get("hanh_dong", "unknown")
            dem_hanh_dong[ha] = dem_hanh_dong.get(ha, 0) + 1

            nd = bg.get("nguoi_dung", "unknown")
            dem_nguoi_dung[nd] = dem_nguoi_dung.get(nd, 0) + 1

            md = bg.get("muc_do", "info")
            dem_muc_do[md] = dem_muc_do.get(md, 0) + 1

        return {
            "tong_ban_ghi": len(tat_ca),
            "theo_hanh_dong": dem_hanh_dong,
            "theo_nguoi_dung": dem_nguoi_dung,
            "theo_muc_do": dem_muc_do,
        }

    def bao_cao(self, limit: int = 50) -> str:
        """Tạo báo cáo nhật ký dạng text."""
        tat_ca = self.tim_kiem(limit=limit)
        tk = self.thong_ke()

        lines = ["=== NHẬT KÝ HOẠT ĐỘNG ===\n"]
        lines.append(f"Tổng bản ghi: {tk['tong_ban_ghi']}\n")

        lines.append("Theo hành động:")
        for ha, dem in tk["theo_hanh_dong"].items():
            lines.append(f"  {ha}: {dem}")

        lines.append(f"\n{limit} hoạt động gần nhất:")
        for bg in tat_ca[-limit:]:
            lines.append(f"  [{bg['thoi_gian']}] {bg['nguoi_dung']} - {bg['hanh_dong']}: {bg['mo_ta']}")

        return "\n".join(lines)

    def xoa(self, so_ngay_giu: int = 90) -> int:
        """Xóa nhật ký cũ hơn số ngày quy định."""
        if not self._log_file.exists():
            return 0

        nguong = time.time() - so_ngay_giu * 86400
        giu_lai = []
        xoa_count = 0

        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    bg = json.loads(line.strip())
                    if bg.get("timestamp", 0) >= nguong:
                        giu_lai.append(line)
                    else:
                        xoa_count += 1
                except (json.JSONDecodeError, ValueError):
                    giu_lai.append(line)

        with open(self._log_file, "w", encoding="utf-8") as f:
            f.writelines(giu_lai)

        self.logger.info(f"Đã xóa {xoa_count} bản ghi cũ")
        return xoa_count
