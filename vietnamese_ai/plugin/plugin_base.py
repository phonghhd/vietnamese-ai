"""
PluginCoSo & Các chuẩn Plugin V-Plugin 2.0 (Extension System) cho Vietnamese AI Framework.
Cung cấp các lớp cơ sở:
- PluginCoSo: Lớp cơ sở tổng quát.
- PluginCongCu: Chuẩn Plugin cung cấp công cụ Function Calling cho Tác tử (TacTu / Agents).
- PluginTriThuc: Chuẩn Plugin cung cấp tri thức, tài liệu và từ điển ngành cho RAG Pipeline.
- PluginGiaiPhap: Chuẩn Plugin đóng gói quy trình giải pháp end-to-end cho ngành chuyên biệt.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class PluginCoSo:
    """Lớp cơ sở bắt buộc cho mọi Plugin."""

    # Meta data của Plugin
    ten: str = "Chưa đặt tên"
    phien_ban: str = "1.0.0"
    mo_ta: str = "Mô tả ngắn về plugin"
    loai: str = "co_so"

    def __init__(self):
        self.trang_thai_hoat_dong = False

    def khoi_dong(self) -> bool:
        """
        Được gọi khi hệ thống vừa nạp Plugin lên.
        Dùng để khởi tạo biến, thiết lập ban đầu.

        Returns:
            True nếu khởi động thành công, False nếu thất bại.
        """
        self.trang_thai_hoat_dong = True
        return True

    def thuc_thi(self, **kwargs) -> Dict[str, Any]:
        """
        Hàm xử lý logic chính của Plugin.

        Args:
            **kwargs: Dữ liệu động truyền từ Framework hoặc Workflow.

        Returns:
            Kết quả tính toán.
        """
        raise NotImplementedError("Plugin phải cài đặt hàm thuc_thi()")

    def tat(self) -> None:
        """
        Được gọi khi Quản trị viên gỡ bỏ (Unload) Plugin khỏi hệ thống.
        Dùng để đóng kết nối, giải phóng bộ nhớ.
        """
        self.trang_thai_hoat_dong = False

    def thong_tin(self) -> Dict[str, Any]:
        """Lấy thông tin mô tả chi tiết của Plugin."""
        return {
            "ten": self.ten,
            "phien_ban": self.phien_ban,
            "mo_ta": self.mo_ta,
            "loai": getattr(self, "loai", "co_so"),
            "trang_thai": self.trang_thai_hoat_dong,
        }


class PluginCongCu(PluginCoSo):
    """
    Chuẩn Plugin công cụ (Tool / Function Calling Plugin).
    Cung cấp các hàm nghiệp vụ có thể gọi tự động bởi Tác tử (TacTu / Agents).
    """

    loai: str = "cong_cu"

    def dang_ky_cong_cu(self) -> List[Dict[str, Any]]:
        """
        Khai báo danh sách các công cụ mà Plugin cung cấp.

        Returns:
            Danh sách dict công cụ theo cấu trúc:
            [
                {
                    "ten": "ten_cong_cu",
                    "mo_ta": "Mô tả hành động của công cụ",
                    "ham": callable,
                    "tham_so": {"ten_ts": {"kieu": "str", "bat_buoc": True}},  # tùy chọn
                    "yeu_cau_xac_nhan": False,
                }
            ]
        """
        raise NotImplementedError("PluginCongCu phải cài đặt dang_ky_cong_cu()")

    def thuc_thi_cong_cu(self, ten_cong_cu: str, **kwargs) -> Any:
        """Thực thi một công cụ cụ thể theo tên."""
        for cc in self.dang_ky_cong_cu():
            if cc.get("ten") == ten_cong_cu:
                ham: Optional[Callable] = cc.get("ham")
                if callable(ham):
                    return ham(**kwargs)
        raise ValueError(f"Không tìm thấy công cụ '{ten_cong_cu}' trong plugin '{self.ten}'")

    def thuc_thi(self, **kwargs) -> Dict[str, Any]:
        """Triển khai hàm thuc_thi mặc định: điều phối tới công cụ được yêu cầu."""
        ten_cc = kwargs.pop("action", None) or kwargs.pop("ten_cong_cu", None)
        if not ten_cc:
            ds_cc = [cc.get("ten") for cc in self.dang_ky_cong_cu()]
            return {
                "thanh_cong": True,
                "plugin": self.ten,
                "danh_sach_cong_cu": ds_cc,
            }
        kq = self.thuc_thi_cong_cu(ten_cc, **kwargs)
        return {"thanh_cong": True, "cong_cu": ten_cc, "ket_qua": kq}


class PluginTriThuc(PluginCoSo):
    """
    Chuẩn Plugin tri thức (Domain Knowledge / RAG Plugin).
    Cung cấp corpus tài liệu, bảng tra cứu và từ điển thuật ngữ nghiệp vụ ngành.
    """

    loai: str = "tri_thuc"

    def nap_nguon_tri_thuc(self) -> List[Dict[str, Any]]:
        """
        Cung cấp danh sách các đoạn văn bản / tài liệu chuyên ngành để đưa vào RAG Pipeline.

        Returns:
            Danh sách dict tài liệu:
            [
                {
                    "tieu_de": "Tên văn bản / chính sách",
                    "noi_dung": "Toàn văn nội dung tài liệu",
                    "nguon": "Nguồn tham chiếu (ví dụ: Thông tư 45/2013)",
                    "metadata": {"chu_de": "..."},
                }
            ]
        """
        return []

    def lay_tu_dien_nghiep_vu(self) -> Dict[str, str]:
        """
        Cung cấp từ điển thuật ngữ chuyên ngành (viết tắt -> giải nghĩa hoặc từ đồng nghĩa).
        """
        return {}

    def tra_cuu_nhanh(self, tu_khoa: str) -> Optional[Any]:
        """Tra cứu nhanh định nghĩa hoặc thông tin theo từ khóa nghiệp vụ."""
        td = self.lay_tu_dien_nghiep_vu()
        tk_clean = tu_khoa.strip().lower()
        for k, v in td.items():
            if k.lower() == tk_clean:
                return v
        return None

    def thuc_thi(self, **kwargs) -> Dict[str, Any]:
        """Triển khai hàm thuc_thi mặc định: tra cứu từ khóa trong tri thức ngành."""
        query = kwargs.get("tu_khoa") or kwargs.get("query") or ""
        kq = self.tra_cuu_nhanh(query) if query else None
        return {
            "thanh_cong": True,
            "plugin": self.ten,
            "tu_khoa": query,
            "ket_qua": kq,
            "so_luong_tai_lieu": len(self.nap_nguon_tri_thuc()),
        }


class PluginGiaiPhap(PluginCoSo):
    """
    Chuẩn Plugin giải pháp toàn diện (End-to-End Solution Plugin).
    Tích hợp quy trình xử lý hoàn chỉnh cho một bài toán nghiệp vụ cụ thể.
    """

    loai: str = "giai_phap"

    def khoi_chay_giai_phap(self, **kwargs) -> Any:
        """Thực thi luồng giải pháp nghiệp vụ hoàn chỉnh."""
        raise NotImplementedError("PluginGiaiPhap phải cài đặt khoi_chay_giai_phap()")

    def thuc_thi(self, **kwargs) -> Dict[str, Any]:
        """Triển khai hàm thuc_thi mặc định: khởi chạy giải pháp."""
        kq = self.khoi_chay_giai_phap(**kwargs)
        return {
            "thanh_cong": True,
            "plugin": self.ten,
            "loai": self.loai,
            "ket_qua": kq,
        }
