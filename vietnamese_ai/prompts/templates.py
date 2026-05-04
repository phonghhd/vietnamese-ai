"""MauPrompt - quản lý prompt templates cho tiếng Việt."""

import re
from typing import Any, Dict, List, Optional


class MauPrompt:
    """
    Quản lý prompt templates với biến và conditional logic.

    Sử dụng:
        >>> mau = MauPrompt("Tóm tắt văn bản sau: {{noi_dung}}")
        >>> prompt = mau.render(noi_dung="Văn bản cần tóm tắt...")
        >>>
        >>> mau = MauPrompt.from_template("phan_tich", "Phân tích {{chu_de}}...")
        >>> mau_danh_sach = MauPrompt.danh_sach_mau_mac_dinh()
    """

    _templates: Dict[str, "MauPrompt"] = {}

    def __init__(
        self,
        mau: str,
        ten: str = "custom",
        mo_ta: str = "",
        bien_mac_dinh: Optional[Dict[str, str]] = None,
    ):
        self.mau = mau
        self.ten = ten
        self.mo_ta = mo_ta
        self.bien_mac_dinh = bien_mac_dinh or {}
        self._bien = self._trich_bien(mau)

    def render(self, **kwargs: Any) -> str:
        """
        Render template với các biến.

        Args:
            **kwargs: Giá trị cho các biến

        Returns:
            Prompt đã render
        """
        ket_qua = self.mau

        # Gộp default values
        gia_tri = {**self.bien_mac_dinh, **kwargs}

        for bien, gia_tri_val in gia_tri.items():
            pattern = r"\{\{\s*" + re.escape(bien) + r"\s*\}\}"
            ket_qua = re.sub(pattern, str(gia_tri_val), ket_qua)

        # Kiểm tra biến chưa được điền
        chua_dien = self._trich_bien(ket_qua)
        if chua_dien:
            missing = ", ".join(chua_dien)
            raise ValueError(f"Chưa điền giá trị cho biến: {missing}")

        return ket_qua

    def danh_sach_bien(self) -> List[str]:
        """Danh sách biến trong template."""
        return list(self._bien)

    def co_bien(self, ten_bien: str) -> bool:
        """Kiểm tra template có biến này không."""
        return ten_bien in self._bien

    def them_bien_mac_dinh(self, ten: str, gia_tri: str) -> None:
        """Thêm giá trị mặc định cho biến."""
        self.bien_mac_dinh[ten] = gia_tri

    @staticmethod
    def _trich_bien(mau: str) -> set:
        """Trích xuất tên biến từ template."""
        return set(re.findall(r"\{\{\s*(\w+)\s*\}\}", mau))

    @classmethod
    def from_template(cls, ten: str, mau: str, mo_ta: str = "") -> "MauPrompt":
        """Tạo và đăng ký template."""
        tpl = cls(mau, ten=ten, mo_ta=mo_ta)
        cls._templates[ten] = tpl
        return tpl

    @classmethod
    def lay_template(cls, ten: str) -> Optional["MauPrompt"]:
        """Lấy template theo tên."""
        return cls._templates.get(ten)

    @classmethod
    def danh_sach_template(cls) -> List[str]:
        """Danh sách tên các template đã đăng ký."""
        return list(cls._templates.keys())

    @classmethod
    def danh_sach_mau_mac_dinh(cls) -> Dict[str, "MauPrompt"]:
        """Tạo các template mặc định cho tiếng Việt."""
        templates = {}

        templates["tom_tat"] = cls(
            ten="tom_tat",
            mo_ta="Tóm tắt văn bản",
            mau=(
                "Hãy tóm tắt văn bản sau một cách ngắn gọn và súc tích.\n\n"
                "Văn bản:\n{{noi_dung}}\n\n"
                "Yêu cầu: {{yeu_cau}}\n\n"
                "Tóm tắt:"
            ),
            bien_mac_dinh={"yeu_cau": "Tóm tắt trong 3-5 câu"},
        )

        templates["phan_tich"] = cls(
            ten="phan_tich",
            mo_ta="Phân tích vấn đề",
            mau=(
                "Phân tích chi tiết vấn đề sau:\n\n"
                "Chủ đề: {{chu_de}}\n"
                "Ngữ cảnh: {{ngu_canh}}\n\n"
                "Hãy phân tích từ các khía cạnh:\n"
                "1. Điểm mạnh\n"
                "2. Điểm yếu\n"
                "3. Đề xuất cải thiện\n"
            ),
            bien_mac_dinh={"ngu_canh": "Không có ngữ cảnh bổ sung"},
        )

        templates["dich_thuat"] = cls(
            ten="dich_thuat",
            mo_ta="Dịch thuật",
            mau=(
                "Dịch đoạn văn sau từ {{nguon}} sang {{dich}}:\n\n"
                "{{noi_dung}}\n\n"
                "Bản dịch:"
            ),
        )

        templates["hoi_dap"] = cls(
            ten="hoi_dap",
            mo_ta="Hỏi đáp",
            mau=(
                "Dựa trên thông tin sau, hãy trả lời câu hỏi.\n\n"
                "Thông tin:\n{{tai_lieu}}\n\n"
                "Câu hỏi: {{cau_hoi}}\n\n"
                "Trả lời:"
            ),
        )

        templates["sinh_code"] = cls(
            ten="sinh_code",
            mo_ta="Sinh code",
            mau=(
                "Viết code {{ngon_ngu}} để thực hiện:\n\n"
                "{{mo_ta}}\n\n"
                "Yêu cầu bổ sung: {{yeu_cau}}\n\n"
                "Code:"
            ),
            bien_mac_dinh={"yeu_cau": "Không có yêu cầu bổ sung"},
        )

        templates["phan_biet"] = cls(
            ten="phan_biet",
            mo_ta="So sánh hai khái niệm",
            mau=(
                "So sánh {{a}} và {{b}}:\n\n"
                "Tiêu chí so sánh:\n{{tieu_chi}}\n\n"
                "Bảng so sánh:"
            ),
            bien_mac_dinh={"tieu_chi": "Tất cả các tiêu chí"},
        )

        for ten, tpl in templates.items():
            cls._templates[ten] = tpl

        return templates

    def __repr__(self) -> str:
        return f"MauPrompt(ten='{self.ten}', bien={self._bien})"
