"""LuongAnToan - guardrails cho LLM output."""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


class LuongAnToan:
    """
    Guardrails cho LLM output - kiểm tra an toàn và chất lượng.

    Hỗ trợ:
    - Content filtering (từ cấm, nội dung nhạy cảm)
    - Format validation (JSON, markdown, code blocks)
    - Length constraints
    - Custom validation rules
    - PII detection (số điện thoại, email, CMND)

    Sử dụng:
        >>> guardrail = LuongAnToan()
        >>> ket_qua = guardrail.kiem_tra("Nội dung cần kiểm tra")
        >>> if ket_qua["an_toan"]:
        ...     print("Nội dung an toàn")
    """

    def __init__(
        self,
        tu_cam: Optional[List[str]] = None,
        toi_da_do_dai: int = 10000,
        toi_thieu_do_dai: int = 0,
        chan_pii: bool = False,
        dinh_dang_yeu_cau: Optional[str] = None,
        rules: Optional[List[Callable[[str], Tuple[bool, str]]]] = None,
    ):
        self.tu_cam = tu_cam or []
        self.toi_da_do_dai = toi_da_do_dai
        self.toi_thieu_do_dai = toi_thieu_do_dai
        self.chan_pii = chan_pii
        self.dinh_dang_yeu_cau = dinh_dang_yeu_cau
        self.rules = rules or []

        self._mau_pii = {
            "so_dien_thoai": r"(?:\+84|0)[1-9]\d{8}",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "cmnd": r"\b\d{9}(?:\d{3})?\b",
            "the_ngan_hang": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        }

    def kiem_tra(self, noi_dung: str) -> Dict[str, Any]:
        """
        Kiểm tra nội dung có an toàn không.

        Args:
            noi_dung: Nội dung cần kiểm tra

        Returns:
            {an_toan, loi, canh_bao, chi_tiet}
        """
        loi = []
        canh_bao = []

        # Kiểm tra độ dài
        if len(noi_dung) > self.toi_da_do_dai:
            loi.append(
                f"Nội dung quá dài ({len(noi_dung)} > {self.toi_da_do_dai} ký tự)"
            )

        if len(noi_dung) < self.toi_thieu_do_dai:
            loi.append(
                f"Nội dung quá ngắn ({len(noi_dung)} < {self.toi_thieu_do_dai} ký tự)"
            )

        # Kiểm tra từ cấm
        noi_dung_lower = noi_dung.lower()
        for tu in self.tu_cam:
            if tu.lower() in noi_dung_lower:
                loi.append(f"Chứa từ cấm: '{tu}'")

        # Kiểm tra PII
        if self.chan_pii:
            pii_phat_hien = self._kiem_tra_pii(noi_dung)
            for loai, matches in pii_phat_hien.items():
                if matches:
                    canh_bao.append(
                        f"Phát hiện {loai}: {len(matches)} kết quả"
                    )

        # Kiểm tra định dạng
        if self.dinh_dang_yeu_cau:
            if not self._kiem_tra_dinh_dang(noi_dung, self.dinh_dang_yeu_cau):
                loi.append(
                    f"Nội dung không đúng định dạng: {self.dinh_dang_yeu_cau}"
                )

        # Custom rules
        for rule in self.rules:
            passed, message = rule(noi_dung)
            if not passed:
                loi.append(message)

        return {
            "an_toan": len(loi) == 0,
            "loi": loi,
            "canh_bao": canh_bao,
            "so_loi": len(loi),
            "so_canh_bao": len(canh_bao),
            "do_dai": len(noi_dung),
        }

    def loc_pii(self, noi_dung: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Lọc PII khỏi nội dung.

        Args:
            noi_dung: Nội dung gốc

        Returns:
            (noi_dung_da_loc, pii_phat_hien)
        """
        ket_qua = noi_dung
        pii_phat_hien = {}

        for loai, mau in self._mau_pii.items():
            matches = re.findall(mau, ket_qua)
            pii_phat_hien[loai] = matches

            if matches:
                if loai == "so_dien_thoai":
                    ket_qua = re.sub(mau, "[SĐT đã ẩn]", ket_qua)
                elif loai == "email":
                    ket_qua = re.sub(mau, "[Email đã ẩn]", ket_qua)
                elif loai == "cmnd":
                    ket_qua = re.sub(mau, "[CMND đã ẩn]", ket_qua)
                elif loai == "the_ngan_hang":
                    ket_qua = re.sub(mau, "[Số thẻ đã ẩn]", ket_qua)

        return ket_qua, pii_phat_hien

    def _kiem_tra_pii(self, noi_dung: str) -> Dict[str, List[str]]:
        """Kiểm tra PII trong nội dung."""
        ket_qua = {}
        for loai, mau in self._mau_pii.items():
            ket_qua[loai] = re.findall(mau, noi_dung)
        return ket_qua

    def _kiem_tra_dinh_dang(
        self, noi_dung: str, dinh_dang: str
    ) -> bool:
        """Kiểm tra nội dung có đúng định dạng không."""
        if dinh_dang == "json":
            import json
            try:
                json.loads(noi_dung)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        elif dinh_dang == "markdown":
            return bool(re.search(r"^#+\s", noi_dung, re.MULTILINE))
        elif dinh_dang == "code":
            return "```" in noi_dung
        elif dinh_dang == "number":
            try:
                float(noi_dung.strip())
                return True
            except ValueError:
                return False
        return True

    def them_tu_cam(self, *tu: str) -> None:
        """Thêm từ cấm."""
        self.tu_cam.extend(tu)

    def them_rule(self, rule: Callable[[str], Tuple[bool, str]]) -> None:
        """Thêm custom validation rule."""
        self.rules.append(rule)

    def them_mau_pii(self, loai: str, mau: str) -> None:
        """Thêm mẫu PII detection."""
        self._mau_pii[loai] = mau

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê cấu hình."""
        return {
            "so_tu_cam": len(self.tu_cam),
            "toi_da_do_dai": self.toi_da_do_dai,
            "chan_pii": self.chan_pii,
            "dinh_dang_yeu_cau": self.dinh_dang_yeu_cau,
            "so_rules": len(self.rules),
            "so_mau_pii": len(self._mau_pii),
        }

    def __repr__(self) -> str:
        return (
            f"LuongAnToan(so_tu_cam={len(self.tu_cam)}, "
            f"chan_pii={self.chan_pii})"
        )

class JSONOutputParser:
    """Ép và trích xuất JSON từ chuỗi văn bản của LLM."""
    
    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """Trích xuất khối JSON từ văn bản trả về."""
        # LLM thường hay bọc JSON trong block markdown ```json ... ```
        pattern = r"```(?:json)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        json_str = matches[0].strip() if matches else text.strip()
        
        import json
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Không thể parse JSON từ đầu ra: {str(e)}\nĐầu ra thô: {text}")

class ToxicityFilter:
    """Màng lọc từ ngữ độc hại cơ bản (cho tiếng Việt)."""
    
    # Danh sách ví dụ, thực tế nên dài và che giấu đi
    BAD_WORDS = ["vkl", "đcm", "vl", "chửi", "đánh nhau", "giết"]
    
    @classmethod
    def check(cls, text: str) -> bool:
        """Kiểm tra có từ khóa xấu không. Trả về True nếu bị chặn."""
        text_lower = text.lower()
        for word in cls.BAD_WORDS:
            # Kiểm tra từ đó có xuất hiện độc lập không
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                return True
        return False
        
    @classmethod
    def filter(cls, text: str) -> str:
        """Che giấu từ độc hại bằng dấu sao."""
        text_lower = text.lower()
        filtered = text
        for word in cls.BAD_WORDS:
            # Thay thế không phân biệt hoa thường
            filtered = re.sub(r'(?i)\b' + re.escape(word) + r'\b', '*' * len(word), filtered)
        return filtered
