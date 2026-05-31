import re
from typing import Any, List, Tuple


class GraphExtractor:
    """
    Trích xuất đồ thị (Graph Extraction) từ văn bản.
    Nhiệm vụ: Chuyển văn bản phi cấu trúc thành các bộ ba (Subject, Relation, Object).
    """

    def __init__(self, llm: Any = None, use_slm: bool = True):
        """
        Args:
            llm: Đối tượng LLM để trích xuất ngữ nghĩa nâng cao.
            use_slm: v27.0.1 - Bật mô hình ngôn ngữ nhỏ (SLM) Rule-based thay vì gọi LLM nặng.
        """
        self.llm = llm
        self.use_slm = use_slm

    def _trich_xuat_bang_slm(self, van_ban: str) -> List[Tuple[str, str, str]]:
        """Sử dụng mô hình nhỏ nội bộ (SLM / Heuristic Regex nâng cao) để bóc tách siêu tốc độ O(N)."""
        mau_quan_he = re.compile(
            r"([A-ZÀ-Ỹ][a-zà-ỹA-ZÀ-Ỹ0-9\s_]+)\s+(là|thuộc|nằm ở|được sáng lập bởi|có liên quan đến|quản lý|sở hữu|phát triển)\s+([A-ZÀ-Ỹ][a-zà-ỹA-ZÀ-Ỹ0-9\s_]+)",
            re.IGNORECASE,
        )
        ket_qua = []
        for match in mau_quan_he.finditer(van_ban):
            chu_the = match.group(1).strip()
            quan_he = match.group(2).strip().lower()
            doi_tuong = match.group(3).strip()
            if len(chu_the) > 1 and len(doi_tuong) > 1:
                ket_qua.append((chu_the, quan_he, doi_tuong))
        return ket_qua

    def _trich_xuat_bang_regex(self, van_ban: str) -> List[Tuple[str, str, str]]:
        """Fallback: Trích xuất cơ bản bằng Regex (Rất thô sơ)."""
        # Ví dụ tìm mẫu "A là B", "A thuộc B"
        mau = re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(là|thuộc|nằm ở|được sáng lập bởi)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
            re.IGNORECASE,
        )
        ket_qua = []
        for match in mau.finditer(van_ban):
            ket_qua.append((match.group(1).strip(), match.group(2).strip(), match.group(3).strip()))
        return ket_qua

    def _trich_xuat_bang_llm(self, van_ban: str) -> List[Tuple[str, str, str]]:
        """Sử dụng LLM để trích xuất các bộ ba."""
        prompt = f"""
Trích xuất các thực thể và mối quan hệ từ văn bản sau thành các bộ ba (Chủ thể, Mối quan hệ, Đối tượng).
Chỉ trả về các bộ ba dưới định dạng: "Chủ thể | Mối quan hệ | Đối tượng", mỗi bộ ba một dòng.
Không giải thích gì thêm.

Văn bản:
{van_ban}
"""
        phan_hoi = ""
        if hasattr(self.llm, "sinh_van_ban"):
            phan_hoi = self.llm.sinh_van_ban(prompt, do_dai=512)
        elif callable(self.llm):
            phan_hoi = self.llm(prompt)

        ket_qua = []
        for dong in phan_hoi.strip().split("\n"):
            dong = dong.strip()
            if "|" in dong:
                parts = [p.strip() for p in dong.split("|")]
                if len(parts) == 3:
                    ket_qua.append((parts[0], parts[1], parts[2]))
        return ket_qua

    def trich_xuat(self, van_ban: str) -> List[Tuple[str, str, str]]:
        """Thực thi trích xuất v27.0.1."""
        if self.use_slm and not self.llm:
            return self._trich_xuat_bang_slm(van_ban)
        if self.llm:
            return self._trich_xuat_bang_llm(van_ban)
        return self._trich_xuat_bang_regex(van_ban)
