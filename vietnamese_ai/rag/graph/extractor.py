import re
from typing import Any, List, Tuple


class GraphExtractor:
    """
    Trích xuất đồ thị (Graph Extraction) từ văn bản.
    Nhiệm vụ: Chuyển văn bản phi cấu trúc thành các bộ ba (Subject, Relation, Object).
    """
    def __init__(self, llm: Any = None):
        """
        Args:
            llm: Đối tượng LLM để trích xuất ngữ nghĩa nâng cao.
                 Nếu None, sẽ dùng Regex cơ bản (không khuyến khích cho GraphRAG thật).
        """
        self.llm = llm

    def _trich_xuat_bang_regex(self, van_ban: str) -> List[Tuple[str, str, str]]:
        """Fallback: Trích xuất cơ bản bằng Regex (Rất thô sơ)."""
        # Ví dụ tìm mẫu "A là B", "A thuộc B"
        mau = re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(là|thuộc|nằm ở|được sáng lập bởi)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.IGNORECASE)
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
        for dong in phan_hoi.strip().split('\n'):
            dong = dong.strip()
            if '|' in dong:
                parts = [p.strip() for p in dong.split('|')]
                if len(parts) == 3:
                    ket_qua.append((parts[0], parts[1], parts[2]))
        return ket_qua

    def trich_xuat(self, van_ban: str) -> List[Tuple[str, str, str]]:
        """Thực thi trích xuất."""
        if self.llm:
            return self._trich_xuat_bang_llm(van_ban)
        return self._trich_xuat_bang_regex(van_ban)
