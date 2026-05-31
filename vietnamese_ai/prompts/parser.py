"""PhanTichDauRa - parse structured output từ LLM."""

import json
import re
from typing import Any, Dict, List, Optional


class PhanTichDauRa:
    """
    Phân tích đầu ra có cấu trúc từ LLM.

    Hỗ trợ parse:
    - JSON từ text
    - Markdown tables
    - Code blocks
    - Numbered/bulleted lists
    - Key-value pairs

    Sử dụng:
        >>> parser = PhanTichDauRa()
        >>> data = parser.phan_tich_json(llm_output)
        >>> items = parser.phan_tich_danh_sach(llm_output)
        >>> table = parser.phan_tich_bang(llm_output)
    """

    def phan_tich_json(self, van_ban: str) -> Optional[Any]:
        """
        Parse JSON từ text (tự động tìm JSON block).

        Args:
            van_ban: Text chứa JSON

        Returns:
            Parsed JSON object hoặc None
        """
        # Thử parse trực tiếp
        try:
            return json.loads(van_ban.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Tìm JSON trong code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", van_ban, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Tìm { } hoặc [ ]
        brace_match = re.search(r"\{.*\}", van_ban, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        bracket_match = re.search(r"\[.*\]", van_ban, re.DOTALL)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def phan_tich_bang(self, van_ban: str) -> List[Dict[str, str]]:
        """
        Parse markdown table.

        Returns:
            [{cot1: gia_tri1, cot2: gia_tri2}, ...]
        """
        lines = van_ban.strip().split("\n")
        table_lines = [
            line.strip()
            for line in lines
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]

        if len(table_lines) < 2:
            return []

        # Header
        header = [c.strip() for c in table_lines[0].split("|")[1:-1]]

        # Bỏ separator line
        rows = []
        for line in table_lines[2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) == len(header):
                row = dict(zip(header, cells))
                rows.append(row)

        return rows

    def phan_tich_danh_sach(self, van_ban: str) -> List[str]:
        """
        Parse numbered or bulleted list.

        Returns:
            Danh sách items
        """
        items = []
        mau = re.compile(r"^\s*(?:\d+[\.\)]\s*|[\-\*\•]\s*)(.+)$", re.MULTILINE)

        for match in mau.finditer(van_ban):
            item = match.group(1).strip()
            if item:
                items.append(item)

        return items

    def phan_tich_code_blocks(self, van_ban: str) -> List[Dict[str, str]]:
        """
        Parse code blocks.

        Returns:
            [{ngon_ngu, code}, ...]
        """
        blocks = []
        mau = re.compile(r"```(\w*)\n?(.*?)\n?```", re.DOTALL)

        for match in mau.finditer(van_ban):
            blocks.append(
                {
                    "ngon_ngu": match.group(1) or "text",
                    "code": match.group(2).strip(),
                }
            )

        return blocks

    def phan_tich_key_value(self, van_ban: str) -> Dict[str, str]:
        """
        Parse key-value pairs.

        Returns:
            {key: value}
        """
        ket_qua = {}
        mau = re.compile(r"^[\-\*]?\s*(.+?):\s*(.+)$", re.MULTILINE)

        for match in mau.finditer(van_ban):
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key and value:
                ket_qua[key] = value

        return ket_qua

    def phan_tich_so(self, van_ban: str) -> List[float]:
        """Trích xuất tất cả số từ text."""
        matches = re.findall(r"-?\d+\.?\d*", van_ban)
        return [float(m) for m in matches]

    def trich_cau_tra_loi(self, van_ban: str) -> str:
        """
        Trích xuất câu trả lời chính từ LLM output.

        Tìm phần text quan trọng nhất (bỏ qua reasoning).
        """
        # Tìm sau "Trả lời:" hoặc "Answer:"
        answer_match = re.search(
            r"(?:Trả lời|Answer|Kết quả):\s*(.+)",
            van_ban,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Tìm câu cuối cùng
        cau = re.split(r"[.!?\n]", van_ban.strip())
        cau = [c.strip() for c in cau if c.strip()]
        if cau:
            return cau[-1]

        return van_ban.strip()

    def __repr__(self) -> str:
        return "PhanTichDauRa()"
