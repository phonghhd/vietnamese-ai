import re
from typing import Tuple


class AhoCorasickTrie:
    """Cấu trúc dữ liệu Trie quét chuỗi siêu tốc O(M+N) v27.0.1"""
    def __init__(self):
        self.root = {}

    def add_word(self, word: str):
        node = self.root
        for char in word.lower():
            node = node.setdefault(char, {})
        node['$'] = True

    def count_matches(self, text: str) -> int:
        count = 0
        text = text.lower()
        # Tìm kiếm O(M+N)
        for i in range(len(text)):
            node = self.root
            for j in range(i, len(text)):
                if text[j] not in node:
                    break
                node = node[text[j]]
                if '$' in node:
                    count += 1
                    # Bỏ qua phần còn lại của từ để tránh đếm đúp
                    break
        return count


class TuongLuaAI:
    """
    Tường lửa (Firewall) cho LLM, chuyên phát hiện và ngăn chặn các cuộc tấn công Prompt Injection,
    Jailbreak, hoặc các chỉ thị độc hại.
    """

    # Các mẫu regex thường thấy trong các cuộc tấn công Prompt Injection
    MAU_HIEM_DOC = [
        r"(?i)\bbỏ\s+qua\s+(tất\s+cả\s+)?(các\s+)?(hướng\s+dẫn|chỉ\s+thị|lệnh)\b",  # ignore previous instructions
        r"(?i)\bignore\s+(all\s+)?(previous\s+)?(instructions|commands)\b",
        r"(?i)\bbạn\s+(bây\s+giờ\s+)?(sẽ\s+)?đóng\s+vai\b",  # you will act as
        r"(?i)\byou\s+(will\s+)?(now\s+)?act\s+as\b",
        r"(?i)\bdan\b",  # DAN (Do Anything Now)
        r"(?i)\bdo\s+anything\s+now\b",
        r"(?i)\blàm\s+bất\s+cứ\s+điều\s+gì\b",
        r"(?i)\bquên\s+(tất\s+cả\s+)?(những\s+)?gì\s+tôi\s+đã\s+nói\b",
        r"(?i)\bsystem\s+prompt\b",
        r"(?i)\btừ\s+giờ\s+trở\s+đi\b",
    ]

    # Từ khóa có nguy cơ cao khi kết hợp với nhau
    TU_KHOA_RUI_RO = [
        "hack",
        "bypass",
        "exploit",
        "lỗ hổng",
        "mật khẩu",
        "password",
        "vô hiệu hóa",
        "hệ thống",
        "trái phép",
    ]

    def __init__(self, ngat_ket_noi_khi_phat_hien: bool = True, use_aho_corasick: bool = True):
        self.ngat_ket_noi = ngat_ket_noi_khi_phat_hien
        self.mau_regex = [re.compile(mau) for mau in self.MAU_HIEM_DOC]
        self.use_aho_corasick = use_aho_corasick
        if self.use_aho_corasick:
            self.trie = AhoCorasickTrie()
            for tu in self.TU_KHOA_RUI_RO:
                self.trie.add_word(tu)

    def kiem_tra_prompt(self, prompt: str) -> Tuple[bool, str]:
        """
        Kiểm tra xem prompt đầu vào có dấu hiệu Prompt Injection hay không.

        Args:
            prompt (str): Chuỗi văn bản người dùng nhập.

        Returns:
            Tuple[bool, str]: (True nếu an toàn, Lý do nếu không an toàn)
        """
        prompt_lower = prompt.lower()

        # 1. Kiểm tra bằng Regex Patterns
        for mau in self.mau_regex:
            if mau.search(prompt_lower):
                return False, "Phát hiện dấu hiệu tấn công Prompt Injection (Pattern matched)."

        # 2. Heuristic check: Mật độ từ khóa rủi ro
        if self.use_aho_corasick:
            so_tu_rui_ro = self.trie.count_matches(prompt_lower)
        else:
            so_tu_rui_ro = sum(1 for tu in self.TU_KHOA_RUI_RO if tu in prompt_lower)

        if so_tu_rui_ro >= 3:
            return False, "Phát hiện quá nhiều từ khóa rủi ro, có khả năng là truy vấn độc hại."

        # 3. Kiểm tra độ dài bất thường (Một số tấn công dùng prompt rất dài để tràn context)
        if len(prompt) > 8000:
            return False, "Prompt vượt quá độ dài an toàn cho phép (8000 ký tự)."

        return True, "An toàn."

    def loc_prompt(self, prompt: str) -> str:
        """
        Lọc bỏ các câu có chứa dấu hiệu Injection nếu có thể, hoặc trả về rỗng nếu quá nguy hiểm.
        (Chức năng nâng cao)
        """
        an_toan, ly_do = self.kiem_tra_prompt(prompt)
        if not an_toan and self.ngat_ket_noi:
            raise ValueError(f"Tường lửa AI đã chặn yêu cầu: {ly_do}")

        return prompt
