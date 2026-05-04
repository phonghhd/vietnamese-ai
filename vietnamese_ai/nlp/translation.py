"""DichThuat - Translation cho tiếng Việt."""

from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class DichThuat:
    """
    Dịch thuật cho tiếng Việt.

    Hỗ trợ:
    - Dictionary-based translation
    - LLM-based translation (nếu có generator)
    - Phrase-level và word-level translation
    - Batch translation

    Sử dụng:
        >>> dich = DichThuat()
        >>> dich.them_tu_dien("en_vi", {"hello": "xin chào", "world": "thế giới"})
        >>> ket_qua = dich.dich("hello world", nguon="en", dich="vi")
    """

    TU_DIEN_EN_VI = {
        "hello": "xin chào",
        "goodbye": "tạm biệt",
        "thank you": "cảm ơn",
        "good morning": "chào buổi sáng",
        "good night": "chúc ngủ ngon",
        "how are you": "bạn khỏe không",
        "yes": "vâng",
        "no": "không",
        "please": "xin vui lòng",
        "sorry": "xin lỗi",
        "computer": "máy tính",
        "artificial intelligence": "trí tuệ nhân tạo",
        "machine learning": "học máy",
        "deep learning": "học sâu",
        "data": "dữ liệu",
        "model": "mô hình",
        "training": "huấn luyện",
        "prediction": "dự đoán",
        "algorithm": "thuật toán",
        "framework": "khung làm việc",
        "library": "thư viện",
        "function": "hàm",
        "class": "lớp",
        "method": "phương thức",
        "variable": "biến",
        "array": "mảng",
        "string": "chuỗi",
        "number": "số",
        "file": "tập tin",
        "error": "lỗi",
        "bug": "lỗi phần mềm",
        "feature": "tính năng",
        "update": "cập nhật",
        "version": "phiên bản",
        "database": "cơ sở dữ liệu",
        "server": "máy chủ",
        "client": "khách hàng",
        "user": "người dùng",
        "system": "hệ thống",
        "network": "mạng",
        "security": "bảo mật",
        "performance": "hiệu suất",
        "memory": "bộ nhớ",
        "disk": "ổ đĩa",
        "cpu": "bộ xử lý",
        "gpu": "bộ xử lý đồ họa",
    }

    TU_DIEN_VI_EN = {v: k for k, v in TU_DIEN_EN_VI.items()}

    def __init__(
        self,
        ham_sinh: Optional[Callable[[str], str]] = None,
        che_do: str = "dictionary",
    ):
        if che_do not in ("dictionary", "llm", "hybrid"):
            raise ValueError("che_do phải là: dictionary, llm, hybrid")

        self.che_do = che_do
        self.ham_sinh = ham_sinh
        self.logger = Logger("DichThuat")

        self._tu_dien: Dict[str, Dict[str, str]] = {
            "en_vi": dict(self.TU_DIEN_EN_VI),
            "vi_en": dict(self.TU_DIEN_VI_EN),
        }

    def dich(
        self,
        van_ban: str,
        nguon: str = "en",
        dich: str = "vi",
    ) -> Dict[str, Any]:
        """
        Dịch văn bản.

        Args:
            van_ban: Văn bản cần dịch
            nguon: Ngôn ngữ nguồn (en, vi)
            dich: Ngôn ngữ đích (en, vi)

        Returns:
            {goc, dich, nguon, dich_lang, che_do}
        """
        key = f"{nguon}_{dich}"

        if self.che_do == "llm" and self.ham_sinh:
            ket_qua = self._dich_llm(van_ban, nguon, dich)
        elif self.che_do == "hybrid" and self.ham_sinh:
            ket_qua = self._dich_hybrid(van_ban, key)
        else:
            ket_qua = self._dich_tu_dien(van_ban, key)

        return {
            "goc": van_ban,
            "dich": ket_qua,
            "nguon": nguon,
            "dich_lang": dich,
            "che_do": self.che_do,
        }

    def _dich_tu_dien(self, van_ban: str, key: str) -> str:
        """Dịch bằng từ điển."""
        tu_dien = self._tu_dien.get(key, {})
        if not tu_dien:
            return van_ban

        ket_qua = van_ban.lower()

        # Ưu tiên cụm từ dài hơn trước
        sorted_phrases = sorted(tu_dien.keys(), key=len, reverse=True)

        da_dich = set()
        for phrase in sorted_phrases:
            if phrase in ket_qua and phrase not in da_dich:
                # Preserve case
                vi_tri = ket_qua.find(phrase)
                while vi_tri != -1:
                    original = van_ban[vi_tri:vi_tri + len(phrase)]
                    translated = tu_dien[phrase]

                    if original[0].isupper():
                        translated = translated[0].upper() + translated[1:]

                    van_ban = van_ban[:vi_tri] + translated + van_ban[vi_tri + len(phrase):]
                    ket_qua = van_ban.lower()

                    da_dich.add(phrase)
                    vi_tri = ket_qua.find(phrase, vi_tri + len(translated))

        return van_ban

    def _dich_llm(self, van_ban: str, nguon: str, dich: str) -> str:
        """Dịch bằng LLM."""
        ten_nguon = {"en": "English", "vi": "Vietnamese"}.get(nguon, nguon)
        ten_dich = {"en": "English", "vi": "Vietnamese"}.get(dich, dich)

        prompt = (
            f"Translate the following text from {ten_nguon} to {ten_dich}. "
            f"Only output the translation, no explanation.\n\n"
            f"Text: {van_ban}\n\nTranslation:"
        )

        try:
            return self.ham_sinh(prompt)
        except Exception as e:
            self.logger.info(f"Lỗi LLM, fallback dictionary: {e}")
            key = f"{nguon}_{dich}"
            return self._dich_tu_dien(van_ban, key)

    def _dich_hybrid(self, van_ban: str, key: str) -> str:
        """Dịch hybrid (dictionary + LLM cho phần chưa dịch)."""
        tu_dien = self._tu_dien.get(key, {})

        # Dịch những gì có trong từ điển
        da_dich = self._dich_tu_dien(van_ban, key)

        # Nếu có nhiều từ chưa dịch, dùng LLM
        tu_chua_dich = 0
        for tu in van_ban.lower().split():
            if tu not in tu_dien:
                tu_chua_dich += 1

        if tu_chua_dich > len(van_ban.split()) * 0.3 and self.ham_sinh:
            return self._dich_llm(
                van_ban,
                key.split("_")[0],
                key.split("_")[1],
            )

        return da_dich

    def them_tu_dien(
        self,
        key: str,
        tu_dien: Dict[str, str],
    ) -> None:
        """Thêm từ điển dịch."""
        if key not in self._tu_dien:
            self._tu_dien[key] = {}
        self._tu_dien[key].update(tu_dien)

    def dich_batch(
        self,
        van_ban_list: List[str],
        nguon: str = "en",
        dich: str = "vi",
    ) -> List[Dict[str, Any]]:
        """Dịch nhiều văn bản."""
        return [self.dich(vb, nguon, dich) for vb in van_ban_list]

    def lay_tu_dien(self, key: str) -> Dict[str, str]:
        """Lấy từ điển theo key."""
        return self._tu_dien.get(key, {}).copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "che_do": self.che_do,
            "co_llm": self.ham_sinh is not None,
            "so_tu_dien": {k: len(v) for k, v in self._tu_dien.items()},
        }

    def __repr__(self) -> str:
        return f"DichThuat(che_do='{self.che_do}')"
