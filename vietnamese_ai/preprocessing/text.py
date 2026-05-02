"""Xử lý văn bản tiếng Việt với underthesea integration."""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

try:
    from underthesea import pos_tag, sentiment, word_tokenize
    _CO_UNDERTHESEA = True
except ImportError:
    _CO_UNDERTHESEA = False


class XuLyVanBan:
    """
    Bộ xử lý văn bản tiếng Việt toàn diện.

    Tính năng:
    - Chuẩn hóa Unicode tiếng Việt
    - Tách từ tiếng Việt chuẩn (underthesea)
    - Gán nhãn từ loại (POS tagging)
    - Phân tích cảm xúc (sentiment)
    - Loại bỏ từ dừng (stopwords)
    - Mã hóa TF-IDF, Bag-of-Words
    - Trích xuất từ khóa

    Sử dụng:
        >>> xl = XuLyVanBan()
        >>> xl.tach_tu("Trí tuệ nhân tạo rất hay")
        ['trí_tuệ_nhân_tạo', 'rất', 'hay']
        >>> xl.phan_tich_cam_xuc("Sản phẩm rất tốt")
        'positive'
    """

    TU_DUNG_TIENG_VIET = {
        "và", "hoặc", "với", "trong", "ngoài", "như", "đã", "còn", "lại",
        "thì", "mà", "của", "cho", "tới", "từ", "theo", "đến", "bởi", "tại",
        "vào", "ra", "lên", "xuống", "về", "đi", "đứng", "nằm", "ngồi",
        "là", "được", "có", "không", "này", "kia", "đó", "ấy", "nọ",
        "rất", "lắm", "quá", "thương", "hay", "đều", "cũng", "đã",
        "sẽ", "đang", "vừa", "mới", "rồi", "xong", "chưa", "chỉ",
        "mỗi", "các", "những", "một", "hai", "ba", "nhiều", "ít",
        "toàn", "hết", "cả", "sự", "việc", "người", "con", "cái",
        "nếu", "thì", "hay", "hoặc", "nhưng", "mà", "vì", "do",
        "nên", "để", "cho", "với", "cùng", "theo", "sau", "trước",
        "ủa", "ơi", "ạ", "à", "vậy", "nữa", "thôi", "nhé", "nè",
    }

    @staticmethod
    def _loai_dau(text: str) -> str:
        """Loại bỏ dấu tiếng Việt, chuyển về ASCII."""
        text = unicodedata.normalize("NFD", text)
        text = text.replace("đ", "d").replace("Đ", "D")
        return "".join(c for c in text if unicodedata.category(c) != "Mn")

    def __init__(self, tu_dung: Optional[set] = None, su_dung_underthesea: bool = True):
        self.tu_dung = tu_dung or self.TU_DUNG_TIENG_VIET
        self._tu_dien: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._su_dung_underthesea = su_dung_underthesea and _CO_UNDERTHESEA

    @property
    def co_underthesea(self) -> bool:
        """Kiểm tra underthesea có sẵn không."""
        return _CO_UNDERTHESEA

    def chuan_hoa(self, text: str) -> str:
        """
        Chuẩn hóa văn bản tiếng Việt.

        - Chuyển về chữ thường
        - Chuẩn hóa Unicode NFC
        - Loại bỏ ký tự đặc biệt
        - Loại bỏ khoảng trắng thừa
        """
        text = text.lower().strip()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(
            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]",
            " ", text
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tach_tu(self, text: str, format_output: str = "list") -> List[str]:
        """
        Tách từ tiếng Việt.

        Sử dụng underthesea nếu có, nếu không dùng split cơ bản.

        Args:
            text: Văn bản đầu vào
            format_output: 'list' hoặc 'text' (nối bằng _)

        Returns:
            Danh sách từ đã tách
        """
        text = self.chuan_hoa(text)

        if self._su_dung_underthesea:
            try:
                tu_list = word_tokenize(text, format="list")
                return tu_list
            except Exception:
                pass

        return text.split()

    def tach_tu_chuoi(self, text: str) -> str:
        """
        Tách từ và trả về chuỗi (từ nối bằng khoảng trắng).

        Hữu ích cho pipeline preprocessing.
        """
        return " ".join(self.tach_tu(text))

    def gan_nhan_tu_loai(self, text: str) -> List[tuple]:
        """
        Gán nhãn từ loại (POS tagging).

        Returns:
            Danh sách (từ, từ_loại) - yêu cầu underthesea

        Raises:
            ImportError: Nếu underthesea chưa cài đặt
        """
        if not _CO_UNDERTHESEA:
            raise ImportError(
                "Cần cài đặt underthesea: pip install underthesea"
            )
        text = self.chuan_hoa(text)
        return pos_tag(text)

    def phan_tich_cam_xuc(self, text: str) -> str:
        """
        Phân tích cảm xúc văn bản.

        Returns:
            'positive', 'negative', hoặc 'neutral'
        """
        if not _CO_UNDERTHESEA:
            raise ImportError(
                "Cần cài đặt underthesea: pip install underthesea"
            )
        return sentiment(text)

    def loai_bo_tu_dung(self, text: str) -> str:
        """Loại bỏ từ dừng khỏi văn bản."""
        cac_tu = self.tach_tu(text)
        tu_loc = [t for t in cac_tu if t not in self.tu_dung]
        return " ".join(tu_loc)

    def xu_ly_day_du(self, text: str) -> str:
        """Xử lý đầy đủ: chuẩn hóa + tách từ + loại bỏ từ dừng."""
        return self.loai_bo_tu_dung(self.chuan_hoa(text))

    def tao_tu_dien(self, cac_van_ban: List[str]) -> Dict[str, int]:
        """Tạo từ điển (vocabulary) từ danh sách văn bản."""
        bo_tu = set()
        for vb in cac_van_ban:
            bo_tu.update(self.tach_tu(vb))
        self._tu_dien = {tu: idx for idx, tu in enumerate(sorted(bo_tu))}
        return self._tu_dien

    def ma_hoa_bo_dem(self, text: str) -> np.ndarray:
        """Mã hóa văn bản bằng bag-of-words."""
        if not self._tu_dien:
            raise RuntimeError("Cần gọi tao_tu_dien() trước.")
        vector = np.zeros(len(self._tu_dien))
        for tu in self.tach_tu(text):
            if tu in self._tu_dien:
                vector[self._tu_dien[tu]] += 1
        return vector

    def ma_hoa_tfidf(self, cac_van_ban: List[str]) -> np.ndarray:
        """
        Mã hóa danh sách văn bản thành ma trận TF-IDF.

        Returns:
            Ma trận (so_van_ban x so_tu) chứa giá trị TF-IDF
        """
        self.tao_tu_dien(cac_van_ban)
        so_vb = len(cac_van_ban)
        so_tu = len(self._tu_dien)

        # TF
        tf = np.zeros((so_vb, so_tu))
        for i, vb in enumerate(cac_van_ban):
            for tu in self.tach_tu(vb):
                if tu in self._tu_dien:
                    tf[i, self._tu_dien[tu]] += 1
            tong = tf[i].sum()
            if tong > 0:
                tf[i] /= tong

        # IDF
        df = np.sum(tf > 0, axis=0)
        self._idf = np.log((so_vb + 1) / (df + 1)) + 1

        return tf * self._idf

    def trich_xuat_tu_khoa(self, text: str, top_n: int = 5) -> List[str]:
        """Trích xuất từ khóa quan trọng từ văn bản."""
        cac_tu = self.tach_tu(self.loai_bo_tu_dung(text))
        dem = Counter(cac_tu)
        return [tu for tu, _ in dem.most_common(top_n)]

    def __repr__(self) -> str:
        engine = "underthesea" if self._su_dung_underthesea else "basic"
        return f"XuLyVanBan(engine='{engine}')"
