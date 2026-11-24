"""PhanTichCamXuc - Phân tích cảm xúc văn bản tiếng Việt."""

from typing import Any, Dict, List

import numpy as np

from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class PhanTichCamXuc:
    """
    Bộ phân tích cảm xúc văn bản tiếng Việt.

    Hỗ trợ 2 chế độ:
    - 'underthesea': Sử dụng thư viện underthesea (nếu có)
    - 'tu_huan': Tự huấn luyện trên dữ liệu người dùng

    Sử dụng:
        >>> # Chế độ underthesea (pre-trained)
        >>> ptx = PhanTichCamXuc(che_do="underthesea")
        >>> ptx.phan_tich("Sản phẩm rất tốt, tôi rất hài lòng")
        {'nhan': 'positive', 'xac_suat': {'positive': 0.85, 'negative': 0.15}}

        >>> # Chế độ tự huấn luyện
        >>> ptx = PhanTichCamXuc(che_do="tu_huan")
        >>> ptx.huan_luyen(van_ban_list, nhan_list)
        >>> ptx.phan_tich("Sản phẩm tuyệt vời")
    """

    TU_CAM_XUC_TICH_CUC = {
        "tốt",
        "tuyệt vời",
        "hay",
        "đẹp",
        "giỏi",
        "xuất sắc",
        "hoàn hảo",
        "tuyệt",
        "thích",
        "vui",
        "hài lòng",
        "ưng ý",
        "tuyệt hảo",
        "đỉnh",
        "chất lượng",
        "nhanh",
        "ổn định",
        "ổn_định",
        "nhiệt tình",
        "nhiệt_tình",
        "rất tốt",
        "rất hay",
        "rất đẹp",
        "rất hài lòng",
        "quá tuyệt",
        "tuyệt_vời",
        "hài_lòng",
        "chất_lượng",
    }

    TU_CAM_XUC_TIEU_CUC = {
        "tệ",
        "kém",
        "dở",
        "tồi",
        "xấu",
        "chậm",
        "lỗi",
        "tệ hại",
        "tệ_hại",
        "tệ_nặng",
        "chán",
        "buồn",
        "ghét",
        "tức",
        "giận",
        "thất vọng",
        "thất_vọng",
        "bực",
        "khó chịu",
        "khó_chịu",
        "rất tệ",
        "rất_tệ",
        "rất kém",
        "rất_kém",
        "quá dở",
        "quá_dở",
        "thật tệ",
        "thật_tệ",
        "thảm họa",
        "thảm_họa",
        "lừa đảo",
        "lừa_đảo",
        "không tốt",
        "không_tốt",
        "không hài lòng",
        "không_hài_lòng",
        "kém chất lượng",
        "kém_chất_lượng",
    }

    def __init__(self, che_do: str = "underthesea"):
        self.che_do = che_do
        self.logger = Logger("PhanTichCamXuc")
        self._xl = XuLyVanBan()
        self._mo_hinh: Any = None
        self._da_huan_luyen = False

        if che_do == "underthesea":
            try:
                from underthesea import sentiment

                self._underthesea_sentiment = sentiment
                self.logger.info("Sử dụng underthesea cho phân tích cảm xúc")
            except ImportError:
                self.logger.warning(
                    "underthesea chưa cài. Chuyển sang chế độ từ điển. "
                    "Cài đặt: pip install underthesea"
                )
                self.che_do = "tu_dien"

    def _diem_tu_dien(self, text: str) -> Dict[str, float]:
        """Tính điểm cảm xúc bằng từ điển tiếng Việt có xử lý cụm từ và phủ định cục bộ."""
        tu_list = self._xl.tach_tu(text)
        text_lower = text.lower()
        tich_cuc_chuan = {w.replace("_", " ") for w in self.TU_CAM_XUC_TICH_CUC}
        tieu_cuc_chuan = {w.replace("_", " ") for w in self.TU_CAM_XUC_TIEU_CUC}
        tu_phu_dinh = {"không", "chưa", "chẳng", "không hề"}

        diem_tich_cuc = 0
        diem_tieu_cuc = 0

        for idx, tu in enumerate(tu_list):
            tu_chuan = tu.lower().replace("_", " ")
            bi_phu_dinh = idx > 0 and tu_list[idx - 1].lower().replace("_", " ") in tu_phu_dinh
            if tu_chuan in tich_cuc_chuan:
                if bi_phu_dinh:
                    diem_tieu_cuc += 1
                else:
                    diem_tich_cuc += 1
            elif tu_chuan in tieu_cuc_chuan:
                if bi_phu_dinh:
                    diem_tich_cuc += 1
                else:
                    diem_tieu_cuc += 1

        for cum in tich_cuc_chuan:
            if " " in cum and cum in text_lower:
                diem_tich_cuc += 1
        for cum in tieu_cuc_chuan:
            if " " in cum and cum in text_lower:
                diem_tieu_cuc += 1

        tong = diem_tich_cuc + diem_tieu_cuc
        if tong == 0:
            return {"positive": 0.5, "negative": 0.5}

        return {
            "positive": diem_tich_cuc / tong,
            "negative": diem_tieu_cuc / tong,
        }

    def phan_tich(self, text: str) -> Dict:
        """
        Phân tích cảm xúc văn bản.

        Args:
            text: Văn bản cần phân tích

        Returns:
            Dict: {'nhan': 'positive'/'negative'/'neutral', 'xac_suat': {...}}
        """
        if self.che_do == "underthesea" and hasattr(self, "_underthesea_sentiment"):
            try:
                ket_qua = self._underthesea_sentiment(text)
                # underthesea trả về 'POS', 'NEG', 'NEU'
                nhan_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
                nhan = nhan_map.get(ket_qua, "neutral")
                return {"nhan": nhan, "xac_suat": {nhan: 1.0}, "nguon": "underthesea"}
            except Exception:
                pass

        if self._da_huan_luyen and self._mo_hinh is not None:
            try:
                vec = self._xl.ma_hoa_tfidf([text])
                nhan_idx = self._mo_hinh.du_doan(vec)[0]
                nhan = self._nhan_map.get(nhan_idx, "neutral")
                return {"nhan": nhan, "xac_suat": {nhan: 1.0}, "nguon": "tu_huan"}
            except Exception:
                pass

        # Fallback: từ điển
        xac_suat = self._diem_tu_dien(text)
        if xac_suat["positive"] > 0.6:
            nhan = "positive"
        elif xac_suat["negative"] > 0.6:
            nhan = "negative"
        else:
            nhan = "neutral"

        return {"nhan": nhan, "xac_suat": xac_suat, "nguon": "tu_dien"}

    def huan_luyen(
        self,
        cac_van_ban: List[str],
        cac_nhan: List[str],
        thuat_toan: str = "logistic",
    ) -> float:
        """
        Huấn luyện mô hình phân tích cảm xúc trên dữ liệu người dùng.

        Args:
            cac_van_ban: Danh sách văn bản
            cac_nhan: Danh nhãn ('positive', 'negative', 'neutral')
            thuat_toan: Thuật toán phân loại

        Returns:
            Độ chính xác trên tập huấn luyện
        """
        from vietnamese_ai.models.classifier import PhanLoai

        self.logger.info(f"Huấn luyện mô hình cảm xúc ({len(cac_van_ban)} mẫu)")

        # Mã hóa văn bản
        tfidf = self._xl.ma_hoa_tfidf(cac_van_ban)

        # Mã hóa nhãn
        nhan_duy_nhat = sorted(set(cac_nhan))
        self._nhan_map = {i: n for i, n in enumerate(nhan_duy_nhat)}
        nhan_map_nguoc = {n: i for i, n in enumerate(nhan_duy_nhat)}
        y = np.array([nhan_map_nguoc[n] for n in cac_nhan])

        # Huấn luyện
        self._mo_hinh = PhanLoai(thuat_toan=thuat_toan)
        self._mo_hinh.huan_luyen(tfidf, y)
        self._da_huan_luyen = True

        diem = self._mo_hinh.danh_gia(tfidf, y)
        self.logger.info(f"Huấn luyện hoàn tất: accuracy={diem:.4f}")
        return diem

    def phan_tich_nhieu(self, cac_van_ban: List[str]) -> List[Dict]:
        """Phân tích cảm xúc cho nhiều văn bản."""
        return [self.phan_tich(vb) for vb in cac_van_ban]

    def thong_ke(self, cac_van_ban: List[str]) -> Dict[str, int]:
        """Thống kê phân bố cảm xúc."""
        ket_qua = self.phan_tich_nhieu(cac_van_ban)
        dem = {"positive": 0, "negative": 0, "neutral": 0}
        for kq in ket_qua:
            dem[kq["nhan"]] = dem.get(kq["nhan"], 0) + 1
        return dem
