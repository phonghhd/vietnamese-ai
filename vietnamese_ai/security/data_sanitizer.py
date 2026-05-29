import re


class DataSanitizer:
    """
    Tiện ích làm sạch dữ liệu (Data Loss Prevention - DLP).
    Quét và tự động che giấu (masking) các thông tin cá nhân (PII)
    trước khi đưa vào VectorDB hoặc trả về cho người dùng.
    """

    MAU_PII = {
        "so_dien_thoai": r"(?:\+84|0)[1-9]\d{8}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "cmnd_cccd": r"\b\d{9}(?:\d{3})?\b",
        "the_ngan_hang": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }

    THAY_THE = {
        "so_dien_thoai": "[SĐT ĐÃ ẨN]",
        "email": "[EMAIL ĐÃ ẨN]",
        "cmnd_cccd": "[CCCD ĐÃ ẨN]",
        "the_ngan_hang": "[THẺ NGÂN HÀNG ĐÃ ẨN]"
    }

    @classmethod
    def lam_sach(cls, van_ban: str) -> str:
        """
        Làm sạch văn bản bằng cách thay thế các mẫu PII bằng placeholder.
        """
        if not van_ban:
            return van_ban

        ket_qua = van_ban
        for loai, mau in cls.MAU_PII.items():
            chuoi_thay_the = cls.THAY_THE[loai]
            ket_qua = re.sub(mau, chuoi_thay_the, ket_qua)

        return ket_qua
