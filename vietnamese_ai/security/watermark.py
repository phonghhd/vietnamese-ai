import binascii
from typing import Optional


class TextWatermarker:
    """
    Công cụ nhúng Thủy ấn vô hình (Invisible Watermark) vào văn bản.
    Sử dụng kỹ thuật Zero-Width Character Steganography.
    - U+200B (Zero-width space) đại diện cho bit 0
    - U+200C (Zero-width non-joiner) đại diện cho bit 1

    Cho phép nhúng Payload (ví dụ: ID của Model, User) vào văn bản sinh ra bởi LLM
    để chống giả mạo và bảo vệ bản quyền.
    """

    CHAR_BIT_0 = "\u200b"
    CHAR_BIT_1 = "\u200c"

    @classmethod
    def nhung_thuy_an(cls, van_ban: str, payload: str) -> str:
        """
        Nhúng payload vào văn bản (ở cuối hoặc xen kẽ).
        """
        if not van_ban or not payload:
            return van_ban

        # Chuyển chuỗi payload thành nhị phân
        # Vi du: "AI" -> hex: 4149 -> binary: 0100000101001001
        binary_payload = bin(int(binascii.hexlify(payload.encode()), 16))[2:]

        # Tạo chuỗi ký tự ẩn
        thuy_an_an = ""
        for bit in binary_payload:
            if bit == "0":
                thuy_an_an += cls.CHAR_BIT_0
            else:
                thuy_an_an += cls.CHAR_BIT_1

        # Dán thủy ấn vào cuối văn bản
        return van_ban + thuy_an_an

    @classmethod
    def giai_ma_thuy_an(cls, van_ban_co_thuy_an: str) -> Optional[str]:
        """
        Trích xuất payload từ văn bản.
        """
        binary_str = ""

        # Lọc ra các ký tự zero-width
        for char in van_ban_co_thuy_an:
            if char == cls.CHAR_BIT_0:
                binary_str += "0"
            elif char == cls.CHAR_BIT_1:
                binary_str += "1"

        if not binary_str:
            return None

        # Chuyển đổi nhị phân về chuỗi text
        try:
            n = int(binary_str, 2)
            # convert int to bytes, sau do decode
            return binascii.unhexlify("%x" % n).decode("utf-8")
        except Exception:
            # Lỗi giải mã do chuỗi nhị phân không hợp lệ
            return None
