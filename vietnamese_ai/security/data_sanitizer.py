from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class DataSanitizer:
    """
    Tiện ích làm sạch dữ liệu & Chống thất thoát dữ liệu (Data Loss Prevention - DLP).
    Quét và tự động che giấu (masking) các thông tin cá nhân (PII) cũng như
    khóa bí mật hệ thống (API Keys, JWT, Private Keys, chuỗi kết nối CSDL)
    trước khi đưa vào VectorDB hoặc trả về cho người dùng.
    """

    MAU_PII = {
        "so_dien_thoai": r"(?<!\d)(?:\+84|0)(?:(?:3|5|7|8|9)(?:[\s.\-]?\d){8}|2\d(?:[\s.\-]?\d){8,9})(?!\d)",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "cmnd_cccd": r"\b0(?:0[1-9]|[1-8]\d|9[0-6])\d{9}\b",
        "the_ngan_hang": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }

    _RE_SO_9_12 = re.compile(r"\b(\d{9}|\d{12})\b")
    _RE_NGU_CANH_CCCD = re.compile(
        r"(?:cccd|cmnd|căn\s*cước|can\s*cuoc|chứng\s*minh|chung\s*minh|định\s*danh|dinh\s*danh|số\s*giấy\s*tờ)\s*(?:công\s*dân|nhân\s*dân|cá\s*nhân)?\s*(?:số|so|no\.?)?\s*[:#\-]?\s*$",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_DON_VI_TIEN_TE = re.compile(
        r"^\s*(?:vnđ|vnd|đồng|dong|đ\b|usd|eur|triệu|trieu|tỷ|ty|ngàn|nghìn|kwh|kg|m2|%)",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_MA_TINH_CCCD_12 = re.compile(r"^0(?:0[1-9]|[1-8]\d|9[0-6])\d{9}$")

    MAU_BI_MAT = {
        "api_key": r"\b(?:vai_[a-fA-F0-9]{16,64}|sk-[a-zA-Z0-9_-]{16,128}|hf_[a-zA-Z0-9]{16,64}|AIza[0-9A-Za-z_-]{20,60})\b",
        "jwt_token": r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b",
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "chuoi_ket_noi_db": r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s\"']+",
    }

    THAY_THE = {
        "so_dien_thoai": "[SĐT ĐÃ ẨN]",
        "email": "[EMAIL ĐÃ ẨN]",
        "cmnd_cccd": "[CCCD ĐÃ ẨN]",
        "the_ngan_hang": "[THẺ NGÂN HÀNG ĐÃ ẨN]",
        "api_key": "[API KEY ĐÃ ẨN]",
        "jwt_token": "[JWT TOKEN ĐÃ ẨN]",
        "private_key": "[PRIVATE KEY ĐÃ ẨN]",
        "chuoi_ket_noi_db": "[DB URI ĐÃ ẨN]",
    }

    @classmethod
    def _la_cccd_hop_le(cls, van_ban: str, start: int, end: int, so: str) -> bool:
        """Phân biệt CCCD/CMND thực tế với số tiền tệ (`100000000 VNĐ`) hoặc mã số kỹ thuật."""
        hau_to = van_ban[end : min(len(van_ban), end + 16)]
        if cls._RE_DON_VI_TIEN_TE.match(hau_to):
            return False
        tien_to = van_ban[max(0, start - 36) : start]
        if cls._RE_NGU_CANH_CCCD.search(tien_to):
            return True
        if len(so) == 12 and cls._RE_MA_TINH_CCCD_12.match(so):
            return True
        return False

    @classmethod
    def _tim_cccd_ngu_canh(cls, van_ban: str) -> List[str]:
        """Tìm toàn bộ chuỗi CCCD/CMND hợp lệ có xét ngữ cảnh và loại trừ số tiền tệ."""
        ket_qua: List[str] = []
        for match in cls._RE_SO_9_12.finditer(van_ban):
            so = match.group(1)
            if cls._la_cccd_hop_le(van_ban, match.start(1), match.end(1), so):
                ket_qua.append(so)
        return ket_qua

    @classmethod
    def _an_cccd_ngu_canh(cls, van_ban: str) -> str:
        """Thay thế có chọn lọc các số CCCD/CMND hợp lệ mà không làm hỏng số tiền VNĐ/đồng."""
        def _thay(match: re.Match[str]) -> str:
            so = match.group(1)
            if cls._la_cccd_hop_le(van_ban, match.start(1), match.end(1), so):
                return cls.THAY_THE["cmnd_cccd"]
            return so

        return cls._RE_SO_9_12.sub(_thay, van_ban)

    @classmethod
    def quet_ro_ri(cls, van_ban: str) -> Dict[str, List[str]]:
        """
        Quét và trả về danh sách các thông tin nhạy cảm (PII & Khóa bí mật) phát hiện được.
        """
        if not van_ban:
            return {}
        phat_hien: Dict[str, List[str]] = {}
        for loai, mau in {**cls.MAU_PII, **cls.MAU_BI_MAT}.items():
            if loai == "cmnd_cccd":
                ket_qua = cls._tim_cccd_ngu_canh(van_ban)
            else:
                ket_qua = re.findall(mau, van_ban)
            if ket_qua:
                phat_hien[loai] = ket_qua
        return phat_hien

    @classmethod
    def lam_sach(cls, van_ban: str, loc_bi_mat: bool = True) -> str:
        """
        Làm sạch văn bản bằng cách thay thế các mẫu PII và khóa bí mật bằng placeholder an toàn.
        """
        if not van_ban:
            return van_ban

        ket_qua = van_ban
        cac_mau = dict(cls.MAU_BI_MAT) if loc_bi_mat else {}
        cac_mau.update(cls.MAU_PII)

        for loai, mau in cac_mau.items():
            if loai == "cmnd_cccd":
                ket_qua = cls._an_cccd_ngu_canh(ket_qua)
            else:
                chuoi_thay_the = cls.THAY_THE[loai]
                ket_qua = re.sub(mau, chuoi_thay_the, ket_qua)

        return ket_qua

    @classmethod
    def quet_dau_doc_du_lieu(
        cls,
        du_lieu: Any,
        an_pii: bool = True,
        loai_bo_mau_doc: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Quét tập dữ liệu huấn luyện (Fine-Tuning) để phát hiện và loại bỏ các mẫu bị đầu độc
        (Data Poisoning / Backdoor Prompt Injection) đồng thời ẩn thông tin cá nhân & khóa bí mật.
        """
        from vietnamese_ai.security.llm_firewall import TuongLuaAI

        bao_cao = {
            "tong_mau_kiem_tra": 0,
            "so_mau_dau_doc_bi_loai": 0,
            "so_mau_da_an_pii": 0,
            "an_toan": True,
        }
        if du_lieu is None or isinstance(du_lieu, tuple):
            return du_lieu, bao_cao

        la_dataset_obj = hasattr(du_lieu, "train") and isinstance(du_lieu.train, list)
        items = du_lieu.train if la_dataset_obj else (du_lieu if isinstance(du_lieu, list) else None)
        if items is None or not items or not isinstance(items[0], dict):
            return du_lieu, bao_cao

        tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=False)
        truong_van_ban = (
            "instruction",
            "input",
            "output",
            "text",
            "van_ban",
            "prompt",
            "response",
            "chosen",
            "rejected",
            "question",
            "answer",
            "context",
        )

        ket_qua_sach: List[Dict[str, Any]] = []
        for mau in items:
            if not isinstance(mau, dict):
                ket_qua_sach.append(mau)
                continue
            bao_cao["tong_mau_kiem_tra"] += 1
            chuoi_gop = " ".join(str(mau[k]) for k in truong_van_ban if k in mau and isinstance(mau[k], str))
            hop_le, _ = tuong_lua.kiem_tra_prompt(chuoi_gop)
            if not hop_le:
                bao_cao["so_mau_dau_doc_bi_loai"] += 1
                bao_cao["an_toan"] = False
                if loai_bo_mau_doc:
                    continue

            mau_moi = dict(mau)
            da_sua_pii = False
            if an_pii:
                for k in truong_van_ban:
                    if k in mau_moi and isinstance(mau_moi[k], str):
                        da_lam_sach = cls.lam_sach(mau_moi[k], loc_bi_mat=True)
                        if da_lam_sach != mau_moi[k]:
                            mau_moi[k] = da_lam_sach
                            da_sua_pii = True
            if da_sua_pii:
                bao_cao["so_mau_da_an_pii"] += 1
            ket_qua_sach.append(mau_moi)

        if not ket_qua_sach and items:
            # Nếu toàn bộ dữ liệu đều bị đầu độc, ném ngoại lệ bảo mật để ngăn huấn luyện mô hình có backdoor
            raise ValueError(
                "Bảo mật: Toàn bộ tập dữ liệu huấn luyện đều chứa chỉ thị độc hại (Data Poisoning / Prompt Injection)!"
            )

        if la_dataset_obj:
            du_lieu.train = ket_qua_sach
            return du_lieu, bao_cao
        return ket_qua_sach, bao_cao
