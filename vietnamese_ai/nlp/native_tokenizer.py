"""Bộ Tách Từ & Chuẩn Hóa Tiếng Việt Siêu Tốc (Fast Vietnamese Native Tokenizer) - Vietnamese AI Framework.

Tối ưu hóa chuyên sâu cho ngữ cảnh tiếng Việt (từ ghép, văn bản hành chính/pháp lý, dấu thanh mới/cũ).
Cung cấp khả năng tăng tốc native C/Rust, tùy chọn tích hợp `underthesea` và fallback Python/NumPy siêu nhẹ.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

try:
    import vietnamese_ai_rust as _rust_core  # type: ignore

    _CO_RUST = True
except ImportError:
    try:
        from vietnamese_ai import vietnamese_ai_rust as _rust_core  # type: ignore

        _CO_RUST = True
    except ImportError:
        _rust_core = None
        _CO_RUST = False


def co_ho_tro_rust() -> bool:
    """Kiểm tra xem hệ thống đã nạp và sẵn sàng mô-đun Rust Native Acceleration (Rayon multi-threading) hay chưa."""
    return _CO_RUST and _rust_core is not None


def _giu_kieu_chu(tu_goc: str, tu_thay_the: str) -> str:
    """Bảo toàn kiểu viết hoa/thường (HOA, Hoa Đầu, thường) khi chuẩn hóa dấu thanh."""
    if not tu_goc:
        return tu_thay_the
    if tu_goc.isupper():
        return tu_thay_the.upper()
    if tu_goc[0].isupper() and tu_goc[1:].islower():
        return tu_thay_the[0].upper() + tu_thay_the[1:]
    if tu_goc[0].isupper():
        return tu_thay_the[0].upper() + tu_thay_the[1:]
    return tu_thay_the


class BoTachTuTiengVietSieuToc:
    """
    Bộ Tách Từ & Chuẩn Hóa Tiếng Việt Siêu Tốc (Fast Vietnamese Native Tokenizer):
    - Tách từ ghép tiếng Việt (Compound Word Segmentation) theo thuật toán Maximum Matching phân tầng độ dài.
    - Chuẩn hóa dấu tiếng Việt kiểu mới (hòa, thúy, ủy, khỏe, tùy) bảo toàn chữ hoa/thường.
    - Chuẩn hóa Unicode NFC, loại bỏ ký tự rác / zero-width spaces.
    - Hỗ trợ chuyển đổi văn bản có dấu sang không dấu (slugify / unaccented).
    - Hỗ trợ tùy chọn backend `underthesea` (nếu cài gói `[vietnamese-nlp]`), mặc định chạy thuần không phụ thuộc nặng.
    """

    BANG_CHUAN_HOA_DAU: Dict[str, str] = {
        "hoà": "hòa",
        "hoá": "hóa",
        "hoả": "hỏa",
        "hoã": "hõa",
        "hoạ": "họa",
        "toà": "tòa",
        "toá": "tóa",
        "toả": "tỏa",
        "toã": "tõa",
        "toạ": "tọa",
        "xoà": "xòa",
        "xoá": "xóa",
        "xoả": "xỏa",
        "xoã": "xõa",
        "xoạ": "xọa",
        "loà": "lòa",
        "loá": "lóa",
        "loả": "lỏa",
        "loã": "lõa",
        "loạ": "lọa",
        "khoà": "khòa",
        "khoá": "khóa",
        "khoả": "khỏa",
        "khoã": "khõa",
        "khoạ": "khọa",
        "khoè": "khòe",
        "khoé": "khóe",
        "khoẻ": "khỏe",
        "khoẽ": "khõe",
        "khoẹ": "khọe",
        "hoè": "hòe",
        "hoé": "hóe",
        "hoẻ": "hỏe",
        "hoẽ": "hõe",
        "hoẹ": "họe",
        "xoè": "xòe",
        "xoé": "xóe",
        "xoẻ": "xỏe",
        "xoẽ": "xõe",
        "xoẹ": "xọe",
        "thuỳ": "thùy",
        "thuý": "thúy",
        "thuỷ": "thủy",
        "thuỹ": "thũy",
        "thuỵ": "thụy",
        "tuỳ": "tùy",
        "tuý": "túy",
        "tuỷ": "tủy",
        "tuỹ": "tũy",
        "tuỵ": "tụy",
        "huỳ": "hùy",
        "huý": "húy",
        "huỷ": "hủy",
        "huỹ": "hũy",
        "huỵ": "hụy",
        "nguỳ": "ngùy",
        "nguý": "ngúy",
        "nguỷ": "ngủy",
        "nguỹ": "ngũy",
        "nguỵ": "ngụy",
        "uỳ": "ùy",
        "uý": "úy",
        "uỷ": "ủy",
        "uỹ": "ũy",
        "uỵ": "ụy",
    }

    TU_DIEN_TU_GHEP_MAC_DINH: Set[str] = {
        "ủy ban nhân dân",
        "hội đồng nhân dân",
        "bộ tư pháp",
        "bộ tài chính",
        "bộ công an",
        "bộ giáo dục và đào tạo",
        "bộ y tế",
        "bộ thông tin và truyền thông",
        "bộ kế hoạch và đầu tư",
        "sở kế hoạch và đầu tư",
        "thủ tướng chính phủ",
        "quốc hội",
        "viện kiểm sát",
        "tòa án nhân dân",
        "cơ quan nhà nước",
        "thẩm quyền",
        "căn cứ",
        "quy phạm pháp luật",
        "văn bản pháp luật",
        "văn bản hành chính",
        "bộ luật",
        "luật pháp",
        "nghị định",
        "thông tư",
        "quyết định",
        "chỉ thị",
        "nghị quyết",
        "công văn",
        "tờ trình",
        "biên bản",
        "quy chế",
        "phụ lục",
        "điều khoản",
        "quy định",
        "hiệu lực",
        "thi hành",
        "quyền lợi",
        "nghĩa vụ",
        "trách nhiệm",
        "xử phạt",
        "vi phạm hành chính",
        "hợp đồng",
        "hợp đồng lao động",
        "giao kết",
        "bồi thường",
        "tranh chấp",
        "chứng thực",
        "công chứng",
        "căn cước công dân",
        "chứng minh nhân dân",
        "hộ khẩu",
        "giấy khai sinh",
        "mã số thuế",
        "chữ ký số",
        "doanh nghiệp",
        "công ty",
        "công ty cổ phần",
        "kinh doanh",
        "đầu tư",
        "tài chính",
        "ngân hàng",
        "chứng khoán",
        "bảo hiểm",
        "phá sản",
        "giải thể",
        "trí tuệ nhân tạo",
        "học máy",
        "học sâu",
        "mạng nơ-ron",
        "xử lý",
        "xử lý ngôn ngữ tự nhiên",
        "ngôn ngữ",
        "tự nhiên",
        "chuyển đổi số",
        "dữ liệu lớn",
        "cơ sở dữ liệu",
        "dữ liệu cá nhân",
        "điện toán đám mây",
        "an toàn thông tin",
        "chuỗi khối",
        "tác tử",
        "mô hình ngôn ngữ",
        "mô hình ngôn ngữ lớn",
        "đồ thị trạng thái",
        "suy luận",
        "truy hồi",
        "học liên kết",
        "lượng tử hóa",
        "đóng gói",
        "bảo mật dữ liệu",
        "phần mềm",
        "hệ điều hành",
        "bán dẫn",
        "tách từ",
        "gán nhãn từ loại",
        "nhận diện thực thể",
        "kinh tế",
        "xã hội",
        "chính trị",
        "văn hóa",
        "giáo dục",
        "y tế",
        "giao thông",
        "môi trường",
        "phát triển",
        "bền vững",
        "chính phủ",
        "quốc gia",
        "quốc tế",
        "thành phố",
        "tỉnh thành",
        "quận huyện",
        "phường xã",
        "thị trấn",
        "nông thôn",
        "việt nam",
        "hà nội",
        "hồ chí minh",
        "đà nẵng",
        "hải phòng",
        "cần thơ",
        "khách hàng",
        "nhân viên",
        "sinh viên",
        "giảng viên",
        "bệnh viện",
        "trường đại học",
        "dịch vụ",
        "sản phẩm",
        "chất lượng",
        "uy tín",
        "thời gian",
        "kết quả",
        "báo cáo",
        "nghiên cứu",
        "thực hiện",
        "hoàn thành",
        "ban hành",
        "áp dụng",
        "phân tích",
        "triển khai",
        "xây dựng",
        "cung cấp",
        "hỗ trợ",
        "bảo vệ",
        "kiểm tra",
        "xác thực",
        "đánh giá",
        "sử dụng",
        "tích hợp",
        "tối ưu",
        "huấn luyện",
        "dự đoán",
        "thông qua",
        "quan trọng",
        "chính xác",
        "an toàn",
        "bảo mật",
        "hiệu quả",
    }

    SPECIAL_TOKENS: Dict[str, int] = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }

    def __init__(
        self,
        tu_dien_bo_sung: Optional[Set[str]] = None,
        su_dung_underthesea: bool = False,
    ) -> None:
        self._tu_dien: Set[str] = set(self.TU_DIEN_TU_GHEP_MAC_DINH)
        if tu_dien_bo_sung:
            for t in tu_dien_bo_sung:
                tu_sach = t.strip().lower()
                if tu_sach:
                    self._tu_dien.add(tu_sach)

        self._tu_dien_theo_do_dai: Dict[int, Set[str]] = {}
        self._trie_tu_ghep: Dict[str, Any] = {}
        self._max_word_len = 1
        for w in self._tu_dien:
            do_dai = len(w.split())
            self._tu_dien_theo_do_dai.setdefault(do_dai, set()).add(w)
            if do_dai > self._max_word_len:
                self._max_word_len = do_dai
            self._chen_trie(w)

        self._rust_tokenizer: Any = None
        if _rust_core is not None and hasattr(_rust_core, "RustViTokenizer"):
            try:
                self._rust_tokenizer = _rust_core.RustViTokenizer(list(self._tu_dien))
            except Exception:
                self._rust_tokenizer = None

        self._underthesea_tokenize = None
        if su_dung_underthesea:
            try:
                from underthesea import word_tokenize

                self._underthesea_tokenize = word_tokenize
            except ImportError:
                self._underthesea_tokenize = None

        self._re_spaces = re.compile(r"[\s\u200b\ufeff]+")
        self._re_punct = re.compile(r"([.,!?;:\"'()\[\]{}–—…/])")
        pattern_dau = "|".join(
            re.escape(k) for k in sorted(self.BANG_CHUAN_HOA_DAU.keys(), key=len, reverse=True)
        )
        self._re_dau_thanh = re.compile(rf"\b({pattern_dau})\b", flags=re.IGNORECASE)
        self._re_dau_cu_nhanh = re.compile(r"[oO][àáảãạèéẻẽẹÀÁẢÃẠÈÉẺẼẸ]|[uU][ỳýỷỹỵỲÝỶỸỴ]")

        self._vocab: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self._id_to_vocab: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        for idx, tu in enumerate(sorted(self._tu_dien), start=len(self.SPECIAL_TOKENS)):
            self._vocab[tu] = idx
            self._vocab[tu.replace(" ", "_")] = idx
            self._id_to_vocab[idx] = tu

    @property
    def co_underthesea(self) -> bool:
        """Cho biết bộ tách từ có đang kích hoạt backend `underthesea` hay không."""
        return self._underthesea_tokenize is not None

    def _chen_trie(self, cum_tu: str) -> None:
        """Chèn từ ghép vào cây tiền tố (Prefix Trie) để khớp O(k) không cấp phát chuỗi tạm."""
        parts = cum_tu.split()
        if len(parts) < 2:
            return
        node = self._trie_tu_ghep
        for p in parts:
            nxt = node.get(p)
            if nxt is None:
                nxt = {}
                node[p] = nxt
            node = nxt
        node["\0"] = True

    def them_tu_dien(self, ds_tu: List[str]) -> None:
        """Bổ sung thêm danh sách từ ghép vào từ điển phân tầng độ dài và cây tiền tố Trie."""
        ds_sach: List[str] = []
        for tu in ds_tu:
            t = tu.strip().lower()
            if t:
                self._tu_dien.add(t)
                so_tieng = len(t.split())
                self._tu_dien_theo_do_dai.setdefault(so_tieng, set()).add(t)
                if so_tieng > self._max_word_len:
                    self._max_word_len = so_tieng
                self._chen_trie(t)
                ds_sach.append(t)
        if ds_sach and self._rust_tokenizer is not None:
            try:
                self._rust_tokenizer.add_words(ds_sach)
            except Exception:
                pass

    def chuan_hoa_unicode(self, van_ban: str) -> str:
        """Chuẩn hóa văn bản sang dạng Unicode NFC và loại bỏ ký tự điều khiển ẩn."""
        if not van_ban:
            return ""
        if _rust_core is not None:
            try:
                sach = str(_rust_core.fast_clean_spaces(van_ban))
                if sach.isascii():
                    return sach
                return unicodedata.normalize("NFC", sach)
            except Exception:
                pass
        vb = van_ban if van_ban.isascii() else unicodedata.normalize("NFC", van_ban)
        return self._re_spaces.sub(" ", vb).strip()

    def chuan_hoa_dau_thanh(self, van_ban: str) -> str:
        """Chuẩn hóa vị trí dấu thanh tiếng Việt về dạng chuẩn hiện đại và bảo toàn kiểu chữ hoa/thường."""
        vb = self.chuan_hoa_unicode(van_ban)
        if not vb or self._re_dau_cu_nhanh.search(vb) is None:
            return vb

        def _thay_the(match: re.Match[str]) -> str:
            goc = match.group(0)
            thay = self.BANG_CHUAN_HOA_DAU.get(goc.lower(), goc)
            return _giu_kieu_chu(goc, thay)

        return self._re_dau_thanh.sub(_thay_the, vb)

    def xoa_dau_tieng_viet(self, van_ban: str) -> str:
        """Chuyển đổi tiếng Việt có dấu sang không dấu siêu nhanh (Unaccented / Slugify)."""
        if not van_ban:
            return ""
        vb = unicodedata.normalize("NFD", van_ban)
        vb = "".join(c for c in vb if unicodedata.category(c) != "Mn")
        vb = vb.replace("đ", "d").replace("Đ", "D")
        return unicodedata.normalize("NFC", vb)

    def tach_tu(
        self,
        van_ban: str,
        noi_tu_ghep: bool = False,
        ky_tu_noi: str = "_",
        chuan_hoa_dau: bool = True,
    ) -> List[str]:
        """
        Tách từ tiếng Việt chuyên sâu theo thuật toán Maximum Matching (hoặc `underthesea` nếu được bật).

        Tham số:
            van_ban: Chuỗi văn bản tiếng Việt cần tách.
            noi_tu_ghep: Nếu True, từ ghép sẽ được nối lại bằng `ky_tu_noi` (vd: `trí_tuệ_nhân_tạo`).
            ky_tu_noi: Ký tự dùng để nối từ ghép (mặc định là `_`).
            chuan_hoa_dau: Tự động chuẩn hóa dấu thanh và Unicode trước khi tách.

        Trả về:
            Danh sách các token từ đơn và từ ghép.
        """
        if not van_ban:
            return []

        if (
            self._underthesea_tokenize is None
            and _rust_core is not None
            and self._rust_tokenizer is not None
            and hasattr(self._rust_tokenizer, "normalize_and_segment")
        ):
            try:
                vb_nfc = (
                    van_ban
                    if (van_ban.isascii() or unicodedata.is_normalized("NFC", van_ban))
                    else unicodedata.normalize("NFC", van_ban)
                )
                return list(
                    self._rust_tokenizer.normalize_and_segment(
                        vb_nfc, noi_tu_ghep, ky_tu_noi, chuan_hoa_dau
                    )
                )
            except Exception:
                pass

        if chuan_hoa_dau:
            van_ban = self.chuan_hoa_dau_thanh(van_ban)
        else:
            van_ban = self.chuan_hoa_unicode(van_ban)

        if self._underthesea_tokenize is not None:
            try:
                ket_qua_ut = self._underthesea_tokenize(van_ban)
                if noi_tu_ghep:
                    return [t.replace(" ", ky_tu_noi) for t in ket_qua_ut]
                return list(ket_qua_ut)
            except Exception:
                pass

        if _rust_core is not None and self._rust_tokenizer is not None:
            try:
                return list(self._rust_tokenizer.segment(van_ban, noi_tu_ghep, ky_tu_noi))
            except Exception:
                pass

        van_ban_phan_tich = self._re_punct.sub(r" \1 ", van_ban)
        cac_tieng = van_ban_phan_tich.split()
        cac_tieng_thuong = [t.lower() for t in cac_tieng]

        ket_qua: List[str] = []
        i = 0
        n = len(cac_tieng)
        trie = self._trie_tu_ghep
        joiner = ky_tu_noi if noi_tu_ghep else " "

        while i < n:
            node = trie.get(cac_tieng_thuong[i])
            if node is None:
                ket_qua.append(cac_tieng[i])
                i += 1
                continue
            best_end = i
            j = i + 1
            while j < n:
                nxt = node.get(cac_tieng_thuong[j])
                if nxt is None:
                    break
                node = nxt
                j += 1
                if "\0" in node:
                    best_end = j
            if best_end > i + 1:
                ket_qua.append(joiner.join(cac_tieng[i:best_end]))
                i = best_end
            else:
                ket_qua.append(cac_tieng[i])
                i += 1

        return ket_qua

    def dem_so_tu(self, van_ban: str, noi_tu_ghep: bool = True, su_dung_rust: bool = True) -> int:
        """Đếm số lượng từ/token tiếng Việt với tốc độ cao (hỗ trợ tăng tốc Rayon đa luồng qua Rust Core khi đếm từ thô)."""
        if not noi_tu_ghep and su_dung_rust and _rust_core is not None and len(van_ban) > 100:
            try:
                return int(_rust_core.fast_token_count(van_ban))
            except Exception:
                pass
        return len(self.tach_tu(van_ban, noi_tu_ghep=noi_tu_ghep))

    def tim_tu_khoa_nhanh(self, van_ban: str, tu_khoa: str) -> List[int]:
        """Tìm nhanh tất cả vị trí xuất hiện của từ khóa bằng Rust SIMD hoặc thuật toán chuỗi Python."""
        if not van_ban or not tu_khoa:
            return []
        if _rust_core is not None:
            try:
                return _rust_core.fast_keyword_search(van_ban, tu_khoa)
            except Exception:
                pass
        indices: List[int] = []
        start = 0
        while True:
            idx = van_ban.find(tu_khoa, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + len(tu_khoa)
        return indices

    dem_token = dem_so_tu

    @property
    def vocab_size(self) -> int:
        """Kích thước từ vựng hiện tại."""
        return len(self._id_to_vocab)

    def __len__(self) -> int:
        return len(self._id_to_vocab)

    @property
    def tu_vung(self) -> Dict[str, int]:
        """Bảng từ điển ánh xạ từ token sang ID."""
        return dict(self._vocab)

    def ma_hoa(
        self,
        van_ban: str,
        them_cls_sep: bool = False,
        do_dai_toi_da: Optional[int] = None,
    ) -> List[int]:
        """
        Mã hóa văn bản tiếng Việt thành chuỗi token IDs (Native Tokenizer Encoding).

        Args:
            van_ban: Chuỗi văn bản tiếng Việt cần mã hóa.
            them_cls_sep: Nếu True, tự động gắn token [CLS] ở đầu và [SEP] ở cuối.
            do_dai_toi_da: Giới hạn độ dài token tối đa (cắt bớt nếu vượt quá).

        Returns:
            Danh sách các token IDs số nguyên.
        """
        if not van_ban:
            return []

        tokens = self.tach_tu(van_ban, noi_tu_ghep=True)
        ids: List[int] = []

        if them_cls_sep:
            ids.append(self.SPECIAL_TOKENS["[CLS]"])

        for tok in tokens:
            t_lower = tok.lower()
            if t_lower in self._vocab:
                ids.append(self._vocab[t_lower])
            else:
                h_id = 10000 + (BoNhungVanBanSieuToc._bam_chuoi(t_lower) % 20000)
                self._vocab[t_lower] = h_id
                self._id_to_vocab[h_id] = tok
                ids.append(h_id)

        if them_cls_sep:
            ids.append(self.SPECIAL_TOKENS["[SEP]"])

        if do_dai_toi_da is not None and do_dai_toi_da > 0:
            ids = ids[:do_dai_toi_da]

        return ids

    encode = ma_hoa

    def giai_ma(self, ids: List[int], bo_special: bool = True) -> str:
        """
        Giải mã danh sách token IDs ngược về văn bản tiếng Việt (Native Tokenizer Decoding).

        Args:
            ids: Danh sách các token ID số nguyên.
            bo_special: Nếu True, loại bỏ các token đặc biệt ([CLS], [SEP], [PAD], [UNK], [MASK]).

        Returns:
            Chuỗi văn bản tiếng Việt tương ứng.
        """
        if not ids:
            return ""

        special_ids = {0, 1, 2, 3, 4}
        cac_tu: List[str] = []

        for i in ids:
            if bo_special and i in special_ids:
                continue
            if i in self._id_to_vocab:
                cac_tu.append(self._id_to_vocab[i].replace("_", " "))
            else:
                cac_tu.append(f"<tok_{i}>")

        return " ".join(cac_tu)

    decode = giai_ma

    def pad(self, ids: List[int], do_dai: int, gia_tri_pad: int = 0) -> List[int]:
        """Padding hoặc cắt bớt danh sách IDs token đến độ dài cố định."""
        if len(ids) >= do_dai:
            return ids[:do_dai]
        return ids + [gia_tri_pad] * (do_dai - len(ids))


class BoNhungVanBanSieuToc:
    """
    Bộ Nhúng Văn Bản Tiếng Việt Siêu Tốc (Fast Vietnamese Native Embedder):
    - Sử dụng Subword, Word Bigram & Syllable Feature Hashing chuẩn hóa L2 vector.
    - Thời gian tính toán vector: < 0.05ms / văn bản.
    - Không cần mô hình học sâu hay GPU, không phụ thuộc PyTorch.
    - Bảo toàn độ tương đồng ngữ nghĩa đặc thù của tiếng Việt.
    """

    def __init__(self, so_chieu: int = 256, tokenizer: Optional[BoTachTuTiengVietSieuToc] = None) -> None:
        self.so_chieu = so_chieu
        self.tokenizer = tokenizer or BoTachTuTiengVietSieuToc()

    @staticmethod
    def _bam_chuoi(chuoi: str) -> int:
        """Hàm băm xác định siêu tốc độc lập với PYTHONHASHSEED."""
        return int.from_bytes(hashlib.blake2b(chuoi.encode("utf-8"), digest_size=4).digest(), "little")

    def nhung(self, van_ban: str) -> np.ndarray:
        """Chuyển đổi văn bản thành vector đặc trưng chuẩn hóa L2."""
        if not van_ban or not van_ban.strip():
            return np.zeros(self.so_chieu, dtype=np.float32)

        tokens = self.tokenizer.tach_tu(van_ban, noi_tu_ghep=True)
        vec = np.zeros(self.so_chieu, dtype=np.float32)
        lower_tokens = [token.lower() for token in tokens]

        for idx, t in enumerate(lower_tokens):
            h1 = self._bam_chuoi(t) % self.so_chieu
            weight = 1.0 / (1.0 + 0.01 * idx)
            vec[h1] += weight

            if idx + 1 < len(lower_tokens):
                bigram = f"{t}|{lower_tokens[idx + 1]}"
                h_bi = self._bam_chuoi(bigram) % self.so_chieu
                vec[h_bi] += 0.45 * weight

            if len(t) >= 4:
                for j in range(len(t) - 2):
                    sub = t[j : j + 3]
                    h_sub = self._bam_chuoi(sub) % self.so_chieu
                    vec[h_sub] += 0.3 * weight

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    def nhung_lo(self, ds_van_ban: List[str]) -> np.ndarray:
        """Nhúng một lô nhiều văn bản cùng lúc thành ma trận 2D."""
        if not ds_van_ban:
            return np.zeros((0, self.so_chieu), dtype=np.float32)
        return np.vstack([self.nhung(vb) for vb in ds_van_ban])

    def do_tuong_dong(self, vb1: str, vb2: str) -> float:
        """Tính độ tương đồng ngữ nghĩa Cosine Similarity giữa 2 văn bản."""
        v1 = self.nhung(vb1)
        v2 = self.nhung(vb2)
        return float(np.dot(v1, v2))

    def tim_kiem_tuong_dong_hang_loat(
        self,
        query_vec: Union[np.ndarray, List[float]],
        ma_tran_vectors: Union[np.ndarray, List[List[float]]],
    ) -> np.ndarray:
        """
        Tính toán Cosine Similarity hàng loạt trên đa lõi CPU (Rust Rayon / NumPy SIMD).

        Tham số:
            query_vec: Vector truy vấn (1D).
            ma_tran_vectors: Ma trận chứa tập hợp các vector cơ sở dữ liệu (2D).

        Trả về:
            Mảng 1D chứa điểm tương đồng cosine của từng vector.
        """
        if len(ma_tran_vectors) == 0:
            return np.zeros(0, dtype=np.float32)

        if _rust_core is not None:
            try:
                q_list = query_vec.tolist() if isinstance(query_vec, np.ndarray) else list(query_vec)
                m_list = (
                    ma_tran_vectors.tolist()
                    if isinstance(ma_tran_vectors, np.ndarray)
                    else [list(v) for v in ma_tran_vectors]
                )
                res = _rust_core.fast_batch_cosine_similarity(q_list, m_list)
                return np.array(res, dtype=np.float32)
            except Exception:
                pass

        q = np.asarray(query_vec, dtype=np.float32)
        m = np.asarray(ma_tran_vectors, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        m_norms = np.linalg.norm(m, axis=1)
        dots = np.dot(m, q)
        denom = q_norm * m_norms
        denom = np.where(denom < 1e-9, 1.0, denom)
        return np.asarray(dots / denom, dtype=np.float32)


FastVietnameseTokenizer = BoTachTuTiengVietSieuToc
NativeVietnameseTokenizer = BoTachTuTiengVietSieuToc
FastVietnameseEmbedder = BoNhungVanBanSieuToc

__all__ = [
    "BoTachTuTiengVietSieuToc",
    "FastVietnameseTokenizer",
    "NativeVietnameseTokenizer",
    "BoNhungVanBanSieuToc",
    "FastVietnameseEmbedder",
    "co_ho_tro_rust",
]
