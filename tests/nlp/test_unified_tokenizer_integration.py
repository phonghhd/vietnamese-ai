"""
Kiểm thử tích hợp & chuẩn hóa hệ sinh thái Tokenizer trong Vietnamese AI Framework.
Bao gồm: BoTachTuTiengVietSieuToc (Native Fast Tokenizer), VietnameseTokenizer (BPE/WordPiece),
và XuLyVanBan (Preprocessing).
"""

from __future__ import annotations

from vietnamese_ai.nlp.native_tokenizer import (
    BoTachTuTiengVietSieuToc,
    FastVietnameseTokenizer,
    NativeVietnameseTokenizer,
)
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.serving.continuous_batching import ContinuousBatcher
from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer


class TestNativeTokenizerProtocol:
    def test_native_tokenizer_protocol_parity(self):
        """Kiểm tra BoTachTuTiengVietSieuToc tuân thủ đầy đủ chuẩn Tokenizer (ma_hoa, giai_ma, encode, decode, pad)."""
        tok = BoTachTuTiengVietSieuToc()

        assert FastVietnameseTokenizer is BoTachTuTiengVietSieuToc
        assert NativeVietnameseTokenizer is BoTachTuTiengVietSieuToc

        # 1. Kích thước từ vựng và len()
        assert tok.vocab_size >= 50
        assert len(tok) == tok.vocab_size
        assert isinstance(tok.tu_vung, dict)
        assert "[CLS]" in tok.tu_vung
        assert "[SEP]" in tok.tu_vung

        # 2. Mã hóa (ma_hoa / encode)
        van_ban = "Trí tuệ nhân tạo và học máy tại Việt Nam"
        ids = tok.ma_hoa(van_ban, them_cls_sep=True)
        assert len(ids) >= 4
        assert ids[0] == tok.SPECIAL_TOKENS["[CLS]"]
        assert ids[-1] == tok.SPECIAL_TOKENS["[SEP]"]

        # Alias encode
        ids_enc = tok.encode(van_ban, them_cls_sep=True)
        assert ids == ids_enc

        # 3. Giải mã (giai_ma / decode)
        giai_ma_str = tok.giai_ma(ids, bo_special=True)
        assert "trí tuệ nhân tạo" in giai_ma_str.lower()
        assert "học máy" in giai_ma_str.lower()

        # Alias decode
        assert tok.decode(ids, bo_special=True) == giai_ma_str

        # 4. Padding
        padded = tok.pad(ids, do_dai=16, gia_tri_pad=0)
        assert len(padded) == 16
        assert padded[-1] == 0

    def test_native_tokenizer_continuous_batching_compatibility(self):
        """Kiểm tra BoTachTuTiengVietSieuToc tương thích trực tiếp với ContinuousBatcher."""
        tok = BoTachTuTiengVietSieuToc()
        prompt = "Trí tuệ nhân tạo Việt Nam"

        ids = ContinuousBatcher._ma_hoa_prompt(prompt, so_tu_vung=tok.vocab_size, tokenizer=tok)
        assert len(ids) > 0

        decoded = ContinuousBatcher._giai_ma_token(ids[0], tokenizer=tok)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


class TestVietnameseTokenizerStandardization:
    def test_transformer_tokenizer_extensions(self):
        """Kiểm tra các alias và phương thức chuẩn hóa của VietnameseTokenizer."""
        corpus = [
            "trí tuệ nhân tạo phát triển mạnh mẽ",
            "học máy và xử lý ngôn ngữ tự nhiên",
            "mô hình ngôn ngữ lớn tiếng việt",
        ]
        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=200, dung_tach_tu_tieng_viet=True)
        tok.huan_luyen(corpus)

        # 1. Properties
        assert tok.vocab_size == len(tok)
        assert len(tok) >= 50

        # 2. tach_tu & dem_so_tu
        tokens = tok.tach_tu("học máy tiếng việt")
        assert len(tokens) > 0
        assert tok.dem_so_tu("học máy tiếng việt") == len(tokens)

        # 3. encode / decode aliases
        ids_mh = tok.ma_hoa("trí tuệ nhân tạo")
        ids_enc = tok.encode("trí tuệ nhân tạo")
        assert ids_mh == ids_enc

        gm_str = tok.giai_ma(ids_mh)
        dec_str = tok.decode(ids_mh)
        assert gm_str == dec_str


class TestXuLyVanBanIntegration:
    def test_xu_ly_van_ban_backed_by_native_tokenizer(self):
        """Kiểm tra XuLyVanBan tự động sử dụng BoTachTuTiengVietSieuToc khi không có underthesea."""
        xl = XuLyVanBan(dung_native_tokenizer=True)
        assert xl.tokenizer is not None
        assert isinstance(xl.tokenizer, BoTachTuTiengVietSieuToc)

        # Tách từ ghép chuyên sâu
        tu_list = xl.tach_tu("Trí tuệ nhân tạo và học máy")
        assert "trí tuệ nhân tạo" in tu_list or "trí_tuệ_nhân_tạo" in tu_list
        assert "học máy" in tu_list or "học_máy" in tu_list

        # Tách từ chuỗi (format text)
        tu_chuoi = xl.tach_tu_chuoi("Trí tuệ nhân tạo và học máy")
        assert "trí_tuệ_nhân_tạo" in tu_chuoi
        assert "học_máy" in tu_chuoi

        # Chuẩn hóa dấu thanh
        chuan = xl.chuan_hoa("Toà án và uỷ ban hoà giải")
        assert "tòa" in chuan
        assert "ủy" in chuan
        assert "hòa" in chuan

        # Xóa dấu tiếng Việt
        khong_dau = xl.xoa_dau("Việt Nam đất nước tuyệt vời")
        assert khong_dau.lower() == "viet nam dat nuoc tuyet voi"


class TestCLITokenizeEncode:
    def test_cli_tokenize_encode_flag(self):
        """Kiểm tra cờ CLI vai tokenize --encode."""
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["tokenize", "Trí tuệ nhân tạo", "--encode"])
        assert args.command == "tokenize"
        assert args.encode is True
