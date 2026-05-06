import pytest


class TestPhanTichCamXuc:
    def test_creation_tu_dien(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        assert ptx.che_do == "tu_dien"

    def test_phan_tich_tu_dien_positive(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        result = ptx.phan_tich("Sản phẩm rất tốt và tuyệt vời")
        assert result["nhan"] in ("positive", "neutral", "negative")
        assert "xac_suat" in result

    def test_phan_tich_tu_dien_negative(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        result = ptx.phan_tich("Sản phẩm rất tệ và kém chất lượng")
        assert "nhan" in result

    def test_phan_tich_neutral(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        result = ptx.phan_tich("Hôm nay trời mưa")
        assert result["nhan"] in ("positive", "neutral", "negative")

    def test_phan_tich_nhieu(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        results = ptx.phan_tich_nhieu(["tốt", "xấu", "bình thường"])
        assert len(results) == 3

    def test_thong_ke(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        tk = ptx.thong_ke(["tốt", "tệ", "bình thường"])
        assert "positive" in tk
        assert "negative" in tk
        assert "neutral" in tk

    def test_huan_luyen(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        van_ban = ["tốt", "tệ", "hay", "dở"] * 10
        nhan = ["positive", "negative", "positive", "negative"] * 10
        score = ptx.huan_luyen(van_ban, nhan)
        assert isinstance(score, float)
        assert ptx._da_huan_luyen is True

    def test_phan_tich_with_trained_model(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        van_ban = ["tốt", "tệ", "hay", "dở"] * 10
        nhan = ["positive", "negative", "positive", "negative"] * 10
        ptx.huan_luyen(van_ban, nhan)
        result = ptx.phan_tich("rất tốt")
        assert "nhan" in result


class TestDichThuat:
    def test_creation(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        assert dich.che_do == "dictionary"

    def test_invalid_che_do(self):
        from vietnamese_ai.nlp.translation import DichThuat

        with pytest.raises(ValueError, match="che_do phải là"):
            DichThuat(che_do="invalid")

    def test_dich_en_vi(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        result = dich.dich("hello", nguon="en", dich="vi")
        assert "xin chào" in result["dich"].lower() or result["dich"] != "hello"

    def test_dich_vi_en(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        result = dich.dich("xin chào", nguon="vi", dich="en")
        assert result["nguon"] == "vi"

    def test_dich_no_dict(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        result = dich.dich("hello", nguon="en", dich="fr")
        assert result["dich"] == "hello"

    def test_dich_batch(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        results = dich.dich_batch(["hello", "world"], nguon="en", dich="vi")
        assert len(results) == 2

    def test_them_tu_dien(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        dich.them_tu_dien("en_fr", {"hello": "bonjour"})
        assert "hello" in dich.lay_tu_dien("en_fr")

    def test_lay_tu_dien(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        td = dich.lay_tu_dien("en_vi")
        assert "hello" in td

    def test_lay_tu_dien_nonexistent(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        td = dich.lay_tu_dien("nonexistent")
        assert td == {}

    def test_thong_ke(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        tk = dich.thong_ke()
        assert tk["che_do"] == "dictionary"
        assert tk["co_llm"] is False

    def test_repr(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        assert "DichThuat" in repr(dich)

    def test_dich_llm(self):
        from vietnamese_ai.nlp.translation import DichThuat

        def mock_sinh(prompt):
            return "xin chào"

        dich = DichThuat(che_do="llm", ham_sinh=mock_sinh)
        result = dich.dich("hello", nguon="en", dich="vi")
        assert result["dich"] == "xin chào"

    def test_dich_llm_error_fallback(self):
        from vietnamese_ai.nlp.translation import DichThuat

        def mock_sinh(prompt):
            raise Exception("LLM error")

        dich = DichThuat(che_do="llm", ham_sinh=mock_sinh)
        result = dich.dich("hello", nguon="en", dich="vi")
        assert result["dich"] != ""

    def test_dich_hybrid(self):
        from vietnamese_ai.nlp.translation import DichThuat

        def mock_sinh(prompt):
            return "translated"

        dich = DichThuat(che_do="hybrid", ham_sinh=mock_sinh)
        result = dich.dich("some unknown text xyz", nguon="en", dich="vi")
        assert "dich" in result

    def test_dich_preserve_case(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat()
        result = dich.dich("Hello", nguon="en", dich="vi")
        assert result["dich"][0].isupper()


class TestNhanDienThucThe:
    def test_creation(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        assert ner._underthesea is None

    def test_nhan_dien_regex(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Email: test@example.com, SĐT: 0912345678")
        loai_list = [r["loai"] for r in results]
        assert "EMAIL" in loai_list
        assert "SO_DIEN_THOAI" in loai_list

    def test_nhan_dien_ngay_thang(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Ngày 01/01/2024 rất đẹp")
        loai_list = [r["loai"] for r in results]
        assert "NGAY_THANG" in loai_list

    def test_nhan_dien_url(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Xem tại https://example.com")
        loai_list = [r["loai"] for r in results]
        assert "URL" in loai_list

    def test_nhan_dien_tien_te(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Giá 500.000 đồng")
        loai_list = [r["loai"] for r in results]
        assert "TIEN_TE" in loai_list

    def test_nhan_dien_dia_danh(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Tôi sống ở Hà Nội")
        loai_list = [r["loai"] for r in results]
        assert "DIA_DANH" in loai_list

    def test_nhan_dien_chuc_danh(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Giám đốc Nguyễn Văn A")
        loai_list = [r["loai"] for r in results]
        assert "CHUC_DANH" in loai_list

    def test_loai_loc(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien(
            "Email: test@example.com ở Hà Nội",
            loai_loc=["EMAIL"],
        )
        loai_list = [r["loai"] for r in results]
        assert all(item == "EMAIL" for item in loai_list)

    def test_them_dia_danh(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ner.them_dia_danh("Buôn Ma Thuột")
        assert "Buôn Ma Thuột" in ner._dia_danh

    def test_them_chuc_danh(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ner.them_chuc_danh("CEO")
        assert "CEO" in ner._chuc_danh

    def test_them_mau(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ner.them_mau("CUSTOM", r"\d{3}-\d{3}")
        assert "CUSTOM" in ner._mau

    def test_thong_ke(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        tk = ner.thong_ke()
        assert tk["so_loai_mau"] > 0
        assert tk["so_dia_danh"] > 0

    def test_repr(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        assert "NhanDienThucThe" in repr(ner)

    def test_custom_patterns(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(
            su_dung_underthesea=False,
            mau_tuy_chinh={"MA_SO": [r"MS-\d+"]},
        )
        results = ner.nhan_dien("Mã số MS-12345")
        loai_list = [r["loai"] for r in results]
        assert "MA_SO" in loai_list

    def test_custom_dict(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(
            su_dung_underthesea=False,
            tu_dien_tuy_chinh={"dia_danh": {"Kon Tum"}, "chuc_danh": {"CTO"}},
        )
        assert "Kon Tum" in ner._dia_danh
        assert "CTO" in ner._chuc_danh

    def test_loai_trung_lap(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        results = ner.nhan_dien("Hà Nội là thủ đô của Hà Nội")
        van_ban_list = [(r["van_ban"], r["loai"]) for r in results if r["loai"] == "DIA_DANH"]
        assert len(van_ban_list) <= 2
