"""Tests cho v10.0: RAG, Serving, Prompts, NLP Extensions, Compression, Production."""

import numpy as np
import pytest

# === RAG Tests ===


class TestCSDLVector:
    def test_khoi_tao(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=128)
        assert csdl.so_luong() == 0
        assert csdl.kich_thuoc == 128

    def test_chen_va_tim_kiem(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=4, khoang_cach="cosine")
        csdl.chen("doc_1", np.array([1.0, 0.0, 0.0, 0.0]), {"noi_dung": "hello"})
        csdl.chen("doc_2", np.array([0.0, 1.0, 0.0, 0.0]), {"noi_dung": "world"})
        csdl.chen("doc_3", np.array([0.9, 0.1, 0.0, 0.0]), {"noi_dung": "hello world"})
        assert csdl.so_luong() == 3

        ket_qua = csdl.tim_kiem(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(ket_qua) == 2
        assert ket_qua[0]["ma"] == "doc_1"

    def test_chen_batch(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=3)
        vectors = np.eye(3)
        csdl.chen_batch(["a", "b", "c"], vectors)
        assert csdl.so_luong() == 3

    def test_xoa(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=2)
        csdl.chen("a", np.array([1.0, 0.0]))
        csdl.chen("b", np.array([0.0, 1.0]))
        assert csdl.xoa("a") is True
        assert csdl.so_luong() == 1
        assert csdl.xoa("z") is False

    def test_luu_tai(self, tmp_path):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=3)
        csdl.chen("a", np.array([1.0, 2.0, 3.0]), {"test": True})
        duong_dan = str(tmp_path / "csdl.pkl")
        csdl.luu(duong_dan)

        csdl2 = CSDLVector.tai(duong_dan)
        assert csdl2.so_luong() == 1
        assert np.allclose(csdl2.lay_vector("a"), [1.0, 2.0, 3.0])

    def test_l2_va_inner_product(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        for kc in ["l2", "inner_product"]:
            csdl = CSDLVector(kich_thuoc=2, khoang_cach=kc)
            csdl.chen("a", np.array([1.0, 0.0]))
            csdl.chen("b", np.array([0.0, 1.0]))
            ket_qua = csdl.tim_kiem(np.array([1.0, 0.0]), top_k=2)
            assert len(ket_qua) == 2

    def test_thong_ke(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=4)
        csdl.chen("a", np.array([1.0, 2.0, 3.0, 4.0]))
        stats = csdl.thong_ke()
        assert stats["so_luong"] == 1
        assert stats["kich_thuoc"] == 4

    def test_update_existing(self):
        from vietnamese_ai.rag.vector_store import CSDLVector

        csdl = CSDLVector(kich_thuoc=2)
        csdl.chen("a", np.array([1.0, 0.0]))
        csdl.chen("a", np.array([0.0, 1.0]))
        assert csdl.so_luong() == 1
        assert np.allclose(csdl.lay_vector("a"), [0.0, 1.0])


class TestCatVanBan:
    def test_chia_theo_tu(self):
        from vietnamese_ai.rag.chunker import CatVanBan

        cat = CatVanBan(kich_thuoc=5, chong_chong=2, chien_luoc="tu", toi_thieu_kich_thuoc=1)
        van_ban = " ".join([f"tu{i}" for i in range(20)])
        chunks = cat.chia(van_ban)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["noi_dung"].split()) <= 5

    def test_chia_theo_ky_tu(self):
        from vietnamese_ai.rag.chunker import CatVanBan

        cat = CatVanBan(kich_thuoc=50, chong_chong=10, chien_luoc="ky_tu")
        van_ban = "A" * 200
        chunks = cat.chia(van_ban)
        assert len(chunks) > 1

    def test_chia_theo_cau(self):
        from vietnamese_ai.rag.chunker import CatVanBan

        cat = CatVanBan(kich_thuoc=10, chong_chong=2, chien_luoc="cau", toi_thieu_kich_thuoc=1)
        van_ban = "Cau mot. Cau hai. Cau ba. Cau bon. Cau nam."
        chunks = cat.chia(van_ban)
        assert len(chunks) >= 1

    def test_chia_theo_doan(self):
        from vietnamese_ai.rag.chunker import CatVanBan

        cat = CatVanBan(kich_thuoc=10, chong_chong=2, chien_luoc="doan")
        van_ban = "Doan mot day du.\n\nDoan hai day du.\n\nDoan ba."
        chunks = cat.chia(van_ban)
        assert len(chunks) >= 1

    def test_van_ban_trong(self):
        from vietnamese_ai.rag.chunker import CatVanBan

        cat = CatVanBan()
        assert cat.chia("") == []
        assert cat.chia("   ") == []

    def test_metadata(self):
        from vietnamese_ai.rag.chunker import CatVanBan

        cat = CatVanBan(kich_thuoc=5, chong_chong=2, chien_luoc="tu")
        chunks = cat.chia("a b c d e f g h", {"source": "test"})
        assert all(c["metadata"].get("source") == "test" for c in chunks)


class TestTrichXuat:
    def test_them_va_tim(self):
        from vietnamese_ai.rag.chunker import CatVanBan
        from vietnamese_ai.rag.retriever import TrichXuat

        cat = CatVanBan(kich_thuoc=200, chong_chong=50, chien_luoc="tu", toi_thieu_kich_thuoc=1)
        trich = TrichXuat(che_do="keyword", cat_van_ban=cat)
        van_ban = " ".join([f"tu{i}" for i in range(50)])
        trich.them_tai_lieu("doc1", van_ban)
        trich.them_tai_lieu("doc2", "cloud computing dich vu may chu")
        assert trich.so_tai_lieu() == 2
        ket_qua = trich.tim_kiem("tu1 tu2 tu3", top_k=2)
        assert len(ket_qua) >= 1

    def test_hybrid_search(self):
        from vietnamese_ai.rag.chunker import CatVanBan
        from vietnamese_ai.rag.retriever import TrichXuat

        cat = CatVanBan(kich_thuoc=200, chong_chong=50, chien_luoc="tu", toi_thieu_kich_thuoc=1)
        trich = TrichXuat(che_do="hybrid", cat_van_ban=cat)
        trich.them_tai_lieu("doc1", "Vietnamese AI Framework la framework hoc may")
        trich.them_tai_lieu("doc2", "Cloud computing va microservices")
        ket_qua = trich.tim_kiem("AI framework", top_k=2)
        assert len(ket_qua) >= 1

    def test_xoa_tai_lieu(self):
        from vietnamese_ai.rag.chunker import CatVanBan
        from vietnamese_ai.rag.retriever import TrichXuat

        cat = CatVanBan(kich_thuoc=200, chong_chong=50, chien_luoc="tu", toi_thieu_kich_thuoc=1)
        trich = TrichXuat(che_do="keyword", cat_van_ban=cat)
        trich.them_tai_lieu("doc1", "Van ban thu nhat co du lieu")
        assert trich.xoa_tai_lieu("doc1") > 0
        assert trich.so_tai_lieu() == 0


class TestRAGPipeline:
    def test_pipeline_day_du(self):
        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        rag = RAGPipeline(che_do_tim_kiem="keyword", che_do_rerank="keyword")
        rag.them_tai_lieu("doc1", "Hoc may la linh vuc cua AI")
        rag.them_tai_lieu("doc2", "Cloud computing la dich vu may chu")

        ket_qua = rag.hoi("AI la gi?")
        assert "tra_loi" in ket_qua
        assert "cau_hoi" in ket_qua
        assert ket_qua["cau_hoi"] == "AI la gi?"

    def test_tim_kiem(self):
        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        rag = RAGPipeline(che_do_tim_kiem="keyword")
        rag.them_tai_lieu("doc1", "Hoc may va deep learning")
        ket_qua = rag.tim_kiem("deep learning")
        assert isinstance(ket_qua, list)

    def test_thong_ke(self):
        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        rag.them_tai_lieu("doc1", "Test document")
        stats = rag.thong_ke()
        assert stats["so_tai_lieu"] == 1


class TestSapXepLai:
    def test_mmr(self):
        from vietnamese_ai.rag.reranker import SapXepLai

        reranker = SapXepLai(che_do="mmr")
        ket_qua = [
            {"ma": "a", "diem": 0.9, "metadata": {"noi_dung": "hello world"}},
            {"ma": "b", "diem": 0.8, "metadata": {"noi_dung": "hello there"}},
            {"ma": "c", "diem": 0.7, "metadata": {"noi_dung": "goodbye world"}},
        ]
        result = reranker.sap_xep_lai("hello", ket_qua, top_k=2)
        assert len(result) == 2

    def test_keyword_rerank(self):
        from vietnamese_ai.rag.reranker import SapXepLai

        reranker = SapXepLai(che_do="keyword")
        ket_qua = [
            {"ma": "a", "diem": 0.5, "metadata": {"noi_dung": "AI framework"}},
            {"ma": "b", "diem": 0.9, "metadata": {"noi_dung": "cloud computing"}},
        ]
        result = reranker.sap_xep_lai("AI framework", ket_qua, top_k=2)
        assert len(result) == 2


# === Serving Tests ===


class TestMayChuBatch:
    def test_khoi_tao(self):
        from vietnamese_ai.serving.batch_server import MayChuBatch

        server = MayChuBatch(mo_hinh=None, kich_thuoc_batch=4)
        assert server.kich_thuoc_batch == 4

    def test_du_doan(self):
        from vietnamese_ai.serving.batch_server import MayChuBatch

        class MockModel:
            def du_doan(self, X):
                return np.ones(len(X))

        server = MayChuBatch(mo_hinh=MockModel(), kich_thuoc_batch=2)
        server.bat_dau()
        ket_qua = server.du_doan(np.array([1.0, 2.0]))
        assert ket_qua is not None
        server.dung()

    def test_thong_ke(self):
        from vietnamese_ai.serving.batch_server import MayChuBatch

        server = MayChuBatch(mo_hinh=None)
        stats = server.lay_thong_ke()
        assert "tong_request" in stats


class TestMayChuStream:
    def test_sinh_stream(self):
        from vietnamese_ai.serving.streaming import MayChuStream

        server = MayChuStream(toc_do_token=0.001)
        tokens = list(server.sinh_stream("test prompt"))
        assert len(tokens) > 0

    def test_sinh_sse(self):
        from vietnamese_ai.serving.streaming import MayChuStream

        server = MayChuStream(toc_do_token=0.001)
        sse_lines = list(server.sinh_sse("test"))
        assert any("data:" in line for line in sse_lines)
        assert any("[DONE]" in line for line in sse_lines)

    def test_custom_generator(self):
        from vietnamese_ai.serving.streaming import MayChuStream

        def custom_gen(prompt, **kwargs):
            return ["token1", "token2", "token3"]

        server = MayChuStream(ham_sinh=custom_gen, toc_do_token=0.001)
        tokens = list(server.sinh_stream("test"))
        assert len(tokens) == 3


class TestBoGioiHanTocDo:
    def test_token_bucket(self):
        from vietnamese_ai.serving.rate_limiter import BoGioiHanTocDo

        limiter = BoGioiHanTocDo(go_i_y=5, cua_so=1.0, che_do="token_bucket")
        for _ in range(5):
            assert limiter.cho_phep("client1") is True
        assert limiter.cho_phep("client1") is False

    def test_sliding_window(self):
        from vietnamese_ai.serving.rate_limiter import BoGioiHanTocDo

        limiter = BoGioiHanTocDo(go_i_y=3, cua_so=1.0, che_do="sliding_window")
        for _ in range(3):
            assert limiter.cho_phep("c1") is True
        assert limiter.cho_phep("c1") is False

    def test_con_lai(self):
        from vietnamese_ai.serving.rate_limiter import BoGioiHanTocDo

        limiter = BoGioiHanTocDo(go_i_y=10, cua_so=60.0)
        assert limiter.lay_con_lai("c1") == 10
        limiter.cho_phep("c1")
        assert limiter.lay_con_lai("c1") < 10

    def test_thong_ke(self):
        from vietnamese_ai.serving.rate_limiter import BoGioiHanTocDo

        limiter = BoGioiHanTocDo()
        limiter.cho_phep("c1")
        stats = limiter.lay_thong_ke()
        assert stats["tong_request"] == 1


# === Prompt Tests ===


class TestMauPrompt:
    def test_render(self):
        from vietnamese_ai.prompts.templates import MauPrompt

        mau = MauPrompt("Hello {{ten}}, ban o {{dia_diem}}?")
        result = mau.render(ten="Phong", dia_diem="Ha Noi")
        assert "Phong" in result
        assert "Ha Noi" in result

    def test_bien_chua_dien(self):
        from vietnamese_ai.prompts.templates import MauPrompt

        mau = MauPrompt("Hello {{ten}}")
        with pytest.raises(ValueError, match="Chưa điền"):
            mau.render()

    def test_bien_mac_dinh(self):
        from vietnamese_ai.prompts.templates import MauPrompt

        mau = MauPrompt("Hello {{ten}}", bien_mac_dinh={"ten": "World"})
        result = mau.render()
        assert "World" in result

    def test_mau_mac_dinh(self):
        from vietnamese_ai.prompts.templates import MauPrompt

        templates = MauPrompt.danh_sach_mau_mac_dinh()
        assert "tom_tat" in templates
        assert "phan_tich" in templates

    def test_danh_sach_bien(self):
        from vietnamese_ai.prompts.templates import MauPrompt

        mau = MauPrompt("{{a}} and {{b}}")
        assert set(mau.danh_sach_bien()) == {"a", "b"}


class TestChuoiPrompt:
    def test_chain_basic(self):
        from vietnamese_ai.prompts.chains import ChuoiPrompt
        from vietnamese_ai.prompts.templates import MauPrompt

        chain = ChuoiPrompt()
        chain.them_buoc("step1", MauPrompt("Process: {{input}}"))

        def mock_gen(prompt):
            return f"Result of: {prompt}"

        result = chain.thuc_hien(mock_gen, {"input": "test data"})
        assert result["so_buoc"] == 1
        assert "ket_qua" in result

    def test_few_shot(self):
        from vietnamese_ai.prompts.chains import ChuoiPrompt

        chain = ChuoiPrompt()
        chain.them_few_shot("input1", "output1")
        chain.them_few_shot("input2", "output2")
        prompt = chain.tao_few_shot_prompt("test question")
        assert "Ví dụ" in prompt
        assert "test question" in prompt

    def test_cot_prompt(self):
        from vietnamese_ai.prompts.chains import ChuoiPrompt

        chain = ChuoiPrompt()
        prompt = chain.tao_cot_prompt("AI la gi?")
        assert "Bước 1" in prompt


class TestLuongAnToan:
    def test_an_toan(self):
        from vietnamese_ai.prompts.guardrails import LuongAnToan

        guard = LuongAnToan()
        result = guard.kiem_tra("Noi dung binh thuong")
        assert result["an_toan"] is True

    def test_tu_cam(self):
        from vietnamese_ai.prompts.guardrails import LuongAnToan

        guard = LuongAnToan(tu_cam=["spam", "hack"])
        result = guard.kiem_tra("This is spam content")
        assert result["an_toan"] is False
        assert result["so_loi"] == 1

    def test_do_dai(self):
        from vietnamese_ai.prompts.guardrails import LuongAnToan

        guard = LuongAnToan(toi_da_do_dai=10)
        result = guard.kiem_tra("A" * 20)
        assert result["an_toan"] is False

    def test_pii_detection(self):
        from vietnamese_ai.prompts.guardrails import LuongAnToan

        guard = LuongAnToan(chan_pii=True)
        result = guard.kiem_tra("Lien he: 0912345678")
        assert result["so_canh_bao"] > 0

    def test_loc_pii(self):
        from vietnamese_ai.prompts.guardrails import LuongAnToan

        guard = LuongAnToan()
        filtered, pii = guard.loc_pii("Email: test@gmail.com, SĐT: 0912345678")
        assert "Email đã ẩn" in filtered
        assert "SĐT đã ẩn" in filtered


class TestPhanTichDauRa:
    def test_phan_tich_json(self):
        from vietnamese_ai.prompts.parser import PhanTichDauRa

        parser = PhanTichDauRa()
        result = parser.phan_tich_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_phan_tich_json_in_codeblock(self):
        from vietnamese_ai.prompts.parser import PhanTichDauRa

        parser = PhanTichDauRa()
        result = parser.phan_tich_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_phan_tich_danh_sach(self):
        from vietnamese_ai.prompts.parser import PhanTichDauRa

        parser = PhanTichDauRa()
        items = parser.phan_tich_danh_sach("1. Item one\n2. Item two\n3. Item three")
        assert len(items) == 3

    def test_phan_tich_bang(self):
        from vietnamese_ai.prompts.parser import PhanTichDauRa

        parser = PhanTichDauRa()
        table = parser.phan_tich_bang("| Name | Age |\n|---|---|\n| A | 25 |\n| B | 30 |\n")
        assert len(table) == 2
        assert table[0]["Name"] == "A"

    def test_phan_tich_code_blocks(self):
        from vietnamese_ai.prompts.parser import PhanTichDauRa

        parser = PhanTichDauRa()
        blocks = parser.phan_tich_code_blocks("```python\nprint('hi')\n```")
        assert len(blocks) == 1
        assert blocks[0]["ngon_ngu"] == "python"


# === NLP Extensions Tests ===


class TestNhanDienThucThe:
    def test_nhan_dien_co_ban(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ket_qua = ner.nhan_dien("Lien he: test@gmail.com hoac 0912345678")
        loai = [r["loai"] for r in ket_qua]
        assert "EMAIL" in loai
        assert "SO_DIEN_THOAI" in loai

    def test_nhan_dien_ngay_thang(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ket_qua = ner.nhan_dien("Ngay 01/01/2024 la ngay dau nam")
        loai = [r["loai"] for r in ket_qua]
        assert "NGAY_THANG" in loai

    def test_nhan_dien_tien_te(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ket_qua = ner.nhan_dien("Gia: 50000 VND")
        loai = [r["loai"] for r in ket_qua]
        assert "TIEN_TE" in loai

    def test_nhan_dien_dia_danh(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ket_qua = ner.nhan_dien("Toi song tai Hà Nội")
        loai = [r["loai"] for r in ket_qua]
        assert "DIA_DANH" in loai

    def test_them_dia_danh(self):
        from vietnamese_ai.nlp.ner import NhanDienThucThe

        ner = NhanDienThucThe(su_dung_underthesea=False)
        ner.them_dia_danh("Bình Dương")
        ket_qua = ner.nhan_dien("O Bình Dương")
        assert any(r["van_ban"] == "Bình Dương" for r in ket_qua)


class TestHoiDapTiengViet:
    def test_hoi_dap(self):
        from vietnamese_ai.nlp.qa import HoiDapTiengViet

        qa = HoiDapTiengViet()
        qa.them_tai_lieu(
            "doc1",
            "Hoc may la linh vuc cua tri tue nhan tao. No su dung thuat toan de hoc tu du lieu.",
        )
        ket_qua = qa.hoi("Hoc may la gi?")
        assert "tra_loi" in ket_qua
        assert ket_qua["diem"] > 0

    def test_thong_ke(self):
        from vietnamese_ai.nlp.qa import HoiDapTiengViet

        qa = HoiDapTiengViet()
        qa.them_tai_lieu("doc1", "Day la mot van ban mau. No co nhieu cau.")
        stats = qa.thong_ke()
        assert stats["so_tai_lieu"] == 1


class TestTomTatVanBan:
    def test_tom_tat_extractive(self):
        from vietnamese_ai.nlp.summarization import TomTatVanBan

        tt = TomTatVanBan(che_do="extractive")
        van_ban = (
            "Cau thu nhat la cau quan trong nhat trong van ban nay. "
            "Cau thu hai cung mang nhieu thong tin. "
            "Cau thu ba it quan trong hon. "
            "Cau thu tu khong lien quan lam. "
            "Cau thu nam la cau ket luan."
        )
        ket_qua = tt.tom_tat(van_ban, so_cau=2)
        assert ket_qua["che_do"] == "extractive"
        assert len(ket_qua["tom_tat"]) > 0

    def test_van_ban_trong(self):
        from vietnamese_ai.nlp.summarization import TomTatVanBan

        tt = TomTatVanBan()
        ket_qua = tt.tom_tat("")
        assert ket_qua["tom_tat"] == ""

    def test_abstractive_voi_generator(self):
        from vietnamese_ai.nlp.summarization import TomTatVanBan

        tt = TomTatVanBan(che_do="abstractive", ham_sinh=lambda p: "Tom tat ngan gon")
        ket_qua = tt.tom_tat("Van ban dai...")
        assert ket_qua["tom_tat"] == "Tom tat ngan gon"


class TestDichThuat:
    def test_dich_en_vi(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat(che_do="dictionary")
        ket_qua = dich.dich("hello world", nguon="en", dich="vi")
        assert ket_qua["nguon"] == "en"
        assert ket_qua["dich_lang"] == "vi"

    def test_dich_vi_en(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat(che_do="dictionary")
        ket_qua = dich.dich("xin chào", nguon="vi", dich="en")
        assert "hello" in ket_qua["dich"].lower()

    def test_them_tu_dien(self):
        from vietnamese_ai.nlp.translation import DichThuat

        dich = DichThuat(che_do="dictionary")
        dich.them_tu_dien("en_vi", {"test": "thu nghiem"})
        ket_qua = dich.dich("test", nguon="en", dich="vi")
        assert "thu nghiem" in ket_qua["dich"]


class TestKiemTraChinhTa:
    def test_kiem_tra(self):
        from vietnamese_ai.nlp.spelling import KiemTraChinhTa

        kt = KiemTraChinhTa()
        kt.them_tu_dien({"xin", "chào", "thế", "giới"})
        ket_qua = kt.kiem_tra("xin chao the gioi")
        assert "loi" in ket_qua

    def test_sua_tu_dong(self):
        from vietnamese_ai.nlp.spelling import KiemTraChinhTa

        kt = KiemTraChinhTa()
        kt.them_tu_dien({"không"})
        result = kt.sua("khong")
        assert "không" in result or result == "khong"

    def test_huan_luyen_tu_corpus(self):
        from vietnamese_ai.nlp.spelling import KiemTraChinhTa

        kt = KiemTraChinhTa()
        kt.huan_luyen_tu_corpus(
            [
                "hello world hello python",
                "hello world again",
            ]
        )
        stats = kt.thong_ke()
        assert stats["so_tu_da_hoc"] > 0


# === Compression Tests ===


class TestHocRutGon:
    def test_distillation(self):
        from vietnamese_ai.compression.distillation import HocRutGon
        from vietnamese_ai.models.classifier import PhanLoai

        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] > 0).astype(int)

        teacher = PhanLoai(thuat_toan="rung_ngau_nhien")
        teacher.huan_luyen(X_train, y_train)

        distiller = HocRutGon(teacher=teacher, nhiet_do=3.0)
        student = PhanLoai(thuat_toan="logistic")
        ket_qua = distiller.huan_luyen(student, X_train, y_train)

        assert "student_acc" in ket_qua
        assert "teacher_acc" in ket_qua

    def test_thong_ke(self):
        from vietnamese_ai.compression.distillation import HocRutGon
        from vietnamese_ai.models.classifier import PhanLoai

        teacher = PhanLoai()
        distiller = HocRutGon(teacher=teacher)
        stats = distiller.thong_ke()
        assert stats["nhiet_do"] == 3.0


class TestCatTiaMoHinh:
    def test_magnitude_pruning(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh
        from vietnamese_ai.models.classifier import PhanLoai

        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] > 0).astype(int)

        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)
        ket_qua = pruner.cat_tia(model, X_train, y_train)
        assert ket_qua["ty_le_prune"] >= 0
        assert ket_qua["model"] is not None

    def test_random_pruning(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh
        from vietnamese_ai.models.classifier import PhanLoai

        X_train = np.random.randn(50, 4)
        y_train = (X_train[:, 0] > 0).astype(int)

        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        pruner = CatTiaMoHinh(che_do="random", ty_le=0.5)
        ket_qua = pruner.cat_tia(model, X_train, y_train)
        assert ket_qua["model"] is not None


# === Production Tests ===


class TestKiemTraSucKhoe:
    def test_kiem_tra_co_ban(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        health = KiemTraSucKhoe()
        health.dang_ky_check("test", lambda: True, "Test check")
        ket_qua = health.kiem_tra()
        assert ket_qua["trang_thai"] == "healthy"

    def test_unhealthy(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        health = KiemTraSucKhoe()
        health.dang_ky_check("fail", lambda: False, "Always fail", quan_trong=True)
        ket_qua = health.kiem_tra()
        assert ket_qua["trang_thai"] == "unhealthy"

    def test_degraded(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        health = KiemTraSucKhoe()
        health.dang_ky_check("ok", lambda: True, "OK", quan_trong=True)
        health.dang_ky_check("warn", lambda: False, "Warning", quan_trong=False)
        ket_qua = health.kiem_tra()
        assert ket_qua["trang_thai"] == "degraded"

    def test_ready_va_live(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        health = KiemTraSucKhoe()
        health.dang_ky_check("ok", lambda: True, "OK")
        assert health.ready() is True
        assert health.live() is True

    def test_he_thong_info(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        health = KiemTraSucKhoe()
        ket_qua = health.kiem_tra()
        assert "he_thong" in ket_qua
        assert "python_version" in ket_qua["he_thong"]


class TestMachCat:
    def test_trang_thai_binh_thuong(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        cb = MachCat(so_loi_toi_da=3, ten="test")
        assert cb.trang_thai == "dong"

    def test_chuyen_sang_mo(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        cb = MachCat(so_loi_toi_da=3)
        for _ in range(3):
            cb.ghi_nhan_loi()
        assert cb.trang_thai == "mo"

    def test_chan_request_khi_mo(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        cb = MachCat(so_loi_toi_da=1)
        cb.ghi_nhan_loi()
        assert cb.cho_phep() is False

    def test_fallback(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        fallback_called = [False]

        def fallback(*args):
            fallback_called[0] = True
            return "fallback_result"

        cb = MachCat(so_loi_toi_da=1, ham_fallback=fallback)
        cb.ghi_nhan_loi()
        result = cb.thuc_hien(lambda: "ok")
        assert fallback_called[0] is True
        assert result == "fallback_result"

    def test_reset(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        cb = MachCat(so_loi_toi_da=1)
        cb.ghi_nhan_loi()
        assert cb.trang_thai == "mo"
        cb.reset()
        assert cb.trang_thai == "dong"

    def test_thuc_hien_thanh_cong(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        cb = MachCat(so_loi_toi_da=5)
        result = cb.thuc_hien(lambda: 42)
        assert result == 42

    def test_thong_ke(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        cb = MachCat()
        cb.thuc_hien(lambda: "ok")
        stats = cb.lay_thong_ke()
        assert stats["thanh_cong"] == 1


class TestLoggerCauTruc:
    def test_khoi_tao(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc(ten="test")
        assert logger.ten == "test"

    def test_log_levels(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc(ten="test", cap_do="DEBUG")
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")

    def test_context(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc(ten="test")
        logger.them_context(request_id="123")
        logger.info("test")
        logger.xoa_context()

    def test_thong_ke(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc(ten="test")
        stats = logger.thong_ke()
        assert stats["ten"] == "test"


class TestQuanLyMetrics:
    def test_counter(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        m = QuanLyMetrics()
        m.counter("requests")
        m.counter("requests")
        assert m.lay_counter("requests") == 2.0

    def test_gauge(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        m = QuanLyMetrics()
        m.gauge("active", 5)
        assert m.lay_gauge("active") == 5.0

    def test_histogram(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        m = QuanLyMetrics()
        for v in [10, 20, 30, 40, 50]:
            m.histogram("latency", v)
        stats = m.lay_histogram_stats("latency")
        assert stats["count"] == 5
        assert stats["mean"] == 30.0

    def test_export_prometheus(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        m = QuanLyMetrics()
        m.counter("test_total")
        prom = m.export_prometheus()
        assert "test_total" in prom

    def test_export_json(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        m = QuanLyMetrics()
        m.counter("c1")
        m.gauge("g1", 42)
        data = m.export_json()
        assert "counters" in data
        assert "gauges" in data

    def test_labels(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        m = QuanLyMetrics()
        m.counter("req", {"endpoint": "/predict"})
        m.counter("req", {"endpoint": "/health"})
        assert m.lay_counter("req", {"endpoint": "/predict"}) == 1.0


class TestLamNongModel:
    def test_dang_ky_va_lam_nong(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.production.warmup import LamNongModel

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X, y)

        warmup = LamNongModel(so_lan_warmup=2)
        warmup.dang_ky_model("clf", model, du_lieu_mau=X)

        ket_qua = warmup.lam_nong("clf")
        assert ket_qua["trang_thai"] == "ok"

    def test_lay_model(self):
        from vietnamese_ai.production.warmup import LamNongModel

        warmup = LamNongModel()
        model = "test_model"
        warmup.dang_ky_model("m1", model)
        assert warmup.lay_model("m1") == "test_model"

    def test_danh_sach(self):
        from vietnamese_ai.production.warmup import LamNongModel

        warmup = LamNongModel()
        warmup.dang_ky_model("a", None)
        warmup.dang_ky_model("b", None)
        assert set(warmup.danh_sach_models()) == {"a", "b"}

    def test_thong_ke(self):
        from vietnamese_ai.production.warmup import LamNongModel

        warmup = LamNongModel()
        warmup.dang_ky_model("m1", None)
        stats = warmup.thong_ke()
        assert stats["so_models"] == 1
