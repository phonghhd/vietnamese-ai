"""Test suite cho Phase 6 - SaaS Platform, No-code Studio, Vietnamese LLM."""

import json
from pathlib import Path

import pytest

from vietnamese_ai.datalake.sample_data import DuLieuMau
from vietnamese_ai.models.classifier import PhanLoai

# ============================================================
# Phase 6.1: SaaS Platform - NenTangDichVu
# ============================================================


class TestNenTangDichVu:
    def setup_method(self):
        from vietnamese_ai.enterprise.platform import NenTangDichVu

        self.ntdv = NenTangDichVu

    def test_khoi_tao(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        assert ntdv is not None

    def test_tao_workspace(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Test Org", "admin")
        assert ws["ten"] == "Test Org"
        assert ws["chu_so_huu"] == "admin"
        assert ws["goi_dich_vu"] == "free"
        assert ws["ma"] is not None

    def test_tao_workspace_invalid_goi(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        with pytest.raises(ValueError):
            ntdv.tao_workspace("Org", "user", goi_dich_vu="invalid")

    def test_tao_workspace_ten_rong(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        with pytest.raises(ValueError):
            ntdv.tao_workspace("", "user")

    def test_lay_workspace(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        ws_info = ntdv.lay_workspace(ws["ma"])
        assert ws_info["ten"] == "Org"

    def test_lay_workspace_khong_ton_tai(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        with pytest.raises(KeyError):
            ntdv.lay_workspace("nonexistent")

    def test_danh_sach_workspaces(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ntdv.tao_workspace("Org1", "user1")
        ntdv.tao_workspace("Org2", "user2")
        ds = ntdv.danh_sach_workspaces()
        assert len(ds) == 2

    def test_cap_nhat_goi(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        ws_updated = ntdv.cap_nhat_goi(ws["ma"], "pro")
        assert ws_updated["goi_dich_vu"] == "pro"

    def test_cap_nhat_goi_invalid(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        with pytest.raises(ValueError):
            ntdv.cap_nhat_goi(ws["ma"], "invalid")

    def test_tao_api_key(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        api_key = ntdv.tao_api_key(ws["ma"])
        assert api_key.startswith("vai_")
        assert len(api_key) > 10

    def test_xac_thuc_api_key(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        api_key = ntdv.tao_api_key(ws["ma"])
        ws_info = ntdv.xac_thuc_api_key(api_key)
        assert ws_info["ma"] == ws["ma"]

    def test_xac_thuc_api_key_invalid(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        with pytest.raises(PermissionError):
            ntdv.xac_thuc_api_key("invalid_key")

    def test_quota_api_keys_free(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin", goi_dich_vu="free")
        ntdv.tao_api_key(ws["ma"])
        with pytest.raises(PermissionError, match="giới hạn"):
            ntdv.tao_api_key(ws["ma"])

    def test_dang_ky_model(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        model_info = ntdv.dang_ky_model(ws["ma"], "test_model", pl, mo_ta="Test")
        assert model_info["ten"] == "test_model"
        assert model_info["ma"] is not None

    def test_quota_models_free(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin", goi_dich_vu="free")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        for i in range(3):
            pl = PhanLoai(thuat_toan="logistic")
            pl.huan_luyen(X, y)
            ntdv.dang_ky_model(ws["ma"], f"model_{i}", pl)

        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)
        with pytest.raises(PermissionError, match="giới hạn"):
            ntdv.dang_ky_model(ws["ma"], "model_extra", pl)

    def test_lay_model(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        model_info = ntdv.dang_ky_model(ws["ma"], "clf", pl)
        mo_hinh = ntdv.lay_model(ws["ma"], model_info["ma"])
        assert mo_hinh.da_huan_luyen

    def test_deploy_model(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        model_info = ntdv.dang_ky_model(ws["ma"], "clf", pl)
        deploy = ntdv.deploy_model(ws["ma"], model_info["ma"])
        assert deploy["trang_thai"] == "hoat_dong"

    def test_du_doan(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50, so_dac_trung=3)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        model_info = ntdv.dang_ky_model(ws["ma"], "clf", pl)
        deploy = ntdv.deploy_model(ws["ma"], model_info["ma"])

        ket_qua = ntdv.du_doan(ws["ma"], deploy["ma"], X[:5])
        assert "ket_qua" in ket_qua
        assert len(ket_qua["ket_qua"]) == 5
        assert ket_qua["so_requests"] == 1

    def test_thong_ke_usage(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        tk = ntdv.thong_ke_usage(ws["ma"])
        assert "usage" in tk
        assert "gioi_han" in tk

    def test_xoa_workspace(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        ws = ntdv.tao_workspace("Org", "admin")
        ntdv.xoa_workspace(ws["ma"])
        assert len(ntdv.danh_sach_workspaces()) == 0

    def test_lay_goi_dich_vu(self, tmp_path):
        ntdv = self.ntdv(str(tmp_path / "saas"))
        goi = ntdv.lay_goi_dich_vu()
        assert "free" in goi
        assert "pro" in goi
        assert "enterprise" in goi

    def test_persistence(self, tmp_path):
        duong_dan = str(tmp_path / "saas")
        ntdv1 = self.ntdv(duong_dan)
        ntdv1.tao_workspace("Org1", "user1")
        ntdv1.tao_workspace("Org2", "user2")

        ntdv2 = self.ntdv(duong_dan)
        assert len(ntdv2.danh_sach_workspaces()) == 2


# ============================================================
# Phase 6.2: No-code Studio - StudioKeoTha
# ============================================================


class TestStudioKeoTha:
    def test_khoi_tao(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        assert studio.ten == "Vietnamese AI Studio"

    def test_them_node(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        node = studio.them_node("du_lieu", "Dữ liệu mẫu")
        assert node["ma"] is not None
        assert node["loai"] == "du_lieu"

    def test_them_node_invalid(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        with pytest.raises(ValueError):
            studio.them_node("invalid_loai", "Test")

    def test_xoa_node(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        node = studio.them_node("du_lieu", "Data")
        studio.xoa_node(node["ma"])
        canvas = studio.lay_canvas()
        assert len(canvas["nodes"]) == 0

    def test_sua_node(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        node = studio.them_node("mo_hinh", "Model", tham_so={"thuat_toan": "logistic"})
        studio.sua_node(node["ma"], {"thuat_toan": "knn"})
        canvas = studio.lay_canvas()
        assert canvas["nodes"][node["ma"]]["tham_so"]["thuat_toan"] == "knn"

    def test_ket_noi(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        n1 = studio.them_node("du_lieu", "Data")
        n2 = studio.them_node("mo_hinh", "Model")
        kn = studio.ket_noi(n1["ma"], n2["ma"])
        assert kn["tu_node"] == n1["ma"]
        assert kn["den_node"] == n2["ma"]

    def test_ket_noi_invalid(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        n1 = studio.them_node("du_lieu", "Data")
        with pytest.raises(KeyError):
            studio.ket_noi(n1["ma"], "nonexistent")

    def test_ket_noi_trung(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        n1 = studio.them_node("du_lieu", "Data")
        n2 = studio.them_node("mo_hinh", "Model")
        studio.ket_noi(n1["ma"], n2["ma"])
        with pytest.raises(ValueError, match="đã tồn tại"):
            studio.ket_noi(n1["ma"], n2["ma"])

    def test_ket_noi_chinh_no(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        n1 = studio.them_node("du_lieu", "Data")
        with pytest.raises(ValueError, match="chính nó"):
            studio.ket_noi(n1["ma"], n1["ma"])

    def test_huy_ket_noi(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        n1 = studio.them_node("du_lieu", "Data")
        n2 = studio.them_node("mo_hinh", "Model")
        studio.ket_noi(n1["ma"], n2["ma"])
        studio.huy_ket_noi(n1["ma"], n2["ma"])
        canvas = studio.lay_canvas()
        assert len(canvas["ket_noi"]) == 0

    def test_chay_pipeline_don_gian(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        n_data = studio.them_node("du_lieu", "Data", tham_so={"so_mau": 100, "so_dac_trung": 3})
        n_model = studio.them_node("mo_hinh", "Logistic", tham_so={"thuat_toan": "logistic"})
        n_eval = studio.them_node("danh_gia", "Đánh giá")
        studio.ket_noi(n_data["ma"], n_model["ma"])
        studio.ket_noi(n_model["ma"], n_eval["ma"])

        ket_qua = studio.chay()
        assert ket_qua["trang_thai"] == "thanh_cong"
        assert ket_qua["so_node"] == 3
        assert ket_qua["tong_thoi_gian"] > 0

    def test_chay_pipeline_voi_tien_xu_ly(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        studio.them_node("du_lieu", "Data", tham_so={"so_mau": 100})
        studio.them_node("tien_xu_ly", "Z-Score", tham_so={"phuong_phap": "zscore"})
        studio.them_node("mo_hinh", "RF", tham_so={"thuat_toan": "rung_ngau_nhien"})
        studio.them_node("danh_gia", "Eval")

        canvas = studio.lay_canvas()
        node_ids = list(canvas["nodes"].keys())
        for i in range(len(node_ids) - 1):
            studio.ket_noi(node_ids[i], node_ids[i + 1])

        ket_qua = studio.chay()
        assert ket_qua["trang_thai"] == "thanh_cong"

    def test_chay_pipeline_trong(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        with pytest.raises(RuntimeError, match="trống"):
            studio.chay()

    def test_lay_canvas(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        studio.them_node("du_lieu", "Data")
        studio.them_node("mo_hinh", "Model")
        canvas = studio.lay_canvas()
        assert len(canvas["nodes"]) == 2
        assert "ket_noi" in canvas

    def test_danh_sach_loai_node(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        loai = studio.danh_sach_loai_node()
        assert "du_lieu" in loai
        assert "mo_hinh" in loai
        assert "danh_gia" in loai

    def test_danh_sach_thuat_toan(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        tt = studio.danh_sach_thuat_toan()
        assert "phan_loai" in tt
        assert "logistic" in tt["phan_loai"]

    def test_templates(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        templates = studio.lay_templates()
        assert "phan_loai_co_ban" in templates
        assert "hoi_quy_nang_cao" in templates

    def test_tai_template(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        canvas = studio.tai_template("phan_loai_co_ban")
        assert len(canvas["nodes"]) == 4
        assert len(canvas["ket_noi"]) == 3

    def test_tai_template_invalid(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        with pytest.raises(KeyError):
            studio.tai_template("nonexistent")

    def test_template_chay(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        studio.tai_template("phan_loai_co_ban")
        ket_qua = studio.chay()
        assert ket_qua["trang_thai"] == "thanh_cong"

    def test_luu_va_tai(self, tmp_path):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        studio.tai_template("phan_loai_co_ban")
        duong_dan = str(tmp_path / "pipeline.json")
        studio.luu(duong_dan)
        assert Path(duong_dan).exists()

        studio2 = StudioKeoTha.tai(duong_dan)
        canvas = studio2.lay_canvas()
        assert len(canvas["nodes"]) == 4

    def test_luu_config_format(self, tmp_path):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        studio.them_node("du_lieu", "Data")
        duong_dan = str(tmp_path / "config.json")
        studio.luu(duong_dan)

        with open(duong_dan, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert "ten" in config
        assert "nodes" in config
        assert "ket_noi" in config


# ============================================================
# Phase 6.3: Vietnamese LLM - VietnameseLLM
# ============================================================


class TestVietnameseLLM:
    CORPUS = [
        "học máy là một nhánh của trí tuệ nhân tạo",
        "trí tuệ nhân tạo đang phát triển rất nhanh",
        "mạng nơ-ron nhân tạo mô phỏng não bộ con người",
        "học sâu là một kỹ thuật mạnh mẽ trong học máy",
        "xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng Việt",
        "học máy rất thú vị và hữu ích cho cuộc sống",
        "trí tuệ nhân tạo thay đổi cách con người làm việc",
        "mạng nơ-ron rất mạnh mẽ cho bài toán nhận dạng",
        "học sâu đã đạt được nhiều thành tựu lớn",
        "xử lý ngôn ngữ tự nhiên là lĩnh vực hấp dẫn",
    ] * 5

    def test_khoi_tao(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=3)
        assert llm.bac == 3

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        with pytest.raises(ValueError):
            VietnameseLLM(bac=1)
        with pytest.raises(ValueError):
            VietnameseLLM(lam_mo=-1)

    def test_huan_luyen(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        ket_qua = llm.huan_luyen(self.CORPUS)
        assert ket_qua["vocab_size"] > 0
        assert ket_qua["tong_tu"] > 0
        assert llm._da_huan_luyen

    def test_huan_luyen_rong(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        with pytest.raises(ValueError):
            llm.huan_luyen([])

    def test_sinh_van_ban(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        van_ban = llm.sinh_van_ban("học máy", do_dai=10)
        assert isinstance(van_ban, str)
        assert len(van_ban) > 0

    def test_sinh_van_ban_chua_huan_luyen(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        with pytest.raises(RuntimeError):
            llm.sinh_van_ban("test")

    def test_sinh_van_ban_nhiet_do(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        vb_conservative = llm.sinh_van_ban("học", do_dai=10, nhiet_do=0.1)
        vb_creative = llm.sinh_van_ban("học", do_dai=10, nhiet_do=2.0)
        assert isinstance(vb_conservative, str)
        assert isinstance(vb_creative, str)

    def test_hoan_thanh_cau(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        lua_chon = llm.hoan_thanh_cau("học máy", so_lua_chon=3, do_dai_toi_da=5)
        assert len(lua_chon) == 3
        for lc in lua_chon:
            assert "van_ban" in lc
            assert "perplexity" in lc

    def test_perplexity(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        ppl = llm.tinh_perplexity("học máy rất hay")
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perplexity_chua_huan_luyen(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        with pytest.raises(RuntimeError):
            llm.tinh_perplexity("test")

    def test_tu_ke_tiep(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        goi_y = llm.lay_tu_ke_tiep("học máy", top_n=3)
        assert isinstance(goi_y, list)
        for gy in goi_y:
            assert "tu" in gy
            assert "xac_suat" in gy

    def test_template(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        van_ban = llm.sinh_theo_template(
            "tin_tuc",
            {"chu_de": "AI đang phát triển mạnh", "nhan_dinh": "đây là xu hướng tất yếu"},
        )
        assert "AI đang phát triển mạnh" in van_ban
        assert "xu hướng tất yếu" in van_ban

    def test_template_invalid(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        with pytest.raises(KeyError):
            llm.sinh_theo_template("nonexistent", {})

    def test_them_template(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        llm.them_template("custom", "Xin chào {ten}!")
        van_ban = llm.sinh_theo_template("custom", {"ten": "Việt Nam"})
        assert van_ban == "Xin chào Việt Nam!"

    def test_them_template_invalid(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        with pytest.raises(ValueError):
            llm.them_template("", "template")
        with pytest.raises(ValueError):
            llm.them_template("ten", "")

    def test_danh_sach_templates(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        templates = llm.danh_sach_templates()
        assert "tin_tuc" in templates
        assert "san_pham" in templates
        assert len(templates) >= 5

    def test_thong_ke(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        tk = llm.thong_ke()
        assert tk["da_huan_luyen"] is True
        assert tk["bac"] == 2
        assert tk["vocab_size"] > 0
        assert tk["tong_ngrams"] > 0

    def test_luu_tai(self, tmp_path):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)

        duong_dan = str(tmp_path / "llm.json")
        llm.luu(duong_dan)
        assert Path(duong_dan).exists()

        llm2 = VietnameseLLM.tai(duong_dan)
        assert llm2._da_huan_luyen
        assert llm2.bac == 2
        assert llm2._vocab_size == llm._vocab_size

        van_ban = llm2.sinh_van_ban("học", do_dai=5)
        assert isinstance(van_ban, str)

    def test_luu_chua_huan_luyen(self, tmp_path):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2)
        with pytest.raises(RuntimeError):
            llm.luu(str(tmp_path / "llm.json"))

    def test_bigram(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        van_ban = llm.sinh_van_ban("trí tuệ", do_dai=10)
        assert isinstance(van_ban, str)

    def test_trigram(self):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        llm = VietnameseLLM(bac=3, toi_thieu_dem=1)
        llm.huan_luyen(self.CORPUS)
        van_ban = llm.sinh_van_ban("học máy", do_dai=10)
        assert isinstance(van_ban, str)


# ============================================================
# Integration tests
# ============================================================


class TestPhase6Integration:
    def test_imports(self):
        from vietnamese_ai import NenTangDichVu, StudioKeoTha, VietnameseLLM

        assert NenTangDichVu is not None
        assert StudioKeoTha is not None
        assert VietnameseLLM is not None

    def test_version(self):
        import vietnamese_ai

        assert vietnamese_ai.__version__ == "11.0.1"

    def test_all_exports_count(self):
        import vietnamese_ai

        assert len(vietnamese_ai.__all__) >= 45

    def test_end_to_end_saas(self, tmp_path):
        from vietnamese_ai.enterprise.platform import NenTangDichVu

        ntdv = NenTangDichVu(str(tmp_path / "saas"))

        ws = ntdv.tao_workspace("AI Startup", "founder", goi_dich_vu="starter")
        api_key = ntdv.tao_api_key(ws["ma"])
        ws_check = ntdv.xac_thuc_api_key(api_key)
        assert ws_check["ma"] == ws["ma"]

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=3)
        pl = PhanLoai(thuat_toan="rung_ngau_nhien")
        pl.huan_luyen(X, y)

        model_info = ntdv.dang_ky_model(ws["ma"], "sentiment", pl)
        deploy = ntdv.deploy_model(ws["ma"], model_info["ma"])

        ket_qua = ntdv.du_doan(ws["ma"], deploy["ma"], X[:5])
        assert len(ket_qua["ket_qua"]) == 5

        tk = ntdv.thong_ke_usage(ws["ma"])
        assert tk["usage"]["predictions"] == 1

    def test_end_to_end_studio(self):
        from vietnamese_ai.studio.builder import StudioKeoTha

        studio = StudioKeoTha()
        studio.tai_template("phan_loai_co_ban")
        ket_qua = studio.chay()
        assert ket_qua["trang_thai"] == "thanh_cong"

    def test_end_to_end_llm(self, tmp_path):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        corpus = [
            "học máy là xu hướng công nghệ mới",
            "trí tuệ nhân tạo giúp cuộc sống tốt đẹp hơn",
            "mạng nơ-ron mô phỏng cách não bộ hoạt động",
            "học sâu đạt nhiều thành tựu trong nhận dạng hình ảnh",
            "xử lý ngôn ngữ tự nhiên rất quan trọng",
        ] * 10

        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        ket_qua = llm.huan_luyen(corpus)
        assert ket_qua["vocab_size"] > 0

        van_ban = llm.sinh_van_ban("học", do_dai=15)
        assert isinstance(van_ban, str)
        assert len(van_ban) > 0

        goi_y = llm.lay_tu_ke_tiep("học máy", top_n=3)
        assert len(goi_y) > 0

        duong_dan = str(tmp_path / "llm.json")
        llm.luu(duong_dan)
        llm2 = VietnameseLLM.tai(duong_dan)
        van_ban2 = llm2.sinh_van_ban("trí tuệ", do_dai=10)
        assert isinstance(van_ban2, str)
