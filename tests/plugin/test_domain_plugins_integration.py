"""
Kiểm thử tích hợp Hệ sinh thái V-Plugin 2.0 (Domain Plugins: Tài chính, Y tế, Quản lý Plugin, AST Sandbox, Agent & RAG).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.plugin.plugin_base import (
    PluginCongCu,
    PluginCoSo,
    PluginGiaiPhap,
    PluginTriThuc,
)
from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin
from vietnamese_ai.plugins.plugin_taichinh import PluginTaiChinh
from vietnamese_ai.plugins.plugin_yte import PluginYTe
from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.sandbox import LoiAnNinh


class TestPluginSDKCore:
    def test_khoi_tao_cac_loai_plugin(self):
        """Kiểm tra các lớp cơ sở PluginCoSo, PluginCongCu, PluginTriThuc, PluginGiaiPhap."""
        p_base = PluginCoSo()
        assert p_base.loai == "co_so"
        assert p_base.khoi_dong() is True
        assert p_base.trang_thai_hoat_dong is True
        p_base.tat()
        assert p_base.trang_thai_hoat_dong is False

        # PluginCongCu mẫu
        class DemoToolPlugin(PluginCongCu):
            ten = "Demo Tool"

            def dang_ky_cong_cu(self):
                return [
                    {
                        "ten": "cong_hai_so",
                        "mo_ta": "Cộng hai số",
                        "ham": lambda a, b: a + b,
                    }
                ]

        tool_p = DemoToolPlugin()
        assert tool_p.loai == "cong_cu"
        assert tool_p.khoi_dong() is True
        res = tool_p.thuc_thi_cong_cu("cong_hai_so", a=10, b=20)
        assert res == 30
        res_exec = tool_p.thuc_thi(action="cong_hai_so", a=5, b=15)
        assert res_exec["thanh_cong"] is True
        assert res_exec["ket_qua"] == 20

        # PluginTriThuc mẫu
        class DemoKnowledgePlugin(PluginTriThuc):
            ten = "Demo Knowledge"

            def nap_nguon_tri_thuc(self):
                return [{"tieu_de": "Tài liệu A", "noi_dung": "Nội dung kiến thức A"}]

            def lay_tu_dien_nghiep_vu(self):
                return {"AI": "Trí tuệ nhân tạo"}

        kn_p = DemoKnowledgePlugin()
        assert kn_p.loai == "tri_thuc"
        assert len(kn_p.nap_nguon_tri_thuc()) == 1
        assert kn_p.tra_cuu_nhanh("ai") == "Trí tuệ nhân tạo"

        # PluginGiaiPhap mẫu
        class DemoSolutionPlugin(PluginGiaiPhap):
            ten = "Demo Solution"

            def khoi_chay_giai_phap(self, **kwargs):
                return {"buoc": 1, "trang_thai": "hoan_thanh"}

        sol_p = DemoSolutionPlugin()
        assert sol_p.loai == "giai_phap"
        res_sol = sol_p.thuc_thi()
        assert res_sol["thanh_cong"] is True
        assert res_sol["ket_qua"]["trang_thai"] == "hoan_thanh"


class TestQuanLyPluginHotSwapAndSandbox:
    def test_nap_plugin_thanh_cong_kem_ast_sandbox(self):
        """Kiểm tra QuanLyPlugin nạp các plugin mẫu và vượt qua AST Sandbox."""
        ql = QuanLyPlugin()

        p_tc = ql.nap_plugin("plugins/plugin_taichinh.py")
        assert p_tc is not None
        assert p_tc.ten == "Tài Chính & Kế Toán"
        assert p_tc.trang_thai_hoat_dong is True

        p_yt = ql.nap_plugin("plugins/plugin_yte.py")
        assert p_yt is not None
        assert p_yt.ten == "Y Tế & Dược Khoa"

        # Kiểm tra phân loại plugin
        assert "plugin_taichinh" in ql.cac_plugin_cong_cu
        assert "plugin_taichinh" in ql.cac_plugin_tri_thuc
        assert "plugin_yte" in ql.cac_plugin_cong_cu
        assert "plugin_yte" in ql.cac_plugin_tri_thuc

        # Lấy toàn bộ công cụ và tri thức
        all_tools = ql.lay_tat_ca_cong_cu()
        assert "tinh_thue_tncn" in all_tools
        assert "tra_cuu_thuoc" in all_tools

        schemas = ql.lay_schema_tat_ca_cong_cu()
        assert len(schemas) >= 6

        all_docs = ql.lay_tat_ca_tri_thuc()
        assert len(all_docs) >= 5

        # Danh sách plugin
        ds = ql.danh_sach_plugin()
        assert len(ds) == 2

        # Gỡ plugin (Hot-Unload)
        ql.go_plugin("plugin_taichinh")
        assert "plugin_taichinh" not in ql.cac_plugin_dang_chay
        assert p_tc.trang_thai_hoat_dong is False

    def test_ast_sandbox_chan_ma_nguy_hiem(self):
        """Kiểm tra PhanTichAST chặn đứng các file plugin có hành vi nguy hiểm."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "plugin_doc_hai.py"
            bad_file.write_text(
                "import os\n"
                "from vietnamese_ai.plugin.plugin_base import PluginCoSo\n"
                "class BadPlugin(PluginCoSo):\n"
                "    def thuc_thi(self):\n"
                "        os.system('rm -rf /')\n",
                encoding="utf-8",
            )

            ql = QuanLyPlugin()
            hop_le, ly_do = ql.kiem_tra_an_ninh(str(bad_file))
            assert hop_le is False
            assert "os" in ly_do

            with pytest.raises(LoiAnNinh):
                ql.nap_plugin(str(bad_file))


class TestPluginTaiChinh:
    def test_tinh_thue_tncn_chinh_xac_7_bac(self):
        """Kiểm tra công thức thuế TNCN với các mức thu nhập khác nhau."""
        p = PluginTaiChinh()

        # Thu nhập 10 triệu (< 11 triệu giảm trừ bản thân) -> Thuế = 0
        kq1 = p.tinh_thue_tncn(thu_nhap_thang=10_000_000)
        assert kq1["thue_tncn_phai_nop"] == 0
        assert kq1["thu_nhap_thuc_nhan"] == 10_000_000

        # Thu nhập 25 triệu, 1 người phụ thuộc (Giảm trừ = 11M + 4.4M = 15.4M)
        # Thu nhập tính thuế = 9.6 triệu -> Bậc 2: 9.6M * 10% - 250k = 710k
        kq2 = p.tinh_thue_tncn(thu_nhap_thang=25_000_000, so_nguoi_phu_thuoc=1)
        assert kq2["thu_nhap_tinh_thue"] == 9_600_000
        assert kq2["bac_thue_cao_nhat"] == 2
        assert kq2["thue_tncn_phai_nop"] == 710_000
        assert kq2["thu_nhap_thuc_nhan"] == 24_290_000

        # Thu nhập 100 triệu, 0 người phụ thuộc
        # Giảm trừ = 11M -> TNTT = 89M -> Bậc 7: 89M * 35% - 9.85M = 21.3M
        kq3 = p.tinh_thue_tncn(thu_nhap_thang=100_000_000)
        assert kq3["bac_thue_cao_nhat"] == 7
        assert kq3["thue_tncn_phai_nop"] == 21_300_000

    def test_tinh_thue_tndn_va_khau_hao_tscd(self):
        """Kiểm tra thuế TNDN và trích khấu hao tài sản cố định."""
        p = PluginTaiChinh()

        # Thuế TNDN
        kq_tndn = p.tinh_thue_tndn(doanh_thu=1_000_000_000, chi_phi_hop_ly=600_000_000)
        assert kq_tndn["thu_nhap_tinh_thue"] == 400_000_000
        assert kq_tndn["thue_tndn_phai_nop"] == 80_000_000
        assert kq_tndn["loi_nhuan_sau_thue"] == 320_000_000

        # Khấu hao TSCĐ (Máy chủ 120 triệu trong 5 năm)
        kq_kh = p.tinh_khau_hao_tscd(nguyen_gia=120_000_000, so_nam_su_dung=5)
        assert kq_kh["muc_khau_hao_nam"] == 24_000_000
        assert kq_kh["muc_khau_hao_thang"] == 2_000_000

    def test_boc_tach_hoa_don_dien_tu(self):
        """Kiểm tra bóc tách hóa đơn điện tử."""
        p = PluginTaiChinh()
        raw_text = (
            "HÓA ĐƠN GIÁ TRỊ GIA TĂNG\n"
            "Số HĐ: HD-2026-00982\n"
            "Đơn vị bán: Công ty TNHH Trí Tuệ Nhân Tạo Việt\n"
            "MST: 0109887766\n"
            "Người mua: Nguyễn Văn A\n"
            "Tổng tiền thanh toán: 15.500.000 VNĐ\n"
        )
        kq = p.boc_tach_hoa_don(raw_text)
        assert kq["so_hoa_don"] == "HD-2026-00982"
        assert kq["mst_nguoi_ban"] == "0109887766"
        assert kq["tong_tien_thanh_toan"] == 15_500_000.0
        assert kq["hop_le"] is True


class TestPluginYTe:
    def test_tra_cuu_duoc_thu_quoc_gia(self):
        """Kiểm tra tra cứu hoạt chất trong Dược thư."""
        p = PluginYTe()

        para = p.tra_cuu_thuoc("Paracetamol")
        assert para["tim_thay"] is True
        assert "Acetaminophen" in para["hoat_chat"]
        assert "gan" in para["chong_chi_dinh"].lower()

        amox = p.tra_cuu_thuoc("amoxicillin 500mg")
        assert amox["tim_thay"] is True
        assert "Penicillin" in amox["nhom_duoc_ly"]

        khong_co = p.tra_cuu_thuoc("Thuốc Lạ Chưa Biết")
        assert khong_co["tim_thay"] is False

    def test_tra_ma_icd10(self):
        """Kiểm tra tra cứu danh mục ICD-10 tiếng Việt."""
        p = PluginYTe()

        kq_j00 = p.tra_ma_icd10("J00")
        assert len(kq_j00) >= 1
        assert "Cảm thường" in kq_j00[0]["ten"]

        kq_tieuduong = p.tra_ma_icd10("đái tháo đường")
        assert len(kq_tieuduong) >= 1
        assert kq_tieuduong[0]["ma"] == "E11"

    def test_sang_loc_trieu_chung_red_flags(self):
        """Kiểm tra thuật toán phân loại cảnh báo đỏ y tế."""
        p = PluginYTe()

        # Dấu hiệu nguy cấp
        kq_khan_cap = p.sang_loc_trieu_chung("Bệnh nhân đau thắt ngực trái và khó thở dữ dội")
        assert kq_khan_cap["muc_do_nghiem_trong"] == "KHẨN_CẤP"
        assert len(kq_khan_cap["dau_hieu_nguy_hiem"]) >= 2
        assert "115" in kq_khan_cap["khuyen_nghi_chuyen_mon"]

        # Dấu hiệu cần khám
        kq_can_kham = p.sang_loc_trieu_chung("Bé bị sốt nhẹ và ho kéo dài 3 ngày")
        assert kq_can_kham["muc_do_nghiem_trong"] == "CẦN_ĐI_KHÁM"

        # Dấu hiệu thông thường
        kq_nhe = p.sang_loc_trieu_chung("Hơi mỏi người do làm việc khuya")
        assert kq_nhe["muc_do_nghiem_trong"] == "THEO_DÕI_TẠI_NHÀ"


class TestTacTuPluginIntegration:
    def test_tac_tu_nap_plugin_va_goi_cong_cu(self):
        """Kiểm tra TacTu tự động nạp công cụ từ PluginTaiChinh và thực thi qua ReAct."""
        plugin = PluginTaiChinh()

        def llm_phan_hoi(prompt: str) -> str:
            # Mô phỏng LLM sinh lệnh ReAct gọi công cụ tinh_thue_tncn
            return (
                "Suy nghĩ: Cần tính thuế TNCN cho mức lương 25 triệu có 1 người phụ thuộc.\n"
                'Hành động: tinh_thue_tncn\n'
                'Tham số: {"thu_nhap_thang": 25000000, "so_nguoi_phu_thuoc": 1}'
            )

        tac_tu = TacTu(llm=llm_phan_hoi, max_iterations=2)
        so_cc = tac_tu.nap_plugin(plugin)
        assert so_cc == 4
        assert "tinh_thue_tncn" in tac_tu.cong_cu

        # Thực thi công cụ qua TacTu
        cc = tac_tu.cong_cu["tinh_thue_tncn"]
        res = cc.chay(thu_nhap_thang=25_000_000, so_nguoi_phu_thuoc=1)
        assert isinstance(res, dict)
        assert res["thue_tncn_phai_nop"] == 710_000


class TestRAGPipelinePluginIntegration:
    def test_rag_pipeline_nap_plugin_tri_thuc(self):
        """Kiểm tra RAGPipeline tự động nạp tri thức từ PluginYTe."""
        rag = RAGPipeline(bat_tuong_lua=False)
        plugin = PluginYTe()

        so_doc = rag.nap_plugin(plugin)
        assert so_doc == 3

        # Truy vấn thông tin vừa nạp
        kq = rag.tim_kiem("Paracetamol liều tối đa mỗi ngày là bao nhiêu?", top_k=2)
        assert len(kq) >= 1
        noi_dung_top = (kq[0].get("noi_dung") or kq[0].get("metadata", {}).get("noi_dung", "")).lower()
        assert "paracetamol" in noi_dung_top
        assert "4g" in noi_dung_top


class TestCLIPluginCommands:
    def test_cli_plugin_parser_va_commands(self):
        """Kiểm tra parser CLI vai plugin và các hành vi list, verify, info, run."""
        from vietnamese_ai.cli.main import _tao_parser, cmd_plugin

        parser = _tao_parser()
        args = parser.parse_args(["plugin", "list"])
        assert args.command == "plugin"
        assert args.action == "list"

        # 1. Test verify
        args_verify = parser.parse_args(["plugin", "verify", "plugins/plugin_taichinh.py"])
        kq_v = cmd_plugin(args_verify)
        assert kq_v["hop_le"] is True

        # 2. Test info
        args_info = parser.parse_args(["plugin", "info", "plugins/plugin_yte.py"])
        kq_info = cmd_plugin(args_info)
        assert kq_info["tim_thay"] is True
        assert kq_info["loai"] == "y_te"

        # 3. Test run tool
        args_run = parser.parse_args([
            "plugin", "run", "plugins/plugin_taichinh.py",
            "--tool", "tinh_thue_tncn",
            "--params", '{"thu_nhap_thang": 30000000, "so_nguoi_phu_thuoc": 0}',
        ])
        kq_run = cmd_plugin(args_run)
        assert kq_run["thanh_cong"] is True
        assert kq_run["ket_qua"]["thue_tncn_phai_nop"] > 0
