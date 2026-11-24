"""Kiểm thử toàn diện Giai đoạn 2: Core Differentiation & Strategic Development."""

from __future__ import annotations

import json
import secrets
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_tai_luoi_cac_subpackage_giai_doan_2():
    """Đảm bảo các namespace mới tải lười chuẩn PEP 562 và không nạp sẵn numpy/sklearn/torch."""
    code = (
        "import sys\n"
        "import vietnamese_ai.nlp\n"
        "import vietnamese_ai.edge\n"
        "import vietnamese_ai.rag.legal\n"
        "import vietnamese_ai.legal\n"
        "import vietnamese_ai.legal.rag\n"
        "import vietnamese_ai.crypto\n"
        "import vietnamese_ai.zkml\n"
        "import vietnamese_ai.classic_ml\n"
        "heavy = [m for m in ('numpy', 'sklearn', 'torch', 'transformers') if m in sys.modules]\n"
        "assert not heavy, f'Nap som goi nang: {heavy}'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_anh_xa_namespace_giai_doan_2():
    """Kiểm tra tính đồng nhất biểu tượng giữa các sub-package tách mới và module gốc."""
    from vietnamese_ai.classic_ml import PhanLoai as ClassicPhanLoai
    from vietnamese_ai.classic_ml import thong_tin_goi_classic_ml
    from vietnamese_ai.crypto import (
        CayMerkleSuyLuan as CryptoMerkle,
    )
    from vietnamese_ai.crypto import (
        KiemToanMatMa,
    )
    from vietnamese_ai.crypto import (
        XacThucSuyLuanMaoHiem as CryptoVerifier,
    )
    from vietnamese_ai.edge import BoGiaiLuongTuGGUF, BoGiaiLuongTuQ4, KiemTraMoiTruongNative
    from vietnamese_ai.legal.rag import (
        BoDocTaiLieuPhapLy,
        BoMauPromptPhapLy,
        BoToiUuNguCanhHanhChinh,
    )
    from vietnamese_ai.models import PhanLoai as ModelsPhanLoai
    from vietnamese_ai.nlp import GanNhanTuLoai, XuLyVanBan, kiem_tra_phu_thuoc_nlp
    from vietnamese_ai.preprocessing.text import XuLyVanBan as OrigXuLyVanBan
    from vietnamese_ai.rag.legal import BoToiUuNguCanhHanhChinh as RagLegalAdmin
    from vietnamese_ai.zkml import XacThucSuyLuanMaoHiem as ZkmlVerifier

    assert ClassicPhanLoai is ModelsPhanLoai
    assert CryptoVerifier is ZkmlVerifier
    assert BoToiUuNguCanhHanhChinh is RagLegalAdmin
    assert XuLyVanBan is OrigXuLyVanBan
    assert BoGiaiLuongTuGGUF is BoGiaiLuongTuQ4
    assert CryptoMerkle is not None
    assert KiemToanMatMa is not None
    assert KiemTraMoiTruongNative is not None
    assert BoDocTaiLieuPhapLy is not None
    assert BoMauPromptPhapLy is not None
    assert GanNhanTuLoai is not None

    nlp_deps = kiem_tra_phu_thuoc_nlp()
    assert "underthesea" in nlp_deps
    classic_info = thong_tin_goi_classic_ml()
    assert classic_info["extra_cai_dat"] == "classic-ml"


def test_nlp_chuan_hoa_dau_giu_chu_hoa_va_pos_tagger():
    """Kiểm tra chuẩn hóa dấu bảo toàn viết hoa/thường và bộ gán nhãn từ loại VLSP."""
    from vietnamese_ai.nlp import BoTachTuTiengVietSieuToc, GanNhanTuLoai

    tok = BoTachTuTiengVietSieuToc()
    chuan = tok.chuan_hoa_dau_thanh("Toà án và Uỷ ban HOÀ giải tuỳ chọn sức khoẻ.")
    assert "Tòa án" in chuan
    assert "Ủy ban" in chuan
    assert "HÒA giải" in chuan
    assert "tùy chọn" in chuan
    assert "khỏe" in chuan

    pos = GanNhanTuLoai(su_dung_underthesea=False)
    tags = pos.gan_nhan("Chính phủ đã ban hành Nghị định 13 rất quan trọng tại Hà Nội.")
    tag_map = {tu.lower(): nhan for tu, nhan in tags}
    assert tag_map["đã"] == "R"
    assert tag_map["ban hành"] == "V"
    assert tag_map["13"] == "M"
    assert tag_map["rất"] == "R"
    assert tag_map["quan trọng"] == "A"
    assert tag_map["tại"] == "E"
    assert tag_map["hà nội"] == "Np"

    chi_tiet = pos.gan_nhan_chi_tiet("Học máy rất hay.")
    assert all("mo_ta" in item for item in chi_tiet)
    assert pos.thong_ke()["backend"] == "native_vlsp_rules"


def test_nlp_ner_chuyen_sau_va_phobert_phan_loai():
    """Kiểm tra NER mở rộng (TO_CHUC, VAN_BAN_PHAP_LUAT, CCCD, MA_SO_THUE, NGUOI) và PhoBERTWrapper."""
    from vietnamese_ai.nlp import NhanDienThucThe, PhoBERTWrapper

    ner = NhanDienThucThe(su_dung_underthesea=False)
    van_ban = (
        "Giám đốc Nguyễn Minh Khôi tại Công ty Cổ phần FPT (MST: 0101248141, CCCD số 001095012345) "
        "áp dụng Nghị định số 13/2023/NĐ-CP tại Hà Nội."
    )
    thuc_the = ner.nhan_dien(van_ban)
    loai_tim_thay = {item["loai"] for item in thuc_the}
    assert "CHUC_DANH" in loai_tim_thay
    assert "NGUOI" in loai_tim_thay
    assert "TO_CHUC" in loai_tim_thay
    assert "MA_SO_THUE" in loai_tim_thay
    assert "CCCD" in loai_tim_thay
    assert "VAN_BAN_PHAP_LUAT" in loai_tim_thay
    assert "DIA_DANH" in loai_tim_thay

    phobert = PhoBERTWrapper(che_do_mo_phong=True, phuong_phap_gop="mean")
    assert "trí_tuệ_nhân_tạo" in phobert.tien_xu_ly_tieng_viet("Trí tuệ nhân tạo Việt Nam").lower()

    with pytest.raises(ValueError, match="phuong_phap_gop"):
        PhoBERTWrapper(phuong_phap_gop="khong_hop_le")

    ds_train = [
        "Nghị định bảo vệ dữ liệu cá nhân và xử phạt vi phạm hành chính",
        "Hợp đồng lao động quy định quyền lợi nghĩa vụ và bồi thường",
        "Thông tư hướng dẫn thi hành luật doanh nghiệp và mã số thuế",
        "Đội tuyển bóng đá thắng lớn với bàn thắng đẹp mắt trên sân vận động",
        "Giải vô địch cầu lông và bóng chuyền thu hút đông đảo khán giả",
        "Trận đấu thể thao kết thúc với tỷ số hòa kịch tính",
    ]
    ds_nhan = ["phap_ly", "phap_ly", "phap_ly", "the_thao", "the_thao", "the_thao"]
    kq_train = phobert.huan_luyen_dau_phan_loai(ds_train, ds_nhan, he_so_chinh_quy=0.5)
    assert kq_train["do_chinh_xac_huan_luyen"] >= 0.99

    du_doan = phobert.du_doan_phan_loai(
        [
            "Nghị định xử phạt vi phạm hành chính về dữ liệu cá nhân",
            "Trận đấu bóng đá hấp dẫn trên sân vận động",
        ]
    )
    assert du_doan[0]["nhan"] == "phap_ly"
    assert du_doan[1]["nhan"] == "the_thao"
    assert abs(sum(du_doan[0]["xac_suat"].values()) - 1.0) < 1e-6


def test_legal_rag_da_loai_van_ban_doc_tep_va_mau_prompt(tmp_path: Path):
    """Kiểm tra RAG hành chính/pháp lý mở rộng: phân loại văn bản, Mục/Phụ lục, OCR/JSON và mẫu prompt."""
    from vietnamese_ai.legal.rag import (
        BoDocTaiLieuPhapLy,
        BoMauPromptPhapLy,
        BoToiUuNguCanhHanhChinh,
    )
    from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh

    cong_van = (
        "BỘ THÔNG TIN VÀ TRUYỀN THÔNG\n"
        "Số: 456/BTTTT-CATTT\n"
        "Hà Nội, ngày 15 tháng 03 năm 2025\n"
        "V/v: Hướng dẫn đảm bảo an toàn thông tin hệ thống AI\n"
        "Kính gửi: Các đơn vị trực thuộc\n"
        "I. YÊU CẦU CHUNG\n"
        "Các hệ thống trí tuệ nhân tạo phải tuân thủ Nghị định số 13/2023/NĐ-CP.\n"
        "II. TỔ CHỨC THỰC HIỆN\n"
        "Báo cáo kết quả rà soát trước ngày 30/06/2025.\n"
        "Phụ lục I: Bảng kiểm an toàn mô hình\n"
        "Kiểm tra tường lửa prompt injection và bảo vệ dữ liệu cá nhân.\n"
    )
    rag = BoToiUuNguCanhHanhChinh(bat_tuong_lua=True)
    meta = rag.trich_xuat_metadata_hanh_chinh(cong_van)
    assert meta["loai_van_ban"] == "Công văn"
    assert "BỘ THÔNG TIN VÀ TRUYỀN THÔNG" in meta["co_quan_ban_hanh"]
    assert "15 tháng 03 năm 2025" in meta["ngay_ban_hanh"]
    assert "Hướng dẫn đảm bảo an toàn thông tin" in meta["trich_yeu"]

    chunks_cv = rag.phan_doan_giu_ngu_canh(cong_van, ten_tai_lieu="Công văn 456")
    assert len(chunks_cv) >= 3
    assert any(c["metadata"]["loai_don_vi"] == "phu_luc" for c in chunks_cv)

    txt_file = tmp_path / "nghi_dinh_13.txt"
    txt_file.write_text(
        "Trang 1/2\n"
        "Chương I. QUY ĐỊNH CHUNG\n"
        "Mục 1. Bảo vệ dữ liệu cá nhân\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "Nghị định này quy định về bảo vệ dữ liệu cá nhân và trách nhiệm của cơ quan.\n"
        "- 2 -\n"
        "Điều 2. Quyền của chủ thể dữ liệu\n"
        "1. Quyền được biết về hoạt động xử lý dữ liệu cá nhân.\n",
        encoding="utf-8",
    )
    kq_nap = rag.nap_tu_tep(txt_file, ten_tai_lieu="Nghị định 13")
    assert kq_nap["dinh_dang"] == "van_ban"
    assert any("Mục 1" in c["ngu_canh_day_du"] for c in kq_nap["chunks"])
    assert all("Trang 1/2" not in c["noi_dung"] for c in kq_nap["chunks"])

    ket_qua = rag.truy_hoi_phap_ly("Quyền của chủ thể dữ liệu là gì?", kq_nap["chunks"], top_k=2)
    prompt = rag.tao_prompt_phap_ly("Quyền của chủ thể dữ liệu là gì?", ket_qua)
    assert "Điều 2" in prompt
    assert "Quyền của chủ thể dữ liệu là gì?" in prompt

    json_file = tmp_path / "hop_dong.json"
    json_file.write_text(
        json.dumps(
            {
                "tieu_de": "Hợp đồng cung cấp dịch vụ phần mềm",
                "noi_dung": "Điều 1. Giá trị hợp đồng là 500.000.000 đồng.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    parser = BoDocTaiLieuPhapLy()
    kq_json = parser.doc_tep(json_file)
    assert kq_json["dinh_dang"] == "json"
    assert "500.000.000 đồng" in kq_json["van_ban"]

    img_file = tmp_path / "scan_dieu_1.png"
    TrichXuatVanBanAnh.tao_anh_van_ban(
        "Điều 1. Quy định bảo mật dữ liệu hành chính",
        duong_dan_luu=img_file,
    )
    kq_img = parser.doc_tep(img_file)
    assert kq_img["dinh_dang"] == "anh_ocr"
    assert "Điều 1" in kq_img["van_ban"]

    thu_vien = BoMauPromptPhapLy()
    assert len(thu_vien.danh_sach_mau()) == 6
    prompt_hd = thu_vien.tao_prompt(
        "ra_soat_hop_dong",
        noi_dung_hop_dong="Bên B chịu phạt 20% nếu giao chậm.",
    )
    assert "Bên B chịu phạt 20%" in prompt_hd


def test_edge_dequant_q4_1_q2_k_q3_k_q4_k_va_tiled_matmul():
    """Kiểm tra chính xác số học các kernel giải lượng tử Q4_1, Q2_K, Q3_K, Q4_K và nhân ma trận phân ô."""
    from vietnamese_ai.edge import BoGiaiLuongTuQ4, KiemTraMoiTruongNative, thong_tin_kien_truc_cpu

    d_q41 = np.float16(0.5).tobytes()
    m_q41 = np.float16(-1.0).tobytes()
    qs_q41 = bytes([0xA2 for _ in range(16)])
    out_q41 = BoGiaiLuongTuQ4.giai_q4_1(d_q41 + m_q41 + qs_q41, so_phan_tu=32)
    assert np.allclose(out_q41[:16], 2 * 0.5 - 1.0, atol=1e-4)
    assert np.allclose(out_q41[16:], 10 * 0.5 - 1.0, atol=1e-4)

    scales_q2k = bytes([0x21 for _ in range(16)])
    qs_q2k = bytes([0b11_10_01_00 for _ in range(64)])
    d_q2k = np.float16(1.0).tobytes()
    dmin_q2k = np.float16(0.5).tobytes()
    out_q2k = BoGiaiLuongTuQ4.giai_q2_k(scales_q2k + qs_q2k + d_q2k + dmin_q2k, so_phan_tu=256)
    assert out_q2k.shape == (256,)
    assert np.allclose(out_q2k[0:32], 0.0 * 1.0 - 2.0 * 0.5, atol=1e-4)
    assert np.allclose(out_q2k[32:64], 1.0 * 1.0 - 2.0 * 0.5, atol=1e-4)
    assert np.allclose(out_q2k[64:96], 2.0 * 1.0 - 2.0 * 0.5, atol=1e-4)
    assert np.allclose(out_q2k[96:128], 3.0 * 1.0 - 2.0 * 0.5, atol=1e-4)

    hmask_q3k = bytes([0xFF for _ in range(32)])
    qs_q3k = bytes([0b01_01_01_01 for _ in range(64)])
    s_raw_q3k = bytearray(12)
    s_raw_q3k[0:4] = bytes([0x22] * 4)
    s_raw_q3k[4:8] = bytes([0x22] * 4)
    s_raw_q3k[8:12] = bytes([0xAA] * 4)
    d_q3k = np.float16(0.5).tobytes()
    out_q3k = BoGiaiLuongTuQ4.giai_q3_k(hmask_q3k + qs_q3k + bytes(s_raw_q3k) + d_q3k, so_phan_tu=256)
    assert out_q3k.shape == (256,)
    assert np.allclose(out_q3k, 1.0 * (34 - 32) * 0.5, atol=1e-4)

    d_q4k = np.float16(0.25).tobytes()
    dmin_q4k = np.float16(0.5).tobytes()
    sc_q4k = bytes([4, 4, 4, 4, 2, 2, 2, 2, 0x24, 0x24, 0x24, 0x24])
    qs_q4k = bytes([0x53 for _ in range(128)])
    block_q4k = d_q4k + dmin_q4k + sc_q4k + qs_q4k
    out_q4k = BoGiaiLuongTuQ4.giai_q4_k(block_q4k, so_phan_tu=256)
    assert out_q4k.shape == (256,)
    assert np.allclose(out_q4k[0:32], 3 * (0.25 * 4) - (0.5 * 2), atol=1e-4)
    assert np.allclose(out_q4k[32:64], 5 * (0.25 * 4) - (0.5 * 2), atol=1e-4)

    raw_w = block_q4k * 4
    w_dense = BoGiaiLuongTuQ4.giai_luong_tu(raw_w, 4 * 256, "q4_k_m").reshape(4, 256)
    x = np.random.RandomState(7).randn(2, 256).astype(np.float32)
    y_tiled = BoGiaiLuongTuQ4.nhan_ma_tran_luong_tu(
        x, raw_w, shape_w=(4, 256), loai_quant="q4_k_m", kich_thuoc_tile=2
    )
    assert np.allclose(y_tiled, x @ w_dense.T, atol=1e-4)

    cpu_info = thong_tin_kien_truc_cpu()
    assert cpu_info["so_loi_logic"] >= 1
    assert len(cpu_info["co_bien_dich_khuyen_nghi"]) >= 1
    tc = KiemTraMoiTruongNative.kiem_tra_toolchain()
    assert "trinh_bien_dich" in tc
    ext_cfg = KiemTraMoiTruongNative.cau_hinh_setuptools_extension("edge_kernel", ["kernel.cpp"])
    assert ext_cfg["language"] == "c++"
    assert ext_cfg["sources"] == ["kernel.cpp"]


def test_edge_gguf_multi_architecture_metadata(tmp_path: Path):
    """Kiểm tra GGUFMemoryMappedStreamer đọc metadata đa kiến trúc (qwen2)."""
    from vietnamese_ai.edge import GGUFMemoryMappedStreamer

    gguf_file = tmp_path / "qwen2_vietnamese.gguf"
    kv_pairs = [
        ("general.name", 8, "Qwen2-Vi-Edge"),
        ("general.architecture", 8, "qwen2"),
        ("qwen2.context_length", 4, 8192),
        ("qwen2.block_count", 4, 28),
        ("qwen2.embedding_length", 4, 3584),
        ("qwen2.attention.head_count", 4, 28),
        ("qwen2.attention.head_count_kv", 4, 4),
    ]
    with open(gguf_file, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 10))
        f.write(struct.pack("<Q", len(kv_pairs)))
        for key, vtype, val in kv_pairs:
            kb = key.encode("utf-8")
            f.write(struct.pack("<Q", len(kb)))
            f.write(kb)
            f.write(struct.pack("<I", vtype))
            if vtype == 8:
                vb = str(val).encode("utf-8")
                f.write(struct.pack("<Q", len(vb)))
                f.write(vb)
            else:
                f.write(struct.pack("<I", int(val)))

    streamer = GGUFMemoryMappedStreamer(gguf_file)
    try:
        info = streamer.lay_thong_tin()
        assert info["kien_truc"] == "qwen2"
        assert info["kien_truc_duoc_ho_tro"] is True
        assert info["context_length"] == 8192
        assert info["block_count"] == 28
        assert info["embedding_length"] == 3584
        assert info["head_count"] == 28
        assert info["head_count_kv"] == 4
    finally:
        streamer.close()


def test_zkml_rfc5114_xac_thuc_nghiem_ngat_va_tu_kiem_toan():
    """Kiểm tra nâng cấp mật mã học RFC 5114, từ chối giả mạo Fiat-Shamir và KiemToanMatMa."""
    from vietnamese_ai.crypto import (
        THONG_SO_NHOM_SCHNORR,
        KiemToanMatMa,
        XacThucSuyLuanMaoHiem,
    )

    assert THONG_SO_NHOM_SCHNORR["do_dai_p_bit"] == 2048
    assert THONG_SO_NHOM_SCHNORR["do_dai_q_bit"] == 256

    bao_cao = KiemToanMatMa.chay_kiem_toan_tu_dong()
    assert bao_cao["tong_quan_dat"] is True
    assert bao_cao["so_hang_muc_dat"] == bao_cao["tong_hang_muc"]

    v = XacThucSuyLuanMaoHiem()
    arr = np.array([0.5, -1.25, 2.0, 4.5], dtype=np.float32)
    ck = v.tao_cam_ket_zkml(arr, layer_id=3, node_id="worker_vn")
    assert XacThucSuyLuanMaoHiem.xac_thuc_cam_ket_nghiem_ngat(ck, tensor=arr) is True

    arr_sai = arr.copy()
    arr_sai[0] += 0.01
    assert XacThucSuyLuanMaoHiem.xac_thuc_cam_ket_nghiem_ngat(ck, tensor=arr_sai) is False

    p = int(THONG_SO_NHOM_SCHNORR["p"])
    q = int(THONG_SO_NHOM_SCHNORR["q"])
    g = int(THONG_SO_NHOM_SCHNORR["g"])
    z_fake = secrets.randbelow(q - 1) + 1
    e_fake = secrets.randbelow(q - 1) + 1
    a_fake = (pow(g, z_fake, p) * pow(v.khoa_cong_khai, q - e_fake, p)) % p
    assert (
        XacThucSuyLuanMaoHiem.xac_thuc_cam_ket(
            y=v.khoa_cong_khai,
            a=a_fake,
            e=e_fake,
            z=z_fake,
            tensor_hash=ck["tensor_hash"],
        )
        is False
    )

    layer_ck = v.sinh_cam_ket_layer("node_a", 0, arr, arr * 2.0)
    assert XacThucSuyLuanMaoHiem.xac_minh_cam_ket(layer_ck, arr, arr * 2.0) is True

    v_khac = XacThucSuyLuanMaoHiem()
    layer_swap = dict(layer_ck)
    layer_swap["public_key"] = v_khac.public_key
    assert XacThucSuyLuanMaoHiem.xac_minh_cam_ket(layer_swap, arr, arr * 2.0) is False


def test_khong_xung_dot_extra_giai_doan_2_va_ban_do_modular():
    """Kiểm tra extra classic-ml, legal-rag đồng bộ không xung đột và cập nhật BAN_DO_MODULAR."""
    from vietnamese_ai.utils import BAN_DO_MODULAR, QuanLyPhuThuoc

    kq = QuanLyPhuThuoc.kiem_tra_xung_dot_phu_thuoc()
    assert kq["hop_le"] is True
    assert kq["so_xung_dot"] == 0
    assert "classic-ml" in QuanLyPhuThuoc.NHOM_PHU_THUOC
    assert "legal-rag" in QuanLyPhuThuoc.NHOM_PHU_THUOC

    exts = BAN_DO_MODULAR["extensions"]
    assert "classic_ml" in exts
    assert "legal_rag" in exts
    assert "crypto_zkml" in exts
