import json
import os
import tempfile

import pytest

from vietnamese_ai.datalake import DocDuLieuStream, HoDuLieu


def tao_file_jsonl_gia(duong_dan: str, so_dong: int):
    """Tạo một file JSONL giả lập."""
    with open(duong_dan, "w", encoding="utf-8") as f:
        for i in range(so_dong):
            data = {"id": i, "van_ban": f"Câu số {i}", "nhan": i % 2}
            f.write(json.dumps(data) + "\n")


def test_doc_du_lieu_stream():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "test_data.jsonl")
        tao_file_jsonl_gia(file_path, 2500)

        # Đọc theo lô 1000
        luong = DocDuLieuStream.doc_jsonl(file_path, kich_thuoc_lo=1000)

        lo_1 = next(luong)
        assert len(lo_1) == 1000
        assert lo_1[0]["id"] == 0

        lo_2 = next(luong)
        assert len(lo_2) == 1000
        assert lo_2[0]["id"] == 1000

        lo_3 = next(luong)
        assert len(lo_3) == 500
        assert lo_3[0]["id"] == 2000

        with pytest.raises(StopIteration):
            next(luong)


def test_ho_du_lieu_va_dvc():
    with tempfile.TemporaryDirectory() as tmp_dir:
        thu_muc_datalake = os.path.join(tmp_dir, "my_datalake")
        ho_du_lieu = HoDuLieu(thu_muc_datalake)

        # 1. Tạo file nguồn (Phiên bản 1)
        file_nguon = os.path.join(tmp_dir, "raw_data.jsonl")
        tao_file_jsonl_gia(file_nguon, 100)

        # 2. Nhập vào hồ (Tạo Hash)
        hash_v1 = ho_du_lieu.nhap_du_lieu(file_nguon, "my_dataset.jsonl", "Bản gốc 100 dòng")
        assert len(hash_v1) == 64  # SHA256

        # Đọc thử từ hồ
        luong_1 = ho_du_lieu.doc_luong_du_lieu("my_dataset.jsonl", kich_thuoc_lo=200)
        lo_1 = next(luong_1)
        assert len(lo_1) == 100

        # 3. Ghi đè file nguồn (Phiên bản 2)
        tao_file_jsonl_gia(file_nguon, 50)

        # Nhập lại vào hồ (Tạo Hash mới)
        hash_v2 = ho_du_lieu.nhap_du_lieu(file_nguon, "my_dataset.jsonl", "Cắt giảm còn 50 dòng")
        assert hash_v1 != hash_v2

        # Đọc thử từ hồ xem đã là 50 chưa
        luong_2 = ho_du_lieu.doc_luong_du_lieu("my_dataset.jsonl", kich_thuoc_lo=200)
        lo_2 = next(luong_2)
        assert len(lo_2) == 50

        # 4. Phục hồi (Checkout) về phiên bản 1
        ho_du_lieu.phuc_hoi_phien_ban("my_dataset.jsonl", hash_v1)

        # Đọc lại từ hồ xem có về 100 không
        luong_3 = ho_du_lieu.doc_luong_du_lieu("my_dataset.jsonl", kich_thuoc_lo=200)
        lo_3 = next(luong_3)
        assert len(lo_3) == 100

        # 5. Kiểm tra lịch sử
        lich_su = ho_du_lieu.xem_lich_su()
        assert len(lich_su) == 2
        assert lich_su[0]["tin_nhan"] == "Bản gốc 100 dòng"
        assert lich_su[1]["tin_nhan"] == "Cắt giảm còn 50 dòng"
