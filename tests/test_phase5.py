"""Test suite cho Phase 5 - Mobile/Edge, NAS, Federated Learning, Real-time Pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from vietnamese_ai.datasets.sample_data import DuLieuMau
from vietnamese_ai.models.classifier import PhanLoai

# ============================================================
# Phase 5.1: Mobile/Edge Deployment - TriKhaiDiDong
# ============================================================


class TestTriKhaiDiDong:
    def setup_method(self):
        from vietnamese_ai.mobile.deployment import TriKhaiDiDong

        self.tkdd = TriKhaiDiDong()
        self.X, self.y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        self.mo_hinh = PhanLoai(thuat_toan="logistic")
        self.mo_hinh.huan_luyen(self.X, self.y)

    def test_khoi_tao(self):
        assert self.tkdd is not None
        assert len(self.tkdd.DINH_DANG_HO_TRO) == 3

    def test_xuat_tflite(self, tmp_path):
        duong_dan = str(tmp_path / "model.tflite")
        ket_qua = self.tkdd.xuat_tflite(
            self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,)
        )
        assert Path(ket_qua).exists()
        assert Path(ket_qua).stat().st_size > 0

    def test_xuat_tflite_with_wrapper(self, tmp_path):
        duong_dan = str(tmp_path / "model.tflite")
        ket_qua = self.tkdd.xuat_tflite(self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,))
        assert Path(ket_qua).exists()

    def test_xuat_tflite_quantized(self, tmp_path):
        duong_dan = str(tmp_path / "model.tflite")
        ket_qua = self.tkdd.xuat_tflite(
            self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,), luong_hoa=True
        )
        assert Path(ket_qua).exists()

    def test_xuat_coreml(self, tmp_path):
        duong_dan = str(tmp_path / "model.coreml")
        ket_qua = self.tkdd.xuat_coreml(
            self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,)
        )
        assert Path(ket_qua).exists()

    def test_xuat_coreml_custom_name(self, tmp_path):
        duong_dan = str(tmp_path / "model.coreml")
        ket_qua = self.tkdd.xuat_coreml(
            self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,), ten_mo_hinh="MyModel"
        )
        with open(ket_qua, "rb") as f:
            magic = f.read(7)
            assert magic == b"VAI_CLM"

    def test_xuat_onnx_mobile_fallback(self, tmp_path):
        duong_dan = str(tmp_path / "model.onnx")
        ket_qua = self.tkdd.xuat_onnx_mobile(
            self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,)
        )
        assert Path(ket_qua).exists()

    def test_luong_hoa_int8(self, tmp_path):
        duong_dan_goc = str(tmp_path / "model.tflite")
        duong_dan_moi = str(tmp_path / "model_int8.tflite")

        self.tkdd.xuat_tflite(self.mo_hinh, duong_dan_goc, kich_thuoc_dau_vao=(5,))
        ket_qua = self.tkdd.luong_hoa_int8(duong_dan_goc, duong_dan_moi)
        assert Path(ket_qua).exists()

    def test_luong_hoa_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.tkdd.luong_hoa_int8("/nonexistent/file", str(tmp_path / "out.tflite"))

    def test_doc_mo_hinh_di_dong(self, tmp_path):
        duong_dan = str(tmp_path / "model.tflite")
        self.tkdd.xuat_tflite(self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,))
        metadata = self.tkdd.doc_mo_hinh_di_dong(duong_dan)
        assert "dinh_dang_phat_hien" in metadata
        assert metadata["dinh_dang_phat_hien"] == "tflite"
        assert "loai_mo_hinh" in metadata

    def test_doc_mo_hinh_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.tkdd.doc_mo_hinh_di_dong("/nonexistent/file")

    def test_benchmark_edge(self):
        ket_qua = self.tkdd.benchmark_edge(
            self.mo_hinh, kich_thuoc_dau_vao=(10, 5), so_lan=5
        )
        assert "thoi_gian_trung_binh_ms" in ket_qua
        assert "throughput_mau_giay" in ket_qua
        assert ket_qua["thoi_gian_trung_binh_ms"] > 0
        assert ket_qua["so_lan_chay"] == 5

    def test_benchmark_invalid_so_lan(self):
        with pytest.raises(ValueError):
            self.tkdd.benchmark_edge(self.mo_hinh, (1, 5), so_lan=0)

    def test_tao_config_deploy(self, tmp_path):
        duong_dan = str(tmp_path / "mobile")
        ket_qua = self.tkdd.tao_config_deploy("my_model", "tflite", duong_dan)
        assert Path(ket_qua).exists()
        config_file = Path(ket_qua) / "mobile_config.json"
        assert config_file.exists()
        with open(config_file) as f:
            config = json.load(f)
        assert config["ten_mo_hinh"] == "my_model"
        assert config["dinh_dang"] == "tflite"

    def test_tao_config_invalid_format(self, tmp_path):
        with pytest.raises(ValueError):
            self.tkdd.tao_config_deploy("model", "invalid", str(tmp_path))

    def test_du_lieu_dau_vao_khong_hop_le(self, tmp_path):
        with pytest.raises(ValueError):
            self.tkdd.xuat_tflite(self.mo_hinh, "", kich_thuoc_dau_vao=(5,))

        with pytest.raises(ValueError):
            self.tkdd.xuat_tflite(self.mo_hinh, str(tmp_path / "m"), kich_thuoc_dau_vao=())

    def test_roundtrip_quantize(self, tmp_path):
        duong_dan_goc = str(tmp_path / "model.tflite")
        duong_dan_quant = str(tmp_path / "model_q.tflite")

        self.tkdd.xuat_tflite(self.mo_hinh, duong_dan_goc, kich_thuoc_dau_vao=(5,))
        self.tkdd.luong_hoa_int8(duong_dan_goc, duong_dan_quant)

        meta = self.tkdd.doc_mo_hinh_di_dong(duong_dan_quant)
        assert meta.get("luong_hoa") is True

    def test_magic_bytes_tflite(self, tmp_path):
        duong_dan = str(tmp_path / "model.tflite")
        self.tkdd.xuat_tflite(self.mo_hinh, duong_dan, kich_thuoc_dau_vao=(5,))
        with open(duong_dan, "rb") as f:
            magic = f.read(7)
        assert magic == b"VAI_TFL"


# ============================================================
# Phase 5.2: NAS - TimKiemKienTruc
# ============================================================


class TestTimKiemKienTruc:
    def setup_method(self):

        self.X, self.y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=3)

    def test_khoi_tao(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(so_fold=3)
        assert tkt.so_fold == 3
        assert tkt.seed == 42

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        with pytest.raises(ValueError):
            TimKiemKienTruc(so_fold=1)

    def test_tim_kiem_ngau_nhien(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(so_fold=3, seed=42, diem_toi_thieu=0.3)
        pham_vi = {
            "so_lop_an": [1, 2],
            "so_neron_lop": [8, 16],
            "ham_kich_hoat": ["relu"],
            "so_vong": [10],
        }
        ket_qua = tkt.tim_kiem_ngau_nhien(self.X, self.y, pham_vi=pham_vi, so_lan=3)
        assert "kien_truc_tot_nhat" in ket_qua
        assert "diem_tot_nhat" in ket_qua
        assert ket_qua["so_lan_thu"] == 3
        assert ket_qua["tong_thoi_gian"] > 0

    def test_tim_kiem_luoi(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(so_fold=2, seed=42)
        luoi = {
            "so_lop_an": [1],
            "so_neron_lop": [8],
            "ham_kich_hoat": ["relu"],
            "so_vong": [10],
        }
        ket_qua = tkt.tim_kiem_luoi(self.X, self.y, luoi_tham_so=luoi)
        assert ket_qua["so_lan_thu"] == 1
        assert ket_qua["kien_truc_tot_nhat"] is not None

    def test_bao_cao_chua_tim_kiem(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(so_fold=2)
        bc = tkt.bao_cao()
        assert "Chưa có kết quả" in bc

    def test_bao_cao_sau_tim_kiem(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(so_fold=2, seed=42, diem_toi_thieu=0.3)
        pham_vi = {"so_lop_an": [1], "so_neron_lop": [8], "ham_kich_hoat": ["relu"], "so_vong": [10]}
        tkt.tim_kiem_ngau_nhien(self.X, self.y, pham_vi=pham_vi, so_lan=2)
        bc = tkt.bao_cao()
        assert "NEURAL ARCHITECTURE SEARCH" in bc

    def test_so_sanh_voi_ml_truyen_thong(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(so_fold=2, seed=42, diem_toi_thieu=0.3)
        pham_vi = {"so_lop_an": [1], "so_neron_lop": [8], "ham_kich_hoat": ["relu"], "so_vong": [10]}
        tkt.tim_kiem_ngau_nhien(self.X, self.y, pham_vi=pham_vi, so_lan=2)
        ket_qua = tkt.so_sanh_voi_ml_truyen_thong(self.X, self.y)
        assert "nas" in ket_qua
        assert "truyen_thong" in ket_qua
        assert len(ket_qua["truyen_thong"]) >= 3

    def test_toi_da_hoa_do_phuc_tap(self):
        from vietnamese_ai.automl.nas import TimKiemKienTruc

        tkt = TimKiemKienTruc(
            so_fold=2, seed=42, diem_toi_thieu=0.3, toi_da_hoa_do_phuc_tap=True
        )
        pham_vi = {
            "so_lop_an": [1, 2],
            "so_neron_lop": [8, 16],
            "ham_kich_hoat": ["relu"],
            "so_vong": [10],
        }
        ket_qua = tkt.tim_kiem_ngau_nhien(self.X, self.y, pham_vi=pham_vi, so_lan=3)
        assert ket_qua["kien_truc_tot_nhat"] is not None


# ============================================================
# Phase 5.3: Federated Learning - HocLienKet
# ============================================================


class TestHocLienKet:
    def setup_method(self):
        self.X, self.y = DuLieuMau.phan_loai_don_gian(so_mau=200, so_dac_trung=5)

    def test_khoi_tao(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=2)
        assert hl.so_client == 3
        assert hl.so_vong == 2

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.federated.learning import HocLienKet

        with pytest.raises(ValueError):
            HocLienKet(so_client=1)
        with pytest.raises(ValueError):
            HocLienKet(so_vong=0)
        with pytest.raises(ValueError):
            HocLienKet(ty_le_client=0)
        with pytest.raises(ValueError):
            HocLienKet(rieng_tu_differntial=-1)

    def test_huan_luyen(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=2, seed=42)
        ket_qua = hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        assert "trong_so_toan_cuc" in ket_qua
        assert "diem_toan_cuc" in ket_qua
        assert ket_qua["diem_toan_cuc"] > 0
        assert ket_qua["so_client"] == 3
        assert ket_qua["so_vong"] == 2
        assert len(ket_qua["lich_su"]) == 2

    def test_huan_luyen_du_lieu_khong_hop_le(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=2, so_vong=1)
        with pytest.raises(ValueError):
            hl.huan_luyen(PhanLoai, np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_du_doan(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=2, seed=42)
        hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        du_doan = hl.du_doan(PhanLoai, self.X[:10], thuat_toan="logistic")
        assert len(du_doan) == 10

    def test_du_doan_chua_huan_luyen(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=2, so_vong=1)
        with pytest.raises(RuntimeError):
            hl.du_doan(PhanLoai, self.X[:5], thuat_toan="logistic")

    def test_lich_su(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=5, seed=42)
        hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        lich_su = hl.lay_lich_su()
        assert len(lich_su) == 5
        for ban_ghi in lich_su:
            assert "vong" in ban_ghi
            assert "diem_toan_cuc" in ban_ghi
            assert "diem_tb_client" in ban_ghi

    def test_bao_cao_chua_huan_luyen(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=2, so_vong=1)
        assert "Chưa có kết quả" in hl.bao_cao()

    def test_bao_cao_sau_huan_luyen(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=2, seed=42)
        hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        bc = hl.bao_cao()
        assert "FEDERATED" in bc

    def test_differential_privacy(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(
            so_client=3, so_vong=2, rieng_tu_differntial=0.1, seed=42
        )
        ket_qua = hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        assert ket_qua["diem_toan_cuc"] > 0

    def test_partial_client_sampling(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=5, so_vong=3, ty_le_client=0.6, seed=42)
        ket_qua = hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        assert len(ket_qua["lich_su"]) == 3

    def test_client_lich_su_chi_tiet(self):
        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=2, ty_le_client=1.0, seed=42)
        ket_qua = hl.huan_luyen(PhanLoai, self.X, self.y, thuat_toan="logistic")
        for ban_ghi in ket_qua["lich_su"]:
            assert ban_ghi["so_client_tham_gia"] == 3
            for ct in ban_ghi["chi_tiet_clients"]:
                assert "client_id" in ct
                assert "diem" in ct


# ============================================================
# Phase 5.4: Real-time Pipeline - PipelineThoiGianThuc
# ============================================================


class TestPipelineThoiGianThuc:
    def setup_method(self):

        self.X, self.y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        self.mo_hinh = PhanLoai(thuat_toan="logistic")
        self.mo_hinh.huan_luyen(self.X, self.y)

    def test_khoi_tao(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        assert p.ten == "PipelineThoiGianThuc"
        assert p.batch_size == 32

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        with pytest.raises(ValueError):
            PipelineThoiGianThuc(kich_thuoc_buffer=0)
        with pytest.raises(ValueError):
            PipelineThoiGianThuc(batch_size=-1)

    def test_dang_ky_mo_hinh(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        assert "test" in p._mo_hinh

    def test_dang_ky_none(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        with pytest.raises(ValueError):
            p.dang_ky_mo_hinh("test", None)

    def test_gui_du_lieu(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        ket_qua = p.gui_du_lieu("test", self.X[0])
        assert ket_qua["trang_thai"] == "da_nhan"

    def test_gui_du_lieu_chua_dang_ky(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        with pytest.raises(KeyError):
            p.gui_du_lieu("nonexistent", self.X[0])

    def test_du_doan(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        ket_qua = p.du_doan("test", self.X[0])
        assert ket_qua["trang_thai"] == "thanh_cong"
        assert ket_qua["latency_ms"] > 0
        assert ket_qua["ket_qua"] is not None

    def test_du_doan_batch(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        ket_qua = p.du_doan_batch("test", self.X[:10])
        assert ket_qua["trang_thai"] == "thanh_cong"
        assert ket_qua["so_mau"] == 10
        assert "throughput_mau_giay" in ket_qua

    def test_du_doan_loi(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        ket_qua = p.du_doan("test", np.array([]))
        assert ket_qua["trang_thai"] == "loi"

    def test_thong_ke(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        for i in range(5):
            p.du_doan("test", self.X[i])
        tk = p.lay_thong_ke()
        assert tk["so_du_doan"] == 5
        assert tk["so_mo_hinh"] == 1
        assert "latency_trung_binh_ms" in tk
        assert "latency_p95_ms" in tk
        assert "queue" in tk

    def test_lich_su(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        for i in range(5):
            p.du_doan("test", self.X[i])
        lich_su = p.lay_lich_su("test", so_luong=5)
        assert len(lich_su) > 0

    def test_xoa_buffer(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        p.du_doan("test", self.X[0])
        p.xoa_buffer()
        tk = p.lay_thong_ke()
        assert tk["so_du_doan"] == 0

    def test_callback(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)

        results = []
        p.dang_ky_callback("test", lambda msg: results.append(msg))
        p.du_doan("test", self.X[0])
        assert len(results) > 0

    def test_repr(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("m1", self.mo_hinh)
        r = repr(p)
        assert "PipelineThoiGianThuc" in r
        assert "mo_hinh=1" in r

    def test_multiple_models(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        m1 = PhanLoai(thuat_toan="logistic")
        m1.huan_luyen(self.X, self.y)
        m2 = PhanLoai(thuat_toan="knn")
        m2.huan_luyen(self.X, self.y)

        p.dang_ky_mo_hinh("logistic", m1)
        p.dang_ky_mo_hinh("knn", m2)

        r1 = p.du_doan("logistic", self.X[0])
        r2 = p.du_doan("knn", self.X[0])
        assert r1["trang_thai"] == "thanh_cong"
        assert r2["trang_thai"] == "thanh_cong"

    def test_gui_du_lieu_voi_nhan(self):
        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc()
        p.dang_ky_mo_hinh("test", self.mo_hinh)
        for i in range(5):
            p.gui_du_lieu("test", self.X[i], nhan=self.y[i])
        tk = p.lay_thong_ke()
        assert tk["feature_store"]["so_features"] >= 1


# ============================================================
# Tests cho MessageQueue và FeatureStore
# ============================================================


class TestMessageQueue:
    def test_publish_consume(self):
        from vietnamese_ai.realtime.pipeline import MessageQueue

        mq = MessageQueue("test")
        mq.publish("topic1", {"data": [1, 2, 3]})
        mq.publish("topic1", {"data": [4, 5, 6]})
        mq.publish("topic2", {"data": [7]})

        tin_nhan = mq.consume("topic1", so_luong=10)
        assert len(tin_nhan) == 2

    def test_subscribe(self):
        from vietnamese_ai.realtime.pipeline import MessageQueue

        mq = MessageQueue("test")
        results = []
        mq.subscribe("test_topic", lambda msg: results.append(msg))
        mq.publish("test_topic", {"value": 42})
        assert len(results) == 1
        assert results[0]["du_lieu"]["value"] == 42

    def test_thong_ke(self):
        from vietnamese_ai.realtime.pipeline import MessageQueue

        mq = MessageQueue("test")
        mq.publish("t1", "data1")
        mq.publish("t2", "data2")
        tk = mq.lay_thong_ke()
        assert tk["tong_tin_nhan"] == 2

    def test_xoa(self):
        from vietnamese_ai.realtime.pipeline import MessageQueue

        mq = MessageQueue("test")
        mq.publish("t1", "data")
        mq.xoa()
        tk = mq.lay_thong_ke()
        assert tk["tong_tin_nhan"] == 0

    def test_backpressure(self):
        from vietnamese_ai.realtime.pipeline import MessageQueue

        mq = MessageQueue("test", kich_thuoc_toi_da=5)
        for i in range(10):
            mq.publish("t", f"data_{i}")
        tk = mq.lay_thong_ke()
        assert tk["trong_queue"] <= 5


class TestFeatureStore:
    def test_cap_nhat_va_lay(self):
        from vietnamese_ai.realtime.pipeline import FeatureStore

        fs = FeatureStore()
        for i in range(10):
            fs.cap_nhat("feature1", np.array([i, i + 1]), nhan=i % 2)

        features = fs.lay_features("feature1")
        assert features.shape == (10, 2)

        labels = fs.lay_labels("feature1")
        assert len(labels) == 10

    def test_cua_so(self):
        from vietnamese_ai.realtime.pipeline import FeatureStore

        fs = FeatureStore()
        for i in range(20):
            fs.cap_nhat("f1", np.array([i]))

        window = fs.lay_window("f1", kich_thuoc=5)
        assert len(window) == 5

    def test_khong_ton_tai(self):
        from vietnamese_ai.realtime.pipeline import FeatureStore

        fs = FeatureStore()
        with pytest.raises(KeyError):
            fs.lay_features("nonexistent")

    def test_thong_ke(self):
        from vietnamese_ai.realtime.pipeline import FeatureStore

        fs = FeatureStore()
        fs.cap_nhat("f1", np.array([1, 2]))
        fs.cap_nhat("f2", np.array([3]))
        tk = fs.thong_ke()
        assert tk["so_features"] == 2


# ============================================================
# Integration test: import từ __init__
# ============================================================


class TestPhase5Integration:
    def test_imports(self):
        from vietnamese_ai import HocLienKet, PipelineThoiGianThuc, TimKiemKienTruc, TriKhaiDiDong

        assert TriKhaiDiDong is not None
        assert TimKiemKienTruc is not None
        assert HocLienKet is not None
        assert PipelineThoiGianThuc is not None

    def test_version(self):
        import vietnamese_ai

        assert vietnamese_ai.__version__ == "9.0.0"

    def test_all_exports_count(self):
        import vietnamese_ai

        assert len(vietnamese_ai.__all__) >= 42

    def test_end_to_end_mobile(self, tmp_path):
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=4)
        pl = PhanLoai(thuat_toan="rung_ngau_nhien")
        pl.huan_luyen(X, y)

        from vietnamese_ai.mobile.deployment import TriKhaiDiDong

        tkdd = TriKhaiDiDong()

        p_tflite = tkdd.xuat_tflite(pl, str(tmp_path / "m.tflite"), kich_thuoc_dau_vao=(4,))
        p_coreml = tkdd.xuat_coreml(pl, str(tmp_path / "m.coreml"), kich_thuoc_dau_vao=(4,))

        assert Path(p_tflite).exists()
        assert Path(p_coreml).exists()

        meta = tkdd.doc_mo_hinh_di_dong(p_tflite)
        assert meta["dinh_dang_phat_hien"] == "tflite"

        bm = tkdd.benchmark_edge(pl, kich_thuoc_dau_vao=(5, 4), so_lan=3)
        assert bm["thoi_gian_trung_binh_ms"] > 0

    def test_end_to_end_federated(self):
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=150, so_dac_trung=3)

        from vietnamese_ai.federated.learning import HocLienKet

        hl = HocLienKet(so_client=3, so_vong=3, seed=42)
        ket_qua = hl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
        assert ket_qua["diem_toan_cuc"] > 0

        du_doan = hl.du_doan(PhanLoai, X[:5], thuat_toan="logistic")
        assert len(du_doan) == 5

    def test_end_to_end_realtime(self):
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc

        p = PipelineThoiGianThuc(kich_thuoc_buffer=500)
        p.dang_ky_mo_hinh("clf", pl)

        for i in range(20):
            result = p.du_doan("clf", X[i])
            assert result["trang_thai"] == "thanh_cong"

        batch = p.du_doan_batch("clf", X[:20])
        assert batch["so_mau"] == 20

        tk = p.lay_thong_ke()
        assert tk["so_du_doan"] == 40  # 20 single + 20 batch
