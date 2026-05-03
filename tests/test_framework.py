"""Test suite cho Vietnamese AI Framework."""

import numpy as np
import pytest

from vietnamese_ai.core.engine import Engine
from vietnamese_ai.core.pipeline import Pipeline
from vietnamese_ai.datasets.sample_data import DuLieuMau
from vietnamese_ai.models.classifier import PhanLoai
from vietnamese_ai.models.clustering import PhanCum
from vietnamese_ai.models.neural_net import MangNron
from vietnamese_ai.models.regression import HoiQuy
from vietnamese_ai.preprocessing.feature_engineering import TaoDacTrung
from vietnamese_ai.preprocessing.numerical import XuLySo
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.metrics import Metrics
from vietnamese_ai.utils.validators import Validator


class TestPhanLoai:
    def setup_method(self):
        self.X, self.y = DuLieuMau.phan_loai_don_gian(so_mau=200)

    def test_logistic(self):
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(self.X, self.y)
        assert pl.da_huan_luyen
        du_doan = pl.du_doan(self.X)
        assert len(du_doan) == len(self.y)

    def test_knn(self):
        pl = PhanLoai(thuat_toan="knn")
        pl.huan_luyen(self.X, self.y)
        assert pl.da_huan_luyen

    def test_svm(self):
        pl = PhanLoai(thuat_toan="svm")
        pl.huan_luyen(self.X, self.y)
        diem = pl.danh_gia(self.X, self.y)
        assert diem > 0.5

    def test_rung_ngau_nhien(self):
        pl = PhanLoai(thuat_toan="rung_ngau_nhien")
        pl.huan_luyen(self.X, self.y)
        diem = pl.danh_gia(self.X, self.y)
        assert diem > 0.8

    def test_bao_cao(self):
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(self.X, self.y)
        bc = pl.bao_cao(self.X, self.y)
        assert "do_chinh_xac" in bc
        assert "f1" in bc

    def test_thuat_toan_khong_hop_le(self):
        with pytest.raises(ValueError):
            PhanLoai(thuat_toan="khong_co")

    def test_du_doan_chua_huan_luyen(self):
        pl = PhanLoai()
        with pytest.raises(RuntimeError):
            pl.du_doan(self.X)

    def test_luu_tai(self, tmp_path):
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(self.X, self.y)
        duong_dan = str(tmp_path / "model.pkl")
        pl.luu(duong_dan)
        pl2 = PhanLoai.tai(duong_dan)
        assert np.array_equal(pl.du_doan(self.X), pl2.du_doan(self.X))


class TestHoiQuy:
    def setup_method(self):
        self.X, self.y = DuLieuMau.hoi_quy_don_gian(so_mau=100)

    def test_tuyen_tinh(self):
        hq = HoiQuy(thuat_toan="tuyen_tinh")
        hq.huan_luyen(self.X, self.y)
        assert hq.da_huan_luyen
        du_doan = hq.du_doan(self.X)
        assert len(du_doan) == len(self.y)

    def test_ridge(self):
        hq = HoiQuy(thuat_toan="ridge")
        hq.huan_luyen(self.X, self.y)
        mse = hq.danh_gia(self.X, self.y)
        assert mse < 10.0

    def test_bao_cao(self):
        hq = HoiQuy(thuat_toan="tuyen_tinh")
        hq.huan_luyen(self.X, self.y)
        bc = hq.bao_cao(self.X, self.y)
        assert "mse" in bc
        assert "r2" in bc


class TestPhanCum:
    def setup_method(self):
        self.X, self.y = DuLieuMau.phan_cum_don_gian(so_mau=300)

    def test_kmeans(self):
        pc = PhanCum(so_cum=3, thuat_toan="kmeans")
        pc.huan_luyen(self.X)
        assert pc.da_huan_luyen
        nhan = pc.du_doan(self.X)
        assert len(nhan) == len(self.X)

    def test_danh_gia(self):
        pc = PhanCum(so_cum=3)
        pc.huan_luyen(self.X)
        diem = pc.danh_gia(self.X)
        assert -1 <= diem <= 1

    def test_lay_tam_cum(self):
        pc = PhanCum(so_cum=3)
        pc.huan_luyen(self.X)
        tam = pc.lay_tam_cum()
        assert tam is not None
        assert tam.shape == (3, 2)


class TestMangNron:
    def setup_method(self):
        self.X, self.y = DuLieuMau.phan_loai_don_gian(so_mau=200, so_dac_trung=5)

    def test_huan_luyen(self):
        mang = MangNron(lop_an=[16, 8], so_vong=50)
        mang.huan_luyen(self.X, self.y)
        assert mang.da_huan_luyen

    def test_du_doan(self):
        mang = MangNron(lop_an=[16, 8], so_vong=50)
        mang.huan_luyen(self.X, self.y)
        du_doan = mang.du_doan(self.X)
        assert len(du_doan) == len(self.y)
        assert all(d in [0, 1] for d in du_doan)

    def test_danh_gia(self):
        mang = MangNron(lop_an=[32, 16], so_vong=100)
        mang.huan_luyen(self.X, self.y)
        diem = mang.danh_gia(self.X, self.y)
        assert diem > 0.5

    def test_lich_su_loss(self):
        mang = MangNron(lop_an=[16], so_vong=30)
        mang.huan_luyen(self.X, self.y)
        loss = mang.lay_lich_su_loss()
        assert len(loss) == 30


class TestXuLyVanBan:
    def test_chuan_hoa(self):
        xl = XuLyVanBan()
        assert xl.chuan_hoa("  Xin   Chào  ") == "xin chào"

    def test_tach_tu(self):
        xl = XuLyVanBan()
        tu = xl.tach_tu("Xử lý ngôn ngữ tự nhiên")
        assert len(tu) >= 3  # underthesea tách từ ghép: "xử lý", "ngôn ngữ", "tự nhiên"

    def test_loai_bo_tu_dung(self):
        xl = XuLyVanBan()
        ket_qua = xl.loai_bo_tu_dung("Đây là một ví dụ")
        assert "là" not in ket_qua.split()
        assert "một" not in ket_qua.split()

    def test_xu_ly_day_du(self):
        xl = XuLyVanBan()
        ket_qua = xl.xu_ly_day_du("  Đây LÀ một Ví Dụ!  ")
        assert "là" not in ket_qua.split()

    def test_ma_hoa_tfidf(self):
        xl = XuLyVanBan()
        van_ban = ["học máy", "mạng nơ-ron", "học sâu"]
        tfidf = xl.ma_hoa_tfidf(van_ban)
        assert tfidf.shape[0] == 3
        assert tfidf.shape[1] > 0

    def test_trich_xuat_tu_khoa(self):
        xl = XuLyVanBan()
        tu_khoa = xl.trich_xuat_tu_khoa("học máy là học máy tốt", top_n=2)
        assert "học" in tu_khoa
        assert "máy" in tu_khoa


class TestXuLySo:
    def test_chuan_hoa_minmax(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        xl = XuLySo()
        ket_qua = xl.chuan_hoa_minmax(data)
        assert ket_qua.min() >= 0
        assert ket_qua.max() <= 1

    def test_chuan_hoa_zscore(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        xl = XuLySo()
        ket_qua = xl.chuan_hoa_zscore(data)
        assert abs(ket_qua.mean()) < 0.01

    def test_xu_ly_gia_tri_thieu(self):
        data = np.array([[1, np.nan], [3, 4], [np.nan, 6]])
        ket_qua = XuLySo.xu_ly_gia_tri_thieu(data, "trung_vi")
        assert not np.isnan(ket_qua).any()

    def test_ma_hoa_nhan(self):
        nhan = np.array(["a", "b", "a", "c"])
        nhan_so, tu_dien = XuLySo.ma_hoa_nhan(nhan)
        assert len(tu_dien) == 3

    def test_chia_du_lieu(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y, ty_le_test=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20


class TestTaoDacTrung:
    def test_dac_trung_da_thuc(self):
        X = np.array([[1, 2], [3, 4]])
        ket_qua = TaoDacTrung.dac_trung_da_thuc(X, bac=2)
        assert ket_qua.shape == (2, 4)

    def test_dac_trung_tuong_tac(self):
        X = np.array([[1, 2], [3, 4]])
        ket_qua = TaoDacTrung.dac_trung_tuong_tac(X)
        assert ket_qua.shape == (2, 3)

    def test_giam_chieu_pca(self):
        X = np.random.randn(50, 10)
        ket_qua = TaoDacTrung.giam_chieu_pca(X, so_chieu=2)
        assert ket_qua.shape == (50, 2)

    def test_chon_dac_trung_phuong_sai(self):
        X = np.array([[1, 0, 5], [2, 0, 6], [3, 0, 7]])
        X_chon, chi_so = TaoDacTrung.chon_dac_trung_phuong_sai(X, nguong=0.1)
        assert 1 not in chi_so
        assert 0 in chi_so
        assert 2 in chi_so


class TestMetrics:
    def test_do_chinh_xac(self):
        y_thuc = np.array([0, 1, 1, 0])
        y_du_doan = np.array([0, 1, 0, 0])
        assert Metrics.do_chinh_xac(y_thuc, y_du_doan) == 0.75

    def test_mse(self):
        y_thuc = np.array([1, 2, 3])
        y_du_doan = np.array([1.1, 2.2, 2.8])
        assert Metrics.mse(y_thuc, y_du_doan) < 0.1

    def test_precision_recall_f1(self):
        y_thuc = np.array([0, 1, 1, 0, 1])
        y_du_doan = np.array([0, 1, 0, 0, 1])
        pr = Metrics.precision_recall_f1(y_thuc, y_du_doan)
        assert "precision" in pr
        assert "recall" in pr
        assert "f1" in pr


class TestValidator:
    def test_kiem_tra_kich_thuoc(self):
        X = np.zeros((10, 5))
        assert Validator.kiem_tra_kich_thuoc(X, (10, 5))
        assert not Validator.kiem_tra_kich_thuoc(X, (10, 3))

    def test_kiem_tra_gia_tri_thieu(self):
        X_sach = np.array([[1, 2], [3, 4]])
        X_nan = np.array([[1, np.nan], [3, 4]])
        assert not Validator.kiem_tra_gia_tri_thieu(X_sach)
        assert Validator.kiem_tra_gia_tri_thieu(X_nan)

    def test_kiem_tra_du_lieu_hop_le(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        hop_le, _ = Validator.kiem_tra_du_lieu_hop_le(X, y)
        assert hop_le

    def test_kiem_tra_nhiem_vu(self):
        y_loai = np.array([0, 1, 0, 1, 0])
        y_hoi_quy = np.array([1.5, 2.3, 3.7, 4.1, 5.9])
        assert Validator.kiem_tra_nhiem_vu(y_loai) == "phan_loai"
        assert Validator.kiem_tra_nhiem_vu(y_hoi_quy) == "hoi_quy"


class TestEngine:
    def test_huan_luyen(self):
        engine = Engine()
        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian()
        ket_qua = engine.huan_luyen(pl, X, y)
        assert ket_qua["trang_thai"] == "thanh_cong"

    def test_du_doan(self):
        engine = Engine()
        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian()
        engine.huan_luyen(pl, X, y)
        du_doan = engine.du_doan(pl, X[:10])
        assert len(du_doan) == 10

    def test_lich_su(self):
        engine = Engine()
        pl = PhanLoai()
        X, y = DuLieuMau.phan_loai_don_gian()
        engine.huan_luyen(pl, X, y)
        lich_su = engine.lay_lich_su()
        assert len(lich_su) == 1


class TestPipeline:
    def test_pipeline(self):
        pipe = Pipeline()
        xl = XuLySo()
        pl = PhanLoai(thuat_toan="logistic")
        pipe.them_buoc("chuan_hoa", xl)
        pipe.them_buoc("phan_loai", pl)

        X, y = DuLieuMau.phan_loai_don_gian()
        X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

        pipe.fit(X_train, y_train)
        du_doan = pipe.predict(X_test)
        assert len(du_doan) == len(X_test)

    def test_danh_sach_buoc(self):
        pipe = Pipeline()
        pipe.them_buoc("buoc1", XuLySo())
        pipe.them_buoc("buoc2", PhanLoai())
        assert pipe.danh_sach_buoc() == ["buoc1", "buoc2"]


class TestWord2Vec:
    def test_huan_luyen(self):
        from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet

        w2v = Word2VecTiengViet(kich_thuoc=10, toi_thieu_dem=1)
        van_ban = [
            "học máy rất thú vị",
            "mạng nơ-ron rất hay",
            "học sâu là xu hướng",
            "máy tính rất thông minh",
        ] * 10
        w2v.huan_luyen(van_ban, so_vong=3)
        assert w2v._da_huan_luyen

    def test_lay_vector(self):
        from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet

        w2v = Word2VecTiengViet(kich_thuoc=10, toi_thieu_dem=1)
        van_ban = ["học máy rất hay"] * 10
        w2v.huan_luyen(van_ban, so_vong=2)
        v = w2v.lay_vector("học")
        assert v is not None
        assert len(v) == 10

    def test_tim_tu_giong(self):
        from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet

        w2v = Word2VecTiengViet(kich_thuoc=10, toi_thieu_dem=1)
        van_ban = ["học máy rất hay", "mạng nơ-ron rất tốt"] * 10
        w2v.huan_luyen(van_ban, so_vong=2)
        tu_giong = w2v.tim_tu_giong("học", top_n=2)
        assert isinstance(tu_giong, list)


class TestPhanTichCamXuc:
    def test_tu_dien(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        ket_qua = ptx.phan_tich("Sản phẩm rất tốt")
        assert ket_qua["nhan"] in ["positive", "negative", "neutral"]

    def test_phan_tich_nhieu(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        ket_qua = ptx.phan_tich_nhieu(["tốt", "xấu"])
        assert len(ket_qua) == 2

    def test_thong_ke(self):
        from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

        ptx = PhanTichCamXuc(che_do="tu_dien")
        tk = ptx.thong_ke(["tốt", "tệ", "bình thường"])
        assert "positive" in tk
        assert "negative" in tk


class TestAutoML:
    def test_phan_loai(self):
        from vietnamese_ai.automl.auto_ml import AutoML

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=3)
        auto = AutoML(so_fold=3)
        ket_qua = auto.fit(X, y)
        assert ket_qua["mo_hinh_tot_nhat"] is not None
        assert ket_qua["diem_tot_nhat"] > 0

    def test_bao_cao(self):
        from vietnamese_ai.automl.auto_ml import AutoML

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=2)
        auto = AutoML(so_fold=3)
        auto.fit(X, y)
        bc = auto.bao_cao()
        assert "AUTOML" in bc


class TestTangCuongVanBan:
    def test_tang_cuong(self):
        from vietnamese_ai.augmentation.text_augmenter import TangCuongVanBan

        tc = TangCuongVanBan(seed=42)
        ket_qua = tc.tang_cuong("Sản phẩm rất tốt", so_mau=5)
        assert len(ket_qua) >= 2
        assert ket_qua[0] == "Sản phẩm rất tốt"

    def test_tang_cuong_tap_du_lieu(self):
        from vietnamese_ai.augmentation.text_augmenter import TangCuongVanBan

        tc = TangCuongVanBan(seed=42)
        vb, nhan = tc.tang_cuong_tap_du_lieu(
            ["tốt", "xấu"], ["positive", "negative"], so_mau_moi_mau=3
        )
        assert len(vb) >= 3  # 2 gốc + ít nhất 1 augmented


class TestGiaiThichMoHinh:
    def test_feature_importance(self):
        from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        pl = PhanLoai(thuat_toan="rung_ngau_nhien")
        pl.huan_luyen(X, y)

        gt = GiaiThichMoHinh(pl, X, y)
        fi = gt.feature_importance()
        assert len(fi) == 5


class TestTheoDoiThiNghiem:
    def test_theo_doi(self, tmp_path):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        td = TheoDoiThiNghiem("test", luu_tai=str(tmp_path))
        td.bat_dau("test_run")
        td.log_param("model", "logistic")
        td.log_metric("accuracy", 0.95)
        ket_qua = td.ket_thuc()
        assert ket_qua["trang_thai"] == "hoan_tat"


class TestMangSau:
    def test_numpy_backend(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

        mang = MangSau(lop_an=[16, 8], so_vong=10, kich_thuoc_batch=32)
        mang.huan_luyen(X_train, y_train)
        assert mang.da_huan_luyen
        du_doan = mang.du_doan(X_test)
        assert len(du_doan) == len(X_test)

    def test_danh_gia(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=3)
        mang = MangSau(lop_an=[8], so_vong=5)
        mang.huan_luyen(X, y)
        diem = mang.danh_gia(X, y)
        assert diem > 0


class TestDuDoanChuoiThoiGian:
    def test_exponential(self):
        from vietnamese_ai.timeseries.forecaster import DuDoanChuoiThoiGian

        du_lieu = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
        dstg = DuDoanChuoiThoiGian(phuong_phap="exponential")
        dstg.huan_luyen(du_lieu)
        du_doan = dstg.du_doan(10)
        assert len(du_doan) == 10

    def test_linear_trend(self):
        from vietnamese_ai.timeseries.forecaster import DuDoanChuoiThoiGian

        du_lieu = np.arange(50, dtype=float) + np.random.randn(50) * 0.1
        dstg = DuDoanChuoiThoiGian(phuong_phap="linear_trend")
        dstg.huan_luyen(du_lieu)
        du_doan = dstg.du_doan(10)
        assert len(du_doan) == 10
        assert du_doan[-1] > du_lieu[-1]  # upward trend

    def test_window_regression(self):
        from vietnamese_ai.timeseries.forecaster import DuDoanChuoiThoiGian

        du_lieu = np.sin(np.linspace(0, 20, 200)) + np.random.randn(200) * 0.1
        dstg = DuDoanChuoiThoiGian(phuong_phap="window_regression", cua_so=10)
        dstg.huan_luyen(du_lieu)
        du_doan = dstg.du_doan(5)
        assert len(du_doan) == 5

    def test_danh_gia(self):
        from vietnamese_ai.timeseries.forecaster import DuDoanChuoiThoiGian

        du_lieu = np.sin(np.linspace(0, 10, 100))
        dstg = DuDoanChuoiThoiGian(phuong_phap="exponential")
        dstg.huan_luyen(du_lieu)
        diem = dstg.danh_gia(np.zeros(10), du_lieu[-10:])
        assert isinstance(diem, float)


class TestPhanLoaiHinhAnh:
    def test_numpy_fallback(self):
        from vietnamese_ai.vision.image_classifier import PhanLoaiHinhAnh

        X = np.random.rand(50, 1, 8, 8)
        y = np.random.randint(0, 3, 50)
        plha = PhanLoaiHinhAnh(so_lop=3)
        plha.huan_luyen(X, y)
        du_doan = plha.du_doan(X[:10])
        assert len(du_doan) == 10


class TestLopDense:
    def test_tien(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        lop = LopDense(5, 3, ham_kich_hoat="relu")
        X = np.random.rand(10, 5)
        ket_qua = lop.tien(X)
        assert ket_qua.shape == (10, 3)

    def test_softmax(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        lop = LopDense(5, 3, ham_kich_hoat="softmax")
        X = np.random.rand(10, 5)
        ket_qua = lop.tien(X)
        assert ket_qua.shape == (10, 3)
        assert np.allclose(ket_qua.sum(axis=1), 1.0)


class TestQuanLyMoHinh:
    def test_dang_ky_va_tai(self, tmp_path):
        from vietnamese_ai.registry.model_registry import QuanLyMoHinh

        ql = QuanLyMoHinh(str(tmp_path / "registry"))

        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl.huan_luyen(X, y)

        ver = ql.dang_ky(pl, ten="test_model", metrics={"accuracy": 0.95})
        assert ver is not None

        mo_hinh = ql.tai("test_model")
        assert mo_hinh is not None

    def test_danh_sach(self, tmp_path):
        from vietnamese_ai.registry.model_registry import QuanLyMoHinh

        ql = QuanLyMoHinh(str(tmp_path / "registry"))

        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl.huan_luyen(X, y)

        ql.dang_ky(pl, ten="model_a", version="1.0")
        ql.dang_ky(pl, ten="model_b", version="1.0")

        ds = ql.danh_sach()
        assert len(ds) == 2

    def test_promote(self, tmp_path):
        from vietnamese_ai.registry.model_registry import QuanLyMoHinh

        ql = QuanLyMoHinh(str(tmp_path / "registry"))

        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl.huan_luyen(X, y)

        ql.dang_ky(pl, ten="model", version="1.0")
        ql.promote("model", "1.0", "production")

        mo_hinh = ql.tai("model", "production")
        assert mo_hinh is not None


class TestXuLyStream:
    def test_them_du_lieu(self):
        from vietnamese_ai.streaming.processor import XuLyStream

        stream = XuLyStream(kich_thuoc_cua_so=50)
        for gv in np.random.randn(20):
            stream.them_du_lieu(gv)
        assert len(stream) == 20

    def test_thong_ke(self):
        from vietnamese_ai.streaming.processor import XuLyStream

        stream = XuLyStream()
        stream.them_nhieu([1.0, 2.0, 3.0, 4.0, 5.0])
        tk = stream.lay_thong_ke()
        assert "trung_binh" in tk
        assert tk["trung_binh"] == 3.0

    def test_bat_thuong(self):
        from vietnamese_ai.streaming.processor import XuLyStream

        stream = XuLyStream(nguong_bat_thuong=2.0)
        du_lieu = list(np.random.randn(20)) + [100.0]
        ket_qua = stream.them_nhieu(du_lieu)
        bat_thuong = [k for k in ket_qua if k["la_bat_thuong"]]
        assert len(bat_thuong) >= 1


class TestXuatONNX:
    def test_khoi_tao(self):
        from vietnamese_ai.export.onnx_export import XuatONNX

        xuat = XuatONNX()
        assert xuat is not None


class TestMultiGPU:
    def test_khoi_tao(self):
        from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer

        trainer = MultiGPUTrainer()
        assert isinstance(trainer.so_gpu, int)

    def test_cpu_fallback(self):
        from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer

        trainer = MultiGPUTrainer()
        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        ket_qua = trainer.huan_luyen(pl, X, y)
        assert "thiet_bi" in ket_qua


class TestPhanTanHuanLuyen:
    def test_huan_luyen(self):
        from vietnamese_ai.distributed.distributed import PhanTanHuanLuyen

        pt = PhanTanHuanLuyen(so_worker=2)
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=200)
        ket_qua = pt.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
        assert ket_qua["so_worker"] == 2
        assert ket_qua["tong_mau"] == 200


class TestModelHub:
    def test_dang_ky_va_tim_kiem(self, tmp_path):
        from vietnamese_ai.hub.model_hub import ModelHub

        hub = ModelHub(str(tmp_path / "hub"))
        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl.huan_luyen(X, y)

        hub.dang_ky(pl, ten="test_model", tac_gia="test", tags=["test"])
        ds = hub.tim_kiem(tags=["test"])
        assert len(ds) >= 1

    def test_danh_gia_sao(self, tmp_path):
        from vietnamese_ai.hub.model_hub import ModelHub

        hub = ModelHub(str(tmp_path / "hub"))
        pl = PhanLoai(thuat_toan="logistic")
        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50)
        pl.huan_luyen(X, y)

        hub.dang_ky(pl, ten="model_a", tac_gia="test")
        hub.danh_gia_sao("test/model_a", sao=5)
        ds = hub.tim_kiem()
        assert ds[0]["danh_gia_sao"] == 5.0


class TestPluginManager:
    def test_dang_ky_plugin(self):
        from vietnamese_ai.plugins.plugin_manager import PluginManager

        pm = PluginManager()

        @pm.dang_ky_plugin("test_plugin")
        def test_func():
            return 42

        plugin = pm.lay_plugin("test_plugin")
        assert plugin() == 42

    def test_hook(self):
        from vietnamese_ai.plugins.plugin_manager import PluginManager

        pm = PluginManager()
        called = []

        @pm.hook("pre_train")
        def on_pre_train(**kwargs):
            called.append(True)

        pm.chay_hook("pre_train")
        assert len(called) == 1

    def test_danh_sach(self):
        from vietnamese_ai.plugins.plugin_manager import PluginManager

        pm = PluginManager()

        @pm.dang_ky_plugin("p1")
        def f1():
            pass

        @pm.dang_ky_plugin("p2")
        def f2():
            pass

        ds = pm.danh_sach()
        assert len(ds) == 2


class TestHeThongXacThuc:
    def test_dang_ky_va_dang_nhap(self, tmp_path):
        from vietnamese_ai.enterprise.auth import HeThongXacThuc

        auth = HeThongXacThuc(str(tmp_path / "auth"))
        auth.dang_ky("admin", "pass123", vai_tro="admin")
        token = auth.dang_nhap("admin", "pass123")
        assert len(token) > 0

    def test_kiem_tra_quyen(self, tmp_path):
        from vietnamese_ai.enterprise.auth import HeThongXacThuc

        auth = HeThongXacThuc(str(tmp_path / "auth"))
        auth.dang_ky("viewer", "pass", vai_tro="viewer")
        token = auth.dang_nhap("viewer", "pass")

        assert auth.kiem_tra_quyen(token, "view") is True
        assert auth.kiem_tra_quyen(token, "train") is False

    def test_api_key(self, tmp_path):
        from vietnamese_ai.enterprise.auth import HeThongXacThuc

        auth = HeThongXacThuc(str(tmp_path / "auth"))
        auth.dang_ky("dev", "pass", vai_tro="developer")
        api_key = auth.tao_api_key("dev")
        assert api_key.startswith("vai_")


class TestNhatKyHoatDong:
    def test_ghi_va_tim_kiem(self, tmp_path):
        from vietnamese_ai.enterprise.audit import NhatKyHoatDong

        nk = NhatKyHoatDong(str(tmp_path / "audit"))
        nk.ghi("train", "admin", "Huấn luyện mô hình")
        nk.ghi("predict", "user1", "Dự đoán dữ liệu mới")

        ket_qua = nk.tim_kiem(nguoi_dung="admin")
        assert len(ket_qua) == 1

    def test_thong_ke(self, tmp_path):
        from vietnamese_ai.enterprise.audit import NhatKyHoatDong

        nk = NhatKyHoatDong(str(tmp_path / "audit"))
        nk.ghi("train", "admin", "Test")
        nk.ghi("predict", "user1", "Test")

        tk = nk.thong_ke()
        assert tk["tong_ban_ghi"] == 2


class TestCloudDeployment:
    def test_tao_docker_config(self, tmp_path):
        from pathlib import Path

        from vietnamese_ai.cloud.deployment import CloudDeployment

        deploy = CloudDeployment()
        duong_dan = deploy.tao_docker_config("test_model", str(tmp_path / "deploy"))
        assert Path(duong_dan).exists()
        assert (Path(duong_dan) / "Dockerfile").exists()


class TestMarketplace:
    def test_dang_ky_va_tim_kiem(self, tmp_path):
        from vietnamese_ai.cloud.marketplace import Marketplace

        mp = Marketplace(str(tmp_path / "market"))
        mp.dang_ky(ten="sentiment", loai="model", tac_gia="test", tags=["nlp"])
        mp.dang_ky(ten="prices", loai="dataset", tac_gia="test", tags=["finance"])

        ds = mp.tim_kiem(category="model")
        assert len(ds) == 1
        assert ds[0]["ten"] == "sentiment"

    def test_danh_gia(self, tmp_path):
        from vietnamese_ai.cloud.marketplace import Marketplace

        mp = Marketplace(str(tmp_path / "market"))
        mp.dang_ky(ten="test", loai="model")
        mp.danh_gia("anonymous/test", sao=4, binh_luan="Hay")

        ds = mp.tim_kiem()
        assert ds[0]["danh_gia_sao"] == 4.0

    def test_thong_ke(self, tmp_path):
        from vietnamese_ai.cloud.marketplace import Marketplace

        mp = Marketplace(str(tmp_path / "market"))
        mp.dang_ky(ten="a", loai="model")
        mp.dang_ky(ten="b", loai="dataset")

        tk = mp.thong_ke()
        assert tk["tong_items"] == 2
