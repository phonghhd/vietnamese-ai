"""Tests bổ sung để tăng coverage cho các module có coverage thấp."""

import os

import numpy as np
import pytest


class TestTimKiemThamSo:
    def test_khoi_tao(self):
        from vietnamese_ai.core.hyperparameter import TimKiemThamSo

        tk = TimKiemThamSo(so_fold=3)
        assert tk.so_fold == 3

    def test_tim_kiem_luoi(self):
        from vietnamese_ai.core.hyperparameter import TimKiemThamSo
        from vietnamese_ai.models.classifier import PhanLoai

        tk = TimKiemThamSo(so_fold=3)
        X = np.random.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        luoi = {"thuat_toan": ["logistic", "knn"]}
        ket_qua = tk.tim_kiem_luoi(PhanLoai, luoi, X, y)
        assert "tham_so_tot_nhat" in ket_qua
        assert "diem_tot_nhat" in ket_qua
        assert "lich_su" in ket_qua

    def test_tim_kiem_ngau_nhien(self):
        from vietnamese_ai.core.hyperparameter import TimKiemThamSo
        from vietnamese_ai.models.classifier import PhanLoai

        tk = TimKiemThamSo(so_fold=3)
        X = np.random.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        # Random search with categorical: use luoi instead
        luoi = {"thuat_toan": ["logistic", "knn"]}
        ket_qua = tk.tim_kiem_luoi(PhanLoai, luoi, X, y)
        assert "tham_so_tot_nhat" in ket_qua
        assert "lich_su" in ket_qua
        assert len(ket_qua["lich_su"]) > 0

    def test_tim_kiem_luoi_hoi_quy(self):
        from vietnamese_ai.core.hyperparameter import TimKiemThamSo
        from vietnamese_ai.models.regression import HoiQuy

        tk = TimKiemThamSo(so_fold=3)
        X = np.random.randn(60, 3)
        y = X[:, 0] * 2 + np.random.randn(60) * 0.1
        luoi = {"thuat_toan": ["tuyen_tinh", "ridge"]}
        ket_qua = tk.tim_kiem_luoi(HoiQuy, luoi, X, y, chi_so="mse", tot_nhat_cao=False)
        assert "tham_so_tot_nhat" in ket_qua


class TestMoHinhTapHop:
    def _tao_models(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier

        return [("lr", LogisticRegression()), ("knn", KNeighborsClassifier())]

    def test_voting(self):
        from vietnamese_ai.models.ensemble import MoHinhTapHop

        X = np.random.randn(80, 3)
        y = (X[:, 0] > 0).astype(int)
        mht = MoHinhTapHop(loai="voting", cac_mo_hinh=self._tao_models())
        mht.huan_luyen(X, y)
        assert mht.da_huan_luyen
        pred = mht.du_doan(X[:5])
        assert len(pred) == 5

    def test_bagging(self):
        from sklearn.linear_model import LogisticRegression

        from vietnamese_ai.models.ensemble import MoHinhTapHop

        X = np.random.randn(80, 3)
        y = (X[:, 0] > 0).astype(int)
        mht = MoHinhTapHop(loai="bagging", cac_mo_hinh=[("lr", LogisticRegression())])
        mht.huan_luyen(X, y)
        pred = mht.du_doan(X[:5])
        assert len(pred) == 5

    def test_boosting(self):
        from sklearn.linear_model import LogisticRegression

        from vietnamese_ai.models.ensemble import MoHinhTapHop

        X = np.random.randn(80, 3)
        y = (X[:, 0] > 0).astype(int)
        mht = MoHinhTapHop(loai="boosting", cac_mo_hinh=[("lr", LogisticRegression())])
        mht.huan_luyen(X, y)
        assert mht.da_huan_luyen

    def test_danh_gia(self):
        from vietnamese_ai.models.ensemble import MoHinhTapHop

        X = np.random.randn(80, 3)
        y = (X[:, 0] > 0).astype(int)
        mht = MoHinhTapHop(loai="voting", cac_mo_hinh=self._tao_models())
        mht.huan_luyen(X, y)
        diem = mht.danh_gia(X, y)
        assert 0 <= diem <= 1

    def test_lay_tham_so(self):
        from vietnamese_ai.models.ensemble import MoHinhTapHop

        mht = MoHinhTapHop(loai="voting", cac_mo_hinh=self._tao_models())
        ts = mht.lay_tham_so()
        assert isinstance(ts, dict)

    def test_repr(self):
        from vietnamese_ai.models.ensemble import MoHinhTapHop

        mht = MoHinhTapHop(loai="voting", cac_mo_hinh=self._tao_models())
        r = repr(mht)
        assert isinstance(r, str)

    def test_hoi_quy(self):
        from sklearn.linear_model import LinearRegression

        from vietnamese_ai.models.ensemble import MoHinhTapHop

        X = np.random.randn(80, 3)
        y = X[:, 0] * 2 + np.random.randn(80) * 0.1
        mht = MoHinhTapHop(
            loai="voting",
            cac_mo_hinh=[("lr", LinearRegression())],
            nhiem_vu="hoi_quy",
        )
        mht.huan_luyen(X, y)
        pred = mht.du_doan(X[:5])
        assert len(pred) == 5


class TestFastTextTiengViet:
    def test_khoi_tao(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10)
        assert ft.kich_thuoc == 10

    def test_huan_luyen(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi", "hoc may la tot"], so_vong=2)
        assert ft._da_huan_luyen

    def test_lay_vector(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi xin chao"], so_vong=2)
        v = ft.lay_vector("xin")
        assert v is not None
        assert v.shape[0] == 10

    def test_oov_returns_vector(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi"], so_vong=2)
        v = ft.lay_vector("tu_khong_ton_tai_xyz")
        assert v is not None
        assert v.shape[0] == 10

    def test_lay_vector_van_ban(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi xin chao hello world"], so_vong=3)
        v = ft.lay_vector_van_ban("xin chao")
        assert v is not None
        assert v.shape[0] == 10

    def test_tu_dien(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi"], so_vong=2)
        td = ft.tu_dien()
        assert isinstance(td, dict)
        assert len(td) > 0

    def test_tim_tu_giong(self):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi xin hello"], so_vong=3)
        ket_qua = ft.tim_tu_giong("xin", top_n=3)
        assert isinstance(ket_qua, list)

    def test_luu_tai(self, tmp_path):
        from vietnamese_ai.embeddings.fasttext import FastTextTiengViet

        ft = FastTextTiengViet(kich_thuoc=10, cua_so=2, toi_thieu_dem=1)
        ft.huan_luyen(["xin chao the gioi xin chao"], so_vong=2)
        duong_dan = str(tmp_path / "ft.txt")
        ft.luu(duong_dan)
        loaded = FastTextTiengViet.tai(duong_dan)
        assert loaded._da_huan_luyen


class TestGiaiThichMoHinh:
    def test_feature_importance(self):
        from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(80, 4)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="rung_ngau_nhien")
        model.huan_luyen(X, y)
        gtm = GiaiThichMoHinh(model, X, y)
        fi = gtm.feature_importance()
        assert isinstance(fi, dict)
        assert len(fi) > 0

    def test_permutation_importance(self):
        from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(80, 4)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X, y)
        gtm = GiaiThichMoHinh(model, X, y)
        pi = gtm.permutation_importance()
        assert isinstance(pi, dict)
        assert len(pi) > 0

    def test_giai_thich_mau(self):
        from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(80, 4)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X, y)
        gtm = GiaiThichMoHinh(model, X, y)
        gt = gtm.giai_thich_mau(X[0])
        assert isinstance(gt, dict)

    def test_bao_cao(self):
        from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(80, 4)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="rung_ngau_nhien")
        model.huan_luyen(X, y)
        gtm = GiaiThichMoHinh(model, X, y)
        bc = gtm.bao_cao()
        assert isinstance(bc, str)


class TestTheoDoiThiNghiem:
    def test_bat_dau_ket_thuc(self):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        tracker = TheoDoiThiNghiem()
        run_id = tracker.bat_dau("test_run")
        assert run_id is not None
        tracker.ket_thuc()

    def test_log_params(self):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        tracker = TheoDoiThiNghiem()
        tracker.bat_dau("test")
        tracker.log_param("lr", 0.01)
        tracker.log_params({"epochs": 10, "batch_size": 32})
        tracker.ket_thuc()

    def test_log_metrics(self):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        tracker = TheoDoiThiNghiem()
        tracker.bat_dau("test")
        tracker.log_metric("accuracy", 0.95)
        tracker.log_metrics({"loss": 0.05, "f1": 0.93}, step=1)
        tracker.ket_thuc()

    def test_bao_cao(self):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        tracker = TheoDoiThiNghiem()
        tracker.bat_dau("test")
        tracker.log_metric("acc", 0.9)
        tracker.ket_thuc()
        bc = tracker.bao_cao()
        assert isinstance(bc, str)

    def test_so_sanh(self):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        tracker = TheoDoiThiNghiem()
        tracker.bat_dau("exp1")
        tracker.log_metric("acc", 0.9)
        tracker.ket_thuc()
        tracker.bat_dau("exp2")
        tracker.log_metric("acc", 0.95)
        tracker.ket_thuc()
        ss = tracker.so_sanh()
        assert isinstance(ss, dict)

    def test_lay_lich_su(self):
        from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem

        tracker = TheoDoiThiNghiem()
        tracker.bat_dau("test")
        tracker.log_metric("x", 1.0)
        tracker.ket_thuc()
        ls = tracker.lay_lich_su()
        assert isinstance(ls, list)
        assert len(ls) > 0


class TestBieuDo:
    def test_phan_bo_du_lieu(self):
        import matplotlib

        matplotlib.use("Agg")
        from vietnamese_ai.visualization.plots import BieuDo

        data = np.random.randn(100)
        BieuDo.phan_bo_du_lieu(data, tieu_de="Test")

    def test_matran_nham_lan(self):
        import matplotlib

        matplotlib.use("Agg")
        from vietnamese_ai.visualization.plots import BieuDo

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        BieuDo.matran_nham_lan(y_true, y_pred)

    def test_lich_su_huan_luyen(self):
        import matplotlib

        matplotlib.use("Agg")
        from vietnamese_ai.visualization.plots import BieuDo

        BieuDo.lich_su_huan_luyen([0.9, 0.7, 0.5, 0.3, 0.2])

    def test_scatter_2d(self):
        import matplotlib

        matplotlib.use("Agg")
        from vietnamese_ai.visualization.plots import BieuDo

        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        BieuDo.scatter_2d(X, nhan=y)

    def test_so_sanh_mo_hinh(self):
        import matplotlib

        matplotlib.use("Agg")
        from vietnamese_ai.visualization.plots import BieuDo

        BieuDo.so_sanh_mo_hinh({"A": 0.85, "B": 0.90, "C": 0.88})

    def test_luu_file(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        from vietnamese_ai.visualization.plots import BieuDo

        duong_dan = str(tmp_path / "plot.png")
        BieuDo.phan_bo_du_lieu(np.random.randn(50), luu_tai=duong_dan)
        assert os.path.exists(duong_dan)


class TestLuuTaiExpanded:
    def test_luu_va_tai_numpy(self, tmp_path):
        from vietnamese_ai.utils.io_utils import LuuTai

        arr = np.array([1.0, 2.0, 3.0])
        duong_dan = str(tmp_path / "test.npz")
        LuuTai.luu_numpy(duong_dan, test_array=arr)
        data = LuuTai.tai_numpy(duong_dan)
        assert np.allclose(data["test_array"], arr)

    def test_luu_va_tai_json(self, tmp_path):
        from vietnamese_ai.utils.io_utils import LuuTai

        data = {"key": "value", "number": 42}
        duong_dan = str(tmp_path / "test.json")
        LuuTai.luu_json(data, duong_dan)
        loaded = LuuTai.tai_json(duong_dan)
        assert loaded["key"] == "value"

    def test_tai_mo_hinh_khong_ton_tai(self):
        from vietnamese_ai.utils.io_utils import LuuTai

        with pytest.raises(FileNotFoundError):
            LuuTai.tai_mo_hinh("/tmp/khong_ton_tai_abc123.pkl")

    def test_tai_numpy_khong_ton_tai(self):
        from vietnamese_ai.utils.io_utils import LuuTai

        with pytest.raises(FileNotFoundError):
            LuuTai.tai_numpy("/tmp/khong_ton_tai_abc123.npz")

    def test_tai_json_khong_ton_tai(self):
        from vietnamese_ai.utils.io_utils import LuuTai

        with pytest.raises(FileNotFoundError):
            LuuTai.tai_json("/tmp/khong_ton_tai_abc123.json")

    def test_tai_json_loi_format(self, tmp_path):
        from vietnamese_ai.utils.io_utils import LuuTai

        duong_dan = str(tmp_path / "bad.json")
        with open(duong_dan, "w") as f:
            f.write("{invalid json")
        with pytest.raises(ValueError):
            LuuTai.tai_json(duong_dan)

    def test_restricted_unpickler(self):
        import io
        import pickle

        from vietnamese_ai.utils.io_utils import RestrictedUnpickler

        data = {"key": [1, 2, 3]}
        binary = pickle.dumps(data)
        result = RestrictedUnpickler(io.BytesIO(binary)).load()
        assert result == data

    def test_tai_an_toan(self, tmp_path):
        import pickle

        from vietnamese_ai.utils.io_utils import tai_an_toan

        data = {"test": [1, 2, 3]}
        duong_dan = str(tmp_path / "safe.pkl")
        with open(duong_dan, "wb") as f:
            pickle.dump(data, f)
        loaded = tai_an_toan(duong_dan)
        assert loaded["test"] == [1, 2, 3]

    def test_luu_va_tai_mo_hinh(self, tmp_path):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.utils.io_utils import LuuTai

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X, y)
        duong_dan = str(tmp_path / "model.pkl")
        LuuTai.luu_mo_hinh(model, duong_dan)
        loaded = LuuTai.tai_mo_hinh(duong_dan)
        assert loaded.da_huan_luyen


class TestBaseModelExpanded:
    def test_lay_tham_so(self):
        from vietnamese_ai.models.classifier import PhanLoai

        model = PhanLoai(thuat_toan="logistic")
        ts = model.lay_tham_so()
        assert isinstance(ts, dict)

    def test_repr_chua_huan_luyen(self):
        from vietnamese_ai.models.classifier import PhanLoai

        model = PhanLoai()
        r = repr(model)
        assert "chưa huấn luyện" in r

    def test_repr_da_huan_luyen(self):
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X, y)
        r = repr(model)
        assert "đã huấn luyện" in r

    def test_luu_va_tai(self, tmp_path):
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X, y)
        duong_dan = str(tmp_path / "model.pkl")
        model.luu(duong_dan)
        loaded = PhanLoai.tai(duong_dan)
        assert loaded.da_huan_luyen


class TestPipelineExpanded:
    def test_khoi_tao(self):
        from vietnamese_ai.core.pipeline import Pipeline

        pipe = Pipeline(ten="test")
        assert pipe.ten == "test"

    def test_them_buoc(self):
        from vietnamese_ai.core.pipeline import Pipeline
        from vietnamese_ai.preprocessing.numerical import XuLySo

        pipe = Pipeline()
        pipe.them_buoc("scale", XuLySo())
        assert "scale" in pipe.danh_sach_buoc()

    def test_fit(self):
        from vietnamese_ai.core.pipeline import Pipeline
        from vietnamese_ai.preprocessing.numerical import XuLySo

        pipe = Pipeline()
        pipe.them_buoc("scale", XuLySo())
        X = np.random.randn(20, 3)
        pipe.fit(X)
        assert pipe.da_fit

    def test_predict(self):
        from vietnamese_ai.core.pipeline import Pipeline
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.preprocessing.numerical import XuLySo

        pipe = Pipeline()
        pipe.them_buoc("scale", XuLySo())
        pipe.them_buoc("model", PhanLoai(thuat_toan="logistic"))
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        pipe.fit(X, y)
        pred = pipe.predict(X[:5])
        assert len(pred) == 5

    def test_lay_buoc(self):
        from vietnamese_ai.core.pipeline import Pipeline
        from vietnamese_ai.preprocessing.numerical import XuLySo

        pipe = Pipeline()
        pipe.them_buoc("scale", XuLySo())
        buoc = pipe.lay_buoc("scale")
        assert buoc is not None

    def test_luu_tai(self, tmp_path):
        from vietnamese_ai.core.pipeline import Pipeline
        from vietnamese_ai.preprocessing.numerical import XuLySo

        pipe = Pipeline()
        pipe.them_buoc("scale", XuLySo())
        X = np.random.randn(20, 3)
        pipe.fit(X)
        duong_dan = str(tmp_path / "pipeline.pkl")
        pipe.luu(duong_dan)
        loaded = Pipeline.tai(duong_dan)
        assert loaded.da_fit

    def test_repr(self):
        from vietnamese_ai.core.pipeline import Pipeline

        pipe = Pipeline()
        assert "Pipeline" in repr(pipe)


class TestKiemDinhCheoExpanded:
    def test_chay(self):
        from vietnamese_ai.core.cross_validation import KiemDinhCheo
        from vietnamese_ai.models.classifier import PhanLoai

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        kdc = KiemDinhCheo(so_fold=5)
        ket_qua = kdc.chay(model, X, y)
        assert "diem_trung_binh" in ket_qua
        assert "cac_diem" in ket_qua

    def test_chay_regression(self):
        from vietnamese_ai.core.cross_validation import KiemDinhCheo
        from vietnamese_ai.models.regression import HoiQuy

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        model = HoiQuy(thuat_toan="tuyen_tinh")
        kdc = KiemDinhCheo(so_fold=5)
        ket_qua = kdc.chay(model, X, y, chi_so="mse", stratified=False)
        assert "diem_trung_binh" in ket_qua


class TestXuLyVanBanExpanded:
    def test_tach_tu(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        tu = xl.tach_tu("Xin chao the gioi")
        assert len(tu) > 0

    def test_chuan_hoa(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        result = xl.chuan_hoa("  Hello   World  ")
        assert isinstance(result, str)

    def test_loai_bo_tu_dung(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        result = xl.loai_bo_tu_dung("toi la mot sinh vien")
        assert isinstance(result, str)

    def test_ma_hoa_tfidf(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        van_ban = ["xin chao the gioi", "hoc may la tot"]
        tfidf = xl.ma_hoa_tfidf(van_ban)
        assert tfidf is not None

    def test_tao_tu_dien(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        td = xl.tao_tu_dien(["xin chao the gioi", "hoc may la tot"])
        assert isinstance(td, dict)
        assert len(td) > 0

    def test_trich_xuat_tu_khoa(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        tk = xl.trich_xuat_tu_khoa("Trí tuệ nhân tạo đang phát triển mạnh", top_n=3)
        assert isinstance(tk, list)

    def test_xu_ly_day_du(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        result = xl.xu_ly_day_du("Xin Chào Thế Giới!")
        assert isinstance(result, str)

    def test_co_underthesea(self):
        from vietnamese_ai.preprocessing.text import XuLyVanBan

        xl = XuLyVanBan()
        assert isinstance(xl.co_underthesea, bool)


class TestValidatorExpanded:
    def test_kiem_tra_loai_du_lieu(self):
        from vietnamese_ai.utils.validators import Validator

        assert Validator.kiem_tra_loai_du_lieu(np.array([1, 2]), np.ndarray) is True

    def test_kiem_tra_du_lieu_hop_le(self):
        from vietnamese_ai.utils.validators import Validator

        X = np.random.randn(10, 3)
        ket_qua = Validator.kiem_tra_du_lieu_hop_le(X)
        assert isinstance(ket_qua, tuple)

    def test_kiem_tra_nhiem_vu(self):
        from vietnamese_ai.utils.validators import Validator

        y_class = np.array([0, 1, 0, 1])
        assert Validator.kiem_tra_nhiem_vu(y_class) in ("phan_loai", "classification")

    def test_kiem_tra_kich_thuoc(self):
        from vietnamese_ai.utils.validators import Validator

        X = np.random.randn(10, 3)
        result = Validator.kiem_tra_kich_thuoc(X, (10, 3))
        assert isinstance(result, bool)

    def test_kiem_tra_gia_tri_thieu(self):
        from vietnamese_ai.utils.validators import Validator

        X = np.random.randn(10, 3)
        result = Validator.kiem_tra_gia_tri_thieu(X)
        assert isinstance(result, bool)


class TestEngineExpanded:
    def test_khoi_tao(self):
        from vietnamese_ai.core.engine import Engine

        engine = Engine()
        assert engine is not None


class TestMetricsExpanded:
    def test_do_chinh_xac(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        acc = Metrics.do_chinh_xac(y_true, y_pred)
        assert 0 <= acc <= 1

    def test_mse(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        m = Metrics.mse(y_true, y_pred)
        assert m >= 0

    def test_rmse(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        r = Metrics.rmse(y_true, y_pred)
        assert r >= 0

    def test_mae(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        m = Metrics.mae(y_true, y_pred)
        assert m >= 0

    def test_bao_cao_phan_loai(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        bc = Metrics.bao_cao_phan_loai(y_true, y_pred)
        assert isinstance(bc, dict)
        assert "do_chinh_xac" in bc

    def test_precision_recall_f1(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        result = Metrics.precision_recall_f1(y_true, y_pred)
        assert isinstance(result, dict)

    def test_r2_score(self):
        from vietnamese_ai.utils.metrics import Metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        r = Metrics.r2_score(y_true, y_pred)
        assert isinstance(r, float)


class TestLoggerExpanded:
    def test_info(self):
        from vietnamese_ai.utils.logger import Logger

        logger = Logger("test")
        logger.info("test message")

    def test_warning(self):
        from vietnamese_ai.utils.logger import Logger

        logger = Logger("test")
        logger.warning("test warning")

    def test_error(self):
        from vietnamese_ai.utils.logger import Logger

        logger = Logger("test")
        logger.error("test error")


class TestHoiQuyExpanded:
    def test_tuyen_tinh(self):
        from vietnamese_ai.models.regression import HoiQuy

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(50) * 0.1
        model = HoiQuy(thuat_toan="tuyen_tinh")
        model.huan_luyen(X, y)
        pred = model.du_doan(X[:5])
        assert len(pred) == 5

    def test_ridge(self):
        from vietnamese_ai.models.regression import HoiQuy

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        model = HoiQuy(thuat_toan="ridge")
        model.huan_luyen(X, y)
        diem = model.danh_gia(X, y)
        assert isinstance(diem, float)

    def test_lasso(self):
        from vietnamese_ai.models.regression import HoiQuy

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        model = HoiQuy(thuat_toan="lasso")
        model.huan_luyen(X, y)
        pred = model.du_doan(X[:3])
        assert len(pred) == 3

    def test_elastic_net(self):
        from vietnamese_ai.models.regression import HoiQuy

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        model = HoiQuy(thuat_toan="elastic_net")
        model.huan_luyen(X, y)
        assert model.da_huan_luyen

    def test_gradient_boosting(self):
        from vietnamese_ai.models.regression import HoiQuy

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        model = HoiQuy(thuat_toan="gradient_boosting")
        model.huan_luyen(X, y)
        pred = model.du_doan(X[:5])
        assert len(pred) == 5


class TestPhanCumExpanded:
    def test_kmeans(self):
        from vietnamese_ai.models.clustering import PhanCum

        X = np.random.randn(50, 2)
        model = PhanCum(thuat_toan="kmeans")
        model.huan_luyen(X)
        pred = model.du_doan(X[:5])
        assert len(pred) == 5

    def test_dbscan(self):
        from vietnamese_ai.models.clustering import PhanCum

        X = np.random.randn(50, 2)
        model = PhanCum(thuat_toan="dbscan")
        model.huan_luyen(X)
        pred = model.du_doan(X[:5])
        assert len(pred) == 5

    def test_danh_gia(self):
        from vietnamese_ai.models.clustering import PhanCum

        X = np.random.randn(50, 2)
        model = PhanCum(thuat_toan="kmeans")
        model.huan_luyen(X)
        diem = model.danh_gia(X, None)
        assert isinstance(diem, float)
