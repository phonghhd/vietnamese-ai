import numpy as np
import pytest


class TestCatTiaMoHinh:
    def test_magnitude_pruning(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5)
        weights = np.random.randn(100)
        model = type("Model", (), {"_trong_so": weights.copy(), "danh_gia": lambda s, X, y: 0.9})()
        result = pruner.cat_tia(model)
        assert result["ty_le_prune"] > 0
        assert result["che_do"] == "magnitude"

    def test_structured_pruning(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="structured", ty_le=0.5)
        weights = np.random.randn(10, 10)
        model = type("Model", (), {"_trong_so": weights.copy()})()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_random_pruning(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="random", ty_le=0.3)
        weights = np.random.randn(100)
        model = type("Model", (), {"_trong_so": weights.copy()})()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_iterative_pruning(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="iterative", ty_le=0.3, so_vong_lap=2)
        weights = np.random.randn(100)
        model = type("Model", (), {"_trong_so": weights.copy()})()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_invalid_che_do(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        with pytest.raises(ValueError, match="che_do phải là"):
            CatTiaMoHinh(che_do="invalid")

    def test_invalid_ty_le_zero(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        with pytest.raises(ValueError, match="ty_le phải"):
            CatTiaMoHinh(ty_le=0)

    def test_invalid_ty_le_one(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        with pytest.raises(ValueError, match="ty_le phải"):
            CatTiaMoHinh(ty_le=1.0)

    def test_model_with_coef(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)
        model = type("Model", (), {
            "coef_": np.random.randn(5, 10),
            "feature_importances_": None,
        })()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_model_with_feature_importances(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)
        model = type("Model", (), {
            "feature_importances_": np.random.randn(10),
        })()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_model_no_weights(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)
        model = type("Model", (), {})()
        result = pruner.cat_tia(model)
        assert result["mask"] is None

    def test_model_with_weights_dict(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)
        model = type("Model", (), {"_weights": {"a": np.random.randn(5), "b": np.random.randn(5)}})()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_lay_mask(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5)
        weights = np.random.randn(100)
        model = type("Model", (), {"_trong_so": weights.copy()})()
        pruner.cat_tia(model)
        assert pruner.lay_mask() is not None

    def test_lay_lich_su(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5)
        weights = np.random.randn(100)
        model = type("Model", (), {"_trong_so": weights.copy()})()
        pruner.cat_tia(model)
        assert len(pruner.lay_lich_su()) == 1

    def test_thong_ke(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5, so_vong_lap=3)
        tk = pruner.thong_ke()
        assert tk["che_do"] == "magnitude"
        assert tk["ty_le"] == 0.5
        assert tk["so_vong_lap"] == 3

    def test_repr(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5)
        assert "magnitude" in repr(pruner)

    def test_structured_1d(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="structured", ty_le=0.5)
        weights = np.random.randn(100)
        model = type("Model", (), {"_trong_so": weights.copy()})()
        result = pruner.cat_tia(model)
        assert result["mask"] is not None

    def test_pruning_with_evaluation(self):
        from vietnamese_ai.compression.pruning import CatTiaMoHinh

        pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)

        class MockModel:
            _trong_so = np.random.randn(100)

            def danh_gia(self, X, y):
                return 0.95

            def huan_luyen(self, X, y):
                pass

        model = MockModel()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = pruner.cat_tia(model, X, y)
        assert result["hieu_suat_truoc"] == 0.95
        assert result["hieu_suat_sau"] is not None


class TestHocRutGon:
    def test_creation(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        distiller = HocRutGon(teacher=teacher, nhiet_do=3.0, alpha=0.5)
        assert distiller.nhiet_do == 3.0
        assert distiller.alpha == 0.5

    def test_invalid_ham_loss(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        with pytest.raises(ValueError, match="ham_loss phải là"):
            HocRutGon(teacher=teacher, ham_loss="invalid")

    def test_temperature_scale(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        distiller = HocRutGon(teacher=teacher, nhiet_do=2.0)
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        scaled = distiller._temperature_scale(probs)
        assert scaled.shape == probs.shape
        assert np.allclose(np.sum(scaled, axis=1), 1.0)

    def test_thong_ke(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        distiller = HocRutGon(teacher=teacher, nhiet_do=3.0, alpha=0.5, ham_loss="mse")
        tk = distiller.thong_ke()
        assert tk["nhiet_do"] == 3.0
        assert tk["ham_loss"] == "mse"

    def test_repr(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        distiller = HocRutGon(teacher=teacher)
        assert "HocRutGon" in repr(distiller)

    def test_lay_lich_su(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        distiller = HocRutGon(teacher=teacher)
        assert distiller.lay_lich_su() == []

    def test_huan_luyen_with_danh_gia(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        class MockTeacher:
            def du_doan_xac_suat(self, X):
                return np.random.dirichlet([1, 1], len(X))

            def danh_gia(self, X, y):
                return 0.9

        class MockStudent:
            def huan_luyen(self, X, y):
                pass

            def danh_gia(self, X, y):
                return 0.85

        teacher = MockTeacher()
        distiller = HocRutGon(teacher=teacher)
        student = MockStudent()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = distiller.huan_luyen(student, X, y)
        assert result["teacher_acc"] == 0.9
        assert result["student_acc"] == 0.85

    def test_huan_luyen_with_soft_labels(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        class MockTeacher:
            def du_doan_xac_suat(self, X):
                return np.random.dirichlet([1, 1], len(X))

            def danh_gia(self, X, y):
                return 0.9

        class MockStudentSoft:
            def huan_luyen_voi_soft_labels(self, X, probs, y):
                pass

            def huan_luyen(self, X, y):
                pass

            def danh_gia(self, X, y):
                return 0.88

        teacher = MockTeacher()
        distiller = HocRutGon(teacher=teacher)
        student = MockStudentSoft()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = distiller.huan_luyen(student, X, y)
        assert result["student_acc"] == 0.88

    def test_huan_luyen_with_val(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        class MockTeacher:
            def du_doan_xac_suat(self, X):
                return np.random.dirichlet([1, 1], len(X))

            def danh_gia(self, X, y):
                return 0.9

        class MockStudent:
            def huan_luyen(self, X, y):
                pass

            def danh_gia(self, X, y):
                return 0.85

        teacher = MockTeacher()
        distiller = HocRutGon(teacher=teacher)
        student = MockStudent()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        X_val = np.random.randn(5, 5)
        y_val = np.random.randint(0, 2, 5)
        result = distiller.huan_luyen(student, X, y, X_val, y_val)
        assert result["student_val_acc"] is not None

    def test_huan_luyen_teacher_du_doan(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        class MockTeacher:
            def du_doan(self, X):
                return np.random.randint(0, 2, len(X))

            def danh_gia(self, X, y):
                return 0.9

        class MockStudent:
            def huan_luyen(self, X, y):
                pass

            def danh_gia(self, X, y):
                return 0.85

        teacher = MockTeacher()
        distiller = HocRutGon(teacher=teacher, nhiet_do=1.0)
        student = MockStudent()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = distiller.huan_luyen(student, X, y)
        assert result["student_acc"] == 0.85

    def test_huan_luyen_ensemble(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        class MockTeacher:
            def du_doan_xac_suat(self, X):
                return np.random.dirichlet([1, 1], len(X))

            def danh_gia(self, X, y):
                return 0.9

        class MockStudent:
            def huan_luyen(self, X, y):
                pass

            def danh_gia(self, X, y):
                return 0.85

        teacher = MockTeacher()
        distiller = HocRutGon(teacher=teacher)
        student = MockStudent()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = distiller.huan_luyen_ensemble(student, [teacher, teacher], X, y)
        assert result["so_teachers"] == 2

    def test_teacher_no_predict(self):
        from vietnamese_ai.compression.distillation import HocRutGon

        teacher = type("Teacher", (), {})()
        distiller = HocRutGon(teacher=teacher)
        with pytest.raises(ValueError, match="Teacher không có phương thức dự đoán"):
            distiller._lay_teacher_probs(np.random.randn(5, 3))


class TestMultiGPUTrainer:
    def test_creation(self):
        from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer

        trainer = MultiGPUTrainer()
        assert isinstance(trainer.so_gpu, int)

    def test_thong_tin_gpu(self):
        from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer

        trainer = MultiGPUTrainer()
        info = trainer.thong_tin_gpu()
        assert isinstance(info, list)

    def test_huan_luyen_cpu_fallback(self):
        from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer

        trainer = MultiGPUTrainer()
        model = type("Model", (), {"huan_luyen": lambda s, X, y: None})()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = trainer.huan_luyen(model, X, y)
        assert "thiet_bi" in result
