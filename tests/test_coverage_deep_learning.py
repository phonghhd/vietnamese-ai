import numpy as np
import pytest


class TestLopDense:
    def test_basic_creation(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5)
        assert layer.dau_vao == 10
        assert layer.dau_ra == 5
        assert layer.trong_so.shape == (10, 5)
        assert layer.bias.shape == (1, 5)

    def test_forward_no_activation(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5)
        X = np.random.randn(3, 10)
        output = layer.tien(X)
        assert output.shape == (3, 5)

    def test_forward_relu(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="relu")
        X = np.random.randn(3, 10)
        output = layer.tien(X)
        assert output.shape == (3, 5)
        assert np.all(output >= 0)

    def test_forward_sigmoid(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="sigmoid")
        X = np.random.randn(3, 10)
        output = layer.tien(X)
        assert output.shape == (3, 5)
        assert np.all(output >= 0) and np.all(output <= 1)

    def test_forward_tanh(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="tanh")
        X = np.random.randn(3, 10)
        output = layer.tien(X)
        assert output.shape == (3, 5)
        assert np.all(output >= -1) and np.all(output <= 1)

    def test_forward_softmax(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="softmax")
        X = np.random.randn(3, 10)
        output = layer.tien(X)
        assert output.shape == (3, 5)
        assert np.allclose(np.sum(output, axis=1), 1.0)

    def test_call(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5)
        X = np.random.randn(3, 10)
        output = layer(X)
        assert output.shape == (3, 5)

    def test_backward_relu(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="relu")
        X = np.random.randn(4, 10)
        layer.tien(X)
        grad = np.random.randn(4, 5)
        result = layer.ve(grad, toc_do_hoc=0.01)
        assert result.shape == (4, 10)

    def test_backward_sigmoid(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="sigmoid")
        X = np.random.randn(4, 10)
        layer.tien(X)
        grad = np.random.randn(4, 5)
        result = layer.ve(grad, toc_do_hoc=0.01)
        assert result.shape == (4, 10)

    def test_backward_tanh(self):
        from vietnamese_ai.deep_learning.layers import LopDense

        layer = LopDense(10, 5, ham_kich_hoat="tanh")
        X = np.random.randn(4, 10)
        layer.tien(X)
        grad = np.random.randn(4, 5)
        result = layer.ve(grad, toc_do_hoc=0.01)
        assert result.shape == (4, 10)


class TestLopDropout:
    def test_training(self):
        from vietnamese_ai.deep_learning.layers import LopDropout

        layer = LopDropout(ty_le=0.5)
        X = np.ones((10, 10))
        output = layer(X, training=True)
        assert output.shape == X.shape

    def test_inference(self):
        from vietnamese_ai.deep_learning.layers import LopDropout

        layer = LopDropout(ty_le=0.5)
        X = np.ones((10, 10))
        output = layer(X, training=False)
        assert np.array_equal(output, X)

    def test_backward_with_mask(self):
        from vietnamese_ai.deep_learning.layers import LopDropout

        layer = LopDropout(ty_le=0.5)
        X = np.ones((10, 10))
        layer(X, training=True)
        grad = np.ones((10, 10))
        result = layer.ve(grad)
        assert result.shape == (10, 10)

    def test_backward_without_mask(self):
        from vietnamese_ai.deep_learning.layers import LopDropout

        layer = LopDropout(ty_le=0.5)
        grad = np.ones((10, 10))
        result = layer.ve(grad)
        assert np.array_equal(result, grad)


class TestLopBatchNorm:
    def test_training(self):
        from vietnamese_ai.deep_learning.layers import LopBatchNorm

        layer = LopBatchNorm(10)
        X = np.random.randn(16, 10)
        output = layer(X, training=True)
        assert output.shape == X.shape

    def test_inference(self):
        from vietnamese_ai.deep_learning.layers import LopBatchNorm

        layer = LopBatchNorm(10)
        X_train = np.random.randn(16, 10)
        layer(X_train, training=True)
        X_test = np.random.randn(4, 10)
        output = layer(X_test, training=False)
        assert output.shape == X_test.shape


class TestMangSau:
    def test_creation(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        mang = MangSau(lop_an=[32, 16], so_vong=5)
        assert mang.lop_an == [32, 16]
        assert mang.so_vong == 5

    def test_co_pytorch(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        mang = MangSau()
        assert isinstance(mang.co_pytorch, bool)

    def test_du_doan_before_train(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        mang = MangSau()
        with pytest.raises(RuntimeError, match="Chưa huấn luyện"):
            mang.du_doan(np.random.randn(1, 5))

    def test_thiet_bi_manual(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        mang = MangSau(thiet_bi="cpu")
        assert mang.thiet_bi == "cpu"

    def test_lay_lich_su_loss(self):
        from vietnamese_ai.deep_learning.mang_sau import MangSau

        mang = MangSau()
        assert mang.lay_lich_su_loss() == []
