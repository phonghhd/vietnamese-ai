"""Test suite cho Phase 7-8 - PyTorch Trainer, Unsloth, GGUF, Transformer, Tokenizer."""

from pathlib import Path

import numpy as np
import pytest

from vietnamese_ai.datasets.sample_data import DuLieuMau

# ============================================================
# Phase 7: PyTorch Trainer
# ============================================================


class TestHuanLuyenPyTorch:
    def test_khoi_tao(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(so_vong=5, kich_thuoc_batch=32)
        assert trainer.so_vong == 5
        assert trainer.thiet_bi in ("cpu", "cuda", "mps")

    def test_khoi_tao_khong_pytorch(self):
        from vietnamese_ai.fine_tuning.pytorch_trainer import _CO_PYTORCH

        if _CO_PYTORCH:
            pytest.skip("PyTorch đã cài, không test ImportError")

    def test_huan_luyen(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2)
        )

        trainer = HuanLuyenPyTorch(so_vong=10, kich_thuoc_batch=32, thiet_bi="cpu")
        ket_qua = trainer.huan_luyen(model, X_train, y_train, X_test, y_test)

        assert ket_qua["tong_thoi_gian"] > 0
        assert ket_qua["so_epoch"] == 10
        assert ket_qua["train_loss_min"] > 0

    def test_du_doan(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

        trainer = HuanLuyenPyTorch(so_vong=5, thiet_bi="cpu")
        trainer.huan_luyen(model, X[:80], y[:80])

        du_doan = trainer.du_doan(X[80:])
        assert len(du_doan) == 20

    def test_du_doan_xac_suat(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=100, so_dac_trung=5)
        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

        trainer = HuanLuyenPyTorch(so_vong=5, thiet_bi="cpu")
        trainer.huan_luyen(model, X[:80], y[:80])

        xac_suat = trainer.du_doan_xac_suat(X[80:])
        assert xac_suat.shape == (20, 2)
        assert np.allclose(xac_suat.sum(axis=1), 1.0, atol=1e-5)

    def test_checkpoint(self, tmp_path):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=5)
        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

        trainer = HuanLuyenPyTorch(so_vong=5, thiet_bi="cpu")
        trainer.huan_luyen(model, X, y)

        duong_dan = str(tmp_path / "checkpoint.pt")
        trainer.luu_checkpoint(duong_dan)
        assert Path(duong_dan).exists()

        model2 = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
        trainer2 = HuanLuyenPyTorch(so_vong=5, thiet_bi="cpu")
        trainer2.tai_checkpoint(duong_dan, model2)
        assert trainer2._model is not None

    def test_lich_su(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=5)
        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

        trainer = HuanLuyenPyTorch(so_vong=10, thiet_bi="cpu")
        trainer.huan_luyen(model, X, y)

        lich_su = trainer.lay_lich_su()
        assert len(lich_su["train_loss"]) == 10
        assert len(lich_su["train_acc"]) == 10

    def test_thong_ke(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(so_vong=5, hon_lep=False)
        tk = trainer.thong_ke()
        assert tk["so_vong"] == 5
        assert "thiet_bi" in tk

    def test_early_stopping(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=80, so_dac_trung=5)
        X_train, X_val = X[:60], X[60:]
        y_train, y_val = y[:60], y[60:]

        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
        trainer = HuanLuyenPyTorch(
            so_vong=100, early_stopping=3, thiet_bi="cpu"
        )
        ket_qua = trainer.huan_luyen(model, X_train, y_train, X_val, y_val)
        assert ket_qua["so_epoch"] <= 100


# ============================================================
# Phase 7: Unsloth Wrapper
# ============================================================


class TestUnslothWrapper:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        wrapper = UnslothWrapper()
        assert wrapper is not None

    def test_danh_sach_models(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        wrapper = UnslothWrapper()
        models = wrapper.danh_sach_models()
        assert "llama3-8b" in models
        assert "mistral-7b" in models

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        wrapper = UnslothWrapper()
        tk = wrapper.thong_ke()
        assert "co_unsloth" in tk
        assert "available_models" in tk
        assert len(tk["available_models"]) >= 5

    def test_tai_model_khong_unsloth(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        wrapper = UnslothWrapper()
        if wrapper.co_unsloth:
            pytest.skip("Unsloth đã cài")
        with pytest.raises(ImportError):
            wrapper.tai_mo_hinh("llama3-8b")


# ============================================================
# Phase 7: HuggingFace Wrapper
# ============================================================


class TestHuggingFaceWrapper:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        wrapper = HuggingFaceWrapper()
        assert wrapper is not None

    def test_danh_sach_models(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        wrapper = HuggingFaceWrapper()
        models = wrapper.danh_sach_models_viet()
        assert "phobert" in models
        assert "vit5-base" in models

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        wrapper = HuggingFaceWrapper()
        tk = wrapper.thong_ke()
        assert "da_tai" in tk
        assert "available_vi_models" in tk
        assert "supported_tasks" in tk


# ============================================================
# Phase 8: GGUF Export
# ============================================================


class TestXuatGGUF:
    def test_khoi_tao(self):
        from vietnamese_ai.export.gguf import XuatGGUF

        xuat = XuatGGUF()
        assert xuat is not None

    def test_danh_sach_quantization(self):
        from vietnamese_ai.export.gguf import XuatGGUF

        xuat = XuatGGUF()
        qt = xuat.danh_sach_quantization()
        assert "f32" in qt
        assert "f16" in qt
        assert "q8_0" in qt
        assert "q4_k_m" in qt

    def test_xuat_numpy(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        trong_so = {
            "layer1.weight": np.random.randn(16, 8).astype(np.float32),
            "layer1.bias": np.random.randn(16).astype(np.float32),
            "layer2.weight": np.random.randn(2, 16).astype(np.float32),
        }

        xuat = XuatGGUF()
        duong_dan = str(tmp_path / "model.gguf")
        ket_qua = xuat.xuat_tu_numpy(trong_so, duong_dan, quantization="f16")
        assert Path(ket_qua).exists()
        assert Path(ket_qua).stat().st_size > 0

    def test_xuat_quantization(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        trong_so = {"weight": np.random.randn(10, 5).astype(np.float32)}
        xuat = XuatGGUF()

        for qt in ["f32", "f16", "q8_0", "q4_k_m"]:
            duong_dan = str(tmp_path / f"model_{qt}.gguf")
            xuat.xuat_tu_numpy(trong_so, duong_dan, quantization=qt)
            assert Path(duong_dan).exists()

    def test_doc_metadata(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        trong_so = {"w": np.random.randn(4, 4).astype(np.float32)}
        xuat = XuatGGUF()
        duong_dan = str(tmp_path / "model.gguf")
        xuat.xuat_tu_numpy(trong_so, duong_dan, thong_tin={"name": "test_model"})

        meta = xuat.doc_metadata(duong_dan)
        assert meta["magic"] == "GGUF"
        assert meta["version"] == 3
        assert meta["n_tensors"] == 1

    def test_doc_metadata_khong_ton_tai(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        xuat = XuatGGUF()
        with pytest.raises(FileNotFoundError):
            xuat.doc_metadata("/nonexistent/file.gguf")

    def test_tao_config_llama_cpp(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        xuat = XuatGGUF()
        duong_dan = str(tmp_path / "llama")
        xuat.tao_config_llama_cpp("my_model", duong_dan)
        assert (Path(duong_dan) / "config.json").exists()
        assert (Path(duong_dan) / "run.sh").exists()

    def test_quantization_invalid(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        xuat = XuatGGUF()
        with pytest.raises(ValueError):
            xuat.xuat_tu_numpy({"w": np.zeros(1)}, str(tmp_path / "m.gguf"), quantization="invalid")


# ============================================================
# Phase 8: MultiHeadAttention
# ============================================================


class TestMultiHeadAttention:
    def test_khoi_tao(self):
        from vietnamese_ai.transformer.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=64, so_dau=4)
        assert mha.d_model == 64
        assert mha.so_dau == 4
        assert mha.d_k == 16

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.transformer.attention import MultiHeadAttention

        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=64, so_dau=3)

    def test_tien(self):
        from vietnamese_ai.transformer.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=32, so_dau=4)
        Q = np.random.randn(2, 10, 32)
        K = np.random.randn(2, 10, 32)
        V = np.random.randn(2, 10, 32)

        output = mha.tien(Q, K, V)
        assert output.shape == (2, 10, 32)

    def test_attention_weights(self):
        from vietnamese_ai.transformer.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=32, so_dau=4)
        Q = np.random.randn(1, 5, 32)
        mha.tien(Q, Q, Q)

        weights = mha.lay_attention_weights()
        assert weights is not None
        assert weights.shape == (1, 4, 5, 5)

    def test_thong_ke(self):
        from vietnamese_ai.transformer.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=64, so_dau=8)
        tk = mha.thong_ke()
        assert tk["d_model"] == 64
        assert tk["so_dau"] == 8


# ============================================================
# Phase 8: TransformerModel
# ============================================================


class TestTransformerModel:
    def test_khoi_tao(self):
        from vietnamese_ai.transformer.model import TransformerModel

        model = TransformerModel(d_model=32, so_dau=4, so_block=2, so_tu_vung=100, so_lop=2)
        assert model.d_model == 32
        assert model.so_block == 2

    def test_tien(self):
        from vietnamese_ai.transformer.model import TransformerModel

        model = TransformerModel(d_model=32, so_dau=4, so_tu_vung=100, so_lop=3, so_block=1)
        input_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        logits = model.tien(input_ids)
        assert logits.shape == (2, 3)

    def test_du_doan(self):
        from vietnamese_ai.transformer.model import TransformerModel

        model = TransformerModel(d_model=32, so_dau=4, so_tu_vung=100, so_lop=2, so_block=1)
        input_ids = np.array([[1, 2, 3], [4, 5, 6]])

        du_doan = model.du_doan(input_ids)
        assert du_doan.shape == (2,)
        assert all(d in [0, 1] for d in du_doan)

    def test_du_doan_xac_suat(self):
        from vietnamese_ai.transformer.model import TransformerModel

        model = TransformerModel(d_model=32, so_dau=4, so_tu_vung=100, so_lop=3, so_block=1)
        input_ids = np.array([[1, 2, 3]])

        xac_suat = model.du_doan_xac_suat(input_ids)
        assert xac_suat.shape == (1, 3)
        assert np.allclose(xac_suat.sum(axis=1), 1.0, atol=1e-5)

    def test_lay_embeddings(self):
        from vietnamese_ai.transformer.model import TransformerModel

        model = TransformerModel(d_model=32, so_dau=4, so_tu_vung=100, so_block=1)
        input_ids = np.array([[1, 2, 3, 4]])

        embeddings = model.lay_embeddings(input_ids)
        assert embeddings.shape == (1, 32)

    def test_thong_ke(self):
        from vietnamese_ai.transformer.model import TransformerModel

        model = TransformerModel(d_model=64, so_dau=8, so_block=4, so_tu_vung=1000)
        tk = model.thong_ke()
        assert "so_tham_so" in tk
        assert "so_tham_so_str" in tk


# ============================================================
# Phase 8: VietnameseTokenizer
# ============================================================


class TestVietnameseTokenizer:
    CORPUS = [
        "học máy là một nhánh của trí tuệ nhân tạo",
        "trí tuệ nhân tạo đang phát triển rất nhanh",
        "mạng nơ-ron nhân tạo mô phỏng não bộ con người",
        "học sâu là một kỹ thuật mạnh mẽ trong học máy",
        "xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng Việt",
    ] * 10

    def test_khoi_tao_bpe(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=200)
        assert tok.che_do == "bpe"

    def test_khoi_tao_wordpiece(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="wordpiece", kich_thuoc_vocab=200)
        assert tok.che_do == "wordpiece"

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        with pytest.raises(ValueError):
            VietnameseTokenizer(che_do="invalid")
        with pytest.raises(ValueError):
            VietnameseTokenizer(kich_thuoc_vocab=10)

    def test_huan_luyen_bpe(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=200)
        ket_qua = tok.huan_luyen(self.CORPUS)
        assert ket_qua["vocab_size"] > 5
        assert tok._da_huan_luyen

    def test_huan_luyen_wordpiece(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="wordpiece", kich_thuoc_vocab=200)
        ket_qua = tok.huan_luyen(self.CORPUS)
        assert ket_qua["vocab_size"] > 5

    def test_ma_hoa_giai_ma(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=300)
        tok.huan_luyen(self.CORPUS)

        ids = tok.ma_hoa("học máy rất hay")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert ids[0] == 2

        text = tok.giai_ma(ids)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_pad(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=200)
        tok.huan_luyen(self.CORPUS)

        ids = tok.ma_hoa("test")
        padded = tok.pad(ids, do_dai=10)
        assert len(padded) == 10

    def test_luu_tai(self, tmp_path):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=200)
        tok.huan_luyen(self.CORPUS)

        duong_dan = str(tmp_path / "tokenizer.json")
        tok.luu(duong_dan)
        assert Path(duong_dan).exists()

        tok2 = VietnameseTokenizer.tai(duong_dan)
        assert tok2._da_huan_luyen
        assert tok2.che_do == "bpe"

        ids = tok2.ma_hoa("học máy")
        assert len(ids) > 0

    def test_thong_ke(self):
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=200)
        tok.huan_luyen(self.CORPUS)

        tk = tok.thong_ke()
        assert tk["da_huan_luyen"] is True
        assert tk["vocab_size"] > 0


# ============================================================
# Integration tests
# ============================================================


class TestPhase78Integration:
    def test_imports(self):
        import vietnamese_ai

        assert vietnamese_ai.__version__ == "9.0.0"
        assert hasattr(vietnamese_ai, "HuanLuyenPyTorch")
        assert hasattr(vietnamese_ai, "UnslothWrapper")
        assert hasattr(vietnamese_ai, "HuggingFaceWrapper")
        assert hasattr(vietnamese_ai, "XuatGGUF")
        assert hasattr(vietnamese_ai, "MultiHeadAttention")
        assert hasattr(vietnamese_ai, "TransformerModel")
        assert hasattr(vietnamese_ai, "VietnameseTokenizer")

    def test_all_exports_count(self):
        import vietnamese_ai

        assert len(vietnamese_ai.__all__) >= 50

    def test_end_to_end_transformer(self):
        from vietnamese_ai.transformer.model import TransformerModel
        from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer

        corpus = [
            "học máy rất thú vị",
            "trí tuệ nhân tạo tuyệt vời",
            "mạng nơ-ron rất mạnh mẽ",
        ] * 20

        tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=100)
        tok.huan_luyen(corpus)

        model = TransformerModel(
            d_model=32, so_dau=4, so_tu_vung=100, so_lop=2, so_block=1
        )

        ids1 = tok.pad(tok.ma_hoa("học máy"), do_dai=10)
        ids2 = tok.pad(tok.ma_hoa("trí tuệ"), do_dai=10)
        input_ids = np.array([ids1, ids2])

        du_doan = model.du_doan(input_ids)
        assert len(du_doan) == 2

        xac_suat = model.du_doan_xac_suat(input_ids)
        assert xac_suat.shape == (2, 2)

    def test_end_to_end_gguf(self, tmp_path):
        from vietnamese_ai.export.gguf import XuatGGUF

        trong_so = {
            "embedding": np.random.randn(100, 32).astype(np.float32),
            "layer1.weight": np.random.randn(32, 16).astype(np.float32),
            "layer1.bias": np.random.randn(16).astype(np.float32),
            "output.weight": np.random.randn(16, 2).astype(np.float32),
        }

        xuat = XuatGGUF()
        duong_dan = str(tmp_path / "model.gguf")
        xuat.xuat_tu_numpy(
            trong_so, duong_dan,
            quantization="q4_k_m",
            thong_tin={"name": "VietnameseAI-Test", "description": "Test model"},
        )

        meta = xuat.doc_metadata(duong_dan)
        assert meta["n_tensors"] == 4
        assert meta["metadata"]["general.name"] == "VietnameseAI-Test"
