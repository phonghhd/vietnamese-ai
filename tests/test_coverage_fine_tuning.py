import json
import os
import tempfile

import numpy as np
import pytest


class TestLoRALayer:
    def test_basic_creation(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        layer = LoRALayer(dau_vao=64, dau_ra=32, rank=8, alpha=16.0)
        assert layer.dau_vao == 64
        assert layer.dau_ra == 32
        assert layer.rank == 8
        assert layer.alpha == 16.0
        assert layer.scaling == 2.0

    def test_invalid_rank_zero(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        with pytest.raises(ValueError, match="rank phải > 0"):
            LoRALayer(dau_vao=64, dau_ra=32, rank=0)

    def test_invalid_rank_negative(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        with pytest.raises(ValueError, match="rank phải > 0"):
            LoRALayer(dau_vao=64, dau_ra=32, rank=-1)

    def test_rank_too_large(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        with pytest.raises(ValueError, match="rank.*phải <= min"):
            LoRALayer(dau_vao=8, dau_ra=16, rank=20)

    def test_forward(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        layer = LoRALayer(dau_vao=64, dau_ra=32, rank=8)
        X = np.random.randn(4, 64)
        output = layer.tien(X)
        assert output.shape == (4, 32)

    def test_backward(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        layer = LoRALayer(dau_vao=64, dau_ra=32, rank=8)
        X = np.random.randn(4, 64)
        layer.tien(X)
        grad = np.random.randn(4, 32)
        result = layer.ve(grad, toc_do_hoc=0.01)
        assert result.shape == (4, 64)
        assert layer._grad_A is not None
        assert layer._grad_B is not None

    def test_gop_trong_so(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        layer = LoRALayer(dau_vao=64, dau_ra=32, rank=8)
        delta = layer.gop_trong_so()
        assert delta.shape == (64, 32)

    def test_so_tham_so(self):
        from vietnamese_ai.fine_tuning.lora import LoRALayer

        layer = LoRALayer(dau_vao=64, dau_ra=32, rank=8)
        assert layer.so_tham_so() == 64 * 8 + 8 * 32


class TestLoRAAdapter:
    def test_basic_creation(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=16, alpha=16.0)
        assert adapter.rank == 16
        assert adapter.alpha == 16.0

    def test_invalid_rank(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        with pytest.raises(ValueError, match="rank phải > 0"):
            LoRAAdapter(rank=0)

    def test_add_layers(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        adapter.them_layer("q_proj", 64, 64)
        adapter.them_layer("v_proj", 64, 64)
        assert len(adapter._layers) == 2
        assert adapter._da_tao is True

    def test_forward(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        adapter.them_layer("q_proj", 64, 32)
        X = np.random.randn(4, 64)
        output = adapter.tien("q_proj", X)
        assert output.shape == (4, 32)

    def test_forward_nonexistent_layer(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        with pytest.raises(KeyError, match="Layer.*không tồn tại"):
            adapter.tien("nonexistent", np.random.randn(4, 64))

    def test_gop_trong_so(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        adapter.them_layer("q_proj", 64, 32)
        adapter.them_layer("v_proj", 64, 32)
        deltas = adapter.gop_trong_so()
        assert "q_proj" in deltas
        assert "v_proj" in deltas
        assert deltas["q_proj"].shape == (64, 32)

    def test_so_tham_so(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        adapter.them_layer("q_proj", 64, 32)
        expected = 64 * 8 + 8 * 32
        assert adapter.so_tham_so() == expected

    def test_ty_le_trainable(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        adapter.them_layer("q_proj", 64, 32)
        ratio = adapter.ty_le_trainable(10000)
        assert ratio > 0

    def test_save_load(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8, alpha=16.0)
        adapter.them_layer("q_proj", 64, 32)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            adapter.luu(path)
            loaded = LoRAAdapter.tai(path)
            assert loaded.rank == 8
            assert loaded.alpha == 16.0
            assert "q_proj" in loaded._layers
        finally:
            os.unlink(path)

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.lora import LoRAAdapter

        adapter = LoRAAdapter(rank=8)
        adapter.them_layer("q_proj", 64, 32)
        tk = adapter.thong_ke()
        assert "rank" in tk
        assert "so_layers" in tk
        assert "so_tham_so" in tk
        assert "cac_layers" in tk


class TestQLoRAAdapter:
    def test_creation_4bit(self):
        from vietnamese_ai.fine_tuning.lora import QLoRAAdapter

        qlora = QLoRAAdapter(rank=8, bits=4)
        assert qlora.bits == 4

    def test_creation_8bit(self):
        from vietnamese_ai.fine_tuning.lora import QLoRAAdapter

        qlora = QLoRAAdapter(rank=8, bits=8)
        assert qlora.bits == 8

    def test_invalid_bits(self):
        from vietnamese_ai.fine_tuning.lora import QLoRAAdapter

        with pytest.raises(ValueError, match="bits phải là 4 hoặc 8"):
            QLoRAAdapter(rank=8, bits=16)

    def test_quantize_4bit(self):
        from vietnamese_ai.fine_tuning.lora import QLoRAAdapter

        qlora = QLoRAAdapter(rank=8, bits=4)
        weights = np.random.randn(10, 10)
        result = qlora.quantize_weights(weights)
        assert result["bits"] == 4
        assert "quantized" in result
        assert "scale" in result

    def test_quantize_8bit(self):
        from vietnamese_ai.fine_tuning.lora import QLoRAAdapter

        qlora = QLoRAAdapter(rank=8, bits=8)
        weights = np.random.randn(10, 10)
        result = qlora.quantize_weights(weights)
        assert result["bits"] == 8

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.lora import QLoRAAdapter

        qlora = QLoRAAdapter(rank=8, bits=4)
        qlora.them_layer("q_proj", 64, 32)
        tk = qlora.thong_ke()
        assert tk["bits"] == 4
        assert tk["loai"] == "QLoRA"


class TestInstructionDataset:
    def test_creation_alpaca(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="alpaca")
        assert ds.che_do == "alpaca"

    def test_creation_sharegpt(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="sharegpt")
        assert ds.che_do == "sharegpt"

    def test_invalid_che_do(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        with pytest.raises(ValueError, match="che_do phải là"):
            InstructionDataset(che_do="invalid")

    def test_tai_tu_list(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        data = [
            {"instruction": "test1", "input": "", "output": "out1"},
            {"instruction": "test2", "input": "in2", "output": "out2"},
        ]
        count = ds.tai_tu_list(data)
        assert count == 2
        assert len(ds) == 2

    def test_them_mau(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.them_mau({"instruction": "test", "input": "", "output": "out"})
        assert len(ds) == 1

    def test_getitem(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.tai_tu_list([{"instruction": "test", "input": "", "output": "out"}])
        assert ds[0]["instruction"] == "test"

    def test_chia_du_lieu(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        data = [{"instruction": f"t{i}", "input": "", "output": f"o{i}"} for i in range(20)]
        ds.tai_tu_list(data)
        result = ds.chia_du_lieu(ty_le_val=0.2)
        assert result["so_train"] + result["so_val"] == 20
        assert ds.train is not None
        assert ds.val is not None

    def test_chia_du_lieu_trong(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        with pytest.raises(RuntimeError, match="Dataset trống"):
            ds.chia_du_lieu()

    def test_train_property_no_split(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.tai_tu_list([{"instruction": "t", "input": "", "output": "o"}])
        assert len(ds.train) == 1

    def test_format_alpaca_no_input(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        result = ds.format_alpaca({"instruction": "Viết bài", "input": "", "output": "Bài viết"})
        assert "Viết bài" in result
        assert "Bài viết" in result

    def test_format_alpaca_with_input(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        result = ds.format_alpaca({"instruction": "Tóm tắt", "input": "Văn bản dài", "output": "Tóm tắt"})
        assert "Tóm tắt" in result
        assert "Văn bản dài" in result

    def test_format_sharegpt(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="sharegpt")
        mau = {"conversations": [
            {"from": "human", "value": "Xin chào"},
            {"from": "gpt", "value": "Chào bạn"},
        ]}
        result = ds.format_sharegpt(mau)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_format_tat_ca_alpaca(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="alpaca")
        ds.tai_tu_list([
            {"instruction": "Test", "input": "", "output": "Out"},
        ])
        results = ds.format_tat_ca()
        assert len(results) == 1

    def test_format_tat_ca_sharegpt(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="sharegpt")
        ds.tai_tu_list([
            {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
        ])
        results = ds.format_tat_ca()
        assert len(results) == 1

    def test_chuyen_doi_sharegpt_sang_alpaca(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="sharegpt")
        ds.tai_tu_list([
            {"conversations": [
                {"from": "human", "value": "Câu hỏi"},
                {"from": "gpt", "value": "Trả lời"},
            ]},
        ])
        result = ds.chuyen_doi_sharegpt_sang_alpaca()
        assert len(result) == 1
        assert result[0]["instruction"] == "Câu hỏi"
        assert result[0]["output"] == "Trả lời"

    def test_chuyen_doi_wrong_mode(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="alpaca")
        with pytest.raises(RuntimeError, match="Chỉ convert"):
            ds.chuyen_doi_sharegpt_sang_alpaca()

    def test_tai_file_json(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        data = [{"instruction": "test", "input": "", "output": "out"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            count = ds.tai_file(path)
            assert count == 1
        finally:
            os.unlink(path)

    def test_tai_file_jsonl(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"instruction": "t1", "input": "", "output": "o1"}\n')
            f.write('{"instruction": "t2", "input": "", "output": "o2"}\n')
            path = f.name
        try:
            count = ds.tai_file(path)
            assert count == 2
        finally:
            os.unlink(path)

    def test_tai_file_not_found(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        with pytest.raises(FileNotFoundError):
            ds.tai_file("/nonexistent/file.json")

    def test_tai_file_dict_wrapped(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        data = {"data": [{"instruction": "test", "input": "", "output": "out"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            count = ds.tai_file(path)
            assert count == 1
        finally:
            os.unlink(path)

    def test_luu(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.tai_tu_list([{"instruction": "test", "input": "", "output": "out"}])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = ds.luu(path)
            assert os.path.exists(result)
        finally:
            os.unlink(path)

    def test_thong_ke_alpaca(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.tai_tu_list([
            {"instruction": "Test instruction", "input": "", "output": "Test output"},
        ])
        tk = ds.thong_ke()
        assert tk["so_mau"] == 1
        assert "do_dai_tb_instruction" in tk

    def test_thong_ke_sharegpt(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="sharegpt")
        ds.tai_tu_list([
            {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
        ])
        tk = ds.thong_ke()
        assert tk["so_mau"] == 1

    def test_thong_ke_trong(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        tk = ds.thong_ke()
        assert tk["so_mau"] == 0


class TestDPOTrainer:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        dpo = DPOTrainer(so_vong=1, beta=0.1)
        assert dpo.beta == 0.1

    def test_invalid_beta(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        with pytest.raises(ValueError, match="beta phải > 0"):
            DPOTrainer(beta=0)

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        dpo = DPOTrainer(so_vong=1, beta=0.1, label_smoothing=0.1)
        tk = dpo.thong_ke()
        assert tk["so_vong"] == 1
        assert tk["beta"] == 0.1
        assert tk["label_smoothing"] == 0.1

    def test_lay_lich_su(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        dpo = DPOTrainer()
        history = dpo.lay_lich_su()
        assert "train_loss" in history
        assert "chosen_rewards" in history


class TestSFTTrainer:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer(so_vong=3, toc_do_hoc=2e-5)
        assert trainer.so_vong == 3

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer()
        tk = trainer.thong_ke()
        assert "so_vong" in tk
        assert "global_step" in tk

    def test_lay_lich_su(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer()
        history = trainer.lay_lich_su()
        assert "train_loss" in history


class TestRewardModel:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        rm = RewardModel(toc_do_hoc=1e-5)
        assert rm.toc_do_hoc == 1e-5

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        rm = RewardModel()
        tk = rm.thong_ke()
        assert "toc_do_hoc" in tk
        assert "score_mean" in tk

    def test_diem_danh_gia_numpy_fallback(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        rm = RewardModel()
        results = rm.diem_danh_gia(None, ["test1", "test2"])
        assert len(results) == 2
        assert "score" in results[0]


class TestInstructionTuningTrainer:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer(so_vong=3)
        assert trainer.so_vong == 3

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer()
        tk = trainer.thong_ke()
        assert "so_vong" in tk
        assert "global_step" in tk

    def test_lay_lich_su(self):
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer()
        history = trainer.lay_lich_su()
        assert "train_loss" in history


class TestHuggingFaceWrapper:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        hf = HuggingFaceWrapper()
        assert hf.da_tai is False

    def test_danh_sach_models_viet(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        hf = HuggingFaceWrapper()
        models = hf.danh_sach_models_viet()
        assert "phobert" in models

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        hf = HuggingFaceWrapper()
        tk = hf.thong_ke()
        assert tk["da_tai"] is False
        assert "available_vi_models" in tk
        assert "supported_tasks" in tk

    def test_du_doan_not_loaded(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        hf = HuggingFaceWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            hf.du_doan(["test"])

    def test_ma_hoa_not_loaded(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        hf = HuggingFaceWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            hf.ma_hoa(["test"])

    def test_luu_model_not_loaded(self):
        from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper

        hf = HuggingFaceWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            hf.luu_model("/tmp/test")


class TestUnslothWrapper:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        assert uw._da_tai is False

    def test_danh_sach_models(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        models = uw.danh_sach_models()
        assert "llama3-8b" in models

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        tk = uw.thong_ke()
        assert "co_unsloth" in tk
        assert "available_models" in tk

    def test_tai_mo_hinh_not_installed(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        uw._co_unsloth = False
        with pytest.raises(ImportError, match="Unsloth chưa cài"):
            uw.tai_mo_hinh("test")

    def test_config_lora_not_loaded(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            uw.config_lora()

    def test_fine_tune_not_loaded(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            uw.fine_tune(None)

    def test_luu_model_not_loaded(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            uw.luu_model("/tmp/test")

    def test_xuat_gguf_not_loaded(self):
        from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

        uw = UnslothWrapper()
        with pytest.raises(RuntimeError, match="Chưa tải model"):
            uw.xuat_gguf("/tmp/test")


class TestPEFTConfig:
    def test_lora_config(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.lora(rank=16, alpha=16.0)
        assert config.rank == 16

    def test_qlora_config(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.qlora(rank=8, bits=4)
        assert config.rank == 8

    def test_to_dict(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.lora()
        d = config.to_dict()
        assert "rank" in d

    def test_from_dict(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        d = {"rank": 16, "alpha": 16.0, "dropout": 0.0, "target_modules": ["q_proj"], "loftq_config": None}
        config = PEFTConfig.from_dict(d)
        assert config.rank == 16


class TestRLHFPipeline:
    def test_creation(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        assert pipeline is not None

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        tk = pipeline.thong_ke()
        assert isinstance(tk, dict)


class TestHuanLuyenPyTorch:
    def test_thong_ke(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu", so_vong=2)
        tk = trainer.thong_ke()
        assert tk["thiet_bi"] == "cpu"
        assert tk["so_vong"] == 2

    def test_co_pytorch(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu")
        assert trainer.co_pytorch is True

    def test_co_gpu_cpu(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu")
        assert trainer.co_gpu is False

    def test_lay_lich_su(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu")
        history = trainer.lay_lich_su()
        assert "train_loss" in history

    def test_du_doan_before_train(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu")
        with pytest.raises(RuntimeError, match="Chưa huấn luyện"):
            trainer.du_doan(np.random.randn(1, 5))

    def test_du_doan_xac_suat_before_train(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu")
        with pytest.raises(RuntimeError, match="Chưa huấn luyện"):
            trainer.du_doan_xac_suat(np.random.randn(1, 5))

    def test_luu_checkpoint_before_train(self):
        pytest.importorskip("torch")
        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        trainer = HuanLuyenPyTorch(thiet_bi="cpu")
        with pytest.raises(RuntimeError, match="Chưa huấn luyện"):
            trainer.luu_checkpoint("/tmp/test.pt")

    def test_full_training_cycle(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        model = nn.Sequential(nn.Linear(5, 3))
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 3, 20)

        trainer = HuanLuyenPyTorch(
            thiet_bi="cpu", so_vong=2, kich_thuoc_batch=8, scheduler="cosine"
        )
        result = trainer.huan_luyen(model, X, y)
        assert result["so_epoch"] == 2
        assert "train_loss_min" in result

        preds = trainer.du_doan(X)
        assert len(preds) == 20

        probs = trainer.du_doan_xac_suat(X)
        assert probs.shape == (20, 3)

    def test_training_with_validation(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        model = nn.Sequential(nn.Linear(5, 3))
        X_train = np.random.randn(20, 5).astype(np.float32)
        y_train = np.random.randint(0, 3, 20)
        X_val = np.random.randn(5, 5).astype(np.float32)
        y_val = np.random.randint(0, 3, 5)

        trainer = HuanLuyenPyTorch(
            thiet_bi="cpu", so_vong=2, kich_thuoc_batch=8,
            early_stopping=5, scheduler="linear"
        )
        result = trainer.huan_luyen(model, X_train, y_train, X_val, y_val)
        assert result["val_loss_min"] is not None

    def test_training_with_step_scheduler(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        model = nn.Sequential(nn.Linear(5, 3))
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 3, 20)

        trainer = HuanLuyenPyTorch(
            thiet_bi="cpu", so_vong=2, kich_thuoc_batch=8, scheduler="step"
        )
        result = trainer.huan_luyen(model, X, y)
        assert result["so_epoch"] == 2

    def test_checkpoint_save_load(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        model = nn.Sequential(nn.Linear(5, 3))
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 3, 20)

        trainer = HuanLuyenPyTorch(thiet_bi="cpu", so_vong=2, kich_thuoc_batch=8)
        trainer.huan_luyen(model, X, y)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.luu_checkpoint(path)
            new_model = nn.Sequential(nn.Linear(5, 3))
            trainer.tai_checkpoint(path, new_model)
        finally:
            os.unlink(path)

    def test_training_with_callback(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        model = nn.Sequential(nn.Linear(5, 3))
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 3, 20)
        callback_args = []

        def callback(epoch, loss, acc):
            callback_args.append((epoch, loss, acc))

        trainer = HuanLuyenPyTorch(thiet_bi="cpu", so_vong=2, kich_thuoc_batch=8)
        trainer.huan_luyen(model, X, y, callback=callback)
        assert len(callback_args) == 2

    def test_gradient_accumulation(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch

        model = nn.Sequential(nn.Linear(5, 3))
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 3, 20)

        trainer = HuanLuyenPyTorch(
            thiet_bi="cpu", so_vong=2, kich_thuoc_batch=4, gradient_accumulation=2
        )
        result = trainer.huan_luyen(model, X, y)
        assert result["so_epoch"] == 2
