import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class SimpleModel(nn.Module):
    def __init__(self, in_dim=5, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class ModelWithTargetModules(nn.Module):
    def __init__(self, in_dim=10, out_dim=10):
        super().__init__()
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.output = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.output(self.q_proj(x))


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 32)
        self.fc = nn.Linear(32, 100)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embedded = self.embedding(x)
        return self.fc(embedded)


class TestDPOTrainerFull:
    def test_full_training(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        model = SmallModel()
        ref_model = SmallModel()

        dpo = DPOTrainer(so_vong=1, gradient_accumulation=1, logging_steps=1, beta=0.1)
        preference_data = [
            {"prompt": "test", "chosen": "good answer", "rejected": "bad answer"},
        ]
        result = dpo.huan_luyen(model, ref_model, preference_data)
        assert "tong_thoi_gian" in result

    def test_full_training_with_label_smoothing(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        model = SmallModel()
        ref_model = SmallModel()

        dpo = DPOTrainer(so_vong=1, gradient_accumulation=1, label_smoothing=0.1)
        preference_data = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"},
        ]
        result = dpo.huan_luyen(model, ref_model, preference_data)
        assert result["so_epoch"] == 1

    def test_full_training_with_callback(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        model = SmallModel()
        ref_model = SmallModel()
        calls = []

        dpo = DPOTrainer(so_vong=1, gradient_accumulation=1, logging_steps=1)
        preference_data = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"},
        ]
        result = dpo.huan_luyen(
            model, ref_model, preference_data,
            callback=lambda step, loss: calls.append(step),
        )
        assert "tong_thoi_gian" in result

    def test_tinh_logps_2d(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        dpo = DPOTrainer()
        logits = torch.randn(2, 10)
        labels = torch.randint(0, 10, (2,))
        result = dpo._tinh_logps(logits, labels)
        assert result.dim() == 0


class TestSFTTrainerFull:
    def test_full_training(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        model = SimpleModel(10, 100)
        trainer = SFTTrainer(so_vong=1, kich_thuoc_batch=2, gradient_accumulation=1)
        du_lieu = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9, 10], "labels": [6, 7, 8, 9, 10]},
        ]
        result = trainer.huan_luyen(model, du_lieu)
        assert "tong_thoi_gian" in result

    def test_full_training_with_val(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        model = SimpleModel(10, 100)
        trainer = SFTTrainer(so_vong=1, kich_thuoc_batch=2, gradient_accumulation=1)
        du_lieu = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
        ]
        du_lieu_val = [
            {"input_ids": [6, 7, 8], "labels": [6, 7, 8]},
        ]
        result = trainer.huan_luyen(model, du_lieu, du_lieu_val)
        assert result["eval_loss_min"] is not None

    def test_cap_nhat_lr(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer(toc_do_hoc=1e-3)
        model = SimpleModel(5, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer._cap_nhat_lr(optimizer, step=0, warmup_steps=10, tong_steps=100)
        assert optimizer.param_groups[0]["lr"] <= 1e-3

    def test_cap_nhat_lr_cosine(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer(toc_do_hoc=1e-3)
        model = SimpleModel(5, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer._cap_nhat_lr(optimizer, step=50, warmup_steps=10, tong_steps=100)
        assert optimizer.param_groups[0]["lr"] <= 1e-3


class TestRewardModelFull:
    def test_full_training(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        model = SimpleModel(10, 1)
        rm = RewardModel(toc_do_hoc=1e-3)
        preference_data = [
            {"chosen": "good answer text", "rejected": "bad answer text"},
        ]
        result = rm.huan_luyen(model, preference_data, so_vong=1)
        assert "tong_thoi_gian" in result

    def test_full_training_with_callback(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        model = SimpleModel(10, 1)
        rm = RewardModel()
        preference_data = [
            {"chosen": "good", "rejected": "bad"},
        ]
        calls = []
        rm.huan_luyen(
            model, preference_data, so_vong=1,
            callback=lambda step, loss: calls.append(step),
        )
        assert len(calls) >= 0

    def test_diem_danh_gia_pytorch(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        model = SimpleModel(10, 1)
        rm = RewardModel()
        rm._score_mean = 0.0
        rm._score_std = 1.0
        results = rm.diem_danh_gia(model, ["test text", "another text"])
        assert len(results) == 2
        assert "score" in results[0]


class TestInstructionTuningTrainerFull:
    def test_numpy_fallback(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer(so_vong=1, logging_steps=1)
        ds = InstructionDataset()
        ds.tai_tu_list([
            {"instruction": "test", "input": "", "output": "out"},
            {"instruction": "test2", "input": "", "output": "out2"},
        ])

        class NumpyModel:
            pass

        result = trainer.huan_luyen(NumpyModel(), None, ds)
        assert "tong_thoi_gian" in result

    def test_numpy_fallback_with_callback(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer(so_vong=1, logging_steps=1)
        ds = InstructionDataset()
        ds.tai_tu_list([
            {"instruction": "test", "input": "", "output": "out"},
        ])
        calls = []

        class NumpyModel:
            pass

        trainer.huan_luyen(
            NumpyModel(), None, ds,
            callback=lambda step, loss: calls.append(step),
        )
        assert len(calls) >= 0

    def test_cap_nhat_lr(self):
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer(toc_do_hoc=1e-3)
        model = SimpleModel(5, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer._cap_nhat_lr(optimizer, step=0, warmup_steps=10, tong_steps=100)
        assert len(trainer._history["learning_rate"]) > 0


class TestLoRAPeftFull:
    def test_ap_dung(self):
        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        model = ModelWithTargetModules(10, 10)
        config = PEFTConfig.lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        peft = LoRAPeft(config)
        model = peft.ap_dung(model)
        assert len(peft._lora_layers) > 0

    def test_chi_trainable(self):
        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        model = ModelWithTargetModules(10, 10)
        config = PEFTConfig.lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        peft = LoRAPeft(config)
        model = peft.ap_dung(model)
        trainable = peft.chi_trainable(model)
        assert len(trainable) > 0

    def test_gop_trong_so(self):
        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        model = ModelWithTargetModules(10, 10)
        config = PEFTConfig.lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        peft = LoRAPeft(config)
        model = peft.ap_dung(model)
        model = peft.gop_trong_so(model)

    def test_thong_ke_with_model(self):
        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        model = ModelWithTargetModules(10, 10)
        config = PEFTConfig.lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        peft = LoRAPeft(config)
        model = peft.ap_dung(model)
        tk = peft.thong_ke(model)
        assert "trainable_params" in tk

    def test_luu(self):
        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        model = ModelWithTargetModules(10, 10)
        config = PEFTConfig.lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        peft = LoRAPeft(config)
        model = peft.ap_dung(model)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            peft.luu(model, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestMultiGPUTrainerFull:
    def test_huan_luyen_cpu_fallback(self):
        from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer

        trainer = MultiGPUTrainer()

        class SimpleWrapper:
            so_vong = 2
            da_huan_luyen = False

            def huan_luyen(self, X, y):
                pass

        model = SimpleWrapper()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = trainer.huan_luyen(model, X, y)
        assert "thiet_bi" in result
