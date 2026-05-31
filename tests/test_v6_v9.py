"""Test suite cho v6.0-v9.0 - PEFT, SFT, DPO, RLHF, GPT, PreTrainer, ModelConfig, Eval."""

import numpy as np
import pytest

# ============================================================
# v6.0: PEFTConfig
# ============================================================


class TestPEFTConfig:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig(phuong_phap="lora", rank=16)
        assert config.phuong_phap == "lora"
        assert config.rank == 16
        assert config.alpha == 16.0

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        with pytest.raises(ValueError):
            PEFTConfig(phuong_phap="invalid")
        with pytest.raises(ValueError):
            PEFTConfig(rank=0)
        with pytest.raises(ValueError):
            PEFTConfig(bits=3)

    def test_preset_lora(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.lora(rank=32, alpha=32.0)
        assert config.phuong_phap == "lora"
        assert config.rank == 32

    def test_preset_qlora(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.qlora(rank=16, bits=4)
        assert config.phuong_phap == "qlora"
        assert config.bits == 4

    def test_scaling(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig(rank=16, alpha=32.0)
        assert config.scaling == 2.0

    def test_to_dict(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig()
        d = config.to_dict()
        assert "phuong_phap" in d
        assert "rank" in d
        assert "scaling" in d

    def test_from_dict(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        d = {"phuong_phap": "lora", "rank": 8, "alpha": 8.0}
        config = PEFTConfig.from_dict(d)
        assert config.rank == 8

    def test_repr(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig()
        r = repr(config)
        assert "PEFTConfig" in r
        assert "lora" in r


# ============================================================
# v6.0: LoRAPeft
# ============================================================


class TestLoRAPeft:
    def test_khoi_tao(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.lora(rank=8)
        peft = LoRAPeft(config)
        assert peft.config.rank == 8

    def test_ap_dung(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        config = PEFTConfig(phuong_phap="lora", rank=4, target_modules=["0", "2"])
        peft = LoRAPeft(config)
        model = peft.ap_dung(model)
        assert len(peft._lora_layers) > 0

    def test_thong_ke(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch chưa cài")

        from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.lora(rank=8)
        peft = LoRAPeft(config)
        tk = peft.thong_ke()
        assert "config" in tk
        assert "so_lora_layers" in tk


# ============================================================
# v6.0: InstructionTuningTrainer
# ============================================================


class TestInstructionTuningTrainer:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer(so_vong=3, toc_do_hoc=2e-5)
        assert trainer.so_vong == 3
        assert trainer.toc_do_hoc == 2e-5

    def test_huan_luyen_numpy(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        dataset = InstructionDataset(che_do="alpaca")
        du_lieu = [
            {"instruction": f"Câu hỏi {i}", "input": "", "output": f"Trả lời {i}"}
            for i in range(10)
        ]
        dataset.tai_tu_list(du_lieu)

        trainer = InstructionTuningTrainer(so_vong=2, logging_steps=5)
        ket_qua = trainer.huan_luyen(None, None, dataset)
        assert ket_qua["so_epoch"] == 2
        assert ket_qua["tong_thoi_gian"] >= 0

    def test_callback(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        dataset = InstructionDataset(che_do="alpaca")
        dataset.tai_tu_list([{"instruction": "test", "input": "", "output": "result"}] * 5)

        calls = []
        trainer = InstructionTuningTrainer(so_vong=1, logging_steps=2)
        trainer.huan_luyen(None, None, dataset, callback=lambda s, _loss: calls.append(s))
        assert len(calls) > 0

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

        trainer = InstructionTuningTrainer()
        tk = trainer.thong_ke()
        assert "so_vong" in tk
        assert "global_step" in tk


# ============================================================
# v6.0: InstructionDataset enhancements
# ============================================================


class TestInstructionDataset:
    def test_alpaca_format(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="alpaca")
        ds.tai_tu_list(
            [
                {"instruction": "Tóm tắt văn bản", "input": "Văn bản dài...", "output": "Tóm tắt"},
                {"instruction": "Dịch sang tiếng Anh", "input": "", "output": "Hello"},
            ]
        )

        formatted = ds.format_tat_ca()
        assert len(formatted) == 2
        assert "Tóm tắt" in formatted[0]

    def test_sharegpt_format(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do="sharegpt")
        ds.tai_tu_list(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Xin chào"},
                        {"from": "gpt", "value": "Chào bạn"},
                    ]
                }
            ]
        )

        formatted = ds.format_tat_ca()
        assert len(formatted) == 1

    def test_chia_du_lieu(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.tai_tu_list(
            [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(20)]
        )
        ket_qua = ds.chia_du_lieu(ty_le_val=0.2)
        assert ket_qua["so_train"] == 16
        assert ket_qua["so_val"] == 4

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset()
        ds.tai_tu_list([{"instruction": "test", "input": "", "output": "result"}])
        tk = ds.thong_ke()
        assert tk["so_mau"] == 1


# ============================================================
# v7.0: SFTTrainer
# ============================================================


class TestSFTTrainer:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer(so_vong=3)
        assert trainer.so_vong == 3

    def test_huan_luyen_numpy(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        du_lieu = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}] * 10
        trainer = SFTTrainer(so_vong=2, logging_steps=5)
        try:
            trainer.huan_luyen(None, du_lieu)
        except ImportError:
            pass

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer()
        tk = trainer.thong_ke()
        assert "so_vong" in tk


# ============================================================
# v7.0: DPOTrainer
# ============================================================


class TestDPOTrainer:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        trainer = DPOTrainer(beta=0.1)
        assert trainer.beta == 0.1

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        with pytest.raises(ValueError):
            DPOTrainer(beta=0)

    def test_huan_luyen_numpy(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        preference_data = [
            {"prompt": "câu hỏi", "chosen": "trả lời tốt", "rejected": "trả lời xấu"},
        ] * 10

        trainer = DPOTrainer(so_vong=2, beta=0.1)
        try:
            trainer.huan_luyen(None, None, preference_data)
        except ImportError:
            pass

    def test_lich_su(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        preference_data = [
            {"prompt": "q", "chosen": "good", "rejected": "bad"},
        ] * 5

        trainer = DPOTrainer(so_vong=1)
        try:
            trainer.huan_luyen(None, None, preference_data)
            lich_su = trainer.lay_lich_su()
            assert len(lich_su["train_loss"]) == 1
        except ImportError:
            lich_su = trainer.lay_lich_su()
            assert "train_loss" in lich_su

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

        trainer = DPOTrainer()
        tk = trainer.thong_ke()
        assert "beta" in tk


# ============================================================
# v7.0: RewardModel
# ============================================================


class TestRewardModel:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        rm = RewardModel()
        assert rm.toc_do_hoc == 1e-5

    def test_huan_luyen_numpy(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        preference_data = [
            {"chosen": "Sản phẩm tốt", "rejected": "Sản phẩm kém"},
        ] * 10

        rm = RewardModel()
        try:
            rm.huan_luyen(None, preference_data, so_vong=2)
        except ImportError:
            pass

    def test_diem_danh_gia(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        rm = RewardModel()
        ket_qua = rm.diem_danh_gia(None, ["văn bản 1", "văn bản 2"])
        assert len(ket_qua) == 2
        for kq in ket_qua:
            assert "score" in kq
            assert "score_normalized" in kq

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.reward_model import RewardModel

        rm = RewardModel()
        tk = rm.thong_ke()
        assert "score_mean" in tk


# ============================================================
# v7.0: RLHFPipeline
# ============================================================


class TestRLHFPipeline:
    def test_khoi_tao(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        assert pipeline is not None

    def test_sft(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        sft_data = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}] * 5
        try:
            pipeline.sft(None, sft_data)
        except ImportError:
            pass

    def test_train_reward_model(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        preference_data = [
            {"chosen": "good", "rejected": "bad"},
        ] * 5
        try:
            pipeline.train_reward_model(None, preference_data, so_vong=1)
        except ImportError:
            pass

    def test_rlhf(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        preference_data = [
            {"prompt": "q", "chosen": "good", "rejected": "bad"},
        ] * 5
        try:
            pipeline.rlhf(None, None, preference_data)
        except ImportError:
            pass

    def test_thong_ke(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()
        tk = pipeline.thong_ke()
        assert "sft_done" in tk
        assert "reward_done" in tk
        assert "rlhf_done" in tk


# ============================================================
# v8.0: GPTModel
# ============================================================


class TestGPTModel:
    def test_khoi_tao(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=2, so_tu_vung=100)
        assert model.d_model == 32
        assert model.so_block == 2

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        with pytest.raises(ValueError):
            GPTModel(d_model=32, so_dau=3)

    def test_tien(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=1, so_tu_vung=100)
        input_ids = np.array([[1, 2, 3, 4, 5]])
        logits = model.tien(input_ids)
        assert logits.shape == (1, 5, 100)

    def test_sinh_tiep(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=1, so_tu_vung=100)
        input_ids = np.array([[1, 2, 3]])
        generated = model.sinh_tiep(input_ids, so_token=5)
        assert generated.shape == (1, 8)

    def test_sinh_tiep_nhiet_do(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=1, so_tu_vung=50)
        input_ids = np.array([[1, 2]])
        result = model.sinh_tiep(input_ids, so_token=3, nhiet_do=0.5)
        assert result.shape == (1, 5)

    def test_sinh_tiep_top_k(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=1, so_tu_vung=50)
        input_ids = np.array([[1]])
        result = model.sinh_tiep(input_ids, so_token=3, top_k=5)
        assert result.shape == (1, 4)

    def test_tinh_loss(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=1, so_tu_vung=100)
        input_ids = np.array([[1, 2, 3, 4]])
        targets = np.array([[2, 3, 4, 5]])
        loss = model.tinh_loss(input_ids, targets)
        assert isinstance(loss, float)
        assert loss > 0

    def test_batch(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=32, so_dau=4, so_block=1, so_tu_vung=100)
        input_ids = np.array([[1, 2, 3], [4, 5, 6]])
        logits = model.tien(input_ids)
        assert logits.shape == (2, 3, 100)

    def test_thong_ke(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel

        model = GPTModel(d_model=64, so_dau=4, so_block=2, so_tu_vung=1000)
        tk = model.thong_ke()
        assert "loai" in tk
        assert "GPT" in tk["loai"]
        assert "so_tham_so" in tk


# ============================================================
# v8.0: TextDataset
# ============================================================


class TestTextDataset:
    def test_khoi_tao(self):
        from vietnamese_ai.transformer.pretrainer import TextDataset

        ds = TextDataset(do_dai_window=32)
        assert ds.do_dai_window == 32

    def test_tai_corpus(self):
        from vietnamese_ai.transformer.pretrainer import TextDataset

        ds = TextDataset(do_dai_window=16, buoc_nhay=8)
        corpus = ["học máy rất hay", "trí tuệ nhân tạo"] * 20
        ket_qua = ds.tai_corpus(corpus)
        assert ket_qua["so_tokens"] > 0
        assert ket_qua["vocab_size"] > 0

    def test_chia_du_lieu(self):
        from vietnamese_ai.transformer.pretrainer import TextDataset

        ds = TextDataset(do_dai_window=10, buoc_nhay=5)
        ds.tai_corpus(["hello world test"] * 50)
        ket_qua = ds.chia_du_lieu()
        assert ket_qua["so_train"] > 0
        assert ket_qua["so_val"] > 0

    def test_iter_batches(self):
        from vietnamese_ai.transformer.pretrainer import TextDataset

        ds = TextDataset(do_dai_window=10, buoc_nhay=5)
        ds.tai_corpus(["abcdef"] * 50)
        ds.chia_du_lieu()

        batches = list(ds.iter_batches(batch_size=4, che_do="train"))
        assert len(batches) > 0
        for input_ids, targets in batches:
            assert input_ids.shape[1] == 9  # window - 1
            assert targets.shape[1] == 9

    def test_ma_hoa_giai_ma(self):
        from vietnamese_ai.transformer.pretrainer import TextDataset

        ds = TextDataset(do_dai_window=10)
        ds.tai_corpus(["hello"])
        ids = ds.ma_hoa("hello")
        assert len(ids) == 5
        decoded = ds.giai_ma(ids)
        assert "h" in decoded

    def test_thong_ke(self):
        from vietnamese_ai.transformer.pretrainer import TextDataset

        ds = TextDataset(do_dai_window=10, buoc_nhay=5)
        ds.tai_corpus(["test"] * 20)
        tk = ds.thong_ke()
        assert "so_tokens" in tk
        assert "vocab_size" in tk


# ============================================================
# v8.0: PreTrainer
# ============================================================


class TestPreTrainer:
    def test_khoi_tao(self):
        from vietnamese_ai.transformer.pretrainer import PreTrainer

        trainer = PreTrainer(so_vong=5, toc_do_hoc=3e-4)
        assert trainer.so_vong == 5

    def test_huan_luyen(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel
        from vietnamese_ai.transformer.pretrainer import PreTrainer, TextDataset

        model = GPTModel(d_model=32, so_dau=4, d_ff=64, so_block=1, so_tu_vung=100)
        dataset = TextDataset(do_dai_window=16, buoc_nhay=8)
        dataset.tai_corpus(["học máy rất hay"] * 30, vocab_size=100)

        trainer = PreTrainer(so_vong=2, kich_thuoc_batch=8, logging_steps=5)
        ket_qua = trainer.huan_luyen(model, dataset)
        assert ket_qua["so_epoch"] == 2
        assert ket_qua["final_perplexity"] > 0

    def test_thong_ke(self):
        from vietnamese_ai.transformer.pretrainer import PreTrainer

        trainer = PreTrainer()
        tk = trainer.thong_ke()
        assert "so_vong" in tk


# ============================================================
# v9.0: ModelConfig
# ============================================================


class TestModelConfig:
    def test_khoi_tao(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        config = ModelConfig(d_model=128, so_dau=4)
        assert config.d_model == 128

    def test_khoi_tao_invalid(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        with pytest.raises(ValueError):
            ModelConfig(d_model=128, so_dau=3)

    def test_preset_tiny(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        config = ModelConfig.from_preset("vnlm-tiny")
        assert config.ten == "vnlm-tiny"
        assert config.d_model == 64

    def test_preset_small(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        config = ModelConfig.from_preset("vnlm-small")
        assert config.d_model == 768
        assert config.so_block == 12

    def test_preset_invalid(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        with pytest.raises(ValueError):
            ModelConfig.from_preset("nonexistent")

    def test_danh_sach_presets(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        presets = ModelConfig.danh_sach_presets()
        assert "vnlm-tiny" in presets
        assert "vnlm-7b" in presets
        assert len(presets) >= 6

    def test_so_tham_so(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        config = ModelConfig.from_preset("vnlm-tiny")
        assert config.so_tham_so > 0
        assert "M" in config.so_tham_so_str or "K" in config.so_tham_so_str

    def test_to_dict(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        config = ModelConfig.from_preset("vnlm-tiny")
        d = config.to_dict()
        assert "d_model" in d
        assert "so_block" in d

    def test_repr(self):
        from vietnamese_ai.llm.model_configs import ModelConfig

        config = ModelConfig.from_preset("vnlm-small")
        r = repr(config)
        assert "vnlm-small" in r


# ============================================================
# v9.0: LMEvalHarness
# ============================================================


class TestLMEvalHarness:
    def test_khoi_tao(self):
        from vietnamese_ai.llm.lm_eval import LMEvalHarness

        harness = LMEvalHarness()
        assert len(harness._tasks) >= 5

    def test_danh_sach_tasks(self):
        from vietnamese_ai.llm.lm_eval import LMEvalHarness

        harness = LMEvalHarness()
        tasks = harness.danh_sach_tasks()
        assert "vie_perplexity" in tasks
        assert "vie_sentiment" in tasks

    def test_dang_ky_task(self):
        from vietnamese_ai.llm.lm_eval import EvalTask, LMEvalHarness

        harness = LMEvalHarness()
        task = EvalTask(ten="custom_task", loai="classification", mo_ta="Test")
        harness.dang_ky_task(task)
        assert "custom_task" in harness.danh_sach_tasks()

    def test_danh_gia(self):
        from vietnamese_ai.llm.lm_eval import LMEvalHarness
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        corpus = ["học máy rất hay", "trí tuệ nhân tạo"] * 10
        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(corpus)

        harness = LMEvalHarness()
        ket_qua = harness.danh_gia(llm, ["vie_perplexity", "vie_sentiment"])
        assert ket_qua["so_tasks"] == 2
        assert "ket_qua" in ket_qua

    def test_bao_cao(self):
        from vietnamese_ai.llm.lm_eval import LMEvalHarness

        harness = LMEvalHarness()
        assert "Chưa có" in harness.bao_cao()

    def test_thong_ke(self):
        from vietnamese_ai.llm.lm_eval import LMEvalHarness

        harness = LMEvalHarness()
        tk = harness.thong_ke()
        assert tk["so_tasks"] >= 5


# ============================================================
# v9.0: BenchmarkRunner
# ============================================================


class TestBenchmarkRunner:
    def test_khoi_tao(self):
        from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()
        assert runner is not None

    def test_chay(self):
        from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        corpus = ["học máy rất hay", "trí tuệ nhân tạo"] * 10
        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(corpus)

        runner = BenchmarkRunner()
        ket_qua = runner.chay(llm, benchmarks=["perplexity", "speed"])
        assert "perplexity" in ket_qua
        assert "speed" in ket_qua
        assert ket_qua["tong_thoi_gian"] >= 0

    def test_bao_cao(self):
        from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        corpus = ["học máy rất hay"] * 10
        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(corpus)

        runner = BenchmarkRunner()
        runner.chay(llm, benchmarks=["perplexity"])
        bc = runner.bao_cao()
        assert "BENCHMARK" in bc
        assert "Perplexity" in bc

    def test_thong_ke(self):
        from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()
        tk = runner.thong_ke()
        assert "co_ket_qua" in tk


# ============================================================
# Integration tests
# ============================================================


class TestV6V9Integration:
    def test_version(self):
        import vietnamese_ai

        assert vietnamese_ai.__version__ == "11.0.1"

    def test_all_exports_count(self):
        import vietnamese_ai

        assert len(vietnamese_ai.__all__) >= 60

    def test_imports_v6(self):
        from vietnamese_ai import (
            InstructionTuningTrainer,
            LoRAAdapter,
            LoRAPeft,
            PEFTConfig,
            QLoRAAdapter,
        )

        assert PEFTConfig is not None
        assert LoRAPeft is not None
        assert InstructionTuningTrainer is not None
        assert LoRAAdapter is not None
        assert QLoRAAdapter is not None

    def test_imports_v7(self):
        from vietnamese_ai import (
            DPOTrainer,
            RewardModel,
            RLHFPipeline,
            SFTTrainer,
        )

        assert SFTTrainer is not None
        assert DPOTrainer is not None
        assert RewardModel is not None
        assert RLHFPipeline is not None

    def test_imports_v8(self):
        from vietnamese_ai import GPTModel, PreTrainer, TextDataset

        assert GPTModel is not None
        assert PreTrainer is not None
        assert TextDataset is not None

    def test_imports_v9(self):
        from vietnamese_ai import BenchmarkRunner, LMEvalHarness, ModelConfig

        assert ModelConfig is not None
        assert LMEvalHarness is not None
        assert BenchmarkRunner is not None

    def test_end_to_end_gpt_pretrain(self):
        from vietnamese_ai.transformer.gpt_model import GPTModel
        from vietnamese_ai.transformer.pretrainer import PreTrainer, TextDataset

        model = GPTModel(d_model=32, so_dau=4, d_ff=64, so_block=1, so_tu_vung=100)
        dataset = TextDataset(do_dai_window=16, buoc_nhay=8)
        dataset.tai_corpus(["học máy rất hay"] * 20, vocab_size=100)

        trainer = PreTrainer(so_vong=1, kich_thuoc_batch=8, logging_steps=10)
        ket_qua = trainer.huan_luyen(model, dataset)
        assert ket_qua["tong_thoi_gian"] > 0

        generated = model.sinh_tiep(np.array([[1, 2, 3]]), so_token=5)
        assert generated.shape == (1, 8)

    def test_end_to_end_peft(self):
        from vietnamese_ai.fine_tuning.peft_config import PEFTConfig

        config = PEFTConfig.lora(rank=8, alpha=16)
        assert config.scaling == 2.0
        d = config.to_dict()
        config2 = PEFTConfig.from_dict(d)
        assert config2.rank == config.rank

    def test_end_to_end_model_config(self):
        from vietnamese_ai.llm.model_configs import ModelConfig
        from vietnamese_ai.transformer.gpt_model import GPTModel

        config = ModelConfig.from_preset("vnlm-tiny")
        model = GPTModel(**config.to_dict())
        tk = model.thong_ke()
        assert tk["d_model"] == config.d_model

    def test_end_to_end_eval(self):
        from vietnamese_ai.llm.lm_eval import LMEvalHarness
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        corpus = ["học máy rất hay", "trí tuệ nhân tạo"] * 10
        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(corpus)

        harness = LMEvalHarness()
        ket_qua = harness.danh_gia(llm, ["vie_perplexity"])
        assert ket_qua["so_tasks"] == 1

    def test_end_to_end_benchmark(self):
        from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        corpus = ["học máy rất hay"] * 10
        llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
        llm.huan_luyen(corpus)

        runner = BenchmarkRunner()
        runner.chay(llm, benchmarks=["perplexity", "speed"])
        bc = runner.bao_cao()
        assert "BENCHMARK" in bc

    def test_end_to_end_rlhf_pipeline(self):
        from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

        pipeline = RLHFPipeline()

        sft_data = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}] * 5
        pref_data = [{"chosen": "good", "rejected": "bad"}] * 5
        dpo_data = [{"prompt": "q", "chosen": "good", "rejected": "bad"}] * 5

        try:
            pipeline.sft(None, sft_data)
            pipeline.train_reward_model(None, pref_data, so_vong=1)
            pipeline.rlhf(None, None, dpo_data)
        except ImportError:
            pass

        tk = pipeline.thong_ke()
        assert "sft_done" in tk
