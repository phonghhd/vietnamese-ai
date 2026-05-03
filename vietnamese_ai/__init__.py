"""
Vietnamese AI Framework - Framework AI thuần tiếng Việt
=======================================================

Framework học máy thuần tiếng Việt, được thiết kế để đơn giản hóa
quá trình phát triển ứng dụng trí tuệ nhân tạo cho người Việt.

Sử dụng:
    >>> from vietnamese_ai import AutoML
    >>> auto = AutoML()
    >>> auto.fit(X_train, y_train)
    >>> du_doan = auto.predict(X_test)
"""

__version__ = "9.0.0"
__author__ = "EvoNet AI Team"


def __getattr__(name: str):
    """Lazy imports - chỉ import khi cần dùng, tránh load dependencies không cần thiết."""
    _lazy_imports = {
        # Core
        "AutoML": ("vietnamese_ai.automl.auto_ml", "AutoML"),
        "BaseModel": ("vietnamese_ai.models.base", "BaseModel"),
        "Engine": ("vietnamese_ai.core.engine", "Engine"),
        "KiemDinhCheo": ("vietnamese_ai.core.cross_validation", "KiemDinhCheo"),
        "Pipeline": ("vietnamese_ai.core.pipeline", "Pipeline"),
        "TimKiemThamSo": ("vietnamese_ai.core.hyperparameter", "TimKiemThamSo"),
        # Models
        "PhanLoai": ("vietnamese_ai.models.classifier", "PhanLoai"),
        "HoiQuy": ("vietnamese_ai.models.regression", "HoiQuy"),
        "PhanCum": ("vietnamese_ai.models.clustering", "PhanCum"),
        "MangNron": ("vietnamese_ai.models.neural_net", "MangNron"),
        "MoHinhTapHop": ("vietnamese_ai.models.ensemble", "MoHinhTapHop"),
        # Preprocessing
        "XuLyVanBan": ("vietnamese_ai.preprocessing.text", "XuLyVanBan"),
        "XuLySo": ("vietnamese_ai.preprocessing.numerical", "XuLySo"),
        "TaoDacTrung": ("vietnamese_ai.preprocessing.feature_engineering", "TaoDacTrung"),
        # NLP
        "PhanTichCamXuc": ("vietnamese_ai.nlp.sentiment", "PhanTichCamXuc"),
        "PhoBERTWrapper": ("vietnamese_ai.nlp.pretrained.phobert", "PhoBERTWrapper"),
        # Embeddings
        "Word2VecTiengViet": ("vietnamese_ai.embeddings.word2vec", "Word2VecTiengViet"),
        "FastTextTiengViet": ("vietnamese_ai.embeddings.fasttext", "FastTextTiengViet"),
        # Deep Learning
        "MangSau": ("vietnamese_ai.deep_learning.mang_sau", "MangSau"),
        # AutoML
        "TimKiemKienTruc": ("vietnamese_ai.automl.nas", "TimKiemKienTruc"),
        # Experiment Tracking
        "TheoDoiThiNghiem": ("vietnamese_ai.experiment_tracking.tracker", "TheoDoiThiNghiem"),
        # Interpretability
        "GiaiThichMoHinh": ("vietnamese_ai.interpretability.explainer", "GiaiThichMoHinh"),
        # Augmentation
        "TangCuongVanBan": ("vietnamese_ai.augmentation.text_augmenter", "TangCuongVanBan"),
        # Export
        "XuatONNX": ("vietnamese_ai.export.onnx_export", "XuatONNX"),
        "XuatGGUF": ("vietnamese_ai.export.gguf", "XuatGGUF"),
        # Distributed
        "PhanTanHuanLuyen": ("vietnamese_ai.distributed.distributed", "PhanTanHuanLuyen"),
        "MultiGPUTrainer": ("vietnamese_ai.distributed.multi_gpu", "MultiGPUTrainer"),
        # Hub
        "ModelHub": ("vietnamese_ai.hub.model_hub", "ModelHub"),
        # Plugins
        "PluginManager": ("vietnamese_ai.plugins.plugin_manager", "PluginManager"),
        # Cloud
        "CloudDeployment": ("vietnamese_ai.cloud.deployment", "CloudDeployment"),
        "Marketplace": ("vietnamese_ai.cloud.marketplace", "Marketplace"),
        # Enterprise
        "HeThongXacThuc": ("vietnamese_ai.enterprise.auth", "HeThongXacThuc"),
        "NhatKyHoatDong": ("vietnamese_ai.enterprise.audit", "NhatKyHoatDong"),
        # Mobile
        "TriKhaiDiDong": ("vietnamese_ai.mobile.deployment", "TriKhaiDiDong"),
        # Federated
        "HocLienKet": ("vietnamese_ai.federated.learning", "HocLienKet"),
        # Realtime
        "PipelineThoiGianThuc": ("vietnamese_ai.realtime.pipeline", "PipelineThoiGianThuc"),
        # SaaS
        "NenTangDichVu": ("vietnamese_ai.saas.platform", "NenTangDichVu"),
        # Studio
        "StudioKeoTha": ("vietnamese_ai.studio.builder", "StudioKeoTha"),
        # Timeseries
        "DuDoanChuoiThoiGian": ("vietnamese_ai.timeseries.forecaster", "DuDoanChuoiThoiGian"),
        # Vision
        "PhanLoaiHinhAnh": ("vietnamese_ai.vision.image_classifier", "PhanLoaiHinhAnh"),
        # Streaming
        "XuLyStream": ("vietnamese_ai.streaming.processor", "XuLyStream"),
        # Registry
        "QuanLyMoHinh": ("vietnamese_ai.registry.model_registry", "QuanLyMoHinh"),
        # Web
        "UngDungWeb": ("vietnamese_ai.web.app", "UngDungWeb"),
        # Visualization
        "BieuDo": ("vietnamese_ai.visualization.plots", "BieuDo"),
        # Utils
        "Logger": ("vietnamese_ai.utils.logger", "Logger"),
        "Metrics": ("vietnamese_ai.utils.metrics", "Metrics"),
        "Validator": ("vietnamese_ai.utils.validators", "Validator"),
        "LuuTai": ("vietnamese_ai.utils.io_utils", "LuuTai"),
        # Transformer
        "MultiHeadAttention": ("vietnamese_ai.transformer.attention", "MultiHeadAttention"),
        "TransformerModel": ("vietnamese_ai.transformer.model", "TransformerModel"),
        "VietnameseTokenizer": ("vietnamese_ai.transformer.tokenizer", "VietnameseTokenizer"),
        "GPTModel": ("vietnamese_ai.transformer.gpt_model", "GPTModel"),
        "PreTrainer": ("vietnamese_ai.transformer.pretrainer", "PreTrainer"),
        "TextDataset": ("vietnamese_ai.transformer.pretrainer", "TextDataset"),
        # LLM
        "VietnameseLLM": ("vietnamese_ai.llm.vietnamese_llm", "VietnameseLLM"),
        "ModelConfig": ("vietnamese_ai.llm.model_configs", "ModelConfig"),
        "LMEvalHarness": ("vietnamese_ai.llm.lm_eval", "LMEvalHarness"),
        "BenchmarkRunner": ("vietnamese_ai.llm.benchmark_runner", "BenchmarkRunner"),
        # Fine-tuning
        "HuanLuyenPyTorch": ("vietnamese_ai.fine_tuning.pytorch_trainer", "HuanLuyenPyTorch"),
        "UnslothWrapper": ("vietnamese_ai.fine_tuning.unsloth_wrapper", "UnslothWrapper"),
        "HuggingFaceWrapper": ("vietnamese_ai.fine_tuning.hf_wrapper", "HuggingFaceWrapper"),
        "LoRAAdapter": ("vietnamese_ai.fine_tuning.lora", "LoRAAdapter"),
        "QLoRAAdapter": ("vietnamese_ai.fine_tuning.lora", "QLoRAAdapter"),
        "PEFTConfig": ("vietnamese_ai.fine_tuning.peft_config", "PEFTConfig"),
        "LoRAPeft": ("vietnamese_ai.fine_tuning.lora_peft", "LoRAPeft"),
        "InstructionTuningTrainer": ("vietnamese_ai.fine_tuning.instruction_trainer", "InstructionTuningTrainer"),
        "SFTTrainer": ("vietnamese_ai.fine_tuning.sft_trainer", "SFTTrainer"),
        "DPOTrainer": ("vietnamese_ai.fine_tuning.dpo_trainer", "DPOTrainer"),
        "RewardModel": ("vietnamese_ai.fine_tuning.reward_model", "RewardModel"),
        "RLHFPipeline": ("vietnamese_ai.fine_tuning.rlhf_pipeline", "RLHFPipeline"),
    }

    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module 'vietnamese_ai' has no attribute '{name}'")


__all__ = [
    "AutoML", "BaseModel", "BenchmarkRunner", "CloudDeployment",
    "DPOTrainer", "DuDoanChuoiThoiGian", "Engine", "FastTextTiengViet",
    "GPTModel", "GiaiThichMoHinh", "HeThongXacThuc", "HocLienKet",
    "HoiQuy", "HuanLuyenPyTorch", "HuggingFaceWrapper",
    "InstructionTuningTrainer", "KiemDinhCheo", "LMEvalHarness",
    "LoRAAdapter", "LoRAPeft", "Logger", "LuuTai", "MangNron",
    "MangSau", "Marketplace", "Metrics", "ModelConfig", "ModelHub",
    "MoHinhTapHop", "MultiGPUTrainer", "MultiHeadAttention",
    "NenTangDichVu", "NhatKyHoatDong", "PEFTConfig", "PhanCum",
    "PhanLoai", "PhanLoaiHinhAnh", "PhanTichCamXuc", "PhanTanHuanLuyen",
    "PhoBERTWrapper", "Pipeline", "PipelineThoiGianThuc", "PluginManager",
    "PreTrainer", "QLoRAAdapter", "QuanLyMoHinh", "RLHFPipeline",
    "RewardModel", "SFTTrainer", "StudioKeoTha", "TangCuongVanBan",
    "TaoDacTrung", "TextDataset", "TheoDoiThiNghiem", "TimKiemKienTruc",
    "TimKiemThamSo", "TransformerModel", "TriKhaiDiDong", "UngDungWeb",
    "UnslothWrapper", "Validator", "VietnameseLLM", "VietnameseTokenizer",
    "Word2VecTiengViet", "XuLySo", "XuLyStream", "XuLyVanBan",
    "XuatGGUF", "XuatONNX",
]
