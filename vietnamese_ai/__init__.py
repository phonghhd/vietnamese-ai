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

__version__ = "11.0.1"
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
        "NhanDienThucThe": ("vietnamese_ai.nlp.ner", "NhanDienThucThe"),
        "HoiDapTiengViet": ("vietnamese_ai.nlp.qa", "HoiDapTiengViet"),
        "TomTatVanBan": ("vietnamese_ai.nlp.summarization", "TomTatVanBan"),
        "DichThuat": ("vietnamese_ai.nlp.translation", "DichThuat"),
        "KiemTraChinhTa": ("vietnamese_ai.nlp.spelling", "KiemTraChinhTa"),
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
        # === v10.0: RAG ===
        "CSDLVector": ("vietnamese_ai.rag.vector_store", "CSDLVector"),
        "TrichXuat": ("vietnamese_ai.rag.retriever", "TrichXuat"),
        "CatVanBan": ("vietnamese_ai.rag.chunker", "CatVanBan"),
        "RAGPipeline": ("vietnamese_ai.rag.rag_pipeline", "RAGPipeline"),
        "SapXepLai": ("vietnamese_ai.rag.reranker", "SapXepLai"),
        "GraphExtractor": ("vietnamese_ai.rag.graph", "GraphExtractor"),
        "NetworkXStore": ("vietnamese_ai.rag.graph", "NetworkXStore"),
        "GraphRetriever": ("vietnamese_ai.rag.graph", "GraphRetriever"),
        "ImageEmbedder": ("vietnamese_ai.rag.multimodal", "ImageEmbedder"),
        "MultimodalStore": ("vietnamese_ai.rag.multimodal", "MultimodalStore"),
        # === v10.0: Serving ===
        "MayChuBatch": ("vietnamese_ai.serving.batch_server", "MayChuBatch"),
        "MayChuStream": ("vietnamese_ai.serving.streaming", "MayChuStream"),
        "BoGioiHanTocDo": ("vietnamese_ai.serving.rate_limiter", "BoGioiHanTocDo"),
        # === v10.0: Prompts ===
        "MauPrompt": ("vietnamese_ai.prompts.templates", "MauPrompt"),
        "ChuoiPrompt": ("vietnamese_ai.prompts.chains", "ChuoiPrompt"),
        "LuongAnToan": ("vietnamese_ai.prompts.guardrails", "LuongAnToan"),
        "PhanTichDauRa": ("vietnamese_ai.prompts.parser", "PhanTichDauRa"),
        # === v10.0: Compression ===
        "HocRutGon": ("vietnamese_ai.compression.distillation", "HocRutGon"),
        "CatTiaMoHinh": ("vietnamese_ai.compression.pruning", "CatTiaMoHinh"),
        # === v10.0: Production ===
        "KiemTraSucKhoe": ("vietnamese_ai.production.health", "KiemTraSucKhoe"),
        "MachCat": ("vietnamese_ai.production.circuit_breaker", "MachCat"),
        "LoggerCauTruc": ("vietnamese_ai.production.logging", "LoggerCauTruc"),
        "QuanLyMetrics": ("vietnamese_ai.production.metrics", "QuanLyMetrics"),
        "LamNongModel": ("vietnamese_ai.production.warmup", "LamNongModel"),
        # === v11.0: SALM ===
        "SelfRefine": ("vietnamese_ai.salm.self_refine", "SelfRefine"),
        "SelfConsistency": ("vietnamese_ai.salm.self_consistency", "SelfConsistency"),
        "AdaptiveLoRA": ("vietnamese_ai.salm.adaptive_lora", "AdaptiveLoRA"),
        "SinhDuLieuTuDong": ("vietnamese_ai.salm.self_data", "SinhDuLieuTuDong"),
        "TestTimeTraining": ("vietnamese_ai.salm.test_time_training", "TestTimeTraining"),
        # === v12.0: Agents ===
        "CongCu": ("vietnamese_ai.agents.tools", "CongCu"),
        "cong_cu": ("vietnamese_ai.agents.tools", "cong_cu"),
        "BoNhoTacTu": ("vietnamese_ai.agents.memory", "BoNhoTacTu"),
        "TacTu": ("vietnamese_ai.agents.agent", "TacTu"),
        "HeThongDaTacTu": ("vietnamese_ai.agents.multi_agent", "HeThongDaTacTu"),
        "TacTuSwarm": ("vietnamese_ai.agents.swarm", "TacTuSwarm"),
        "HeThongSwarm": ("vietnamese_ai.agents.swarm", "HeThongSwarm"),
        "MoA": ("vietnamese_ai.agents.moa", "MoA"),
        "LapKeHoachMCTS": ("vietnamese_ai.agents.mcts_planning", "LapKeHoachMCTS"),
        # === v13.0: Edge & DePIN ===
        "NodeLlamaEngine": ("vietnamese_ai.edge.node_llama", "NodeLlamaEngine"),
        "EdgeRouter": ("vietnamese_ai.edge.intelligent_router", "EdgeRouter"),
        # === v15.0: Extreme Efficiency ===
        "BitLinear": ("vietnamese_ai.compression.extreme", "BitLinear"),
        "SpeculativeEngine": ("vietnamese_ai.serving.speculative", "SpeculativeEngine"),
    }

    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module 'vietnamese_ai' has no attribute '{name}'")


__all__ = [
    # v1.0 - v9.0
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
    # v10.0: RAG & v14.0: Advanced RAG
    "CSDLVector", "TrichXuat", "CatVanBan", "RAGPipeline", "SapXepLai",
    "GraphExtractor", "NetworkXStore", "GraphRetriever", "ImageEmbedder", "MultimodalStore",
    # v10.0: Serving
    "MayChuBatch", "MayChuStream", "BoGioiHanTocDo",
    # v10.0: Prompts
    "MauPrompt", "ChuoiPrompt", "LuongAnToan", "PhanTichDauRa",
    # v10.0: NLP Extensions
    "NhanDienThucThe", "HoiDapTiengViet", "TomTatVanBan", "DichThuat", "KiemTraChinhTa",
    # v10.0: Compression
    "HocRutGon", "CatTiaMoHinh",
    # v10.0: Production
    "KiemTraSucKhoe", "MachCat", "LoggerCauTruc", "QuanLyMetrics", "LamNongModel",
    # v11.0: SALM
    "SelfRefine", "SelfConsistency", "AdaptiveLoRA", "SinhDuLieuTuDong", "TestTimeTraining",
    # v12.0: Agents
    "CongCu", "cong_cu", "BoNhoTacTu", "TacTu", "HeThongDaTacTu",
    "TacTuSwarm", "HeThongSwarm", "MoA", "LapKeHoachMCTS",
    # v13.0: Edge & DePIN
    "NodeLlamaEngine", "EdgeRouter",
    # v15.0: Extreme Efficiency
    "BitLinear", "SpeculativeEngine",
]
